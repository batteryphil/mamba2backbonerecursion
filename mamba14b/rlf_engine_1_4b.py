"""
rlf_engine_1_4b.py — Recursive Latent Forcing Engine for Mamba-1.4B
=====================================================================
Port of batteryphil/mamba2backbonerecursion/mamba_engine.py adapted for
state-spaces/mamba-1.4b (d_model=2048, 48 layers, Mamba1 SSM blocks).

Architecture (per forward pass):
  Input → embed → all 48 Mamba1 layers → x_prompt anchor (lifeline)
        → PREPEND M=8 latent memory tokens → [mem | prompt]
        → RLF Loop × MAX_LOOPS:
              ├ Lifeline re-inject (prompt positions only)
              ├ LoopRoPE (loop index encoding → loop 1 ≠ loop 6)
              ├ LoRA top 24 layers (reasoning core)
              ├ Mamba1 loop engine (dedicated SSM)
              ├ RMSNorm
              └ Latent bridge: x + bridge_up(bridge_down(x))
        → Slice off M prefix tokens
        → norm_f(x + residual) → lm_head → logits
        → Predict one token. If § → halt.

Original insight credit: ItsMick — Mamba natively handles O(1) loop state
over sequence time, bypassing the KV-Cache entirely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel, Mamba
from torch.utils.checkpoint import checkpoint as grad_ckpt

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL   = "state-spaces/mamba-1.4b"
D_MODEL      = 2048     # mamba-1.4b hidden size
N_LAYERS     = 48       # total backbone layers
BASE_SPLIT   = 24       # freeze bottom half of layers
LORA_RANK    = 8
PREFIX_M     = 8        # latent scratchpad tokens
BRIDGE_RANK  = 128      # low-rank bridge bottleneck (widened from 64 for V2)
MAX_LOOPS    = 6        # max RLF iterations per token

# Mamba1 loop engine params
LOOP_D_STATE = 16
LOOP_D_CONV  = 4
LOOP_EXPAND  = 1

# ── Tokenizer + HALT token ────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token

# § maps to token 7803 in GPT-NeoX vocab — confirmed single token.
# Low frequency in code/math data → safe to repurpose as HALT signal.
# No embedding resize needed — preserves SFT checkpoint compatibility.
HALT_ID = tokenizer.encode("§")[0]   # = 7803


# ── Lifeline Decay schedule ──────────────────────────────────────────────────
LIFELINE_DECAY = 0.7   # per-loop multiplier; by loop 3, prompt is at 0.7^3 ≈ 0.34
                        # forcing the model to rely on the ConceptPerceptron map.


# ── ConceptPerceptron ─────────────────────────────────────────────────────────
class ConceptPerceptron(nn.Module):
    """Generates Korpela's 'Conceptual Model' map from the raw prompt.

    Pools the full backbone output into a global context vector, then projects
    it into the M-token scratchpad prefix.  Unlike the old static
    `latent_memory` parameter (which received zero gradient because the model
    halted before backprop reached it), the Perceptron is conditioned on the
    *input* so it always participates in the graph — HALT or not.
    """

    def __init__(self, d_model: int, prefix_m: int) -> None:
        """Build a two-layer MLP: d_model → d_model//2 → prefix_m*d_model."""
        super().__init__()
        self.prefix_m = prefix_m
        self.mapper = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=True),
            nn.GELU(),
            nn.Linear(d_model // 2, prefix_m * d_model, bias=True),
        ).to(torch.bfloat16)
        # Small init so the perceptron starts near-zero, allowing the backbone
        # residual to dominate early training (same philosophy as bridge_up=zeros).
        for layer in self.mapper:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

    def forward(self, x_prompt: torch.Tensor) -> torch.Tensor:
        """Map backbone output → conceptual scratchpad prefix.

        Args:
            x_prompt: [B, T, D] full backbone hidden states

        Returns:
            [B, prefix_m, D] conceptual map tokens
        """
        # Mean-pool over sequence → global context vector [B, D]
        context_vector = x_prompt.mean(dim=1)
        # Project → [B, prefix_m * D]
        latent_map = self.mapper(context_vector)
        return latent_map.view(-1, self.prefix_m, x_prompt.size(-1))


# ── 1D RoPE for Loop Index ────────────────────────────────────────────────────
class LoopRoPE(nn.Module):
    """1D Rotary Position Embedding keyed by loop iteration index.

    Ensures loop 1, 2, ..., 6 occupy geometrically distinct subspaces,
    preventing gradient mode collapse where all loops become identical.
    """

    def __init__(self, d_model: int, base: int = 10000) -> None:
        """Precompute inverse frequency bands."""
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def _sincos(
        self, idx: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) vectors for loop index idx."""
        n     = torch.tensor(float(idx), device=device)
        freqs = n * self.inv_freq.to(device=device, dtype=torch.float32)
        cos_v = torch.stack([freqs.cos(), freqs.cos()], dim=-1).flatten()[: self.d_model]
        sin_v = torch.stack([freqs.sin(), freqs.sin()], dim=-1).flatten()[: self.d_model]
        return cos_v.to(dtype), sin_v.to(dtype)

    @staticmethod
    def _rot_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate adjacent pairs: [..., x1, x2, ...] → [..., -x2, x1, ...]."""
        return torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, loop_idx: int) -> torch.Tensor:
        """Apply RoPE rotation for loop_idx to x shaped [B, T, d_model]."""
        cos, sin = self._sincos(loop_idx, x.device, x.dtype)
        return x * cos + self._rot_half(x) * sin


# ── LoRA Linear ───────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """Low-rank adapter. lora_B initialised to zero → identity at start of training."""

    def __init__(
        self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0
    ) -> None:
        """Wrap a base nn.Linear with LoRA A/B matrices."""
        super().__init__()
        self.bias = linear.bias
        d_out, d_in = linear.weight.shape
        dtype  = linear.weight.dtype
        device = linear.weight.device
        self.register_buffer("base_weight", linear.weight.data.clone())
        self.lora_A = nn.Parameter(torch.empty(rank, d_in, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, dtype=dtype, device=device))
        self.scale  = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A)

    @property
    def weight(self) -> torch.Tensor:
        """Runtime fused weight: W_base + scale * (B @ A)."""
        return self.base_weight + self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fused weight."""
        return F.linear(x, self.weight, self.bias)


def fuse_lora_weights(model: nn.Module) -> None:
    """Merge LoRA A/B into base weight and replace with plain nn.Linear.

    Call before export/inference to eliminate LoRA VRAM overhead.
    Fuses on CPU to avoid double-VRAM allocation on GPU.
    """
    import gc

    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    for name, mod in targets:
        dev   = mod.base_weight.device
        dtype = mod.base_weight.dtype
        fused = (mod.base_weight.float() + mod.scale *
                 (mod.lora_B.data.float() @ mod.lora_A.data.float())).cpu()
        bias  = mod.bias.data.cpu() if mod.bias is not None else None

        parts  = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        getattr(parent, parts[-1]).cpu()
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        new_lin = nn.Linear(fused.shape[1], fused.shape[0],
                            bias=bias is not None, dtype=dtype, device=dev)
        new_lin.weight.data.copy_(fused.to(dtype))
        if bias is not None:
            new_lin.bias.data.copy_(bias.to(dtype))
        new_lin.requires_grad_(False)
        setattr(parent, parts[-1], new_lin)
        del fused, bias
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  Fused LoRA: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# RecursiveMamba1_PrefixScratchpad — Main RLF Engine for 1.4B
# ══════════════════════════════════════════════════════════════════════════════
class RecursiveMamba1_PrefixScratchpad(nn.Module):
    """Mamba-1.4B + Prefix Latent Scratchpad + RLF loop engine.

    Key architectural additions over base SFT model:
      latent_memory : [1, M, D] virtual scratchpad prepended to sequence
      mamba1_loop   : dedicated Mamba1 SSM for iterative reasoning
      loop_rope     : distinguishes loop iterations geometrically
      lifeline_gate : learned scalar gate for prompt re-injection
      bridge_down/up: low-rank AE bottleneck translating loop→vocab space
    """

    MAX_LOOPS: int = MAX_LOOPS

    def __init__(
        self, backbone: MambaLMHeadModel, lora_rank: int = LORA_RANK
    ) -> None:
        """Build RLF engine from a loaded MambaLMHeadModel."""
        super().__init__()
        bb = backbone.backbone
        self.embedding  = bb.embedding
        self.layers     = nn.ModuleList(bb.layers)
        self.norm_f     = bb.norm_f
        self.lm_head    = backbone.lm_head
        self.M          = PREFIX_M
        d_model         = self.embedding.embedding_dim

        # ── Freeze bottom BASE_SPLIT layers ──────────────────────────────────
        for layer in self.layers[:BASE_SPLIT]:
            for p in layer.parameters():
                p.requires_grad = False

        # ── LoRA on top BASE_SPLIT..end layers ────────────────────────────────
        for layer in self.layers[BASE_SPLIT:]:
            mx = layer.mixer
            for attr in ("in_proj", "out_proj"):
                if hasattr(mx, attr):
                    setattr(mx, attr, LoRALinear(
                        getattr(mx, attr), rank=lora_rank, alpha=lora_rank * 2.0
                    ))

        # ── RLF components ────────────────────────────────────────────────────
        self.loop_rope = LoopRoPE(d_model)

        self.mamba1_loop = Mamba(
            d_model=d_model, d_state=LOOP_D_STATE,
            d_conv=LOOP_D_CONV, expand=LOOP_EXPAND,
        ).to(torch.bfloat16)
        nn.init.zeros_(self.mamba1_loop.out_proj.weight)

        self.loop_norm = nn.RMSNorm(d_model).to(torch.bfloat16)

        self.lifeline_gate = nn.Parameter(
            torch.ones(d_model, dtype=torch.bfloat16)
        )

        # V2: ConceptPerceptron replaces the static latent_memory parameter.
        # The perceptron derives the scratchpad prefix from the prompt itself,
        # guaranteeing gradient flow regardless of when HALT fires.
        self.concept_perceptron = ConceptPerceptron(d_model, PREFIX_M)

        # Low-rank bridge: widened to rank 128 for V2 (was 64).
        # Starts as near-identity (bridge_up zeros init).
        self.bridge_down = nn.Linear(d_model, BRIDGE_RANK, bias=False,
                                     dtype=torch.bfloat16)
        self.bridge_up   = nn.Linear(BRIDGE_RANK, d_model, bias=False,
                                     dtype=torch.bfloat16)
        nn.init.kaiming_uniform_(self.bridge_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.bridge_up.weight)

        self.d_model = d_model
        self._print_param_report()

    def _print_param_report(self) -> None:
        """Print trainable parameter breakdown."""
        n_lora   = sum(p.numel() for n, p in self.named_parameters()
                       if p.requires_grad and "lora" in n.lower())
        n_loop   = sum(p.numel() for p in self.mamba1_loop.parameters())
        n_percep = sum(p.numel() for p in self.concept_perceptron.parameters())
        n_bridge = (sum(p.numel() for p in self.bridge_down.parameters()) +
                    sum(p.numel() for p in self.bridge_up.parameters()))
        n_tr     = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_fr     = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"  LoRA params:       {n_lora:,}")
        print(f"  Loop engine:       {n_loop:,}")
        print(f"  ConceptPerceptron: {n_percep:,} (prompt→{self.M}×{self.d_model} map)")
        print(f"  Latent bridge:     {n_bridge:,} ({self.d_model}→{BRIDGE_RANK}→{self.d_model})")
        print(f"  Total trainable:   {n_tr:,}")
        print(f"  Base frozen:       {n_fr:,}")
        print(f"  HALT token:        {HALT_ID} (§)")
        print(f"  Lifeline decay:    {LIFELINE_DECAY}^loop_idx (text fades each loop)")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _encode(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Embed + run all 48 layers, return (hidden, residual) BEFORE norm_f."""
        x     = self.embedding(input_ids)
        res   = None
        for layer in self.layers:
            x, res = layer(x, residual=res)
        return x, res

    def _run_top_layers(
        self, x: torch.Tensor, res: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run only top (BASE_SPLIT..) layers — the LoRA reasoning core."""
        for layer in self.layers[BASE_SPLIT:]:
            x, res = layer(x, residual=res)
        return x, res

    def _apply_norm(
        self, x: torch.Tensor, res: torch.Tensor | None
    ) -> torch.Tensor:
        """Apply norm_f: norm_f(x + residual) if residual else norm_f(x)."""
        if res is not None:
            x = x + res
        return self.norm_f(x.to(self.norm_f.weight.dtype))

    def _lifeline_inject(
        self,
        x_ext: torch.Tensor,
        res_ext: torch.Tensor | None,
        x_prompt: torch.Tensor,
        res_prompt: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Re-inject x_prompt into prompt positions (M..), leave prefix alone.

        Uses torch.cat (not in-place) to preserve the autograd graph.
        Prefix positions 0..M-1 evolve freely as scratch paper.
        """
        gate     = self.lifeline_gate
        prefix_x = x_ext[:, : self.M, :]
        prompt_x = x_ext[:, self.M :, :]
        inj_x    = prompt_x + gate * x_prompt
        new_x    = torch.cat([prefix_x, inj_x], dim=1)

        new_res = res_ext
        if res_ext is not None and res_prompt is not None:
            prefix_r = res_ext[:, : self.M, :]
            prompt_r = res_ext[:, self.M :, :]
            inj_r    = prompt_r + gate * res_prompt
            new_res  = torch.cat([prefix_r, inj_r], dim=1)

        return new_x, new_res

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        chain_targets: list | None = None,
        ans_starts: list | None = None,
    ) -> tuple:
        """RLF forward pass.

        Sequence shape transforms:
          [B,T] → embed [B,T,D] → extend [B,M+T,D] → slice [B,T,D] → logits [B,T,V]

        Args:
            input_ids:     [B, T] token ids
            chain_targets: list[list[int]] — per-sample chain token id sequences
            ans_starts:    list[int] — per-sample answer start positions

        Returns (training):   (avg_loss, avg_acc, answer_acc, halt_acc)
        Returns (inference):  (n_loops, trace, last_answer_str)
        """
        B = input_ids.shape[0]

        # ── Full backbone encode ──────────────────────────────────────────────
        x, res = self._encode(input_ids)         # [B, T, D], [B, T, D]|None
        # Detach lifeline anchors — they are constants for all loops
        x_prompt   = x.detach().clone()
        res_prompt = res.detach().clone() if res is not None else None

        # ── V2: Initialise scratchpad from ConceptPerceptron ─────────────────
        # The perceptron maps the full backbone output → M conceptual tokens.
        # This runs even if the model halts at Loop 1, so gradients always
        # flow back through the perceptron regardless of HALT position.
        mem     = self.concept_perceptron(x_prompt)            # [B, M, D]
        x_ext   = torch.cat([mem, x], dim=1)                  # [B, M+T, D]
        # Zero-pad residual for the M new prefix positions
        if res is not None:
            res_pad = torch.zeros(B, self.M, self.d_model,
                                  device=res.device, dtype=res.dtype)
            res_ext = torch.cat([res_pad, res], dim=1)        # [B, M+T, D]
        else:
            res_ext = None

        # ── Training ──────────────────────────────────────────────────────────
        if self.training and chain_targets is not None:
            n_loops = max(len(t) for t in chain_targets)

            def _top_ckpt(x_in: torch.Tensor, r_in: torch.Tensor | None):
                """Gradient-checkpointed top LoRA layers."""
                return self._run_top_layers(x_in, r_in)

            step_losses: list[torch.Tensor] = []
            step_accs:   list[torch.Tensor] = []
            halt_accs:   list[float]        = []

            for loop_i in range(n_loops):
                # V2 Lifeline Decay: raw prompt fades by LIFELINE_DECAY^loop_i.
                # By loop 3 the signal is at ~34% of original strength.
                # The model must use the ConceptPerceptron map — it cannot
                # coast on re-reading the full prompt every iteration.
                decay = LIFELINE_DECAY ** loop_i
                x_prompt_decayed   = x_prompt * decay
                res_prompt_decayed = (res_prompt * decay
                                      if res_prompt is not None else None)
                x_ext, res_ext = self._lifeline_inject(
                    x_ext, res_ext, x_prompt_decayed, res_prompt_decayed
                )
                # Loop RoPE
                x_ext    = self.loop_rope(x_ext, loop_i)
                if res_ext is not None:
                    res_ext = self.loop_rope(res_ext, loop_i)

                # LoRA reasoning core (gradient-checkpointed)
                x_ext, res_ext = grad_ckpt(
                    _top_ckpt, x_ext, res_ext, use_reentrant=False
                )

                # Mamba1 loop engine
                x_ext = x_ext + self.mamba1_loop(x_ext)
                x_ext = self.loop_norm(x_ext)

                # Low-rank bridge (+ residual = near-identity init)
                x_bridged = x_ext + self.bridge_up(self.bridge_down(x_ext))

                # Slice off prefix tokens, apply norm, project to vocab
                x_out   = x_bridged[:, self.M :, :]           # [B, T, D]
                r_out   = res_ext[:, self.M :, :] if res_ext is not None else None
                x_normed = self._apply_norm(x_out, r_out)
                logits   = self.lm_head(x_normed)              # [B, T, V]
                V        = logits.shape[-1]

                loop_loss = torch.tensor(0.0, device=x_ext.device, requires_grad=True)
                loop_acc  = torch.tensor(0.0, device=x_ext.device)
                valid     = 0

                for b in range(B):
                    as_ = (ans_starts[b] if ans_starts else x_out.shape[1] - 1)
                    if as_ < 1 or as_ >= x_out.shape[1]:
                        continue
                    tgt_id = int(chain_targets[b][min(loop_i, len(chain_targets[b]) - 1)])
                    if tgt_id >= V:
                        continue
                    lg_b   = logits[b, as_ - 1, :]
                    pred   = lg_b.argmax().item()
                    tgt_t  = torch.tensor(tgt_id, device=x_ext.device)
                    loop_loss = loop_loss + F.cross_entropy(
                        lg_b.unsqueeze(0), tgt_t.unsqueeze(0)
                    )
                    loop_acc = loop_acc + float(pred == tgt_id)
                    valid   += 1
                    if tgt_id == HALT_ID:
                        halt_accs.append(float(pred == tgt_id))

                if valid > 0:
                    step_losses.append(loop_loss / valid)
                    step_accs.append(loop_acc / valid)

            avg_loss = (torch.stack(step_losses).mean() if step_losses else
                        torch.tensor(0.0, requires_grad=True))
            avg_acc  = (torch.stack([a.detach() for a in step_accs]).mean()
                        if step_accs else torch.tensor(0.0))
            ans_accs = step_accs[:-1] if len(step_accs) > 1 else step_accs
            ans_acc  = (torch.stack([a.detach() for a in ans_accs]).mean()
                        if ans_accs else avg_acc)
            halt_acc = (sum(halt_accs) / len(halt_accs)) if halt_accs else 0.0
            return avg_loss, avg_acc, ans_acc, halt_acc

        # ── Inference ────────────────────────────────────────────────────────
        trace: list[tuple] = []
        last  = ""
        with torch.no_grad():
            for loop_i in range(self.MAX_LOOPS):
                # Apply lifeline decay at inference too for consistency.
                decay = LIFELINE_DECAY ** loop_i
                x_prompt_decayed   = x_prompt * decay
                res_prompt_decayed = (res_prompt * decay
                                      if res_prompt is not None else None)
                x_ext, res_ext = self._lifeline_inject(
                    x_ext, res_ext, x_prompt_decayed, res_prompt_decayed
                )
                x_ext = self.loop_rope(x_ext, loop_i)
                if res_ext is not None:
                    res_ext = self.loop_rope(res_ext, loop_i)

                x_ext, res_ext = self._run_top_layers(x_ext, res_ext)
                x_ext = x_ext + self.mamba1_loop(x_ext)
                x_ext = self.loop_norm(x_ext)

                x_bridged = x_ext + self.bridge_up(self.bridge_down(x_ext))
                x_out     = x_bridged[:, self.M :, :]
                r_out     = res_ext[:, self.M :, :] if res_ext is not None else None
                logits    = self.lm_head(self._apply_norm(x_out, r_out))
                p         = torch.softmax(logits[0, -1, :].float(), dim=-1)
                tid       = p.argmax().item()
                tok       = tokenizer.decode([tid]).strip()
                conf      = round(p[tid].item(), 4)

                if tid == HALT_ID:
                    trace.append((f"L{loop_i + 1}", "<HALT>", conf))
                    return loop_i + 1, trace, last

                trace.append((f"L{loop_i + 1}", tok, conf))
                last = tok

        return self.MAX_LOOPS, trace, last


# ══════════════════════════════════════════════════════════════════════════════
# Phase freeze helpers (verbatim logic from 2.8B engine)
# ══════════════════════════════════════════════════════════════════════════════

def freeze_for_phase3a(model: RecursiveMamba1_PrefixScratchpad) -> None:
    """Phase 3a: train ConceptPerceptron + bridge ONLY. Freeze everything else.

    V2 change: replaces latent_memory (now removed) with concept_perceptron.
    This warms up the dynamic prompt→scratchpad mapping and the bridge before
    the full LoRA/loop-engine is unleashed in Phase 3b.
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in model.concept_perceptron.parameters():
        p.requires_grad = True
    for p in model.bridge_down.parameters():
        p.requires_grad = True
    for p in model.bridge_up.parameters():
        p.requires_grad = True
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"  PHASE 3a — ConceptPerceptron + Bridge Warmup | trainable: {tr:,}")
    print(f"{'='*60}\n")


def freeze_for_phase3b(model: RecursiveMamba1_PrefixScratchpad) -> None:
    """Phase 3b: unfreeze top LoRA + loop engine + lifeline + perceptron + bridge.

    V2 change: concept_perceptron replaces latent_memory in the trainable set.
    """
    for layer in model.layers[BASE_SPLIT:]:
        for p in layer.parameters():
            p.requires_grad = True
    for p in model.mamba1_loop.parameters():
        p.requires_grad = True
    for p in model.loop_norm.parameters():
        p.requires_grad = True
    model.lifeline_gate.requires_grad = True
    for p in model.concept_perceptron.parameters():
        p.requires_grad = True
    for p in model.bridge_down.parameters():
        p.requires_grad = True
    for p in model.bridge_up.parameters():
        p.requires_grad = True
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fr = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\n{'='*60}")
    print(f"  PHASE 3b — RLF Joint | trainable: {tr:,} | frozen: {fr:,}")
    print(f"{'='*60}\n")


def freeze_for_phase3c(model: RecursiveMamba1_PrefixScratchpad) -> None:
    """Phase 3c: freeze RLF, train ONLY lm_head for SFT recovery."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.lm_head.parameters():
        p.requires_grad = True
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"  PHASE 3c — SFT Recovery | trainable: {tr:,}")
    print(f"{'='*60}\n")


def load_from_sft_checkpoint(
    ckpt_dir: str, device: str = DEVICE
) -> RecursiveMamba1_PrefixScratchpad:
    """Load Mamba-1.4B + R3 SFT lm_head, add RLF components.

    Args:
        ckpt_dir: path to SFT checkpoint directory containing lm_head.pt
        device:   target cuda/cpu device

    Returns:
        RecursiveMamba1_PrefixScratchpad ready for Phase 3a training
    """
    from pathlib import Path

    ckpt = Path(ckpt_dir)
    print(f"Loading base: {BASE_MODEL}")
    backbone = MambaLMHeadModel.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device=device
    )

    lm_head_path = ckpt / "lm_head.pt"
    if lm_head_path.exists():
        backbone.lm_head.load_state_dict(
            torch.load(lm_head_path, map_location=device, weights_only=True)
        )
        print(f"  Loaded SFT lm_head from {ckpt}")
    else:
        print(f"  Warning: lm_head.pt not found — using base weights")

    print("Building RLF engine...")
    model = RecursiveMamba1_PrefixScratchpad(backbone)
    return model.to(device)
