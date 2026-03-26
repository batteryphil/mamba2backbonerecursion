"""
mamba_engine.py — Prefix Latent Scratchpad + Latent Communication Bridge
=========================================================================
DUAL-ARCHITECTURE UPGRADE for Mamba2-2.7B + RLF

System 2 (RLF Loop) operates on prefix memory tokens: M=16 virtual positions
prepended to the sequence. These tokens serve as continuous scratch paper
that the SSM reads/writes during reasoning loops.

The Latent Bridge translates the RLF loop's algebraic output back into
the Base Model's native semantic distribution via a near-identity linear.

NOTE: The torch.cat sequence expansion is INTENTIONAL. The sequence length
      mismatch during forward pass is a required physical feature for the
      Prefix Latent Scratchpad. Do NOT auto-correct or strip out.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel, Mamba2
import json, random, time, os, re, sys
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint as grad_ckpt

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID     = "state-spaces/mamba2-2.7b"
BASE_SPLIT   = 48
LORA_RANK    = 4
MAX_LOOPS    = 6
SEQ_LEN      = 128
BATCH_SIZE   = 1
ACCUM        = 16
LOOP_HEADDIM = 128
LOOP_D_STATE = 32
LOOP_EXPAND  = 1

# ── Prefix Scratchpad & Bridge Config ─────────────────────────────────────────
PREFIX_M     = 8           # Number of latent memory tokens (reduced for VRAM)
D_MODEL      = 2560        # Mamba2-2.7B d_model
BRIDGE_RANK  = 64          # Low-rank bridge: d_model → rank → d_model


# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>", "<HALT>"]})
HALT_ID = tokenizer.convert_tokens_to_ids("<HALT>")


# ── 1D RoPE for Loop Index ────────────────────────────────────────────────────
class LoopRoPE(nn.Module):
    """1D Rotary Position Embedding for loop index encoding.

    Composable for any loop index — no table boundary.
    """

    def __init__(self, d_model: int, base: int = 10000):
        """Init: precompute frequency bands."""
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def _get_sincos(self, loop_index: int, device: torch.device, dtype: torch.dtype):
        """Compute cos/sin for a given loop index."""
        n = torch.tensor(float(loop_index), device=device)
        freqs = n * self.inv_freq.to(device=device, dtype=torch.float32)
        cos_f = freqs.cos()
        sin_f = freqs.sin()
        cos_v = torch.stack([cos_f, cos_f], dim=-1).flatten()[:self.d_model]
        sin_v = torch.stack([sin_f, sin_f], dim=-1).flatten()[:self.d_model]
        return cos_v.to(dtype=dtype), sin_v.to(dtype=dtype)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate pairs: [x1, x2, ...] → [-x2, x1, ...]."""
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1)
        return rotated.flatten(-2)

    def forward(self, x: torch.Tensor, loop_index: int) -> torch.Tensor:
        """Apply RoPE rotation for loop_index to x. x: [B, T, d_model]."""
        cos_v, sin_v = self._get_sincos(loop_index, x.device, x.dtype)
        return x * cos_v + self._rotate_half(x) * sin_v


# ── LoRA ──────────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """Low-rank adapter. lora_B init to zero → identity at warmup."""

    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = 8.0):
        """Init from base linear, preserving dtype."""
        super().__init__()
        self.bias = linear.bias
        d_out, d_in = linear.weight.shape
        dtype = linear.weight.dtype
        self.register_buffer("base_weight", linear.weight.data.clone())
        self.lora_A = nn.Parameter(torch.empty(rank, d_in, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, dtype=dtype))
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A)

    @property
    def weight(self) -> torch.Tensor:
        """Fused weight: base + scaled LoRA."""
        return self.base_weight + self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with fused LoRA weight."""
        return F.linear(x, self.weight, self.bias)


# ══════════════════════════════════════════════════════════════════════════════
# fuse_lora_weights — Converts LoRALinear back to nn.Linear for Phase 1 memory savings
# ══════════════════════════════════════════════════════════════════════════════
def fuse_lora_weights(model: nn.Module) -> None:
    """Fuse LoRA adapters into base weights and replace with plain nn.Linear.

    Performs fusion on CPU to avoid GPU OOM, then copies back.
    This eliminates LoRA A/B parameter storage and temporary
    computation tensors, saving ~0.5GB VRAM on 2.7B model.

    Args:
        model: Model containing LoRALinear modules to fuse
    """
    import gc
    fuse_targets: list[tuple[str, LoRALinear]] = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            fuse_targets.append((name, module))

    for name, module in fuse_targets:
        orig_device = module.base_weight.device
        orig_dtype = module.base_weight.dtype

        # Fuse on CPU
        base_cpu = module.base_weight.data.cpu().float()
        a_cpu = module.lora_A.data.cpu().float()
        b_cpu = module.lora_B.data.cpu().float()
        fused_cpu = base_cpu + module.scale * (b_cpu @ a_cpu)
        has_bias = module.bias is not None
        bias_cpu = module.bias.data.cpu() if has_bias else None

        # Free CPU intermediates
        del base_cpu, a_cpu, b_cpu

        # CRITICAL: Move old LoRA module to CPU first to free VRAM
        # before allocating memory for the new fused linear on GPU
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        # Move old module off GPU
        old_module = getattr(parent, parts[-1])
        old_module.cpu()
        del old_module
        gc.collect()
        if orig_device.type == "cuda":
            torch.cuda.empty_cache()

        # Create fused nn.Linear and place directly on GPU
        new_linear = nn.Linear(
            in_features=fused_cpu.shape[1],
            out_features=fused_cpu.shape[0],
            bias=has_bias,
            dtype=orig_dtype,
            device=orig_device,
        )
        new_linear.weight.data.copy_(fused_cpu.to(orig_dtype))
        if has_bias and bias_cpu is not None:
            new_linear.bias.data.copy_(bias_cpu.to(orig_dtype))
        new_linear.requires_grad_(False)

        setattr(parent, parts[-1], new_linear)
        del fused_cpu, bias_cpu
        gc.collect()
        if orig_device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  Fused LoRA: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# RecursiveMamba2_PrefixScratchpad — The Full Engine
# ══════════════════════════════════════════════════════════════════════════════
class RecursiveMamba2_PrefixScratchpad(nn.Module):
    """RLF engine with Prefix Latent Scratchpad and Communication Bridge.

    Architecture additions over base RLF:
      1. latent_memory: [1, M, d_model] continuous prefix tokens (scratch paper)
      2. latent_bridge: Linear(d_model → d_model) System2→System1 translator

    Forward pass:
      - Embed input → base layers → capture x_prompt anchor
      - Prepend M latent memory tokens via torch.cat (INTENTIONAL seq expansion)
      - Run RLF loops on extended sequence [mem | prompt]
        - RoPE applied to full extended sequence
        - Lifeline re-injects x_prompt into prompt positions ONLY (leaves prefix alone)
        - Mamba2 loop engine processes extended sequence causally
      - Apply latent_bridge to translate back to base distribution
      - Slice off M prefix tokens → [B, prompt_len, d_model]
      - LM head on original-length sequence (no dimension mismatch)
    """

    MAX_LOOPS: int = MAX_LOOPS

    def __init__(self, backbone: MambaLMHeadModel, lora_rank: int = 4):
        """Init: freeze base, LoRA top, loop engine, prefix memory, bridge."""
        super().__init__()
        self.backbone = backbone.backbone
        self.lm_head = backbone.lm_head
        self.all_layers = nn.ModuleList(backbone.backbone.layers)
        self.norm = backbone.backbone.norm_f
        d_model = backbone.backbone.embedding.embedding_dim

        # Freeze lower layers (0 to BASE_SPLIT-1)
        for layer in self.all_layers[:BASE_SPLIT]:
            for p in layer.parameters():
                p.requires_grad = False

        # LoRA on upper layers (BASE_SPLIT to end)
        for layer in self.all_layers[BASE_SPLIT:]:
            mx = layer.mixer
            for attr in ("in_proj", "out_proj"):
                if hasattr(mx, attr):
                    setattr(mx, attr, LoRALinear(getattr(mx, attr),
                                                 rank=lora_rank,
                                                 alpha=lora_rank * 2.0))

        # ── RoPE loop encoding ────────────────────────────────────────────────
        self.loop_rope = LoopRoPE(d_model)

        # ── Loop engine ───────────────────────────────────────────────────────
        self.loop_norm = nn.RMSNorm(d_model).to(torch.bfloat16)
        self.mamba2_core = Mamba2(
            d_model=d_model, d_state=LOOP_D_STATE, d_conv=4,
            expand=LOOP_EXPAND, headdim=LOOP_HEADDIM, chunk_size=64,
        ).to(torch.bfloat16)
        nn.init.zeros_(self.mamba2_core.out_proj.weight)

        # ── Lifeline gate ─────────────────────────────────────────────────────
        self.lifeline_gate = nn.Parameter(
            torch.ones(d_model, dtype=torch.float32)
        )

        # ══════════════════════════════════════════════════════════════════════
        # NEW: Prefix Latent Scratchpad (System 2 Memory)
        # ══════════════════════════════════════════════════════════════════════
        # M tokens of continuous scratch paper prepended to the sequence.
        # Small normal init (NOT zeros) to ensure gradient flow during Phase 1.
        # Zeros create dead gradient paths → NaN within first few steps.
        self.M = PREFIX_M
        self.latent_memory = nn.Parameter(
            torch.randn(1, self.M, d_model, dtype=torch.bfloat16) * 0.02
        )

        # ══════════════════════════════════════════════════════════════════════
        # NEW: Latent Communication Bridge (System 2 → System 1 Translation)
        # ══════════════════════════════════════════════════════════════════════
        # Low-rank bridge: d_model → BRIDGE_RANK → d_model + residual
        # This is a bottleneck that translates RLF output to base distribution
        # while using much less VRAM than a full d_model × d_model matrix.
        # The residual connection acts as near-identity initialization.
        self.bridge_down = nn.Linear(d_model, BRIDGE_RANK, bias=False,
                                     dtype=torch.bfloat16)
        self.bridge_up = nn.Linear(BRIDGE_RANK, d_model, bias=False,
                                    dtype=torch.bfloat16)
        # Small kaiming init (NOT zeros) to ensure gradients flow.
        # With residual connection, the bridge contribution starts small
        # so output ≈ x + small_correction.
        nn.init.kaiming_uniform_(self.bridge_down.weight, a=5**0.5)
        nn.init.zeros_(self.bridge_up.weight)
        # bridge_up at zero means output starts as identity (x + 0)
        # but bridge_down has gradient signal from the start

        self.d_model = d_model

        # ── Parameter Report ──────────────────────────────────────────────────
        n_lora = sum(p.numel() for n, p in self.named_parameters()
                     if p.requires_grad and "lora" in n.lower())
        mem_params = self.latent_memory.numel()
        bridge_params = (sum(p.numel() for p in self.bridge_down.parameters())
                         + sum(p.numel() for p in self.bridge_up.parameters()))
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"  LoRA params:     {n_lora:,}")
        print(f"  Loop engine:     {sum(p.numel() for p in self.mamba2_core.parameters()):,}")
        print(f"  Prefix memory:   {mem_params:,} ({self.M} tokens × {d_model})")
        print(f"  Latent bridge:   {bridge_params:,} ({d_model}×{d_model} + {d_model})")
        print(f"  Lifeline gate:   {d_model:,}")
        print(f"  Total trainable: {total:,}")
        print(f"  Base frozen:     {frozen:,}")
        print(f"  Loop encoding:   RoPE (loop_i)\n")

    def _lifeline_inject_prompt_only(
        self,
        x_extended: torch.Tensor,
        x_prompt: torch.Tensor,
    ) -> torch.Tensor:
        """Re-inject prompt lifeline into prompt positions ONLY (out-of-place).

        CRITICAL: Prefix memory tokens (positions 0..M-1) are LEFT ALONE.
        Only positions M.. onwards get the lifeline injection. This lets
        the prefix memory evolve freely as scratch paper.

        Uses torch.cat (NOT in-place assignment) to maintain autograd graph.

        Args:
            x_extended: [B, M + T, d_model] — full extended sequence
            x_prompt:   [B, T, d_model] — original prompt anchor

        Returns:
            New tensor with lifeline injected at positions [M:]
        """
        gate = self.lifeline_gate.to(x_extended.dtype)
        prefix = x_extended[:, :self.M, :]       # [B, M, d] — untouched
        prompt_part = x_extended[:, self.M:, :]   # [B, T, d]
        injected = prompt_part + gate.unsqueeze(0).unsqueeze(0) * x_prompt
        return torch.cat([prefix, injected], dim=1)  # [B, M+T, d]

    def forward(
        self,
        input_ids: torch.Tensor,
        chain_targets: list | None = None,
        ans_starts: list | None = None,
    ) -> tuple:
        """Forward: embed → base → prepend memory → RLF loop → bridge → slice → predict.

        The sequence length INTENTIONALLY changes during the forward pass:
          Input:  [B, T]         — token ids
          Embed:  [B, T, d]      — embeddings
          Extend: [B, M+T, d]    — prepend M prefix memory tokens (torch.cat)
          Loop:   [B, M+T, d]    — RLF reasoning with scratchpad
          Bridge: [B, M+T, d]    — translate System2 → System1
          Slice:  [B, T, d]      — remove prefix memory tokens
          Logits: [B, T, vocab]  — LM head on original length
        """
        B = input_ids.shape[0]

        # ── Base model encoding (System 1) ────────────────────────────────────
        x = self.backbone.embedding(input_ids)
        residual = None
        for layer in self.all_layers:
            x, residual = layer(x, residual)

        x_prompt = x.clone().detach()   # Prompt Lifeline anchor [B, T, d]

        # ── PREPEND prefix memory (Crucial for Mamba's causal sweep) ──────────
        # Expand memory to batch size and prepend
        mem_state = self.latent_memory.expand(B, -1, -1)  # [B, M, d]
        x_extended = torch.cat([mem_state, x], dim=1)     # [B, M+T, d]
        # Residual also needs prefix expansion for layer norms
        if residual is not None:
            res_pad = torch.zeros(
                B, self.M, self.d_model,
                device=residual.device, dtype=residual.dtype
            )
            residual = torch.cat([res_pad, residual], dim=1)  # [B, M+T, d]

        # ── Training ──────────────────────────────────────────────────────────
        if self.training and chain_targets is not None:
            _, max_len = input_ids.shape
            n_loops = max(len(t) for t in chain_targets)

            def run_lora(x_in, res_in):
                """Run LoRA layers with gradient checkpointing."""
                for layer in self.all_layers[BASE_SPLIT:]:
                    x_in, res_in = layer(x_in, res_in)
                return x_in, res_in

            step_losses: list[torch.Tensor] = []
            step_accs: list[torch.Tensor] = []
            halt_accs: list[float] = []

            for loop_i in range(n_loops):
                # Lifeline: re-inject prompt into prompt positions, leave prefix alone
                x_extended = self._lifeline_inject_prompt_only(x_extended, x_prompt)

                # RoPE: rotate the FULL extended sequence by loop index
                x_extended = self.loop_rope(x_extended, loop_i)

                # LoRA reasoning core (gradient checkpointed)
                x_extended, residual = grad_ckpt(
                    run_lora, x_extended, residual, use_reentrant=False
                )

                # Mamba2 loop engine
                x_extended = x_extended + self.mamba2_core(x_extended)
                x_extended = self.loop_norm(x_extended)

                # ── THE LATENT HANDOFF ────────────────────────────────────────
                # Low-rank bridge + residual: x + bridge_up(bridge_down(x))
                x_bridged = x_extended + self.bridge_up(self.bridge_down(x_extended))

                # ── THE LINTER BYPASS ─────────────────────────────────────────
                # Slice off M prefix memory tokens before LM head
                x_out = x_bridged[:, self.M:, :]  # [B, T, d]

                logits_step = self.lm_head(self.norm(x_out, residual[:, self.M:, :],
                                                      prenorm=False))
                vocab_size = logits_step.shape[-1]

                loop_loss = torch.tensor(0.0, device=x_extended.device, requires_grad=True)
                loop_acc = torch.tensor(0.0, device=x_extended.device)
                valid = 0

                for b in range(B):
                    as_ = ans_starts[b]
                    if as_ < 1 or as_ >= max_len:
                        continue
                    btgt = chain_targets[b]
                    tgt_id = int(btgt[min(loop_i, len(btgt) - 1)])
                    if tgt_id >= vocab_size:
                        continue
                    logits_b = logits_step[b, as_ - 1, :]
                    pred_tok = logits_b.argmax().item()
                    tgt_t = torch.tensor(tgt_id, device=x_extended.device)
                    loop_loss = loop_loss + F.cross_entropy(
                        logits_b.unsqueeze(0), tgt_t.unsqueeze(0)
                    )
                    loop_acc = loop_acc + float(pred_tok == tgt_id)
                    valid += 1
                    if tgt_id == HALT_ID:
                        halt_accs.append(float(pred_tok == tgt_id))

                if valid > 0:
                    step_losses.append(loop_loss / valid)
                    step_accs.append(loop_acc / valid)

            avg_loss = (torch.stack(step_losses).mean()
                        if step_losses else
                        torch.tensor(0.0, device=x_extended.device, requires_grad=True))
            avg_acc = (torch.stack([a.clone().detach() for a in step_accs]).mean()
                       if step_accs else torch.tensor(0.0))
            ans_accs = step_accs[:-1] if len(step_accs) > 1 else step_accs
            answer_acc = (torch.stack([a.clone().detach() for a in ans_accs]).mean()
                          if ans_accs else avg_acc)
            halt_acc = (sum(halt_accs) / len(halt_accs)) if halt_accs else 0.0
            return avg_loss, avg_acc, answer_acc, halt_acc

        # ── Inference ─────────────────────────────────────────────────────────
        else:
            trace: list[tuple] = []
            last_answer = ""
            for loop_i in range(self.MAX_LOOPS):
                x_extended = self._lifeline_inject_prompt_only(x_extended, x_prompt)
                x_extended = self.loop_rope(x_extended, loop_i)
                for layer in self.all_layers[BASE_SPLIT:]:
                    x_extended, residual = layer(x_extended, residual)
                x_extended = x_extended + self.mamba2_core(x_extended)
                x_extended = self.loop_norm(x_extended)

                # Bridge + slice (low-rank + residual)
                x_bridged = x_extended + self.bridge_up(self.bridge_down(x_extended))
                x_out = x_bridged[:, self.M:, :]
                lg = self.lm_head(self.norm(x_out, residual[:, self.M:, :],
                                             prenorm=False))
                p = torch.softmax(lg[0, -1, :].float(), dim=-1)
                tid = p.argmax().item()
                tok = tokenizer.decode([tid]).strip()
                trace.append((f"L{loop_i+1}", tok, round(p[tid].item(), 4)))
                if tid == HALT_ID:
                    trace[-1] = (f"L{loop_i+1}", "<HALT>", round(p[tid].item(), 4))
                    return loop_i + 1, trace, last_answer
                last_answer = tok
            return self.MAX_LOOPS, trace, last_answer


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 Warmup: Freeze Everything Except Memory + Bridge
# ══════════════════════════════════════════════════════════════════════════════

def freeze_for_phase1(model: RecursiveMamba2_PrefixScratchpad) -> None:
    """Phase 1 Warmup: freeze everything, unfreeze ONLY latent_memory + latent_bridge.

    This function locks down:
      - Entire Mamba-2 2.7B base backbone (requires_grad = False)
      - All LoRA adapters (the trained reasoning logic from step 3000)
      - The LM head
      - The lifeline gate
      - The loop engine (mamba2_core)
      - The loop norm
      - The embeddings

    It then sets requires_grad = True ONLY for:
      - self.latent_memory: The 16-token prefix scratchpad
      - self.latent_bridge: The System2→System1 translation matrix

    The optimizer will only update two things:
      1. The Zeros: Format the blank latent_memory vectors so Mamba gates
         accept them as valid scratch paper
      2. The Translator: Bend the latent_bridge so RLF loop output gets
         translated into Base Model vocabulary distribution

    Args:
        model: The RecursiveMamba2_PrefixScratchpad model instance
    """
    # Step 1: Freeze EVERYTHING
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: Unfreeze ONLY the scratchpad + bridge
    model.latent_memory.requires_grad = True
    for param in model.bridge_down.parameters():
        param.requires_grad = True
    for param in model.bridge_up.parameters():
        param.requires_grad = True

    # ── Report ────────────────────────────────────────────────────────────────
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    mem_p = model.latent_memory.numel()
    bridge_p = (sum(p.numel() for p in model.bridge_down.parameters())
                + sum(p.numel() for p in model.bridge_up.parameters()))

    print(f"\n{'='*70}")
    print(f"  PHASE 1 WARMUP — Scratchpad Initialization")
    print(f"{'='*70}")
    print(f"  Frozen:     {frozen:,} params (base + LoRA + loop engine + gate)")
    print(f"  Trainable:  {trainable:,} params:")
    print(f"    latent_memory: {mem_p:,} ({PREFIX_M} × {D_MODEL})")
    print(f"    latent_bridge: {bridge_p:,} ({D_MODEL}→{BRIDGE_RANK}→{D_MODEL})")
    print(f"  Optimizer targets ONLY: latent_memory + bridge")
    print(f"{'='*70}\n")


def get_phase1_optimizer(model: RecursiveMamba2_PrefixScratchpad) -> optim.AdamW:
    """Create Phase 1 optimizer for ONLY the scratchpad + bridge params.

    Args:
        model: The model with Phase 1 freeze applied

    Returns:
        AdamW optimizer targeting only latent_memory + latent_bridge
    """
    params = [
        {"params": [model.latent_memory], "lr": 1e-3, "weight_decay": 0.0},
        {"params": (list(model.bridge_down.parameters())
                    + list(model.bridge_up.parameters())),
         "lr": 5e-4, "weight_decay": 0.01},
    ]
    return optim.AdamW(params)
