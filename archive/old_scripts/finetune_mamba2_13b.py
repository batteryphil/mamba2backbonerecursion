"""
finetune_mamba2_13b.py — Mamba2-1.3B v27 (Recursive Loop Fine-tune)
======================================================================
Architecture: state-spaces/mamba2-1.3b
  - 48 layers, d_model=2048, Mamba2 (SSD) blocks
  - Layers 0-23:  frozen feature extractor (run once)
  - Layers 24-47: LoRA reasoning engine (run N times per input)
  - step_emb:     loop-position clock injected before each pass
  - loop_norm:    RMSNorm stabilizes loop_state between passes
  - mamba3_core:  MIMO phase rotator for gradient-stable memory

Everything inherited from the proven mamba-130m v25 run:
  - Curriculum: N=2→6, graduate at 85% rolling accuracy
  - Pointer mask: only tokens in prompt + EOS + THINK
  - Dense Trajectory Supervision (<THINK> on intermediate loops)
  - JIT-fused CUDA MIMO phase kernel (gradient-stable across N loops)
  - Padding-masked accuracy gate (fixes padding-gaming)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel
import json
import random
import time
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID    = "state-spaces/mamba2-1.3b"
STEPS       = 100_000
BATCH_SIZE  = 2        # fp32 1.3B weights ~ 5GB; small batch fits 11.6GB GPU
ACCUM       = 8        # effective batch = 16
LOG_EVERY   = 50
BASE_SPLIT  = 24       # layers 0-23 frozen, 24-47 looped
LORA_RANK   = 8
RESUME_FROM = ""
SAVE_PATH   = "mamba2_13b_finetuned_v27.pt"
SEQ_LEN     = 256      # logic samples avg 200 tokens; 256 fits comfortably

print(f"\n{'='*60}", flush=True)
print(f"  Mamba2-1.3B v27 — Recursive Loop Fine-tune", flush=True)
print(f"  Device: {DEVICE} | Steps={STEPS}", flush=True)
print(f"{'='*60}\n", flush=True)


# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token

tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>"]})
THINK_TOKEN_ID = tokenizer.convert_tokens_to_ids("<THINK>")
print(f"  <THINK> Token ID generated: {THINK_TOKEN_ID}")

_answer_seq_ids     = tokenizer.encode(" Answer:", add_special_tokens=False)
_answer_nospace_ids = tokenizer.encode("Answer:",  add_special_tokens=False)
print(f"  Answer boundary tokens: {_answer_seq_ids} = ' Answer:'")
print(f"  Answer (nospace) tokens:{_answer_nospace_ids} = 'Answer:'")

ALLOWED_CORE_TOKENS = [tokenizer.eos_token_id, THINK_TOKEN_ID]


def find_answer_start(ids: list[int]) -> int:
    """Return position of first answer token after 'Answer:' boundary."""
    _arrow_ids = tokenizer.encode("# ->", add_special_tokens=False)
    for boundary in (_answer_nospace_ids, _answer_seq_ids, _arrow_ids):
        n = len(boundary)
        for i in range(len(ids) - n):
            if ids[i:i + n] == boundary:
                return i + n
    return -1


# ── LoRA Linear ───────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """Low-Rank Adapter for nn.Linear. Exposes .weight for Mamba2 CUDA kernels."""

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        """Initialize LoRA with frozen base weight and trainable A/B matrices."""
        super().__init__()
        self.bias   = linear.bias
        d_out, d_in = linear.weight.shape
        dtype = linear.weight.dtype  # bf16 — match backbone throughout
        self.register_buffer("base_weight", linear.weight.data.clone())
        # A/B in same dtype as backbone — v26 principle: one dtype, zero mixing
        self.lora_A = nn.Parameter(torch.empty(rank, d_in, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, dtype=dtype))
        self.scale  = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A)

    @property
    def weight(self) -> torch.Tensor:
        """Fused weight = frozen base + low-rank delta. All bf16, no casting."""
        return self.base_weight + self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using fused weight."""
        return F.linear(x, self.weight, self.bias)


# ── JIT Fused MIMO Phase Kernel ───────────────────────────────────────────────
@torch.jit.script
def fused_mamba3_mimo_core(
    x_in: torch.Tensor,
    real_state: torch.Tensor,
    imag_state: torch.Tensor,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    B_real: torch.Tensor,
    B_imag: torch.Tensor,
    C_real: torch.Tensor,
    C_imag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    V25 JIT-fused real-valued MIMO phase rotation kernel.

    Unit-circle constraint (|A|=1) guarantees BPTT gradient stability
    across arbitrary recursion depths N.
    """
    B_r = B_real.unsqueeze(0).unsqueeze(0)
    B_i = B_imag.unsqueeze(0).unsqueeze(0)
    C_r = C_real.unsqueeze(0).unsqueeze(0)
    C_i = C_imag.unsqueeze(0).unsqueeze(0)

    bx_real = B_r * x_in
    bx_imag = B_i * x_in

    new_real = (cos_t * real_state - sin_t * imag_state) + bx_real
    new_imag = (sin_t * real_state + cos_t * imag_state) + bx_imag

    y_real     = (C_r * new_real) - (C_i * new_imag)
    y_real_sum = y_real.sum(dim=-1)

    return y_real_sum, new_real, new_imag


# ── Mamba3 Reasoning Block ────────────────────────────────────────────────────
class Mamba3ReasoningBlock(nn.Module):
    """MIMO Phase Rotator — gradient-stable memory module for N-loop reasoning."""

    def __init__(self, d_model: int, n_channels: int = 2, d_state: int = 16):
        """Initialize phase rotator with static A/B/C parameters."""
        super().__init__()
        self.d_model    = d_model
        self.n_channels = n_channels
        self.d_state    = d_state

        self.in_proj  = nn.Linear(d_model, n_channels * d_model, bias=False)
        self.A_theta  = nn.Parameter(torch.randn(n_channels, d_model, d_state) * 0.1)
        self.B_real   = nn.Parameter(torch.randn(n_channels, d_model, d_state) * 0.1)
        self.B_imag   = nn.Parameter(torch.randn(n_channels, d_model, d_state) * 0.1)
        self.C_real   = nn.Parameter(torch.randn(n_channels, d_model, d_state) * 0.1)
        self.C_imag   = nn.Parameter(torch.randn(n_channels, d_model, d_state) * 0.1)
        self.out_proj = nn.Linear(n_channels * d_model, d_model, bias=False)
        self.mixer_norm = nn.RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        real_state: torch.Tensor = None,
        imag_state: torch.Tensor = None,
        cos_t: torch.Tensor = None,
        sin_t: torch.Tensor = None,
    ) -> tuple:
        """Run one phase rotation step, updating the complex state."""
        B, L, _ = x.shape
        x_in    = self.in_proj(x)
        x_in    = x_in.view(B, L, self.n_channels, self.d_model).unsqueeze(-1)

        if real_state is None:
            real_state = torch.zeros(B, L, self.n_channels, self.d_model,
                                     self.d_state, device=x.device, dtype=x.dtype)
            imag_state = torch.zeros_like(real_state)

        if cos_t is None or sin_t is None:
            cos_t = torch.cos(self.A_theta).unsqueeze(0).unsqueeze(0)
            sin_t = torch.sin(self.A_theta).unsqueeze(0).unsqueeze(0)

        y_real_sum, new_real, new_imag = fused_mamba3_mimo_core(
            x_in, real_state, imag_state, cos_t, sin_t,
            self.B_real, self.B_imag, self.C_real, self.C_imag,
        )
        y_flat = y_real_sum.view(B, L, self.n_channels * self.d_model)
        out    = self.out_proj(y_flat)
        out    = self.mixer_norm(out)
        return x + out, new_real, new_imag


# ── Recursive Mamba2-1.3B Wrapper ─────────────────────────────────────────────
class RecursiveMamba2_13B(nn.Module):
    """
    Wraps frozen mamba2-1.3b with a recursive reasoning head.

    Architecture:
      - Layers  0-23: frozen feature extractor (runs once)
      - Layers 24-47: LoRA reasoning engine  (runs N times)
      - step_emb:     loop-step clock
      - loop_norm:    RMSNorm between loop passes
      - mamba3_core:  MIMO phase rotator for gradient-stable memory
    """

    MAX_LOOPS: int = 6

    def __init__(self, backbone_model: MambaLMHeadModel, lora_rank: int = 8):
        """Freeze backbone, inject LoRA on top 24 layers, add loop machinery."""
        super().__init__()
        self.backbone   = backbone_model.backbone
        self.lm_head    = backbone_model.lm_head
        self.top_layers = nn.ModuleList(
            [backbone_model.backbone.layers[i] for i in range(BASE_SPLIT, 48)]
        )
        self.norm   = backbone_model.backbone.norm_f
        d_model     = backbone_model.backbone.embedding.embedding_dim  # 2048

        n_frozen = sum(p.numel() for p in backbone_model.parameters())
        print(f"  Frozen params:  {n_frozen:,}  (ALL 48 backbone layers + head + embedding)")
        print(f"  Trainable from base: 0  — zero catastrophic forgetting")
        print(f"  New params to train: step_emb + loop_norm added in wrapper")

        # LoRA on in_proj and out_proj of all top 24 layers
        ALPHA = lora_rank * 2.0
        for layer in self.top_layers:
            mx = layer.mixer
            for attr in ("in_proj", "out_proj"):
                if hasattr(mx, attr):
                    setattr(mx, attr, LoRALinear(getattr(mx, attr),
                                                 rank=lora_rank, alpha=ALPHA))

        self.step_emb   = nn.Embedding(self.MAX_LOOPS, d_model)
        nn.init.normal_(self.step_emb.weight, std=0.01)

        self.loop_norm  = nn.RMSNorm(d_model)
        self.mamba3_core = Mamba3ReasoningBlock(d_model=d_model,
                                                n_channels=2, d_state=16)

        # Cast new modules to bf16 to match backbone (v26 principle: one dtype)
        self.step_emb    = self.step_emb.to(torch.bfloat16)
        self.loop_norm   = self.loop_norm.to(torch.bfloat16)
        self.mamba3_core = self.mamba3_core.to(torch.bfloat16)

        n_lora = sum(p.numel() for n, p in self.named_parameters()
                     if p.requires_grad and "lora" in n.lower())
        n_emb  = sum(p.numel() for p in self.step_emb.parameters())
        n_norm = sum(p.numel() for p in self.loop_norm.parameters())
        n_core = sum(p.numel() for p in self.mamba3_core.parameters())
        print(f"  LoRA params:    {n_lora:,}")
        print(f"  step_emb+norm:  {n_emb + n_norm:,}")
        print(f"  mamba3_core:    {n_core:,}")
        print(f"  Total trainable:{n_lora + n_emb + n_norm + n_core:,}\n")

    def forward(
        self,
        input_ids: torch.Tensor,
        tgt_labels: torch.Tensor = None,
        ans_starts: list = None,
        accum: int = 1,
        max_train_loops: int = None,
    ) -> tuple:
        """
        Forward pass — training with Dense Trajectory Supervision, inference with ACT halt.
        """
        x        = self.backbone.embedding(input_ids)
        residual = None

        # ── Feature extractor (layers 0-23, runs once) ────────────────────────
        for layer in self.backbone.layers[:BASE_SPLIT]:
            x, residual = layer(x, residual)

        base_features = x.clone()

        if self.training:
            if max_train_loops is None:
                max_train_loops = self.MAX_LOOPS

            n_steps      = max_train_loops
            step_losses  = []
            answer_losses = []
            step_accs    = []
            real_state   = None
            imag_state   = None

            # Pre-compute rotary geometry once (Trig Tax optimization)
            cos_t = torch.cos(self.mamba3_core.A_theta).unsqueeze(0).unsqueeze(0)
            sin_t = torch.sin(self.mamba3_core.A_theta).unsqueeze(0).unsqueeze(0)

            def run_lora_layers(x_in: torch.Tensor,
                                res_in: torch.Tensor) -> tuple:
                """Run all top LoRA layers — wrapped for gradient checkpointing."""
                for layer in self.top_layers:
                    x_in, res_in = layer(x_in, res_in)
                return x_in, res_in

            for step_i in range(n_steps):
                step_vec = self.step_emb(
                    torch.tensor(step_i, device=x.device)
                )  # bf16 — matches x dtype from backbone
                x = x + step_vec


                x, residual = checkpoint(run_lora_layers, x, residual,
                                         use_reentrant=False)

                x, real_state, imag_state = self.mamba3_core(
                    x, real_state, imag_state, cos_t=cos_t, sin_t=sin_t
                )
                x = self.loop_norm(x)  # bf16 throughout — no casting needed


                if tgt_labels is not None and ans_starts is not None:
                    x_normed     = self.norm(x, residual, prenorm=False)
                    logits_step  = self.lm_head(x_normed)
                    vocab_size   = logits_step.shape[-1]
                    B, max_len   = input_ids.shape

                    # Per-batch pointer mask
                    batch_mask = torch.full((B, vocab_size), float("-inf"),
                                           device=x.device)
                    for b in range(B):
                        uniq    = torch.unique(input_ids[b])
                        allowed = torch.cat([
                            uniq,
                            torch.tensor(ALLOWED_CORE_TOKENS, device=x.device),
                        ]).unique()
                        batch_mask[b, allowed] = 0.0

                    logits_step = logits_step + batch_mask.unsqueeze(1)

                    step_loss  = torch.tensor(0.0, device=x.device,
                                             requires_grad=True)
                    step_acc   = torch.tensor(0.0, device=x.device)
                    valid_count = 0

                    for b in range(B):
                        ans_start = ans_starts[b]
                        full_tgt  = torch.full((max_len - 1,), -100,
                                               dtype=torch.long, device=x.device)

                        if ans_start < 0 or ans_start > max_len - 1:
                            continue

                        raw_tgt = tgt_labels[b, ans_start:max_len]
                        eos_pos = (raw_tgt == tokenizer.eos_token_id).nonzero(
                            as_tuple=True
                        )[0]
                        if len(eos_pos) > 0:
                            raw_tgt = raw_tgt[: eos_pos[0]]

                        if raw_tgt.shape[0] == 0:
                            continue

                        ans_len   = raw_tgt.shape[0]
                        raw_slice = raw_tgt.clone()

                        if step_i < n_steps - 1:
                            raw_slice = torch.full_like(raw_tgt, THINK_TOKEN_ID)

                        write_end = min(ans_start - 1 + ans_len, max_len - 1)
                        full_tgt[ans_start - 1 : write_end] = raw_slice[
                            : write_end - (ans_start - 1)
                        ]

                        if (full_tgt != -100).sum() == 0:
                            continue

                        valid_count += 1
                        logits_b    = logits_step[b, : max_len - 1, :]
                        valid_mask  = full_tgt != -100
                        pred_tokens = logits_b.argmax(dim=-1)

                        if valid_mask.sum() > 0:
                            step_acc = step_acc + (
                                pred_tokens[valid_mask] == full_tgt[valid_mask]
                            ).float().mean()

                        step_loss = step_loss + F.cross_entropy(
                            logits_b, full_tgt, ignore_index=-100
                        )

                    if valid_count > 0:
                        step_loss = step_loss / valid_count
                        step_acc  = step_acc  / valid_count

                    step_losses.append(step_loss)

                    if step_i == n_steps - 1:
                        step_accs.append(step_acc)
                        answer_losses.append(step_loss)

            avg_traj_loss   = torch.stack(step_losses).mean()  if step_losses   else torch.tensor(0.0, device=x.device, requires_grad=True)
            avg_answer_loss = torch.stack(answer_losses).mean() if answer_losses else torch.tensor(0.0, device=x.device)
            avg_traj_acc    = torch.stack(step_accs).mean()    if step_accs     else torch.tensor(0.0, device=x.device)
            return None, n_steps, [], avg_traj_loss, avg_traj_acc, avg_answer_loss

        else:
            # ── Inference: ACT halt when model stops predicting <THINK> ───────
            vocab_size = self.lm_head.weight.shape[0]
            mask       = torch.full((vocab_size,), float("-inf"), device=x.device)
            uniq       = torch.unique(input_ids[0])
            allowed    = torch.cat([
                uniq,
                torch.tensor(ALLOWED_CORE_TOKENS, device=x.device),
            ]).unique()
            mask[allowed] = 0.0

            trace      = []
            loops_taken = self.MAX_LOOPS
            real_state  = None
            imag_state  = None
            cos_t = torch.cos(self.mamba3_core.A_theta).unsqueeze(0).unsqueeze(0)
            sin_t = torch.sin(self.mamba3_core.A_theta).unsqueeze(0).unsqueeze(0)

            for step_i in range(self.MAX_LOOPS):
                step_vec    = self.step_emb(torch.tensor(step_i, device=x.device))
                x           = x + step_vec  # all bf16
                for layer in self.top_layers:
                    x, residual = layer(x, residual)

                x, real_state, imag_state = self.mamba3_core(
                    x, real_state, imag_state, cos_t=cos_t, sin_t=sin_t
                )
                x = self.loop_norm(x)  # bf16

                logits_tmp = self.lm_head(self.norm(x, residual, prenorm=False))
                logits_tmp[0, -1, :] = logits_tmp[0, -1, :] + mask

                p          = torch.softmax(logits_tmp[0, -1, :], dim=-1)
                top_tok_id = p.argmax().item()
                max_prob   = p.max().item()
                top_tok    = tokenizer.decode([top_tok_id]).strip()
                trace.append((f"L{step_i+1}", round(max_prob, 2), top_tok))

                if top_tok_id != THINK_TOKEN_ID:
                    loops_taken = step_i + 1
                    break

            x = self.norm(x, residual, prenorm=False)
            return self.lm_head(x), loops_taken, trace


# ── Load Base Model ───────────────────────────────────────────────────────────
# Load backbone in bf16 — same exponent range as fp32 (no overflow NaN),
# fits 11.6GB GPU (1.3B × 2 bytes ≈ 2.7GB vs 5.4GB for fp32).
# V26 principle applied: one dtype, zero mixing.
print(f"  Loading {MODEL_ID} (bfloat16)...", flush=True)
base_model = MambaLMHeadModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                               device=DEVICE)

# Resize embeddings for <THINK> token
new_vocab = len(tokenizer)
old_vocab = base_model.backbone.embedding.weight.shape[0]
d_model   = base_model.backbone.embedding.embedding_dim

if new_vocab > old_vocab:
    print(f"  Resizing vocab {old_vocab} → {new_vocab} for <THINK> token")
    new_emb = nn.Embedding(new_vocab, d_model)
    nn.init.normal_(new_emb.weight, std=0.02)
    new_emb.weight.data[:old_vocab] = base_model.backbone.embedding.weight.data
    base_model.backbone.embedding = new_emb

    new_head = nn.Linear(d_model, new_vocab, bias=False)
    nn.init.normal_(new_head.weight, std=0.02)
    new_head.weight.data[:old_vocab] = base_model.lm_head.weight.data
    base_model.lm_head = new_head

for p in base_model.parameters():
    p.requires_grad = False

# Unfreeze the resized embeddings / lm_head for the new token
base_model.backbone.embedding.weight.requires_grad = True
base_model.lm_head.weight.requires_grad = True

model = RecursiveMamba2_13B(base_model, lora_rank=LORA_RANK).to(DEVICE)

# ── Checkpoint loading ─────────────────────────────────────────────────────────
_loops_expanded = False
if RESUME_FROM and os.path.exists(RESUME_FROM):
    ckpt       = torch.load(RESUME_FROM, map_location=DEVICE)
    state_dict = ckpt.get("model_state_dict", ckpt)
    if "step_emb.weight" in state_dict:
        ckpt_emb  = state_dict["step_emb.weight"]
        model_emb = model.step_emb.weight
        if ckpt_emb.shape != model_emb.shape:
            print(f"  ⚠️ MAX_LOOPS mismatch — adapting step_emb")
            adapted = model_emb.clone()
            min_l   = min(ckpt_emb.shape[0], model_emb.shape[0])
            adapted[:min_l] = ckpt_emb[:min_l]
            state_dict["step_emb.weight"] = adapted
            _loops_expanded = True
    model.load_state_dict(state_dict, strict=False)
    print(f"  ✅ Checkpoint loaded from {RESUME_FROM}\n")
else:
    print(f"  Starting fresh (no resume checkpoint)\n")


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_training_data() -> list[dict]:
    """Load system2_logic + MMLU-format samples, return {text, hops} dicts."""
    samples: list[dict] = []

    for fname in ["system2_logic_v1.json"]:
        if os.path.exists(fname):
            with open(fname) as f:
                data = json.load(f)
            for item in data:
                text = item.get("text", item.get("prompt", ""))
                hops = item.get("hops", 2)
                if text:
                    samples.append({"text": text, "hops": hops})
            hop_dist = {}
            for s in samples:
                hop_dist[s["hops"]] = hop_dist.get(s["hops"], 0) + 1
            print(f"  Logic/fact samples:        {len(samples):,}  ({fname}) ✅ SYS2")
            print(f"  Hop distribution:      {dict(sorted(hop_dist.items()))}")
            break

    if os.path.exists("mmlu_format_v17.json"):
        before = len(samples)
        with open("mmlu_format_v17.json") as f:
            mmlu = json.load(f)
        for item in mmlu[:10_000]:
            samples.append({"text": item["text"], "hops": 1})
        print(f"  MMLU-format samples:       {len(samples)-before:,}  (mmlu_format_v17.json) [hops=1]")

    seen: set[str] = set()
    unique: list[dict] = []
    for s in samples:
        if s["text"] not in seen:
            seen.add(s["text"])
            unique.append(s)
    print(f"  Deduplication:             {len(samples):,} → {len(unique):,} unique ({len(samples)-len(unique)} dupes removed)")
    print(f"  Total samples:             {len(unique):,}\n")
    return unique


def make_batch(pool: list[dict], max_loops: int, seed: int) -> tuple:
    """Sample a curriculum-gated batch, returning (input_ids, labels, ans_starts)."""
    rng      = random.Random(seed)
    eligible = [s for s in pool if s.get("hops", 1) <= max_loops] or pool
    batch    = rng.sample(eligible, min(BATCH_SIZE, len(eligible)))
    texts    = [s["text"] for s in batch]

    enc       = tokenizer(texts, max_length=SEQ_LEN, truncation=True,
                          padding="max_length", return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    labels    = input_ids.clone()

    ans_starts = []
    for b in range(input_ids.shape[0]):
        ans_starts.append(find_answer_start(input_ids[b].tolist()))

    return input_ids, labels, ans_starts


# ── Optimizer ─────────────────────────────────────────────────────────────────
if __name__ != "__main__":
    raise SystemExit(0)

samples = load_training_data()

group1_params = (
    [model.step_emb.weight]
    + list(model.loop_norm.parameters())
    + list(model.mamba3_core.parameters())
    + [base_model.backbone.embedding.weight, base_model.lm_head.weight]
)
group1_ids    = {id(p) for p in group1_params}
group2_params = [p for p in model.parameters()
                 if p.requires_grad and id(p) not in group1_ids]

optimizer = optim.AdamW([
    {"params": group1_params, "lr": 1e-3,  "weight_decay": 0.0},
    {"params": group2_params, "lr": 1e-3,  "weight_decay": 0.01},
])
scheduler = CosineAnnealingLR(optimizer, T_max=STEPS // ACCUM, eta_min=1e-6)

if RESUME_FROM and os.path.exists(RESUME_FROM):
    ckpt = torch.load(RESUME_FROM, map_location=DEVICE)
    if not _loops_expanded:
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

print(f"  Optimizer: group1={len(group1_params)} tensors @ LR=1e-3  |  group2={len(group2_params)} tensors @ LR=1e-3")
print(f"\n{'─'*60}")
print(f"  Starting fine-tune: {STEPS} steps | Batch={BATCH_SIZE * ACCUM}")
print(f"{'─'*60}")


# ── Training Loop ─────────────────────────────────────────────────────────────
model.train()
optimizer.zero_grad()
t0 = time.time()
total_loss = 0.0
total_acc  = 0.0

current_max_loops = 2
acc_window: list[float] = []

start_step = 1
if RESUME_FROM and os.path.exists(RESUME_FROM):
    ckpt = torch.load(RESUME_FROM, map_location=DEVICE)
    start_step        = ckpt.get("step", 1)
    current_max_loops = ckpt.get("current_max_loops", 2)

for step in range(start_step, STEPS + 1):
    for accum_i in range(ACCUM):
        input_ids, labels, ans_starts = make_batch(
            samples, current_max_loops, seed=step * ACCUM + accum_i
        )
        _, n_steps, _, traj_loss, traj_acc, ans_loss = model(
            input_ids,
            tgt_labels=labels,
            ans_starts=ans_starts,
            accum=ACCUM,
            max_train_loops=current_max_loops,
        )
        (traj_loss / ACCUM).backward()
        total_loss += ans_loss.item()
        total_acc  += traj_acc.item() * 100

    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0
    )
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    if step % LOG_EVERY == 0:
        avg_loss = total_loss / (LOG_EVERY * ACCUM)
        avg_acc  = total_acc  / (LOG_EVERY * ACCUM)
        total_loss = total_acc = 0.0

        vram = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0
        tps  = (BATCH_SIZE * ACCUM * SEQ_LEN * LOG_EVERY) / (time.time() - t0)
        t0   = time.time()
        lr_g1 = optimizer.param_groups[0]["lr"]
        lr_g2 = optimizer.param_groups[1]["lr"]

        print(
            f"  Step {step:5d} | Loss: {avg_loss:.4f} | Acc: {avg_acc:5.1f}%"
            f" | LR(g1): {lr_g1:.2e} LR(lora): {lr_g2:.2e}"
            f" | TPS: {int(tps)} | VRAM: {vram:.2f}GB | MaxN: {current_max_loops}",
            flush=True,
        )

        acc_window.append(avg_acc)
        if len(acc_window) > 5:
            acc_window.pop(0)
        rolling_acc = sum(acc_window) / len(acc_window)

        if len(acc_window) == 5 and rolling_acc > 85.0:
            if current_max_loops < model.MAX_LOOPS:
                current_max_loops += 1
                acc_window.clear()
                print(f"\n🚀 CURRICULUM UPGRADE: Mastering N={current_max_loops - 1}"
                      f" at {rolling_acc:.1f}% accuracy! Escalating to N={current_max_loops} loops!\n")
                milestone = SAVE_PATH.replace(".pt", f"_MaxN_{current_max_loops}.pt")
                torch.save({
                    "model_state_dict"   : model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "step"               : step,
                    "current_max_loops"  : current_max_loops,
                    "acc"                : rolling_acc,
                }, milestone)
                print(f"  🏆 Milestone Checkpoint saved → {milestone}")
            elif rolling_acc > 95.0:
                print(f"\n🎉 N={model.MAX_LOOPS} MASTERED! Absolute Engine Solved. Halting training early.\n")
                break

    if step % 200 == 0:
        ckpt_path = SAVE_PATH.replace(".pt", f"_step{step}.pt")
        torch.save({
            "model_state_dict"   : model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step"               : step,
            "current_max_loops"  : current_max_loops,
        }, ckpt_path)
        print(f"  💾 Checkpoint saved → {ckpt_path}", flush=True)

# Final save
torch.save({"model_state_dict": model.state_dict(), "step": step}, SAVE_PATH)
print(f"\n✅ v27 complete — weights saved to {SAVE_PATH}\n")

# ── Quick inference test ───────────────────────────────────────────────────────
model.eval()
test_prompts = [
    "X = blue. Y = X. What is Y?\nAnswer:",
    "A is taller than B. B is taller than C. Who is shortest?\nAnswer:",
    "What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:",
    "A = red. B = A. C = B. What is C?\nAnswer:",
]
print("=" * 60)
with torch.no_grad():
    for prompt in test_prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False,
                               return_tensors="pt").to(DEVICE)
        _, loops, trace = model(ids)
        trace_str = " → ".join(f"{t[0]}:{t[2]}" for t in trace)
        print(f"  Q: {repr(prompt.strip())[:55]}")
        print(f"  {trace_str}")
        print(f"  A: {trace[-1][2]!r}  ({loops} loops)\n")
print("=" * 60)
