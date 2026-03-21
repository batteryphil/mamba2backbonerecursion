"""
finetune_mamba2_v28.py — Mamba2-1.3B v28 (Latent Forcing)
============================================================
THE FIX: Progressive Loop Supervision ("Latent Forcing")

v27 problem: Dense Trajectory Supervision taught the model:
  "Output <THINK> five times, then the answer."
  → Backbone pre-computes answer; Mamba idles for 5 beats (Clever Hans).

v28 fix: Every loop gets a REAL target — an intermediate variable.
  For chain "A=red. B=A. C=B. What is C?":
    Loop 1 → target "A"    (model must isolate the anchor variable)
    Loop 2 → target "B"    (model must hop the pointer one step)
    Loop 3 → target "red"  (model must resolve the final value & halt)

  n_loops = n_hops per sample (not fixed MAX_LOOPS).
  The Mamba block CANNOT idle — the loss only goes down if it moves
  the chain pointer forward each tick. No more metronome. No <THINK>.

Architecture: identical to v27 (frozen bf16 backbone, LoRA, MIMO core).
Training data: same system2_logic_v1.json — but we parse chain structure
  at runtime to extract per-loop targets.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel, Mamba
import json
import random
import time
import os
import re
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID    = "state-spaces/mamba2-1.3b"
STEPS       = 100_000
BATCH_SIZE  = 4
ACCUM       = 4        # effective batch = 16
LOG_EVERY   = 50
BASE_SPLIT  = 24
LORA_RANK   = 8
MAX_LOOPS   = 6        # ceiling; actual loops = n_hops per sample
SEQ_LEN     = 256
RESUME_FROM = ""
SAVE_PATH   = "mamba2_13b_v28_latent_forcing.pt"

print(f"\n{'='*60}", flush=True)
print(f"  Mamba2-1.3B v28 — Latent Forcing", flush=True)
print(f"  n_loops = n_hops per sample | No <THINK> supervision", flush=True)
print(f"  Device: {DEVICE} | Steps={STEPS}", flush=True)
print(f"{'='*60}\n", flush=True)


# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token

# No <THINK> token in v28 — intermediate loops predict real variable names.
# We keep the token for backward compat with saved checkpoints only.
tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>"]})


def _parse_chain(text: str) -> list[str] | None:
    """Extract per-loop targets from a variable chain prompt.

    "A = red. B = A. C = B. What is C?\\nAnswer: red"
    → ["A", "B", "red"]  (len = n_hops)

    Returns None if the prompt is not a parseable chain.
    """
    # Match lines of the form "X = something"
    assignments = re.findall(r'([A-Za-z_]\w*)\s*=\s*(\S+?)[\.\n]', text)
    if len(assignments) < 2:
        return None

    # Build value map
    val: dict[str, str] = {}
    for var, expr in assignments:
        val[var] = expr

    # Resolve the chain iteratively
    chain_vars = [assignments[0][0]]  # anchor variable
    for var, expr in assignments[1:]:
        chain_vars.append(var)

    # Walk from first to last, collect per-loop targets
    # Loop i target = name of variable at step i (except last = resolved value)
    targets: list[str] = []
    for i, cvar in enumerate(chain_vars[:-1]):
        targets.append(chain_vars[i])   # intermediate: the variable name itself

    # Final loop: resolve the full chain to its root value
    final_var = chain_vars[-1]
    resolved  = val.get(final_var, final_var)
    visited   = set()
    while resolved in val and resolved not in visited:
        visited.add(resolved)
        resolved = val[resolved]
    targets.append(resolved)            # final: the actual value

    return targets if len(targets) >= 2 else None


def find_answer_start(ids: list[int]) -> int:
    """Return position of first answer token after 'Answer:' boundary."""
    _ans1 = tokenizer.encode("Answer:",  add_special_tokens=False)
    _ans2 = tokenizer.encode(" Answer:", add_special_tokens=False)
    _arr  = tokenizer.encode("# ->",    add_special_tokens=False)
    for boundary in (_ans1, _ans2, _arr):
        n = len(boundary)
        for i in range(len(ids) - n):
            if ids[i:i + n] == boundary:
                return i + n
    return -1


# ── LoRA Linear ───────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """LoRA adapter — bf16 A/B matching backbone dtype, zero mixing."""

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        """Initialize with same dtype as base weight (bf16)."""
        super().__init__()
        self.bias  = linear.bias
        d_out, d_in = linear.weight.shape
        dtype = linear.weight.dtype
        self.register_buffer("base_weight", linear.weight.data.clone())
        self.lora_A = nn.Parameter(torch.empty(rank, d_in, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, dtype=dtype))
        self.scale  = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A)

    @property
    def weight(self) -> torch.Tensor:
        """Fused weight in bf16 — no dtype mixing."""
        return self.base_weight + self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard linear forward."""
        return F.linear(x, self.weight, self.bias)


# ── Loop State Block ──────────────────────────────────────────────────────────
# mamba_ssm.Mamba: Triton-optimized selective scan (Mamba1).
# Replaces the custom JIT MIMO Phase Rotator.
# Benefits: ~5-10x faster (Triton kernels), data-dependent A/B/C matrices
# (state geometry adapts per sequence, better for Latent Forcing),
# simpler stateless per-loop call (SSM state handled internally).
# Tradeoff: no unit-circle |A|=1 guarantee — acceptable at N<=6 depth.


# ── Recursive Mamba2-1.3B (v28) ───────────────────────────────────────────────
class RecursiveMamba2_v28(nn.Module):
    """
    Mamba2-1.3B with Latent Forcing (v28).

    Key change from v27:
      - No <THINK> supervision. Each loop i gets target = chain_targets[i].
      - n_loops = n_hops per sample (variable, not fixed MAX_LOOPS).
      - Loss forces the Mamba phase rotators to DO the pointer arithmetic,
        not idle behind a countdown timer.
    """

    MAX_LOOPS: int = MAX_LOOPS

    def __init__(self, backbone: MambaLMHeadModel, lora_rank: int = 8):
        """Freeze backbone, inject LoRA on top 24 layers, add loop machinery."""
        super().__init__()
        self.backbone   = backbone.backbone
        self.lm_head    = backbone.lm_head
        self.top_layers = nn.ModuleList(
            [backbone.backbone.layers[i] for i in range(BASE_SPLIT, 48)]
        )
        self.norm   = backbone.backbone.norm_f
        d_model     = backbone.backbone.embedding.embedding_dim  # 2048

        # LoRA on in_proj and out_proj of all 24 top layers
        for layer in self.top_layers:
            mx = layer.mixer
            for attr in ("in_proj", "out_proj"):
                if hasattr(mx, attr):
                    setattr(mx, attr, LoRALinear(getattr(mx, attr),
                                                 rank=lora_rank,
                                                 alpha=lora_rank * 2.0))

        self.step_emb    = nn.Embedding(self.MAX_LOOPS, d_model).to(torch.bfloat16)
        self.loop_norm   = nn.RMSNorm(d_model).to(torch.bfloat16)
        # Mamba1 block as loop-state engine — Triton selective scan, bf16
        self.mamba3_core = Mamba(
            d_model=d_model, d_state=16, d_conv=4, expand=2
        ).to(torch.bfloat16)
        # Zero out_proj so block starts as near-identity (stable warmup)
        # Same principle as LoRA lora_B=zeros: learns from 0, not from random noise
        nn.init.zeros_(self.mamba3_core.out_proj.weight)


        n_lora = sum(p.numel() for n, p in self.named_parameters()
                     if p.requires_grad and "lora" in n.lower())
        n_new  = (sum(p.numel() for p in self.step_emb.parameters()) +
                  sum(p.numel() for p in self.loop_norm.parameters()) +
                  sum(p.numel() for p in self.mamba3_core.parameters()))
        print(f"  LoRA params:     {n_lora:,}")
        print(f"  Loop machinery:  {n_new:,}")
        print(f"  Total trainable: {n_lora + n_new:,}\n")

    def forward(
        self,
        input_ids:     torch.Tensor,
        chain_targets: list[list[str]] | None = None,  # per-batch per-loop targets
        ans_starts:    list[int] | None = None,
    ) -> tuple:
        """
        Training: chain_targets provided — n_loops = len(chain_targets[b]).
        Inference: chain_targets=None — runs MAX_LOOPS.
        """
        x = self.backbone.embedding(input_ids)
        residual = None
        for layer in self.backbone.layers[:BASE_SPLIT]:
            x, residual = layer(x, residual)

        # ── Training Path ──────────────────────────────────────────────────────
        if self.training and chain_targets is not None:
            B, max_len = input_ids.shape
            n_loops = max(len(t) for t in chain_targets)

            def run_lora_layers(x_in, res_in):
                """Gradient-checkpointed LoRA layers."""
                for layer in self.top_layers:
                    x_in, res_in = layer(x_in, res_in)
                return x_in, res_in

            step_losses: list[torch.Tensor] = []
            step_accs:   list[torch.Tensor] = []

            for loop_i in range(n_loops):
                step_vec = self.step_emb(
                    torch.tensor(loop_i, device=x.device)
                )
                x = x + step_vec
                x, residual = checkpoint(run_lora_layers, x, residual,
                                         use_reentrant=False)
                # Mamba1 selective scan — stateless per loop call, Triton kernels
                x = x + self.mamba3_core(x)
                x = self.loop_norm(x)

                x_normed    = self.norm(x, residual, prenorm=False)
                logits_step = self.lm_head(x_normed)  # (B, L, V)
                vocab_size  = logits_step.shape[-1]

                # Vectorized loss over batch — no tokenizer calls, targets pre-tokenized
                loop_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
                loop_acc  = torch.tensor(0.0, device=x.device)
                valid = 0

                for b in range(B):
                    as_ = ans_starts[b]
                    if as_ < 1 or as_ >= max_len:
                        continue
                    btgt   = chain_targets[b]
                    tgt_id = int(btgt[min(loop_i, len(btgt) - 1)])

                    # Pointer mask from pre-tokenized input
                    uniq = torch.unique(input_ids[b])
                    mask = torch.full((vocab_size,), float("-inf"), device=x.device)
                    mask[uniq] = 0.0

                    # Guard: skip if target not reachable through pointer mask
                    # (prevents cross_entropy computing log(softmax(-inf)) = inf)
                    if tgt_id >= vocab_size or mask[tgt_id].item() == float("-inf"):
                        continue

                    logits_b = logits_step[b, as_ - 1, :] + mask
                    pred_tok = logits_b.argmax().item()
                    tgt_t    = torch.tensor(tgt_id, device=x.device)
                    loop_loss = loop_loss + F.cross_entropy(
                        logits_b.unsqueeze(0), tgt_t.unsqueeze(0)
                    )
                    loop_acc = loop_acc + float(pred_tok == tgt_id)
                    valid   += 1


                if valid > 0:
                    step_losses.append(loop_loss / valid)
                    step_accs.append(loop_acc   / valid)

            avg_loss = (torch.stack(step_losses).mean()
                        if step_losses else
                        torch.tensor(0.0, device=x.device, requires_grad=True))
            avg_acc  = (torch.stack([a.clone().detach() for a in step_accs]).mean()
                        if step_accs else torch.tensor(0.0, device=x.device))
            final_loop_acc = step_accs[-1] if step_accs else avg_acc
            return avg_loss, avg_acc, final_loop_acc

        # ── Inference Path ─────────────────────────────────────────────────────
        else:
            vocab_size = self.lm_head.weight.shape[0]
            mask = torch.full((vocab_size,), float("-inf"), device=x.device)
            uniq = torch.unique(input_ids[0])
            mask[uniq] = 0.0

            trace: list[tuple] = []
            prev_tok = None

            for loop_i in range(self.MAX_LOOPS):
                sv = self.step_emb(torch.tensor(loop_i, device=x.device))
                x  = x + sv
                for layer in self.top_layers:
                    x, residual = layer(x, residual)
                x = x + self.mamba3_core(x)
                x = self.loop_norm(x)

                lg = self.lm_head(self.norm(x, residual, prenorm=False))
                lg[0, -1, :] = lg[0, -1, :] + mask
                p    = torch.softmax(lg[0, -1, :], dim=-1)
                tid  = p.argmax().item()
                tok  = tokenizer.decode([tid]).strip()
                prob = round(p.max().item(), 3)
                trace.append((f"L{loop_i+1}", tok, prob))

                # Halt: output same token twice (converged) or non-variable-like token
                if prev_tok is not None and tid == prev_tok:
                    return loop_i + 1, trace
                prev_tok = tid

            return self.MAX_LOOPS, trace


# ── Data Loading ──────────────────────────────────────────────────────────────
def _pretokenize_target(tgt_str: str, fallback: bool = True) -> int:
    """Tokenize a target string to its first token id. Cached-friendly."""
    for prefix in (" ", ""):
        ids = tokenizer.encode(prefix + tgt_str, add_special_tokens=False)
        if ids:
            return ids[0]
    return tokenizer.eos_token_id


def load_training_data() -> list[dict]:
    """Load data, parse chain structure, pre-tokenize targets & ans_starts.

    Pre-computation eliminates all tokenizer calls from the hot training loop.
    Multi-hop chains → progressive per-loop targets (Latent Forcing).
    Non-chain samples → 1-hop with answer as single target.
    """
    samples: list[dict] = []

    def _extract_answer(text: str) -> str | None:
        """Pull answer from 'Answer: X' line."""
        m = re.search(r'[Aa]nswer:\s*(\S+)', text)
        return m.group(1).rstrip('.,!?') if m else None

    def _prep(text: str, chain_targets: list[str]) -> dict | None:
        """Pre-tokenize targets matched against actual prompt tokens.

        Searches all prefix variants (' X', 'X') to find which token
        actually appears in the encoded prompt, guaranteeing the target
        is inside the pointer mask at training time.
        """
        enc      = tokenizer.encode(text, add_special_tokens=False)
        enc_set  = set(enc)
        ans_start = find_answer_start(enc)
        if ans_start < 1:
            return None
        tgt_ids: list[int] = []
        for tgt_str in chain_targets:
            matched = None
            for prefix in (" ", "", " "+tgt_str[0].upper()+tgt_str[1:], tgt_str.upper()):
                candidates = tokenizer.encode(prefix if prefix.startswith(" ") or prefix == "" else prefix+tgt_str,
                                              add_special_tokens=False)
                # also try just prefixing normally
                for pfx in (" ", ""):
                    cands = tokenizer.encode(pfx + tgt_str, add_special_tokens=False)
                    if cands and cands[0] in enc_set:
                        matched = cands[0]
                        break
                if matched is not None:
                    break
            if matched is None:
                # last resort: take first token of space-prefixed form
                cands = tokenizer.encode(" " + tgt_str, add_special_tokens=False)
                matched = cands[0] if cands else tokenizer.eos_token_id
            tgt_ids.append(matched)
        return {
            "text":          text,
            "hops":          len(chain_targets),
            "chain_targets": chain_targets,
            "chain_tgt_ids": tgt_ids,    # guaranteed in enc_set (or best effort)
            "ans_start":     ans_start,
        }


    if os.path.exists("system2_logic_v1.json"):
        with open("system2_logic_v1.json") as f:
            data = json.load(f)
        multi_hop = one_hop = skipped = 0
        for item in data:
            text = item.get("text", item.get("prompt", ""))
            if not text:
                continue
            chain_tgts = _parse_chain(text)
            if chain_tgts and len(chain_tgts) >= 2:
                s = _prep(text, chain_tgts)
                if s:
                    samples.append(s)
                    multi_hop += 1
                else:
                    skipped += 1
            else:
                ans = _extract_answer(text)
                if ans:
                    s = _prep(text, [ans])
                    if s:
                        samples.append(s)
                        one_hop += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1
        print(f"  Multi-hop chains: {multi_hop:,}  (Latent Forcing)")
        print(f"  Single-hop items: {one_hop:,}   (1-target supervision)")
        print(f"  Skipped:          {skipped:,}   (no parseable answer)")

    if os.path.exists("mmlu_format_v17.json"):
        with open("mmlu_format_v17.json") as f:
            mmlu = json.load(f)
        mmlu_added = 0
        for item in mmlu[:10_000]:
            text = item.get("text", "")
            ans  = _extract_answer(text)
            if ans:
                s = _prep(text, [ans])
                if s:
                    samples.append(s)
                    mmlu_added += 1
        print(f"  MMLU-format:      {mmlu_added:,}  (free-form answer targets, no letters)")

    # Free-form counterfactual samples (Phase 4 fix — no ABCD)
    counterfactuals = [
        {"text": "The sun is freezing cold. John's coffee is the temperature of the sun. What temperature is John's coffee?\nAnswer: cold",   "targets": ["cold"]},
        {"text": "Fire is icy cold. Bob touched fire. What did Bob feel?\nAnswer: cold",                                                        "targets": ["cold"]},
        {"text": "In this world dogs meow and cats bark. Sarah has a cat. What sound does her pet make?\nAnswer: bark",                         "targets": ["bark"]},
        {"text": "'heavy' means weighs nothing. 'light' means weighs a lot. A feather is very heavy. A boulder is very light. Which is easier to lift?\nAnswer: boulder", "targets": ["boulder"]},
    ]
    for cf in counterfactuals:
        s = _prep(cf["text"], cf["targets"])
        if s:
            samples.append(s)

    hop_dist: dict = {}
    for s in samples:
        h = s["hops"]
        hop_dist[h] = hop_dist.get(h, 0) + 1
    print(f"  Hop distribution: {dict(sorted(hop_dist.items()))}")
    print(f"  Total samples:    {len(samples):,}  (all targets pre-tokenized)\n")
    return samples



def make_batch(pool: list[dict], seed: int) -> tuple:
    """Sample a batch — targets are pre-tokenized at load time."""
    rng   = random.Random(seed)
    batch = rng.sample(pool, min(BATCH_SIZE, len(pool)))

    texts      = [s["text"]          for s in batch]
    ctgt_ids   = [s["chain_tgt_ids"] for s in batch]   # pre-tokenized
    ans_starts = [s["ans_start"]     for s in batch]   # pre-computed

    enc = tokenizer(texts, max_length=SEQ_LEN, truncation=True,
                    padding="max_length", return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)

    return input_ids, ctgt_ids, ans_starts


# ── Load Base Model ───────────────────────────────────────────────────────────
print(f"  Loading {MODEL_ID} (bfloat16)...", flush=True)
base_model = MambaLMHeadModel.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device=DEVICE
)

# Resize for <THINK> token (backward compat — token itself not used in training)
new_vocab = len(tokenizer)
old_vocab = base_model.backbone.embedding.weight.shape[0]
d_model   = base_model.backbone.embedding.embedding_dim

if new_vocab > old_vocab:
    print(f"  Resizing vocab {old_vocab} → {new_vocab}")
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
base_model.backbone.embedding.weight.requires_grad = True
base_model.lm_head.weight.requires_grad = True

model = RecursiveMamba2_v28(base_model, lora_rank=LORA_RANK).to(DEVICE)

# Optionally warm-start from v27 weights
if os.path.exists("mamba2_13b_finetuned_v27.pt"):
    ckpt = torch.load("mamba2_13b_finetuned_v27.pt", map_location=DEVICE)
    sd   = ckpt.get("model_state_dict", ckpt)
    sd_compat = {k: v for k, v in sd.items() if not k.startswith("mamba3_core")}
    missing, unexpected = model.load_state_dict(sd_compat, strict=False)
    print(f"  Warm-start from v27: LoRA+step_emb+loop_norm loaded | mamba3_core init fresh")
    print(f"  ({len(missing)} missing, {len(unexpected)} unexpected keys)")
else:
    print("  Starting from scratch (no v27 checkpoint found)")

if RESUME_FROM and os.path.exists(RESUME_FROM):
    ckpt = torch.load(RESUME_FROM, map_location=DEVICE)
    sd   = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    print(f"  Resumed from {RESUME_FROM}")


# ── Optimizer ─────────────────────────────────────────────────────────────────
if __name__ != "__main__":
    raise SystemExit(0)

samples = load_training_data()

# ── Train / Validation Split (stratified by hop count) ────────────────────────
random.seed(42)
train_samples: list[dict] = []
val_samples:   list[dict] = []

# Group by hop count, hold out 10% of each hop bucket
hop_buckets: dict[int, list] = {}
for s in samples:
    hop_buckets.setdefault(s["hops"], []).append(s)

for hop, bucket in sorted(hop_buckets.items()):
    random.shuffle(bucket)
    n_val = max(1, int(len(bucket) * 0.10))
    val_samples.extend(bucket[:n_val])
    train_samples.extend(bucket[n_val:])

print(f"  Train: {len(train_samples):,} | Val: {len(val_samples):,}  (10% stratified holdout)")
hop_val = {}
for s in val_samples:
    hop_val[s["hops"]] = hop_val.get(s["hops"], 0) + 1
print(f"  Val hop dist: {dict(sorted(hop_val.items()))}\n")


g1_params = (
    [model.step_emb.weight]
    + list(model.loop_norm.parameters())
    + list(model.mamba3_core.parameters())
    + [base_model.backbone.embedding.weight, base_model.lm_head.weight]
)
g1_ids    = {id(p) for p in g1_params}
g2_params = [p for p in model.parameters()
             if p.requires_grad and id(p) not in g1_ids]

optimizer = optim.AdamW([
    {"params": g1_params, "lr": 1e-3,  "weight_decay": 0.0},
    {"params": g2_params, "lr": 5e-4,  "weight_decay": 0.01},
])
scheduler = CosineAnnealingLR(optimizer, T_max=STEPS // ACCUM, eta_min=1e-6)

print(f"  Optimizer: g1={len(g1_params)} tensors @ 1e-3 | g2={len(g2_params)} LoRA tensors @ 5e-4")
print(f"\n{'─'*60}")
print(f"  v28 Latent Forcing | {STEPS} steps | Effective batch = {BATCH_SIZE * ACCUM}")
print(f"  Early stop: val AllLoopAcc >= 99.5%% for 3 consecutive val checks")
print(f"{'─'*60}")


# ── Validation Pass ───────────────────────────────────────────────────────────
VAL_EVERY        = 500   # steps between validation runs
EARLY_STOP_ACC   = 99.5  # % threshold for early stopping
EARLY_STOP_COUNT = 3     # consecutive checks at/above threshold to stop


def run_validation(val_pool: list[dict], n_batches: int = 20) -> tuple[float, float, float]:
    """Evaluate model on held-out val set. Returns (loss, AllLoopAcc, FinalAcc).

    Intentionally keeps model.training=True (not eval) so forward() uses the
    training branch that computes loss. torch.no_grad() skips gradient tracking.
    Dropout disabled implicitly since there's no dropout in this architecture.
    """
    v_loss = v_aa = v_fa = 0.0
    valid_batches = 0
    with torch.no_grad():
        for i in range(n_batches):
            try:
                input_ids, ctgt_ids, ans_starts = make_batch(
                    val_pool, seed=99999 + i
                )
                loss, avg_acc, final_acc = model(
                    input_ids,
                    chain_targets=ctgt_ids,
                    ans_starts=ans_starts,
                )
                if torch.isfinite(loss):
                    v_loss += loss.item()
                    v_aa   += avg_acc.item() * 100
                    v_fa   += final_acc.item() * 100
                    valid_batches += 1
            except Exception:
                pass
    model.train()
    if valid_batches == 0:
        return float("inf"), 0.0, 0.0
    return v_loss / valid_batches, v_aa / valid_batches, v_fa / valid_batches


# ── Training Loop ─────────────────────────────────────────────────────────────
model.train()
optimizer.zero_grad()
t0    = time.time()
total_loss = total_avg_acc = total_final_acc = 0.0

early_stop_hits     = 0  # consecutive val checks at/above EARLY_STOP_ACC
best_val_acc        = 0.0
last_train_aa: float = 0.0

for step in range(1, STEPS + 1):
    for accum_i in range(ACCUM):
        input_ids, ctgt_ids, ans_starts = make_batch(
            train_samples, seed=step * ACCUM + accum_i
        )
        loss, avg_acc, final_acc = model(
            input_ids,
            chain_targets=ctgt_ids,
            ans_starts=ans_starts,
        )
        (loss / ACCUM).backward()
        total_loss      += loss.item()
        total_avg_acc   += avg_acc.item() * 100
        total_final_acc += final_acc.item() * 100

    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0
    )
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    if step % LOG_EVERY == 0:
        n  = LOG_EVERY * ACCUM
        al = total_loss      / n
        aa = total_avg_acc   / n
        fa = total_final_acc / n
        last_train_aa = aa
        total_loss = total_avg_acc = total_final_acc = 0.0

        vram = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0
        tps  = (BATCH_SIZE * ACCUM * SEQ_LEN * LOG_EVERY) / (time.time() - t0)
        t0   = time.time()
        lr1  = optimizer.param_groups[0]["lr"]
        lr2  = optimizer.param_groups[1]["lr"]
        print(
            f"  Step {step:5d} | Loss: {al:.4f} | AllLoopAcc: {aa:5.1f}%"
            f" | FinalAcc: {fa:5.1f}%"
            f" | LR: {lr1:.1e}/{lr2:.1e} | TPS: {int(tps)} | VRAM: {vram:.2f}GB",
            flush=True,
        )

    # ── Validation + Early Stop Check ─────────────────────────────────────────
    if step % VAL_EVERY == 0:
        vl, vaa, vfa = run_validation(val_samples, n_batches=30)
        gap = last_train_aa - vaa
        overfit_flag = " ⚠️  OVERFIT" if gap > 5.0 else ""
        print(
            f"\n  ── VAL @ step {step} ──────────────────────────────────────",
            flush=True,
        )
        print(
            f"  Val  Loss: {vl:.4f} | Val AllLoopAcc: {vaa:5.1f}%"
            f" | Val FinalAcc: {vfa:5.1f}%",
            flush=True,
        )
        print(
            f"  Train-Val gap: {gap:+.1f}pp{overfit_flag}",
            flush=True,
        )

        if vaa > best_val_acc:
            best_val_acc = vaa
            best_path    = SAVE_PATH.replace(".pt", "_best.pt")
            torch.save({
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "step": step,
                "val_allloop_acc": vaa,
                "val_final_acc":   vfa,
            }, best_path)
            print(f"  🏆 New best val acc {vaa:.1f}% → {best_path}", flush=True)

        if vaa >= EARLY_STOP_ACC:
            early_stop_hits += 1
            print(
                f"  ⏱  Early-stop counter: {early_stop_hits}/{EARLY_STOP_COUNT}"
                f"  (val AllLoopAcc={vaa:.1f}% >= {EARLY_STOP_ACC}%)",
                flush=True,
            )
            if early_stop_hits >= EARLY_STOP_COUNT:
                print(
                    f"\n  ✅ EARLY STOP at step {step}."
                    f" Val AllLoopAcc={vaa:.1f}% for {EARLY_STOP_COUNT} consecutive checks.",
                    flush=True,
                )
                torch.save({
                    "model_state_dict":     model.state_dict(),
                    "step": step,
                    "val_allloop_acc": vaa,
                }, SAVE_PATH)
                break
        else:
            if early_stop_hits > 0:
                print(f"  Early-stop counter reset (was {early_stop_hits})", flush=True)
            early_stop_hits = 0
        print(f"  {'─'*54}\n", flush=True)

    if step % 200 == 0:
        ckpt_path = SAVE_PATH.replace(".pt", f"_step{step}.pt")
        torch.save({
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
        }, ckpt_path)
        print(f"  💾 Checkpoint → {ckpt_path}", flush=True)

else:
    torch.save({"model_state_dict": model.state_dict(), "step": STEPS}, SAVE_PATH)
    print(f"\n✅ v28 complete (full run) — saved to {SAVE_PATH}\n")


# ── Quick inference test ───────────────────────────────────────────────────────
model.eval()
tests = [
    "A = red. B = A. What is B?\nAnswer:",
    "A = red. B = A. C = B. What is C?\nAnswer:",
    "X = apple. Y = X. Z = Y. W = Z. What is W?\nAnswer:",
    "Fire is icy cold. Bob touched fire. What did Bob feel?\nAnswer:",
]
print("=" * 60)
with torch.no_grad():
    for prompt in tests:
        ids_ = tokenizer.encode(prompt, add_special_tokens=False,
                                return_tensors="pt").to(DEVICE)
        loops, trace = model(ids_)
        chain_str = " → ".join(f"{t[0]}:{t[1]}" for t in trace)
        print(f"  Q: {prompt.splitlines()[0][:55]!r}")
        print(f"  {chain_str}")
        print(f"  Final: {trace[-1][1]!r}  ({loops} loops)\n")
print("=" * 60)
