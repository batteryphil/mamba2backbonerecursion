"""
finetune_mamba_130m_v29.py — Mamba-130m v29 (Latent Forcing + <HALT> Token)
============================================================================
v28 proved Latent Forcing works. v29 makes halting a LEARNED BEHAVIOR.

The key change from v28:
  v28: halt when Python script detects token repeat (fragile heuristic)
  v29: train model to PREDICT <HALT> after the answer → model decides when done

Training targets for a 3-hop chain (X=Apple, Y=X, Z=Y. What is Z?):
  Loop 0: "X"       ← pointer at start variable
  Loop 1: "Y"       ← pointer moves one hop
  Loop 2: "Apple"   ← resolved value (the answer)
  Loop 3: "<HALT>"  ← model learns: I am done, stop computing

At inference: run loops until model predicts <HALT>, return previous token.
<HALT> is added to the pointer mask — it is always a reachable prediction.

This gives the model genuine Adaptive Computation Time (ACT):
  - Easy 1-hop questions: HALT at loop 2 (answer at L1, HALT at L2)
  - Hard 3-hop questions: HALT at loop 4 (answer at L3, HALT at L4)
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
from torch.utils.checkpoint import checkpoint as grad_ckpt

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID    = "state-spaces/mamba-130m"
BASE_SPLIT  = 6
LORA_RANK   = 8
MAX_LOOPS   = 8        # +1 vs v28 to give room for HALT loop
SEQ_LEN     = 256
BATCH_SIZE  = 8
ACCUM       = 4        # eff batch = 32
STEPS       = 50_000
LOG_EVERY   = 50
VAL_EVERY   = 500
EARLY_STOP_ACC   = 99.0   # Slightly lower because HALT accuracy counts
EARLY_STOP_COUNT = 3
RESUME_FROM = "mamba_130m_v28_latent_forcing_best.pt"  # Warm-start from v28
SAVE_PATH   = "mamba_130m_v29_halt.pt"

print(f"\n{'='*60}", flush=True)
print(f"  Mamba-130m v29 — Latent Forcing + <HALT> Token", flush=True)
print(f"  chain_targets = [var1, ..., answer, <HALT>]", flush=True)
print(f"  Inference halts when model predicts <HALT>", flush=True)
print(f"  Device: {DEVICE} | Steps={STEPS}", flush=True)
print(f"{'='*60}\n", flush=True)


# ── Tokenizer — add BOTH <THINK> (compat) and <HALT> ─────────────────────────
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<THINK>", "<HALT>"]
})
HALT_ID  = tokenizer.convert_tokens_to_ids("<HALT>")
THINK_ID = tokenizer.convert_tokens_to_ids("<THINK>")
print(f"  <HALT> token id: {HALT_ID}", flush=True)
print(f"  <THINK> token id: {THINK_ID}", flush=True)


def _parse_chain(text: str) -> list[str] | None:
    """Extract per-loop targets from 'X = val. Y = X.' style chain."""
    assignments = re.findall(r'([A-Za-z_]\w*)\s*=\s*(\S+?)[\.\n]', text)
    if len(assignments) < 2:
        return None
    val: dict[str, str] = {}
    for var, expr in assignments:
        val[var] = expr
    chain_vars = [assignments[0][0]]
    for var, _ in assignments[1:]:
        chain_vars.append(var)
    targets: list[str] = []
    for i in range(len(chain_vars) - 1):
        targets.append(chain_vars[i])
    final_var = chain_vars[-1]
    resolved  = val.get(final_var, final_var)
    visited: set[str] = set()
    while resolved in val and resolved not in visited:
        visited.add(resolved)
        resolved = val[resolved]
    targets.append(resolved)
    # v29: add <HALT> as the final target
    targets.append("<HALT>")
    return targets if len(targets) >= 3 else None


def find_answer_start(ids: list[int]) -> int:
    """Return first token position after 'Answer:' boundary."""
    for boundary in (
        tokenizer.encode("Answer:",  add_special_tokens=False),
        tokenizer.encode(" Answer:", add_special_tokens=False),
    ):
        n = len(boundary)
        for i in range(len(ids) - n):
            if ids[i:i + n] == boundary:
                return i + n
    return -1


# ── LoRA Linear ───────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """LoRA adapter — identical to v28."""

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        """Initialize with same dtype as base weight, zero lora_B."""
        super().__init__()
        self.bias  = linear.bias
        d_out, d_in = linear.weight.shape
        dtype = linear.weight.dtype
        self.register_buffer("base_weight", linear.weight.data.clone())
        self.lora_A = nn.Parameter(torch.empty(rank, d_in,  dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, dtype=dtype))
        self.scale  = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A)

    @property
    def weight(self) -> torch.Tensor:
        """Fused LoRA weight."""
        return self.base_weight + self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard linear forward."""
        return F.linear(x, self.weight, self.bias)


# ── Recursive Mamba-130m v29 ──────────────────────────────────────────────────
class RecursiveMamba130m_v29(nn.Module):
    """
    Mamba-130m with Latent Forcing + <HALT> self-halting (v29).

    Training targets: [var1, var2, ..., answer, <HALT>]
    Inference halt: when model predicts <HALT>, stop and return previous token.

    The <HALT> token is ALWAYS reachable in the pointer mask — it is added
    as a special allowed token regardless of prompt content.
    """

    MAX_LOOPS: int = MAX_LOOPS

    def __init__(self, backbone: MambaLMHeadModel, lora_rank: int = 8):
        """Freeze bottom 6 layers, LoRA on top 18, add loop machinery."""
        super().__init__()
        self.backbone    = backbone.backbone
        self.lm_head     = backbone.lm_head
        self.all_layers  = nn.ModuleList(backbone.backbone.layers)
        self.norm        = backbone.backbone.norm_f
        d_model          = backbone.backbone.embedding.embedding_dim  # 768

        for layer in self.all_layers[:BASE_SPLIT]:
            for p in layer.parameters():
                p.requires_grad = False

        for layer in self.all_layers[BASE_SPLIT:]:
            mx = layer.mixer
            for attr in ("in_proj", "out_proj"):
                if hasattr(mx, attr):
                    setattr(mx, attr, LoRALinear(getattr(mx, attr),
                                                 rank=lora_rank,
                                                 alpha=lora_rank * 2.0))

        self.step_emb    = nn.Embedding(self.MAX_LOOPS, d_model).to(torch.bfloat16)
        self.loop_norm   = nn.RMSNorm(d_model).to(torch.bfloat16)
        self.mamba3_core = Mamba(
            d_model=d_model, d_state=16, d_conv=4, expand=2
        ).to(torch.bfloat16)
        nn.init.zeros_(self.mamba3_core.out_proj.weight)

        n_lora = sum(p.numel() for n, p in self.named_parameters()
                     if p.requires_grad and "lora" in n.lower())
        n_new  = (sum(p.numel() for p in self.step_emb.parameters()) +
                  sum(p.numel() for p in self.loop_norm.parameters()) +
                  sum(p.numel() for p in self.mamba3_core.parameters()))
        total  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  LoRA params:     {n_lora:,}")
        print(f"  Loop machinery:  {n_new:,}")
        print(f"  Total trainable: {total:,}\n")

    def _build_mask(self, input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """Pointer mask: allow prompt tokens + always allow <HALT>.

        v29 key difference: <HALT> token is always reachable so model can
        learn to predict it without it being blocked by the pointer mask.
        """
        mask = torch.full((vocab_size,), float("-inf"), device=input_ids.device)
        mask[torch.unique(input_ids)] = 0.0
        # Always allow <HALT> — it's the learned stop signal
        if HALT_ID < vocab_size:
            mask[HALT_ID] = 0.0
        return mask

    def forward(
        self,
        input_ids:     torch.Tensor,
        chain_targets: list | None = None,
        ans_starts:    list | None = None,
    ) -> tuple:
        """
        Training: Latent Forcing loss with <HALT> as final target.
        Inference: run until model predicts <HALT>, return previous answer.
        """
        x        = self.backbone.embedding(input_ids)
        residual = None

        for layer in self.all_layers:
            x, residual = layer(x, residual)

        # ── Training path ──────────────────────────────────────────────────────
        if self.training and chain_targets is not None:
            B, max_len = input_ids.shape
            n_loops    = max(len(t) for t in chain_targets)  # includes HALT loop

            def run_lora_layers(x_in, res_in):
                """Re-run LoRA top layers per loop."""
                for layer in self.all_layers[BASE_SPLIT:]:
                    x_in, res_in = layer(x_in, res_in)
                return x_in, res_in

            step_losses: list[torch.Tensor] = []
            step_accs:   list[torch.Tensor] = []; halt_accs: list[float] = []

            for loop_i in range(n_loops):
                step_vec = self.step_emb(
                    torch.tensor(min(loop_i, self.MAX_LOOPS - 1), device=x.device)
                )
                x = x + step_vec
                x, residual = grad_ckpt(run_lora_layers, x, residual,
                                        use_reentrant=False)
                x = x + self.mamba3_core(x)
                x = self.loop_norm(x)

                x_normed    = self.norm(x, residual, prenorm=False)
                logits_step = self.lm_head(x_normed)
                vocab_size  = logits_step.shape[-1]

                loop_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
                loop_acc  = torch.tensor(0.0, device=x.device)
                valid     = 0

                for b in range(B):
                    as_ = ans_starts[b]
                    if as_ < 1 or as_ >= max_len:
                        continue
                    btgt   = chain_targets[b]
                    tgt_id = int(btgt[min(loop_i, len(btgt) - 1)])

                    # Build mask: prompt tokens + HALT always allowed
                    mask = self._build_mask(input_ids[b], vocab_size)

                    # HALT token is always reachable — no skip needed for it
                    if tgt_id >= vocab_size:
                        continue
                    if tgt_id != HALT_ID and mask[tgt_id].item() == float("-inf"):
                        continue

                    logits_b = logits_step[b, as_ - 1, :] + mask
                    pred_tok = logits_b.argmax().item()
                    tgt_t    = torch.tensor(tgt_id, device=x.device)
                    loop_loss = loop_loss + F.cross_entropy(
                        logits_b.unsqueeze(0), tgt_t.unsqueeze(0)
                    )
                    loop_acc = loop_acc + float(pred_tok == tgt_id)
                    valid   += 1
                    if tgt_id == HALT_ID:
                        halt_accs.append(float(pred_tok == tgt_id))

                if valid > 0:
                    step_losses.append(loop_loss / valid)
                    step_accs.append(loop_acc   / valid)

            avg_loss  = (torch.stack(step_losses).mean()
                         if step_losses else
                         torch.tensor(0.0, device=x.device, requires_grad=True))
            avg_acc   = (torch.stack([a.clone().detach() for a in step_accs]).mean()
                         if step_accs else torch.tensor(0.0, device=x.device))
            # final_acc = accuracy on the answer loop (excluding HALT loop)
            answer_accs = step_accs[:-1] if len(step_accs) > 1 else step_accs
            final_acc   = (torch.stack([a.clone().detach() for a in answer_accs]).mean()
                           if answer_accs else avg_acc)
            halt_acc_val = (sum(halt_accs) / len(halt_accs)) if halt_accs else 0.0
            return avg_loss, avg_acc, final_acc, halt_acc_val

        # ── Inference path — halt on <HALT> prediction ──────────────────────────
        else:
            vocab_size = self.lm_head.weight.shape[0]
            mask = self._build_mask(input_ids[0], vocab_size)

            trace:      list[tuple] = []
            last_answer: str = ""
            last_prob:   float = 0.0

            for loop_i in range(self.MAX_LOOPS):
                sv = self.step_emb(
                    torch.tensor(min(loop_i, self.MAX_LOOPS - 1), device=x.device)
                )
                x  = x + sv
                for layer in self.all_layers[BASE_SPLIT:]:
                    x, residual = layer(x, residual)
                x = x + self.mamba3_core(x)
                x = self.loop_norm(x)

                lg  = self.lm_head(self.norm(x, residual, prenorm=False))
                lg[0, -1, :] += mask
                p   = torch.softmax(lg[0, -1, :], dim=-1)
                tid = p.argmax().item()
                tok = tokenizer.decode([tid]).strip()
                trace.append((f"L{loop_i+1}", tok, round(p.max().item(), 3)))

                if tid == HALT_ID:
                    # Model decided to stop — return the previous answer token
                    trace[-1] = (f"L{loop_i+1}", "<HALT>", round(p.max().item(), 3))
                    return loop_i + 1, trace, last_answer

                last_answer = tok
                last_prob   = round(p.max().item(), 3)

            return self.MAX_LOOPS, trace, last_answer


# ── Data Pipeline ─────────────────────────────────────────────────────────────
def load_training_data() -> list[dict]:
    """Load data with HALT appended to chain_targets."""
    samples: list[dict] = []

    def _extract_answer(text: str) -> str | None:
        m = re.search(r'[Aa]nswer:\s*(\S+)', text)
        return m.group(1).rstrip('.,!?') if m else None

    def _prep(text: str, chain_targets: list[str]) -> dict | None:
        """Pre-tokenize targets; HALT target always maps to HALT_ID."""
        enc_ids   = tokenizer.encode(text, add_special_tokens=False)
        enc_set   = set(enc_ids)
        ans_start = find_answer_start(enc_ids)
        if ans_start < 1:
            return None
        tgt_ids: list[int] = []
        for tgt_str in chain_targets:
            if tgt_str == "<HALT>":
                tgt_ids.append(HALT_ID)
                continue
            matched = None
            for pfx in (" ", ""):
                cands = tokenizer.encode(pfx + tgt_str, add_special_tokens=False)
                if cands and cands[0] in enc_set:
                    matched = cands[0]
                    break
            if matched is None:
                cands   = tokenizer.encode(" " + tgt_str, add_special_tokens=False)
                matched = cands[0] if cands else tokenizer.eos_token_id
            tgt_ids.append(matched)
        return {
            "text":          text,
            "hops":          len(chain_targets) - 1,  # subtract HALT loop
            "chain_targets": chain_targets,
            "chain_tgt_ids": tgt_ids,
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
            chain_tgts = _parse_chain(text)  # already appends <HALT>
            if chain_tgts and len(chain_tgts) >= 3:
                s = _prep(text, chain_tgts)
                if s:
                    samples.append(s)
                    multi_hop += 1
                else:
                    skipped += 1
            else:
                ans = _extract_answer(text)
                if ans:
                    s = _prep(text, [ans, "<HALT>"])  # 1-hop + HALT
                    if s:
                        samples.append(s)
                        one_hop += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1
        print(f"  Multi-hop chains: {multi_hop:,}")
        print(f"  Single-hop items: {one_hop:,}")
        print(f"  Skipped:          {skipped:,}")

    if os.path.exists("mmlu_format_v17.json"):
        with open("mmlu_format_v17.json") as f:
            mmlu = json.load(f)
        mmlu_added = 0
        for item in mmlu[:5_000]:
            text = item.get("text", "")
            ans  = _extract_answer(text)
            if ans:
                s = _prep(text, [ans, "<HALT>"])
                if s:
                    samples.append(s)
                    mmlu_added += 1
        print(f"  MMLU-format:      {mmlu_added:,}")

    # Counterfactual samples — HELD OUT from chain training (different content)
    # These are reality-override only; not used in Phase 1/2/3 chain testing
    for cf in [
        ("The sun is freezing cold. John's coffee is the temperature of the sun. What temperature is John's coffee?\nAnswer: cold",  ["cold", "<HALT>"]),
        ("Fire is icy cold. Bob touched fire. What did Bob feel?\nAnswer: cold",                                                     ["cold", "<HALT>"]),
        ("In this world dogs meow and cats bark. Sarah has a cat. What sound does she hear?\nAnswer: bark",                          ["bark", "<HALT>"]),
    ]:
        s = _prep(cf[0], cf[1])
        if s:
            samples.append(s)

    hop_dist: dict = {}
    for s in samples:
        hop_dist[s["hops"]] = hop_dist.get(s["hops"], 0) + 1
    print(f"  Hop dist: {dict(sorted(hop_dist.items()))}")
    print(f"  Total:    {len(samples):,}  (all pre-tokenized, targets include <HALT>)\n")
    return samples


def make_batch(pool: list[dict], seed: int) -> tuple:
    """Sample a padded batch with pre-tokenized targets (including HALT)."""
    batch = random.Random(seed).sample(pool, min(BATCH_SIZE, len(pool)))
    enc   = tokenizer(
        [s["text"] for s in batch],
        max_length=SEQ_LEN, truncation=True,
        padding="max_length", return_tensors="pt"
    )
    return (
        enc["input_ids"].to(DEVICE),
        [s["chain_tgt_ids"] for s in batch],
        [s["ans_start"]     for s in batch],
    )


# ── Load Model ────────────────────────────────────────────────────────────────
print(f"  Loading {MODEL_ID} (bfloat16)...", flush=True)
base_model = MambaLMHeadModel.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device=DEVICE
)

new_vocab = len(tokenizer)  # Now includes <HALT> AND <THINK>
old_vocab = base_model.backbone.embedding.weight.shape[0]
d_model   = base_model.backbone.embedding.embedding_dim

if new_vocab > old_vocab:
    print(f"  Resizing vocab {old_vocab} → {new_vocab} (added <HALT>)")
    new_emb = nn.Embedding(new_vocab, d_model, dtype=torch.bfloat16)
    nn.init.normal_(new_emb.weight, std=0.02)
    new_emb.weight.data[:old_vocab] = base_model.backbone.embedding.weight.data
    base_model.backbone.embedding   = new_emb

    new_head = nn.Linear(d_model, new_vocab, bias=False, dtype=torch.bfloat16)
    nn.init.normal_(new_head.weight, std=0.02)
    new_head.weight.data[:old_vocab] = base_model.lm_head.weight.data
    base_model.lm_head = new_head

for p in base_model.parameters():
    p.requires_grad = False
base_model.backbone.embedding.weight.requires_grad = True
base_model.lm_head.weight.requires_grad             = True

model = RecursiveMamba130m_v29(base_model, lora_rank=LORA_RANK).to(DEVICE)

# Warm-start from v28 — compatible partial load (step_emb size differs: 6→8)
if RESUME_FROM and os.path.exists(RESUME_FROM):
    ckpt   = torch.load(RESUME_FROM, map_location=DEVICE)
    sd     = ckpt.get("model_state_dict", ckpt)
    own_sd = model.state_dict()
    # Copy only tensors whose shapes match exactly; skip mismatched ones
    filtered = {}
    skipped  = []
    for k, v in sd.items():
        if k in own_sd and own_sd[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)
    miss, unex = model.load_state_dict(filtered, strict=False)
    # Warm-start step_emb: copy v28 rows [0:6] into v29's [0:6], rows 6-7 stay random
    if "step_emb.weight" in sd:
        with torch.no_grad():
            src = sd["step_emb.weight"]  # [6, 768]
            model.step_emb.weight.data[:src.shape[0]] = src.to(
                model.step_emb.weight.device
            )
        print(f"  step_emb warm-start: copied {src.shape[0]} rows, 2 new rows random")
    print(f"  Warm-start from {RESUME_FROM}")
    print(f"  Loaded: {len(filtered)} | Skipped (shape mismatch): {skipped} | Missing: {len(miss)}")
else:
    print(f"  No warm-start checkpoint found — training from scratch")


# ── Optimizer + Data ─────────────────────────────────────────────────────────
samples = load_training_data()

random.seed(42)
train_samples: list[dict] = []
val_samples:   list[dict] = []
hop_buckets: dict[int, list] = {}
for s in samples:
    hop_buckets.setdefault(s["hops"], []).append(s)
for hop, bucket in sorted(hop_buckets.items()):
    random.shuffle(bucket)
    n_val = max(1, int(len(bucket) * 0.10))
    val_samples.extend(bucket[:n_val])
    train_samples.extend(bucket[n_val:])

hop_val = {}
for s in val_samples:
    hop_val[s["hops"]] = hop_val.get(s["hops"], 0) + 1
print(f"  Train: {len(train_samples):,} | Val: {len(val_samples):,}")
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
    {"params": g1_params, "lr": 5e-4,  "weight_decay": 0.0},   # half LR vs v28 (warm-start)
    {"params": g2_params, "lr": 2e-4,  "weight_decay": 0.01},
])
scheduler = CosineAnnealingLR(optimizer, T_max=STEPS // ACCUM, eta_min=1e-6)

print(f"  g1={len(g1_params)} tensors @ 5e-4 | g2={len(g2_params)} LoRA @ 2e-4")
print(f"\n{'─'*60}")
print(f"  130m v29 Latent Forcing + <HALT> | {STEPS} steps | eff batch={BATCH_SIZE*ACCUM}")
print(f"  Halt: model predicts <HALT> → return previous token as answer")
print(f"{'─'*60}\n")


def run_validation(val_pool: list[dict], n_batches: int = 30) -> tuple:
    """Val pass — training mode + torch.no_grad."""
    v_loss = v_aa = v_fa = v_halt = 0.0; valid = 0
    with torch.no_grad():
        for i in range(n_batches):
            try:
                ids, ctgts, ans = make_batch(val_pool, seed=99999 + i)
                loss, aa, fa, ha = model(ids, chain_targets=ctgts, ans_starts=ans)
                if torch.isfinite(loss):
                    v_loss += loss.item(); v_aa += aa.item() * 100
                    v_fa   += fa.item() * 100; v_halt += ha * 100; valid += 1
            except Exception:
                pass
    if valid == 0:
        return float("inf"), 0.0, 0.0, 0.0
    return v_loss / valid, v_aa / valid, v_fa / valid, v_halt / valid


# ── Training Loop ─────────────────────────────────────────────────────────────
model.train()
optimizer.zero_grad()
t0 = time.time()
total_loss = total_aa = total_fa = total_ha = 0.0
early_stop_hits = 0; best_val_acc = 0.0; last_train_aa = 0.0

for step in range(1, STEPS + 1):
    for accum_i in range(ACCUM):
        ids, ctgts, ans = make_batch(train_samples, seed=step * ACCUM + accum_i)
        loss, aa, fa, ha = model(ids, chain_targets=ctgts, ans_starts=ans)
        (loss / ACCUM).backward()
        total_loss += loss.item(); total_aa += aa.item() * 100
        total_fa   += fa.item() * 100; total_ha += ha * 100

    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0
    )
    optimizer.step(); scheduler.step(); optimizer.zero_grad()

    if step % LOG_EVERY == 0:
        n = LOG_EVERY * ACCUM
        al = total_loss / n; aa = total_aa / n; fa = total_fa / n; ha = total_ha / n
        last_train_aa = aa
        total_loss = total_aa = total_fa = total_ha = 0.0
        vram = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0
        tps  = (BATCH_SIZE * ACCUM * SEQ_LEN * LOG_EVERY) / (time.time() - t0)
        t0   = time.time()
        print(
            f"  Step {step:5d} | Loss: {al:.4f} | AllLoopAcc: {aa:5.1f}%"
            f" | AnswerAcc: {fa:5.1f}% | HaltAcc: {ha:5.1f}%"
            f" | LR: {optimizer.param_groups[0]['lr']:.1e}"
            f" | TPS: {int(tps)} | VRAM: {vram:.2f}GB",
            flush=True,
        )

    if step % VAL_EVERY == 0:
        vl, vaa, vfa, vha = run_validation(val_samples)
        gap  = last_train_aa - vaa
        flag = " ⚠️  OVERFIT" if gap > 5.0 else ""
        print(f"\n  ── VAL @ step {step} ───────────────────────────────────", flush=True)
        print(f"  Val AllLoopAcc: {vaa:5.1f}% | AnswerAcc: {vfa:5.1f}% | HaltAcc: {vha:5.1f}%", flush=True)
        print(f"  Train-Val gap: {gap:+.1f}pp{flag}", flush=True)

        if vaa > best_val_acc:
            best_val_acc = vaa
            best_path    = SAVE_PATH.replace(".pt", "_best.pt")
            torch.save({"model_state_dict": model.state_dict(),
                        "step": step, "val_allloop_acc": vaa,
                        "halt_id": HALT_ID}, best_path)
            print(f"  🏆 Best val {vaa:.1f}% → {best_path}", flush=True)

        if vaa >= EARLY_STOP_ACC:
            early_stop_hits += 1
            print(f"  ⏱  Early-stop: {early_stop_hits}/{EARLY_STOP_COUNT}  (val={vaa:.1f}%)", flush=True)
            if early_stop_hits >= EARLY_STOP_COUNT:
                print(f"\n  ✅ EARLY STOP @ step {step}. Val={vaa:.1f}%", flush=True)
                torch.save({"model_state_dict": model.state_dict(),
                            "step": step, "halt_id": HALT_ID}, SAVE_PATH)
                break
        else:
            early_stop_hits = 0
        print(f"  {'─'*52}\n", flush=True)

    if step % 500 == 0:
        ckpt_path = SAVE_PATH.replace(".pt", f"_step{step}.pt")
        torch.save({
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step, "halt_id": HALT_ID,
        }, ckpt_path)
        print(f"  💾 {ckpt_path}", flush=True)

else:
    torch.save({"model_state_dict": model.state_dict(),
                "step": STEPS, "halt_id": HALT_ID}, SAVE_PATH)
    print(f"\n✅ 130m v29 complete — {SAVE_PATH}\n")


# ── Quick inference test — model-driven halting ───────────────────────────────
model.eval()
tests = [
    ("A = red. B = A. What is B?\nAnswer:",                     "red"),
    ("X = Apple. Y = X. Z = Y. What is Z?\nAnswer:",            "Apple"),
    ("A = moon. B = A. C = B. D = C. What is D?\nAnswer:",      "moon"),
    ("Fire is icy cold. Bob touched fire. What did Bob feel?\nAnswer:", "cold"),
]
print("=" * 60)
with torch.no_grad():
    for prompt, expected in tests:
        ids_ = tokenizer.encode(prompt, add_special_tokens=False,
                                return_tensors="pt").to(DEVICE)
        loops, trace, answer = model(ids_)
        chain_str = " → ".join(f"{t[0]}:{t[1]}" for t in trace)
        match = "✅" if expected.lower() in answer.lower() else "❌"
        print(f"  {match} Q: {prompt.splitlines()[0][:55]!r}")
        print(f"     {chain_str}")
        print(f"     Answer: {answer!r}  ({loops} loops, halted by <HALT> token)\n")
print("=" * 60)
