"""
phase1_warmup.py — Prefix Scratchpad Initialization for Mamba2-2.7B + RLF
===========================================================================
Phase 1: Format the blank latent_memory and calibrate the latent_bridge.

What gets trained:
  - latent_memory (16 × 2560 = 40,960 params)  — scratch paper formatting
  - latent_bridge (2560 × 2560 + 2560 = 6,556,160 params) — S2→S1 translator

What stays frozen:
  - Mamba-2 2.7B base backbone (2.7B params)
  - LoRA adapters (step 3000 reasoning logic)
  - Loop engine, lifeline gate, embeddings, LM head

Expected: Fast convergence (~500-1000 steps). The bridge starts as identity,
so initial loss should be similar to the base model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import json, random, time, os, re, sys
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint as grad_ckpt

from mamba_engine import (
    RecursiveMamba2_PrefixScratchpad, freeze_for_phase1, get_phase1_optimizer,
    fuse_lora_weights, LoRALinear, tokenizer, HALT_ID,
    DEVICE, MODEL_ID, BASE_SPLIT, LORA_RANK, SEQ_LEN, BATCH_SIZE, ACCUM,
    PREFIX_M, D_MODEL, BRIDGE_RANK,
)

# ── Phase 1 Config ────────────────────────────────────────────────────────────
PHASE1_STEPS     = 1_000     # Memory formatting converges fast
LOG_EVERY        = 50
VAL_EVERY        = 200
SAVE_PATH        = "mamba2_2.7b_phase1_scratchpad.pt"
CKPT_PATH        = "saved_weights/mamba2_2.7b_rlf_rope_best_step3000_val97.3.pt"

# ── Live log ──────────────────────────────────────────────────────────────────
class Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, log_path: str):
        """Init with log file path."""
        self.terminal = sys.stdout
        self.log = open(log_path, 'w')
    def write(self, message: str) -> None:
        """Write to both outputs."""
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self) -> None:
        """Flush both outputs."""
        self.terminal.flush()
        self.log.flush()

sys.stdout = Tee('training_phase1.log')

print(f"\n{'='*70}")
print(f"  PHASE 1 WARMUP — Prefix Latent Scratchpad Initialization")
print(f"  Target: Format latent_memory + Calibrate latent_bridge")
print(f"  Frozen: Base 2.7B + LoRA + Loop engine + Gate + Embeddings")
print(f"  Base ckpt: {CKPT_PATH}")
print(f"  Device: {DEVICE} | Steps: {PHASE1_STEPS}")
print(f"{'='*70}\n")


# ── Data helpers (from mamba_engine) ──────────────────────────────────────────
def find_answer_start(ids: list[int]) -> int:
    """Find first position after 'Answer:' boundary."""
    for boundary in (
        tokenizer.encode("Answer:",  add_special_tokens=False),
        tokenizer.encode(" Answer:", add_special_tokens=False),
        tokenizer.encode("\nAnswer:", add_special_tokens=False),
    ):
        n = len(boundary)
        for i in range(len(ids) - n + 1):
            if ids[i:i + n] == boundary:
                return min(i + n, len(ids) - 1)
    return -1


def _parse_v2_chain(answer: str) -> list[str] | None:
    """Extract per-loop targets from v2_clean answer format."""
    m = re.match(
        r'\[Reasoning\]\s*(.+?)\s*\[Answer\]\s*(.+?)\s*<HALT>',
        answer.strip()
    )
    if not m:
        return None

    reasoning = m.group(1).strip()
    final_ans = m.group(2).strip()
    targets: list[str] = []
    steps = [s.strip() for s in reasoning.split(';') if s.strip()]

    for step in steps:
        arith_m = re.search(r'=\s*(\S+)\s*$', step)
        if arith_m:
            targets.append(arith_m.group(1)); continue
        assign_m = re.search(r'←\s*(\S+?)(?:\s*=\s*(\S+))?\s*$', step)
        if assign_m:
            targets.append(assign_m.group(2) or assign_m.group(1)); continue
        cont_m = re.search(r'→\s*(\S+)\s*$', step)
        if cont_m:
            targets.append(cont_m.group(1)); continue
        if step.lower().startswith('given'):
            continue
        step_m = re.search(r'Step \d+:\s*(.+)', step)
        if step_m:
            words = step_m.group(1).split()
            if words:
                targets.append(words[-1])
            continue

    if not targets or targets[-1] != final_ans:
        targets.append(final_ans)
    targets.append("<HALT>")
    return targets if len(targets) >= 2 else None


def _parse_chain(text: str) -> list[str] | None:
    """Extract per-loop targets + HALT from chain text."""
    assignments = re.findall(r'([A-Za-z_]\w*)\s*=\s*(\S+?)[.\n]', text)
    if len(assignments) < 2:
        return None
    val: dict[str, str] = {}
    for var, expr in assignments:
        val[var] = expr
    chain_vars = [assignments[0][0]]
    for var, _ in assignments[1:]:
        chain_vars.append(var)
    targets: list[str] = [chain_vars[i] for i in range(len(chain_vars) - 1)]
    final_var = chain_vars[-1]
    resolved = val.get(final_var, final_var)
    visited: set[str] = set()
    while resolved in val and resolved not in visited:
        visited.add(resolved)
        resolved = val[resolved]
    targets.append(resolved)
    targets.append("<HALT>")
    return targets if len(targets) >= 3 else None


def _parse_override(sample: dict) -> list[str]:
    """Override: single direct answer then HALT."""
    return [sample["answer"], "<HALT>"]


def load_training_data() -> list[dict]:
    """Load training data with chain parsing."""
    samples: list[dict] = []

    def _prep(text: str, tgt_strs: list[str]) -> dict | None:
        """Prepare a sample for training."""
        enc_ids = tokenizer.encode(text, add_special_tokens=False)
        ans_start = find_answer_start(enc_ids)
        if ans_start < 1:
            return None
        ans_start = min(ans_start, len(enc_ids) - 1)
        tgt_ids: list[int] = []
        for ts in tgt_strs:
            if ts == "<HALT>":
                tgt_ids.append(HALT_ID); continue
            for pfx in (" ", ""):
                cands = tokenizer.encode(pfx + ts, add_special_tokens=False)
                if cands: tgt_ids.append(cands[0]); break
            else:
                tgt_ids.append(tokenizer.eos_token_id)
        return {
            "text": text, "hops": len(tgt_strs) - 1,
            "chain_targets": tgt_strs, "chain_tgt_ids": tgt_ids,
            "ans_start": ans_start,
        }

    for fname, label in [("system2_logic_v3_curriculum.json", "v3_curriculum"),
                          ("system2_logic_v2_clean.json", "v2_clean")]:
        if not os.path.exists(fname):
            continue
        with open(fname) as f:
            data = json.load(f)
        ok = 0
        for item in data:
            if "[Reasoning]" in item.get("answer", ""):
                ct = _parse_v2_chain(item["answer"])
                full_text = item["text"] + item["answer"]
                s = _prep(full_text, ct) if ct and len(ct) >= 2 else None
            elif item.get("type") == "override":
                ct = _parse_override(item)
                s = _prep(item["text"], ct) if ct and len(ct) >= 2 else None
            else:
                ct = _parse_chain(item["text"])
                s = _prep(item["text"], ct) if ct and len(ct) >= 2 else None
            if s:
                samples.append(s)
                ok += 1
        print(f"  {label}: {ok:,} samples loaded")
        if ok > 0:
            break

    hop_dist = {}
    for s in samples: hop_dist[s["hops"]] = hop_dist.get(s["hops"], 0) + 1
    print(f"  Total: {len(samples):,} | Hop dist: {dict(sorted(hop_dist.items()))}\n")
    return samples


def make_batch(pool: list[dict], seed: int) -> tuple:
    """Sample a padded batch."""
    batch = random.Random(seed).sample(pool, min(BATCH_SIZE, len(pool)))
    enc = tokenizer(
        [s["text"] for s in batch],
        max_length=SEQ_LEN, truncation=True, padding="max_length", return_tensors="pt"
    )
    return (
        enc["input_ids"].to(DEVICE),
        [s["chain_tgt_ids"] for s in batch],
        [s["ans_start"] for s in batch],
    )


# ── Load Model with Checkpoint ───────────────────────────────────────────────
print(f"  Loading {MODEL_ID} (bfloat16)...", flush=True)
base_model = MambaLMHeadModel.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device=DEVICE
)

# Expand vocab for <THINK> and <HALT> tokens
new_vocab = len(tokenizer)
old_vocab = base_model.backbone.embedding.weight.shape[0]
d_model = base_model.backbone.embedding.embedding_dim

if new_vocab > old_vocab:
    ne = nn.Embedding(new_vocab, d_model, dtype=torch.bfloat16)
    nn.init.normal_(ne.weight, std=0.02)
    ne.weight.data[:old_vocab] = base_model.backbone.embedding.weight.data
    base_model.backbone.embedding = ne
    nh = nn.Linear(d_model, new_vocab, bias=False, dtype=torch.bfloat16)
    nn.init.normal_(nh.weight, std=0.02)
    nh.weight.data[:old_vocab] = base_model.lm_head.weight.data
    base_model.lm_head = nh

for p in base_model.parameters():
    p.requires_grad = False

# Create the Prefix Scratchpad model
model = RecursiveMamba2_PrefixScratchpad(base_model, lora_rank=LORA_RANK).to(DEVICE)

# Load step 3000 RLF checkpoint
print(f"  Loading RLF checkpoint: {CKPT_PATH}", flush=True)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
own_sd = model.state_dict()
loaded = 0
skipped = 0
for k, v in sd.items():
    if k in own_sd and own_sd[k].shape == v.shape:
        own_sd[k] = v
        loaded += 1
    else:
        skipped += 1
model.load_state_dict(own_sd, strict=False)
print(f"  Loaded {loaded} tensors, skipped {skipped} (new: latent_memory, bridge_down, bridge_up)")

# Fuse LoRA weights into base weights and convert to nn.Linear
# This eliminates LoRA temporary allocations and saves ~0.5GB VRAM
print(f"  Fusing LoRA weights for Phase 1 memory savings...")
fuse_lora_weights(model)

# ── PHASE 1 FREEZE ───────────────────────────────────────────────────────────
freeze_for_phase1(model)
optimizer = get_phase1_optimizer(model)
scheduler = CosineAnnealingLR(optimizer, T_max=PHASE1_STEPS, eta_min=1e-6)

# ── Data ──────────────────────────────────────────────────────────────────────
samples = load_training_data()
if not samples:
    raise RuntimeError("No training data found.")

random.seed(42)
hop_buckets: dict[int, list] = {}
for s in samples:
    hop_buckets.setdefault(s["hops"], []).append(s)
train_samples: list[dict] = []
val_samples: list[dict] = []
for hop, bucket in sorted(hop_buckets.items()):
    random.shuffle(bucket)
    n_val = max(1, int(len(bucket) * 0.10))
    val_samples.extend(bucket[:n_val])
    train_samples.extend(bucket[n_val:])

print(f"  Train: {len(train_samples):,} | Val: {len(val_samples):,}")

vram_init = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0
print(f"\n  VRAM after init: {vram_init:.2f}GB")
print(f"  Prefix memory: {PREFIX_M} tokens × {D_MODEL} = {PREFIX_M * D_MODEL:,} params")
print(f"  Latent bridge:  {D_MODEL}→{BRIDGE_RANK}→{D_MODEL} = "
      f"{D_MODEL*BRIDGE_RANK + BRIDGE_RANK*D_MODEL:,} params")
print(f"\n{'─'*70}")
print(f"  Starting Phase 1 warmup...")
print(f"{'─'*70}\n")


# ── Validation ────────────────────────────────────────────────────────────────
def run_validation(val_pool: list, n_batches: int = 30) -> tuple:
    """Run validation and return (loss, all_acc, ans_acc, halt_acc)."""
    vl = vaa = vfa = vha = 0.0
    valid = 0
    # NOTE: Do NOT call model.eval() — the forward pass dispatches on
    # self.training to return 4 values (train) vs 3 values (inference).
    # We want the 4-value training path. torch.no_grad() is sufficient.
    with torch.no_grad():
        for i in range(n_batches):
            try:
                ids, ct, ans = make_batch(val_pool, seed=99999 + i)
                loss, aa, fa, ha = model(ids, chain_targets=ct, ans_starts=ans)
                if torch.isfinite(loss):
                    vl += loss.item(); vaa += aa.item()*100
                    vfa += fa.item()*100; vha += ha*100; valid += 1
            except Exception as e:
                print(f"  val batch {i} failed: {e}")
    if valid == 0:
        return float("inf"), 0.0, 0.0, 0.0
    return vl/valid, vaa/valid, vfa/valid, vha/valid


# ── Training Loop ─────────────────────────────────────────────────────────────
model.train()
optimizer.zero_grad()
t0 = time.time()
total_loss = total_aa = total_fa = total_ha = 0.0
best_val_acc = 0.0

for step in range(1, PHASE1_STEPS + 1):
    for accum_i in range(ACCUM):
        ids, ct, ans = make_batch(train_samples, seed=step * ACCUM + accum_i)
        loss, aa, fa, ha = model(ids, chain_targets=ct, ans_starts=ans)
        (loss / ACCUM).backward()
        total_loss += loss.item()
        total_aa += aa.item() * 100
        total_fa += fa.item() * 100
        total_ha += ha * 100

    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0
    )
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    if step % LOG_EVERY == 0:
        n = LOG_EVERY * ACCUM
        al = total_loss / n
        aa = total_aa / n
        fa = total_fa / n
        ha = total_ha / n
        total_loss = total_aa = total_fa = total_ha = 0.0
        vram = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0
        tps = (BATCH_SIZE * ACCUM * SEQ_LEN * LOG_EVERY) / (time.time() - t0)
        t0 = time.time()
        mem_norm = model.latent_memory.data.norm().item()
        bridge_norm = (model.bridge_down.weight.data.norm().item()
                       + model.bridge_up.weight.data.norm().item())
        print(
            f"  Step {step:5d} | Loss: {al:.4f} | AllLoop: {aa:5.1f}%"
            f" | Halt: {ha:5.1f}%"
            f" | Mem‖={mem_norm:.4f} | Bridge‖={bridge_norm:.4f}"
            f" | TPS: {int(tps)} | VRAM: {vram:.2f}GB",
            flush=True
        )

    if step % VAL_EVERY == 0:
        vl, vaa, vfa, vha = run_validation(val_samples)
        print(f"\n  ── VAL @ step {step} {'─'*44}", flush=True)
        print(f"  Val AllLoop: {vaa:5.1f}% | Answer: {vfa:5.1f}% | Halt: {vha:5.1f}%",
              flush=True)

        if vaa > best_val_acc:
            best_val_acc = vaa
            best_path = SAVE_PATH.replace(".pt", "_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "step": step, "val_allloop_acc": vaa,
                "halt_id": HALT_ID, "no_mask": True,
                "rope": True, "d_model": d_model,
                "model_id": MODEL_ID,
                "prefix_m": PREFIX_M,
                "has_bridge": True,
                "phase": "phase1_warmup",
            }, best_path)
            print(f"  🏆 Best val {vaa:.1f}% → {best_path}", flush=True)

        print(f"  {'─'*60}\n", flush=True)

# ── Final save ────────────────────────────────────────────────────────────────
torch.save({
    "model_state_dict": model.state_dict(),
    "step": PHASE1_STEPS,
    "halt_id": HALT_ID, "no_mask": True,
    "rope": True, "d_model": d_model,
    "model_id": MODEL_ID,
    "prefix_m": PREFIX_M,
    "has_bridge": True,
    "phase": "phase1_warmup",
}, SAVE_PATH)
print(f"\n✅ Phase 1 warmup complete — {SAVE_PATH}")

# ── Quick smoke test ──────────────────────────────────────────────────────────
model.eval()
print(f"\n{'='*70}")
print(f"  Phase 1 Smoke Test — Reasoning should still work")
print(f"{'='*70}")

smoke_tests = [
    ("P = blue. Q = P. R = Q. What is R?\nAnswer:", "blue", "3-hop"),
    ("X = red. Y = X. Z = Y. W = Z. What is W?\nAnswer:", "red", "4-hop"),
    ("A = gold. B = A. C = B. D = C. E = D. F = E. What is F?\nAnswer:", "gold", "6-hop"),
]

for prompt, expected, label in smoke_tests:
    ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        n, trace, answer = model(ids)
    ok = expected.lower() in answer.lower()
    icon = "✅" if ok else "❌"
    print(f"  {icon} {label}: got='{answer}' want='{expected}' loops={n}")
    for lbl, tok, prob in trace:
        print(f"    {lbl:5s}  {tok!r:14s}  p={prob:.4f}")

print(f"\n{'='*70}")
print(f"  Memory ‖latent_memory‖ = {model.latent_memory.data.norm():.6f}")
print(f"  Bridge ‖down‖ = {model.bridge_down.weight.data.norm():.6f}")
print(f"  Bridge ‖up‖ = {model.bridge_up.weight.data.norm():.6f}")
print(f"{'='*70}\n")
