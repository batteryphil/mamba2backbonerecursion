"""
phase3_adversarial_training.py — Adversarial Generalization Training
=====================================================================
Phase 3: Force the reasoning core to generalize beyond clean A=B syntax.
Trains on 3 adversarial formats: Variable Chaos, Semantic Prose, Distractors.

SEQ_LEN bumped to 512 (from 128) because adversarial samples average 211 tokens.
Mamba is O(n) in sequence length — NOT O(n²) like Transformers — so 4× longer
sequences only add ~0.8GB activation memory, fitting comfortably in 12GB.

What gets trained (same as Phase 2):
  - LoRA adapters (layers 48-63)
  - Loop engine (mamba2_core)
  - Loop norm + Lifeline gate
  - Latent memory (prefix scratchpad)
  - Latent bridge (bridge_down + bridge_up)

What stays frozen:
  - Mamba-2 2.7B base backbone
  - Embeddings + LM head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json, random, time, os, re, sys, gc
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel, Mamba2
from torch.optim.lr_scheduler import CosineAnnealingLR

from mamba_engine import (
    RecursiveMamba2_PrefixScratchpad, LoRALinear, LoopRoPE,
    tokenizer, HALT_ID,
    DEVICE, MODEL_ID, BASE_SPLIT, LORA_RANK,
    PREFIX_M, D_MODEL, BRIDGE_RANK, MAX_LOOPS,
    LOOP_HEADDIM, LOOP_D_STATE, LOOP_EXPAND,
    BATCH_SIZE, ACCUM,
)

# ── Phase 3 Config — Override SEQ_LEN ─────────────────────────────────────
SEQ_LEN          = 512          # 4× increase from Phase 2's 128
PHASE3_STEPS     = 3_000
LOG_EVERY        = 50
VAL_EVERY        = 500
CKPT_EVERY       = 500
PHASE2_CKPT      = "mamba2_2.7b_phase2_joint_best.pt"
STEP3000_CKPT    = "saved_weights/mamba2_2.7b_rlf_rope_best_step3000_val97.3.pt"
SAVE_PATH        = "mamba2_2.7b_phase3_adversarial.pt"
EARLY_STOP_ACC   = 93.0
EARLY_STOP_COUNT = 3

# ── Live log ──────────────────────────────────────────────────────────────
class Tee:
    """Write to both stdout and log file."""
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

sys.stdout = Tee('training_phase3.log')

print(f"\n{'='*70}")
print(f"  PHASE 3 — Adversarial Generalization Training")
print(f"  Formats: Variable Chaos / Semantic Prose / Distractor Injection")
print(f"  SEQ_LEN: {SEQ_LEN} (up from 128)")
print(f"  Device: {DEVICE} | Steps: {PHASE3_STEPS}")
print(f"{'='*70}\n")


# ── Data Parsing ──────────────────────────────────────────────────────────
def _parse_v2_chain(answer: str) -> list[str] | None:
    """Extract per-loop targets from [Reasoning]...[Answer]...<HALT>."""
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
    """Extract per-loop targets from variable assignment chain text."""
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


def load_training_data() -> list[dict]:
    """Load Phase 3 adversarial data + existing curriculum for replay."""
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

    # ── Load Phase 3 adversarial curriculum ───────────────────────────────
    phase3_file = "phase3_adversarial_curriculum.json"
    if os.path.exists(phase3_file):
        with open(phase3_file) as f:
            adv_data = json.load(f)
        ok = 0
        type_counts: dict[str, int] = {}
        for item in adv_data:
            ct = _parse_v2_chain(item.get("answer", ""))
            if ct and len(ct) >= 2:
                s = _prep(item["text"], ct)
                if s:
                    s["format"] = item.get("type", "unknown")
                    samples.append(s)
                    ok += 1
                    t = item.get("type", "?")
                    type_counts[t] = type_counts.get(t, 0) + 1
        print(f"  Phase 3 adversarial: {ok:,} samples loaded")
        for t, c in sorted(type_counts.items()):
            print(f"    {t}: {c}")
    else:
        print(f"  ⚠️  {phase3_file} not found — run generate_phase3_data.py first")

    # ── Load existing curriculum for replay (prevent catastrophic forgetting) ─
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
                ct = [item["answer"], "<HALT>"]
                s = _prep(item["text"], ct) if ct and len(ct) >= 2 else None
            else:
                ct = _parse_chain(item["text"])
                s = _prep(item["text"], ct) if ct and len(ct) >= 2 else None
            if s:
                s["format"] = "original"
                samples.append(s)
                ok += 1
        print(f"  {label} (replay): {ok:,} samples loaded")
        if ok > 0:
            break

    hop_dist: dict[int, int] = {}
    for s in samples:
        hop_dist[s["hops"]] = hop_dist.get(s["hops"], 0) + 1
    format_dist: dict[str, int] = {}
    for s in samples:
        fmt = s.get("format", "?")
        format_dist[fmt] = format_dist.get(fmt, 0) + 1
    print(f"  Total: {len(samples):,}")
    print(f"  Format dist: {dict(sorted(format_dist.items()))}")
    print(f"  Hop dist: {dict(sorted(hop_dist.items()))}\n")
    return samples


def make_batch(pool: list[dict], seed: int) -> tuple:
    """Sample a padded batch with SEQ_LEN=512."""
    batch = random.Random(seed).sample(pool, min(BATCH_SIZE, len(pool)))
    enc = tokenizer(
        [s["text"] for s in batch],
        max_length=SEQ_LEN, truncation=True, padding="max_length",
        return_tensors="pt",
    )
    return (
        enc["input_ids"].to(DEVICE),
        [s["chain_tgt_ids"] for s in batch],
        [s["ans_start"] for s in batch],
    )


# ══════════════════════════════════════════════════════════════════════════
# Phase 3 Unfreeze (same as Phase 2)
# ══════════════════════════════════════════════════════════════════════════
def unfreeze_for_phase3(model: RecursiveMamba2_PrefixScratchpad) -> None:
    """Unfreeze reasoning engine for Phase 3 adversarial training.

    Same components as Phase 2: LoRA + loop engine + gate + memory + bridge.
    """
    for param in model.parameters():
        param.requires_grad = False

    for layer in model.all_layers[BASE_SPLIT:]:
        mx = layer.mixer
        for attr in ("in_proj", "out_proj"):
            module = getattr(mx, attr, None)
            if module is not None and isinstance(module, LoRALinear):
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True

    for param in model.mamba2_core.parameters():
        param.requires_grad = True
    for param in model.loop_norm.parameters():
        param.requires_grad = True
    model.lifeline_gate.requires_grad = True
    model.latent_memory.requires_grad = True
    for param in model.bridge_down.parameters():
        param.requires_grad = True
    for param in model.bridge_up.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\n{'='*70}")
    print(f"  PHASE 3 — Adversarial Unfreeze")
    print(f"{'='*70}")
    print(f"  Frozen:     {frozen:,}")
    print(f"  Trainable:  {trainable:,}")
    print(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════════════════════
# Model Loading — Load Phase 2 best checkpoint
# ══════════════════════════════════════════════════════════════════════════
print(f"  Loading {MODEL_ID} (bfloat16)...", flush=True)
base_model = MambaLMHeadModel.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device=DEVICE
)

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

model = RecursiveMamba2_PrefixScratchpad(base_model, lora_rank=LORA_RANK).to(DEVICE)

# Load Phase 2 checkpoint (contains LoRA + loop engine + memory + bridge)
ckpt_path = PHASE2_CKPT
if not os.path.exists(ckpt_path):
    ckpt_path = "mamba2_2.7b_phase2_joint.pt"
if not os.path.exists(ckpt_path):
    # Fall back to step 3000 LoRA + Phase 1 memory/bridge
    print(f"  ⚠️  No Phase 2 checkpoint found, loading step 3000...")
    ckpt_path = STEP3000_CKPT

print(f"  Loading checkpoint: {ckpt_path}...", flush=True)
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
own_sd = model.state_dict()
loaded = 0
for k, v in sd.items():
    if k in own_sd and own_sd[k].shape == v.shape:
        own_sd[k] = v
        loaded += 1
model.load_state_dict(own_sd, strict=False)
print(f"  Loaded {loaded} tensors")
del ckpt, sd, own_sd
gc.collect()
torch.cuda.empty_cache()

# Apply Phase 3 unfreeze
unfreeze_for_phase3(model)

# ── Data ──────────────────────────────────────────────────────────────────
samples = load_training_data()
if not samples:
    raise RuntimeError("No training data. Run generate_phase3_data.py first.")

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

# ── Optimizer ─────────────────────────────────────────────────────────────
g_memory = [model.latent_memory]
g_bridge = (list(model.bridge_down.parameters())
            + list(model.bridge_up.parameters()))
g_gate = [model.lifeline_gate]
g_loop = (list(model.mamba2_core.parameters())
           + list(model.loop_norm.parameters()))
g_lora = [p for n, p in model.named_parameters()
           if p.requires_grad and "lora" in n.lower()]

# Lower LR for Phase 3 — fine-tuning on top of fine-tuning
optimizer = optim.AdamW([
    {"params": g_memory, "lr": 3e-5, "weight_decay": 0.0},
    {"params": g_bridge, "lr": 3e-5, "weight_decay": 0.01},
    {"params": g_gate,   "lr": 3e-5, "weight_decay": 0.0},
    {"params": g_loop,   "lr": 5e-6, "weight_decay": 0.01},
    {"params": g_lora,   "lr": 5e-6, "weight_decay": 0.01},
])
scheduler = CosineAnnealingLR(optimizer, T_max=PHASE3_STEPS, eta_min=1e-7)

vram_init = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0
print(f"\n{'─'*70}")
print(f"  Phase 3 | SEQ_LEN={SEQ_LEN} | {PHASE3_STEPS} steps")
print(f"  BS={BATCH_SIZE} × ACCUM={ACCUM} = effective {BATCH_SIZE*ACCUM}")
print(f"  LR: LoRA/Loop=5e-6 | Memory/Bridge/Gate=3e-5")
print(f"  VRAM after init: {vram_init:.2f}GB")
print(f"{'─'*70}\n")


# ── Validation ────────────────────────────────────────────────────────────
def run_validation(val_pool: list, n_batches: int = 30) -> tuple:
    """Run validation (keep model.training=True for correct dispatch)."""
    vl = vaa = vfa = vha = 0.0
    valid = 0
    with torch.no_grad():
        for i in range(n_batches):
            try:
                ids, ct, ans = make_batch(val_pool, seed=99999 + i)
                loss, aa, fa, ha = model(ids, chain_targets=ct, ans_starts=ans)
                if torch.isfinite(loss):
                    vl += loss.item(); vaa += aa.item()*100
                    vfa += fa.item()*100; vha += ha*100; valid += 1
            except Exception as e:
                print(f"  val batch {i} error: {e}")
    if valid == 0:
        return float("inf"), 0.0, 0.0, 0.0
    return vl/valid, vaa/valid, vfa/valid, vha/valid


# ── Training Loop ─────────────────────────────────────────────────────────
model.train()
optimizer.zero_grad()
t0 = time.time()
total_loss = total_aa = total_fa = total_ha = 0.0
early_stop_hits = 0
best_val_acc = 0.0
last_train_aa = 0.0

for step in range(1, PHASE3_STEPS + 1):
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
        last_train_aa = aa
        total_loss = total_aa = total_fa = total_ha = 0.0
        vram = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0
        tps = (BATCH_SIZE * ACCUM * SEQ_LEN * LOG_EVERY) / (time.time() - t0)
        t0 = time.time()
        g = model.lifeline_gate.data
        mem_norm = model.latent_memory.data.norm().item()
        print(
            f"  Step {step:5d} | Loss: {al:.4f} | AllLoop: {aa:5.1f}%"
            f" | Halt: {ha:5.1f}%"
            f" | Gate μ={g.mean():.3f} σ={g.std():.4f}"
            f" | Mem‖={mem_norm:.2f}"
            f" | LR: {optimizer.param_groups[-1]['lr']:.1e}"
            f" | TPS: {int(tps)} | VRAM: {vram:.2f}GB",
            flush=True
        )

    if step % VAL_EVERY == 0:
        vl, vaa, vfa, vha = run_validation(val_samples)
        gap = last_train_aa - vaa
        flag = " ⚠️ OVERFIT" if gap > 10.0 else ""
        print(f"\n  ── VAL @ step {step} {'─'*44}", flush=True)
        print(f"  Val AllLoop: {vaa:5.1f}% | Answer: {vfa:5.1f}%"
              f" | Halt: {vha:5.1f}%", flush=True)
        print(f"  Train-Val gap: {gap:+.1f}pp{flag}", flush=True)

        if vaa > best_val_acc:
            best_val_acc = vaa
            best_path = SAVE_PATH.replace(".pt", "_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "step": step, "val_allloop_acc": vaa,
                "halt_id": HALT_ID, "rope": True,
                "d_model": d_model, "model_id": MODEL_ID,
                "prefix_m": PREFIX_M, "has_bridge": True,
                "phase": "phase3_adversarial",
                "seq_len": SEQ_LEN,
            }, best_path)
            print(f"  🏆 Best val {vaa:.1f}% → {best_path}", flush=True)

        if vaa >= EARLY_STOP_ACC:
            early_stop_hits += 1
            print(f"  ⏱  Early-stop: {early_stop_hits}/{EARLY_STOP_COUNT}",
                  flush=True)
            if early_stop_hits >= EARLY_STOP_COUNT:
                print(f"\n  ✅ EARLY STOP @ step {step}. Val={vaa:.1f}%\n",
                      flush=True)
                break
        else:
            early_stop_hits = 0
        print(f"  {'─'*60}\n", flush=True)

    if step % CKPT_EVERY == 0:
        torch.save({
            "model_state_dict": model.state_dict(),
            "step": step, "halt_id": HALT_ID, "rope": True,
            "d_model": d_model, "model_id": MODEL_ID,
            "prefix_m": PREFIX_M, "has_bridge": True,
            "phase": "phase3_adversarial",
            "seq_len": SEQ_LEN,
        }, SAVE_PATH.replace(".pt", f"_step{step}.pt"))
        print(f"  💾 step {step} checkpoint", flush=True)
else:
    torch.save({
        "model_state_dict": model.state_dict(),
        "step": PHASE3_STEPS, "halt_id": HALT_ID, "rope": True,
        "d_model": d_model, "model_id": MODEL_ID,
        "prefix_m": PREFIX_M, "has_bridge": True,
        "phase": "phase3_adversarial",
        "seq_len": SEQ_LEN,
    }, SAVE_PATH)
    print(f"\n✅ Phase 3 adversarial training complete — {SAVE_PATH}")


# ── Phase 3 OOD Evaluation ───────────────────────────────────────────────
model.eval()
print(f"\n{'='*70}")
print(f"  PHASE 3 ADVERSARIAL EVAL")
print(f"{'='*70}")

eval_tests = [
    # In-distribution format tests
    ("A=titan. B=A. C=B. D=C. What is D?\nAnswer:", "titan", "4-hop var (ID)"),
    # Variable chaos
    ("Var_42 is set to omega. Var_77 <- Var_42. Var_13 equals Var_77. "
     "What is the value of Var_13?\nAnswer:", "omega", "3-hop chaos (ID)"),
    # Semantic prose
    ("The secret vault password is 'cipher'. Alice memorizes the password. "
     "Alice whispers it to Bob. Bob texts it to Charlie. "
     "What is the password that Charlie received?\nAnswer:",
     "cipher", "3-hop prose (ID)"),
    # Distractor chain
    ("A = nebula. The Eiffel Tower grows in the summer. B = A. "
     "I need to buy milk. C = B. What is C?\nAnswer:",
     "nebula", "3-hop distractor (ID)"),
    # OOD: 10-hop variable chaos
    ("Var_10 = quantum. Var_20 <- Var_10. Var_30 is set to Var_20. "
     "Var_40 equals Var_30. Var_50 <- Var_40. Var_60 = Var_50. "
     "Var_70 is set to Var_60. Var_80 <- Var_70. Var_90 equals Var_80. "
     "Var_99 <- Var_90. What is the value of Var_99?\nAnswer:",
     "quantum", "10-hop chaos (OOD format)"),
    # OOD: Test 1 reprise — kinship
    ("The secret password is 'void'. John memorizes it. "
     "John whispers it to Michael. Michael passes the intel to Sarah. "
     "Sarah securely messages David. David briefs Emma. "
     "What is the password that Emma received?\nAnswer:",
     "void", "5-hop prose (kinship-style)"),
]

for prompt, expected, label in eval_tests:
    ids_ = tokenizer.encode(prompt, add_special_tokens=False,
                            return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        n, trace, answer = model(ids_)
    hit = ("✅" if expected.lower() in answer.lower()
           else f"❌ got={answer!r}")
    print(f"\n  {label}: {hit}")
    for lbl, tok, prob in trace:
        mark = (" ← ✅" if tok.lower() == expected.lower()
                else (" <HALT>" if tok == "<HALT>" else ""))
        print(f"    {lbl:5s}  {tok!r:14s}  p={prob:.4f}{mark}")

print(f"\n{'='*70}")
print(f"  Phase 3 complete.")
print(f"{'='*70}\n")
