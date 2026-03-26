"""
phase2_joint_training.py — Joint Training: Reasoning Core + Scratchpad + Bridge
=================================================================================
Phase 2: Unfreeze LoRA + loop engine alongside latent_memory + bridge.
Train on extended 15-hop curriculum for OOD length generalization.

What gets trained (unfrozen):
  - LoRA adapters (layers 48-63, in_proj + out_proj)
  - Loop engine (mamba2_core)
  - Loop norm
  - Lifeline gate
  - Latent memory (prefix scratchpad)
  - Latent bridge (bridge_down + bridge_up)

What stays frozen:
  - Mamba-2 2.7B base backbone layers 0-63 (base weights, not LoRA)
  - Embeddings
  - LM head

Gradient checkpointing on LoRA layers to fit 12GB VRAM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json, random, time, os, re, sys
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel, Mamba2
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint as grad_ckpt

from mamba_engine import (
    RecursiveMamba2_PrefixScratchpad, LoRALinear, LoopRoPE,
    tokenizer, HALT_ID,
    DEVICE, MODEL_ID, BASE_SPLIT, LORA_RANK, SEQ_LEN, BATCH_SIZE, ACCUM,
    PREFIX_M, D_MODEL, BRIDGE_RANK, MAX_LOOPS,
    LOOP_HEADDIM, LOOP_D_STATE, LOOP_EXPAND,
)

# ── Phase 2 Config ────────────────────────────────────────────────────────────
PHASE2_STEPS     = 3_000
LOG_EVERY        = 50
VAL_EVERY        = 500
PHASE1_CKPT      = "mamba2_2.7b_phase1_scratchpad.pt"
PHASE1_BEST      = "mamba2_2.7b_phase1_scratchpad_best.pt"
STEP3000_CKPT    = "saved_weights/mamba2_2.7b_rlf_rope_best_step3000_val97.3.pt"
SAVE_PATH        = "mamba2_2.7b_phase2_joint.pt"
EARLY_STOP_ACC   = 95.0
EARLY_STOP_COUNT = 3

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

sys.stdout = Tee('training_phase2.log')

print(f"\n{'='*70}")
print(f"  PHASE 2 — Joint Training: Reasoning Core + Scratchpad + Bridge")
print(f"  Goal: OOD 15-hop generalization via extended curriculum")
print(f"  Device: {DEVICE} | Steps: {PHASE2_STEPS}")
print(f"{'='*70}\n")


# ── Generate 15-hop curriculum ────────────────────────────────────────────────
def generate_extended_curriculum() -> list[dict]:
    """Generate 11-15 hop chains for OOD training.

    Uses single-letter variables and short colors to fit SEQ_LEN=128.
    A 15-hop chain with 1-letter vars ≈ 90 tokens (fits 128).

    Returns:
        list of training samples with reasoning chains
    """
    rng = random.Random(2024)
    # Short colors only (3-5 chars) to minimize token count
    colors = [
        "red", "blue", "gold", "pink", "gray", "tan", "teal",
        "jade", "plum", "aqua", "lime", "rose", "navy", "sage",
    ]
    # Single letters for variables — max 26 hops
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    samples: list[dict] = []
    for n_hops in range(11, 16):
        for _ in range(200):
            color = rng.choice(colors)
            # Pick n_hops+1 unique letters and shuffle
            vars_used = letters[:n_hops + 1]
            rng.shuffle(vars_used)

            # Build compact chain: "A=red. B=A. C=B. ... What is Z?"
            first = vars_used[0]
            parts = [f"{first}={color}."]
            for i in range(1, len(vars_used)):
                parts.append(f"{vars_used[i]}={vars_used[i-1]}.")
            last = vars_used[-1]
            prompt = " ".join(parts) + f" What is {last}?\nAnswer: "

            # Build compact reasoning: "B←A; C←B; ..."
            r_steps = []
            for i in range(1, len(vars_used)):
                r_steps.append(f"{vars_used[i]}←{vars_used[i-1]}")
            reasoning = "; ".join(r_steps)
            answer = f" [Reasoning] {reasoning} [Answer] {color} <HALT>"

            samples.append({
                "text": prompt + answer,
                "answer": answer,
                "type": "chain",
                "hops": n_hops,
            })

    rng.shuffle(samples)
    return samples


# ── Data helpers ──────────────────────────────────────────────────────────────
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
    """Load training data + extended 15-hop curriculum."""
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

    # Load existing curriculum (1-10 hops)
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

    # Generate and add 11-15 hop curriculum
    ext = generate_extended_curriculum()
    ext_ok = 0
    ext_fail_reasons: dict[str, int] = {}
    for item in ext:
        if "[Reasoning]" in item.get("answer", ""):
            ct = _parse_v2_chain(item["answer"])
            full = item["text"]
            if ct is None:
                ext_fail_reasons["parse_fail"] = ext_fail_reasons.get("parse_fail", 0) + 1
                continue
            if len(ct) < 2:
                ext_fail_reasons["too_short"] = ext_fail_reasons.get("too_short", 0) + 1
                continue
            s = _prep(full, ct)
            if s is None:
                # Debug: why did _prep fail?
                enc_ids = tokenizer.encode(full, add_special_tokens=False)
                ans_start = find_answer_start(enc_ids)
                if ext_fail_reasons.get("prep_fail", 0) == 0:
                    print(f"    DEBUG first prep_fail: toks={len(enc_ids)} ans_start={ans_start}")
                    print(f"    text[:80]={full[:80]!r}")
                ext_fail_reasons["prep_fail"] = ext_fail_reasons.get("prep_fail", 0) + 1
                continue
        else:
            ct = _parse_chain(item["text"])
            s = _prep(item["text"], ct) if ct and len(ct) >= 2 else None
            if s is None:
                ext_fail_reasons["chain_fail"] = ext_fail_reasons.get("chain_fail", 0) + 1
                continue
        samples.append(s)
        ext_ok += 1
    print(f"  Extended 11-15 hop: {ext_ok:,} samples generated")
    if ext_fail_reasons:
        print(f"  Extended failures: {ext_fail_reasons}")

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


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 Unfreeze: Reasoning core + scratchpad + bridge
# ══════════════════════════════════════════════════════════════════════════════
def unfreeze_for_phase2(model: RecursiveMamba2_PrefixScratchpad) -> None:
    """Unfreeze reasoning core alongside scratchpad + bridge.

    Unfrozen:
      - LoRA adapters (lora_A, lora_B on layers 48-63)
      - Loop engine (mamba2_core)
      - Loop norm
      - Lifeline gate
      - Latent memory
      - Bridge (bridge_down + bridge_up)

    Frozen (stays frozen):
      - Base backbone weights (base_weight buffers in LoRALinear)
      - Embeddings
      - LM head

    Args:
        model: The RecursiveMamba2_PrefixScratchpad model instance
    """
    # Step 1: Freeze EVERYTHING first
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: Unfreeze LoRA adapters on upper layers
    for layer in model.all_layers[BASE_SPLIT:]:
        mx = layer.mixer
        for attr in ("in_proj", "out_proj"):
            module = getattr(mx, attr, None)
            if module is not None and isinstance(module, LoRALinear):
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True

    # Step 3: Unfreeze loop engine + norm + gate
    for param in model.mamba2_core.parameters():
        param.requires_grad = True
    for param in model.loop_norm.parameters():
        param.requires_grad = True
    model.lifeline_gate.requires_grad = True

    # Step 4: Unfreeze scratchpad + bridge (from Phase 1)
    model.latent_memory.requires_grad = True
    for param in model.bridge_down.parameters():
        param.requires_grad = True
    for param in model.bridge_up.parameters():
        param.requires_grad = True

    # Report
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_lora = sum(p.numel() for n, p in model.named_parameters()
                 if p.requires_grad and "lora" in n.lower())
    n_loop = sum(p.numel() for p in model.mamba2_core.parameters())
    n_mem = model.latent_memory.numel()
    n_bridge = (sum(p.numel() for p in model.bridge_down.parameters())
                + sum(p.numel() for p in model.bridge_up.parameters()))

    print(f"\n{'='*70}")
    print(f"  PHASE 2 — Joint Training Unfreeze")
    print(f"{'='*70}")
    print(f"  Frozen:     {frozen:,} (base backbone + embeddings + lm_head)")
    print(f"  Trainable:  {trainable:,}:")
    print(f"    LoRA:          {n_lora:,}")
    print(f"    Loop engine:   {n_loop:,}")
    print(f"    Lifeline gate: {model.lifeline_gate.numel():,}")
    print(f"    Memory:        {n_mem:,} ({PREFIX_M} × {D_MODEL})")
    print(f"    Bridge:        {n_bridge:,} ({D_MODEL}→{BRIDGE_RANK}→{D_MODEL})")
    print(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Model Loading (2-stage: step3000 LoRA + Phase 1 memory/bridge)
# ══════════════════════════════════════════════════════════════════════════════
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

# Stage 1: Load step 3000 LoRA weights (reasoning core)
print(f"  Stage 1: Loading LoRA from {STEP3000_CKPT}...", flush=True)
ckpt_rlf = torch.load(STEP3000_CKPT, map_location=DEVICE, weights_only=False)
sd_rlf = ckpt_rlf.get("model_state_dict", ckpt_rlf)
own_sd = model.state_dict()
loaded_rlf = 0
for k, v in sd_rlf.items():
    if k in own_sd and own_sd[k].shape == v.shape:
        own_sd[k] = v
        loaded_rlf += 1
model.load_state_dict(own_sd, strict=False)
print(f"  Loaded {loaded_rlf} tensors from step 3000 (LoRA + loop engine + gate)")
del ckpt_rlf, sd_rlf, own_sd
import gc; gc.collect()
torch.cuda.empty_cache()

# Stage 2: Load Phase 1 memory + bridge weights (CPU to avoid VRAM spike)
phase1_path = PHASE1_BEST if os.path.exists(PHASE1_BEST) else PHASE1_CKPT
print(f"  Stage 2: Loading memory+bridge from {phase1_path}...", flush=True)
ckpt_p1 = torch.load(phase1_path, map_location="cpu", weights_only=False)
sd_p1 = ckpt_p1.get("model_state_dict", ckpt_p1)
p1_loaded = 0
for key in ["latent_memory", "bridge_down.weight", "bridge_up.weight"]:
    if key in sd_p1:
        param = dict(model.named_parameters()).get(key, None)
        buf = dict(model.named_buffers()).get(key, None)
        target = param if param is not None else buf
        if target is not None:
            target.data.copy_(sd_p1[key].to(target.device))
            p1_loaded += 1
print(f"  Loaded {p1_loaded} Phase 1 tensors (memory + bridge)")
del ckpt_p1, sd_p1
import gc; gc.collect()
torch.cuda.empty_cache()

# Apply Phase 2 unfreeze
unfreeze_for_phase2(model)

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

# ── Optimizer (lower LR for joint training) ───────────────────────────────────
g_memory = [model.latent_memory]
g_bridge = (list(model.bridge_down.parameters())
            + list(model.bridge_up.parameters()))
g_gate = [model.lifeline_gate]
g_loop = (list(model.mamba2_core.parameters())
           + list(model.loop_norm.parameters()))
g_lora = [p for n, p in model.named_parameters()
           if p.requires_grad and "lora" in n.lower()]

optimizer = optim.AdamW([
    {"params": g_memory, "lr": 5e-5, "weight_decay": 0.0},
    {"params": g_bridge, "lr": 5e-5, "weight_decay": 0.01},
    {"params": g_gate,   "lr": 5e-5, "weight_decay": 0.0},
    {"params": g_loop,   "lr": 1e-5, "weight_decay": 0.01},
    {"params": g_lora,   "lr": 1e-5, "weight_decay": 0.01},
])
scheduler = CosineAnnealingLR(optimizer, T_max=PHASE2_STEPS // ACCUM, eta_min=1e-7)

vram_init = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0
print(f"\n{'─'*70}")
print(f"  Phase 2 | LoRA + Loop + Memory + Bridge | {PHASE2_STEPS} steps")
print(f"  BS={BATCH_SIZE} × ACCUM={ACCUM} = effective {BATCH_SIZE*ACCUM}")
print(f"  LR: LoRA/Loop=1e-5 | Memory/Bridge/Gate=5e-5")
print(f"  VRAM after init: {vram_init:.2f}GB")
print(f"{'─'*70}\n")


# ── Validation ────────────────────────────────────────────────────────────────
def run_validation(val_pool: list, n_batches: int = 30) -> tuple:
    """Run validation (keep model.training=True for correct forward dispatch)."""
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


# ── Training Loop ─────────────────────────────────────────────────────────────
model.train()
optimizer.zero_grad()
t0 = time.time()
total_loss = total_aa = total_fa = total_ha = 0.0
early_stop_hits = 0
best_val_acc = 0.0
last_train_aa = 0.0

for step in range(1, PHASE2_STEPS + 1):
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
        print(f"  Val AllLoop: {vaa:5.1f}% | Answer: {vfa:5.1f}% | Halt: {vha:5.1f}%",
              flush=True)
        print(f"  Train-Val gap: {gap:+.1f}pp{flag}", flush=True)

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
                "phase": "phase2_joint",
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

    if step % 1000 == 0:
        torch.save({
            "model_state_dict": model.state_dict(),
            "step": step, "halt_id": HALT_ID, "rope": True,
            "d_model": d_model, "model_id": MODEL_ID,
            "prefix_m": PREFIX_M, "has_bridge": True,
            "phase": "phase2_joint",
        }, SAVE_PATH.replace(".pt", f"_step{step}.pt"))
        print(f"  💾 step {step} checkpoint", flush=True)
else:
    torch.save({
        "model_state_dict": model.state_dict(),
        "step": PHASE2_STEPS, "halt_id": HALT_ID, "rope": True,
        "d_model": d_model, "model_id": MODEL_ID,
        "prefix_m": PREFIX_M, "has_bridge": True,
        "phase": "phase2_joint",
    }, SAVE_PATH)
    print(f"\n✅ Phase 2 joint training complete — {SAVE_PATH}")


# ── OOD Length Test ───────────────────────────────────────────────────────────
model.eval()
print(f"\n{'='*70}")
print(f"  PHASE 2 OOD TEST — trained 1-15 hops, testing 20/25/30")
print(f"{'='*70}")


def make_ood_chain(n: int, color: str) -> str:
    """Generate n-hop test chain."""
    vars_l = [f"X{i}" for i in range(1, n + 2)]
    chain = f"{vars_l[0]}={color}. "
    for i in range(1, len(vars_l)):
        chain += f"{vars_l[i]}={vars_l[i-1]}. "
    chain += f"What is {vars_l[-1]}?\nAnswer:"
    return chain


ood_tests = [
    (make_ood_chain(4, "blue"), "blue", "4-hop (in-dist)"),
    (make_ood_chain(10, "gold"), "gold", "10-hop (curriculum)"),
    (make_ood_chain(15, "diamond"), "diamond", "15-hop (curriculum)"),
    (make_ood_chain(20, "quantum"), "quantum", "20-hop (OOD)"),
    (make_ood_chain(25, "nebula"), "nebula", "25-hop (OOD)"),
]

for prompt, expected, label in ood_tests:
    ids_ = tokenizer.encode(prompt, add_special_tokens=False,
                            return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        n, trace, answer = model(ids_)
    hit = ("✅ GENERALIZES" if expected.lower() in answer.lower()
           else f"❌ FAILS → {answer!r}")
    print(f"\n  {label}: {hit}")
    for lbl, tok, prob in trace:
        mark = (" ← ✅" if tok.lower() == expected.lower()
                else (" <HALT>" if tok == "<HALT>" else ""))
        print(f"    {lbl:5s}  {tok!r:14s}  p={prob:.4f}{mark}")

print(f"\n{'='*70}")
print(f"  Phase 2 complete. If 20-hop works: TRUE OOD GENERALIZATION.")
print(f"{'='*70}\n")
