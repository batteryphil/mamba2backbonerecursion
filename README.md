# Mamba Latent Forcing — Recursive Reasoning Engine (v28)

A recursive neural language model that performs **step-by-step computation** in latent space, proven via controlled experiments against the unmodified base model.

The key finding: after Latent Forcing training, the model moves a variable-chain pointer one hop per loop pass (`A → B → red`). The base model without training produces the same token with identical probability on every loop pass — proof that the behavior is learned, not architectural.

---

## What's in this repo

| File | Purpose |
|------|---------|
| `finetune_mamba_130m_v28.py` | **Main training script** — Latent Forcing on 130m |
| `finetune_mamba2_v28.py` | Latent Forcing on 1.3B model |
| `baseline_vs_v28.py` | **Proof experiment** — base vs trained side-by-side |
| `diagnostic_big_v28.py` | Full 5-phase diagnostic suite (OOD, latent probe, halt, override, accuracy) |
| `diagnostic_suite_v28.py` | Compact 4-phase diagnostic |
| `system2_logic_v1.json` | Chain-following training data (multi-hop variable assignments) |
| `mmlu_format_v17.json` | MMLU-style factual QA training data |

---

## Quick Start

### 1. Install dependencies
```bash
pip install torch transformers mamba-ssm
```

### 2. Train the 130m model
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
python -u finetune_mamba_130m_v28.py 2>&1 | tee train.log
```

Training converges in ~1,500 steps (~1 hour on any RTX GPU).  
Saves: `mamba_130m_v28_latent_forcing_best.pt` (257 MB)

### 3. Run the proof experiment (base vs trained)
```bash
python baseline_vs_v28.py 2>&1 | tee proof.log
```

You will see a table like this — BASE is frozen, V28 moves:
```
Chain: 3-hop: X→Y→Z=Apple  |  Expected: 'Apple'
Loop    BASE (no training)      V28 (Latent Forcing)
──────  ──────────────────────  ──────────────────────
L1      'Z'   p=0.412           'X'     p=0.993
L2      'Z'   p=0.412           'Y'     p=1.000
L3      'Z'   p=0.412           'Y'     p=0.591
L4      'Z'   p=0.412           'Apple' p=0.999 ✅
```

### 4. Run the full diagnostic suite
```bash
python diagnostic_big_v28.py 2>&1 | tee diagnostics.log
```

Runs 5 phases:
- **Phase 1**: OOD extrapolation (4-10 hop chains, trained on 1-3)
- **Phase 2**: Per-loop latent probe — what does each loop output?
- **Phase 3**: Dynamic halt — does compute scale with difficulty?
- **Phase 4**: Reality override — can context beat pretrained priors?
- **Phase 5**: In-distribution accuracy sweep (50 samples per hop count)

---

## How Latent Forcing Works

Standard recursive training: supervise only the **final** answer.
→ Model learns a "delay gate" — holds answer, outputs filler tokens, emits answer at the end.

**Latent Forcing**: supervise **every loop** with the corresponding chain step.

For `A = red. B = A. What is B?`:
```
Loop 0 target: "A"    ← point at the anchor variable
Loop 1 target: "red"  ← resolve to the value
```

For `X = Apple. Y = X. Z = Y. What is Z?`:
```
Loop 0 target: "X"      ← identify start
Loop 1 target: "Y"      ← move one hop
Loop 2 target: "Apple"  ← reach the resolved value
```

The loss function checks every intermediate loop — the model cannot "cheat" by waiting:

```python
for loop_i in range(n_loops):
    # Get this loop's specific target token
    tgt_id = chain_targets[b][min(loop_i, len(chain_targets[b]) - 1)]

    # Pointer mask: model can only output tokens present in the prompt
    mask = torch.full((vocab_size,), float("-inf"))
    mask[unique_input_tokens] = 0.0

    # Loss at the answer position for this specific loop
    logits_b  = logits_step[b, ans_start - 1, :] + mask
    loop_loss += F.cross_entropy(logits_b.unsqueeze(0), torch.tensor([tgt_id]))

total_loss = torch.stack(loop_losses).mean()
```

---

## Architecture

```
Input
  │
  └──► Embedding (d=768)
         │
         ▼
     Layers 0-5   ← FROZEN (stable base features)
         │
   ┌─────▼─────────────────────────────────────────┐
   │         LOOP  (runs N times)                  │
   │                                               │
   │   + step_emb[loop_i]  ← "which tick am I?"   │
   │         │                                     │
   │     Layers 6-23 (LoRA rank=8)                 │
   │         │                                     │
   │   + mamba3_core(x)    ← Mamba1 stateful scan  │
   │         │                                     │
   │     loop_norm (RMSNorm)                       │
   │         │                                     │
   │   lm_head → loss vs chain_targets[loop_i]     │
   └───────────────────────────────────────────────┘
         │
       Answer
```

**Trainable parameters**: 43.3M of 130M total  
**VRAM**: 0.46 GB  
**Training speed**: ~2,000-4,000 TPS

---

## Results

### Training
| Step | Loss | AllLoopAcc | FinalAcc |
|------|------|-----------|---------|
| 50 | 2.231 | 65.9% | 68.8% |
| 100 | 0.313 | 87.7% | 92.1% |
| 500 | 0.011 | 99.9% | 99.9% |
| 1,500 *(early stop)* | 0.001 | **100.0%** | **100.0%** |

Validation AllLoopAcc: **100.0%** | Train-Val gap: **+0.0pp**

### Reality Override
| Test | Base | V28 |
|------|------|-----|
| "Fire is icy cold. Bob felt?" | ❌ | ✅ cold |
| "Cats bark. Sarah's cat sounds?" | ❌ | ✅ bark |
| "Sun is freezing. Coffee temp?" | ❌ | ✅ cold |
| "Sugar tastes bitter. Tom tasted?" | ❌ | ✅ bitter |
| "Up means down. Sarah jumped up = ?" | ❌ | ✅ down |
| **Score** | **0/5** | **5/5** |

---

## Training Configuration

```python
MODEL        = "state-spaces/mamba-130m"
BATCH_SIZE   = 8
ACCUM        = 4           # effective batch = 32
STEPS        = 50_000      # early stop fired at 1,500
SEQ_LEN      = 256
LR_loop      = 1e-3        # step_emb, loop_norm, mamba3_core
LR_lora      = 5e-4        # LoRA adapters
CLIP         = 1.0         # gradient norm clipping
VAL_SPLIT    = 10%         # stratified by hop count
EARLY_STOP   = val AllLoopAcc >= 99.5% × 3 consecutive checks
```

---

## Data Format

Training data lives in `system2_logic_v1.json`. Each sample:

```json
{
  "text": "A = red. B = A. C = B. What is C?\nAnswer: red",
  "hops": 3,
  "chain_targets": ["A", "B", "red"],
  "chain_tgt_ids": [32, 33, 1445],
  "ans_start": 16
}
```

All targets are **pre-tokenized at load time** — no tokenizer calls in the training hot loop.

---

## Known Limitations

1. **OOD hop length**: Trained on 1-3 hops, fails at 4+. Fix: add longer chains to training data.
2. **Halt detection**: Current "token repeats = done" logic fires too early on some chains, returning the variable letter instead of the resolved word.
3. **Pointer mask**: At inference, only allows tokens present in the prompt. This is intentional but can cause failures when the target word is not in the prompt.

---

## Version History

| Version | Key Change |
|---------|-----------|
| v25 | MIMO Phase Rotator (unit-circle BPTT, JIT CUDA) |
| v26 | 130m fine-tune, THINK-token supervision |
| v27 | 1.3B scale, bf16, LoRA on top 24 layers |
| **v28** | **Latent Forcing** — per-loop supervision, Mamba1 scan block, 0.0pp val gap |

---

## Paper

A full research paper with experimental proof is in:  
`latent_forcing_paper.md` (in the project AI artifacts directory)

The paper includes the controlled baseline experiment tables, architecture details, training data format, code excerpts, and discussion of limitations.
