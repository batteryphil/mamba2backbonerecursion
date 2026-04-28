# Mamba Recursive Latent Forcing (RLF)

> **Enabling multi-hop latent reasoning on State Space Models without attention — O(1) memory, baremetal deployable.**

---

## What This Is

This repository contains the full implementation of **Recursive Latent Forcing (RLF)**:
an architectural technique for enabling multi-step variable retention and reasoning on
Mamba SSM models, without attention mechanisms, KV-cache growth, or external memory.

**Original insight credit: ItsMick** — foundational discovery that Mamba natively
handles O(1) loop state over sequence time, enabling explicit iterative computation
that bypasses the KV-Cache entirely.

---

## The Problem It Solves

Standard Mamba (and all SSMs) suffer from **exponential state decay** over long sequences.
Variables assigned early in a prompt (`apples=6, price=0.50`) are attenuated to near-zero
by the time the model generates the answer. SFT training cannot fix this — it's architectural.

RLF fixes it by adding:
1. **Prefix latent scratchpad** — 8 learnable virtual tokens that persist across reasoning loops
2. **Lifeline re-injection** — the original prompt is re-injected at every loop iteration
3. **LoopRoPE** — geometric loop index encoding preventing mode collapse across iterations
4. **Dedicated loop SSM** — a lightweight auxiliary Mamba for iterative computation
5. **Low-rank latent bridge** — translates loop output back to vocabulary distribution
6. **HALT token** — clean termination signal without embedding resize

See [`theory/RLF_THEORY.md`](theory/RLF_THEORY.md) for the full technical document.

---

## Repository Structure

```
mamba2backbonerecursion/
│
├── mamba14b/                   ← Mamba-1.4B RLF (current)
│   ├── rlf_engine_1_4b.py      ← Main engine: RecursiveMamba1_PrefixScratchpad
│   ├── rlf_dataset.py          ← Training data: var chains, math, sequences, bugs
│   ├── rlf_trainer_1_4b.py     ← 3-phase training orchestrator (3a/3b/3c)
│   ├── rlf_chain_test.py       ← Chain accuracy validation (20 held-out tests)
│   ├── rlf_benchmark.py        ← Extended 54-test benchmark (RLF inference mode)
│   ├── auto_eval_rlf.sh        ← Auto-eval trigger (runs after training completes)
│   ├── extended_benchmark.py   ← SFT-mode 54-test benchmark
│   ├── latent_benchmark.py     ← Original SFT benchmark (21 tests)
│   ├── reasoning_probe.py      ← Spacer depth vs accuracy sweep
│   └── targeted_data_gen.py    ← Hybrid CoT + direct code training data generator
│
├── mamba28b/                   ← Mamba-2.8B RLF (Phase 7.5 Golden, archived)
│   ├── mamba_engine.py         ← 2.8B RecursiveMamba2_PrefixScratchpad
│   ├── mamba1_engine.py        ← 1.4B experimental engine (predecessor)
│   ├── dataset_rlf.py          ← Original adversarial chain dataset
│   ├── train_2_8b_rlf.py       ← 2.8B VRAM-conscious RLF trainer
│   ├── phase1_warmup.py        ← Scratchpad warmup
│   ├── phase2_joint_training.py← Joint RLF training
│   ├── phase3_adversarial_training.py
│   ├── phase4_engram_integration.py
│   ├── phase5_rlf_recovery.py  ← Recovery fine-tune
│   ├── the_crucible.py         ← Structural validation (O(1) VRAM proof)
│   ├── dataset_rlf.py          ← Variable chain adversarial dataset
│   ├── test_babi.py            ← bAbI multi-hop reasoning tests
│   ├── test_asymptotic.py      ← Asymptotic state retention tests
│   ├── PHASE14_LATENT_ENGINE_REPORT.md
│   └── PHASE7_FIX_REPORT.md
│
├── theory/                     ← Technical documentation
│   ├── RLF_THEORY.md           ← Full architectural theory + math
│   └── BENCHMARK_RESULTS.md    ← All historical benchmark results
│
├── ssm_infer.c / ssm_infer.h   ← Baremetal C inference engine (UEFI)
├── ssm_weights.c / ssm_weights.h
├── bpe_tokenizer.c / .h        ← BPE tokenizer in C
├── export_mamba_baremetal.py   ← Export weights to baremetal binary format
├── llama2_efi_mamba.c          ← UEFI EFI application (no OS required)
├── ARCHITECTURE.md
├── TUTORIAL.md
├── REPRODUCE.md
└── requirements.txt
```

---

## Mamba-1.4B RLF — Quick Start

### Requirements

```bash
pip install mamba-ssm transformers torch
```

### Training (full 3-phase, ~7 hours)

```bash
cd mamba14b
python rlf_trainer_1_4b.py --phase all
```

Or phase by phase:
```bash
python rlf_trainer_1_4b.py --phase 3a   # 2000 steps — memory warmup (~1hr)
python rlf_trainer_1_4b.py --phase 3b   # 8000 steps — RLF training (~5.5hrs)
python rlf_trainer_1_4b.py --phase 3c   # 1000 steps — SFT recovery (~40min)
```

Resume from checkpoint:
```bash
python rlf_trainer_1_4b.py --phase 3b --resume
```

### Evaluation

```bash
# Chain accuracy test (primary RLF validation)
python rlf_chain_test.py

# Full 54-test benchmark with R5 comparison
python rlf_benchmark.py

# Spacer depth sweep (historical SFT reasoning probe)
python reasoning_probe.py
```

### Auto-eval (waits for training, then runs all tests)

```bash
bash auto_eval_rlf.sh &
```

---

## Architecture Summary

```
Input: "A=42. B=A. C=B. What is C?"
  │
  ▼ embed + all 48 Mamba1 layers (bottom 24 frozen)
  │
  x_prompt (lifeline anchor — saved, never modified)
  │
  ▼ PREPEND 8 latent memory tokens
  │
  [mem₁..mem₈ | tok₁..tokₙ]  ← extended sequence
  │
  ┌─────── RLF Loop (×6 max) ──────────────────────┐
  │  1. lifeline_inject(prompt positions only)       │
  │  2. LoopRoPE(loop_index) — geometric distinction │
  │  3. Top 24 LoRA layers — reasoning core          │
  │  4. mamba1_loop SSM — iterative state            │
  │  5. loop_norm                                    │
  │  6. bridge_up(bridge_down(x)) — vocab translate  │
  │  7. slice off prefix → lm_head → predict token   │
  │     if token == § → HALT                         │
  └────────────────────────────────────────────────--┘
  │
  Output trace:
    L1 → "42"   (resolves A=42)
    L2 → "42"   (resolves B=A=42)
    L3 → "42"   (resolves C=B=42)
    L4 → "§"    (HALT)
```

---

## Key Properties

| Property | Value |
|---|---|
| Base model | state-spaces/mamba-1.4b |
| Parameters (total) | 1,372M |
| Parameters (trainable RLF) | ~150M (top LoRA + loop engine + bridge) |
| Parameters (frozen) | ~634M (bottom 24 layers) |
| Prefix memory | 8 tokens × 2048 = 16,384 params |
| Loop engine | Mamba1 (d_model=2048, d_state=16) = 13.2M params |
| HALT token | `§` = token 7803 in GPT-NeoX vocab |
| Max reasoning loops | 6 |
| Inference VRAM | ~2.9GB |
| Training VRAM (Phase 3a) | 2.86GB |

---

## Results

### Pre-RLF (R5 SFT Hybrid baseline)

| Category | Score |
|---|---|
| Code Generation | 75% |
| Mathematics | 88% |
| Data Structures | 100% |
| Logic/Sequences | 0% |
| Word Problems | 17% |
| Bug Fixing | 20% |
| **Overall (54 tests)** | **56%** |

### Post-RLF (expected — in training)

| Category | Expected |
|---|---|
| Word Problems | 55-65% |
| Logic/Sequences | 40-60% |
| Bug Fixing | 40-50% |
| Code Generation | 75% (preserved) |
| **Overall** | **~70-75%** |

See [`theory/BENCHMARK_RESULTS.md`](theory/BENCHMARK_RESULTS.md) for full historical results.

---

## Training Data

### Mamba-1.4B (mamba14b/rlf_dataset.py)

15,000 samples across 4 chain types:

| Type | Mix | Example |
|---|---|---|
| Variable pointer | 50% | `A=42. B=A. What is B?` → `[42, 42, §]` |
| Math chains | 30% | `qty=6. price=0.50. cost=qty*price. What is cost?` → `[3.00, §]` |
| Sequence patterns | 15% | `x1=2. x2=x1*3. x3=x2*3. What is x4?` → `[54, §]` |
| Bug fix | 5% | `op=minus. fix=plus. What is fixed expr?` → `[a plus b, §]` |

Adversarial mode (40% probability): adds distractor facts, prose sentences,
and random chaos strings to test variable isolation under noise.

### Prior SFT Rounds (base capability)

50,000+ distilled CoT samples trained over 5 rounds:
- Code generation (Python functions, classes, algorithms)
- Mathematical reasoning (arithmetic, word problems)
- Bug detection and fixing
- Data structures and algorithms

---

## Validated Theory

### O(1) Memory (proven on 2.8B model)

The Crucible test (`mamba28b/the_crucible.py`) proved:
- VRAM delta across >20 recursion loops: **ΔV=0.00MB**
- State-tracking proof: base ❌ → RLF ✅
- Kill-shot isolation (Lobotomy test): **100%**

### The Spacer Contribution

Prior to RLF, spacer tokens (`====`) were used as a crude approximation:
- 0 spacers → 40% reasoning accuracy
- 4 spacers → 60% reasoning accuracy (best)
- These validate that iterative computation helps, but can't persist variables

RLF replaces spacers with a proper architectural solution.

---

## Baremetal Deployment

The C inference engine (`ssm_infer.c`, `llama2_efi_mamba.c`) enables deployment
as a UEFI application — no operating system required.

```
+------------------+
|  UEFI Firmware   |
|  llama2_efi_mamba.c ← mamba weights (BF16 quantized .mamb binary)
|  ssm_infer.c     ← O(n) forward pass in pure C
|  bpe_tokenizer.c ← BPE tokenizer
+------------------+
      No OS. No Python. No CUDA. Runs on CPU in UEFI shell.
```

Export to baremetal format:
```bash
python export_mamba_baremetal.py
```

This is the only published LLM with reasoning capability designed to run in
pre-boot firmware without an operating system.

---

## Citation / Attribution

If you use this work:

```
batteryphil/mamba2backbonerecursion (2026)
Recursive Latent Forcing for Mamba SSM Latent Reasoning
Original insight: ItsMick — O(1) loop state over sequence time in Mamba
```
