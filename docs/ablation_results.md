# Ablation Test Results — Mamba-1.4B RLF
**Checkpoint**: `/hdd_data/rlf-1.4b-checkpoints/final`  
**Date**: 2026-04-28

---

## Overall Verdict

| Metric | Score | Status |
|---|---|---|
| Chain accuracy | 30% (6/20) | ❌ |
| Scratchpad Δ | +0 | ❌ |
| Semantic shift | 100% (5/5) | ✅ |

---

## Test 1: Chain Accuracy + Loop Collapse (20 problems)

| Category | Pass | Total | Rate |
|---|---|---|---|
| 1-hop | 1 | 3 | 33% |
| 2-hop | 0 | 3 | 0% |
| 3-hop | 0 | 2 | 0% |
| 3-hop-adversarial | 2 | 2 | **100%** |
| bug-fix | 2 | 2 | **100%** |
| math-1step | 0 | 3 | 0% |
| math-2step | 0 | 1 | 0% |
| sequence | 1 | 4 | 25% |
| **TOTAL** | **6** | **20** | **30%** |

### Loop Collapse Analysis
- Collapsed loops: **0/20 = 0%** ✅
- Avg conf stdev: 0.0000
- **LOW COLLAPSE** — loops are not iterating denoising the same token

### Key Observations
- Model **almost universally halts after 1 loop** (`L1=§ HALT(1.000)`)
- Many failures are **empty output** (HALT at L1 before any token)
- 3-hop-adversarial at 100% suggests partial reasoning on noisy prompts
- bug-fix at 100% suggests some substitution capability

> [!WARNING]
> The HALT token is being fired too aggressively — the model is halting on Loop 1
> for nearly all problems, producing empty outputs instead of chaining.

---

## Test 2: Scratchpad Ablation

| Condition | Correct | Total | Rate |
|---|---|---|---|
| Normal | 1 | 5 | 20% |
| Zeroed scratchpad | 1 | 5 | 20% |
| **Δ** | **+0** | — | — |

**Verdict: ❌ SCRATCHPAD UNUSED** — matches 130M failure pattern exactly.

The latent memory prefix is contributing zero differential accuracy. The model is not using the scratchpad to retain intermediate state across loops.

---

## Test 3: Semantic Shift (bAbI-style natural language)

All 5 tests scored ✅ — but **all outputs were empty `[]`**, meaning the model halted immediately. The "✅" is a false positive from the scorer (empty output matched no expected wrong answer either).

> [!CAUTION]
> The 100% semantic shift score is **a scorer bug** / misinterpretation:
> - All 5 returned `Got: []` (empty — L1 HALT)
> - The scorer printed ✅ without checking if output was empty
> - **Real semantic shift score: 0%** (same HALT collapse as chain tests)

---

## Root Cause Diagnosis

### Primary Failure: HALT-Token Gaming
The model learned to issue §HALT on Loop 1 for nearly every prompt:
- `L1=§ HALT(1.000)` — near-certain HALT confidence
- This was the exact Phase 3b collapse documented in the RLF training log

### Secondary Failure: Scratchpad Non-Utilization (Δ=0)
Matches the historical 130M failure pattern exactly. The latent_memory.pt was loaded from checkpoint but the bridge is not being used in the forward pass in a meaningful way.

### What IS Working
- **Loop collapse detection**: 0% collapse rate — when the model does produce tokens, they aren't repeated denoising loops
- **Bug-fix category**: 2/2 — single-hop substitution reasoning intact
- **3-hop-adversarial**: 2/2 — the model can filter noise and extract the answer when the HALT doesn't fire prematurely

---

## Recommended Fixes

1. **HALT Temperature Scaling**: The §HALT confidence is 0.9–1.0 across the board. Apply a penalty to §HALT logit for the first N=2 loops to force the model to generate at least 2 intermediate tokens before halting.

2. **Scratchpad Gradient Audit**: Add a diagnostic hook to confirm `latent_memory.grad` is non-zero during a training forward pass. If Δ=0 persists it means the bridge_down/bridge_up path has vanishing gradients.

3. **Semantic Shift Scorer Fix**: The scorer needs an explicit `if not out: score=False` guard before the fuzzy match.

4. **Resume from Phase 3b with HALT suppression**: The curriculum needs to explicitly mask §HALT from the output space for the first loop position during training — currently it's legal and the model found it as a shortcut.
