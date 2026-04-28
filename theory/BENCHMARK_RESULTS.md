# Benchmark Results — RLF Mamba Models

## Model Lineage

```
state-spaces/mamba-1.4b (base)
  └─ R1: SFT 10k samples          → 48% benchmark
  └─ R2: SFT 25k samples          → 48% benchmark
  └─ R3: SFT 50k + targeted       → 76% benchmark  ← best SFT
  └─ R4: + CoT reasoning training → 52% benchmark  (code regression)
  └─ R5: Hybrid 60k steps         → 67% benchmark
  └─ RLF: Prefix scratchpad + RLF → TBD (training April 28, 2026)
```

---

## Extended Benchmark Results (54 tests, 300 max tokens)

### R5 SFT Hybrid (current best hybrid)

| Category | Pass | Total | Rate |
|---|---|---|---|
| Algorithm | 3 | 4 | 75% |
| Bug Fix | 1 | 5 | 20% |
| Code Gen | 6 | 8 | 75% |
| Code Gen (Complex) | 2 | 6 | 33% |
| Completion | 2 | 3 | 67% |
| Data Structures | 4 | 4 | 100% |
| Logic | 0 | 4 | 0% |
| Math | 7 | 8 | 88% |
| Strings | 4 | 4 | 100% |
| Unit Tests | 0 | 2 | 0% |
| Word Problem | 1 | 6 | 17% |
| **TOTAL** | **30** | **54** | **56%** |

**Throughput:** 15.3 tok/s  
**Bigram repetition:** 0.061 (baseline mamba: ~0.31)  
**Attractor check:** CLEAN

### R3 SFT Direct (best benchmark-only)

| Category | Pass | Total | Rate |
|---|---|---|---|
| Code Generation | 6 | 8 | 75% |
| Mathematics | 4 | 5 | 80% |
| Bug Fixing | 1 | 3 | 33% |
| Algorithm | 3 | 3 | 100% |
| Data Structures | 2 | 2 | 100% |
| Code Quality | 0 | 2 | 0% |
| **TOTAL** | **16** | **21** | **76%** |

---

## Reasoning Probe Results (spacer depth sweep)

Tests whether latent "thinking" (spacer tokens) improves multi-step reasoning.

| Spacer Count | R3 | R4 | R5 |
|---|---|---|---|
| 0 spacers | 20% | 40% | 40% |
| 4 spacers | 20% | 60% | 60% |
| 8 spacers | 20% | 20% | 40% |
| 16 spacers | 20% | 40% | 40% |

**Conclusion:** 4 spacers optimal. Latent thinking adds +1 correct answer vs no spacers.

---

## 2.8B Model — Phase 7.5 Golden Checkpoint

Validated on BIG-Bench Lite / General Logic (16 probes):

| Metric | Score |
|---|---|
| BIG-Bench Lite | 75.0% (12/16) |
| The Crucible (structural validation) | 3/3 proofs passed |
| O(1) VRAM verification | ΔV=0.00MB across >20 loops |
| Lobotomy test (kill-shot isolation) | 100% |
| Conversational multi-hop | ✅ (John/Mary apples: correct) |

---

## Training Speed Reference

| Model | Steps | Wall Time | Notes |
|---|---|---|---|
| Mamba-1.4B R3 | 20,000 | ~3.5 hrs | Direct SFT |
| Mamba-1.4B R5 | 60,000 | ~10 hrs | Hybrid SFT |
| Mamba-1.4B RLF 3a | 2,000 | ~62 min | Memory+bridge only (2.86GB) |
| Mamba-1.4B RLF 3b | 8,000 | ~5.5 hrs | Full RLF (est.) |
| Mamba-1.4B RLF 3c | 1,000 | ~40 min | SFT recovery |

---

## Inference Speed

| Model | Throughput | VRAM |
|---|---|---|
| base mamba-1.4b | ~18 tok/s | 2.8GB |
| R5 SFT | ~15.3 tok/s | 3.4GB |
| RLF inference (6 loops/token) | ~2-3 tok/s (est.) | 2.9GB |

RLF is slower per token (6 backbone passes per output token) but uses **less
VRAM than the SFT model** because the prefix scratchpad replaces the LoRA adapter.
