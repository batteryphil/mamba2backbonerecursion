# Mamba RLF Engine — Recursive Latent Forcing Research

> **Status**: Active research. 130M v2 retraining in progress (March 2026).

A recursive reasoning architecture grafted onto frozen **Mamba-2 2.7B** and **Mamba-1 130M** backbones. The engine uses Recursive Latent Forcing (RLF) loops with a Prefix Latent Scratchpad and Latent Communication Bridge to perform multi-hop chain reasoning at **O(1) memory** — no KV-cache, no activation accumulation.

---

## Research Findings

### ✅ Confirmed Achievements

#### 1. O(1) Memory — Formally Confirmed
The prefix scratchpad mechanism maintains **constant VRAM** regardless of loop depth. Confirmed via a controlled 100-sample × 16-loop sweep:

```
VRAM at loop 1:  265 MB
VRAM at loop 16: 265 MB   ← flat
Delta:           0 MB      ✅ No KV-cache accumulation
```

This is the core architectural claim and it holds. A Transformer equivalent running 16 attention passes would grow linearly.

#### 2. Temporal Ablation — Loop Delta Proven (+49%)

A rigorous 3-arm scientific experiment proved that **recursive looping is the mechanism**, not weight memorization:

| Arm | Setup | Accuracy |
|-----|-------|----------|
| A | Stock mamba-130m (5-shot baseline) | **36%** |
| B | Trained weights, **max_loops=1** (lobotomy) | **0%** |
| C | Trained weights, **max_loops=16** (full RLF) | **49%** |

**Loop delta: +49%.** The raw trained weights with 1 loop score 0%. More loops score 49%. The mechanism is confirmed architecture-dependent, not memorization.

#### 3. Phase 5 RLF Recovery — 83% Accuracy
After a regression introduced during Phase 4 Engram training (RLF accuracy dropped from ~75% → 50%), a targeted Phase 5 recovery run at LR=1e-4 with the gate head frozen successfully recovered and exceeded previous performance:

```
Phase 4 Regression: 50% RLF Acc
Phase 5 Recovery:   83% RLF Acc   ✅ (+33% recovery)
```
Checkpoint: `saved_weights/mamba130m_phase5_recovery_best.pt`

#### 4. Engram Contextual Gating — Gate Polarization Works
The `engram_gate_head` (a trainable MLP classifier on the hidden state at the answer boundary) successfully learned to discriminate factual vs. poison injections:

```
Gate on matching fact:   0.89 → ACCEPT
Gate on mismatched fact: 0.11 → REJECT
Accuracy at convergence: 100% gate discrimination
```

The gate correctly identifies coherent vs. incoherent context — the mechanism for selective CPU factual offloading is architecturally sound.

#### 5. Discovered: Latent Sequence Replay Mechanism
The ablation study revealed the model's actual multi-hop mechanism is **latent sequence replay**, not abstract logical recursion. The training supervision at position `ans_start-1` with target `chain_targets[loop_i]` causes the model to replay the input sequence token-by-token across loops:

```
Loop 1 → 'V'      (first token of V1=Blue.)
Loop 2 → '1'
Loop 3 → '='
Loop 4 → 'Blue'   ← answer appears here
Loop 5 → '.'
Loop 6 → 'V'
```

The answer token slides into loop 4 at the value's position in the original sequence. This is a novel finding — temporal position is the reasoning primitive.

---

### ❌ Confirmed Failures & Problems

#### 1. Phase 4 Engram Integration — RLF Regression
Integrating the Engram gate (Phase 4) caused a severe RLF accuracy regression from ~75% → 50%. Root cause: joint training of the gate alongside the RLF engine created conflicting gradient signals. The gate loss dominated, suppressing the RLF reasoning signal.

**Fix**: Phase 5 recovery — freeze gate, retrain RLF at LR=1e-4.

#### 2. 2.7B Phase 2 Joint Training — Overfitting on In-Distribution
The 2.7B model achieved 97.3% val accuracy on 2-6 hop chains but **generalized poorly** to 7+ hop OOD chains. Arm A (stock 5-shot) outperformed the trained model on deep OOD chains in the ablation, suggesting the model memorized the 3-6 hop distribution rather than learning a generalizable algorithm.

**Root cause**: Training distribution too narrow (hops 2-6 only). The solution for v2 retraining is hops 2-8 in Phase 2 and adversarial hardening in Phase 3.

#### 3. Baremetal UEFI Deployment — AVX2 Incompatibility
Attempted to run the Mamba-2 2.7B model in a bare-metal UEFI C environment (QEMU) to validate CPU inference for the factual offloading use-case. Failed due to:
- **`#UD` (Invalid Opcode) exception** — the UEFI firmware environment does not support AVX2 SIMD instructions, which the compiled SSM inference kernel uses
- The 2.85GB model bin loads correctly, but weight processing requires AVX2 for matrix ops
- Stripping AVX2 objects from the build removes ~40% of inference performance

**Status**: Blocked. CPU-only SSM inference in UEFI is feasible but requires full scalar fallback implementation in DjiBLAS.

#### 4. Halt Prediction — Never Reliably Learned
Across all training runs (Phase 1-5, both 130M and 2.7B), `halt_acc` remained near 0.00 in Phases 1-2 and rarely exceeded 0.20 in Phase 3. The model learns to predict the correct answer token at the correct loop but almost never correctly predicts HALT at the right loop.

**Root cause**: The HALT token is at the end of the target sequence (`chain_targets[-1]`). With `max_loops=6`, the training signal for HALT only fires at `loop_i = len(target_ids) - 1`, which is typically beyond the loop budget. The model never sees a clean HALT gradient.

**Proposed fix**: Explicit HALT curriculum — train 30% of examples with short 1-2 hop chains where HALT falls within the loop budget.

#### 5. Inference Position Bug — Training/Inference Mismatch
The model's training supervision applied loss at `logits[ans_start-1]` (the token before the answer colon). But the original inference code read `logits[-1]` (the very last position), causing the model to predict the next contextual token (always `V`, the variable prefix) instead of the answer. This explains why early inference tests showed 0% accuracy despite high training accuracy.

**Fix**: Inference now reads at `logits[T_orig-2]` to match the supervised position.

---

## Architecture (130M)

```
┌──────────────────────────────────────────────────────────┐
│  Frozen Mamba-1 130M Base (45.3M params, bfloat16)       │
│  Layers 0-17: Pure base (frozen)                         │
│  Layers 18-23: Base + LoRA-4 adapters (trainable)        │
│                                                          │
│  Prefix Latent Scratchpad  [8 tokens × 768]              │
│  Latent Communication Bridge  [768→64→768, low-rank]     │
│  Recursive Loop Engine  [Mamba1 core, 3.8M params]       │
│  Lifeline Gate  [768-dim learnable residual scalar]       │
│  Loop RoPE Encoding  [position by loop index]            │
└──────────────────────────────────────────────────────────┘

Total trainable: 45.6M params (LoRA + engine + bridge + memory)
Total frozen:    45.3M params (base backbone)
```

---

## Training Pipeline — Mamba-130M v2 (Current)

Fully automated. Scaled-down from 2.7B overfitting run. Wider hop range prevents memorization.

```bash
# Run all 3 phases automatically
python train_130m.py

# Resume from a specific phase
python train_130m.py --phase 2
```

| Phase | Steps | LR | Hops | Purpose |
|-------|-------|----|------|---------|
| 1 | 1500 | 1e-3 | 2-5 | Warmup: prefix memory + bridge only |
| 2 | 2000 | 5e-4 | 2-8 | Joint: LoRA + engine, wider distribution |
| 3 | 1500 | 1e-4 | 2-8 | Adversarial: chaos + prose distractors |

Early stops at rolling(100) acc ≥ target per phase.

---

## Checkpoints

| File | Description | RLF Acc |
|------|-------------|---------|
| `mamba130m_v1_archive_best.pt` | Phase 5 recovery (archived v1) | 83% |
| `mamba130m_v2_best.pt` | Current v2 training run (in progress) | — |
| `mamba2_2.7b_rlf_rope_best_step3000_val97.3.pt` | 2.7B Phase 2 best | 97.3%* |

> *97.3% on in-distribution 2-6 hop chains. Generalizes poorly to 7+ hops (OOD).

---

## Requirements

- **GPU**: 12GB+ VRAM (tested on RTX 3060 12GB)
- **Python**: 3.10+
- **CUDA**: 11.8+

```bash
pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install causal-conv1d>=1.2.0
pip install mamba-ssm>=2.0.0
pip install transformers accelerate
```

---

## Key Files

| File | Purpose |
|------|---------|
| `mamba1_engine.py` | Core: `RecursiveMamba1_PrefixScratchpad` (130M engine) |
| `mamba_engine.py` | Core: `RecursiveMamba2_PrefixScratchpad` (2.7B engine) |
| `train_130m.py` | Automated 3-phase pipeline for 130M |
| `temporal_ablation.py` | 3-arm ablation study script |
| `phase4_engram_integration.py` | Engram contextual gate training |
| `phase5_rlf_recovery.py` | RLF recovery after Engram regression |
| `quick_test.py` | Smoke test for inference |

---

## How It Works

1. Input is prefixed with **8 learnable scratchpad tokens**
2. Full sequence passes through frozen Mamba backbone
3. LoRA adapters modify upper layer computations
4. **Loop engine** replays the sequence with RoPE loop position encoding
5. At each loop, `lm_head` reads the answer-position logit
6. **Latent bridge** translates loop output back to base token distribution
7. Loops repeat up to `max_loops` times or until `<HALT>` is predicted
8. Memory is **O(1)** — no KV-cache, prefix scratchpad is a fixed parameter

---

## Credits

- **[Djiby Diop](https://github.com/Djiby-diop)** — Original bare-metal LLM inference runtime, DjiBLAS math library, GGUF loader, BPE tokenizer, ion-engine architecture, and UEFI boot system.
- **[State Spaces / Mamba](https://github.com/state-spaces/mamba)** — Albert Gu and Tri Dao's Mamba and Mamba-2 architectures.
- **Recursive Latent Forcing (RLF)** — Recursive reasoning loop concept adapted for SSM architectures.

## License

See upstream repositories for license details.
