# Mamba2 Recursive Latent Forcing (RLF) — Reasoning Engine

A recursive reasoning architecture grafted onto a frozen **Mamba-2 2.7B** backbone. The model uses Recursive Latent Forcing (RLF) loops with a Prefix Latent Scratchpad and Latent Communication Bridge to perform multi-hop chain reasoning — maintaining **O(1) memory** per reasoning loop with no KV cache accumulation.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Frozen Mamba-2 2.7B Base (2.06B params, bfloat16)      │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Layers 0-47: Pure base (frozen)                  │  │
│  │  Layers 48-63: Base + LoRA adapters (trainable)   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Prefix Latent Scratchpad (8 tokens × 2560)             │
│  ┌─────────────────────────────────────┐                │
│  │  Persistent working memory across   │                │
│  │  recursive loops (nn.Parameter)     │                │
│  └─────────────────────────────────────┘                │
│                                                         │
│  Recursive Loop Engine (Mamba2 core, 19.9M params)      │
│  ┌─────────────────────────────────────┐                │
│  │  RoPE loop encoding (loop position) │                │
│  │  Lifeline gate (residual bypass)    │                │
│  │  Loop norm (LayerNorm)              │                │
│  └─────────────────────────────────────┘                │
│                                                         │
│  Latent Bridge (2560 → 64 → 2560, low-rank)            │
│  ┌─────────────────────────────────────┐                │
│  │  Translates loop output back to     │                │
│  │  base model's semantic distribution │                │
│  └─────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────┘

Total trainable: 21.6M params (1.0% of model)
Total frozen:    2.06B params (99.0% of model)
```

## Key Results

| Metric | Value |
|--------|-------|
| Phase 1 (warmup) | 95.1% accuracy, 1000 steps |
| Phase 2 (joint) | 93.6% val accuracy, 3000 steps |
| Phase 3 (adversarial) | Training in progress |
| VRAM usage | 5.5-5.7 GB (fits 12GB GPU) |
| O(1) memory | ✅ Verified: +0.0000 GB across 6 loops |
| Base model preserved | ✅ Frozen backbone untouched |

## Requirements

- **GPU**: 12GB+ VRAM (tested on RTX 3060 12GB)
- **Python**: 3.10+
- **CUDA**: 11.8+
- **Base model**: `state-spaces/mamba2-2.7b` (auto-downloaded from HuggingFace)

## Installation

```bash
git clone https://github.com/batteryphil/mamba2backbonerecursion.git
cd mamba2backbonerecursion

pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

> **Note**: `mamba-ssm` requires `causal-conv1d` and `triton`. Install in order:
> ```bash
> pip install causal-conv1d>=1.2.0
> pip install mamba-ssm>=2.0.0
> ```

## Training Pipeline (3 Phases)

Training MUST be run in sequence — each phase loads the previous checkpoint.

### Phase 1 — Scratchpad & Bridge Warmup

Initializes the prefix scratchpad and latent bridge while keeping everything else frozen.

```bash
python phase1_warmup.py
```

- **Duration**: ~2 hours on 12GB GPU
- **Trainable**: 348K params (memory + bridge only)
- **Output**: `mamba2_2.7b_phase1_scratchpad.pt`
- **Target**: >90% accuracy on 1-11 hop chains

### Phase 2 — Joint Training with LoRA

Unfreezes LoRA adapters and the loop engine for end-to-end reasoning optimization.

```bash
python phase2_joint_training.py
```

- **Duration**: ~14 hours on 12GB GPU
- **Trainable**: 21.6M params (LoRA + loop + memory + bridge)
- **Requires**: Phase 1 checkpoint + `saved_weights/mamba2_2.7b_rlf_rope_best_step3000_val97.3.pt`
- **Output**: `mamba2_2.7b_phase2_joint_best.pt`
- **Note**: Uses accidental SGDR warm restarts (T_max mismatch creates 16 LR cycles)

### Phase 3 — Adversarial Generalization

Forces the model to generalize beyond clean `A=B` syntax using 3 adversarial formats.

```bash
# Generate adversarial curriculum first
python generate_phase3_data.py

# Then train
python phase3_adversarial_training.py
```

- **Duration**: ~20 hours on 12GB GPU (SEQ_LEN=512)
- **Trainable**: 21.6M params (same as Phase 2)
- **Requires**: Phase 2 best checkpoint
- **Output**: `mamba2_2.7b_phase3_adversarial_best.pt`

#### Adversarial Formats

| Format | Example | Purpose |
|--------|---------|---------|
| Variable Chaos | `Var_42 is set to omega. Var_77 <- Var_42.` | Break single-letter dependency |
| Semantic Prose | `Alice whispers it to Bob. Bob texts it to Charlie.` | Force prose-to-logic translation |
| Distractor Chain | `A=red. The Eiffel Tower grows... B=A.` | Train noise rejection |

## Evaluation

```bash
# Run the 4 adversarial stress tests
python evaluate_phase2.py

# Comprehensive benchmark
python comprehensive_eval_2.7b.py
```

## File Structure

```
├── mamba_engine.py              # Core: RecursiveMamba2_PrefixScratchpad architecture
├── mamba_block.py               # Mamba block wrapper
├── config.py                    # Model configuration
│
├── phase1_warmup.py             # Phase 1: Scratchpad + bridge warmup
├── phase2_joint_training.py     # Phase 2: Joint LoRA + loop training
├── phase3_adversarial_training.py  # Phase 3: Adversarial generalization
│
├── generate_phase3_data.py      # Phase 3 curriculum generator
├── data_builder_v2.py           # Chain reasoning data builder
├── data_pipeline.py             # Data loading utilities
├── clean_training_data.py       # Data cleaning
├── inject_curriculum.py         # Curriculum injection
│
├── evaluate_phase2.py           # 4-test adversarial evaluation
├── comprehensive_eval.py        # Full benchmark (130M)
├── comprehensive_eval_2.7b.py   # Full benchmark (2.7B)
├── full_inference_test.py       # Inference testing
├── quick_test.py                # Quick smoke test
│
├── system2_logic_v2_clean.json  # Curriculum data (1-11 hops)
├── system2_logic_v3_curriculum.json  # Extended curriculum
├── phase3_adversarial_curriculum.json  # Adversarial data (5-20 hops)
│
├── finetune_mamba2_130m_v34.py  # Earlier 130M experiments
├── finetune_mamba2_2.7b.py      # 2.7B fine-tuning base
├── distill_mamba2.py            # Distillation utilities
├── launch_phase2.sh             # Auto-launcher for Phase 2
└── requirements.txt             # Python dependencies
```

## How It Works

1. **Input** is tokenized and prefixed with 8 learnable scratchpad tokens
2. The extended sequence passes through the frozen Mamba-2 backbone
3. At layers 48-63, LoRA adapters modify the computation
4. The **loop engine** (a small Mamba2 core) processes the output with RoPE loop encoding
5. The **lifeline gate** controls how much loop output vs. base output to use
6. The **latent bridge** translates loop algebra back to the base model's token distribution
7. Steps 2-6 repeat for up to 6 loops, or until the model emits `<HALT>`
8. Memory is O(1) per loop — no KV cache, no activation accumulation

## Credits & Acknowledgments

- **[Djiby Diop](https://github.com/Djiby-diop)** — Original creator of the bare-metal LLM inference runtime ([llm-baremetal](https://github.com/Djiby-diop/llm-baremetal)). The C inference engine, DjiBLAS math library, GGUF loader, BPE tokenizer, ion-engine architecture, and UEFI boot system are his original work. This project builds on his runtime to deploy the trained reasoning model on bare-metal hardware.

- **[State Spaces / Mamba](https://github.com/state-spaces/mamba)** — Albert Gu and Tri Dao's Mamba and Mamba-2 architectures, which serve as the frozen backbone.

- **Recursive Latent Forcing (RLF)** — The recursive reasoning loop concept, adapted for SSM architectures.

## License

See upstream repositories for license details.
