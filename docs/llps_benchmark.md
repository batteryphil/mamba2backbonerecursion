# Phase 3: LLPS Benchmark Results

## Environment

- **Device**: CPU (no GPU available, no mamba-ssm CUDA kernels)
- **Runs**: 3 (2.8B), 10 (130M)
- **Loops per run**: 7
- **Prompt**: `[LOGIC] X=5. Y=X*2. Z=Y+3. W=Z-X. Output W. ====` (31 tokens)

## Results: Mamba-2.8B (2560 hidden, 64 layers)

| Approach | Avg Loop ms | Median ms | p95 ms | LLPS | Notes |
|----------|------------|-----------|--------|------|-------|
| Original (re-tokenize) | 1100.71 | 1093.16 | 1187.93 | 0.9 | Sequence grows each loop |
| Stateful cache | 468.79 | 452.40 | 534.02 | 2.1 | Single-token recurrent step |

**Speedup: 2.35x** on CPU with 2.8B model

## Results: Mamba-130M (768 hidden, 24 layers)

| Approach | Avg Loop ms | Median ms | p95 ms | LLPS | Notes |
|----------|------------|-----------|--------|------|-------|
| Original (re-tokenize) | 103.43 | 101.81 | 109.39 | 9.7 | Sequence grows each loop |
| Stateful cache | 32.64 | 31.03 | 37.42 | 30.6 | Single-token recurrent step |

**Speedup: 3.17x** on CPU with 130M model

## Detailed Statistics (2.8B)

| Metric | Original (ms) | Stateful (ms) | Speedup |
|--------|--------------|---------------|---------|
| avg | 1100.71 | 468.79 | 2.35x |
| median | 1093.16 | 452.40 | 2.42x |
| p95 | 1187.93 | 534.02 | 2.22x |
| min | 1021.77 | 436.85 | 2.34x |
| max | 1195.47 | 542.18 | 2.20x |
| stdev | 43.98 | 35.00 | 1.26x |
| n (samples) | 21 | 21 | -- |

## Analysis

The stateful approach processes a single token per iteration (O(1) per step), while
the original re-tokenizes the entire prompt + spacers each loop (O(n) per step where
n grows).

### Why CPU speedup is 2-3x (not higher)

On CPU without CUDA kernels, the bottleneck is matrix multiplication, which scales
with model size regardless of sequence length. The re-tokenization overhead is
relatively small compared to per-layer matmul compute.

### Expected GPU speedup

On GPU with CUDA kernels (`mamba-ssm`, `causal-conv1d`), the speedup should be
significantly larger because:

1. The CUDA selective scan kernel optimizes the `seq_len=1` decode case
2. Re-tokenization cost matters more when per-token compute is fast
3. The conv1d decode path (rolling window + dot product) is much cheaper than
   full 1D convolution in prefill mode
4. Total cost: original is O(loops^2), stateful is O(loops)

Expected GPU speedup: **5-15x** for 7 loops, growing with loop count.
