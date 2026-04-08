# Contribution: True O(1) Stateful Loop Engine

## Architectural Change

The existing latent loop implementation rebuilds SSM state from scratch every iteration:

```python
# Original — O(n) per loop, n grows each step
for lp in range(MAX_LOOPS):
    toks = tok(prompt + "=" * lp, ...)       # re-tokenize expanding string
    h = model(**toks, ...).hidden_states[-1]  # full forward pass on entire sequence
```

This is functionally equivalent to pause tokens with a growing prompt. The SSM state
is rebuilt from scratch each time — no recurrent state is carried forward.

The new `stateful_engine.py` uses MambaCache for true O(1) recurrent iteration:

```python
# Stateful — O(1) per loop, constant regardless of history
out = model(input_ids=prompt_ids, use_cache=True, ...)  # prefill once
cache = out.cache_params

for lp in range(MAX_LOOPS):
    step = model(input_ids=spacer, cache_params=cache,   # single-token step
                 cache_position=pos, use_cache=True, ...)
    h = step.hidden_states[-1][0, -1, :]                 # read from cache
```

Each loop is a single-token recurrent step. Sequence length never grows.
Memory usage is constant. This is the correct way to use an SSM recurrently.

## Key API Finding

The plan assumed the standard `past_key_values` transformer API. Mamba uses a
different interface:

- `cache_params` (not `past_key_values`) — passes MambaCache to model
- `cache_position` — **required** when passing cache manually; shape determines
  prefill (shape=conv_kernel) vs decode (shape=1) mode
- Cache is updated **in-place** — same object, mutated

See `docs/cache_api_findings.md` for full details.

## Results

### Mamba-2.8B (CPU, 64 layers, 2560 hidden)

| Approach | Avg Loop ms | LLPS | Speedup |
|----------|-------------|------|---------|
| Original (re-tokenize) | 1100.71 | 0.9 | — |
| Stateful cache | 468.79 | 2.1 | **2.35x** |

### Mamba-130M (CPU, 24 layers, 768 hidden)

| Approach | Avg Loop ms | LLPS | Speedup |
|----------|-------------|------|---------|
| Original (re-tokenize) | 103.43 | 9.7 | — |
| Stateful cache | 32.64 | 30.6 | **3.17x** |

### Correctness

- **Prefill match**: Loop 0 hidden states identical (cosine sim = 1.0000)
- **Generate from cache**: Works without fallback (no kill switch triggered)
- **ACT proportionality**: Hard prompt h-delta (50.6) > Easy (19.2) — 2.6x ratio

On GPU with CUDA kernels, the speedup should be significantly higher because
the original approach's total cost is O(loops^2) while the stateful approach
is O(loops). The CUDA selective scan kernel also specifically optimizes the
`seq_len=1` decode path.

## Files Changed

| File | Status | Description |
|------|--------|-------------|
| `stateful_engine.py` | **NEW** | O(1) StatefulLoopEngine implementation |
| `validate_stateful.py` | **NEW** | Phase 2 correctness validation script |
| `benchmark_llps.py` | **NEW** | Phase 3 LLPS benchmark script |
| `session_memory.py` | **MODIFIED** | `latent_turn()` upgraded to O(1) cache iteration |
| `mamba_engine.py` | **UNCHANGED** | Original training engine preserved |
| `docs/cache_api_findings.md` | **NEW** | Phase 0 MambaCache API documentation |
| `docs/correctness_validation.md` | **NEW** | Phase 2 comparison results |
| `docs/llps_benchmark.md` | **NEW** | Phase 3 latency measurements |
| `docs/blockers.md` | **NEW** | Kill switch status and environment blockers |

## What Remains

### Requires Fine-Tuned Checkpoint + GPU

- [ ] Proof 3 validation: variable tracking W=8
- [ ] ACT proportionality with HaltingHead loop counts
- [ ] Kill-shot ablation (full run vs 2-loop lobotomy)
- [ ] GPU LLPS benchmark with 2.8B model (expected >10x speedup)

### Future Work (from plan)

- [ ] **NPU HaltingHead dispatch**: Move halting decision to NPU for latency hiding
- [ ] **State delta encoding**: Save only the delta between cache states for more
  compact session cartridges (currently ~5MB full cache, could be <1KB deltas)
- [ ] **Batched iteration**: Process multiple conversations simultaneously using
  `max_batch_size > 1` in MambaCache
- [ ] **torch.compile integration**: MambaCache marks tensors as static addresses;
  should be compatible with `torch.compile` for additional speedup
