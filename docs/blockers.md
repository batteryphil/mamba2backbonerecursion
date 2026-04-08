# Blockers

## No GPU Available

- **Phase**: All
- **Impact**: Cannot measure GPU-specific latency; no CUDA fast-path kernels
- **Mitigation**: Both 2.8B and 130M models tested on CPU (slow path). API correctness confirmed, benchmarks captured.
- **Resolution**: Re-run benchmarks on GPU for production-representative numbers

## No Fine-Tuned Checkpoint

- **Phase**: 2, 3
- **Impact**: Cannot validate Proof 3 (W=8), ACT proportionality with HaltingHead
- **Mitigation**: Structural correctness confirmed; checkpoint-dependent tests documented as pending
- **Resolution**: Run `validate_stateful.py` and `benchmark_llps.py` with `checkpoints/mamba-2.8b-latent`

## transformers 5.3.0 API Change

- **Phase**: 0
- **Impact**: `MambaCache` moved from `transformers.cache_utils` to `transformers.models.mamba.modeling_mamba`
- **Resolution**: Import via `from transformers import MambaCache` (auto-import works)
- **Note**: The plan assumed `past_key_values` API; actual Mamba API uses `cache_params` + `cache_position`

## No Kill Switches Triggered

All kill switch conditions in the plan were avoided:
- `use_cache=True` works correctly
- `generate()` accepts `cache_params` with pre-built cache
- MambaCache exposes `conv_states` and `ssm_states` as public attributes
