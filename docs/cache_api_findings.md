# Phase 0: MambaCache API Findings

## Environment

- **transformers version**: 5.3.0
- **MambaCache location**: `transformers.models.mamba.modeling_mamba` (NOT `transformers.cache_utils`)
- **Import**: `from transformers import MambaCache` (auto-imported from model module)
- **Python**: 3.14
- **GPU**: Not available at inspection time (CPU-only system)

## MambaCache Structure

```python
class MambaCache:
    conv_states: list[torch.Tensor]   # [num_layers] x [batch, intermediate_size, conv_kernel_size]
    ssm_states:  list[torch.Tensor]   # [num_layers] x [batch, intermediate_size, ssm_state_size]
```

- One `conv_state` and one `ssm_state` per layer
- Tensors are pre-allocated at cache creation (not grown dynamically)
- Updated **in-place** via `update_conv_state()` and `update_ssm_state()` — the cache
  object returned from the forward pass is the **same object**, mutated
- `reset()` zeros all states in-place (preserves static addresses for torch.compile)

### Constructor

```python
MambaCache(
    config: PreTrainedConfig,
    max_batch_size: int,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str | None = None
)
```

Reads `config.intermediate_size`, `config.state_size`, `config.conv_kernel`,
`config.num_hidden_layers` from the model config.

## Critical API Difference from Plan

The plan assumes the standard `past_key_values` API. Mamba uses a **different interface**:

| Plan assumed | Actual Mamba API |
|---|---|
| `past_key_values=cache` | `cache_params=cache` |
| `out.past_key_values` | `out.cache_params` |
| No position tracking | `cache_position` is **required** |

## Forward Method Signature

```python
MambaForCausalLM.forward(
    input_ids=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_params=None,         # MambaCache instance
    labels=None,
    output_hidden_states=None,
    return_dict=None,
    use_cache=None,
    cache_position=None,       # REQUIRED when cache_params is provided
    logits_to_keep=0,
)
```

## cache_position Semantics

`cache_position` controls whether the model is in **prefill** or **decode** mode.
The discriminator is `cache_position.shape[0]`:

| Mode | cache_position | What happens |
|---|---|---|
| Prefill | `torch.arange(0, config.conv_kernel)` (shape = conv_kernel, typically 4) | Full sequence processed; conv state initialized via padding + conv1d |
| Decode | `torch.tensor([position])` (shape = 1) | Single token step; conv state updated via rolling window + sum |

This is checked in `MambaMixer.slow_forward`:
```python
if cache_position.shape[0] == self.conv_kernel_size:
    # prefill path
else:
    # decode path (single-token recurrent step)
```

**The position value in decode mode doesn't matter for correctness** — `cache_position`
is clamped to `[0, conv_kernel_size-1]` in `update_conv_state`. Any value >= 1 triggers
decode mode because `shape[0] == 1 != conv_kernel_size`.

## Automatic Cache Creation

When `use_cache=True` and `cache_params=None`:
- Model creates a fresh `MambaCache` internally
- Sets `cache_position = torch.arange(0, config.conv_kernel)`
- This is the prefill path — processes full input_ids in one pass

When `use_cache=True` and `cache_params` is provided but `cache_position` is None:
- **Raises ValueError** — you must provide cache_position for manual forward calls

## Generate Integration

`MambaForCausalLM.prepare_inputs_for_generation` accepts `cache_params`:
```python
gen_out = model.generate(
    input_ids=...,
    cache_params=cache,      # pre-built cache
    max_new_tokens=...,
    use_cache=True,
    ...
)
```

When `cache_params` is provided to `generate()`, it does NOT create a new cache.
The generate method internally manages `cache_position` for subsequent tokens.

## Output Format

```python
out = model(input_ids=..., use_cache=True, output_hidden_states=True)
out.cache_params          # MambaCache (same object, mutated in-place)
out.hidden_states         # tuple of tensors, one per layer + final norm
out.hidden_states[-1]     # [batch, seq_len, hidden_size] — final layer output after norm
out.last_hidden_state     # same as hidden_states[-1] when output_hidden_states=True
```

## Correct O(1) Iteration Pattern

```python
device = model.device

# 1. Prefill — build SSM state from prompt
out = model(
    input_ids=prompt_ids,
    use_cache=True,               # creates cache, uses prefill path
    output_hidden_states=True
)
cache = out.cache_params
seq_len = prompt_ids.shape[1]

# 2. Iterate — single-token recurrent steps, O(1) each
spacer = torch.tensor([[spacer_id]], device=device)
for step in range(max_loops):
    step_out = model(
        input_ids=spacer,
        cache_params=cache,
        cache_position=torch.tensor([seq_len + step], device=device),
        use_cache=True,
        output_hidden_states=True
    )
    # cache is mutated in-place, step_out.cache_params is the same object
    h = step_out.hidden_states[-1][0, -1, :].float()
    # ... halting check with h ...

# 3. Generate from accumulated state
gen_ids = model.generate(
    input_ids=spacer,             # minimal input; cache has full context
    cache_params=cache,
    cache_position=torch.tensor([seq_len + max_loops], device=device),
    max_new_tokens=100,
    use_cache=True,
    do_sample=False,
)
```

## Session Serialization

`MambaCache` exposes `conv_states` and `ssm_states` as public list attributes.
The existing `session_memory.py` already serializes these correctly:

```python
state = {
    "conv_states": [s.cpu() for s in cache.conv_states],
    "ssm_states":  [s.cpu() for s in cache.ssm_states],
}
```

Reconstruction requires creating a fresh `MambaCache(config, ...)` and copying
tensors back into each slot.

## Blockers

- **No GPU available** on inspection machine — all API findings from source reading
- **No checkpoint** at `checkpoints/mamba-2.8b-latent/` — base model
  `state-spaces/mamba-2.8b-hf` can be used for structural testing
