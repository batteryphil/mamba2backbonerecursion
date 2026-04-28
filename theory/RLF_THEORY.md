# Recursive Latent Forcing (RLF) — Theory & Architecture

## Overview

Recursive Latent Forcing (RLF) is an architectural technique that enables
multi-hop latent reasoning on State Space Models (SSMs) without attention mechanisms.

**Original insight credit: ItsMick** — foundational discovery that Mamba natively
handles O(1) loop state over sequence time, enabling explicit multi-hop computation
without KV-cache growth.

---

## The Core Problem: SSM State Decay

Standard Mamba (and all SSMs) maintain a fixed-size hidden state vector `h ∈ R^d`.
At each sequence position, the SSM update is:

```
h_t = A·h_{t-1} + B·x_t
y_t = C·h_t + D·x_t
```

Where `A` is a diagonal decay matrix with values in (0,1). This means early context
is **exponentially attenuated** over long sequences. For multi-step reasoning:

```
Prompt: "apples=6, price=0.50... [200 tokens later] ...What is cost?"
         ↑ stored in h_0                               ↑ h_200 ≈ A^200 · h_0 ≈ 0
```

The values `apples=6` and `price=0.50` have effectively **decayed to zero** by the
time the model generates the answer.

**SFT alone cannot fix this.** You can train on 1 million word problems but the
architecture physically cannot hold early variables across long contexts.

---

## The RLF Solution: Prefix Scratchpad + Iterative Re-injection

### 1. Prefix Latent Memory

M=8 learnable virtual tokens are prepended to the sequence before each reasoning loop:

```python
mem_state = self.latent_memory.expand(B, -1, -1)  # [B, M, D]
x_extended = torch.cat([mem_state, x], dim=1)      # [B, M+T, D]
```

These tokens act as **continuous scratch paper** — they are not tied to any input
token and can hold arbitrary learned representations. They evolve freely through
each loop while the prompt positions receive lifeline re-injection.

**Key constraint:** Prefix positions 0..M-1 are never overwritten by the lifeline.
Only positions M.. (the actual prompt) receive re-injection. This separates
"working memory" (prefix) from "reference memory" (lifeline).

### 2. Lifeline Re-injection

At every loop iteration, the original prompt encoding `x_prompt` is re-injected into
the prompt positions of the extended sequence:

```python
def _lifeline_inject(x_ext, x_prompt):
    gate = self.lifeline_gate  # learned scalar gate per dimension
    prefix = x_ext[:, :M, :]   # untouched — free scratch space
    prompt_part = x_ext[:, M:, :]
    injected = prompt_part + gate * x_prompt  # re-anchor to original context
    return torch.cat([prefix, injected], dim=1)
```

This is the key that prevents variables from decaying: the original values are
**re-injected at every loop**, so no matter how many iterations run, the model
always has access to `apples=6, price=0.50` at full strength.

### 3. LoopRoPE — Geometric Loop Distinction

Without position encoding on the loop index, all 6 iterations of the loop would
apply the same transformation and the model would learn a fixed point (loop collapse).

LoopRoPE encodes the loop iteration number as a 1D rotary position embedding:

```python
class LoopRoPE(nn.Module):
    def forward(self, x, loop_idx):
        cos, sin = self._sincos(loop_idx, x.device, x.dtype)
        return x * cos + self._rot_half(x) * sin
```

Loop 1, Loop 2, ..., Loop 6 each occupy **geometrically distinct subspaces** in the
hidden state manifold. The model learns different operations per loop index, enabling
true iterative computation rather than repeated identical projections.

### 4. Dedicated Loop Engine

A small auxiliary Mamba SSM (`mamba1_loop`) processes the extended sequence
`[mem | prompt]` at each iteration:

```python
x_ext = x_ext + self.mamba1_loop(x_ext)  # residual: preserves info
x_ext = self.loop_norm(x_ext)
```

This is separate from the backbone — it's a lightweight SSM (~13M params) specialized
for iterative state computation. Initialized with `out_proj.weight = 0` so it starts
as identity and learns incrementally.

### 5. Latent Bridge (System 2 → System 1 Translation)

The loop engine operates in a different activation distribution than the backbone.
A low-rank bottleneck translates between them:

```python
# d_model → 64 → d_model (near-identity init)
x_bridged = x + bridge_up(bridge_down(x))
```

`bridge_up` initialized to zeros → output starts as `x + 0 = x` (identity).
As training progresses, the bridge learns to translate loop state into vocabulary
predictions that the LM head can interpret correctly.

---

## Training Protocol (3 Phases)

### Phase 3a — Scratchpad Warmup (2000 steps)

**Freeze:** Everything except `latent_memory` + `bridge_down/up`  
**Goal:** Initialize the prefix memory tokens into a useful activation range.

Zero-initialized memory creates dead gradient paths. Phase 3a uses small-normal
init (σ=0.02) and trains bridge to connect loop output to vocabulary space before
any reasoning weights are touched.

### Phase 3b — RLF Joint Training (8000 steps)

**Unfreeze:** Top N/2 LoRA layers + loop engine + lifeline gate + memory + bridge  
**Data:** Adversarial chain curriculum (variable pointer chains, math chains, sequences)

Chain target format:
```
Prompt: "A=42. B=A. C=B. What is C?"
Targets: [token("42"), token("42"), token("42"), HALT_ID]
```

Each loop predicts one chain step. Loss computed at `ans_start` position each loop.
Adversarial mode adds distractor facts + prose to test variable isolation.

### Phase 3c — SFT Recovery (1000 steps)

**Freeze:** All RLF components  
**Unfreeze:** LM head only  
**Data:** Original SFT dataset (code + math)

Prevents catastrophic forgetting of code generation ability from prior SFT rounds.
Runs 1 loop (loop_i=0) with causal LM loss on next-token prediction.

---

## Inference

At inference time, the model generates **one reasoning chain token per forward pass**:

```
Input: "V1=42. V2=V1. V3=V2. What is V3?"
Loop 1: → "42"   (resolves V1=42)
Loop 2: → "42"   (resolves V2=V1=42)
Loop 3: → "42"   (resolves V3=V2=42)
Loop 4: → "§"    (HALT — done)
```

MAX_LOOPS=6 hard cap prevents infinite generation. HALT token `§` (token 7803 in
GPT-NeoX vocab) signals termination without embedding resize.

---

## Why O(1) Memory

Unlike transformer KV-cache (which grows linearly with sequence length), the RLF
prefix scratchpad is fixed size:

- Prefix memory: `1 × M × D = 1 × 8 × 2048 = 16,384 parameters`
- This is constant regardless of input length
- Total VRAM during RLF inference: **~2.9GB** for the 1.4B model

A transformer reasoning model at the same quality level would require KV-cache
proportional to context length — 10-50x more memory for long reasoning chains.

---

## Comparison with Prior Work

| Approach | Memory Model | Multi-hop | Baremetal |
|---|---|---|---|
| Standard Mamba (base) | O(1) SSM state | ❌ Decays | ✅ |
| Mamba + SFT spacers (`====`) | O(1) + delimiter | ~20% retention | ✅ |
| Neural Turing Machine | External memory tape | ✅ | ❌ |
| Differentiable Neural Computer | External memory + heads | ✅ | ❌ |
| Transformer (KV-cache) | O(n) attention | ✅ | ❌ |
| **RLF (this work)** | **O(1) prefix + lifeline** | **✅ Full retention** | **✅** |

RLF is the only approach that achieves full variable retention with O(1) memory
and maintains baremetal deployability.

---

## Known Failure Modes

1. **Loop collapse** — mitigated by LoopRoPE. If LoopRoPE is removed, all loops
   converge to the same output within 200 training steps.

2. **Dead scratchpad** — if `latent_memory` is initialized to zeros, Phase 3b
   fails to differentiate prefix from prompt positions. Requires σ=0.02 init.

3. **Bridge saturation** — if `bridge_up` is not initialized to zeros, the bridge
   outputs large values before the loop engine is trained, causing NaN in early steps.

4. **Residual propagation** — Mamba1 layers return `(x, residual)` tuples. The
   residual must be zero-padded and tracked separately through the prefix expansion,
   then applied as `norm_f(x + residual)` after slicing. Dropping residual tracking
   causes incorrect normalization.
