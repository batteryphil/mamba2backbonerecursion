"""
cpu_infer.py — Pure-PyTorch CPU inference for Mamba2-2.7B + RLF
================================================================
Monkey-patches all Triton/CUDA-only ops in mamba_ssm with pure PyTorch
equivalents so the full model runs on CPU.

Patches applied:
  1. Block.fused_add_norm → False (uses torch.nn.LayerNorm path)
  2. Mamba2.use_mem_eff_path → False (skips mamba_split_conv1d_scan_combined)
  3. mamba_chunk_scan_combined → ssd_minimal_discrete (pure PyTorch SSM)
  4. RMSNormGated.forward → rms_norm_ref (pure PyTorch RMSNorm)
  5. causal_conv1d_fn → None (forces nn.Conv1d fallback path)
  6. Block/MixerModel norm_f → torch.nn.RMSNorm (replaces Triton RMSNorm)

Usage:
  python cpu_infer.py --prompt "What is 2 + 2?"
  python cpu_infer.py                              # interactive REPL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
import os
import gc

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Monkey-patch mamba_ssm BEFORE any model imports
# ══════════════════════════════════════════════════════════════════════════════

# --- Patch 1: Pure-PyTorch RMSNorm reference (from mamba_ssm source) --------
def rms_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    z: torch.Tensor | None = None,
    eps: float = 1e-6,
    group_size: int | None = None,
    norm_before_gate: bool = True,
    upcast: bool = True,
) -> torch.Tensor:
    """Pure PyTorch RMSNorm — replaces Triton kernel.

    Supports optional gating with z (SiLU activation).

    Args:
        x: Input tensor
        weight: Norm weight parameter
        bias: Optional bias
        z: Optional gating tensor
        eps: Epsilon for numerical stability
        group_size: Group size for group norm variant
        norm_before_gate: Whether to normalize before or after gating
        upcast: Whether to upcast to float32

    Returns:
        Normalized tensor
    """
    from einops import rearrange

    dtype = x.dtype
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
        out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out *= F.silu(z)
    return out.to(dtype)


# --- Patch 2: Pure-PyTorch SSD scan (from mamba_ssm.modules.ssd_minimal) ----
def ssd_minimal_discrete_cpu(
    X: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    block_len: int,
    initial_states: torch.Tensor | None = None,
) -> tuple:
    """Pure PyTorch selective state-space scan.

    This is the reference implementation from the Mamba2 paper (Listing 1).
    Runs entirely on CPU with no Triton dependency.

    Args:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
        block_len: Chunk size for blocked computation
        initial_states: Optional initial SSM states

    Returns:
        Tuple of (output, final_state)
    """
    from einops import rearrange, repeat

    assert X.dtype == A.dtype == B.dtype == C.dtype
    # Pad sequence length to be divisible by block_len
    orig_len = X.shape[1]
    if orig_len % block_len != 0:
        pad_len = block_len - (orig_len % block_len)
        X = F.pad(X, (0, 0, 0, 0, 0, pad_len))
        A = F.pad(A, (0, 0, 0, pad_len))
        B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
        C = F.pad(C, (0, 0, 0, 0, 0, pad_len))

    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # Segment sum (stable version)
    def segsum(x: torch.Tensor) -> torch.Tensor:
        """Stable segment sum for SSM computation."""
        T = x.size(-1)
        x = repeat(x, "... d -> ... d e", e=T)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
        x = x.masked_fill(~mask, 0)
        x_segsum = torch.cumsum(x, dim=-2)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    # 1. Intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. State for each chunk
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Inter-chunk recurrence
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. State → output per chunk
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    # Trim back to original length
    Y = Y[:, :orig_len]
    return Y, final_state


# --- Patch 3: Replacement for mamba_chunk_scan_combined --------------------
def mamba_chunk_scan_cpu(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    **kwargs,
) -> torch.Tensor:
    """CPU-compatible wrapper for mamba_chunk_scan_combined.

    Converts the Mamba2 calling convention to ssd_minimal_discrete format.

    Args:
        x: (B, L, nheads, headdim)
        dt: (B, L, nheads)
        A: (nheads,) — should be negative
        B: (B, L, ngroups, d_state)
        C: (B, L, ngroups, d_state)
        chunk_size: Block size for chunked computation
        D: Optional skip connection parameter
        z: Optional gating tensor (unused in minimal impl)
        dt_bias: Optional bias added to dt
        dt_softplus: Whether to apply softplus to dt

    Returns:
        Output tensor (B, L, nheads, headdim)
    """
    from einops import repeat

    # Apply dt_bias and softplus
    if dt_bias is not None:
        dt = dt + dt_bias
    if dt_softplus:
        dt = F.softplus(dt)

    # Expand A: (nheads,) → (B, L, nheads)
    A_expanded = repeat(A, "h -> b l h", b=x.shape[0], l=x.shape[1])

    # SSD expects: X = x * dt, A_disc = A * dt
    dtype = x.dtype
    x_f = x.float()
    dt_f = dt.float()
    A_f = A_expanded.float()
    B_f = B.float()
    C_f = C.float()

    # Expand B, C if ngroups < nheads (group query)
    nheads = x.shape[2]
    ngroups = B.shape[2]
    if ngroups < nheads:
        repeats = nheads // ngroups
        B_f = repeat(B_f, "b l g n -> b l (g r) n", r=repeats)
        C_f = repeat(C_f, "b l g n -> b l (g r) n", r=repeats)

    X_disc = x_f * dt_f.unsqueeze(-1)
    A_disc = A_f * dt_f

    Y, _ = ssd_minimal_discrete_cpu(X_disc, A_disc, B_f, C_f, chunk_size)
    Y = Y.to(dtype)

    # D skip connection
    if D is not None:
        if D.dim() == 1:
            # D is (nheads,) — broadcast
            Y = Y + x * D.unsqueeze(-1)
        else:
            # D is (nheads, headdim)
            Y = Y + x * D

    return Y


# --- Apply monkey-patches --------------------------------------------------
print("[CPU] Applying pure-PyTorch monkey-patches...")

# Patch mamba_chunk_scan_combined
import mamba_ssm.ops.triton.ssd_combined as ssd_combined_mod
ssd_combined_mod.mamba_chunk_scan_combined = mamba_chunk_scan_cpu

# Also patch at the import site in mamba2 module
import mamba_ssm.modules.mamba2 as mamba2_mod
mamba2_mod.mamba_chunk_scan_combined = mamba_chunk_scan_cpu
# Force non-fused path
mamba2_mod.causal_conv1d_fn = None
mamba2_mod.causal_conv1d_update = None

# Patch RMSNormGated to use pure PyTorch
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as _TritonRMSNormGated


class RMSNormGatedCPU(nn.Module):
    """CPU-compatible RMSNormGated replacement.

    Uses pure PyTorch rms_norm_ref instead of Triton kernel.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5,
                 norm_before_gate: bool = True, group_size: int | None = None,
                 device=None, dtype=None):
        """Initialize CPU RMSNorm with gating support."""
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass using pure PyTorch RMSNorm."""
        return rms_norm_ref(x, self.weight, self.bias, z=z, eps=self.eps,
                            group_size=self.group_size,
                            norm_before_gate=self.norm_before_gate)

# Patch the RMSNormGated class in the mamba2 module
mamba2_mod.RMSNormGated = RMSNormGatedCPU
import mamba_ssm.ops.triton.layernorm_gated as lng_mod
lng_mod.RMSNorm = RMSNormGatedCPU

# Patch layer_norm_fn and rms_norm_fn used by Block and MixerModel
def layer_norm_fn_cpu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None = None,
    x1: torch.Tensor | None = None,
    weight1: torch.Tensor | None = None,
    bias1: torch.Tensor | None = None,
    eps: float = 1e-6,
    dropout_p: float = 0.0,
    rowscale: torch.Tensor | None = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    is_rms_norm: bool = False,
    return_dropout_mask: bool = False,
) -> torch.Tensor | tuple:
    """Pure PyTorch replacement for Triton layer_norm_fn.

    Full-signature compatible with mamba_ssm's Triton version.

    Args:
        x: Input hidden states
        weight: Norm weight
        bias: Optional norm bias
        residual: Optional residual for fused add+norm
        x1: Optional parallel branch (ignored on CPU)
        weight1: Optional parallel weight (ignored on CPU)
        bias1: Optional parallel bias (ignored on CPU)
        eps: Epsilon for stability
        dropout_p: Dropout probability (ignored on CPU inference)
        rowscale: Per-row scaling (ignored on CPU)
        prenorm: If True, returns (normed, residual) tuple
        residual_in_fp32: Whether to keep residual in fp32
        is_rms_norm: Whether to use RMSNorm
        return_dropout_mask: Whether to return mask (ignored)

    Returns:
        Normalized tensor, or (normalized, residual) if prenorm=True
    """
    # Fused add
    if residual is not None:
        x = x + residual
    residual_out = x
    if residual_in_fp32:
        residual_out = residual_out.float()

    # Normalize
    x_float = x.float()
    if is_rms_norm:
        rstd = 1.0 / torch.sqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
        out = (x_float * rstd * weight.float())
    else:
        mean = x_float.mean(dim=-1, keepdim=True)
        rstd = 1.0 / torch.sqrt((x_float - mean).square().mean(dim=-1, keepdim=True) + eps)
        out = ((x_float - mean) * rstd * weight.float())

    if bias is not None:
        out = out + bias.float()
    out = out.to(x.dtype)

    if prenorm:
        return out, residual_out
    return out


def rms_norm_fn_cpu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None = None,
    x1: torch.Tensor | None = None,
    weight1: torch.Tensor | None = None,
    bias1: torch.Tensor | None = None,
    eps: float = 1e-6,
    dropout_p: float = 0.0,
    rowscale: torch.Tensor | None = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    return_dropout_mask: bool = False,
    # Also accept gated-norm kwargs (used by RMSNormGated)
    z: torch.Tensor | None = None,
    group_size: int | None = None,
    norm_before_gate: bool = True,
) -> torch.Tensor | tuple:
    """Pure PyTorch rms_norm_fn — matches both Triton signatures.

    Handles two calling conventions:
      1. Block/MixerModel style: residual, prenorm, residual_in_fp32
      2. RMSNormGated style: z, group_size, norm_before_gate

    Args:
        x: Input tensor
        weight: Norm weight
        bias: Optional bias
        residual: Optional residual for fused add+norm
        eps: Epsilon for stability
        prenorm: If True, returns (normed, residual) tuple
        residual_in_fp32: Keep residual fp32
        z: Optional gating tensor
        group_size: Group norm size
        norm_before_gate: Gate before or after norm

    Returns:
        Normalized tensor, or (normalized, residual) tuple
    """
    # If z is provided, use the gated variant (RMSNormGated path)
    if z is not None:
        return rms_norm_ref(x, weight, bias, z=z, eps=eps,
                            group_size=group_size,
                            norm_before_gate=norm_before_gate)

    # Otherwise use the residual variant (Block/MixerModel path)
    return layer_norm_fn_cpu(
        x, weight, bias, residual=residual,
        eps=eps, prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
        is_rms_norm=True,
    )


import mamba_ssm.ops.triton.layer_norm as ln_mod
ln_mod.layer_norm_fn = layer_norm_fn_cpu
ln_mod.rms_norm_fn = rms_norm_fn_cpu

# Also monkey-patch the RMSNorm CLASS in layer_norm.py (used by Block.norm)
_OrigTritonRMSNorm = ln_mod.RMSNorm


class RMSNormCPU(nn.Module):
    """CPU-compatible replacement for mamba_ssm's Triton RMSNorm.

    The Triton version's forward() accepts: x, residual, prenorm, residual_in_fp32.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5,
                 dropout_p: float = 0.0, device=None, dtype=None):
        """Initialize with same API as Triton RMSNorm."""
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.drop = nn.Dropout(dropout_p) if dropout_p > 0.0 else None
        self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None,
                prenorm: bool = False,
                residual_in_fp32: bool = False) -> torch.Tensor | tuple:
        """Forward with residual support — matches Triton RMSNorm API."""
        return rms_norm_fn_cpu(
            x, self.weight, self.bias,
            residual=residual, eps=self.eps,
            dropout_p=self.drop.p if self.drop is not None and self.training else 0.0,
            prenorm=prenorm, residual_in_fp32=residual_in_fp32,
        )


ln_mod.RMSNorm = RMSNormCPU

# Patch at Block import site
import mamba_ssm.modules.block as block_mod
block_mod.layer_norm_fn = layer_norm_fn_cpu
block_mod.RMSNorm = RMSNormCPU

# Patch at MixerModel import site
import mamba_ssm.models.mixer_seq_simple as mixer_mod
mixer_mod.layer_norm_fn = layer_norm_fn_cpu
mixer_mod.rms_norm_fn = rms_norm_fn_cpu
mixer_mod.RMSNorm = RMSNormCPU

print("[CPU] All patches applied ✓")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Now import model code (after patches are in place)
# ══════════════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mamba_engine import (
    RecursiveMamba2_PrefixScratchpad,
    fuse_lora_weights,
    tokenizer,
    HALT_ID,
    MODEL_ID,
)


CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mamba2_2.7b_phase2_joint_best.pt",
)
DEVICE = "cpu"
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.9


def patch_model_for_cpu(model: nn.Module) -> None:
    """Disable fused_add_norm and use_mem_eff_path on all submodules.

    Also replaces Triton-only RMSNorm/RMSNormGated with CPU-compatible versions.

    Args:
        model: The loaded model to patch
    """
    for module in model.modules():
        if hasattr(module, "fused_add_norm"):
            module.fused_add_norm = False
        if hasattr(module, "use_mem_eff_path"):
            module.use_mem_eff_path = False

    # Replace Triton RMSNorm and RMSNormGated instances with CPU versions
    replacements = []
    for name, module in model.named_modules():
        for attr_name, child in module.named_children():
            # Skip if already a CPU replacement or PyTorch built-in
            if isinstance(child, (RMSNormCPU, RMSNormGatedCPU, nn.RMSNorm)):
                continue
            child_module_path = type(child).__module__ or ""
            is_triton_rms = (type(child).__name__ == "RMSNorm"
                             and "triton" in child_module_path
                             and hasattr(child, "weight"))
            is_triton_gated = isinstance(child, _TritonRMSNormGated)
            if is_triton_rms:
                # Block.norm or MixerModel.norm_f — has residual/prenorm signature
                cpu_norm = RMSNormCPU(child.weight.shape[0], eps=child.eps)
                cpu_norm.weight = child.weight
                replacements.append((module, attr_name, cpu_norm))
            elif is_triton_gated and not isinstance(child, RMSNormGatedCPU):
                # Mamba2.norm — has z/gating signature
                cpu_norm = RMSNormGatedCPU(
                    child.weight.shape[0], eps=child.eps,
                    norm_before_gate=getattr(child, "norm_before_gate", True),
                    group_size=getattr(child, "group_size", None),
                )
                cpu_norm.weight = child.weight
                replacements.append((module, attr_name, cpu_norm))

    for parent, attr_name, new_module in replacements:
        setattr(parent, attr_name, new_module)
    if replacements:
        print(f"  Replaced {len(replacements)} Triton norm modules with CPU versions")


def load_model_cpu(checkpoint_path: str) -> RecursiveMamba2_PrefixScratchpad:
    """Load model on CPU with all Triton ops patched out.

    Args:
        checkpoint_path: Path to .pt checkpoint

    Returns:
        Model in eval mode on CPU
    """
    print(f"Loading backbone: {MODEL_ID} (CPU mode)")
    from mamba_ssm import MambaLMHeadModel

    backbone = MambaLMHeadModel.from_pretrained(
        MODEL_ID, dtype=torch.float32, device="cpu"
    )

    # Patch BEFORE building wrapper
    patch_model_for_cpu(backbone)

    # Resize embeddings
    new_vocab = len(tokenizer)
    old_embed = backbone.backbone.embedding
    old_vocab = old_embed.weight.shape[0]
    if new_vocab > old_vocab:
        print(f"  Expanding vocab: {old_vocab} → {new_vocab}")
        new_embed = nn.Embedding(new_vocab, old_embed.embedding_dim)
        new_embed.weight.data[:old_vocab] = old_embed.weight.data.float()
        nn.init.normal_(new_embed.weight.data[old_vocab:], mean=0.0, std=0.02)
        backbone.backbone.embedding = new_embed

        old_head = backbone.lm_head
        if hasattr(old_head, "weight") and old_head.weight.shape[0] == old_vocab:
            new_head = nn.Linear(old_head.in_features, new_vocab, bias=old_head.bias is not None)
            new_head.weight.data[:old_vocab] = old_head.weight.data.float()
            nn.init.zeros_(new_head.weight.data[old_vocab:])
            backbone.lm_head = new_head

    print("Building RLF wrapper...")
    model = RecursiveMamba2_PrefixScratchpad(backbone, lora_rank=4)

    print(f"Loading checkpoint: {os.path.basename(checkpoint_path)}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)

    print("Fusing LoRA weights...")
    fuse_lora_weights(model)

    # Patch the full model again after construction
    patch_model_for_cpu(model)

    # Convert everything to float32 for CPU
    model = model.float()
    model.eval()

    del backbone, ckpt, state
    gc.collect()

    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"RAM usage: ~{param_mb:.0f} MB")
    print()
    return model


def generate_rlf(
    model: RecursiveMamba2_PrefixScratchpad,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    top_p: float = TOP_P,
) -> str:
    """Generate text using the RLF reasoning loop on CPU.

    Args:
        model: Loaded model in eval mode
        prompt: Input text
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus sampling threshold

    Returns:
        Generated text string
    """
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        # RLF reasoning loop
        n_loops, trace, last_answer = model(input_ids)
        print(f"  RLF: {n_loops} loops")
        for step, tok, conf in trace:
            print(f"    {step}: '{tok}' (p={conf})")

        # Autoregressive generation
        generated_ids = input_ids.clone()
        generated_tokens = []

        for i in range(max_new_tokens):
            x = model.backbone.embedding(generated_ids)
            residual = None
            for layer in model.all_layers:
                x, residual = layer(x, residual)

            logits = model.lm_head(model.norm(x, residual, prenorm=False))
            next_logits = logits[0, -1, :].float()

            if temperature > 0:
                next_logits = next_logits / temperature
                if top_k > 0:
                    topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < topk_vals[-1]] = float("-inf")
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cum_probs > top_p
                    remove[1:] = remove[:-1].clone()
                    remove[0] = False
                    sorted_logits[remove] = float("-inf")
                    next_logits = torch.zeros_like(next_logits).scatter(0, sorted_idx, sorted_logits)
                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = next_logits.argmax(dim=-1, keepdim=True)

            token_id = next_id.item()
            if token_id == tokenizer.eos_token_id or token_id == HALT_ID:
                break
            generated_tokens.append(token_id)
            generated_ids = torch.cat([generated_ids, next_id.unsqueeze(0)], dim=1)

            if (i + 1) % 10 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()

    if generated_tokens:
        sys.stdout.write("\n")
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def interactive_repl(model: RecursiveMamba2_PrefixScratchpad) -> None:
    """Interactive REPL for CPU inference.

    Args:
        model: Loaded model in eval mode
    """
    print("═" * 60)
    print("  Mamba2-2.7B + RLF  ·  CPU Inference")
    print("  Type 'quit' or Ctrl+C to exit")
    print("═" * 60)
    print()

    while True:
        try:
            prompt = input(">>> ").strip()
            if not prompt or prompt.lower() in ("quit", "exit", "q"):
                break
            output = generate_rlf(model, prompt)
            print(f"\n{output}\n")
        except KeyboardInterrupt:
            print("\n\nExiting.")
            break
        except Exception as e:
            print(f"\n  Error: {e}\n")
            import traceback
            traceback.print_exc()


def main() -> int:
    """Entry point.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(description="Mamba2-2.7B + RLF CPU Inference")
    parser.add_argument("--prompt", "-p", type=str, default=None)
    parser.add_argument("--max-tokens", "-n", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", "-t", type=float, default=TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--checkpoint", "-c", type=str, default=CHECKPOINT)
    args = parser.parse_args()

    model = load_model_cpu(args.checkpoint)

    if args.prompt:
        output = generate_rlf(model, args.prompt,
                              max_new_tokens=args.max_tokens,
                              temperature=args.temperature,
                              top_k=args.top_k, top_p=args.top_p)
        print(output)
    else:
        interactive_repl(model)

    return 0


if __name__ == "__main__":
    sys.exit(main())
