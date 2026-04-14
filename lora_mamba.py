"""
lora_mamba.py
=============
Post-backbone LoRA adapter for Mamba SSM models.

Design rationale
----------------
The Mamba backbone uses fused Triton kernels (mamba_ssm.ops.triton) for the
selective scan. These are non-differentiable through standard PyTorch autograd.

PostBackboneLoRA bypasses this entirely: it inserts its residual connection
AFTER the backbone returns hidden states — the adapter never touches the SSM
kernels, making all adapter weights fully trainable with zero kernel modifications.

Usage
-----
    from lora_mamba import PostBackboneLoRA, load_post_lora

    adapter = PostBackboneLoRA(d_model=2560, rank=16, alpha=32.0, n_layers=6)
    adapter = adapter.to(device)
    load_post_lora(adapter, "lora_oo_r16_final.pt", device=device)
    adapter.eval()

    # Inference
    h = model.backbone(input_ids)      # [B, L, 2560]
    h = adapter(h)                     # [B, L, 2560]  ← LoRA correction injected
    logits = model.lm_head(h)
"""

import torch
import torch.nn as nn
from typing import Optional


class LoRALayer(nn.Module):
    """Single rank-decomposed residual adapter layer.

    Implements h = h + (alpha/rank) * B(A(h)) where A and B are low-rank
    projection matrices. Initialized so B=0 → zero contribution at init.
    """

    def __init__(self, d_model: int, rank: int = 16, alpha: float = 32.0):
        """Initialise a single LoRA layer.

        Args:
            d_model: Hidden dimension of the base model.
            rank:    Rank of the low-rank decomposition.
            alpha:   Scaling factor (effective LR multiplier = alpha / rank).
        """
        super().__init__()
        self.rank    = rank
        self.scaling = alpha / rank
        self.lora_A  = nn.Linear(d_model, rank,    bias=False)
        self.lora_B  = nn.Linear(rank,    d_model, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)  # zero-init → no effect at start

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply LoRA residual: h + scaling * B(A(h))."""
        return h + self.scaling * self.lora_B(self.lora_A(h))


class PostBackboneLoRA(nn.Module):
    """Stack of LoRA layers applied after the Mamba backbone.

    The adapter operates on the full [B, L, D] hidden state tensor returned
    by model.backbone(). Multiple stacked layers allow deeper adaptation
    without increasing rank.

    Args:
        d_model:  Hidden dimension (2560 for mamba-2.8b).
        rank:     LoRA rank (default 16).
        alpha:    LoRA alpha scaling (default 32.0).
        n_layers: Number of stacked LoRA layers (default 6).
    """

    def __init__(self, d_model: int = 2560, rank: int = 16,
                 alpha: float = 32.0, n_layers: int = 6):
        """Initialise the post-backbone LoRA adapter stack."""
        super().__init__()
        self.layers = nn.ModuleList(
            [LoRALayer(d_model, rank, alpha) for _ in range(n_layers)]
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply all LoRA layers sequentially to hidden state h.

        Args:
            h: Hidden state tensor [B, L, D]

        Returns:
            Corrected hidden state [B, L, D]
        """
        for layer in self.layers:
            h = layer(h)
        return h


def load_post_lora(adapter: PostBackboneLoRA, path: str,
                   device: str = "cuda",
                   dtype: Optional[torch.dtype] = None) -> None:
    """Load saved adapter weights into a PostBackboneLoRA instance.

    Handles both flat state_dicts and wrapped {'state_dict': ..., 'meta': ...}
    checkpoint formats produced by train_lora_oo_trainer.py.

    Args:
        adapter: PostBackboneLoRA instance to load into.
        path:    Path to the .pt checkpoint file.
        device:  Target device for loaded weights.
        dtype:   Optional dtype to cast weights to (e.g. torch.bfloat16).
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    if dtype is not None:
        sd = {k: v.to(dtype=dtype) for k, v in sd.items()}

    adapter.load_state_dict(sd)
    adapter.to(device)
