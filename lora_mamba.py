"""
lora_mamba.py - Post-backbone LoRA adapter for Mamba SSM models.
Ported from batteryphil/mamba-2.8b-latent for mamba-1.4b (d_model=2048).

Design rationale
----------------
The Mamba backbone uses fused Triton kernels for the selective scan.
These are non-differentiable through standard PyTorch autograd.

PostBackboneLoRA bypasses this entirely: it inserts its residual connection
AFTER the backbone returns hidden states — the adapter never touches the SSM
kernels, making all adapter weights fully trainable with zero kernel modifications.
"""

import torch
import torch.nn as nn
from typing import Optional


class LoRALayer(nn.Module):
    """Single rank-decomposed residual adapter layer.

    Implements h = h + (alpha/rank) * B(A(h)) where A and B are low-rank
    projection matrices. Initialized so B=0 -> zero contribution at init.
    """

    def __init__(self, d_model: int, rank: int = 16, alpha: float = 32.0) -> None:
        """Initialise a single LoRA layer."""
        super().__init__()
        self.rank    = rank
        self.scaling = alpha / rank
        self.lora_A  = nn.Linear(d_model, rank,    bias=False)
        self.lora_B  = nn.Linear(rank,    d_model, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)  # zero-init -> no effect at start

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply LoRA residual: h + scaling * B(A(h))."""
        return h + self.scaling * self.lora_B(self.lora_A(h))


class PostBackboneLoRA(nn.Module):
    """Stack of LoRA layers applied after the Mamba backbone.

    The adapter operates on the full [B, L, D] hidden state tensor returned
    by model.backbone(). Multiple stacked layers allow deeper adaptation
    without increasing rank.
    """

    def __init__(self, d_model: int = 2048, rank: int = 16,
                 alpha: float = 32.0, n_layers: int = 6) -> None:
        """Initialise the post-backbone LoRA adapter stack."""
        super().__init__()
        self.layers = nn.ModuleList(
            [LoRALayer(d_model, rank, alpha) for _ in range(n_layers)]
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply all LoRA layers sequentially to hidden state h."""
        for layer in self.layers:
            h = layer(h)
        return h


def load_post_lora(adapter: PostBackboneLoRA, path: str,
                   device: str = "cuda",
                   dtype: Optional[torch.dtype] = None) -> None:
    """Load saved adapter weights into a PostBackboneLoRA instance."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    sd   = ckpt.get("state_dict", ckpt)
    if dtype is not None:
        sd = {k: v.to(dtype=dtype) for k, v in sd.items()}
    adapter.load_state_dict(sd)
    adapter.to(device)
