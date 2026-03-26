"""Pure PyTorch Mamba SSM block — no CUDA kernels, CPU-friendly.

Implements the S6 selective state-space model from:
    "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

This is a from-scratch implementation that runs on CPU without
the mamba-ssm package or causal-conv1d dependencies.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """Single Mamba SSM block with selective scan.

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension.
        d_conv: Local convolution width.
        expand_factor: Expansion factor for inner dimension.
        dt_rank: Rank for delta projection. "auto" = d_model // 16.
        bias: Use bias in linear projections.
        conv_bias: Use bias in conv1d.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dt_rank: str = "auto",
        bias: bool = False,
        conv_bias: bool = True,
    ) -> None:
        """Initialize MambaBlock."""
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand_factor

        if dt_rank == "auto":
            self.dt_rank = max(d_model // 16, 1)
        else:
            self.dt_rank = int(dt_rank)

        # Input projection: x -> (z, x_proj)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Depthwise conv1d on x branch
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias,
        )

        # SSM parameters projection: x -> (dt, B, C)
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + d_state * 2, bias=False
        )

        # Delta (dt) projection
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # SSM parameters A and D
        # A is initialized as a structured matrix (diagonal)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.expand(self.d_inner, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        emotion_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional emotion conditioning.

        Args:
            x: Input tensor [B, T, d_model].
            emotion_emb: Optional emotion embedding [B, d_model] to add.

        Returns:
            Output tensor [B, T, d_model].
        """
        residual = x
        x = self.norm(x)

        # Add emotion conditioning if provided
        if emotion_emb is not None:
            x = x + emotion_emb.unsqueeze(1)

        # Input projection -> split into x and gate (z)
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)  # [B, T, d_inner] each

        # Depthwise conv on x branch (causal)
        x_conv = x_branch.transpose(1, 2)  # [B, d_inner, T]
        x_conv = self.conv1d(x_conv)[:, :, :x_branch.shape[1]]
        x_conv = x_conv.transpose(1, 2)  # [B, T, d_inner]
        x_branch = F.silu(x_conv)

        # SSM scan
        y = self._ssm_scan(x_branch)

        # Gate and project output
        y = y * F.silu(z)
        output = self.out_proj(y)

        return output + residual

    def _ssm_scan(self, x: torch.Tensor) -> torch.Tensor:
        """Selective scan — the core of Mamba.

        Computes the discretized SSM recurrence:
            h[t] = A_bar * h[t-1] + B_bar * x[t]
            y[t] = C * h[t] + D * x[t]

        Args:
            x: Input tensor [B, T, d_inner].

        Returns:
            Output tensor [B, T, d_inner].
        """
        batch, seq_len, d_inner = x.shape

        # Project x to get dt, B, C (data-dependent = selective)
        x_dbc = self.x_proj(x)  # [B, T, dt_rank + 2*d_state]
        dt, B, C = torch.split(
            x_dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Delta (discretization step size)
        dt = self.dt_proj(dt)  # [B, T, d_inner]
        dt = F.softplus(dt)  # Ensure positive

        # A matrix (negative for stability)
        A = -torch.exp(self.A_log)  # [d_inner, d_state]

        # Discretize: A_bar = exp(dt * A), B_bar = dt * B
        # For efficiency, compute element-wise
        dt_A = torch.einsum("btd,dn->btdn", dt, A)  # [B, T, d_inner, d_state]
        A_bar = torch.exp(dt_A)

        dt_B = torch.einsum("btd,btn->btdn", dt, B)  # [B, T, d_inner, d_state]

        # Sequential scan (CPU-friendly loop)
        h = torch.zeros(
            batch, d_inner, self.d_state,
            device=x.device, dtype=x.dtype
        )
        ys = []

        for t in range(seq_len):
            # h = A_bar * h + B_bar * x
            h = A_bar[:, t] * h + dt_B[:, t] * x[:, t].unsqueeze(-1)
            # y = C * h + D * x
            y_t = torch.einsum("bdn,bn->bd", h, C[:, t])
            y_t = y_t + self.D * x[:, t]
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # [B, T, d_inner]


class MambaStack(nn.Module):
    """Stack of Mamba blocks.

    Args:
        n_layers: Number of Mamba blocks.
        d_model: Model dimension.
        d_state: SSM state dimension.
        d_conv: Convolution width.
        expand_factor: Expansion factor.
    """

    def __init__(
        self,
        n_layers: int = 6,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
    ) -> None:
        """Initialize MambaStack."""
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        emotion_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through all Mamba layers.

        Args:
            x: Input tensor [B, T, d_model].
            emotion_emb: Optional emotion embedding [B, d_model].

        Returns:
            Output tensor [B, T, d_model].
        """
        for layer in self.layers:
            x = layer(x, emotion_emb=emotion_emb)
        return self.final_norm(x)
