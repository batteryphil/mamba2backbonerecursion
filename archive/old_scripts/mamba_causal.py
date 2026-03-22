import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Attempt to load optimized kernels
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from causal_conv1d import causal_conv1d_fn
    HAS_OPTIMIZED_KERNELS = True
except ImportError:
    HAS_OPTIMIZED_KERNELS = False

def _ssm_scan_cpu(x, dt, A, B, C, D, z=None):
    """Pure-PyTorch selective-scan recurrence (CPU fallback)."""
    B_sz, L, D_inner = x.shape
    N = A.shape[1]
    dtype = x.dtype
    A, D, dt, B, C = A.to(dtype), D.to(dtype), dt.to(dtype), B.to(dtype), C.to(dtype)
    h = torch.zeros(B_sz, D_inner, N, device=x.device, dtype=dtype)
    ys = []
    for t in range(L):
        dA = torch.exp(dt[:, t, :].unsqueeze(-1) * A.unsqueeze(0))
        dB = dt[:, t, :].unsqueeze(-1) * B[:, t, :].unsqueeze(1)
        h = dA * h + dB * x[:, t, :].unsqueeze(-1)
        y_t = (h * C[:, t, :].unsqueeze(1)).sum(-1)
        y_t = y_t + D.unsqueeze(0) * x[:, t, :]
        ys.append(y_t)
    y = torch.stack(ys, dim=1)
    if z is not None:
        y = y * F.silu(z.to(dtype))
    return y

class CausalMambaBlock(nn.Module):
    """
    Standard Autoregressive (Causal) Mamba Block.
    Refactored from BiMambaBlock for Next-Token Prediction.
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)

        # 🚀 Shared Linear Projections
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # 🚀 Causal SSM Parameters (Unidirectional)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, 
            kernel_size=d_conv, padding=d_conv-1, 
            groups=self.d_inner, bias=True
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        L = u.shape[1]
        xz = self.in_proj(u)
        x_in, z = xz.chunk(2, dim=-1)

        if HAS_OPTIMIZED_KERNELS and u.is_cuda:
            # --- Causal Execution (Fused) ---
            # Transpose for Conv1d: (B, L, D) -> (B, D, L)
            x_c = causal_conv1d_fn(
                x_in.transpose(1, 2), 
                self.conv1d.weight.squeeze(1), 
                self.conv1d.bias, 
                activation="silu"
            )
            
            x_proj = self.x_proj(x_c.transpose(1, 2))
            dt, B, C = torch.split(x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = F.linear(dt, self.dt_proj.weight, self.dt_proj.bias)
            
            y = selective_scan_fn(
                x_c, dt.transpose(1, 2), -torch.exp(self.A_log), 
                B.transpose(1, 2), C.transpose(1, 2), self.D.float(), 
                z.transpose(1, 2), delta_softplus=True
            ).transpose(1, 2)
            
            return self.out_proj(y)
        else:
            # Manual Fallback (Causal)
            x = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
            x = F.silu(x)
            x_p = self.x_proj(x)
            dt, B, C = torch.split(x_p, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = F.softplus(self.dt_proj(dt))
            y = _ssm_scan_cpu(x, dt, -torch.exp(self.A_log), B, C, self.D, z)
            
            return self.out_proj(y)

# Compatibility Alias
MambaBlock = CausalMambaBlock
