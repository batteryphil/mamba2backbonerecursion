import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm

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

class BiMambaBlock(nn.Module):
    """Optimized Bidirectional Mamba block.
    Shares O(N^2) linear projections (in_proj, out_proj) for VRAM efficiency.
    Maintains separate O(N) SSM kernels for forward and backward passes.
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)

        # 🚀 MASSIVE VRAM SAVER: Shared Linear Projections
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # 🚀 ENGINE A: Forward Pass Parameters
        self.fwd_conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv-1, groups=self.d_inner, bias=True)
        self.fwd_x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.fwd_dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.fwd_A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.fwd_D = nn.Parameter(torch.ones(self.d_inner))

        # 🚀 ENGINE B: Backward Pass Parameters
        self.bwd_conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv-1, groups=self.d_inner, bias=True)
        self.bwd_x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.bwd_dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.bwd_A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.bwd_D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        L = u.shape[1]
        xz = self.in_proj(u)
        x_in, z = xz.chunk(2, dim=-1)

        if HAS_OPTIMIZED_KERNELS and u.is_cuda:
            # --- Forward Execution (Fused) ---
            x_fwd_c = causal_conv1d_fn(x_in.transpose(1, 2), self.fwd_conv1d.weight.squeeze(1), self.fwd_conv1d.bias, activation="silu")
            x_proj_fwd = self.fwd_x_proj(x_fwd_c.transpose(1, 2))
            dt_fwd, B_fwd, C_fwd = torch.split(x_proj_fwd, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt_fwd = F.linear(dt_fwd, self.fwd_dt_proj.weight, self.fwd_dt_proj.bias)
            y_fwd = selective_scan_fn(
                x_fwd_c, dt_fwd.transpose(1, 2), -torch.exp(self.fwd_A_log), 
                B_fwd.transpose(1, 2), C_fwd.transpose(1, 2), self.fwd_D.float(), 
                z.transpose(1, 2), delta_softplus=True
            ).transpose(1, 2)

            # --- Backward Execution (Fused) ---
            x_flip, z_flip = x_in.flip([1]), z.flip([1])
            x_bwd_c = causal_conv1d_fn(x_flip.transpose(1, 2), self.bwd_conv1d.weight.squeeze(1), self.bwd_conv1d.bias, activation="silu")
            x_proj_bwd = self.bwd_x_proj(x_bwd_c.transpose(1, 2))
            dt_bwd, B_bwd, C_bwd = torch.split(x_proj_bwd, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt_bwd = F.linear(dt_bwd, self.bwd_dt_proj.weight, self.bwd_dt_proj.bias)
            y_bwd = selective_scan_fn(
                x_bwd_c, dt_bwd.transpose(1, 2), -torch.exp(self.bwd_A_log), 
                B_bwd.transpose(1, 2), C_bwd.transpose(1, 2), self.bwd_D.float(), 
                z_flip.transpose(1, 2), delta_softplus=True
            ).transpose(1, 2).flip([1])
            
            return self.out_proj(y_fwd + y_bwd)
        else:
            # Manual Fallback
            return self._forward_manual(u, x_in, z)

    def _forward_manual(self, u, x_in, z):
        L = u.shape[1]
        x_fwd = self.fwd_conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_fwd = F.silu(x_fwd)
        x_proj_fwd = self.fwd_x_proj(x_fwd)
        dt_fwd, B_fwd, C_fwd = torch.split(x_proj_fwd, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_fwd = F.softplus(self.fwd_dt_proj(dt_fwd))
        y_fwd = _ssm_scan_cpu(x_fwd, dt_fwd, -torch.exp(self.fwd_A_log), B_fwd, C_fwd, self.fwd_D, z)

        x_flip, z_flip = x_in.flip([1]), z.flip([1])
        x_bwd = self.bwd_conv1d(x_flip.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_bwd = F.silu(x_bwd)
        x_proj_bwd = self.bwd_x_proj(x_bwd)
        dt_bwd, B_bwd, C_bwd = torch.split(x_proj_bwd, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_bwd = F.softplus(self.bwd_dt_proj(dt_bwd))
        y_bwd = _ssm_scan_cpu(x_bwd, dt_bwd, -torch.exp(self.bwd_A_log), B_bwd, C_bwd, self.bwd_D, z_flip).flip([1])

        return self.out_proj(y_fwd + y_bwd)

# Hot-swap alias for architecture compatibility
MambaBlock = BiMambaBlock

class TimestepEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
    def forward(self, t):
        device = t.device
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MambaDiffusion(nn.Module):
    def __init__(self, img_size=32, in_channels=3, patch_size=4, d_model=256, n_layers=4):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.time_mlp = nn.Sequential(
            TimestepEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mamba": BiMambaBlock(d_model),
                "norm": nn.LayerNorm(d_model)
            })
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.final_linear = nn.Linear(d_model, patch_size * patch_size * in_channels)

    def forward(self, x, t):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        t_emb = self.time_mlp(t).unsqueeze(1)
        for layer in self.layers:
            h = layer["norm"](x + t_emb)
            x = x + layer["mamba"](h)
        x = self.final_norm(x)
        x = self.final_linear(x)
        x = x.transpose(1, 2).reshape(B, C * self.patch_size**2, H // self.patch_size, W // self.patch_size)
        x = self.custom_depatchify(x, B, C, H, W)
        return x

    def custom_depatchify(self, x, B, C, H, W):
        x = x.reshape(B, C, self.patch_size, self.patch_size, H // self.patch_size, W // self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, C, H, W)
        return x

class DiffusionEngine:
    def __init__(self, model, n_steps=1000, device="cpu"):
        self.model = model.to(device)
        self.n_steps = n_steps
        self.device = device
        self.beta = torch.linspace(1e-4, 0.02, n_steps, device=device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def noise(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])[:, None, None, None]
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise, noise

    @torch.no_grad()
    def sample(self, shape):
        self.model.eval()
        x = torch.randn(shape, device=self.device)
        for i in tqdm(reversed(range(self.n_steps)), desc="Sampling", total=self.n_steps):
            t = torch.tensor([i] * shape[0], device=self.device)
            pred_noise = self.model(x, t)
            alpha_t = self.alpha[i]
            alpha_bar_t = self.alpha_bar[i]
            beta_t = self.beta[i]
            noise = torch.randn_like(x) if i > 0 else 0
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / (torch.sqrt(1 - alpha_bar_t))) * pred_noise) + torch.sqrt(beta_t) * noise
        return x
