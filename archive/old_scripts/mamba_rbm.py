import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

# Attempt to load optimized kernels
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from causal_conv1d import causal_conv1d_fn
    HAS_OPTIMIZED_KERNELS = True
except ImportError:
    HAS_OPTIMIZED_KERNELS = False

@dataclass
class Config:
    vocab_size: int = 50258
    d_model: int = 1024
    n_layers: int = 8      # Depth of the recursive stack
    seq_len: int = 1024
    n_reasoning: int = 3   # Internal loop iterations for Path B
    n_memory_layers: int = 0    # Extra high-d_state memory consolidation layers appended after n_layers
    memory_d_state: int = 32    # d_state used by memory layers (double the normal 16)

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

class DualCausalMambaBlock(nn.Module):
    """
    Dual-Path Causal Mamba: 
    Path A (Intuition) + Path B (Logic).
    Shares projections for VRAM efficiency while maintaining zero-leakage causality.
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

        # --- Path A: Intuition (Causal) ---
        self.a_conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv-1, groups=self.d_inner, bias=True)
        self.a_x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.a_dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.a_A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.a_D = nn.Parameter(torch.ones(self.d_inner))

        # --- Path B: Logic (Causal) ---
        self.b_conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv-1, groups=self.d_inner, bias=True)
        self.b_x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.b_dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.b_A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.b_D = nn.Parameter(torch.ones(self.d_inner))

    def _run_ssm(self, x_in, z, mode='a'):
        conv = self.a_conv1d if mode == 'a' else self.b_conv1d
        x_proj_layer = self.a_x_proj if mode == 'a' else self.b_x_proj
        dt_proj_layer = self.a_dt_proj if mode == 'a' else self.b_dt_proj
        A_log = self.a_A_log if mode == 'a' else self.b_A_log
        D = self.a_D if mode == 'a' else self.b_D
        L = x_in.shape[1]

        if HAS_OPTIMIZED_KERNELS and x_in.is_cuda:
            x_c = causal_conv1d_fn(x_in.transpose(1, 2), conv.weight.squeeze(1), conv.bias, activation="silu")
            x_p = x_proj_layer(x_c.transpose(1, 2))
            dt, B, C = torch.split(x_p, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = F.linear(dt, dt_proj_layer.weight, dt_proj_layer.bias)
            y = selective_scan_fn(
                x_c, dt.transpose(1, 2), -torch.exp(A_log), 
                B.transpose(1, 2), C.transpose(1, 2), D.float(), 
                z.transpose(1, 2), delta_softplus=True
            ).transpose(1, 2)
            return y
        else:
            x = conv(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
            x = F.silu(x)
            x_p = x_proj_layer(x)
            dt, B, C = torch.split(x_p, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = F.softplus(dt_proj_layer(dt))
            return _ssm_scan_cpu(x, dt, -torch.exp(A_log), B, C, D, z)

    def forward(self, u: torch.Tensor, use_logic=False) -> torch.Tensor:
        xz = self.in_proj(u)
        x_in, z = xz.chunk(2, dim=-1)
        
        if not use_logic:
            # Intuition only (Standard context building)
            return self.out_proj(self._run_ssm(x_in, z, mode='a'))
        else:
            # Logic path (Recursive deliberation)
            return self.out_proj(self._run_ssm(x_in, z, mode='b'))

class RecursiveMambaLM(nn.Module):
    """
    RBM (Recurrent Bidirectional Mamba) -> PROTOCOL v6.2 (Dual Causal Reasoning):
    Uses a Dual-Path Causal engine. Path A is 'Intuition'. Path B is 'Reasoning' (Recursive).
    Zero-cheating architecture for Autoregressive stability.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(0.05)  # 🛡️ Protocol v6.6: Embedding dropout

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mamba": DualCausalMambaBlock(config.d_model),
                "norm": nn.LayerNorm(config.d_model)
            })
            for _ in range(config.n_layers)
        ])

        # 🧠 Memory Consolidation Layers: higher d_state for long-range retention
        # Randomly initialized — learn to consolidate state across recursive passes
        self.memory_layers = nn.ModuleList([
            nn.ModuleDict({
                "mamba": DualCausalMambaBlock(config.d_model, d_state=config.memory_d_state),
                "norm": nn.LayerNorm(config.d_model)
            })
            for _ in range(config.n_memory_layers)
        ])

        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.pass_dropout = nn.Dropout(0.1)
        self.lm_head.weight = self.token_embed.weight


    def forward(self, input_ids):
        x = self.token_embed(input_ids)
        if self.training:
            x = self.embed_dropout(x)  # 🛡️ Protocol v6.6: Only drop during training, not inference
        n = self.config.n_reasoning
        
        # Parallel Path Multiplier: 
        # Both paths contribute to the update.
        res_scale = 1.0 / math.sqrt(2.0) 

        for layer in self.layers:
            residual = x
            normed_x = layer["norm"](x)
            
            # --- PARALLEL EXECUTION PROTOCOL v6.3 ---
            # 1. Path A: Intuition (Context Anchor)
            intuition_out = layer["mamba"](normed_x, use_logic=False)
            
            # 2. Path B: Logic (Recursive Deliberation)
            # Starting from raw normed_x instead of intuition's output to prevent error chaining
            logic_out = normed_x 
            for _ in range(n):
                logic_res = logic_out
                # Use Logic Path B + LayerNorm anchor to prevent explosion
                logic_out = layer["mamba"](layer["norm"](logic_out), use_logic=True) 
                logic_out = self.pass_dropout(logic_out)
                logic_out = logic_out + logic_res
            
            # Weighted merge of Intuition + Logic
            x = (intuition_out + logic_out) * res_scale + residual

        # 🧠 Memory Consolidation Layers (added after standard layers in same recursive pass)
        for mem_layer in self.memory_layers:
            residual = x
            normed_x = mem_layer["norm"](x)
            intuition_out = mem_layer["mamba"](normed_x, use_logic=False)
            logic_out = normed_x
            for _ in range(n):
                logic_res = logic_out
                logic_out = mem_layer["mamba"](mem_layer["norm"](logic_out), use_logic=True)
                logic_out = self.pass_dropout(logic_out)
                logic_out = logic_out + logic_res
            x = (intuition_out + logic_out) * res_scale + residual

        x = self.final_norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=50, temperature=0.8, top_k=40):
        self.eval()
        current_ids = prompt_ids
        for _ in range(max_new_tokens):
            logits = self.forward(current_ids[:, -self.config.seq_len:])
            next_token_logits = logits[:, -1, :] / (temperature + 1e-8)
            v, _ = torch.topk(next_token_logits, top_k)
            next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            current_ids = torch.cat([current_ids, next_token], dim=1)
            if next_token.item() == 50256: break
        return current_ids
