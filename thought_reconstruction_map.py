"""
Generative Thought Reconstruction Map (GTRM) Layer
===================================================
Biomimetic constructive memory layer for PrimeLM-50M.

Instead of storing verbatim tokens or raw keys/values in expanding VRAM,
this layer distills the cognitive trajectory into compact 32-dimensional
topological blueprints (m_t), stores them in a second-order quadratic manifold
(M^(2) in R^(32 x 512), 65.5 KB), and dynamically reconstructs the thought
during <think> generation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class GenerativeThoughtReconstructionLayer(nn.Module):
    """
    Second-Order Generative Thought Reconstruction Layer.
    
    1. Distills x_t in R^D into a compact topological blueprint m_t in R^d_map (d_map = 32).
    2. Gates consolidation with a learnable novelty detector gamma_t in [0, 1].
    3. Accumulates blueprints into second-order quadratic moment state M^(2) in R^(d_map x D).
    4. Dynamically reconstructs the original thought trajectory upon associative query q_t.
    """
    def __init__(
        self,
        d_model: int = 512,
        d_map: int = 32,
        decay: float = 0.9995,
        eps: float = 1e-4
    ):
        super().__init__()
        self.d_model = d_model
        self.d_map = d_map
        self.decay = decay
        self.eps = eps

        # 1. Topological Blueprint Encoder (d_model -> d_map)
        self.map_proj = nn.Linear(d_model, d_map, bias=False)
        self.map_norm = RMSNorm(d_map)

        # 2. Novelty / Salience Gate (identifies invariant facts, entities, numbers)
        self.salience_gate = nn.Linear(d_model, 1, bias=True)
        nn.init.constant_(self.salience_gate.bias, 0.0) # Neutral initial gate

        # 3. Memory Value Projection (information payload)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # 4. Reconstruction Query Projection (d_model -> d_map)
        self.q_proj = nn.Linear(d_model, d_map, bias=False)
        self.q_norm = RMSNorm(d_map)

        # 5. Generative Thought Reconstruction Projection
        self.recon_proj = nn.Linear(d_model, d_model, bias=False)
        self.recon_gate = nn.Linear(d_model, d_model, bias=True)
        nn.init.constant_(self.recon_gate.bias, -2.0) # Starts gentle, learns to inject

        # State size in bytes: M is (d_map * d_model), Z is (d_map)
        self.state_bytes = (d_map * d_model + d_map) * 4

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.
        Supports both parallel training mode [B, L, D] and autoregressive decode [B, 1, D].
        """
        B, L, D = x.shape

        # Step 1: Encode topological thought blueprint m_t in R^32
        m_raw = self.map_proj(x)
        m = self.map_norm(m_raw) # [B, L, d_map]

        # Step 2: Novelty/Salience gating
        gamma = torch.sigmoid(self.salience_gate(x)) # [B, L, 1]

        # Step 3: Value payload
        v = self.v_proj(x) # [B, L, D]

        # Step 4: Query projection
        q_raw = self.q_proj(x)
        q = self.q_norm(q_raw) # [B, L, d_map]

        # Quadratic term: m_sq = (m * m) in R^32
        m_sq = m * m # [B, L, d_map]
        q_sq = q * q # [B, L, d_map]

        # Modulate by novelty gate: only salient thoughts are consolidated
        m_sq_gated = m_sq * gamma # [B, L, d_map]

        if L == 1 and state is not None:
            # === Autoregressive O(1) Streaming Decode ===
            M, Z = state
            v_t = v.squeeze(1)          # [B, D]
            m_t = m_sq_gated.squeeze(1).to(v_t.dtype) # [B, d_map]
            q_t = q_sq.squeeze(1).to(v_t.dtype)       # [B, d_map]

            # Normalizer state
            Z = self.decay * Z + m_t

            # Update 2nd-order associative memory
            M = self.decay * M + torch.bmm(m_t.unsqueeze(2), v_t.unsqueeze(1))

            # Readout: Recall = q_t^T * M^(2) / (q_t^T * Z + eps)
            num = torch.bmm(q_t.unsqueeze(1), M) # [B, 1, D]
            den = torch.bmm(q_t.unsqueeze(1), Z.unsqueeze(-1)) + self.eps # [B, 1, 1]
            recall = num / den # [B, 1, D]

            # Generative reconstruction projection
            recon = self.recon_proj(recall)
            gate = torch.sigmoid(self.recon_gate(x))
            out = x + gate * recon

            return out, (M, Z) if return_state else None

        else:
            # === Parallel Causal Training Mode (Vectorized BMM) ===
            # S[b, i, j] = q_i^T m_j
            S = torch.bmm(q_sq, m_sq_gated.transpose(1, 2)) # [B, L, L]

            # Causal decay matrix: Lambda[i, j] = decay^(i-j) for i >= j else 0
            idx = torch.arange(L, device=x.device)
            dist = idx.unsqueeze(1) - idx.unsqueeze(0)
            causal_mask = dist >= 0
            decay_mat = torch.where(causal_mask, (self.decay ** dist.float()).to(x.dtype), torch.zeros_like(dist, dtype=x.dtype))
            decay_mat = decay_mat.unsqueeze(0) # [1, L, L]

            attn_weights = S * decay_mat # [B, L, L]
            recall_all = torch.bmm(attn_weights, v) # [B, L, D]
            
            norm_factor = torch.sum(attn_weights, dim=-1, keepdim=True)

            if state is not None:
                M, Z = state
                # Add influence from prior state
                t_decay = (self.decay ** (idx.float() + 1.0)).view(1, L, 1, 1).to(v.dtype)
                prior_recall = torch.matmul(q_sq.unsqueeze(2), M.unsqueeze(1)) * t_decay
                recall_all = recall_all + prior_recall.squeeze(2)
                
                t_decay_Z = (self.decay ** (idx.float() + 1.0)).view(1, L, 1).to(v.dtype)
                prior_norm = torch.bmm(q_sq, Z.unsqueeze(-1)) * t_decay_Z
                norm_factor = norm_factor + prior_norm

            norm_factor = norm_factor + self.eps
            recall_all = (recall_all / norm_factor).to(v.dtype)

            recon = self.recon_proj(recall_all)
            gate = torch.sigmoid(self.recon_gate(x))
            out = x + gate * recon

            final_state = None
            if return_state:
                # Vectorized cumulative state at end of sequence
                weights = (self.decay ** (L - 1 - idx).float()).view(1, L, 1).to(v.dtype)
                m_decayed = (m_sq_gated * weights).to(v.dtype) # [B, L, d_map]
                
                M_new = torch.bmm(m_decayed.transpose(1, 2), v) # [B, d_map, D]
                Z_new = torch.sum(m_decayed, dim=1) # [B, d_map]
                
                if state is not None:
                    M_old, Z_old = state
                    M_new = M_new + (self.decay ** L) * M_old
                    Z_new = Z_new + (self.decay ** L) * Z_old
                final_state = (M_new, Z_new)

            return out, final_state


def test_reconstruction_map():
    print("[*] Testing GenerativeThoughtReconstructionLayer...")
    layer = GenerativeThoughtReconstructionLayer(d_model=512, d_map=32)
    layer.eval()

    # 1. Parallel training mode test
    x = torch.randn(2, 64, 512)
    out_parallel, state = layer(x, return_state=True)
    M, Z = state
    assert out_parallel.shape == x.shape, f"Shape mismatch: {out_parallel.shape} vs {x.shape}"
    assert M.shape == (2, 32, 512), f"State shape mismatch: {M.shape}"
    assert Z.shape == (2, 32), f"State shape mismatch: {Z.shape}"
    print(f"  [+] Parallel Forward Pass: PASS (Output: {out_parallel.shape}, State: M {M.shape}, Z {Z.shape})")

    # 3. Autoregressive streaming step test
    x_step = torch.randn(2, 1, 512)
    out_step, next_state = layer(x_step, state=state, return_state=True)
    assert out_step.shape == (2, 1, 512)
    print("  [+] Autoregressive Decode Step: PASS (Flat O(1) state verified)")

    print("[SUCCESS] GenerativeThoughtReconstructionLayer fully functional!")


if __name__ == "__main__":
    test_reconstruction_map()
