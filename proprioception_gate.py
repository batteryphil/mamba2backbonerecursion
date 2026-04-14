import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricProprioceptionGate(nn.Module):
    """
    Geometric Proprioception Gate for Mamba SSM hidden state degeneration detection.

    Computes three geometric signals from the hidden state trajectory:
      - Velocity  : L2 norm of consecutive state differences (high = erratic change)
      - Drift     : 1 - cosine_similarity of consecutive states (high = direction change)
      - Stagnation: 1 - variance of rolling window  ← INVERTED coherence

    The Stagnation signal is the key fix. Raw variance (coherence) drops to zero
    when the model loops degenerately, which makes W_g vanish (gain → 1.0, no correction).
    By passing (1 - coherence), we map:
      healthy text   → low stagnation → small W_g contribution
      degenerate loop → high stagnation → large W_g dampening

    This ensures the gate fires HARDER on degenerate states, not on healthy ones.
    """

    def __init__(self, d_model: int = 2560, window_size: int = 8):
        """Initialise gate with d_model hidden dimension and rolling window size."""
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size

        # W_g maps [velocity, drift, stagnation] → d_model correction vector
        self.W_g = nn.Linear(3, d_model, bias=False)
        nn.init.zeros_(self.W_g.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Apply proprioceptive gain correction to hidden state sequence h.

        Args:
            h: Hidden state tensor of shape [B, L, D]

        Returns:
            h_out: Corrected hidden state of shape [B, L, D]
        """
        B, L, D = h.shape

        # Shift h by one position to compare consecutive states
        h_prev = F.pad(h[:, :-1, :], (0, 0, 1, 0), value=0.0)

        # Signal 1: Velocity — magnitude of state change per step
        # High velocity = fast-moving geometry (either healthy surprise or erratic loop entry)
        velocity = torch.norm(h - h_prev, p=2, dim=-1, keepdim=True)

        # Signal 2: Drift — directional instability
        # High drift (cosine_sim → -1) = states pointing in inconsistent directions
        # We invert cosine_similarity so HIGH value = HIGH instability
        drift = (1.0 - F.cosine_similarity(h, h_prev, dim=-1)).unsqueeze(-1)

        # Signal 3: Stagnation — INVERTED variance (the key fix)
        # Raw variance:  high for healthy text, near-zero for degenerate loops
        # Stagnation:    1 - variance, maps near-zero variance → high stagnation
        # This ensures W_g fires HARD when the model is stuck in a repeat loop.
        if L >= self.window_size:
            h_padded = F.pad(h, (0, 0, self.window_size - 1, 0))
            windows  = h_padded.unfold(1, self.window_size, 1)
            # Normalise variance to [0,1] range before inverting
            raw_var  = torch.var(windows, dim=-1).mean(dim=-1, keepdim=True)
            # Clamp normalised var to [0,1] then invert
            stagnation = 1.0 - raw_var.clamp(0.0, 1.0)
        else:
            # Short sequences: assume healthy (no stagnation)
            stagnation = torch.zeros(B, L, 1, device=h.device, dtype=h.dtype)

        # Combine all three signals: [velocity, drift, stagnation]
        signals = torch.cat([velocity, drift, stagnation], dim=-1)

        # Apply learned dampening: gain > 1.0 amplifies coherent directions,
        # gain < 1.0 suppresses degenerate attractors.
        gain  = 1.0 + self.W_g(signals)
        h_out = h * gain

        return h_out
