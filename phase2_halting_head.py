#!/usr/bin/env python3
"""
phase2_halting_head.py -- HaltingHead MLP for Project TinyRefinementModel

A small MLP attached to the final hidden state of the Mamba-1.4B student.
Predicts a binary halting signal: 1 = stop emitting ==== spacers, 0 = continue.
Trained jointly with the LM head under BCEWithLogitsLoss.

This module is imported by phase2_sculptor_trainer.py.
"""

import torch
import torch.nn as nn


class HaltingHead(nn.Module):
    """Two-layer MLP predicting when to stop emitting spacer tokens.

    Takes the final hidden state from the Mamba backbone and outputs
    a single logit. Post-sigmoid > 0.5 means "emit answer now."

    Architecture: hidden_dim -> 256 -> GELU -> 256 -> 1
    Weight init: Xavier uniform for stability during early training.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        """Initialize the HaltingHead.

        Args:
            hidden_dim: Dimension of the Mamba backbone hidden state.
            dropout: Dropout rate applied between layers.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable training.

        Sets the final linear layer bias to -2.0 so the initial sigmoid
        output is ~0.11, biasing the head toward 'continue reasoning'
        rather than halting immediately at the start of training.
        """
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Bias the output layer strongly toward 'don't halt yet'
        final_linear = [m for m in self.net.modules() if isinstance(m, nn.Linear)][-1]
        with torch.no_grad():
            final_linear.bias.fill_(-2.0)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Compute per-position halting logits from hidden states.

        Accepts either (batch, hidden_dim) or (batch, seq_len, hidden_dim).
        When 3D, processes every position to produce per-position predictions.

        Args:
            hidden_state: Tensor of shape (batch, hidden_dim) or
                          (batch, seq_len, hidden_dim).

        Returns:
            Tensor of shape (batch, 1) or (batch, seq_len, 1) -- raw logits.
        """
        return self.net(hidden_state)

    def predict(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Inference-mode prediction with sigmoid applied.

        Args:
            hidden_state: Same as forward().

        Returns:
            Tensor of shape (batch, 1) with values in [0, 1].
            Values > 0.5 indicate "halt spacer emission."
        """
        with torch.no_grad():
            logit = self.forward(hidden_state)
        return torch.sigmoid(logit)

    def extra_repr(self) -> str:
        """String representation for model summary."""
        return f"hidden_dim={self.hidden_dim}"
