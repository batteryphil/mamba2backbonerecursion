import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from mamba_causal import CausalMambaBlock

@dataclass
class Config:
    vocab_size: int = 50258
    d_model: int = 1024
    n_layers: int = 11
    seq_len: int = 1024

class MambaLM(nn.Module):
    """
    Standard Autoregressive Language Model using Causal Mamba layers.
    Refactored from DiM_LLM for Next-Token Prediction.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        
        # 1. Standard Token Embedding
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        
        # 2. Causal Mamba Backbone
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mamba": CausalMambaBlock(config.d_model),
                "norm": nn.LayerNorm(config.d_model)
            })
            for _ in range(config.n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.d_model)
        
        # 3. Standard LM Head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Optional: Weight Tying
        self.lm_head.weight = self.token_embed.weight
        
    def forward(self, input_ids):
        """
        Standard AR Forward Pass
        input_ids: (B, L)
        Returns: logits (B, L, V)
        """
        # Token Embeddings
        x = self.token_embed(input_ids) # (B, L, D)
        
        # Process Layers
        for layer in self.layers:
            # Pre-norm residual connection
            h = layer["norm"](x)
            x = x + layer["mamba"](h)
            
        # Final Norm and Projection
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        return logits

def calculate_ar_loss(logits, targets):
    """
    Next-Token Prediction Loss calculation.
    """
    B, L, V = logits.shape
    # Flatten for Cross Entropy
    return F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
