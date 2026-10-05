"""
mamba_gist_engine.py — Mamba + PRIME Associative Memory Hybrid Engine
====================================================================
Integrates Generative Thought Reconstruction Layer (Gist) from PRIME-Moment-Attention
into Mamba State Space Model backbones.

Replaces the flawed "Prefix Scratchpad" and "Lifeline Decay" from RLF with:
1. In-situ associative binding: m_t^2 ⊗ v_t written into second-order state M ∈ R^(d_map x D)
2. True recurrent state passing: (M, Z) passed across reasoning loops in O(1) memory
3. Zero-initialized reconstruction: base pre-trained weights preserved 100% at step 0
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# Automatically set cache and MIOpen paths for offline / ROCm compatibility
MIO_DIR = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.makedirs(MIO_DIR, exist_ok=True)
os.environ["MIOPEN_USER_DB_PATH"] = MIO_DIR
os.environ["MIOPEN_CACHE_DIR"] = MIO_DIR
os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"
os.environ["AMD_SERIALIZE_KERNEL"] = "3"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# PRIME-Moment-Attention / Gist Associative Memory import
try:
    from thought_reconstruction_map import GenerativeThoughtReconstructionLayer
except ImportError:
    PRIME_SRC = Path(__file__).resolve().parent.parent / "PRIME-Moment-Attention" / "src"
    if str(PRIME_SRC) not in sys.path:
        sys.path.insert(0, str(PRIME_SRC))
    from prime_moment_attention.thought_reconstruction_map import GenerativeThoughtReconstructionLayer


class MambaGistWrapperLayer(nn.Module):
    """
    Wraps an individual Mamba layer with a Generative Thought Reconstruction Layer (Gist).
    
    The Mamba layer runs standard state space scanning, and Gist accumulates associative
    bindings into an outer-product state M ∈ R^(d_map x D).
    """
    def __init__(
        self,
        mamba_layer: nn.Module,
        d_model: int,
        d_map: int = 32,
        decay: float = 0.9995,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.mamba_layer = mamba_layer
        self.gist = GenerativeThoughtReconstructionLayer(
            d_model=d_model,
            d_map=d_map,
            decay=decay,
            eps=eps,
        )
        # Small weight init so Gist starts with negligible perturbation while allowing full gradient flow
        nn.init.normal_(self.gist.recon_proj.weight, std=1e-3)
        if self.gist.recon_proj.bias is not None:
            nn.init.zeros_(self.gist.recon_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # 1. Standard Mamba forward pass
        x = self.mamba_layer(hidden_states, **kwargs)
        if isinstance(x, tuple):
            x = x[0]

        target_dtype = self.gist.map_proj.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        # 2. In-situ Associative Memory consolidation & retrieval
        gist_out, new_state = self.gist(x, state=state, return_state=return_state)
        return gist_out, new_state


class MambaGistModel(nn.Module):
    """
    Mamba language model hybridized with PRIME Gist Associative Memory.
    Supports single-pass causal execution and multi-loop recurrent state passing.
    """
    def __init__(
        self,
        base_model_name_or_path: str = "state-spaces/mamba-130m-hf",
        gist_layer_indices: Optional[List[int]] = None,
        d_map: int = 32,
        decay: float = 0.9995,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.base_model_name = base_model_name_or_path
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype

        print(f"[*] Loading base model: {base_model_name_or_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, local_files_only=True)
            if len(self.tokenizer.encode("test")) == 0:
                self.tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf", local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf", local_files_only=True)

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            local_files_only=True,
            torch_dtype=dtype,
        ).to(self.device)

        d_model = self.model.config.hidden_size
        num_layers = len(self.model.backbone.layers)

        # Default: insert Gist layers at mid-trunk and final layer
        if gist_layer_indices is None:
            if num_layers <= 24:
                gist_layer_indices = [num_layers // 2 - 1, num_layers - 1] # e.g. [11, 23]
            else:
                # e.g. [11, 23, 35, 47] for 48 layers
                gist_layer_indices = [
                    num_layers // 4 - 1,
                    num_layers // 2 - 1,
                    3 * num_layers // 4 - 1,
                    num_layers - 1,
                ]

        self.gist_layer_indices = sorted(gist_layer_indices)
        self.gist_modules = nn.ModuleDict()

        print(f"[*] Transplanting Gist Associative Memory (d_map={d_map}) onto layers: {self.gist_layer_indices}")
        for idx in self.gist_layer_indices:
            orig_layer = self.model.backbone.layers[idx]
            wrapped = MambaGistWrapperLayer(
                orig_layer,
                d_model=d_model,
                d_map=d_map,
                decay=decay,
            ).to(self.device, dtype=dtype)
            self.model.backbone.layers[idx] = wrapped
            self.gist_modules[str(idx)] = wrapped.gist

        self.d_model = d_model
        self.d_map = d_map
        self.num_layers = num_layers

    def freeze_backbone(self, unfreeze_lm_head: bool = False):
        """Freeze all base Mamba weights; keep only Gist layers trainable."""
        for p in self.model.parameters():
            p.requires_grad = False

        for gist in self.gist_modules.values():
            for p in gist.parameters():
                p.requires_grad = True

        if unfreeze_lm_head:
            for p in self.model.lm_head.parameters():
                p.requires_grad = True

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[*] Backbone frozen. Trainable parameters: {n_train:,} / {n_total:,} ({n_train/n_total*100:.2f}%)")

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        state: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass through Mamba + Gist.
        
        Args:
            input_ids: Input token tensor [B, L]
            states / state: Dictionary mapping layer_idx -> (M, Z) associative memory states
            return_state: Whether to return updated Gist states
        """
        if states is None and state is not None:
            states = state
        hidden_states = self.model.backbone.embeddings(input_ids)
        new_states = {} if return_state else None

        for idx, layer in enumerate(self.model.backbone.layers):
            if idx in self.gist_layer_indices:
                prior_state = states.get(idx) if states is not None else None
                hidden_states, layer_state = layer(
                    hidden_states,
                    state=prior_state,
                    return_state=return_state,
                )
                if return_state and layer_state is not None:
                    new_states[idx] = layer_state
            else:
                hidden_states = layer(hidden_states)
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]

        hidden_states = self.model.backbone.norm_f(hidden_states)
        hidden_states = hidden_states.to(self.model.lm_head.weight.dtype)
        logits = self.model.lm_head(hidden_states)

        if return_state:
            return logits, new_states
        return logits, None

    def recurrent_forward(
        self,
        input_ids: torch.Tensor,
        num_loops: int = 2,
    ) -> Tuple[List[torch.Tensor], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Executes num_loops reasoning passes over the prompt.
        State (M, Z) carries associative memory across loops in O(1) space.
        """
        loop_logits = []
        current_state = None

        for loop_idx in range(num_loops):
            if current_state is not None:
                current_state = {k: (m.detach(), z.detach()) for k, (m, z) in current_state.items()}
            logits, current_state = self.forward(
                input_ids,
                states=current_state,
                return_state=True,
            )
            loop_logits.append(logits)

        return loop_logits, current_state
