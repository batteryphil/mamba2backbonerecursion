import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from tqdm import tqdm
from mamba_diffusion import MambaBlock

@dataclass
class Config:
    vocab_size: int = 50258
    d_model: int = 1024
    n_layers: int = 11
    seq_len: int = 1024 # CRITICAL: MUST MATCH TRAINING


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, :, :])

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:, :seq_len, ...].to(x.device),
            self.sin_cached[:, :seq_len, ...].to(x.device),
        )

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


class DiM_LLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        
        # Token Embedder
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        
        # Time Embedding (Continuous t_norm [0, 1])
        self.time_mlp = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, 2 * config.d_model) # Expanded for AdaLN (gamma, beta)
        )
        
        # New projection layer for the upcoming C++ context memory bank
        self.context_norm = nn.LayerNorm(config.d_model)
        self.context_proj = nn.Linear(config.d_model, 2 * config.d_model)

        # 🚀 RoPE (Rotary Position Embeddings)
        self.rope = RotaryEmbedding(config.d_model)
        
        # 🚀 Self-Conditioning Projection
        self.self_cond_proj = nn.Linear(config.vocab_size, config.d_model, bias=False)

        # Mamba Backbone
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mamba": MambaBlock(config.d_model),
                "norm": nn.LayerNorm(config.d_model)
            })
            for _ in range(config.n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.d_model)
        # Standard language modeling head
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)
        
        # 🚀 Weight Tying (VRAM Optimization)
        self.output_proj.weight = self.token_embed.weight
        
        self.zero_init_ada()

    def zero_init_ada(self):
        """Force gamma/beta to start at 0 so the model behaves as an identity map initially."""
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)
        nn.init.zeros_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)

    def forward(self, input_ids, t_norm, context_bank=None, self_cond=None):
        """
        input_ids: (B, L)
        t_norm: (B,) - Diffusion timestep [0, 1]
        context_bank: (B, 1, D) or (B, L, D) - Optional future memory hook
        self_cond: (B, L, V) - Optional self-conditioning logits/probs
        """
        x = self.token_embed(input_ids) # (B, L, D)
        
        # 🚀 Apply RoPE (Spatial Grounding)
        B, L, D = x.shape
        cos, sin = self.rope(x, seq_len=L)
        x = apply_rotary_pos_emb(x, cos, sin)
        
        # 🚀 Apply Self-Conditioning
        if self_cond is not None:
            x = x + self.self_cond_proj(self_cond)
        
        # Generate t_emb (B, 1, 2*D)
        t_emb = self.time_mlp(t_norm.unsqueeze(-1).float()).unsqueeze(1) 
        
        # If context_bank is provided, combine it with t_emb
        if context_bank is not None:
            t_emb = t_emb + self.context_proj(self.context_norm(context_bank))

        h = x
        for layer in self.layers:
            # 🎨 True AdaLN (Adaptive Layer Normalization)
            # Split embedding into gamma and beta parameters
            gamma, beta = t_emb.chunk(2, dim=-1)
            
            # Apply AdaLN: LayerNorm(x) * (1 + gamma) + beta
            h = layer["norm"](h) * (1 + gamma) + beta
            h = h + layer["mamba"](h)
            
        h = self.final_norm(h)
        return self.output_proj(h)

class MaskedDiffusionEngine:
    def __init__(self, model, config, device="cuda", ema_decay=0.999):
        self.model = model.to(device)
        self.vocab_size = config.vocab_size
        self.mask_id = config.vocab_size - 1
        self.device = device
        self.seq_len = config.seq_len
        
        # EMA Setup
        self.ema_decay = ema_decay
        self.ema_model = None # Will be set in train_llm.py if used

    def update_ema(self):
        if self.ema_model is None:
            return
        with torch.no_grad():
            for p_ema, p_model in zip(self.ema_model.parameters(), self.model.parameters()):
                p_ema.data.mul_(self.ema_decay).add_(p_model.data, alpha=1 - self.ema_decay)

    def get_mask_ratio(self, t_norm: torch.Tensor) -> torch.Tensor:
        """
        Calculates the mask ratio for unmasking schedule.
        V4 Warp: Hits 0.0 at t <= 0.15 (85% mark), forcing the commitment cliff.
        """
        # Map t_norm [0.15, 1.0] -> [0.0, 1.0]. t_norm < 0.15 results in 0.0 mask ratio.
        warped_t = torch.clamp((t_norm - 0.15) / (1.0 - 0.15), min=0.0)
        # Final commitment schedule (returns 1.0 at t=1.0, 0.0 at t=0.15)
        return torch.sin(warped_t * math.pi / 2)

    def compute_snr(self, t_norm: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Signal-to-Noise Ratio for Discrete Masked Diffusion.
        SNR = (1 - mask_ratio) / mask_ratio
        """
        m = self.get_mask_ratio(t_norm)
        return (1.0 - m) / (m + 1e-8)

    def forward_process(self, input_ids, context_bank=None, loss_mask=None, gamma=5.0):
        B, L = input_ids.shape
        t_norm = torch.rand(B, device=self.device)
        mask_ratio = self.get_mask_ratio(t_norm)
        
        rand_tensor = torch.rand(B, L, device=self.device)
        mask_bool = rand_tensor < mask_ratio.unsqueeze(-1)
        
        if loss_mask is not None:
            mask_bool = mask_bool & (loss_mask == 1)
        
        masked_inputs = input_ids.clone()
        masked_inputs[mask_bool] = self.mask_id
        
        # 🚀 Self-Conditioning Training Loop
        self_cond = None
        if torch.rand(1).item() < 0.5:
            with torch.no_grad():
                # Initial pass to get prediction
                initial_logits = self.model(masked_inputs, t_norm, context_bank=context_bank)
                self_cond = F.softmax(initial_logits, dim=-1).detach()
        
        logits = self.model(masked_inputs, t_norm, context_bank=context_bank, self_cond=self_cond)
        logits = torch.clamp(logits, -20, 20) # 🛡️ Stability Guard
        
        # 1. Per-token Cross-Entropy
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size), 
            input_ids.view(-1), 
            reduction='none'
        ).view(B, L)
        
        # 2. Average over sequence length (masked tokens only) for per-sequence losses
        mask_float = mask_bool.float()
        unmasked_count = mask_float.sum(dim=1) + 1e-8
        per_batch_loss = (loss * mask_float).sum(dim=1) / unmasked_count
        
        # 3. Calculate Min-SNR Weights: w(t) = min(SNR(t), gamma)
        snr = self.compute_snr(t_norm)
        weights = torch.clamp(snr, max=gamma)
        
        # 4. Final weight reduction
        weighted_loss = (per_batch_loss * weights).mean()
        
        return weighted_loss

    @torch.no_grad()
    def sample(self, n_samples=1, steps=32, prompt_ids=None, base_temp=1.2, min_temp=0.1, context_bank=None):
        # Use EMA model for sampling if available
        sampling_model = self.ema_model if self.ema_model is not None else self.model
        sampling_model.eval()
        
        # Initialize with all [MASK]
        current_ids = torch.full((n_samples, self.seq_len), self.mask_id, dtype=torch.long, device=self.device)
        
        # If prompt is provided, place it at the start and fix it
        prompt_len = 0
        if prompt_ids is not None:
            prompt_len = min(prompt_ids.shape[1], self.seq_len)
            current_ids[:, :prompt_len] = prompt_ids[:, :prompt_len]
        
        mask_indices = torch.ones((n_samples, self.seq_len), dtype=torch.bool, device=self.device)
        mask_indices[:, :prompt_len] = False # Prompt tokens are not masked

        for step in tqdm(range(steps), desc="Unmasking Sequence"):
            progress = step / steps
            t_scalar = 1.0 - progress
            t_norm = torch.full((n_samples,), t_scalar, device=self.device)
            
            # Predict
            logits = sampling_model(current_ids, t_norm, context_bank=context_bank)
            
            # 🚀 TAPS (Time-Annealed Perturbation Sampling) Schedule
            # High temperature & Top-K early to break the "Repeat Curse", cooling down to lock grammar
            current_temp = base_temp * (1.0 - progress) + min_temp * progress
            current_top_k = int(200 * (1.0 - progress) + 10 * progress)
            
            # Top-K filtering
            v, _ = torch.topk(logits, current_top_k)
            logits[logits < v[:, :, [-1]]] = -float('Inf')

            # Stochastic Sampling with Annealed Temperature
            probs = F.softmax(logits / current_temp, dim=-1) # (B, L, V)
            B, L, V = probs.shape
            
            # Sample using Multinomial
            flat_probs = probs.reshape(-1, V)
            sampled_ids = torch.multinomial(flat_probs, 1).reshape(B, L)
            
            # Calculate confidence (use max prob from softmax for selection confidence)
            max_probs, _ = torch.max(probs, dim=-1)
            
            # Identify which tokens are currently [MASK] AND NOT part of the prompt
            is_masked = (current_ids == self.mask_id) & mask_indices
            
            # Calculate how many tokens to unmask this step
            ratio_to_unmask = self.get_mask_ratio(torch.tensor(t_scalar - (1.0/steps)))
            maskable_len = self.seq_len - prompt_len
            tokens_to_keep = int((1.0 - ratio_to_unmask.item()) * maskable_len)
            
            # Only consider confidence for currently masked tokens (low-confidence remasking)
            confidence = max_probs.masked_fill(~is_masked, -1.0) 
            _, indices_to_unmask = torch.topk(confidence, tokens_to_keep, dim=-1)
            
            # Fill sampled IDs into the selected indices
            current_ids.scatter_(1, indices_to_unmask, sampled_ids.gather(1, indices_to_unmask))

        return current_ids

    @torch.no_grad()
    def adaptive_sample(
        self,
        n_samples: int = 1,
        prompt_ids: torch.Tensor | None = None,
        base_temp: float = 1.2,
        min_temp: float = 0.1,
        context_bank: torch.Tensor | None = None,
        max_steps: int = 1000,
        min_steps: int = 32,
        entropy_threshold: float = 0.02
    ) -> tuple[torch.Tensor, int, float]:
        """
        Dynamic Entropy-Steered Inference for DiM-LLM.
        Uses Shannon Entropy as an early exit condition to optimize inference speed.
        """
        sampling_model = self.ema_model if self.ema_model is not None else self.model
        sampling_model.eval()
        
        # Initialize with all [MASK]
        current_ids = torch.full((n_samples, self.seq_len), self.mask_id, dtype=torch.long, device=self.device)
        
        # If prompt is provided, place it at the start and fix it
        prompt_len = 0
        if prompt_ids is not None:
            prompt_len = min(prompt_ids.shape[1], self.seq_len)
            current_ids[:, :prompt_len] = prompt_ids[:, :prompt_len]
        
        mask_indices = torch.ones((n_samples, self.seq_len), dtype=torch.bool, device=self.device)
        mask_indices[:, :prompt_len] = False # Prompt tokens are not masked
        
        steps_taken = 0
        avg_entropy_val = 1.0 # default high entropy

        for step in tqdm(range(max_steps), desc="Adaptive Unmasking"):
            steps_taken += 1
            progress = step / max_steps
            t_scalar = 1.0 - progress
            # 3060-optimized explicit type hinting
            t_norm = torch.full((n_samples,), t_scalar, dtype=torch.float32, device=self.device)
            
            # Predict
            device_type = 'cuda' if 'cuda' in str(self.device).lower() else 'cpu'
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits = sampling_model(current_ids, t_norm, context_bank=context_bank)
            
            # Convert logits to float32 for stable softmax and entropy calculations
            logits = logits.to(torch.float32)

            # 🚀 TAPS (Time-Annealed Perturbation Sampling) Schedule
            current_temp = base_temp * (1.0 - progress) + min_temp * progress
            current_top_k = int(200 * (1.0 - progress) + 10 * progress)
            
            # Top-K filtering
            v, _ = torch.topk(logits, current_top_k)
            logits[logits < v[:, :, [-1]]] = -float('Inf')

            # Stochastic Sampling with Annealed Temperature
            probs = F.softmax(logits / current_temp, dim=-1) # (B, L, V)
            B, L, V = probs.shape
            
            # --- Dynamic Entropy Calculation (Shannon) ---
            # Identify which tokens are currently [MASK] AND NOT part of the prompt
            is_masked = (current_ids == self.mask_id) & mask_indices
            
            if is_masked.any():
                # H = -\sum p \log p
                log_probs = torch.log(probs.clamp(min=1e-10))
                entropy = -(probs * log_probs).sum(dim=-1) # (B, L)
                
                # Average entropy only over the active [MASK] tokens
                avg_entropy = (entropy * is_masked.float()).sum() / is_masked.float().sum()
                avg_entropy_val = avg_entropy.item()
                
                # ── The 'Golden Ratio' Logic ──────────────────────────────────
                if avg_entropy_val < 0.1:
                    # We are in 'Fact/Code' territory. Tighten the screws.
                    current_knob = 0.2  # Very restrictive (quick exit)
                    dynamic_max_steps = 32
                else:
                    # We are in 'Creative/Reasoning' territory. Let it breathe.
                    current_knob = 0.02 # Very permissive (deep denoising)
                    dynamic_max_steps = 256
                
                # Inference-Time Scaling
                # Decay the threshold slightly as t_norm goes to 0 (allow more precision)
                current_threshold = current_knob * (0.8 + 0.2 * t_scalar)
                
                # Early Exit Condition
                if (step >= min_steps and avg_entropy_val < current_threshold) or (step >= dynamic_max_steps):
                    break
            
            # Sample using Multinomial
            flat_probs = probs.reshape(-1, V)
            sampled_ids = torch.multinomial(flat_probs, 1).reshape(B, L)
            
            # Calculate confidence (use max prob from softmax for selection confidence)
            max_probs, _ = torch.max(probs, dim=-1)
            
            # Calculate how many tokens to unmask this step
            ratio_to_unmask = self.get_mask_ratio(torch.tensor(t_scalar - (1.0/max_steps)))
            maskable_len = self.seq_len - prompt_len
            tokens_to_keep = int((1.0 - ratio_to_unmask.item()) * maskable_len)
            
            # Only consider confidence for currently masked tokens (low-confidence remasking)
            confidence = max_probs.masked_fill(~is_masked, -1.0) 
            _, indices_to_unmask = torch.topk(confidence, tokens_to_keep, dim=-1)
            
            # Fill sampled IDs into the selected indices
            current_ids.scatter_(1, indices_to_unmask, sampled_ids.gather(1, indices_to_unmask))

        return current_ids, steps_taken, avg_entropy_val




