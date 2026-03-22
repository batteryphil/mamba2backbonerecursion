"""
mind_reader.py — Interactive "Mind Reader" script for Mamba-130M v17.
This script demonstrates the dynamic computation of the recursive reasoning model,
showing it literally change its mind as it thinks through a problem.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel
import os

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS     = "mamba130m_finetuned_v19.pt"
CONF_THR    = 0.85

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token

# ── LoRA Linear ───────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.bias  = linear.bias
        d_out, d_in = linear.weight.shape
        self.register_buffer("base_weight", linear.weight.data.clone())
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.scale  = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A)

    @property
    def weight(self) -> torch.Tensor:
        return self.base_weight + self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

# ── Recursive Mamba Wrapper ───────────────────────────────────────────────────
class RecursiveMamba130M(nn.Module):
    MAX_LOOPS: int = 10

    def __init__(self, backbone_model: MambaLMHeadModel, lora_rank: int = 8):
        super().__init__()
        self.backbone   = backbone_model.backbone
        self.lm_head    = backbone_model.lm_head
        self.top_layers = nn.ModuleList([backbone_model.backbone.layers[i] for i in range(6, 24)])
        self.norm       = backbone_model.backbone.norm_f
        d_model         = backbone_model.backbone.embedding.embedding_dim

        ALPHA = lora_rank * 2.0
        for layer in self.top_layers:
            mx = layer.mixer
            for attr in ("in_proj", "x_proj", "dt_proj"):
                if hasattr(mx, attr):
                    setattr(mx, attr, LoRALinear(getattr(mx, attr), rank=lora_rank, alpha=ALPHA))

        self.step_emb = nn.Embedding(self.MAX_LOOPS, d_model)
        nn.init.normal_(self.step_emb.weight, std=0.01)
        self.loop_norm = nn.RMSNorm(d_model)

    def forward(self, input_ids: torch.Tensor, print_trace=True) -> tuple:
        x = self.backbone.embedding(input_ids)
        residual = None

        for layer in self.backbone.layers[:6]:
            x, residual = layer(x, residual)

        base_features = x.clone()
        loops_taken = self.MAX_LOOPS

        for step_i in range(self.MAX_LOOPS):
            step_vec = self.step_emb(torch.tensor(step_i, device=x.device))
            x = x + step_vec
            for layer in self.top_layers:
                x, residual = layer(x, residual)
            x = x + base_features
            x = self.loop_norm(x)

            # Check confidence on current state
            with torch.no_grad():
                logits_tmp = self.lm_head(self.norm(x, residual, prenorm=False))
                p = torch.softmax(logits_tmp[0, -1, :], dim=-1)
                
                max_prob = p.max().item()
                entropy  = -(p * (p + 1e-12).log()).sum().item()
                top_tok_id = p.argmax().item()
                top_tok  = tokenizer.decode([top_tok_id]).strip()
                
                if print_trace:
                    print(f"  [Loop {step_i+1}] Token: {top_tok!r} (Confidence: {max_prob*100:.0f}%)")

                if max_prob > CONF_THR:
                    loops_taken = step_i + 1
                    break

        if print_trace:
            print(f"  👉 Final Output: {top_tok}")

        x = self.norm(x, residual, prenorm=False)
        return self.lm_head(x), loops_taken

# ── Load Model ────────────────────────────────────────────────────────────────
def load_model():
    print("Loading base mamba-130m...")
    base_model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m", dtype=torch.float32, device=DEVICE)
    for p in base_model.parameters(): p.requires_grad = False
    
    print("Wrapping with Recursive reasoning head...")
    model = RecursiveMamba130M(base_model, lora_rank=8).to(DEVICE)
    model.eval()
    
    if os.path.exists(WEIGHTS):
        print(f"Loading finetuned weights: {WEIGHTS}")
        ckpt = torch.load(WEIGHTS, map_location=DEVICE)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
    else:
        print(f"⚠️ Warning: Could not find {WEIGHTS}, running untrained!")
        
    return model

def interactive_loop():
    model = load_model()
    print("\n" + "="*60)
    print("  MAMBA-130M RECURSIVE REASONING: MIND READER")
    print("  Type 'quit' or 'exit' to stop.")
    print("  Remember to end prompts with 'Answer:'")
    print("="*60 + "\n")
    
    while True:
        try:
            prompt = input("\nQuestion: ").strip()
            if prompt.lower() in ("quit", "exit", ""):
                break
                
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                model(input_ids, print_trace=True)
                
        except KeyboardInterrupt:
            break
        except EOFError:
            break

if __name__ == "__main__":
    interactive_loop()
