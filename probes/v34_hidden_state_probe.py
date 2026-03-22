import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel
import re

DEVICE = "cuda"

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>", "<HALT>"]})
HALT_ID = tokenizer.convert_tokens_to_ids("<HALT>")

def find_answer_start(ids: list[int]) -> int:
    for boundary in (
        tokenizer.encode("Answer:",  add_special_tokens=False),
        tokenizer.encode(" Answer:", add_special_tokens=False),
        tokenizer.encode("\nAnswer:", add_special_tokens=False),
    ):
        n = len(boundary)
        for i in range(len(ids) - n + 1):
            if ids[i:i + n] == boundary:
                return min(i + n, len(ids) - 1)
    return -1

class LoopRoPE(nn.Module):
    def __init__(self, d_model: int, base: int = 10000):
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def _get_sincos(self, loop_index: int, device: torch.device, dtype: torch.dtype):
        n = torch.tensor(float(loop_index), device=device)
        freqs = n * self.inv_freq.to(device=device, dtype=torch.float32)
        cos_f = freqs.cos()
        sin_f = freqs.sin()
        cos_v = torch.stack([cos_f, cos_f], dim=-1).flatten()[:self.d_model]
        sin_v = torch.stack([sin_f, sin_f], dim=-1).flatten()[:self.d_model]
        return cos_v.to(dtype=dtype), sin_v.to(dtype=dtype)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1)
        return rotated.flatten(-2)

    def forward(self, x: torch.Tensor, loop_index: int) -> torch.Tensor:
        cos_v, sin_v = self._get_sincos(loop_index, x.device, x.dtype)
        return x * cos_v + self._rotate_half(x) * sin_v

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.bias  = linear.bias
        d_out, d_in = linear.weight.shape
        dtype = linear.weight.dtype
        self.register_buffer("base_weight", linear.weight.data.clone())
        self.lora_A = nn.Parameter(torch.empty(rank, d_in,  dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, dtype=dtype))
        self.scale  = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A)
    @property
    def weight(self) -> torch.Tensor:
        return self.base_weight + self.scale * (self.lora_B @ self.lora_A)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

from mamba_ssm import Mamba2
class RecursiveMamba2_v34(nn.Module):
    MAX_LOOPS: int = 16
    def __init__(self, backbone: MambaLMHeadModel, lora_rank: int = 8):
        super().__init__()
        self.backbone   = backbone.backbone
        self.lm_head    = backbone.lm_head
        self.all_layers = nn.ModuleList(backbone.backbone.layers)
        self.norm       = backbone.backbone.norm_f
        d_model         = backbone.backbone.embedding.embedding_dim

        for layer in self.all_layers[:6]:
            for p in layer.parameters():
                p.requires_grad = False

        for layer in self.all_layers[6:]:
            mx = layer.mixer
            for attr in ("in_proj", "out_proj"):
                if hasattr(mx, attr):
                    setattr(mx, attr, LoRALinear(getattr(mx, attr), rank=lora_rank, alpha=lora_rank * 2.0))

        self.loop_rope   = LoopRoPE(d_model)
        self.loop_norm   = nn.RMSNorm(d_model).to(torch.bfloat16)
        self.mamba2_core = Mamba2(d_model=d_model, d_state=64, d_conv=4, expand=2, headdim=64, chunk_size=64).to(torch.bfloat16)
        self.lifeline_gate = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
        self.d_model = d_model

    def _lifeline_inject(self, x: torch.Tensor, x_prompt: torch.Tensor) -> torch.Tensor:
        gate = self.lifeline_gate.to(x.dtype)
        return x + gate.unsqueeze(0).unsqueeze(0) * x_prompt

    def forward(self, x):
        pass # Not used directly in probe

print("Loading model for Hidden State Probe...")
base_model = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-130m", dtype=torch.bfloat16, device=DEVICE)

new_vocab = len(tokenizer)
old_vocab = base_model.backbone.embedding.weight.shape[0]
d_model = base_model.backbone.embedding.embedding_dim
if new_vocab > old_vocab:
    ne = nn.Embedding(new_vocab, d_model, dtype=torch.bfloat16)
    ne.weight.data[:old_vocab] = base_model.backbone.embedding.weight.data
    base_model.backbone.embedding = ne
    nh = nn.Linear(d_model, new_vocab, bias=False, dtype=torch.bfloat16)
    nh.weight.data[:old_vocab] = base_model.lm_head.weight.data
    base_model.lm_head = nh

model = RecursiveMamba2_v34(base_model, lora_rank=8).to(DEVICE)
ckpt = torch.load("mamba2_130m_v34_rope_best.pt", map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

tests = [
    ("Base (1.0)", "P = algorithm. Q = P. R = Q. S = R. T = S. U = T. V = U. W = V. What is W?\nAnswer:", ["P", "Q", "R", "S", "T", "U", "V", "algorithm", "<HALT>"], "none"),
    ("True Zero (0.0)", "P = algorithm. Q = P. R = Q. S = R. T = S. U = T. V = U. W = V. What is W?\nAnswer:", ["P", "Q", "R", "S", "T", "U", "V", "algorithm", "<HALT>"], "zero"),
    ("Noise Injection", "P = algorithm. Q = P. R = Q. S = R. T = S. U = T. V = U. W = V. What is W?\nAnswer:", ["P", "Q", "R", "S", "T", "U", "V", "algorithm", "<HALT>"], "noise"),
    ("Shuffle Injection", "P = algorithm. Q = P. R = Q. S = R. T = S. U = T. V = U. W = V. What is W?\nAnswer:", ["P", "Q", "R", "S", "T", "U", "V", "algorithm", "<HALT>"], "shuffle"),
]

base_gate = model.lifeline_gate.data.clone()

for label, prompt, expected, mode in tests:
    print(f"\n{'='*95}")
    print(f"PROBE: {label}")
    print(f"PROMPT: {prompt.strip()}")
    print(f"{'='*95}")
    
    ids_ = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    ans_start = find_answer_start(ids_[0].tolist())
    
    with torch.no_grad():
        x = model.backbone.embedding(ids_)
        residual = None
        for layer in model.all_layers:
            x, residual = layer(x, residual)
            
        x_prompt = x.clone().detach()
        if mode == "zero":
            x_prompt = x_prompt * 0.0
        elif mode == "noise":
            x_prompt = torch.randn_like(x_prompt) * x_prompt.std() + x_prompt.mean()
        elif mode == "shuffle":
            idx = torch.randperm(x_prompt.shape[1])
            x_prompt = x_prompt[:, idx, :]
        
        for loop_i in range(12):
            x = model._lifeline_inject(x, x_prompt)
            x = model.loop_rope(x, loop_i)
            for layer in model.all_layers[6:]:
                x, residual = layer(x, residual)
            x = x + model.mamba2_core(x)
            x = model.loop_norm(x)
            
            # Predict from the exact hidden state at ans_start - 1
            h = model.norm(x, residual, prenorm=False)
            logits = model.lm_head(h)
            logits_b = logits[0, ans_start - 1, :]
            probs = torch.softmax(logits_b.float(), dim=-1)
            
            exp_tok = expected[loop_i] if loop_i < len(expected) else "???"
            
            topk = torch.topk(logits_b, 3) 
            vals = topk.values.tolist()
            inds = topk.indices.tolist()
            
            margin = vals[0] - vals[1]
            top_tok = tokenizer.decode([inds[0]]).strip()
            if inds[0] == HALT_ID: top_tok = "<HALT>"
            
            match_marker = "✅" if (top_tok.lower() == exp_tok.lower() or (top_tok == "<HALT>" and exp_tok == "<HALT>")) else "❌"
            
            print(f"L{loop_i+1:<2d} [Exp: {exp_tok:>10s}] | Top: {top_tok:>10s} {match_marker} | Margin: {margin:6.2f} | Top 3: ", end="")
            for j in range(3):
                tok = tokenizer.decode([inds[j]]).strip()
                if inds[j] == HALT_ID: tok = "<HALT>"
                print(f"{tok}({probs[inds[j]]:.3f})  ", end="")
            print()
            
            if inds[0] == HALT_ID:
                break
