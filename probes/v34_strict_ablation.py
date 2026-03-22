import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel

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
        return torch.stack([freqs.cos(), freqs.cos()], dim=-1).flatten()[:self.d_model].to(dtype=dtype), \
               torch.stack([freqs.sin(), freqs.sin()], dim=-1).flatten()[:self.d_model].to(dtype=dtype)
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(-2)
    def forward(self, x: torch.Tensor, loop_index: int) -> torch.Tensor:
        cos_v, sin_v = self._get_sincos(loop_index, x.device, x.dtype)
        return x * cos_v + self._rotate_half(x) * sin_v

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.bias  = linear.bias
        self.register_buffer("base_weight", linear.weight.data.clone())
        self.lora_A = nn.Parameter(torch.empty(rank, linear.weight.shape[1], dtype=linear.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(linear.weight.shape[0], rank, dtype=linear.weight.dtype))
        self.scale  = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A)
    @property
    def weight(self): return self.base_weight + self.scale * (self.lora_B @ self.lora_A)
    def forward(self, x): return F.linear(x, self.weight, self.bias)

from mamba_ssm import Mamba2
class RecursiveMamba2_v34(nn.Module):
    def __init__(self, backbone: MambaLMHeadModel):
        super().__init__()
        self.backbone, self.lm_head, self.all_layers = backbone.backbone, backbone.lm_head, nn.ModuleList(backbone.backbone.layers)
        self.norm, d_model = backbone.backbone.norm_f, backbone.backbone.embedding.embedding_dim
        for layer in self.all_layers[6:]:
            for attr in ("in_proj", "out_proj"):
                if hasattr(layer.mixer, attr): setattr(layer.mixer, attr, LoRALinear(getattr(layer.mixer, attr)))
        self.loop_rope = LoopRoPE(d_model)
        self.loop_norm = nn.RMSNorm(d_model).to(torch.bfloat16)
        self.mamba2_core = Mamba2(d_model=d_model, d_state=64, d_conv=4, expand=2, headdim=64, chunk_size=64).to(torch.bfloat16)
        self.lifeline_gate = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
    def _lifeline_inject(self, x, x_prompt): return x + self.lifeline_gate.to(x.dtype).unsqueeze(0).unsqueeze(0) * x_prompt

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

model = RecursiveMamba2_v34(base_model).to(DEVICE)
ckpt = torch.load("mamba2_130m_v34_rope_best.pt", map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

base_gate = model.lifeline_gate.data.clone()
ram_idx = torch.where(base_gate > 1.01)[0]
alu_idx = torch.where(base_gate < 0.97)[0]

def run_trace(model_obj, noise_type="none"):
    prompt = "P = algorithm. Q = P. R = Q. S = R. T = S. U = T. V = U. W = V. What is W?\nAnswer:"
    ids_ = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    ans_start = find_answer_start(ids_[0].tolist())
    trace = []
    with torch.no_grad():
        x = model_obj.backbone.embedding(ids_)
        residual = None
        for layer in model_obj.all_layers[:6]: x, residual = layer(x, residual)
        x_prompt = x.clone().detach()
        if noise_type == "noise":
            x_prompt = torch.randn_like(x_prompt) * x_prompt.std() + x_prompt.mean()
        elif noise_type == "shuffle":
            idx = torch.randperm(x_prompt.shape[1])
            x_prompt = x_prompt[:, idx, :]
        for loop_i in range(8):
            x = model_obj._lifeline_inject(x, x_prompt)
            x = model_obj.loop_rope(x, loop_i)
            for layer in model_obj.all_layers[6:]: x, residual = layer(x, residual)
            x = x + model_obj.mamba2_core(x)
            x = model_obj.loop_norm(x)
            logits = model_obj.lm_head(model_obj.norm(x, residual, prenorm=False))[0, ans_start - 1, :]
            top_id = logits.argmax().item()
            tok = tokenizer.decode([top_id]).strip()
            if top_id == HALT_ID: tok = "<HALT>"
            trace.append(tok)
    return trace

print("\n--- BASE (Gate unchanged) ---")
print(run_trace(model))

print("\n--- TRUE ZERO ABLATION (Lifeline Removed: Gate = 0.0) ---")
model.lifeline_gate.data.fill_(0.0)
print(run_trace(model))

print("\n--- RAM ISOLATION (Only RAM Dims active, everything else 0.0) ---")
new_gate = torch.zeros_like(base_gate)
new_gate[ram_idx] = base_gate[ram_idx]
model.lifeline_gate.data.copy_(new_gate)
print(run_trace(model))

print("\n--- ALU ISOLATION (Only ALU Dims active, everything else 0.0) ---")
new_gate = torch.zeros_like(base_gate)
new_gate[alu_idx] = base_gate[alu_idx]
model.lifeline_gate.data.copy_(new_gate)
print(run_trace(model))

print("\n--- NEUTRAL ISOLATION (Only Neutral Dims active, RAM/ALU 0.0) ---")
new_gate = base_gate.clone()
new_gate[ram_idx] = 0.0
new_gate[alu_idx] = 0.0
model.lifeline_gate.data.copy_(new_gate)
print(run_trace(model))

print("\n--- NOISE INJECTION (Full Gate, Gaussian Noise x_prompt) ---")
model.lifeline_gate.data.copy_(base_gate)
print(run_trace(model, "noise"))

print("\n--- SHUFFLED INJECTION (Full Gate, Permuted x_prompt) ---")
print(run_trace(model, "shuffle"))
