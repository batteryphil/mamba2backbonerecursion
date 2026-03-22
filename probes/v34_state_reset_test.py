import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel
from mamba_ssm import Mamba2

class LoopRoPE(nn.Module):
    def __init__(self, d_model: int, base: int = 10000):
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def _get_sincos(self, loop_index: int, device: torch.device, dtype: torch.dtype):
        n = torch.tensor(float(loop_index), device=device)
        freqs = n * self.inv_freq.to(device=device, dtype=torch.float32)
        emb = torch.stack([freqs, freqs], dim=-1).flatten()[:self.d_model]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, loop_index: int) -> torch.Tensor:
        cos, sin = self._get_sincos(loop_index, x.device, x.dtype)
        return x * cos + self._rotate_half(x) * sin

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.bias = linear.bias
        self.register_buffer("base_weight", linear.weight.data.clone())
        self.lora_A = nn.Parameter(torch.empty(rank, linear.weight.shape[1], dtype=linear.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(linear.weight.shape[0], rank, dtype=linear.weight.dtype))
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A)

    @property
    def weight(self):
        return self.base_weight + self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class RecursiveMamba2_v34(nn.Module):
    def __init__(self, backbone: MambaLMHeadModel, lora_rank: int = 8):
        super().__init__()
        self.backbone = backbone.backbone
        self.lm_head = backbone.lm_head
        self.all_layers = nn.ModuleList(self.backbone.layers)
        self.norm = self.backbone.norm_f
        d_model = self.backbone.embedding.embedding_dim

        for layer in self.all_layers[6:]:
            for attr in ("in_proj", "out_proj"):
                if hasattr(layer.mixer, attr):
                    setattr(layer.mixer, attr, LoRALinear(getattr(layer.mixer, attr), rank=lora_rank, alpha=lora_rank * 2.0))

        self.loop_rope = LoopRoPE(d_model)
        self.loop_norm = nn.RMSNorm(d_model).to(torch.bfloat16)
        self.mamba2_core = Mamba2(d_model=d_model, d_state=64, d_conv=4, expand=2, headdim=64, chunk_size=64).to(torch.bfloat16)
        self.lifeline_gate = nn.Parameter(torch.ones(d_model, dtype=torch.float32))

    def _lifeline_inject(self, x, x_prompt):
        return x + self.lifeline_gate.to(x.dtype).unsqueeze(0).unsqueeze(0) * x_prompt

DEVICE = "cuda"
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>", "<HALT>"]})
HALT_ID = tokenizer.convert_tokens_to_ids("<HALT>")

def find_answer_start(ids: list[int]) -> int:
    for boundary in (tokenizer.encode("Answer:", add_special_tokens=False),
                     tokenizer.encode(" Answer:", add_special_tokens=False),
                     tokenizer.encode("\nAnswer:", add_special_tokens=False)):
        n = len(boundary)
        for i in range(len(ids) - n + 1):
            if ids[i:i + n] == boundary:
                return min(i + n, len(ids) - 1)
    return -1

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

def run_12hop_test(residual_decay=1.0, x_decay=1.0, h_limit=None, mode="normal"):
    prompt = "A = algorithm. B = A. C = B. D = C. E = D. F = E. G = F. H = G. I = H. J = I. K = J. L = K. M = L. What is M?\nAnswer:"
    ids_ = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    ans_start = find_answer_start(ids_[0].tolist())
    
    trace = []
    with torch.no_grad():
        x = model.backbone.embedding(ids_)
        residual = None
        for layer in model.all_layers[:6]: x, residual = layer(x, residual)
        x_prompt = x.clone().detach()
            
        for loop_i in range(15):
            x = model._lifeline_inject(x, x_prompt)
            x = model.loop_rope(x, loop_i)
            
            for layer in model.all_layers[6:]:
                x, residual = layer(x, residual)
            
            x = x + model.mamba2_core(x)
            x = model.loop_norm(x)
            
            logits = model.lm_head(model.norm(x, residual, prenorm=False))[0, ans_start - 1, :]
            top_id = logits.argmax().item()
            tok = tokenizer.decode([top_id]).strip()
            if top_id == HALT_ID: tok = "<HALT>"
            trace.append(tok)
            
            if residual is not None:
                residual = residual * residual_decay
            x = x * x_decay
            
    return trace

print("\n--- 13-HOP STATE SATURATION TEST ---")
print(f"Base:          {run_12hop_test()}")
print(f"Res Decay 0.9: {run_12hop_test(residual_decay=0.9)}")
print(f"Res Decay 0.5: {run_12hop_test(residual_decay=0.5)}")
print(f"Res Decay 0.0: {run_12hop_test(residual_decay=0.0)}")
print(f"X Decay 0.9:   {run_12hop_test(x_decay=0.9)}")
print(f"X Decay 0.7:   {run_12hop_test(x_decay=0.7)}")
