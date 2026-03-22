"""
v31_probe.py — test the no-mask v31 model with full diagnostic
Runs fixed 5 loops, no halt, prints raw logit trace for top-5 tokens.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel, Mamba2

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
CKPT      = "mamba2_130m_v31_nomask_best.pt"
BASE_SPLIT = 6
MAX_LOOPS  = 8
LOOP_HEADDIM = 64
LOOP_D_STATE = 64

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>", "<HALT>"]})
HALT_ID = tokenizer.convert_tokens_to_ids("<HALT>")


class LoRALinear(nn.Module):
    def __init__(self, linear, rank=8, alpha=16.0):
        super().__init__()
        self.bias = linear.bias
        d_out, d_in = linear.weight.shape
        dtype = linear.weight.dtype
        self.register_buffer("base_weight", linear.weight.data.clone())
        self.lora_A = nn.Parameter(torch.empty(rank, d_in, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, dtype=dtype))
        self.scale  = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A)
    @property
    def weight(self): return self.base_weight + self.scale * (self.lora_B @ self.lora_A)
    def forward(self, x): return F.linear(x, self.weight, self.bias)


class V31Model(nn.Module):
    """Inference-only wrapper for v31 no-mask model."""

    def __init__(self, bb, rank=8):
        super().__init__()
        self.backbone   = bb.backbone
        self.lm_head    = bb.lm_head
        self.all_layers = nn.ModuleList(bb.backbone.layers)
        self.norm       = bb.backbone.norm_f
        d = bb.backbone.embedding.embedding_dim
        for layer in self.all_layers[:BASE_SPLIT]:
            for p in layer.parameters(): p.requires_grad = False
        for layer in self.all_layers[BASE_SPLIT:]:
            for attr in ("in_proj", "out_proj"):
                if hasattr(layer.mixer, attr):
                    setattr(layer.mixer, attr,
                            LoRALinear(getattr(layer.mixer, attr), rank, rank*2.0))
        self.step_emb    = nn.Embedding(MAX_LOOPS, d).to(torch.bfloat16)
        self.loop_norm   = nn.RMSNorm(d).to(torch.bfloat16)
        self.mamba2_core = Mamba2(d_model=d, d_state=LOOP_D_STATE, d_conv=4,
                                  expand=2, headdim=LOOP_HEADDIM, chunk_size=64
                                  ).to(torch.bfloat16)

    def run(self, input_ids, n=8):
        """Fixed n loops — no halt, no mask, raw full-vocab logits."""
        x, res = self.backbone.embedding(input_ids), None
        for layer in self.all_layers:
            x, res = layer(x, res)
        trace = []
        for loop_i in range(n):
            sv  = self.step_emb(torch.tensor(min(loop_i, MAX_LOOPS-1), device=x.device))
            x   = x + sv
            for layer in self.all_layers[BASE_SPLIT:]:
                x, res = layer(x, res)
            x   = x + self.mamba2_core(x)
            x   = self.loop_norm(x)
            lg  = self.lm_head(self.norm(x, res, prenorm=False))[0, -1, :]
            p   = torch.softmax(lg, dim=-1)
            top = p.topk(5)
            t5  = [(tokenizer.decode([i]).strip(), round(top.values[j].item(), 4))
                   for j, i in enumerate(top.indices.tolist())]
            tok, prob = t5[0]
            trace.append((f"L{loop_i+1}", tok, prob, t5))
            if tok == "<HALT>": break
        return trace


def load_v31():
    raw = MambaLMHeadModel.from_pretrained(
        "state-spaces/mamba2-130m", dtype=torch.bfloat16, device=DEVICE)
    nv = len(tokenizer); ov = raw.backbone.embedding.weight.shape[0]
    d  = raw.backbone.embedding.embedding_dim
    if nv > ov:
        e = nn.Embedding(nv, d, dtype=torch.bfloat16)
        nn.init.normal_(e.weight, std=0.02)
        e.weight.data[:ov] = raw.backbone.embedding.weight.data
        raw.backbone.embedding = e
        h = nn.Linear(d, nv, bias=False, dtype=torch.bfloat16)
        nn.init.normal_(h.weight, std=0.02)
        h.weight.data[:ov] = raw.lm_head.weight.data
        raw.lm_head = h
    for p in raw.parameters(): p.requires_grad = False
    raw.backbone.embedding.weight.requires_grad = True
    raw.lm_head.weight.requires_grad = True
    m = V31Model(raw).to(DEVICE)
    sd = torch.load(CKPT, map_location=DEVICE)
    sd = sd.get("model_state_dict", sd)
    m.load_state_dict(sd, strict=False)
    m.eval()
    print(f"  ✅ v31 loaded from {CKPT}")
    return m


SEP = "=" * 70

def probe(model, prompt, answer, label):
    inp = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    print(f"\n  {label}  |  Answer: {answer!r}")
    print(f"  {'─'*65}")
    with torch.no_grad():
        trace = model.run(inp, n=8)
    for lbl, tok, prob, t5 in trace:
        hit = " ✅" if tok.strip().lower() == answer.strip().lower() else (
              " <HALT>" if tok == "<HALT>" else "")
        print(f"    {lbl}  {tok!r:12s} p={prob:.4f}  top5={[(t[0],t[1]) for t in t5]}{hit}")


if __name__ == "__main__":
    print(SEP)
    print("  v31 FULL-VOCAB PROBE — no mask, raw 50k logits")
    print(SEP)
    model = load_v31()

    chains = [
        ("A = red. B = A. What is B?\nAnswer:",              "red",   "1-hop: A→B=red"),
        ("X = Apple. Y = X. Z = Y. What is Z?\nAnswer:",    "Apple", "3-hop: X→Y→Z=Apple"),
        ("A = moon. B = A. C = B. D = C. What is D?\nAnswer:", "moon", "4-hop: A→B→C→D=moon"),
        ("A = sky. B = A. C = B. D = C. E = D. What is E?\nAnswer:", "sky", "5-hop"),
    ]
    for prompt, answer, label in chains:
        probe(model, prompt, answer, label)

    print(f"\n{SEP}")
    print("  REALITY OVERRIDE — held-out prompts, no mask")
    print(SEP)
    overrides = [
        ("This river flows uphill. A ball thrown into it will travel which direction?\nAnswer:", "up"),
        ("Fire is icy cold. Bob touched fire. What did Bob feel?\nAnswer:", "cold"),
        ("Rocks are soft. Alice dropped a rock on her foot. Did it hurt?\nAnswer:", "no"),
    ]
    for prompt, answer in overrides:
        probe(model, prompt, answer, prompt.splitlines()[0][:55])

    print(f"\n{SEP}")
