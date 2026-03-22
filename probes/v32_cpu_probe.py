"""
v32_cpu_probe.py — CPU-mode inference test against mamba2_130m_v32_lifeline_best.pt
Training stays on GPU. This runs entirely on CPU — no conflict.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel, Mamba2

DEVICE       = "cpu"   # force CPU — training keeps GPU
CKPT         = "mamba2_130m_v32_lifeline_best.pt"
BASE_SPLIT   = 6
MAX_LOOPS    = 8
LORA_RANK    = 8
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


class V32Model(nn.Module):
    def __init__(self, bb, rank=8):
        super().__init__()
        self.backbone    = bb.backbone
        self.lm_head     = bb.lm_head
        self.all_layers  = nn.ModuleList(bb.backbone.layers)
        self.norm        = bb.backbone.norm_f
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
        self.lifeline_gate = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))

    def run(self, input_ids):
        """Full-vocab inference with prompt lifeline, no mask."""
        x, res = self.backbone.embedding(input_ids), None
        for layer in self.all_layers:
            x, res = layer(x, res)
        x_prompt = x.clone().detach()   # THE LIFELINE

        trace = []; last_answer = ""
        for loop_i in range(MAX_LOOPS):
            x   = x + self.lifeline_gate * x_prompt
            sv  = self.step_emb(torch.tensor(min(loop_i, MAX_LOOPS-1)))
            x   = x + sv
            for layer in self.all_layers[BASE_SPLIT:]:
                x, res = layer(x, res)
            x   = x + self.mamba2_core(x)
            x   = self.loop_norm(x)
            lg  = self.lm_head(self.norm(x, res, prenorm=False))[0, -1, :]
            p   = torch.softmax(lg.float(), dim=-1)   # float32 for CPU stability
            top = p.topk(5)
            tid = top.indices[0].item()
            tok = tokenizer.decode([tid]).strip()
            t5  = [(tokenizer.decode([top.indices[j].item()]).strip(),
                    round(top.values[j].item(), 4)) for j in range(5)]
            trace.append((f"L{loop_i+1}", tok, round(p[tid].item(), 4), t5))
            if tid == HALT_ID:
                trace[-1] = (f"L{loop_i+1}", "<HALT>", round(p[tid].item(), 4), t5)
                return trace, last_answer
            last_answer = tok
        return trace, last_answer


def load():
    print(f"  Loading mamba2-130m on CPU...", flush=True)
    raw = MambaLMHeadModel.from_pretrained(
        "state-spaces/mamba2-130m", dtype=torch.bfloat16, device="cpu")
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
    m = V32Model(raw).to("cpu")
    ckpt = torch.load(CKPT, map_location="cpu")
    sd   = ckpt.get("model_state_dict", ckpt)
    m.load_state_dict(sd, strict=False)
    m.eval()
    val_acc = ckpt.get("val_allloop_acc", "?")
    print(f"  ✅ Loaded {CKPT}  (val AllLoop: {val_acc:.1f}%)\n")
    return m


SEP = "=" * 68

def probe(model, prompt, expected, label):
    ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        trace, answer = model.run(ids)
    hit = "✅" if expected.lower() in answer.lower() else "❌"
    print(f"  {hit} {label}")
    for lbl, tok, prob, t5 in trace:
        mark = " ← ANSWER ✅" if tok.lower() == expected.lower() else (
               " <HALT>" if tok == "<HALT>" else "")
        print(f"     {lbl}  {tok!r:14s} p={prob:.4f}  top3={[(t[0],t[1]) for t in t5[:3]]}{mark}")
    print()


if __name__ == "__main__":
    print(SEP)
    print("  v32 CPU PROBE — prompt lifeline, no mask, full 50k vocab")
    print(f"  Checkpoint: {CKPT}")
    print(SEP + "\n")

    model = load()

    print("── CHAIN TRAVERSAL ─────────────────────────────────────────────────"); print()
    chains = [
        ("A = red. B = A. What is B?\nAnswer:",                    "red",         "1-hop: A→B=red"),
        ("X = Apple. Y = X. Z = Y. What is Z?\nAnswer:",          "Apple",       "3-hop: X→Y→Z=Apple"),
        ("A = moon. B = A. C = B. D = C. What is D?\nAnswer:",    "moon",        "4-hop: A→B→C→D=moon"),
        ("P = carburetor. Q = P. R = Q. What is R?\nAnswer:",     "carburetor",  "3-hop: random word"),
        ("M = photosynthesis. N = M. O = N. P = O. What is P?\nAnswer:", "photosynthesis", "4-hop: random word"),
        ("A = sky. B = A. C = B. D = C. E = D. What is E?\nAnswer:", "sky",      "5-hop: A→…→E=sky"),
    ]
    for prompt, expected, label in chains:
        probe(model, prompt, expected, label)

    print("── REALITY OVERRIDE ─────────────────────────────────────────────────"); print()
    overrides = [
        ("Fire is icy cold. Bob touched fire. What did Bob feel?\nAnswer:", "cold"),
        ("This river flows uphill. A ball dropped in will travel which direction?\nAnswer:", "up"),
        ("In this world gravity pushes things away. A ball dropped falls which direction?\nAnswer:", "up"),
    ]
    for prompt, expected in overrides:
        probe(model, prompt, expected, prompt.splitlines()[0][:55])

    print(SEP)
    print("  Training continues on GPU — checkpoint preserved.")
    print(SEP)
