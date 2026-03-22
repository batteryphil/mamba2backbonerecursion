"""
fixed_loop_probe.py — No halting. Exactly 5 loops. Raw trace.
=============================================================
Runs BASE, NoLF, and v28 for a hardcoded 5 loops each.
Zero halt logic. Zero interruption. Just the raw numbers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel, Mamba

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_V28   = "mamba_130m_v28_latent_forcing_best.pt"
CKPT_NOLF  = "mamba_130m_v28_nolf_best.pt"
BASE_SPLIT = 6
N_LOOPS    = 5          # HARDCODED — exactly 5 loops, no halt whatsoever
SEP = "=" * 68

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>"]})

def enc(p):
    return tokenizer.encode(p, add_special_tokens=False, return_tensors="pt").to(DEVICE)

def pmask(ids, vocab):
    m = torch.full((vocab,), float("-inf"), device=DEVICE)
    m[torch.unique(ids[0])] = 0.0
    return m

def top3(lg):
    p = torch.softmax(lg, dim=-1)
    t = p.topk(3)
    return [(tokenizer.decode([i]).strip(), round(t.values[j].item(), 4))
            for j, i in enumerate(t.indices.tolist())]


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


class RecurrentModel(nn.Module):
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
            for attr in ("in_proj","out_proj"):
                if hasattr(layer.mixer, attr):
                    setattr(layer.mixer, attr,
                            LoRALinear(getattr(layer.mixer, attr), rank, rank*2.0))
        self.step_emb    = nn.Embedding(6, d).to(torch.bfloat16)
        self.loop_norm   = nn.RMSNorm(d).to(torch.bfloat16)
        self.mamba3_core = Mamba(d_model=d, d_state=16, d_conv=4, expand=2).to(torch.bfloat16)

    def run_fixed(self, input_ids, n=N_LOOPS):
        """Run EXACTLY n loops. No halt. No questions asked."""
        x, res = self.backbone.embedding(input_ids), None
        for layer in self.all_layers:
            x, res = layer(x, res)
        vocab = self.lm_head.weight.shape[0]
        mask  = pmask(input_ids, vocab)
        trace = []
        for loop_i in range(n):
            sv = self.step_emb(torch.tensor(min(loop_i, 5), device=x.device))
            x  = x + sv
            for layer in self.all_layers[BASE_SPLIT:]:
                x, res = layer(x, res)
            x = x + self.mamba3_core(x)
            x = self.loop_norm(x)
            lg    = self.lm_head(self.norm(x, res, prenorm=False))[0, -1, :]
            lg_m  = lg + mask
            t3    = top3(lg_m)
            tok, prob = t3[0]
            trace.append((f"L{loop_i+1}", tok, prob, t3))
        return trace


class BaselineModel(nn.Module):
    def __init__(self, bb):
        super().__init__()
        self.backbone = bb.backbone
        self.lm_head  = bb.lm_head
        self.norm     = bb.backbone.norm_f
    def run_fixed(self, input_ids, n=N_LOOPS):
        """Frozen forward pass repeated n times — deterministic floor."""
        trace = []
        for loop_i in range(n):
            x, res = self.backbone.embedding(input_ids), None
            for layer in self.backbone.layers:
                x, res = layer(x, res)
            vocab = self.lm_head.weight.shape[0]
            lg_m  = self.lm_head(self.norm(x, res, prenorm=False))[0, -1, :] + pmask(input_ids, vocab)
            t3    = top3(lg_m)
            tok, prob = t3[0]
            trace.append((f"L{loop_i+1}", tok, prob, t3))
        return trace


def prep_backbone():
    raw = MambaLMHeadModel.from_pretrained(
        "state-spaces/mamba-130m", dtype=torch.bfloat16, device=DEVICE)
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
    return raw


def load(ckpt_path, backbone):
    backbone.backbone.embedding.weight.requires_grad = True
    backbone.lm_head.weight.requires_grad = True
    m  = RecurrentModel(backbone).to(DEVICE)
    sd = torch.load(ckpt_path, map_location=DEVICE)
    sd = sd.get("model_state_dict", sd)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


def print_trace(label, trace, answer):
    print(f"\n  ── {label} ──────────────────────────────────────────")
    for loop_label, tok, prob, t3 in trace:
        hit = " ✅" if answer.lower() in tok.lower() else ""
        print(f"    {loop_label}  {tok!r:12s}  p={prob:.4f}  "
              f"top3={[(t[0], t[1]) for t in t3]}{hit}")


def probe(base_m, nolf_m, v28_m, prompt, answer, label):
    inp = enc(prompt)
    print(f"\n{SEP}")
    print(f"  CHAIN: {label}   |   Answer: {answer!r}")
    print(f"  Loops: {N_LOOPS} (hardcoded, zero halt)")
    print(SEP)
    with torch.no_grad():
        print_trace("BASE  (no training, frozen repeat)", base_m.run_fixed(inp), answer)
        print_trace("NoLF  (same arch, final-only loss)", nolf_m.run_fixed(inp), answer)
        print_trace("v28   (Latent Forcing, per-loop loss)", v28_m.run_fixed(inp), answer)


if __name__ == "__main__":
    print(SEP)
    print(f"  FIXED-LOOP PROBE — {N_LOOPS} loops, zero halt, raw trace")
    print(f"  Models: BASE | NoLF | v28 Latent Forcing")
    print(SEP)

    print("\n  Loading models...", flush=True)
    raw0  = prep_backbone()
    base_m = BaselineModel(raw0).to(DEVICE); base_m.eval()

    raw1   = prep_backbone()
    nolf_m = load(CKPT_NOLF, raw1)

    raw2  = prep_backbone()
    v28_m = load(CKPT_V28, raw2)
    print("  ✅ All models loaded\n")

    chains = [
        ("A = red. B = A. What is B?\nAnswer:",              "red",   "1-hop: A→B=red"),
        ("X = Apple. Y = X. Z = Y. What is Z?\nAnswer:",    "Apple", "3-hop: X→Y→Z=Apple"),
        ("A = moon. B = A. C = B. D = C. What is D?\nAnswer:", "moon", "4-hop: A→B→C→D=moon"),
        ("A = sky. B = A. C = B. D = C. E = D. What is E?\nAnswer:", "sky", "5-hop: A→B→C→D→E=sky"),
    ]

    for prompt, answer, label in chains:
        probe(base_m, nolf_m, v28_m, prompt, answer, label)

    print(f"\n{SEP}")
    print("  Raw trace complete. No halt logic applied.")
    print(f"  If v28 hits the answer at L4 and holds → Latent Forcing proven.")
    print(f"  If NoLF skips straight to answer at L1 → it learned a shortcut, not the algorithm.")
    print(SEP)
