"""
ablation_comparison.py — 3-Way Proof: BASE vs NoLF vs v28
===========================================================
Compares three models on the same prompts:

  BASE  : stock mamba-130m, no training, no loops
  NoLF  : same arch as v28, trained on FINAL ANSWER ONLY (no per-loop supervision)
  v28   : Latent Forcing — supervised at EVERY loop against chain_targets[i]

If Latent Forcing is real:
  BASE  → same token every loop (deterministic function, no state)
  NoLF  → may converge to correct final answer but WITHOUT intermediate pointer movement
  v28   → shows step-by-step pointer traversal (X → Y → Apple) per loop

This is the scientifically valid ablation that previous comparison lacked.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel, Mamba
import os

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_V28   = "mamba_130m_v28_latent_forcing_best.pt"
CKPT_NOLF  = "mamba_130m_v28_nolf_best.pt"
BASE_SPLIT = 6
MAX_LOOPS  = 8
SEP        = "=" * 60

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>"]})


def enc(prompt: str) -> torch.Tensor:
    """Tokenize prompt to GPU tensor."""
    return tokenizer.encode(prompt, add_special_tokens=False,
                            return_tensors="pt").to(DEVICE)


def pmask(input_ids: torch.Tensor, vocab: int) -> torch.Tensor:
    """Pointer mask — restrict logits to tokens in prompt."""
    mask = torch.full((vocab,), float("-inf"), device=DEVICE)
    mask[torch.unique(input_ids[0])] = 0.0
    return mask


def top5_str(logits: torch.Tensor) -> list[tuple]:
    """Return top-5 (token, prob) pairs."""
    p   = torch.softmax(logits, dim=-1)
    top = p.topk(5)
    return [(tokenizer.decode([i]).strip(), round(top.values[j].item(), 3))
            for j, i in enumerate(top.indices.tolist())]


# ── LoRA linear (shared by both NoLF and v28) ─────────────────────────────────
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
    def weight(self):
        return self.base_weight + self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


# ── Shared inference architecture (used by both NoLF and v28) ─────────────────
class RecurrentModel(nn.Module):
    """Shared inference model for both v28 and NoLF — same arch."""

    def __init__(self, backbone, max_loops=MAX_LOOPS, lora_rank=8):
        super().__init__()
        self.MAX_LOOPS  = max_loops
        self.backbone   = backbone.backbone
        self.lm_head    = backbone.lm_head
        self.all_layers = nn.ModuleList(backbone.backbone.layers)
        self.norm       = backbone.backbone.norm_f
        d = backbone.backbone.embedding.embedding_dim

        for layer in self.all_layers[:BASE_SPLIT]:
            for p in layer.parameters():
                p.requires_grad = False

        for layer in self.all_layers[BASE_SPLIT:]:
            mx = layer.mixer
            for attr in ("in_proj", "out_proj"):
                if hasattr(mx, attr):
                    setattr(mx, attr, LoRALinear(getattr(mx, attr), lora_rank, lora_rank * 2.0))

        self.step_emb    = nn.Embedding(6, d).to(torch.bfloat16)
        self.loop_norm   = nn.RMSNorm(d).to(torch.bfloat16)
        self.mamba3_core = Mamba(d_model=d, d_state=16, d_conv=4, expand=2).to(torch.bfloat16)

    def run(self, input_ids: torch.Tensor) -> list:
        """Stateful loop inference with smart halt.

        Halt rule: stop on token repeat ONLY if the token is NOT a single
        uppercase letter. Single uppercase letters (A-Z) are variable pointers
        mid-traversal — halting on them prematurely cuts the chain before it
        resolves to the actual answer word.
        """
        x, res = self.backbone.embedding(input_ids), None
        for layer in self.all_layers:
            x, res = layer(x, res)

        vocab = self.lm_head.weight.shape[0]
        mask  = pmask(input_ids, vocab)
        trace = []
        prev  = None

        for loop_i in range(self.MAX_LOOPS):
            sv = self.step_emb(torch.tensor(min(loop_i, 5), device=x.device))
            x  = x + sv
            for layer in self.all_layers[BASE_SPLIT:]:
                x, res = layer(x, res)
            x = x + self.mamba3_core(x)
            x = self.loop_norm(x)

            lg  = self.lm_head(self.norm(x, res, prenorm=False))
            lg[0, -1, :] += mask
            t5  = top5_str(lg[0, -1, :])
            tok, prob = t5[0]
            trace.append((f"L{loop_i+1}", tok, prob, t5))

            # Smart halt: only stop on repeat if it's a resolved word (not a
            # single uppercase variable letter like A, B, X, Y, Z)
            is_var_letter = len(tok) == 1 and tok.isupper()
            if prev is not None and tok == prev and not is_var_letter:
                return trace
            prev = tok
        return trace


class BaselineModel(nn.Module):
    """Frozen backbone — no loops, deterministic repeat baseline."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.backbone
        self.lm_head  = backbone.lm_head
        self.norm     = backbone.backbone.norm_f

    def run(self, input_ids: torch.Tensor, n: int = MAX_LOOPS) -> list:
        """Repeat same forward pass N times — no state update."""
        trace = []
        for loop_i in range(n):
            x, res = self.backbone.embedding(input_ids), None
            for layer in self.backbone.layers:
                x, res = layer(x, res)
            vocab = self.lm_head.weight.shape[0]
            lg_m  = self.lm_head(self.norm(x, res, prenorm=False))[0, -1, :] + pmask(input_ids, vocab)
            t5    = top5_str(lg_m)
            tok, prob = t5[0]
            trace.append((f"L{loop_i+1}", tok, prob, t5))
        return trace


def load_all():
    """Load backbone, build BASE, NoLF, and v28 models."""
    print("  Loading mamba-130m backbone x2...", flush=True)

    def prep_backbone():
        raw = MambaLMHeadModel.from_pretrained(
            "state-spaces/mamba-130m", dtype=torch.bfloat16, device=DEVICE
        )
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

    # BASE
    raw0   = prep_backbone()
    base_m = BaselineModel(raw0).to(DEVICE)
    base_m.eval()
    print("  ✅ BASE ready", flush=True)

    # NoLF
    nolf_m = None
    if os.path.exists(CKPT_NOLF):
        raw1   = prep_backbone()
        raw1.backbone.embedding.weight.requires_grad = True
        raw1.lm_head.weight.requires_grad            = True
        nolf_m = RecurrentModel(raw1, max_loops=MAX_LOOPS).to(DEVICE)
        sd     = torch.load(CKPT_NOLF, map_location=DEVICE)
        sd     = sd.get("model_state_dict", sd)
        nolf_m.load_state_dict(sd, strict=False)
        nolf_m.eval()
        print(f"  ✅ NoLF ready ({CKPT_NOLF})", flush=True)
    else:
        print(f"  ⏳ NoLF checkpoint not found ({CKPT_NOLF}) — run training first", flush=True)

    # v28
    v28_m = None
    if os.path.exists(CKPT_V28):
        raw2  = prep_backbone()
        raw2.backbone.embedding.weight.requires_grad = True
        raw2.lm_head.weight.requires_grad            = True
        v28_m = RecurrentModel(raw2, max_loops=MAX_LOOPS).to(DEVICE)
        sd    = torch.load(CKPT_V28, map_location=DEVICE)
        sd    = sd.get("model_state_dict", sd)
        v28_m.load_state_dict(sd, strict=False)
        v28_m.eval()
        print(f"  ✅ v28  ready ({CKPT_V28})", flush=True)
    else:
        print(f"  ⏳ v28  checkpoint not found ({CKPT_V28})", flush=True)

    return base_m, nolf_m, v28_m


def compare_probe(base_m, nolf_m, v28_m, prompt, answer, label):
    """Side-by-side loop trace for all three models."""
    inp = enc(prompt)
    print(f"\n  {'─'*64}")
    print(f"  Chain: {label}  |  Answer: {answer!r}")
    print(f"  {'─'*64}")

    base_tr = base_m.run(inp, n=MAX_LOOPS)
    nolf_tr = nolf_m.run(inp) if nolf_m else None
    v28_tr  = v28_m.run(inp)  if v28_m  else None

    hdr_base = "BASE (no train)"
    hdr_nolf = "NoLF (final-only)"
    hdr_v28  = "v28 (Latent Forcing)"
    print(f"  {'Loop':<6}  {hdr_base:<20}  {hdr_nolf:<22}  {hdr_v28:<22}")
    print(f"  {'─'*6}  {'─'*20}  {'─'*22}  {'─'*22}")

    for i in range(MAX_LOOPS):
        bl, bw, bp, _ = base_tr[i]
        nw = np = "—"
        vw = vp = "—"
        if nolf_tr and i < len(nolf_tr):
            _, nw, np, _ = nolf_tr[i]
        if v28_tr and i < len(v28_tr):
            _, vw, vp, _ = v28_tr[i]
        b_star  = " ✅" if isinstance(bw, str) and answer.lower() in bw.lower() else "   "
        n_star  = " ✅" if isinstance(nw, str) and answer.lower() in nw.lower() else "   "
        v_star  = " ✅" if isinstance(vw, str) and answer.lower() in vw.lower() else "   "
        print(f"  {bl:<6}  "
              f"{str(bw)!r:10s} p={str(bp):<6}{b_star}  "
              f"{str(nw)!r:10s} p={str(np):<6}{n_star}  "
              f"{str(vw)!r:10s} p={str(vp):<6}{v_star}")


def reality_override_3way(base_m, nolf_m, v28_m):
    """Phase 4 — 3-way reality override test with HELD-OUT (novel) prompts."""
    print(f"\n{SEP}")
    print(f"  REALITY OVERRIDE — 3-way (HELD-OUT prompts, not in training data)")
    print(SEP)

    # IMPORTANT: these prompts were NOT in the training data
    tests = [
        ("Rocks are soft and feathers are hard. Alice dropped a rock on her foot. Did it hurt?\nAnswer:",
         "yes", "no"),
        ("In this country, red lights mean go and green lights mean stop. The light is red. Should the car move?\nAnswer:",
         "yes", "no"),
        ("This river flows uphill. A ball thrown into it will travel which direction?\nAnswer:",
         "up", "down"),
        ("Salt is sweet and sugar is salty in this universe. Tom added sugar to his coffee. It tasted?\nAnswer:",
         "salty", "sweet"),
    ]

    base_pass = nolf_pass = v28_pass = 0
    with torch.no_grad():
        for prompt, ctx_ans, prior_ans in tests:
            inp   = enc(prompt)
            b_tok = base_m.run(inp, n=2)[0][1]
            n_tok = nolf_m.run(inp)[-1][1] if nolf_m else "—"
            v_tok = v28_m.run(inp)[-1][1]  if v28_m  else "—"

            def score(tok, ctx, prior):
                if ctx.lower() in tok.lower():    return "CTX ✅", True
                if prior.lower() in tok.lower():  return "PRIOR ❌", False
                return f"? ({tok!r})", False

            bs, bp = score(b_tok, ctx_ans, prior_ans)
            ns, np = score(n_tok, ctx_ans, prior_ans)
            vs, vp = score(v_tok, ctx_ans, prior_ans)
            base_pass += bp; nolf_pass += np; v28_pass += vp

            q = prompt.splitlines()[0][:60]
            print(f"  Q: {q!r}")
            print(f"     BASE  {b_tok!r:10s} → {bs}")
            print(f"     NoLF  {n_tok!r:10s} → {ns}")
            print(f"     v28   {v_tok!r:10s} → {vs}\n")

    print(f"  SCORES | BASE: {base_pass}/4 | NoLF: {nolf_pass}/4 | v28: {v28_pass}/4")


if __name__ == "__main__":
    print(SEP)
    print("  3-WAY ABLATION: BASE vs NoLF vs v28 Latent Forcing")
    print(f"  BASE:  stock mamba-130m, no training")
    print(f"  NoLF:  same arch, final-answer-only supervision")
    print(f"  v28:   Latent Forcing — per-loop supervision")
    print(SEP)

    base_m, nolf_m, v28_m = load_all()

    probes = [
        ("A = red. B = A. What is B?\nAnswer:",             "red",   "1-hop: A→B=red"),
        ("X = Apple. Y = X. Z = Y. What is Z?\nAnswer:",   "Apple", "3-hop: X→Y→Z=Apple"),
        ("A = moon. B = A. C = B. D = C. What is D?\nAnswer:", "moon", "4-hop: A→B→C→D=moon"),
    ]

    print(f"\n{SEP}")
    print(f"  PHASE 2 LATENT PROBE — Per-loop token comparison")
    print(f"  Key: does per-loop supervision (v28) cause pointer movement that NoLF cannot?")
    print(SEP)

    with torch.no_grad():
        for prompt, answer, label in probes:
            compare_probe(base_m, nolf_m, v28_m, prompt, answer, label)

        reality_override_3way(base_m, nolf_m, v28_m)

    print("\n  INTERPRETATION:")
    print("  If BASE=flat, NoLF=flat, v28=pointer movement → Latent Forcing PROVEN")
    print("  If BASE=flat, NoLF=pointer, v28=pointer → architecture does it (claim collapses)")
    print("  If all flat → neither method works")
    print(f"\n{SEP}")
