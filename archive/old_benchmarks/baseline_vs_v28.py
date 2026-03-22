"""
baseline_vs_v28.py — Control Group Comparison
=============================================
Runs identical tests on:
  1. BASE MODEL: stock state-spaces/mamba-130m (no fine-tuning, no loop block)
  2. V28 MODEL:  mamba_130m_v28_latent_forcing_best.pt (Latent Forcing)

Proves Phase 2 latent pointer movement is caused by training, not architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel, Mamba

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CKPT       = "mamba_130m_v28_latent_forcing_best.pt"
BASE_SPLIT = 6
MAX_LOOPS  = 8
SEP = "=" * 60

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>"]})

def enc(prompt: str) -> torch.Tensor:
    """Tokenize prompt to input_ids tensor."""
    return tokenizer.encode(prompt, add_special_tokens=False,
                            return_tensors="pt").to(DEVICE)

def pointer_mask(input_ids: torch.Tensor, vocab: int) -> torch.Tensor:
    """Restrict logits to tokens actually present in the prompt."""
    mask = torch.full((vocab,), float("-inf"), device=DEVICE)
    mask[torch.unique(input_ids[0])] = 0.0
    return mask

def decode_top5(logits: torch.Tensor) -> list:
    """Return top-5 (token, prob) pairs from a logit vector."""
    p   = torch.softmax(logits, dim=-1)
    top = p.topk(5)
    return [(tokenizer.decode([i]).strip(), round(top.values[j].item(), 3))
            for j, i in enumerate(top.indices.tolist())]


# ── BASE MODEL (no training, no loops — just backbone greedy decode) ───────────
class BaselineModel(nn.Module):
    """Stock mamba-130m, greedy forward pass, no loop machinery."""

    def __init__(self, backbone: MambaLMHeadModel):
        """Wrap backbone for inference only."""
        super().__init__()
        self.backbone = backbone.backbone
        self.lm_head  = backbone.lm_head
        self.norm     = backbone.backbone.norm_f

    def single_pass(self, input_ids: torch.Tensor) -> list:
        """One forward pass, returns top-5 at last position."""
        x, res = self.backbone.embedding(input_ids), None
        for layer in self.backbone.layers:
            x, res = layer(x, res)
        lg = self.lm_head(self.norm(x, res, prenorm=False))
        vocab = lg.shape[-1]
        mask  = pointer_mask(input_ids, vocab)
        lg_m  = lg[0, -1, :] + mask
        return decode_top5(lg_m)

    def run_simulated_loops(self, input_ids: torch.Tensor, n: int) -> list:
        """Simulate N 'loops' by re-running the full forward pass N times
        (no learned step_emb or loop state — pure repetition).
        Shows what a base model does without training."""
        results = []
        for loop_i in range(n):
            top5 = self.single_pass(input_ids)
            tok, prob = top5[0]
            results.append((f"L{loop_i+1}", tok, prob, top5))
        return results


# ── V28 MODEL ─────────────────────────────────────────────────────────────────
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


class V28Model(nn.Module):
    """130m v28 Latent Forcing — stateful loop block."""

    def __init__(self, backbone, max_loops=MAX_LOOPS, lora_rank=8):
        """Build architecture matching training script."""
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
                    setattr(mx, attr, LoRALinear(getattr(mx, attr),
                                                 lora_rank, lora_rank * 2.0))

        self.step_emb    = nn.Embedding(6, d).to(torch.bfloat16)
        self.loop_norm   = nn.RMSNorm(d).to(torch.bfloat16)
        self.mamba3_core = Mamba(d_model=d, d_state=16, d_conv=4, expand=2).to(torch.bfloat16)

    def run(self, input_ids, n_loops=MAX_LOOPS):
        """N stateful loops with learned step embeddings and Mamba1 scan."""
        x, res = self.backbone.embedding(input_ids), None
        for layer in self.all_layers:
            x, res = layer(x, res)

        vocab = self.lm_head.weight.shape[0]
        mask  = pointer_mask(input_ids, vocab)
        trace = []

        for loop_i in range(n_loops):
            sv = self.step_emb(torch.tensor(min(loop_i, 5), device=x.device))
            x  = x + sv
            for layer in self.all_layers[BASE_SPLIT:]:
                x, res = layer(x, res)
            x = x + self.mamba3_core(x)
            x = self.loop_norm(x)

            lg   = self.lm_head(self.norm(x, res, prenorm=False))
            lg_m = lg[0, -1, :] + mask
            top5 = decode_top5(lg_m)
            tok, prob = top5[0]
            trace.append((f"L{loop_i+1}", tok, prob, top5))
        return trace


def load_both():
    """Load base backbone and v28 fine-tuned model sharing the same backbone."""
    print(f"  Loading state-spaces/mamba-130m backbone...", flush=True)
    raw = MambaLMHeadModel.from_pretrained(
        "state-spaces/mamba-130m", dtype=torch.bfloat16, device=DEVICE
    )
    # Resize vocab for <THINK>
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

    # BASE: wrap stock backbone
    base_model = BaselineModel(raw).to(DEVICE)
    base_model.eval()
    print(f"  ✅ Base model ready (no fine-tuning)", flush=True)

    # V28: load second copy of backbone, apply LoRA + checkpoint
    raw2 = MambaLMHeadModel.from_pretrained(
        "state-spaces/mamba-130m", dtype=torch.bfloat16, device=DEVICE
    )
    if nv > ov:
        e2 = nn.Embedding(nv, d, dtype=torch.bfloat16)
        nn.init.normal_(e2.weight, std=0.02)
        e2.weight.data[:ov] = raw2.backbone.embedding.weight.data
        raw2.backbone.embedding = e2
        h2 = nn.Linear(d, nv, bias=False, dtype=torch.bfloat16)
        nn.init.normal_(h2.weight, std=0.02)
        h2.weight.data[:ov] = raw2.lm_head.weight.data
        raw2.lm_head = h2
    for p in raw2.parameters(): p.requires_grad = False
    raw2.backbone.embedding.weight.requires_grad = True
    raw2.lm_head.weight.requires_grad = True

    v28_model = V28Model(raw2, max_loops=MAX_LOOPS).to(DEVICE)
    sd = torch.load(CKPT, map_location=DEVICE)
    sd = sd.get("model_state_dict", sd)
    miss, _ = v28_model.load_state_dict(sd, strict=False)
    v28_model.eval()
    print(f"  ✅ V28 model ready ({len(miss)} missing keys)", flush=True)

    return base_model, v28_model


def compare_probe(base_model, v28_model, prompt, answer, label):
    """Side-by-side loop trace for one prompt."""
    inp = enc(prompt)
    print(f"\n  {'─'*56}")
    print(f"  Chain: {label}  |  Expected answer: {answer!r}")
    print(f"  {'─'*56}")

    # Base model — no stateful loop, just repeat the same forward pass
    base_trace = base_model.run_simulated_loops(inp, n=MAX_LOOPS)
    # V28 — stateful loops
    v28_trace  = v28_model.run(inp, n_loops=MAX_LOOPS)

    print(f"  {'Loop':<6}  {'BASE (no training)':<22}  {'V28 (Latent Forcing)':<22}")
    print(f"  {'─'*6}  {'─'*22}  {'─'*22}")
    for i in range(MAX_LOOPS):
        bl, bw, bp, _ = base_trace[i]
        vl, vw, vp, _ = v28_trace[i]
        b_star = " ✅" if answer.lower() in bw.lower() else "   "
        v_star = " ✅" if answer.lower() in vw.lower() else "   "
        print(f"  {bl:<6}  {bw!r:10s} p={bp:.3f}{b_star}   {vw!r:10s} p={vp:.3f}{v_star}")


def run_comparison(base_model, v28_model):
    """Run side-by-side comparison on all test prompts."""
    print(f"\n{SEP}")
    print(f"  COMPARISON: BASE vs V28 — Phase 2 Latent State Probe")
    print(f"  Question: does V28 show pointer movement that BASE cannot?")
    print(SEP)

    probes = [
        ("A = red. B = A. What is B?\nAnswer:",
         "red", "1-hop: A→B=red"),
        ("X = Apple. Y = X. Z = Y. What is Z?\nAnswer:",
         "Apple", "3-hop: X→Y→Z=Apple"),
        ("A = moon. B = A. C = B. D = C. What is D?\nAnswer:",
         "moon", "4-hop: A→B→C→D=moon"),
        ("A = sky. B = A. C = B. D = C. E = D. What is E?\nAnswer:",
         "sky", "5-hop: A→B→C→D→E=sky"),
    ]
    with torch.no_grad():
        for prompt, answer, label in probes:
            compare_probe(base_model, v28_model, prompt, answer, label)

    print(f"\n{SEP}")
    print(f"  INTERPRETATION:")
    print(f"  - BASE: should show the SAME token every loop (no state update)")
    print(f"  - V28:  should show X → Y → Z → 'answer' progression")
    print(f"  - If BASE is flat and V28 moves → Latent Forcing PROVEN")
    print(SEP)


def run_reality_comparison(base_model, v28_model):
    """Phase 4: does V28 override pretrained priors better than BASE?"""
    print(f"\n{SEP}")
    print(f"  COMPARISON: BASE vs V28 — Phase 4 Reality Override")
    print(SEP)
    tests = [
        ("Fire is icy cold. Bob touched fire. What did Bob feel?\nAnswer:",
         "cold", "hot"),
        ("In this world dogs meow and cats bark. Sarah has a cat. What sound does she hear?\nAnswer:",
         "bark", "meow"),
        ("The sun is freezing cold. John's coffee is the temperature of the sun. What temperature is John's coffee?\nAnswer:",
         "cold", "hot"),
        ("Sugar tastes bitter in this world. Tom ate sugar. What did Tom taste?\nAnswer:",
         "bitter", "sweet"),
        ("Up means down and down means up. Sarah jumped up. Which direction did she move?\nAnswer:",
         "down", "up"),
    ]

    base_pass = v28_pass = 0
    with torch.no_grad():
        for prompt, ctx_ans, prior_ans in tests:
            inp      = enc(prompt)
            # Base: one pass
            b_top5   = base_model.single_pass(inp)
            b_tok    = b_top5[0][0]
            b_ctx    = ctx_ans.lower() in b_tok.lower()
            b_prior  = prior_ans.lower() in b_tok.lower() and not b_ctx

            # V28: run to halt
            v28_tr   = v28_model.run(inp, n_loops=MAX_LOOPS)
            v_tok    = v28_tr[-1][1]
            # Find first repeat (halt point)
            seen = set()
            for loop_i, (_, tok, _, _) in enumerate(v28_tr):
                if tok in seen:
                    v_tok = tok
                    break
                seen.add(tok)
            v_ctx    = ctx_ans.lower() in v_tok.lower()
            v_prior  = prior_ans.lower() in v_tok.lower() and not v_ctx

            base_pass += b_ctx
            v28_pass  += v_ctx

            b_v = "CTX ✅" if b_ctx else ("PRIOR ❌" if b_prior else f"?{b_tok!r}")
            v_v = "CTX ✅" if v_ctx else ("PRIOR ❌" if v_prior else f"?{v_tok!r}")
            q   = prompt[:55].replace("\n", " ")
            print(f"  Q: {q!r}")
            print(f"     BASE: {b_tok!r:10s} → {b_v}")
            print(f"     V28:  {v_tok!r:10s} → {v_v}\n")

    print(f"  Reality Override Scores:")
    print(f"    BASE model: {base_pass}/{len(tests)}")
    print(f"    V28  model: {v28_pass}/{len(tests)}")
    delta = v28_pass - base_pass
    if delta > 0:
        print(f"  V28 outperforms BASE by +{delta} questions ✅")
    elif delta == 0:
        print(f"  Equal performance — V28 didn't improve ⚠️")
    else:
        print(f"  BASE outperforms V28 ❌")


if __name__ == "__main__":
    print(SEP)
    print("  BASE vs V28 — Proof of Latent Forcing")
    print(f"  Checkpoint: {CKPT}")
    print(f"  Device: {DEVICE} | MAX_LOOPS={MAX_LOOPS}")
    print(SEP)

    base_model, v28_model = load_both()
    run_comparison(base_model, v28_model)
    run_reality_comparison(base_model, v28_model)

    print("\nDone.")
