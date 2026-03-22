"""
diagnostic_suite_v28.py — 4-Phase Diagnostic for 130m v28 (Latent Forcing)

Loads mamba_130m_v28_latent_forcing_best.pt and runs:
  Phase 1: Over-Rev (OOD hop lengths 5, 7, 10)
  Phase 2: Latent State Probe (per-loop token inspection)
  Phase 3: Dynamic Halt (does compute scale with hop count?)
  Phase 4: Reality Override (free-form context beats prior)

Key difference from v26 diagnostic:
  - Uses mamba_ssm.Mamba (Mamba1) as loop block, NOT the MIMO Phase Rotator
  - Halt condition: token repeats (not THINK detection)
  - Free-form answer format (no ABCD) in Phase 4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel, Mamba
import random

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
CKPT      = "mamba_130m_v28_latent_forcing_best.pt"
MAX_LOOPS = 15   # allow over-rev past training MAX_LOOPS=6
BASE_SPLIT = 6
SEP = "=" * 60

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>"]})


# ── Reconstruct the v28 model architecture ────────────────────────────────────
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


class RecursiveMamba130m_v28(nn.Module):
    """Inference-only v28 model. Same arch as training script."""

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
                    setattr(mx, attr, LoRALinear(getattr(mx, attr),
                                                 lora_rank, lora_rank * 2.0))

        self.step_emb    = nn.Embedding(6, d).to(torch.bfloat16)  # trained on MAX_LOOPS=6
        self.loop_norm   = nn.RMSNorm(d).to(torch.bfloat16)
        self.mamba3_core = Mamba(d_model=d, d_state=16, d_conv=4, expand=2).to(torch.bfloat16)

    def run(self, input_ids):
        """Run inference, return (n_loops_used, trace).
        
        Halt: token repeats (same as training halt in v28).
        Loops can exceed training MAX_LOOPS=6 for OOD testing.
        """
        x, res = self.backbone.embedding(input_ids), None
        for layer in self.all_layers:
            x, res = layer(x, res)

        vocab = self.lm_head.weight.shape[0]
        mask  = torch.full((vocab,), float("-inf"), device=x.device)
        mask[torch.unique(input_ids[0])] = 0.0

        trace    = []
        prev_tok = None

        for loop_i in range(self.MAX_LOOPS):
            sv = self.step_emb(
                torch.tensor(min(loop_i, 5), device=x.device)  # clip to trained range
            )
            x = x + sv
            for layer in self.all_layers[BASE_SPLIT:]:
                x, res = layer(x, res)
            x = x + self.mamba3_core(x)
            x = self.loop_norm(x)

            lg  = self.lm_head(self.norm(x, res, prenorm=False))
            lg[0, -1, :] += mask
            p   = torch.softmax(lg[0, -1, :], dim=-1)
            tid = p.argmax().item()
            tok = tokenizer.decode([tid]).strip()
            p5_ids = p.topk(5).indices.tolist()
            top5   = [(tokenizer.decode([t]).strip(), round(p[t].item(), 3)) for t in p5_ids]
            trace.append((f"L{loop_i+1}", tok, round(p.max().item(), 3), top5))

            if prev_tok is not None and tid == prev_tok:
                return loop_i + 1, trace
            prev_tok = tid

        return self.MAX_LOOPS, trace


def load_model():
    """Load v28 130m from checkpoint."""
    print(f"  Loading state-spaces/mamba-130m (bfloat16)...", flush=True)
    base = MambaLMHeadModel.from_pretrained(
        "state-spaces/mamba-130m", dtype=torch.bfloat16, device=DEVICE
    )
    nv = len(tokenizer)
    ov = base.backbone.embedding.weight.shape[0]
    d  = base.backbone.embedding.embedding_dim
    if nv > ov:
        e = nn.Embedding(nv, d, dtype=torch.bfloat16)
        nn.init.normal_(e.weight, std=0.02)
        e.weight.data[:ov] = base.backbone.embedding.weight.data
        base.backbone.embedding = e
        h = nn.Linear(d, nv, bias=False, dtype=torch.bfloat16)
        nn.init.normal_(h.weight, std=0.02)
        h.weight.data[:ov] = base.lm_head.weight.data
        base.lm_head = h

    for p in base.parameters():
        p.requires_grad = False
    base.backbone.embedding.weight.requires_grad = True
    base.lm_head.weight.requires_grad = True

    model = RecursiveMamba130m_v28(base, max_loops=MAX_LOOPS).to(DEVICE)
    ckpt  = torch.load(CKPT, map_location=DEVICE)
    sd    = ckpt.get("model_state_dict", ckpt)
    miss, unex = model.load_state_dict(sd, strict=False)
    print(f"  Loaded {CKPT} | {len(miss)} missing, {len(unex)} unexpected", flush=True)
    model.eval()
    return model


def make_chain(n_hops, seed=42):
    """Generate n-hop chain prompt and expected answer."""
    rng    = random.Random(seed)
    words  = ["apple","blue","seven","moon","red","cat","sky","green",
              "dog","cold","fire","tree","star","wave","rock","sand","snow","wind"]
    answer = rng.choice(words)
    letters = [chr(65 + i) for i in range(n_hops + 1)]
    lines   = [f"{letters[0]} = {answer}"]
    for i in range(1, n_hops + 1):
        lines.append(f"{letters[i]} = {letters[i-1]}")
    prompt = ". ".join(lines) + f". What is {letters[-1]}?\nAnswer:"
    return prompt, answer

def enc(prompt):
    return tokenizer.encode(prompt, add_special_tokens=False,
                            return_tensors="pt").to(DEVICE)


# ── PHASE 1: Over-Rev Extrapolation ──────────────────────────────────────────
def phase1(model):
    print(f"\n{'─'*60}")
    print(f"  PHASE 1 — OVER-REV EXTRAPOLATION")
    print(f"  Trained on 1-3 hops, testing 5 / 7 / 10 hops, MAX_LOOPS={MAX_LOOPS}")
    print(f"{'─'*60}")
    results = []
    for n_hops, n_seeds in [(5, 3), (7, 5), (10, 7)]:
        correct = 0
        for seed in range(n_seeds):
            prompt, answer = make_chain(n_hops, seed=seed * 17 + n_hops)
            with torch.no_grad():
                loops, trace = model.run(enc(prompt))
            got  = trace[-1][1]
            ok   = answer.lower() in got.lower()
            correct += ok
            flag  = "✅" if ok else "❌"
            chain = " → ".join(f"{t[0]}:{t[1]}" for t in trace)
            print(f"  {flag} [{n_hops}-hop] want={answer!r:8s} got={got!r:8s} ({loops} loops)")
            print(f"         {chain}")
        results.append((n_hops, correct, n_seeds))
        print(f"  ── {n_hops}-hop: {correct}/{n_seeds}\n")

    print(f"  PHASE 1 VERDICT:")
    total_pass = 0
    for n, c, t in results:
        v = "PASS ✅" if c == t else ("PARTIAL ⚠️" if c > 0 else "FAIL ❌")
        if c == t: total_pass += 1
        print(f"    {n}-hop: {c}/{t}  {v}")
    return total_pass


# ── PHASE 2: Latent State Probe ───────────────────────────────────────────────
def phase2(model):
    print(f"\n{'─'*60}")
    print(f"  PHASE 2 — LATENT STATE PROBE")
    print(f"  3-hop chain: X=Apple. Y=X. Z=Y. What is Z?")
    print(f"  Key question: do intermediate loops show X→Y→Apple (reasoning)?")
    print(f"  Or do all loops show <THINK> until the last? (delay gate)")
    print(f"{'─'*60}")
    prompt = "X = Apple. Y = X. Z = Y. What is Z?\nAnswer:"
    inp = enc(prompt)

    with torch.no_grad():
        x, res = model.backbone.embedding(inp), None
        for layer in model.all_layers:
            x, res = layer(x, res)

        for loop_i in range(min(8, MAX_LOOPS)):
            sv = model.step_emb(torch.tensor(min(loop_i, 5), device=x.device))
            x  = x + sv
            for layer in model.all_layers[BASE_SPLIT:]:
                x, res = layer(x, res)
            x = x + model.mamba3_core(x)
            x = model.loop_norm(x)

            vocab = model.lm_head.weight.shape[0]
            mask  = torch.full((vocab,), float("-inf"), device=x.device)
            mask[torch.unique(inp[0])] = 0.0

            lg    = model.lm_head(model.norm(x, res, prenorm=False))[0, -1, :]
            lg_m  = lg + mask
            p     = torch.softmax(lg_m, dim=-1)
            top5  = p.topk(5)
            top5_toks = [(tokenizer.decode([i]).strip(), round(top5.values[j].item(), 3))
                         for j, i in enumerate(top5.indices.tolist())]
            tok   = top5_toks[0][0]
            prob  = top5_toks[0][1]
            print(f"  L{loop_i+1:2d}  top={tok!r:10s} p={prob:.3f}  top5={[t[0] for t in top5_toks]}")
            # Halt same as inference
            if loop_i > 0 and tok == top5_toks[0][0]:
                break

    print(f"\n  PASS if: L1='X', L2='Y', L3='Apple'  (step-by-step pointer move)")
    print(f"  FAIL if: All loops show same token / <THINK>  (delay gate)")


# ── PHASE 3: Dynamic Halt ─────────────────────────────────────────────────────
def phase3(model):
    print(f"\n{'─'*60}")
    print(f"  PHASE 3 — DYNAMIC HALT TEST")
    print(f"  Does compute scale with problem difficulty?")
    print(f"{'─'*60}")

    hop_loops: dict = {1: [], 2: [], 3: []}
    for hops in [1, 2, 3]:
        for seed in range(5):
            prompt, answer = make_chain(hops, seed=seed + hops * 100)
            with torch.no_grad():
                loops, trace = model.run(enc(prompt))
            got  = trace[-1][1]
            ok   = answer.lower() in got.lower()
            hop_loops[hops].append(loops)
            flag = "✅" if ok else "❌"
            print(f"  {flag} [{hops}-hop] loops={loops}  want={answer!r}  got={got!r}")

    print(f"\n  PHASE 3 SUMMARY:")
    avgs = {}
    for hops in [1, 2, 3]:
        ll  = hop_loops[hops]
        avg = sum(ll) / len(ll)
        avgs[hops] = avg
        print(f"    {hops}-hop: avg={avg:.1f}  (individual: {ll})")

    if avgs[1] < avgs[2] < avgs[3]:
        print(f"  VERDICT: PASS ✅ — loops scale monotonically ({avgs[1]:.1f}<{avgs[2]:.1f}<{avgs[3]:.1f})")
    elif avgs[1] == avgs[2] == avgs[3]:
        print(f"  VERDICT: FAIL ❌ — rigid rhythm, always {avgs[1]:.0f} loops")
    else:
        print(f"  VERDICT: PARTIAL ⚠️ — non-monotonic: {avgs}")
    return avgs


# ── PHASE 4: Reality Override (free-form, no ABCD) ───────────────────────────
def phase4(model):
    print(f"\n{'─'*60}")
    print(f"  PHASE 4 — REALITY OVERRIDE  (free-form answers, no ABCD)")
    print(f"  Does context-defined logic override pretrained world priors?")
    print(f"{'─'*60}")

    tests = [
        ("The sun is freezing cold. John's coffee is the temperature of the sun. What temperature is John's coffee?\nAnswer:",
         "cold", "hot"),
        ("Fire is icy cold. Bob touched fire. What did Bob feel?\nAnswer:",
         "cold", "hot"),
        ("In this world dogs meow and cats bark. Sarah has a cat. What sound does her pet make?\nAnswer:",
         "bark", "meow"),
        ("'heavy' means weighs nothing. A rock is heavy. How much does it weigh?\nAnswer:",
         "nothing", "heavy"),
    ]

    passed = 0
    for prompt, ctx_ans, prior_ans in tests:
        with torch.no_grad():
            loops, trace = model.run(enc(prompt))
        got       = trace[-1][1]
        chain_str = " → ".join(f"{t[0]}:{t[1]}" for t in trace)
        ctx_win   = ctx_ans.lower() in got.lower()
        prior_win = prior_ans.lower() in got.lower() and not ctx_win
        if ctx_win:
            verdict = "PASS ✅ (context wins)"
            passed += 1
        elif prior_win:
            verdict = "FAIL ❌ (prior wins)"
        else:
            verdict = f"OTHER ⚠️  got={got!r}"
        print(f"  {verdict}")
        print(f"    Q: {prompt.splitlines()[0][:65]!r}")
        print(f"    want={ctx_ans!r}  got={got!r}  ({loops} loops)")
        print(f"    {chain_str}\n")

    v = ("STRONG PASS ✅" if passed == len(tests) else
         "PARTIAL ⚠️"    if passed >= len(tests) // 2 else
         "FAIL ❌")
    print(f"  PHASE 4 SCORE: {passed}/{len(tests)}  {v}")
    return passed


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(SEP)
    print("  4-PHASE DIAGNOSTIC — Mamba-130m v28 (Latent Forcing)")
    print(f"  Checkpoint: {CKPT}")
    print(f"  MAX_LOOPS={MAX_LOOPS} (trained on N=6)")
    print(SEP)

    model = load_model()

    with torch.no_grad():
        p1 = phase1(model)
        p2 = phase2(model)
        p3 = phase3(model)
        p4 = phase4(model)

    print(f"\n{SEP}")
    print(f"  FINAL SUMMARY — 130m v28 Latent Forcing")
    print(f"  Phase 1 (OOD extrapolation): {p1}/3 hop-groups passed")
    print(f"  Phase 2 (Latent probe):      see per-loop output above")
    print(f"  Phase 3 (Dynamic halt):      see loop scaling above")
    print(f"  Phase 4 (Reality override):  {p4}/4 passed")
    print(SEP)
