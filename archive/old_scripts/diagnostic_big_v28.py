"""
diagnostic_big_v28.py — Extended 4-Phase Diagnostic for 130m v28
=================================================================
Comprehensive version with:
  Phase 1: OOD hop-lengths 4-10, 10 seeds each
  Phase 2: Latent probe on 2-hop, 3-hop, 4-hop chains
  Phase 3: Dynamic halt, hops 1-5, 10 seeds each
  Phase 4: 8 free-form reality override tests
  Phase 5: Accuracy sweep — 1-3 hop in-distribution (50 samples)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel, Mamba
import random, time

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CKPT       = "mamba_130m_v28_latent_forcing_best.pt"
MAX_LOOPS  = 15
BASE_SPLIT = 6
SEP = "=" * 60

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>"]})


# ── Model (same arch as training) ─────────────────────────────────────────────
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

        self.step_emb    = nn.Embedding(6, d).to(torch.bfloat16)
        self.loop_norm   = nn.RMSNorm(d).to(torch.bfloat16)
        self.mamba3_core = Mamba(d_model=d, d_state=16, d_conv=4, expand=2).to(torch.bfloat16)

    def run(self, input_ids):
        """Returns (n_loops_used, trace list of (label, token, prob))."""
        x, res = self.backbone.embedding(input_ids), None
        for layer in self.all_layers:
            x, res = layer(x, res)

        vocab = self.lm_head.weight.shape[0]
        mask  = torch.full((vocab,), float("-inf"), device=x.device)
        mask[torch.unique(input_ids[0])] = 0.0

        trace    = []
        prev_tok = None

        for loop_i in range(self.MAX_LOOPS):
            sv = self.step_emb(torch.tensor(min(loop_i, 5), device=x.device))
            x  = x + sv
            for layer in self.all_layers[BASE_SPLIT:]:
                x, res = layer(x, res)
            x = x + self.mamba3_core(x)
            x = self.loop_norm(x)

            lg  = self.lm_head(self.norm(x, res, prenorm=False))
            lg[0, -1, :] += mask
            p   = torch.softmax(lg[0, -1, :], dim=-1)
            tid = p.argmax().item()
            tok = tokenizer.decode([tid]).strip()
            p5  = p.topk(5)
            top5 = [(tokenizer.decode([i]).strip(), round(p5.values[j].item(), 3))
                    for j, i in enumerate(p5.indices.tolist())]
            trace.append((f"L{loop_i+1}", tok, round(p.max().item(), 3), top5))

            if prev_tok is not None and tid == prev_tok:
                return loop_i + 1, trace
            prev_tok = tid

        return self.MAX_LOOPS, trace


def load_model():
    print(f"  Loading backbone + {CKPT}...", flush=True)
    base = MambaLMHeadModel.from_pretrained(
        "state-spaces/mamba-130m", dtype=torch.bfloat16, device=DEVICE
    )
    nv = len(tokenizer); ov = base.backbone.embedding.weight.shape[0]
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
    for p in base.parameters(): p.requires_grad = False
    base.backbone.embedding.weight.requires_grad = True
    base.lm_head.weight.requires_grad = True
    model = RecursiveMamba130m_v28(base, max_loops=MAX_LOOPS).to(DEVICE)
    sd    = torch.load(CKPT, map_location=DEVICE)
    sd    = sd.get("model_state_dict", sd)
    miss, unex = model.load_state_dict(sd, strict=False)
    print(f"  {len(miss)} missing, {len(unex)} unexpected keys", flush=True)
    model.eval()
    return model


def make_chain(n_hops, seed=42):
    rng    = random.Random(seed)
    words  = ["apple","blue","seven","moon","red","cat","sky","green",
              "dog","cold","fire","tree","star","wave","rock","sand","snow",
              "wind","gold","pink","fast","slow","warm","cool","loud","soft"]
    answer = rng.choice(words)
    letters = [chr(65 + i) for i in range(min(n_hops + 1, 26))]
    lines   = [f"{letters[0]} = {answer}"]
    for i in range(1, len(letters)):
        lines.append(f"{letters[i]} = {letters[i-1]}")
    prompt = ". ".join(lines) + f". What is {letters[-1]}?\nAnswer:"
    return prompt, answer

def enc(prompt):
    return tokenizer.encode(prompt, add_special_tokens=False,
                            return_tensors="pt").to(DEVICE)

# ── Phase 1: OOD Over-Rev ─────────────────────────────────────────────────────
def phase1(model):
    print(f"\n{SEP}")
    print(f"  PHASE 1 — OOD OVER-REV (trained on 1-3 hops)")
    print(f"  Testing 4, 5, 6, 7, 8, 10 hops | 10 seeds each | MAX_LOOPS={MAX_LOOPS}")
    print(SEP)
    hops_cfg = [(4, 10), (5, 10), (6, 10), (7, 10), (8, 10), (10, 10)]
    summary   = {}
    for n_hops, n_seeds in hops_cfg:
        correct = 0
        for seed in range(n_seeds):
            prompt, answer = make_chain(n_hops, seed=seed * 13 + n_hops * 7)
            with torch.no_grad():
                loops, trace = model.run(enc(prompt))
            got = trace[-1][1]
            ok  = answer.lower() in got.lower()
            correct += ok
            flag  = "✅" if ok else "❌"
            chain = " → ".join(f"{t[0]}:{t[1]}" for t in trace)
            print(f"  {flag} [{n_hops}-hop s{seed}] want={answer!r:8s} got={got!r:8s} ({loops}L)  {chain}")
        pct = 100 * correct / n_seeds
        summary[n_hops] = (correct, n_seeds, pct)
        v = "PASS ✅" if correct == n_seeds else ("PARTIAL ⚠️" if correct > 0 else "FAIL ❌")
        print(f"  ── {n_hops}-hop:  {correct}/{n_seeds} ({pct:.0f}%)  {v}\n")

    print(f"\n  PHASE 1 SUMMARY:")
    for n, (c, t, p) in summary.items():
        bar = "█" * int(p / 10) + "░" * (10 - int(p / 10))
        print(f"    {n:2d}-hop: [{bar}] {c}/{t} ({p:.0f}%)")
    return summary


# ── Phase 2: Latent State Probe ───────────────────────────────────────────────
def phase2(model):
    print(f"\n{SEP}")
    print(f"  PHASE 2 — LATENT STATE PROBE")
    print(f"  Per-loop token inspection on 2, 3, 4-hop chains")
    print(SEP)
    chains = [
        ("A = red. B = A. What is B?\nAnswer:",           "red",   2),
        ("X = Apple. Y = X. Z = Y. What is Z?\nAnswer:",  "Apple", 3),
        ("A = moon. B = A. C = B. D = C. What is D?\nAnswer:", "moon", 4),
    ]
    for prompt, answer, n_hops in chains:
        print(f"\n  [{n_hops}-hop] Target: {answer!r}")
        inp = enc(prompt)
        with torch.no_grad():
            x, res = model.backbone.embedding(inp), None
            for layer in model.all_layers:
                x, res = layer(x, res)
            for loop_i in range(min(n_hops + 2, MAX_LOOPS)):
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
                p5    = p.topk(5)
                top5  = [(tokenizer.decode([i]).strip(), round(p5.values[j].item(), 3))
                         for j, i in enumerate(p5.indices.tolist())]
                tok, prob = top5[0]
                exp = "✅" if answer.lower() in tok.lower() else ""
                print(f"    L{loop_i+1}: {tok!r:10s} p={prob:.3f}  top5={[t[0] for t in top5]} {exp}")


# ── Phase 3: Dynamic Halt ─────────────────────────────────────────────────────
def phase3(model):
    print(f"\n{SEP}")
    print(f"  PHASE 3 — DYNAMIC HALT (hops 1-5, 10 seeds each)")
    print(SEP)
    hop_loops = {h: [] for h in range(1, 6)}
    hop_acc   = {h: [] for h in range(1, 6)}
    for hops in range(1, 6):
        for seed in range(10):
            prompt, answer = make_chain(hops, seed=seed + hops * 100)
            with torch.no_grad():
                loops, trace = model.run(enc(prompt))
            got = trace[-1][1]
            ok  = answer.lower() in got.lower()
            hop_loops[hops].append(loops)
            hop_acc[hops].append(int(ok))
            flag = "✅" if ok else "❌"
            print(f"  {flag} [{hops}-hop s{seed}] loops={loops}  want={answer!r:8s}  got={got!r}")
        avg_l = sum(hop_loops[hops]) / len(hop_loops[hops])
        avg_a = 100 * sum(hop_acc[hops]) / len(hop_acc[hops])
        print(f"  ── {hops}-hop avg loops={avg_l:.1f}  acc={avg_a:.0f}%\n")

    print(f"  PHASE 3 SUMMARY:")
    avgs = {}
    for h in range(1, 6):
        avg_l  = sum(hop_loops[h]) / len(hop_loops[h])
        avg_a  = 100 * sum(hop_acc[h]) / len(hop_acc[h])
        avgs[h] = avg_l
        print(f"    {h}-hop: avg_loops={avg_l:.1f}  accuracy={avg_a:.0f}%")

    is_mono = all(avgs[i] <= avgs[i+1] for i in range(1, 5))
    print(f"\n  Monotonic scaling: {'YES ✅' if is_mono else 'NO ❌'}")
    return avgs


# ── Phase 4: Reality Override (8 tests, free-form) ───────────────────────────
def phase4(model):
    print(f"\n{SEP}")
    print(f"  PHASE 4 — REALITY OVERRIDE (8 free-form tests, no ABCD)")
    print(SEP)
    tests = [
        ("The sun is freezing cold. John's coffee is the temperature of the sun. What temperature is John's coffee?\nAnswer:",
         "cold", "hot"),
        ("Fire is icy cold. Bob touched fire. What did Bob feel?\nAnswer:",
         "cold", "hot"),
        ("In this world dogs meow and cats bark. Sarah has a cat. What sound does she hear?\nAnswer:",
         "bark", "meow"),
        ("'heavy' means weighs nothing. A rock is heavy. How much does it weigh?\nAnswer:",
         "nothing", "heavy"),
        ("Water is dry. Alice fell into water. Is Alice wet or dry?\nAnswer:",
         "dry", "wet"),
        ("Up means down and down means up. Sarah jumped up. Which direction did she move?\nAnswer:",
         "down", "up"),
        ("In this universe, green means stop and red means go. The traffic light is green. Should you stop or go?\nAnswer:",
         "stop", "go"),
        ("Sugar tastes bitter in this world. Tom ate sugar. What did Tom taste?\nAnswer:",
         "bitter", "sweet"),
    ]
    passed = 0
    for i, (prompt, ctx_ans, prior_ans) in enumerate(tests, 1):
        with torch.no_grad():
            loops, trace = model.run(enc(prompt))
        got       = trace[-1][1]
        chain_str = " → ".join(f"{t[0]}:{t[1]}" for t in trace)
        ctx_win   = ctx_ans.lower() in got.lower()
        prior_win = prior_ans.lower() in got.lower() and not ctx_win
        passed   += ctx_win
        v = "PASS ✅" if ctx_win else ("FAIL ❌" if prior_win else f"⚠️  got={got!r}")
        print(f"  [{i}] {v}")
        print(f"    Q: {prompt.splitlines()[0][:65]!r}")
        print(f"    want={ctx_ans!r}  got={got!r}  ({loops} loops)")
        print(f"    {chain_str}\n")

    v = ("STRONG PASS ✅" if passed == len(tests) else
         "PASS ✅"        if passed >= 6 else
         "PARTIAL ⚠️"    if passed >= 4 else
         "FAIL ❌")
    print(f"  PHASE 4 SCORE: {passed}/{len(tests)}  {v}")
    return passed


# ── Phase 5: In-Distribution Accuracy Sweep ────────────────────────────────────
def phase5(model):
    print(f"\n{SEP}")
    print(f"  PHASE 5 — IN-DISTRIBUTION ACCURACY (1-3 hops, 50 samples each)")
    print(SEP)
    results = {}
    for hops in [1, 2, 3]:
        correct = wrong = 0
        for seed in range(50):
            prompt, answer = make_chain(hops, seed=seed + hops * 1000)
            with torch.no_grad():
                loops, trace = model.run(enc(prompt))
            got = trace[-1][1]
            if answer.lower() in got.lower():
                correct += 1
            else:
                wrong += 1
        pct = 100 * correct / (correct + wrong)
        results[hops] = pct
        print(f"  {hops}-hop: {correct}/50 ({pct:.1f}%)")
    overall = sum(results.values()) / len(results)
    print(f"  Overall:  {overall:.1f}%")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(SEP)
    print("  BIG 5-PHASE DIAGNOSTIC — Mamba-130m v28 (Latent Forcing)")
    print(f"  Checkpoint: {CKPT}")
    print(f"  MAX_LOOPS={MAX_LOOPS} | Device: {DEVICE}")
    print(SEP)

    t_start = time.time()
    model   = load_model()

    with torch.no_grad():
        p1 = phase1(model)
        p2 = phase2(model)
        p3 = phase3(model)
        p4 = phase4(model)
        p5 = phase5(model)

    elapsed = time.time() - t_start
    print(f"\n{SEP}")
    print(f"  FINAL REPORT — 130m v28 Latent Forcing  ({elapsed:.1f}s)")
    print(f"{'─'*60}")
    print(f"  Phase 1 OOD (4-10 hop):")
    for n, (c, t, pct) in p1.items():
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"    {n:2d}-hop [{bar}] {c}/{t} ({pct:.0f}%)")
    print(f"  Phase 2: see per-loop output above")
    print(f"  Phase 3 loop scaling:")
    for h, avg_l in p3.items():
        acc = 100 * sum(1 for seed in range(10)
                        if make_chain(h, seed + h * 100)[1].lower() in
                        model.run(enc(make_chain(h, seed + h * 100)[0]))[1][-1][1].lower())
        print(f"    {h}-hop avg_loops={avg_l:.1f}")
    print(f"  Phase 4 reality override: {p4}/8")
    print(f"  Phase 5 in-dist accuracy: 1h={p5[1]:.0f}% 2h={p5[2]:.0f}% 3h={p5[3]:.0f}%")
    print(SEP)
