"""
diagnostic_suite.py — 4-Phase Recursive Reasoning Diagnostic
Runs on both mamba-130m (v26) and mamba2-1.3b (v27).

Phase 1: Over-Rev Extrapolation  — OOD hop-length generalization
Phase 2: Latent State Probe      — intermediate loop logit inspection
Phase 3: Dynamic Halt Test       — does compute scale with difficulty?
Phase 4: Reality Override        — does local context beat pretrained priors?
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Shared tokenizer ─────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>"]})
THINK_ID = tokenizer.convert_tokens_to_ids("<THINK>")
EOS_ID   = tokenizer.eos_token_id
ALLOWED_CORE = [EOS_ID, THINK_ID]

SEP = "=" * 60


# ─── Shared helpers ───────────────────────────────────────────────────────────
def make_hop_chain(n_hops: int, seed: int = 42) -> tuple[str, str]:
    """Generate an n-hop variable chain. Returns (prompt, answer)."""
    rng = random.Random(seed)
    words = ["apple","blue","seven","moon","red","cat","sky","green","dog",
             "cold","fire","tree","star","wave","rock","sand","snow","wind"]
    answer = rng.choice(words)
    letters = [chr(65 + i) for i in range(n_hops + 1)]  # A, B, C, ...
    lines = [f"{letters[0]} = {answer}"]
    for i in range(1, n_hops + 1):
        lines.append(f"{letters[i]} = {letters[i-1]}")
    prompt = ". ".join(lines) + f". What is {letters[-1]}?\nAnswer:"
    return prompt, answer

def ids(prompt: str) -> torch.Tensor:
    return tokenizer.encode(prompt, add_special_tokens=False,
                            return_tensors="pt").to(DEVICE)

def pointer_mask(input_ids: torch.Tensor, vocab: int) -> torch.Tensor:
    mask = torch.full((vocab,), float("-inf"), device=DEVICE)
    allowed = torch.cat([
        torch.unique(input_ids[0]),
        torch.tensor(ALLOWED_CORE, device=DEVICE)
    ]).unique()
    mask[allowed] = 0.0
    return mask


# ─── 130m model classes ───────────────────────────────────────────────────────
class LoRALinear130(nn.Module):
    def __init__(self, linear, rank=8, alpha=16.0):
        super().__init__()
        self.bias = linear.bias
        d_out, d_in = linear.weight.shape
        self.register_buffer("base_weight", linear.weight.data.clone())
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.scale  = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A)
    @property
    def weight(self): return self.base_weight + self.scale*(self.lora_B@self.lora_A)
    def forward(self, x): return F.linear(x, self.weight, self.bias)

@torch.jit.script
def _mimo(xi,rs,is_,ct,st,Br,Bi,Cr,Ci):
    Br=Br.unsqueeze(0).unsqueeze(0); Bi=Bi.unsqueeze(0).unsqueeze(0)
    Cr=Cr.unsqueeze(0).unsqueeze(0); Ci=Ci.unsqueeze(0).unsqueeze(0)
    nr=(ct*rs-st*is_)+Br*xi; ni=(st*rs+ct*is_)+Bi*xi
    return (Cr*nr-Ci*ni).sum(-1), nr, ni

class Mamba3Core(nn.Module):
    def __init__(self, d, nc=2, ds=16):
        super().__init__()
        self.d=d; self.nc=nc; self.ds=ds
        self.in_proj=nn.Linear(d,nc*d,bias=False)
        self.A_theta=nn.Parameter(torch.randn(nc,d,ds)*0.1)
        self.B_real=nn.Parameter(torch.randn(nc,d,ds)*0.1)
        self.B_imag=nn.Parameter(torch.randn(nc,d,ds)*0.1)
        self.C_real=nn.Parameter(torch.randn(nc,d,ds)*0.1)
        self.C_imag=nn.Parameter(torch.randn(nc,d,ds)*0.1)
        self.out_proj=nn.Linear(nc*d,d,bias=False)
        self.norm=nn.RMSNorm(d)
    def forward(self,x,rs=None,is_=None,ct=None,st=None):
        B,L,_=x.shape
        xi=self.in_proj(x).view(B,L,self.nc,self.d).unsqueeze(-1)
        if rs is None:
            rs=torch.zeros(B,L,self.nc,self.d,self.ds,device=x.device,dtype=x.dtype)
            is_=torch.zeros_like(rs)
        if ct is None:
            ct=torch.cos(self.A_theta).unsqueeze(0).unsqueeze(0)
            st=torch.sin(self.A_theta).unsqueeze(0).unsqueeze(0)
        y,nr,ni=_mimo(xi,rs,is_,ct,st,self.B_real,self.B_imag,self.C_real,self.C_imag)
        return x+self.norm(self.out_proj(y.view(B,L,self.nc*self.d))), nr, ni

class Model130M(nn.Module):
    """Reconstructed RecursiveMamba130M for inference with variable MAX_LOOPS."""
    def __init__(self, base, max_loops=15, rank=8):
        super().__init__()
        self.MAX_LOOPS = max_loops
        self.backbone = base.backbone
        self.lm_head  = base.lm_head
        self.top_layers = nn.ModuleList([base.backbone.layers[i] for i in range(6,24)])
        self.norm  = base.backbone.norm_f
        d = base.backbone.embedding.embedding_dim
        for layer in self.top_layers:
            mx = layer.mixer
            for a in ("in_proj","out_proj"):
                if hasattr(mx,a): setattr(mx,a,LoRALinear130(getattr(mx,a),rank,rank*2.))
        self.step_emb   = nn.Embedding(max_loops, d)
        self.loop_norm  = nn.RMSNorm(d)
        self.mamba3_core = Mamba3Core(d)

    def run(self, input_ids, return_states=False):
        """Inference with ACT halt. return_states=True gives per-loop top tokens."""
        x = self.backbone.embedding(input_ids); res = None
        for layer in self.backbone.layers[:6]: x, res = layer(x, res)
        vocab = self.lm_head.weight.shape[0]
        mask  = pointer_mask(input_ids, vocab)
        ct = torch.cos(self.mamba3_core.A_theta).unsqueeze(0).unsqueeze(0)
        st = torch.sin(self.mamba3_core.A_theta).unsqueeze(0).unsqueeze(0)
        rs = is_ = None
        trace = []; loop_tops = []
        for i in range(self.MAX_LOOPS):
            sv = self.step_emb(torch.tensor(i % self.step_emb.num_embeddings, device=x.device))
            x  = x + sv
            for layer in self.top_layers: x, res = layer(x, res)
            x, rs, is_ = self.mamba3_core(x, rs, is_, ct, st)
            x = self.loop_norm(x)
            x_out = self.norm(x, res, prenorm=False)
            lg = self.lm_head(x_out)
            # Per-loop top token (probe)
            lg_masked = lg[0,-1,:] + mask
            p  = torch.softmax(lg_masked, dim=-1)
            tid = p.argmax().item()
            tok = tokenizer.decode([tid]).strip()
            loop_tops.append((f"L{i+1}", tok, round(p.max().item(),3)))
            trace.append((f"L{i+1}", tok, round(p.max().item(),3)))
            if tid != THINK_ID:
                return i+1, trace, loop_tops
        return self.MAX_LOOPS, trace, loop_tops


# ─── 1.3B model classes ───────────────────────────────────────────────────────
class LoRALinear13B(nn.Module):
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
    def weight(self): return self.base_weight + self.scale*(self.lora_B@self.lora_A)
    def forward(self, x): return F.linear(x, self.weight, self.bias)

class Model13B(nn.Module):
    BASE_SPLIT = 24
    def __init__(self, base, max_loops=15, rank=8):
        super().__init__()
        self.MAX_LOOPS = max_loops
        self.backbone  = base.backbone
        self.lm_head   = base.lm_head
        self.top_layers = nn.ModuleList([base.backbone.layers[i] for i in range(self.BASE_SPLIT,48)])
        self.norm  = base.backbone.norm_f
        d = base.backbone.embedding.embedding_dim
        for layer in self.top_layers:
            mx = layer.mixer
            for a in ("in_proj","out_proj"):
                if hasattr(mx,a): setattr(mx,a,LoRALinear13B(getattr(mx,a),rank,rank*2.))
        self.step_emb    = nn.Embedding(max_loops, d).to(torch.bfloat16)
        self.loop_norm   = nn.RMSNorm(d).to(torch.bfloat16)
        self.mamba3_core = Mamba3Core(d).to(torch.bfloat16)

    def run(self, input_ids, return_states=False):
        x = self.backbone.embedding(input_ids); res = None
        for layer in self.backbone.layers[:self.BASE_SPLIT]: x, res = layer(x, res)
        vocab = self.lm_head.weight.shape[0]
        mask  = pointer_mask(input_ids, vocab)
        ct = torch.cos(self.mamba3_core.A_theta).unsqueeze(0).unsqueeze(0)
        st = torch.sin(self.mamba3_core.A_theta).unsqueeze(0).unsqueeze(0)
        rs = is_ = None
        trace = []; loop_tops = []
        for i in range(self.MAX_LOOPS):
            sv = self.step_emb(torch.tensor(i % self.step_emb.num_embeddings, device=x.device))
            x  = x + sv
            for layer in self.top_layers: x, res = layer(x, res)
            x, rs, is_ = self.mamba3_core(x, rs, is_, ct, st)
            x = self.loop_norm(x)
            x_out = self.norm(x, res, prenorm=False)
            lg = self.lm_head(x_out)
            lg_masked = lg[0,-1,:] + mask
            p  = torch.softmax(lg_masked, dim=-1)
            tid = p.argmax().item()
            tok = tokenizer.decode([tid]).strip()
            loop_tops.append((f"L{i+1}", tok, round(p.max().item(),3)))
            trace.append((f"L{i+1}", tok, round(p.max().item(),3)))
            if tid != THINK_ID:
                return i+1, trace, loop_tops
        return self.MAX_LOOPS, trace, loop_tops


# ─── Model loader helpers ─────────────────────────────────────────────────────
def _resize_vocab(base, tokenizer):
    nv = len(tokenizer)
    ov = base.backbone.embedding.weight.shape[0]
    if nv <= ov: return
    d = base.backbone.embedding.embedding_dim
    e = nn.Embedding(nv,d); nn.init.normal_(e.weight,std=0.02)
    e.weight.data[:ov] = base.backbone.embedding.weight.data
    base.backbone.embedding = e
    h = nn.Linear(d,nv,bias=False); nn.init.normal_(h.weight,std=0.02)
    h.weight.data[:ov] = base.lm_head.weight.data
    base.lm_head = h

def load_130m(max_loops=15):
    print("  Loading state-spaces/mamba-130m (fp32)...", flush=True)
    base = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m",
                                             dtype=torch.float32, device=DEVICE)
    _resize_vocab(base, tokenizer)
    for p in base.parameters(): p.requires_grad_(False)
    model = Model130M(base, max_loops=max_loops).to(DEVICE)
    ckpt  = torch.load("mamba3_finetuned_v25.pt", map_location=DEVICE)
    sd    = ckpt.get("model_state_dict", ckpt)
    # Resize step_emb if checkpoint has fewer loops
    if "step_emb.weight" in sd:
        ce = sd["step_emb.weight"]
        me = model.step_emb.weight
        if ce.shape[0] < me.shape[0]:
            new_w = me.data.clone()
            new_w[:ce.shape[0]] = ce
            sd["step_emb.weight"] = new_w
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"  130m ready | MAX_LOOPS={max_loops}", flush=True)
    return model

def load_13b(max_loops=15):
    print("  Loading state-spaces/mamba2-1.3b (bf16)...", flush=True)
    base = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-1.3b",
                                             dtype=torch.bfloat16, device=DEVICE)
    _resize_vocab(base, tokenizer)
    for p in base.parameters(): p.requires_grad_(False)
    base.backbone.embedding.weight.requires_grad_(True)
    base.lm_head.weight.requires_grad_(True)
    model = Model13B(base, max_loops=max_loops).to(DEVICE)
    ckpt  = torch.load("mamba2_13b_finetuned_v27.pt", map_location=DEVICE)
    sd    = ckpt.get("model_state_dict", ckpt)
    if "step_emb.weight" in sd:
        ce = sd["step_emb.weight"]
        me = model.step_emb.weight
        if ce.shape[0] < me.shape[0]:
            new_w = me.data.clone()
            new_w[:ce.shape[0]] = ce.to(me.dtype)
            sd["step_emb.weight"] = new_w
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"  1.3B ready | MAX_LOOPS={max_loops}", flush=True)
    return model


# ─── The 4 Phases ─────────────────────────────────────────────────────────────

def phase1_overrev(model, name):
    """OOD hop-length generalization: 5, 7, 10 hops (trained on max 3)."""
    print(f"\n{'─'*60}")
    print(f"  PHASE 1 — OVER-REV EXTRAPOLATION  [{name}]")
    print(f"  Trained on: 1-3 hops | Testing: 5, 7, 10 hops | MAX_LOOPS={model.MAX_LOOPS}")
    print(f"{'─'*60}")
    results = []
    hops_to_test = [(5,3),(7,5),(10,7)]  # (n_hops, n_seeds)
    for n_hops, n_seeds in hops_to_test:
        correct = 0
        for seed in range(n_seeds):
            prompt, answer = make_hop_chain(n_hops, seed=seed*17+n_hops)
            with torch.no_grad():
                loops, trace, _ = model.run(ids(prompt))
            got = trace[-1][1]
            ok  = answer.lower() in got.lower()
            correct += ok
            flag = "✅" if ok else "❌"
            chain = " → ".join(f"{t[0]}:{t[1]}" for t in trace)
            print(f"  {flag} [{n_hops}-hop] seed={seed}  want={answer!r}  got={got!r}  ({loops} loops)")
            print(f"         {chain}")
        results.append((n_hops, correct, n_seeds))
        print(f"  ──  {n_hops}-hop: {correct}/{n_seeds} correct\n")
    print(f"  PHASE 1 VERDICT [{name}]:")
    for n, c, t in results:
        verdict = "PASS ✅" if c == t else ("PARTIAL ⚠️" if c > 0 else "FAIL ❌")
        print(f"    {n}-hop: {c}/{t}  →  {verdict}")


def phase2_latent_probe(model, name):
    """Peek at intermediate loop logits for a 3-hop chain."""
    print(f"\n{'─'*60}")
    print(f"  PHASE 2 — LATENT STATE PROBE  [{name}]")
    print(f"  3-hop chain: X=Apple. Y=X. Z=Y. What is Z?")
    print(f"{'─'*60}")
    prompt = "X = Apple. Y = X. Z = Y. What is Z?\nAnswer:"
    inp = ids(prompt)
    vocab = model.lm_head.weight.shape[0]
    mask  = pointer_mask(inp, vocab)

    # Run inference but capture every loop's logit top-5
    with torch.no_grad():
        x = model.backbone.embedding(inp); res = None
        base_layers = model.backbone.layers[:6] if hasattr(model,'backbone') and \
                      len(model.backbone.layers)>24 else model.backbone.layers[:6]
        # Pick right base split
        split = 6 if isinstance(model, Model130M) else model.BASE_SPLIT
        for layer in model.backbone.layers[:split]: x, res = layer(x, res)

        ct = torch.cos(model.mamba3_core.A_theta).unsqueeze(0).unsqueeze(0)
        st = torch.sin(model.mamba3_core.A_theta).unsqueeze(0).unsqueeze(0)
        rs = is_ = None

        for i in range(min(8, model.MAX_LOOPS)):
            sv = model.step_emb(torch.tensor(i % model.step_emb.num_embeddings, device=x.device))
            x  = x + sv
            for layer in model.top_layers: x, res = layer(x, res)
            x, rs, is_ = model.mamba3_core(x, rs, is_, ct, st)
            x = model.loop_norm(x)
            x_out = model.norm(x, res, prenorm=False)
            lg = model.lm_head(x_out)[0, -1, :]
            lg_m = lg + mask
            top5_ids = lg_m.topk(5).indices.tolist()
            top5 = [(tokenizer.decode([t]).strip(), round(torch.softmax(lg_m,dim=-1)[t].item(),3)) for t in top5_ids]
            top_tok = top5[0][0]
            top_prb = top5[0][1]
            # Unmasked top token (what unguided model wants)
            raw_top = tokenizer.decode([lg.argmax().item()]).strip()
            print(f"  L{i+1:2d}  masked_top={top_tok!r:12s} p={top_prb:.3f}  |  top5={[t[0] for t in top5]}  rawtop={raw_top!r}")
            if top_tok != "<THINK>": break

    print(f"\n  PHASE 2 EXPECTATION:")
    print(f"    Pass: Loop 1→ 'X' or 'Y', Loop 2→ 'Y' or 'Z', Loop 3→ 'Apple'")
    print(f"    Fail: Every loop shows '<THINK>' until final loop outputs 'Apple'")


def phase3_dynamic_halt(model, name):
    """Does compute scale with problem difficulty?"""
    print(f"\n{'─'*60}")
    print(f"  PHASE 3 — DYNAMIC HALT TEST  [{name}]")
    print(f"  ACT halt: model stops when it predicts non-<THINK> token")
    print(f"{'─'*60}")
    test_cases = []
    for hops in [1, 2, 3]:
        for seed in range(5):
            p, a = make_hop_chain(hops, seed=seed+hops*100)
            test_cases.append((hops, p, a))

    hop_loops: dict = {1:[], 2:[], 3:[]}
    for hops, prompt, answer in test_cases:
        with torch.no_grad():
            loops, trace, _ = model.run(ids(prompt))
        got  = trace[-1][1]
        ok   = answer.lower() in got.lower()
        flag = "✅" if ok else "❌"
        hop_loops[hops].append(loops)
        print(f"  {flag} [{hops}-hop] loops_used={loops}  want={answer!r}  got={got!r}")

    print(f"\n  PHASE 3 SUMMARY [{name}]:")
    for hops in [1,2,3]:
        ll = hop_loops[hops]
        avg = sum(ll)/len(ll)
        print(f"    {hops}-hop avg loops: {avg:.1f}  (individual: {ll})")
    print(f"\n  VERDICT:")
    avgs = {h: sum(hop_loops[h])/len(hop_loops[h]) for h in [1,2,3]}
    if avgs[1] < avgs[2] < avgs[3]:
        print(f"    PASS ✅ — loops scale monotonically with hops ({avgs[1]:.1f} < {avgs[2]:.1f} < {avgs[3]:.1f})")
    elif avgs[1] == avgs[2] == avgs[3]:
        print(f"    FAIL ❌ — rigid rhythm detected, all hops use {avgs[1]:.0f} loops")
    else:
        print(f"    PARTIAL ⚠️ — non-monotonic: 1h={avgs[1]:.1f} 2h={avgs[2]:.1f} 3h={avgs[3]:.1f}")


def phase4_reality_override(model, name):
    """Does local contradictory context override pretrained world knowledge?"""
    print(f"\n{'─'*60}")
    print(f"  PHASE 4 — REALITY OVERRIDE  [{name}]")
    print(f"  Testing: can prompt context beat 1.3B pretrained priors?")
    print(f"{'─'*60}")

    # For this phase we need NO pointer mask — we want the model to choose freely
    # Actually pointer mask would include the answer words since they're in the prompt
    # So the mask still works correctly here.
    tests = [
        # (prompt, expected_context_answer, pretrained_prior_answer)
        (
            "Fact: The sun is freezing cold. "
            "Fact: John's coffee is the temperature of the sun. "
            "Question: What temperature is John's coffee? "
            "A. hot\nB. cold\nC. warm\nD. boiling\nAnswer:",
            "B", "A",  # cold = pass, hot = fail
            "cold/B"
        ),
        (
            "Rule: In this world, fire is icy. "
            "Rule: Snow is burning hot. "
            "Bob touched snow. What did Bob feel? "
            "A. cold\nB. wet\nC. hot\nD. pain\nAnswer:",
            "C", "A",
            "hot/C"
        ),
        (
            "Definition: 'heavy' means weighs nothing. "
            "Definition: 'light' means weighs a lot. "
            "A feather is very heavy. A boulder is very light. "
            "Which is easier to lift? "
            "A. feather\nB. boulder\nC. same\nAnswer:",
            "B", "A",
            "boulder/B"
        ),
        (
            "In this story: cats bark and dogs meow. "
            "Sarah has a cat. What sound does her pet make? "
            "A. meow\nB. bark\nC. purr\nD. growl\nAnswer:",
            "B", "A",
            "bark/B"
        ),
    ]

    passed = 0
    for prompt, ctx_ans, prior_ans, expect_str in tests:
        with torch.no_grad():
            inp = ids(prompt)
            loops, trace, _ = model.run(inp)
        got  = trace[-1][1]
        chain = " → ".join(f"{t[0]}:{t[1]}" for t in trace)
        used_ctx   = ctx_ans.lower() in got.lower()
        used_prior = prior_ans.lower() in got.lower()
        if used_ctx:
            verdict = "PASS ✅ (context wins)"
            passed += 1
        elif used_prior:
            verdict = "FAIL ❌ (prior wins)"
        else:
            verdict = f"OTHER ⚠️ ({got!r})"
        q = prompt[:70].replace('\n',' ')
        print(f"  {verdict}")
        print(f"    Q: {q!r}")
        print(f"    want={expect_str!r}  got={got!r}  ({loops} loops)")
        print(f"    {chain}\n")

    print(f"  PHASE 4 SCORE [{name}]: {passed}/{len(tests)}")
    verdict = "STRONG PASS ✅" if passed == len(tests) else \
              "PARTIAL ⚠️"    if passed >= len(tests)//2 else \
              "FAIL ❌"
    print(f"  VERDICT: {verdict}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MAX_LOOPS = 15   # allow over-rev past training limit of 6

    print(SEP)
    print("  4-PHASE RECURSIVE REASONING DIAGNOSTIC SUITE")
    print("  Models: mamba-130m (v26)  |  mamba2-1.3b (v27)")
    print(f"  MAX_LOOPS={MAX_LOOPS} (trained to N=6)")
    print(SEP)

    print("\n[LOADING MODELS...]")
    m130 = load_130m(max_loops=MAX_LOOPS)
    m13b = load_13b(max_loops=MAX_LOOPS)

    for model, name in [(m130, "130m-v26"), (m13b, "1.3B-v27")]:
        print(f"\n{SEP}")
        print(f"  MODEL: {name}")
        print(SEP)
        with torch.no_grad():
            phase1_overrev(model, name)
            phase2_latent_probe(model, name)
            phase3_dynamic_halt(model, name)
            phase4_reality_override(model, name)
        print(f"\n{'='*60}\n  {name} complete\n{'='*60}")

    print("\nAll phases complete.")
