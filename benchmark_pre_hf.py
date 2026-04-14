"""
benchmark_pre_hf.py
===================
Complete pre-upload benchmark for the OO-SomaMind 2.8B model stack.

Sections
--------
  1. SYSTEM     — GPU, VRAM baseline
  2. THROUGHPUT — Tokens/sec at seq lengths [64, 128, 256, 512]
  3. PERPLEXITY — Cross-entropy loss on standard English text (WikiText sample)
  4. REPETITION — Bigram rep rate across 20 diverse prompts (default mode)
  5. OO DOMAIN  — Recall quality on 12 OO ontology probes (oo_domain mode)
  6. HALT HEAD  — p_halt separation gap (HIGH vs LOW prompts)
  7. GATE       — Proprioception gate degeneration response (W_g norm, gate_diff)
  8. ROUTING    — Inference mode auto-detection accuracy (12 routing checks)

Output: benchmark_pre_hf_results.txt
"""

import os, sys, time, json, math, torch, torch.nn as nn
from collections import Counter
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/home/phil/.gemini/antigravity/scratch/RM3_Project")

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm import MambaLMHeadModel
from safetensors.torch import load_file
from transformers import AutoTokenizer

MODEL_DIR   = "/home/phil/.gemini/antigravity/scratch/Syrin_Mamba/Syrin_Mamba_Enterprise_Pack/mamba-2.8b-latent"
GATE_PATH   = "/home/phil/.gemini/antigravity/scratch/RM3_Project/proprio_gate_2.8b.pt"
LORA_PATH   = "/home/phil/.gemini/antigravity/scratch/RM3_Project/lora_oo_r16.pt"
HALT_V2     = "/home/phil/.gemini/antigravity/scratch/RM3_Project/halting_head_v2.pt"
HALT_V1     = "/home/phil/.gemini/antigravity/scratch/RM3_Project/halting_head.pt"
RESULT_PATH = "/home/phil/.gemini/antigravity/scratch/RM3_Project/benchmark_pre_hf_results.txt"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

lines = []

def pr(msg=""):
    """Print and record a line."""
    print(msg)
    lines.append(str(msg))

def sep(title=""):
    """Section separator."""
    if title:
        bar = f"{'─' * 3} {title} {'─' * (60 - len(title))}"
    else:
        bar = "─" * 66
    pr(bar)

# ─────────────────────────────────────────────────────────────────────────────
# Sampling helpers
# ─────────────────────────────────────────────────────────────────────────────
def top_p_sample(logits: torch.Tensor, temperature: float = 0.4,
                 top_p: float = 0.9, rep_penalty: float = 1.4,
                 prev_ids: Optional[list] = None) -> int:
    """Sample next token with top-p + repetition penalty."""
    if rep_penalty != 1.0 and prev_ids:
        for tid in set(prev_ids):
            logits[tid] = (logits[tid] / rep_penalty if logits[tid] > 0
                           else logits[tid] * rep_penalty)
    logits = logits / max(temperature, 1e-6)
    probs  = torch.softmax(logits, dim=-1)
    sp, si = torch.sort(probs, descending=True)
    cum    = torch.cumsum(sp, dim=-1)
    cut    = (cum - sp) > top_p
    sp[cut] = 0.0
    sp /= sp.sum()
    return si[torch.multinomial(sp, 1)].item()


def ngram_repeat(ids: list, n: int = 4) -> bool:
    """Detect if last n-gram appeared before."""
    if len(ids) < n * 2:
        return False
    last = tuple(ids[-n:])
    return last in [tuple(ids[i:i+n]) for i in range(len(ids) - n)]


def generate(model, adapter, gate, tok, prompt: str,
             max_tok: int = 80, mode: str = "default",
             seed: int = 42) -> tuple:
    """Generate from the full stack. Returns (text, rep_pct, tok_count)."""
    torch.manual_seed(seed)
    eos    = tok.eos_token_id
    ids    = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=64).input_ids.to(DEVICE)
    out    = []
    use_oo = (mode == "oo_domain")
    T    = 0.4 if use_oo else 0.8
    tp   = 0.9 if use_oo else 0.95
    rp   = 1.4 if use_oo else 1.1
    ng   = 4   if use_oo else None

    with torch.no_grad():
        cur = ids
        for _ in range(max_tok):
            h = model.backbone(cur)
            if adapter is not None:
                h = adapter(h)
            if gate is not None:
                h = gate(h)
            logits = model.lm_head(h)[0, -1, :].float()
            if use_oo:
                nxt = top_p_sample(logits, T, tp, rp, out)
            else:
                nxt = logits.argmax().item()
            if nxt == eos:
                break
            out.append(nxt)
            if ng and ngram_repeat(out, ng):
                break
            cur = torch.cat([cur, torch.tensor([[nxt]], device=DEVICE)], dim=1)

    text = tok.decode(out, skip_special_tokens=True).strip()
    w    = text.split()
    bg   = [(w[i], w[i+1]) for i in range(len(w)-1)]
    rep  = 100.0 * (len(bg) - len(set(bg))) / max(1, len(bg))
    return text, rep, len(out)


# ─────────────────────────────────────────────────────────────────────────────
# Load stack
# ─────────────────────────────────────────────────────────────────────────────
pr("=" * 66)
pr("  OO-SomaMind 2.8B  — Pre-HF Upload Benchmark")
pr(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
pr("=" * 66)
pr()

sep("1. SYSTEM")
import subprocess
gpu_info = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
     "--format=csv,noheader"], text=True).strip()
pr(f"  GPU      : {gpu_info}")
pr(f"  PyTorch  : {torch.__version__}")
pr(f"  Device   : {DEVICE}")
pr(f"  CUDA     : {torch.version.cuda}")
vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
pr(f"  VRAM     : {vram_total:.1f} GB total")
pr()

pr("  Loading model stack...")
t_load = time.perf_counter()
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tok.pad_token = tok.eos_token

cfg   = MambaConfig(d_model=2560, n_layer=64, vocab_size=50280, pad_vocab_size_multiple=8)
model = MambaLMHeadModel(cfg, dtype=torch.bfloat16, device=DEVICE)
sd    = load_file(os.path.join(MODEL_DIR, "model.safetensors"))
if "lm_head.weight" not in sd and "backbone.embedding.weight" in sd:
    sd["lm_head.weight"] = sd["backbone.embedding.weight"]
model.load_state_dict(sd, strict=False)
model.eval()

from proprioception_gate import GeometricProprioceptionGate
gate_sd = torch.load(GATE_PATH, map_location=DEVICE)
gate    = GeometricProprioceptionGate(d_model=2560, window_size=8)
gate.load_state_dict(gate_sd)
gate    = gate.to(DEVICE, dtype=torch.bfloat16).eval()

from lora_mamba import PostBackboneLoRA, load_post_lora
adapter = PostBackboneLoRA(d_model=2560, rank=16, alpha=32.0, n_layers=6).to(DEVICE)
load_post_lora(adapter, LORA_PATH, device=DEVICE)
adapter.eval()

load_time = time.perf_counter() - t_load
vram_used = torch.cuda.memory_allocated() / 1024**3
vram_pct  = 100 * vram_used / vram_total
pr(f"  Load time : {load_time:.1f}s")
pr(f"  VRAM used : {vram_used:.2f} GB  ({vram_pct:.1f}%  of {vram_total:.1f} GB)")
pr(f"  VRAM free : {vram_total - vram_used:.2f} GB available for sessions/SWARM")
total_params = sum(p.numel() for p in model.parameters())
pr(f"  Params    : {total_params/1e9:.3f}B backbone + 522K LoRA + ~8K gate")
pr()

# ─────────────────────────────────────────────────────────────────────────────
sep("2. THROUGHPUT — tokens/sec at various sequence lengths")
# ─────────────────────────────────────────────────────────────────────────────
seq_lengths = [64, 128, 256, 512]
tput_results = []
WARMUP = 3
REPS   = 8

for seq_len in seq_lengths:
    ids  = torch.randint(100, 50000, (1, seq_len), device=DEVICE)
    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model.backbone(ids)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(REPS):
            _ = model.backbone(ids)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tps     = (seq_len * REPS) / elapsed
    tput_results.append((seq_len, tps))
    flag = "✓" if tps >= 1500 else "~" if tps >= 800 else "✗"
    pr(f"  {flag} seq={seq_len:>4}  →  {tps:>7,.0f} tok/s  ({elapsed*1000/REPS:.1f} ms/pass)")

avg_tps = sum(t for _, t in tput_results) / len(tput_results)
pr(f"       avg  →  {avg_tps:>7,.0f} tok/s")
pr()

# ─────────────────────────────────────────────────────────────────────────────
sep("3. PERPLEXITY — cross-entropy on English text sample")
# ─────────────────────────────────────────────────────────────────────────────
# Wikitext-style sample paragraphs
WIKI_SAMPLE = [
    "The mitochondria are often called the powerhouse of the cell because they generate most of the cell's supply of ATP, used as a source of chemical energy.",
    "The French Revolution was a period of radical political and societal change in France that began with the Estates General of 1789 and ended with the formation of the French Consulate in November 1799.",
    "In mathematics, the Pythagorean theorem states that the square of the hypotenuse is equal to the sum of the squares of the other two sides of a right triangle.",
    "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data.",
    "The speed of light in a vacuum is approximately 299,792,458 metres per second. This universal physical constant is denoted by the letter c.",
    "The human brain contains approximately 86 billion neurons, each connected to thousands of other neurons through synapses, forming a complex network.",
    "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.",
    "Climate change refers to long-term shifts in temperatures and weather patterns, primarily caused by human activities since the 1800s.",
]

losses = []
with torch.no_grad():
    for text in WIKI_SAMPLE:
        ids  = tok.encode(text, return_tensors="pt").to(DEVICE)
        ids  = ids[:, :256]
        h    = model.backbone(ids)
        h    = adapter(h)
        h    = gate(h)
        logits = model.lm_head(h).float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = ids[..., 1:].contiguous()
        loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)),
                                      shift_labels.view(-1))
        losses.append(loss.item())

avg_loss = sum(losses) / len(losses)
perplexity = math.exp(avg_loss)
flag = "✓" if perplexity < 30 else "~" if perplexity < 60 else "✗"
pr(f"  {flag} Avg CE loss  : {avg_loss:.4f}")
pr(f"  {flag} Perplexity   : {perplexity:.2f}  (lower = better; GPT-2 baseline ~29)")
pr()

# ─────────────────────────────────────────────────────────────────────────────
sep("4. REPETITION — bigram rep rate, 20 diverse prompts, default mode")
# ─────────────────────────────────────────────────────────────────────────────
DIVERSE_PROMPTS = [
    "The capital of France is",
    "To make a good cup of coffee, you should",
    "Explain the difference between a list and a tuple in Python.",
    "The best way to stay healthy is",
    "Once upon a time in a land far away,",
    "The theory of relativity states that",
    "In order to solve this math problem, first",
    "The most important feature of a good API is",
    "When debugging code, the first step is",
    "The history of the Roman Empire began",
    "Water is composed of two hydrogen atoms and",
    "To deploy a Docker container, run the command",
    "The stock market crashed in 1929 because",
    "A neural network learns by",
    "The fastest animal on land is the cheetah, which can reach speeds of",
    "In quantum mechanics, the uncertainty principle states",
    "The process of photosynthesis converts",
    "To write a recursive function in Python, you need",
    "The largest ocean on Earth is",
    "To reverse a string in Python, you can",
]

rep_scores = []
for i, prompt in enumerate(DIVERSE_PROMPTS):
    text, rep, ntok = generate(model, adapter, gate, tok, prompt, mode="default")
    rep_scores.append(rep)
    flag = "✓" if rep < 20 else "~" if rep < 40 else "✗"
    # Print just summary per-prompt
    pr(f"  {flag} [{rep:>4.0f}%rep | {ntok:>2}t] {prompt[:50]!r}")

avg_rep = sum(rep_scores) / len(rep_scores)
good    = sum(1 for r in rep_scores if r < 20)
flag    = "✓" if avg_rep < 20 else "~" if avg_rep < 35 else "✗"
pr()
pr(f"  {flag} Summary: {good}/{len(rep_scores)} under 20% rep  |  avg={avg_rep:.1f}%")
pr()

# ─────────────────────────────────────────────────────────────────────────────
sep("5. OO DOMAIN — recall quality (oo_domain mode, T=0.4, top-p=0.9)")
# ─────────────────────────────────────────────────────────────────────────────
OO_PROBES = [
    ("[OO] What is the limbion engine responsible for?",       ["limbion", "temporal", "sequen", "chronion", "phase"]),
    ("[OO] What is the governor in the OO organism?",          ["governor", "decision", "BATTERFYL", "policy", "halt"]),
    ("[OO] What is the DNA hash in the OO organism?",          ["hash", "DNA", "integrity", "cryptograph", "nvram", "boot"]),
    ("[SELF] What are you?",                                    ["Mamba", "OO", "backbone", "cognitive", "engine"]),
    ("[SWARM:MAIN] What is the SWARM protocol?",               ["SWARM", "agent", "fork", "cache", "snapshot"]),
    ("[OO] What is the thanatosion engine?",                   ["thanatosion", "shutdown", "graceful", "DNA", "CUDA"]),
    ("[OO] What does the GATE_COHERENCE_VIOLATION event mean?",["GATE", "coherence", "degenera", "proprioception", "warden"]),
    ("[OO] What is Phase C of the OO organism lifecycle?",     ["Phase C", "Active", "reasoning", "governor", "loop"]),
    ("[OO] What is the warden bus?",                           ["warden", "bus", "broadcast", "engine", "event"]),
    ("[OO] What is BATTERFYL?",                                ["BATTERFYL", "policy", "halt", "governor", "arbitr"]),
    ("[SELF] How do you detect degeneration loops?",           ["proprioception", "gate", "velocity", "drift", "coherence", "degenera"]),
    ("[OO] What is the zygotron boot sequence?",               ["zygotron", "boot", "DNA", "hash", "engine", "phase"]),
]

oo_scores = []
for prompt, keywords in OO_PROBES:
    text, rep, ntok = generate(model, adapter, gate, tok, prompt,
                                max_tok=100, mode="oo_domain")
    matched   = sum(1 for kw in keywords if kw.lower() in text.lower())
    recall    = matched / len(keywords)
    oo_scores.append(recall)
    flag = "✓" if recall >= 0.5 and rep < 30 else "~" if recall >= 0.25 else "✗"
    pr(f"  {flag} recall={recall:.0%} rep={rep:.0f}% [{ntok}t]  {prompt[:50]!r}")
    if recall < 0.5:
        pr(f"      missing: {[kw for kw in keywords if kw.lower() not in text.lower()]}")
    else:
        pr(f"      → {text[:100]!r}")

avg_recall = sum(oo_scores) / len(oo_scores)
good_oo    = sum(1 for s in oo_scores if s >= 0.5)
flag       = "✓" if avg_recall >= 0.55 else "~" if avg_recall >= 0.35 else "✗"
pr()
pr(f"  {flag} OO recall: {good_oo}/{len(oo_scores)} >= 50%  |  avg={avg_recall:.0%}")
pr()

# ─────────────────────────────────────────────────────────────────────────────
sep("6. HALT HEAD — p_halt separation (v2 vs v1)")
# ─────────────────────────────────────────────────────────────────────────────
def eval_halt_head(path, label):
    """Run the halting head over high/low prompts and report separation."""
    if not os.path.exists(path):
        pr(f"  [SKIP] {label}: not found")
        return None
    raw = torch.load(path, map_location=DEVICE, weights_only=True)

    # Auto-detect format and architecture
    if isinstance(raw, dict) and "state_dict" in raw:
        sd     = raw["state_dict"]
        d_in   = raw.get("d_input", 2560)
    else:
        sd   = raw
        d_in = 2560

    # Reconstruct MLP architecture from state_dict keys
    # Works for any Linear chain in the 'net' sequential
    layer_weights = sorted(
        [(k, v) for k, v in sd.items() if "weight" in k],
        key=lambda x: int(x[0].split(".")[1])
    )
    layers = []
    for i, (k, w) in enumerate(layer_weights):
        in_f, out_f = w.shape[1], w.shape[0]
        layers.append(nn.Linear(in_f, out_f))
        if i < len(layer_weights) - 1:
            layers.append(nn.GELU())
    head = nn.Sequential(*layers).to(DEVICE)

    # Build matching state dict
    new_sd = {}
    w_keys = sorted([k for k in sd if "weight" in k],
                    key=lambda x: int(x.split(".")[1]))
    b_keys = sorted([k for k in sd if "bias" in k],
                    key=lambda x: int(x.split(".")[1]))
    lin_idx = 0
    for mod_idx, mod in enumerate(head):
        if isinstance(mod, nn.Linear):
            new_sd[f"{mod_idx}.weight"] = sd[w_keys[lin_idx]]
            new_sd[f"{mod_idx}.bias"]   = sd[b_keys[lin_idx]]
            lin_idx += 1
    head.load_state_dict(new_sd)
    head = nn.Sequential(nn.Sequential(*head)).to(DEVICE)  # wrap for call
    # final layer should output a scalar, wrap with sigmoid
    class WrappedHead(nn.Module):
        def __init__(self, net): super().__init__(); self.net = net[0]; self.sig = nn.Sigmoid()
        def forward(self, x): return self.sig(self.net(x)).squeeze(-1)
    head_fn = WrappedHead(head).eval()

    high_scores, low_scores = [], []
    with torch.no_grad():
        for group, store in [(HIGH_PROMPTS, high_scores), (LOW_PROMPTS, low_scores)]:
            for prompt in group:
                ids = tok(prompt, return_tensors="pt",
                          truncation=True, max_length=64).input_ids.to(DEVICE)
                h   = model.backbone(ids)
                vec = h[0, -1, :d_in].float().unsqueeze(0)
                p   = head_fn(vec).item()
                store.append(p)

    avg_hi = sum(high_scores) / len(high_scores)
    avg_lo = sum(low_scores)  / len(low_scores)
    gap    = avg_hi - avg_lo
    flag   = "✓" if gap > 0.15 else "~" if gap > 0 else "✗"
    pr(f"  {flag} {label}:  HIGH={avg_hi:.3f}  LOW={avg_lo:.3f}  sep={gap:+.3f}")
    return gap


HIGH_PROMPTS = [
    "[OO] Critical: limbion stall detected in zone A.",
    "[SWARM:AGENT_2] Emergency: DNA hash mismatch on boot.",
    "[OO] BATTERFYL threshold exceeded. Escalate to governor.",
    "Explain the entire architecture of a modern transformer model in detail.",
    "Solve this multi-step logic puzzle: All A are B, some B are C, no C are D...",
]
LOW_PROMPTS = [
    "2 + 2 = ?",
    "What color is the sky?",
    "Name the capital of Japan.",
    "Hello!",
    "What is 10 divided by 2?",
]

gap_v2 = eval_halt_head(HALT_V2, "Halt Head v2 (OO semantics)")
gap_v1 = eval_halt_head(HALT_V1, "Halt Head v1 (legacy)      ")
pr()



# ─────────────────────────────────────────────────────────────────────────────
sep("7. GATE — Geometric Proprioception Gate diagnostics")
# ─────────────────────────────────────────────────────────────────────────────
wg_norm   = gate.W_g.weight.data.norm().item()
flag_norm = "✓" if 2.0 < wg_norm < 6.0 else "~"
pr(f"  {flag_norm} W_g norm : {wg_norm:.4f}  (healthy range 2.0–6.0)")

# Coherent hidden state
ids_clean = tok("The capital of France is Paris.", return_tensors="pt").input_ids.to(DEVICE)
with torch.no_grad():
    h_clean  = model.backbone(ids_clean)
    h_gated  = gate(h_clean)
diff_clean = (h_gated - h_clean).norm().item()

# Degenerate hidden state (repeating token sequence)
ids_degen = torch.full((1, 64), tok.eos_token_id, dtype=torch.long, device=DEVICE)
with torch.no_grad():
    h_degen  = model.backbone(ids_degen)
    h_dgated = gate(h_degen)
diff_degen = (h_dgated - h_degen).norm().item()

ratio = diff_degen / max(diff_clean, 1e-8)
flag_ratio = "✓" if ratio >= 1.5 else "~" if ratio >= 1.0 else "✗"
pr(f"  ✓ gate_diff (healthy input)    : {diff_clean:>8.2f}")
pr(f"  ✓ gate_diff (degenerate input) : {diff_degen:>8.2f}")
pr(f"  {flag_ratio} degenerate/healthy ratio  : {ratio:>8.2f}x  (target ≥ 1.5x heavier correction)")
pr()

# ─────────────────────────────────────────────────────────────────────────────
sep("8. ROUTING — Inference mode auto-detection")
# ─────────────────────────────────────────────────────────────────────────────
# Import the routing function from stateful_engine
engine_path = "/home/phil/.gemini/antigravity/scratch/ItsMick_mamba2backbonerecursion"
sys.path.insert(0, engine_path)
try:
    from stateful_engine import detect_inference_mode, INFERENCE_MODES
    routing_available = True
except ImportError as e:
    pr(f"  [SKIP] Could not import stateful_engine: {e}")
    routing_available = False

if routing_available:
    ROUTING_CASES = [
        ("[OO] What is the limbion engine?",            "oo_domain"),
        ("[SELF] What are you?",                         "oo_domain"),
        ("[SWARM:MAIN] Handoff requested.",              "oo_domain"),
        ("[WARDEN] LIMBION_STALL detected.",             "oo_domain"),
        ("/oo_status",                                   "oo_domain"),
        ("/fork session_a",                              "oo_domain"),
        ("def calculate_primes(n):",                     "code"),
        ("import numpy as np",                           "code"),
        ("What is the capital of Germany?",              "default"),
        ("The weather today is",                         "default"),
        ("Who are you?",                                 "identity"),
        ("What is your architecture?",                   "identity"),
    ]
    correct = 0
    for prompt, expected in ROUTING_CASES:
        detected = detect_inference_mode(prompt)
        ok = detected == expected
        correct += ok
        flag = "✓" if ok else "✗"
        pr(f"  {flag} {expected:<12} ← {prompt[:45]!r}")

    pr()
    pr(f"  Routing accuracy: {correct}/{len(ROUTING_CASES)}")
    pr(f"  Modes available : {list(INFERENCE_MODES.keys())}")
pr()

# ─────────────────────────────────────────────────────────────────────────────
sep()
pr("  BENCHMARK COMPLETE")
sep()

# ── Final score card ──
pr()
pr("  SCORE CARD")
pr(f"  VRAM footprint    : {vram_used:.2f} GB / {vram_total:.1f} GB  ({vram_pct:.0f}%)")
pr(f"  Avg throughput    : {avg_tps:,.0f} tok/s")
pr(f"  Perplexity        : {perplexity:.2f}")
pr(f"  Repetition (gen)  : {avg_rep:.1f}%  ({good}/{len(rep_scores)} prompts clean)")
pr(f"  OO domain recall  : {avg_recall:.0%}  ({good_oo}/{len(OO_PROBES)} probes >= 50%)")
if gap_v2 is not None:
    status = "PASS" if gap_v2 > 0.15 else "MARGINAL" if gap_v2 > 0 else "FAIL"
    pr(f"  Halt head v2 sep  : {gap_v2:+.3f}  [{status}]")
pr(f"  Gate W_g norm     : {wg_norm:.4f}")
pr(f"  Gate degen ratio  : {ratio:.2f}x")
if routing_available:
    pr(f"  Mode routing      : {correct}/{len(ROUTING_CASES)}")

pr()
pr(f"  Results → {RESULT_PATH}")
pr("=" * 66)

with open(RESULT_PATH, "w") as f:
    f.write("\n".join(lines))
