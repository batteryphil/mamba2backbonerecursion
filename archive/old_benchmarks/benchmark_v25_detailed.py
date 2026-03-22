"""
benchmark_v25_detailed.py
=========================
Comprehensive benchmark of Mamba3-130M v25, MaxN=6 checkpoint.
Tests in-distribution accuracy, OOD generalisation, loop-depth ablation,
error taxonomy, and baseline comparison.
"""
import sys, os, json, random, time, importlib.util, collections
import torch

os.chdir('/home/phil/Desktop/mambadiff/mambadiff llm tts')

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "mamba3_finetuned_v25.pt"        # Final v26 fully-trained MaxN=6 weights
RESULTS    = {}  # collect all results for the final report

# ── Load module + model ────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("finetune_mamba3", "finetune_mamba3.py")
mod  = importlib.util.module_from_spec(spec)
sys.modules["finetune_mamba3"] = mod
try:
    spec.loader.exec_module(mod)
except SystemExit:
    pass

model = mod.model.to(DEVICE)
tok   = mod.tokenizer

ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded {CHECKPOINT}  step={ckpt.get('step','?')}  maxn_saved={ckpt.get('current_max_loops','?')}")


# ── Inference ──────────────────────────────────────────────────────────────

def predict(text: str, max_loops: int = 6) -> tuple[str, int, list, float]:
    """Return (pred_str, loops_used, trace, ms)."""
    model.MAX_LOOPS = max_loops
    ids = tok.encode(text, add_special_tokens=False, max_length=512, truncation=True)
    inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    t0  = time.time()
    with torch.no_grad():
        out = model(inp)
    ms = (time.time() - t0) * 1000
    logits, loops_taken, trace = out[0], out[1], out[2] if isinstance(out, tuple) else (out, 6, [])
    pred = trace[-1][2] if trace else tok.decode([logits[0, -1, :].argmax().item()]).strip()
    return pred, loops_taken, trace, ms


def score_samples(samples, max_loops=6, n=None, seed=42) -> dict:
    """Evaluate samples. Returns full stats dict."""
    random.seed(seed)
    pool = random.sample(samples, min(n or len(samples), len(samples)))
    correct, wrong_list, loop_counts, times = 0, [], [], []
    for item in pool:
        text   = item["text"]
        answer = str(item.get("answer", "")).strip().lower()
        split  = text.rfind("Answer:")
        prompt = text[:split].rstrip() + "\nAnswer:" if split >= 0 else text
        pred, loops, _, ms = predict(prompt, max_loops=max_loops)
        ok = pred.lower().strip() == answer
        if ok:
            correct += 1
        else:
            wrong_list.append({"prompt": prompt[-80:], "pred": pred, "ans": answer})
        loop_counts.append(loops)
        times.append(ms)
    total = len(pool)
    return {
        "correct": correct, "total": total,
        "acc": correct / total * 100,
        "wrong": wrong_list[:8],
        "avg_loops": sum(loop_counts) / total,
        "avg_ms":    sum(times) / total,
    }


# ── Load data ──────────────────────────────────────────────────────────────
print("Loading datasets...")
with open("system2_logic_v1.json") as f:
    s2 = json.load(f)

hop1 = [s for s in s2 if s.get("hops") == 1]
hop2 = [s for s in s2 if s.get("hops") == 2]
hop3 = [s for s in s2 if s.get("hops") == 3]

# Taxonomy by task type
by_type = collections.defaultdict(list)
for item in s2:
    t = item["text"]
    if "apple" in t or "coin" in t or "earns" in t or "spends" in t or "Change:" in t:
        by_type["arithmetic"].append(item)
    elif "in the" in t or "inside" in t or "on the" in t.lower():
        by_type["spatial"].append(item)
    elif "owns" in t or "have" in t or "pet" in t:
        by_type["property"].append(item)
    elif "likes" in t.lower() or "chose" in t or "matched" in t or "copies" in t:
        by_type["name_chain"].append(item)
    else:
        by_type["variable_binding"].append(item)

print(f"  Loaded {len(s2)} samples: hop1={len(hop1)} hop2={len(hop2)} hop3={len(hop3)}")
for k, v in sorted(by_type.items()):
    print(f"    {k:20s}: {len(v)}")


print("\n" + "="*70)
print("  MAMBA3-130M v25 | MaxN=6 | DETAILED BENCHMARK")
print("="*70)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: LOOP-DEPTH ABLATION (the scientific proof)
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  SECTION 1 — Loop-depth ablation (2-hop tasks, N=1..6)")
print("  Hypothesis: accuracy increases monotonically with more loops")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
abl = {}
random.seed(42)
ablation_pool = random.sample(hop2, min(300, len(hop2)))
for n in range(1, 7):
    r = score_samples(ablation_pool, max_loops=n, n=300, seed=42)
    abl[n] = r
    bar = "█" * int(r["acc"] / 5)
    print(f"  N={n}: {r['correct']:3d}/{r['total']}  {r['acc']:5.1f}%  {bar}")
RESULTS["ablation"] = abl


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: PER-HOP ACCURACY AT FULL N=6
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  SECTION 2 — Per-hop accuracy (MaxN=6, 150 samples each)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
hops = {}
for label, pool in [("1-hop", hop1), ("2-hop", hop2), ("3-hop", hop3)]:
    r = score_samples(pool, max_loops=6, n=min(150, len(pool)), seed=7)
    hops[label] = r
    bar = "█" * int(r["acc"] / 5)
    print(f"  {label}: {r['correct']:3d}/{r['total']}  {r['acc']:5.1f}%  avg={r['avg_loops']:.1f} loops  {bar}")
RESULTS["per_hop"] = hops


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: TASK-TYPE BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  SECTION 3 — Task-type accuracy breakdown (N=6, 150 samples each)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
task_results = {}
for tname, pool in sorted(by_type.items()):
    if not pool:
        continue
    r = score_samples(pool, max_loops=6, n=min(150, len(pool)), seed=13)
    task_results[tname] = r
    bar = "█" * int(r["acc"] / 5)
    print(f"  {tname:20s}: {r['correct']:3d}/{r['total']}  {r['acc']:5.1f}%  {bar}")
RESULTS["task_types"] = task_results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: OUT-OF-DISTRIBUTION (OOD) — fresh sentences, same token pool
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  SECTION 4 — Out-of-distribution test (novel prompts, same vocab)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
ood = [
    # Variable binding (novel phrasing)
    {"t": "Variable Q holds orange. Variable R is set to Q. What is R?\nAnswer:", "a": "orange", "cat": "var_bind"},
    {"t": "Let G equal blue. H is assigned the value of G. H equals?\nAnswer:", "a": "blue", "cat": "var_bind"},
    {"t": "M = pink. N = M. O = N. What is O?\nAnswer:", "a": "pink", "cat": "var_bind_3hop"},
    {"t": "Define T as purple. U copies T. V mirrors U. V is?\nAnswer:", "a": "purple", "cat": "var_bind_3hop"},
    {"t": "X is red. Y is the same as X. Z is the same as Y. Z?\nAnswer:", "a": "red", "cat": "var_bind_3hop"},
    # Spatial (novel phrasing)
    {"t": "The hat is inside the drawer. The drawer is in the closet. Where is the hat?\nAnswer:", "a": "closet", "cat": "spatial"},
    {"t": "My pen is on the desk. The desk is in the office. The pen is in the?\nAnswer:", "a": "office", "cat": "spatial"},
    {"t": "The coin is in the jar. The jar is in the bag. The bag is on the shelf. The coin is on the?\nAnswer:", "a": "shelf", "cat": "spatial_3hop"},
    # Name chains (novel names)
    {"t": "Tom likes green. Sue copies Tom. Sue likes?\nAnswer:", "a": "green", "cat": "name_chain"},
    {"t": "Zoe chose yellow. Max copies Zoe's pick. Max chose?\nAnswer:", "a": "yellow", "cat": "name_chain"},
    {"t": "Anna likes blue. Ben copies Anna. Cal copies Ben. Cal likes?\nAnswer:", "a": "blue", "cat": "name_chain_3hop"},
    # Arithmetic (correct labels now)
    {"t": "Start: X=3. Change: +4. End: X=?\nAnswer:", "a": "7", "cat": "arithmetic"},
    {"t": "Start: X=8. Change: -3. End: X=?\nAnswer:", "a": "5", "cat": "arithmetic"},
    {"t": "Mike has 5 apples. Mike earns 2 apples. Mike now has?\nAnswer:", "a": "7", "cat": "arithmetic"},
    {"t": "Lisa has 9 coins. Lisa spends 4 coins. Lisa now has?\nAnswer:", "a": "5", "cat": "arithmetic"},
    {"t": "Start: Y=6. Change: -6. End: Y=?\nAnswer:", "a": "0", "cat": "arithmetic_zero"},
]

ood_by_cat = collections.defaultdict(lambda: {"correct": 0, "total": 0})
ood_correct = 0
ood_lines = []
for item in ood:
    pred, loops, _, ms = predict(item["t"], max_loops=6)
    ok = pred.lower().strip() == item["a"]
    if ok:
        ood_correct += 1
    ood_by_cat[item["cat"]]["total"] += 1
    if ok:
        ood_by_cat[item["cat"]]["correct"] += 1
    status = "✅" if ok else "❌"
    ood_lines.append(f"  {status} [{loops}L] pred={pred!r:8s} ans={item['a']!r:6s} | {item['t'][:55].strip()!r}")

for line in ood_lines:
    print(line)

print(f"\n  OOD total:  {ood_correct}/{len(ood)} = {ood_correct/len(ood)*100:.0f}%")
print("\n  By category:")
for cat, d in sorted(ood_by_cat.items()):
    acc = d["correct"] / d["total"] * 100 if d["total"] else 0
    ok  = "✅" if acc == 100 else ("⚠️ " if acc >= 50 else "❌")
    print(f"    {ok} {cat:20s}: {d['correct']}/{d['total']} = {acc:.0f}%")
RESULTS["ood"] = {"correct": ood_correct, "total": len(ood), "by_cat": dict(ood_by_cat)}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: BASELINE COMPARISON (N=1 vs N=6 on hard tasks)
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  SECTION 5 — Baseline vs full model (N=1 vs N=6 on 2-hop tasks)")
print("  N=1 = frozen backbone output (no LoRA reasoning)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
random.seed(99)
hard_pool = random.sample(hop2, min(200, len(hop2)))
r_n1 = score_samples(hard_pool, max_loops=1, n=200, seed=99)
r_n6 = score_samples(hard_pool, max_loops=6, n=200, seed=99)
print(f"  N=1 (baseline): {r_n1['correct']:3d}/200 = {r_n1['acc']:5.1f}%")
print(f"  N=6 (full):     {r_n6['correct']:3d}/200 = {r_n6['acc']:5.1f}%")
print(f"  Improvement:    +{r_n6['acc'] - r_n1['acc']:.1f}pp  (×{r_n6['acc']/max(r_n1['acc'],0.1):.1f} relative)")
RESULTS["baseline"] = {"n1": r_n1, "n6": r_n6}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: FAILURE MODE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  SECTION 6 — Failure mode analysis")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
# Show wrong predictions from the task-type breakdown
for tname, r in sorted(task_results.items()):
    wrongs = r.get("wrong", [])
    if wrongs:
        print(f"\n  [{tname}] failures (showing up to 3):")
        for w in wrongs[:3]:
            print(f"    ❌ pred={w['pred']!r:10s} ans={w['ans']!r:10s} | ...{w['prompt'][-60:]!r}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: LIVE REASONING TRACES
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  SECTION 7 — Live reasoning traces at MaxN=6")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
trace_demos = [
    ("2-hop var",  "Let X = blue. Y points to X. What is Y?\nAnswer:"),
    ("3-hop var",  "A = red. B = A. C = B. D = C. What is D?\nAnswer:"),
    ("Spatial 2",  "The key is in the bag. The bag is in the box. Where is the key?\nAnswer:"),
    ("Name chain", "Alice likes green. Bob copies Alice. Carol copies Bob. Carol likes?\nAnswer:"),
    ("Arithmetic", "Start: N=7. Change: -3. End: N=?\nAnswer:"),
    ("Arith. 0",   "Start: K=5. Change: -5. End: K=?\nAnswer:"),
]
trace_rows = []
for label, prompt in trace_demos:
    pred, loops, trace, ms = predict(prompt, max_loops=6)
    trace_str = " → ".join([f"{t[0]}:{t[2]}" for t in trace])
    print(f"\n  [{label}]  {prompt.strip()!r}")
    print(f"    Trace: {trace_str}")
    print(f"    Answer: {pred!r}  ({loops} loops, {ms:.0f}ms)")
    trace_rows.append((label, prompt, pred, loops, trace_str, ms))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: PERFORMANCE PROFILE
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  SECTION 8 — Inference performance by loop count")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
perf_prompt = "X = blue. Y = X. What is Y?\nAnswer:"
for n in range(1, 7):
    runs = [predict(perf_prompt, max_loops=n)[3] for _ in range(5)]
    avg_ms = sum(runs) / len(runs)
    print(f"  N={n}: avg={avg_ms:.1f}ms")


print("\n" + "="*70)
print("  BENCHMARK COMPLETE")
print("="*70)

# Save raw results for report
import json as _json
with open("/tmp/bench_results.json", "w") as f:
    _json.dump({k: {kk: vv for kk, vv in (v.items() if isinstance(v, dict) else {}.items()) if kk != "wrong"} for k, v in RESULTS.items()}, f, indent=2, default=str)
print("Raw results saved to /tmp/bench_results.json")
