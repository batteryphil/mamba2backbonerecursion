"""
benchmark_v25.py
================
Benchmark for the Mamba3 v25 MaxN=4 checkpoint.
Uses mod.model + mod.tokenizer directly from finetune_mamba3 module.

Benchmarks:
  1. Per-hop accuracy (1-hop, 2-hop, 3-hop) at MaxN=4
  2. N=1 vs N=2 vs N=3 vs N=4 loop-depth ablation
  3. Out-of-distribution fresh examples
  4. Live reasoning trace display
"""
import sys, os, json, random, time, importlib.util
import torch

os.chdir('/home/phil/Desktop/mambadiff/mambadiff llm tts')

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "mamba3_finetuned_v25_MaxN_4.pt"

# ── Load the module (classes + pre-built model + tokenizer) ──────────────────
print("Loading finetune_mamba3 module ...", flush=True)
spec = importlib.util.spec_from_file_location("finetune_mamba3", "finetune_mamba3.py")
mod  = importlib.util.module_from_spec(spec)
sys.modules["finetune_mamba3"] = mod
try:
    spec.loader.exec_module(mod)
except SystemExit:
    pass   # training guard fired — classes and model already in module

model     = mod.model.to(DEVICE)
tok       = mod.tokenizer
THINK_ID  = mod.THINK_TOKEN_ID

# Load the best checkpoint weights
print(f"Loading checkpoint: {CHECKPOINT} ...", flush=True)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
saved_step = ckpt.get("step", "?")
saved_maxn = ckpt.get("current_max_loops", "?")
print(f"  ✅ Step={saved_step}  MaxN_at_save={saved_maxn}")

print("\n" + "="*65)
print("  Mamba3-130M v25 MaxN=4 | Benchmark Results")
print("="*65)


# ── Inference helper ─────────────────────────────────────────────────────────

def predict(text: str, max_loops: int = 4) -> tuple[str, int, list, float]:
    """Run inference. Returns (pred_token, loops_used, trace, ms)."""
    model.MAX_LOOPS = max_loops
    ids = tok.encode(text, add_special_tokens=False, max_length=512, truncation=True)
    inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    t0 = time.time()
    with torch.no_grad():
        out = model(inp)
    elapsed = (time.time() - t0) * 1000
    # Model returns (logits, loops_taken, trace) in inference mode.
    # The returned logits are UNMASKED — pointer mask was applied inside the loop.
    # The trace captures the MASKED predictions per loop; use the final one.
    if isinstance(out, tuple):
        logits, loops_taken, trace = out[0], out[1], out[2]
    else:
        logits, loops_taken, trace = out, max_loops, []
    # trace[-1] = (loop_label, max_prob, pred_token_str)
    if trace:
        pred_str = trace[-1][2]   # correct masked answer from final loop
    else:
        pred_id  = logits[0, -1, :].argmax().item()
        pred_str = tok.decode([pred_id]).strip()
    return pred_str, loops_taken, trace, elapsed


def eval_dataset(samples: list, max_loops: int = 4, n: int = 200) -> dict:
    """Evaluate on n samples, return accuracy stats."""
    random.shuffle(samples)
    subset  = samples[:n]
    correct = 0
    for item in subset:
        text   = item["text"]
        answer = item.get("answer", "").strip().lower()
        # Prompt: strip the trailing "Answer: X" to simulate real inference
        split  = text.rfind("Answer:")
        prompt = text[:split].rstrip() + "\nAnswer:" if split >= 0 else text
        pred, _, _, _ = predict(prompt, max_loops=max_loops)
        if pred.lower().strip() == answer:
            correct += 1
    return {"correct": correct, "total": len(subset), "acc": correct / len(subset) * 100}


# ── Load test data ────────────────────────────────────────────────────────────
print("\nLoading system2_logic_v1.json ...", flush=True)
with open("system2_logic_v1.json") as f:
    s2 = json.load(f)

hop1 = [s for s in s2 if s.get("hops", 2) == 1]
hop2 = [s for s in s2 if s.get("hops", 2) == 2]
hop3 = [s for s in s2 if s.get("hops", 2) == 3]
print(f"  1-hop: {len(hop1)}  2-hop: {len(hop2)}  3-hop: {len(hop3)}")

random.seed(42)

# ── BENCHMARK 1: Per-hop accuracy at MaxN=4 ──────────────────────────────────
print("\n" + "─"*65)
print("  BENCHMARK 1 — Per-hop accuracy at N=4 (150 samples each)")
print("─"*65)
for label, pool in [("1-hop", hop1), ("2-hop", hop2), ("3-hop", hop3)]:
    r = eval_dataset(pool, max_loops=4, n=min(150, len(pool)))
    bar = "█" * int(r["acc"] / 5)
    print(f"  {label}: {r['correct']:3d}/{r['total']} = {r['acc']:5.1f}%  {bar}")


# ── BENCHMARK 2: N=1 vs N=2 vs N=3 vs N=4 ablation on 2-hop tasks ───────────
print("\n" + "─"*65)
print("  BENCHMARK 2 — Loop-depth ablation on 2-hop tasks")
print("  (Proves genuine recursive compute — more loops = better accuracy)")
print("─"*65)
random.seed(99)
ablation_pool = random.sample(hop2, min(200, len(hop2)))
for n_loops in [1, 2, 3, 4]:
    correct = 0
    for item in ablation_pool:
        text   = item["text"]
        answer = item.get("answer", "").strip().lower()
        split  = text.rfind("Answer:")
        prompt = text[:split].rstrip() + "\nAnswer:" if split >= 0 else text
        pred, _, _, _ = predict(prompt, max_loops=n_loops)
        if pred.lower().strip() == answer:
            correct += 1
    acc = correct / len(ablation_pool) * 100
    bar = "█" * int(acc / 5)
    print(f"  N={n_loops}: {correct:3d}/{len(ablation_pool)} = {acc:5.1f}%  {bar}")


# ── BENCHMARK 3: Fresh out-of-distribution examples ──────────────────────────
print("\n" + "─"*65)
print("  BENCHMARK 3 — Fresh OOD examples (never-seen sentences)")
print("─"*65)
ood_samples = [
    # 2-hop variable chains with novel phrasing
    {"text": "Let Q = orange. Set R = Q. R is?\nAnswer:", "answer": "orange"},
    {"text": "Define M as purple. N follows M. What is N?\nAnswer:", "answer": "purple"},
    {"text": "P = green. Q = P. Value of Q?\nAnswer:", "answer": "green"},
    {"text": "A = pink. B = A. C = B. C equals?\nAnswer:", "answer": "pink"},
    {"text": "Set U = brown. V copies U. What is V?\nAnswer:", "answer": "brown"},
    # Spatial chains
    {"text": "The key is in the bag. The bag is on the shelf. Where is the key?\nAnswer:", "answer": "shelf"},
    {"text": "The coin is inside the jar. The jar is in the closet. Where is the coin?\nAnswer:", "answer": "closet"},
    # Arithmetic
    {"text": "Carol has 4 coins. Carol earns 3 coins. Carol now has?\nAnswer:", "answer": "7"},
    {"text": "Tom has 9 apples. Tom spends 4 apples. Tom now has?\nAnswer:", "answer": "5"},
    # Named person chains
    {"text": "Sam likes red. Pat copies Sam's choice. Pat likes?\nAnswer:", "answer": "red"},
    {"text": "Kim chose blue. Lee matched Kim's pick. Lee picked?\nAnswer:", "answer": "blue"},
]
ood_correct = 0
for item in ood_samples:
    pred, loops, _, ms = predict(item["text"], max_loops=4)
    ok = pred.lower().strip() == item["answer"]
    if ok:
        ood_correct += 1
    status = "✅" if ok else "❌"
    prompt_short = item["text"][:50].replace("\nAnswer:", "").strip()
    print(f"  {status} [{loops}loops] pred={pred!r:10s} ans={item['answer']!r:8s} | {prompt_short!r}")

print(f"\n  OOD accuracy: {ood_correct}/{len(ood_samples)} = {ood_correct/len(ood_samples)*100:.0f}%")


# ── BENCHMARK 4: Live reasoning trace ────────────────────────────────────────
print("\n" + "─"*65)
print("  BENCHMARK 4 — Live reasoning trace (internal loop progression)")
print("─"*65)
demo_cases = [
    "X = blue. Y = X. Z = Y. What is Z?\nAnswer:",
    "The ball is in the cup. The cup is in the box. Where is the ball?\nAnswer:",
    "Alice has 6 apples. Bob gives 3 apples to Alice. Alice now has?\nAnswer:",
    "Let A = red. Set B = A. C copies B. C is?\nAnswer:",
    "dave likes yellow. Eve copies Dave's color. What does Eve like?\nAnswer:",
]
for prompt in demo_cases:
    pred, loops, trace, ms = predict(prompt, max_loops=4)
    print(f"\n  Prompt: {prompt.strip()!r}")
    print(f"  Trace:  {trace}")
    print(f"  Answer: {pred!r}  ({loops} loop{'s' if loops != 1 else ''}, {ms:.1f}ms)")

print("\n" + "="*65)
print("  Benchmark complete.")
print("="*65)
