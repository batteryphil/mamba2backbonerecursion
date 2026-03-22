"""
baseline_mamba130m.py
=====================
Run the exact same questions as benchmark_v25_detailed.py against
the raw, unmodified state-spaces/mamba-130m pretrained model.
No LoRA, no recursive loops, no pointer mask, no special tokens.
Just greedy argmax from the frozen pretrained weights.
"""
import sys, os, json, random, time, torch
os.chdir('/home/phil/Desktop/mambadiff/mambadiff llm tts')

from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*70)
print("  BASELINE: state-spaces/mamba-130m  (NO fine-tuning, NO loops)")
print("="*70)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
model     = MambaLMHeadModel.from_pretrained(
    "state-spaces/mamba-130m", dtype=torch.float32, device=DEVICE
)
model.eval()
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}\n")


def predict_base(text: str, max_new: int = 1) -> str:
    """Greedy single-token prediction from the raw backbone."""
    ids = tokenizer.encode(text, add_special_tokens=False, max_length=512, truncation=True)
    inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = model(inp)
    logits = out.logits[0, -1, :]          # last position
    tok_id = logits.argmax().item()
    return tokenizer.decode([tok_id]).strip()


# ── Section 1: Per-hop accuracy (150 samples each, same seed as benchmark) ──
print("━"*70)
print("  SECTION 1 — Per-hop accuracy (greedy argmax, 150 samples)")
print("━"*70)

with open("system2_logic_v1.json") as f:
    data = json.load(f)

hop1 = [s for s in data if s.get("hops") == 1]
hop2 = [s for s in data if s.get("hops") == 2]
hop3 = [s for s in data if s.get("hops") == 3]

def score(pool, n=150, seed=7):
    """Score pool using base model greedy prediction."""
    random.seed(seed)
    samples = random.sample(pool, min(n, len(pool)))
    correct, times = 0, []
    wrong_ex = []
    for item in samples:
        text   = item["text"]
        answer = str(item.get("answer", "")).strip().lower()
        split  = text.rfind("Answer:")
        prompt = text[:split].rstrip() + "\nAnswer:" if split >= 0 else text
        t0   = time.time()
        pred = predict_base(prompt)
        times.append((time.time() - t0)*1000)
        ok = pred.lower().strip() == answer
        if ok:
            correct += 1
        else:
            wrong_ex.append((pred, answer, prompt[-60:]))
    acc = correct / len(samples) * 100
    return correct, len(samples), acc, wrong_ex[:3], sum(times)/len(times)

for label, pool in [("1-hop", hop1), ("2-hop", hop2), ("3-hop", hop3)]:
    c, t, acc, _, avg_ms = score(pool, n=min(150, len(pool)))
    bar = "█" * int(acc / 5)
    print(f"  {label}: {c:3d}/{t}  {acc:5.1f}%  {bar}")


# ── Section 2: Task-type breakdown ──────────────────────────────────────────
import collections
by_type = collections.defaultdict(list)
for item in data:
    t = item["text"]
    if "apple" in t or "coin" in t or "earns" in t or "spends" in t or "Change:" in t:
        by_type["arithmetic"].append(item)
    elif "in the" in t or "inside" in t or "on the" in t.lower():
        by_type["spatial"].append(item)
    elif "owns" in t or "have" in t or "pet" in t or "has a?" in t:
        by_type["property"].append(item)
    elif "likes" in t.lower() or "chose" in t or "matched" in t or "copies" in t:
        by_type["name_chain"].append(item)
    else:
        by_type["variable_binding"].append(item)

print()
print("━"*70)
print("  SECTION 2 — Task-type breakdown (150 samples each)")
print("━"*70)
for tname, pool in sorted(by_type.items()):
    if not pool:
        continue
    c, t, acc, wrongs, _ = score(pool, n=150, seed=13)
    bar = "█" * int(acc / 5)
    print(f"  {tname:20s}: {c:3d}/{t}  {acc:5.1f}%  {bar}")


# ── Section 3: Exact OOD prompts from benchmark ──────────────────────────────
print()
print("━"*70)
print("  SECTION 3 — Exact OOD prompts (same 16 as benchmark)")
print("━"*70)
ood = [
    {"t": "Variable Q holds orange. Variable R is set to Q. What is R?\nAnswer:", "a": "orange"},
    {"t": "Let G equal blue. H is assigned the value of G. H equals?\nAnswer:", "a": "blue"},
    {"t": "M = pink. N = M. O = N. What is O?\nAnswer:", "a": "pink"},
    {"t": "Define T as purple. U copies T. V mirrors U. V is?\nAnswer:", "a": "purple"},
    {"t": "X is red. Y is the same as X. Z is the same as Y. Z?\nAnswer:", "a": "red"},
    {"t": "The hat is inside the drawer. The drawer is in the closet. Where is the hat?\nAnswer:", "a": "closet"},
    {"t": "My pen is on the desk. The desk is in the office. The pen is in the?\nAnswer:", "a": "office"},
    {"t": "The coin is in the jar. The jar is in the bag. The bag is on the shelf. The coin is on the?\nAnswer:", "a": "shelf"},
    {"t": "Tom likes green. Sue copies Tom. Sue likes?\nAnswer:", "a": "green"},
    {"t": "Zoe chose yellow. Max copies Zoe's pick. Max chose?\nAnswer:", "a": "yellow"},
    {"t": "Anna likes blue. Ben copies Anna. Cal copies Ben. Cal likes?\nAnswer:", "a": "blue"},
    {"t": "Start: X=3. Change: +4. End: X=?\nAnswer:", "a": "7"},
    {"t": "Start: X=8. Change: -3. End: X=?\nAnswer:", "a": "5"},
    {"t": "Mike has 5 apples. Mike earns 2 apples. Mike now has?\nAnswer:", "a": "7"},
    {"t": "Lisa has 9 coins. Lisa spends 4 coins. Lisa now has?\nAnswer:", "a": "5"},
    {"t": "Start: Y=6. Change: -6. End: Y=?\nAnswer:", "a": "0"},
]

ood_correct = 0
for item in ood:
    pred = predict_base(item["t"])
    ok   = pred.lower().strip() == item["a"]
    if ok: ood_correct += 1
    status = "✅" if ok else "❌"
    print(f"  {status} pred={pred!r:10s} ans={item['a']!r:6s} | {item['t'][:50].strip()!r}")

print(f"\n  OOD total: {ood_correct}/{len(ood)} = {ood_correct/len(ood)*100:.0f}%")


# ── Section 4: Same MMLU-style questions ─────────────────────────────────────
print()
print("━"*70)
print("  SECTION 4 — MMLU-style multiple-choice (same 4 as benchmark)")
print("━"*70)
mmlu = [
    {"t": "What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:", "a": "B"},
    {"t": "Which planet is smallest?\nA. Venus\nB. Earth\nC. Mercury\nD. Mars\nAnswer:", "a": "C"},
    {"t": "Water boils at?\nA. 50C\nB. 100C\nC. 150C\nD. 200C\nAnswer:", "a": "B"},
    {"t": "Capital of France?\nA. Berlin\nB. Rome\nC. Paris\nD. London\nAnswer:", "a": "C"},
]
for item in mmlu:
    pred = predict_base(item["t"])
    ok   = pred.strip().upper() == item["a"]
    status = "✅" if ok else "❌"
    print(f"  {status} pred={pred!r:10s} ans={item['a']!r} | {item['t'][:50].strip()!r}")


print()
print("="*70)
print("  BASELINE COMPLETE")
print("="*70)
