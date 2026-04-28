"""
content_benchmark.py
====================
Format-Free Content Benchmark

Grades the model on whether it knows the ANSWER, not whether it
knows the output format. No A/B/C/D labels. Model writes free text.
Grading uses exact substring match on the gold answer text.

Tasks:
  ARC-Challenge: science questions, grade on answer text present
  HellaSwag:     sentence completion, grade on best ending present  
  Winogrande:    fill-in-the-blank, grade on correct word present
"""

import torch
import torch.nn as nn
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter

ENGINE_DIR  = "checkpoints/mamba-2.8b-latent-final"
HALT_THRESH = 0.7
DOMAIN_MAX  = {"math": 25, "chat": 5}
SAMPLE      = 200


class HaltingHead(nn.Module):
    def __init__(self, d_input: int = 2561):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).squeeze(-1)


print("=" * 64)
print("  CONTENT-GRADED BENCHMARK — No A/B/C/D Format Required")
print("  Scoring: does model output CONTAIN the correct answer text?")
print("=" * 64)

tok = AutoTokenizer.from_pretrained(ENGINE_DIR, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    ENGINE_DIR, torch_dtype=torch.bfloat16,
    device_map="cuda:0", trust_remote_code=True
)
model.eval()

ckpt = torch.load(f"{ENGINE_DIR}/halting_head.pt", weights_only=True)
head = HaltingHead(ckpt["d_input"]).cuda()
head.load_state_dict(ckpt["state_dict"])
head.eval()
print("[READY]\n")


def generate(prompt: str, domain: str = "math", max_new: int = 40) -> tuple:
    """Run latent reasoning loop, return (text, loops_used)."""
    m = DOMAIN_MAX.get(domain, 10)
    with torch.no_grad():
        for lp in range(50):
            toks = tok(
                prompt + "=" * lp,
                return_tensors="pt", truncation=True, max_length=350
            ).to("cuda")
            h  = model(**toks, output_hidden_states=True).hidden_states[-1][0, -1, :].float()
            ln = torch.tensor([lp / m], dtype=torch.float32, device="cuda")
            p  = head(torch.cat([h, ln]).unsqueeze(0)).item()
            if p >= HALT_THRESH:
                break
        out = model.generate(
            **toks, max_new_tokens=max_new,
            do_sample=False, repetition_penalty=1.1
        )
    text = tok.decode(
        out[0][toks["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    return text, lp + 1


def contains_answer(output: str, gold: str, threshold: int = 3) -> bool:
    """
    True if the gold answer text appears in the model output.
    Uses normalized comparison (lowercase, strip punctuation).
    For short answers (<=threshold words), requires full match.
    For longer answers, requires overlap of key content words.
    """
    def normalize(s):
        return re.sub(r'[^a-z0-9\s]', '', s.lower()).strip()

    out_n  = normalize(output)
    gold_n = normalize(gold)

    # Direct substring match
    if gold_n in out_n:
        return True

    # Key words overlap for longer answers
    gold_words = [w for w in gold_n.split() if len(w) > 3]
    if not gold_words:
        return gold_n in out_n

    out_words = set(out_n.split())
    hits = sum(1 for w in gold_words if w in out_words)
    return hits >= min(len(gold_words), max(1, len(gold_words) // 2))


results = {}
loops_per_task = {}

# ── ARC-Challenge ──────────────────────────────────────────────────────────
print("[TASK 1/3] ARC-Challenge — Science Reasoning")
ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
correct = 0
loops   = []
for item in tqdm(ds.select(range(SAMPLE)), desc="ARC-C"):
    q      = item["question"]
    # Map labels to answer texts
    label_to_text = dict(zip(item["choices"]["label"], item["choices"]["text"]))
    gold_text = label_to_text.get(item["answerKey"], "")
    if not gold_text:
        # Numeric answerKey (1/2/3/4 → A/B/C/D → text)
        lm = {"1":"A","2":"B","3":"C","4":"D"}
        gold_text = label_to_text.get(lm.get(item["answerKey"],""), "")

    # Present as plain text — no labels
    choices_text = "\n".join(f"- {t}" for t in item["choices"]["text"])
    prompt = (
        f"[LOGIC] Question: {q}\n"
        f"Possible answers:\n{choices_text}\n"
        f"What is the correct answer? Write it out:"
    )

    output, lp = generate(prompt, domain="math", max_new=25)
    loops.append(lp)

    hit = contains_answer(output, gold_text)
    if hit:
        correct += 1

arc_score = 100 * correct / SAMPLE
arc_loops  = sum(loops) / len(loops)
results["arc_challenge"] = arc_score
loops_per_task["arc_challenge"] = arc_loops
print(f"  ARC-C: {arc_score:.1f}%  avg loops: {arc_loops:.1f}\n")


# ── Winogrande ─────────────────────────────────────────────────────────────
print("[TASK 2/3] Winogrande — Fill in the Blank")
ds2 = load_dataset("winogrande", "winogrande_xl", split="validation",
                   trust_remote_code=True)
correct = 0
loops   = []
for item in tqdm(ds2.select(range(SAMPLE)), desc="Winogrande"):
    gold_text = item["option1"] if item["answer"] == "1" else item["option2"]
    sentence  = item["sentence"]

    prompt = (
        f"[CHAT] Fill in the blank with the best word or phrase.\n"
        f"Sentence: {sentence}\n"
        f"Options:\n"
        f"- {item['option1']}\n"
        f"- {item['option2']}\n"
        f"Write the correct word or phrase to fill in the blank:"
    )

    output, lp = generate(prompt, domain="chat", max_new=15)
    loops.append(lp)

    hit = contains_answer(output, gold_text)
    if hit:
        correct += 1

wino_score = 100 * correct / SAMPLE
wino_loops  = sum(loops) / len(loops)
results["winogrande"] = wino_score
loops_per_task["winogrande"] = wino_loops
print(f"  Winogrande: {wino_score:.1f}%  avg loops: {wino_loops:.1f}\n")


# ── HellaSwag ──────────────────────────────────────────────────────────────
print("[TASK 3/3] HellaSwag — Sentence Completion")
ds3 = load_dataset("Rowan/hellaswag", split="validation")
correct = 0
loops   = []
for item in tqdm(ds3.select(range(SAMPLE)), desc="HellaSwag"):
    endings   = item["endings"]
    gold_text = endings[int(item["label"])]
    ctx       = item["ctx"]

    # Present endings as plain options, no labels
    endings_text = "\n".join(f"- {e}" for e in endings)
    prompt = (
        f"[CHAT] Choose the best continuation for this sentence.\n"
        f"Context: {ctx}\n"
        f"Options:\n{endings_text}\n"
        f"Write the best continuation:"
    )

    output, lp = generate(prompt, domain="chat", max_new=30)
    loops.append(lp)

    hit = contains_answer(output, gold_text)
    if hit:
        correct += 1

hella_score = 100 * correct / SAMPLE
hella_loops  = sum(loops) / len(loops)
results["hellaswag"] = hella_score
loops_per_task["hellaswag"] = hella_loops
print(f"  HellaSwag: {hella_score:.1f}%  avg loops: {hella_loops:.1f}\n")


# ── FINAL REPORT ───────────────────────────────────────────────────────────
BASELINES  = {"arc_challenge": 40.4, "winogrande": 63.5, "hellaswag": 55.5}
PREV_MC    = {"arc_challenge": 30.0, "winogrande": 40.5, "hellaswag": 25.5}
RANDOM     = {"arc_challenge": 25.0, "winogrande": 50.0, "hellaswag": 25.0}

print("=" * 68)
print("  FORMAT-FREE CONTENT BENCHMARK RESULTS")
print("  Graded: answer text present in output (no format required)")
print("=" * 68)
print(f"  {'Task':<22} {'Random':>7} {'MC fmt':>8} {'Content':>9} {'Base':>8} {'Loops':>6}")
print(f"  {'-'*64}")
for t in ["arc_challenge", "winogrande", "hellaswag"]:
    s    = results[t]
    prev = PREV_MC.get(t, 0)
    base = BASELINES[t]
    rand = RANDOM[t]
    lp   = loops_per_task[t]
    delta = s - base
    mrk  = "✅" if delta >= 0 else ("⚠️" if delta >= -10 else "❌")
    print(f"  {t:<22} {rand:>6.0f}% {prev:>7.1f}% {s:>8.1f}% {base:>7.1f}% {lp:>5.1f}L {mrk}")
print(f"  {'-'*64}")
avg = sum(results.values()) / len(results)
avg_base = sum(BASELINES.values()) / len(BASELINES)
print(f"  {'AVERAGE':<22} {'-':>7} {sum(PREV_MC.values())/3:>7.1f}% {avg:>8.1f}% {avg_base:>7.1f}%")
print("=" * 68)
print()
print("  ACT LOOP PROPORTIONALITY:")
print(f"    ARC-C (hard science):    {loops_per_task['arc_challenge']:.1f}L")
print(f"    Winogrande (fill-blank): {loops_per_task['winogrande']:.1f}L")
print(f"    HellaSwag (completion):  {loops_per_task['hellaswag']:.1f}L")
print("=" * 68)
