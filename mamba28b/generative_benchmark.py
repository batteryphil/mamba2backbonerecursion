"""
generative_benchmark.py
========================
Full Generative Benchmark — All 4 Tasks with Reasoning Enabled

Runs ARC-Challenge, HellaSwag, PIQA, and Winogrande through the
actual HaltingHead latent loop. Uses aggressive letter extraction
and chain-of-thought forcing to get clean A/B/C/D answers.

Key difference from lm_eval: model.generate() is called, not
log-likelihood, so the dark latent loops fire on every question.
"""

import torch
import torch.nn as nn
import re
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

ENGINE_DIR  = "checkpoints/mamba-2.8b-latent"
HALT_THRESH = 0.7
DOMAIN_MAX  = {"math": 25, "chat": 5}
SAMPLE      = 200   # samples per task


class HaltingHead(nn.Module):
    def __init__(self, d_input: int = 2561):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


print("=" * 62)
print("  FULL GENERATIVE BENCHMARK — ALL 4 TASKS")
print("  (HaltingHead reasoning loops ENABLED)")
print("=" * 62)

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
print("[READY] Engine loaded.\n")

LABEL_MAP = {"1": "A", "2": "B", "3": "C", "4": "D"}


def generate(prompt: str, domain: str = "math", max_new: int = 25) -> tuple:
    """Run the latent reasoning loop and return (text, loops)."""
    m = DOMAIN_MAX.get(domain, 10)
    p = 0.0
    lp = 0
    with torch.no_grad():
        for lp in range(50):
            toks = tok(
                prompt + "=" * lp,
                return_tensors="pt",
                truncation=True,
                max_length=400
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


def extract_letter(text: str, valid: list = None) -> str:
    """Robustly extract A/B/C/D from model output."""
    valid = valid or ["A", "B", "C", "D"]
    t = text.upper().strip()
    # Pattern 1: "The answer is A" / "Answer: B" / "(C)" / "A."
    for pat in [
        r'(?:ANSWER\s*(?:IS)?|THE\s+ANSWER\s+IS)\s*[:\-]?\s*([A-D])\b',
        r'\bANSWER\s*:\s*([A-D])\b',
        r'^\s*([A-D])[\.:\)]\s',
        r'\(([A-D])\)',
        r'\b([A-D])\b',
    ]:
        m = re.search(pat, t)
        if m and m.group(1) in valid:
            return m.group(1)
    return "N/A"


results = {}
loop_totals = {}

# ── TASK 1: ARC-Challenge ──────────────────────────────────────────
print("[TASK 1/4] ARC-Challenge")
ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
correct = 0
loops = []
for item in tqdm(ds.select(range(SAMPLE)), desc="ARC-C"):
    q      = item["question"]
    labels = item["choices"]["label"]
    texts  = item["choices"]["text"]
    truth  = LABEL_MAP.get(item["answerKey"], item["answerKey"])
    prompt = (
        f"[LOGIC] Question: {q}\n"
        + "\n".join(f"{l}: {t}" for l, t in zip(labels, texts))
        + "\nThink step by step. The answer is letter:"
    )
    out, lp = generate(prompt, domain="math")
    guess = extract_letter(out, [LABEL_MAP.get(l, l) for l in labels])
    guess = LABEL_MAP.get(guess, guess)
    if guess == truth:
        correct += 1
    loops.append(lp)
score = 100 * correct / SAMPLE
results["arc_challenge"] = score
loop_totals["arc_challenge"] = sum(loops) / len(loops)
print(f"  ARC-C: {score:.1f}%  avg loops: {loop_totals['arc_challenge']:.1f}\n")

# ── TASK 2: HellaSwag ──────────────────────────────────────────────
print("[TASK 2/4] HellaSwag")
ds = load_dataset("Rowan/hellaswag", split="validation")
correct = 0
loops = []
for item in tqdm(ds.select(range(SAMPLE)), desc="HellaSwag"):
    ctx     = item["ctx"]
    endings = item["endings"]
    truth   = str(int(item["label"]))
    letters = ["A", "B", "C", "D"]
    truth_l = letters[int(truth)] if int(truth) < 4 else "A"
    prompt  = (
        f"[CHAT] Complete this sentence: {ctx}\n"
        + "\n".join(f"{l}: {e}" for l, e in zip(letters, endings))
        + "\nThe best completion is letter:"
    )
    out, lp = generate(prompt, domain="chat")
    guess = extract_letter(out, letters)
    if guess == truth_l:
        correct += 1
    loops.append(lp)
score = 100 * correct / SAMPLE
results["hellaswag"] = score
loop_totals["hellaswag"] = sum(loops) / len(loops)
print(f"  HellaSwag: {score:.1f}%  avg loops: {loop_totals['hellaswag']:.1f}\n")

# ── TASK 3: PIQA ───────────────────────────────────────────────────
print("[TASK 3/4] PIQA")
ds = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
correct = 0
loops = []
for item in tqdm(ds.select(range(SAMPLE)), desc="PIQA"):
    goal  = item["goal"]
    sol1  = item["sol1"]
    sol2  = item["sol2"]
    truth = "A" if item["label"] == 0 else "B"
    prompt = (
        f"[CHAT] Goal: {goal}\n"
        f"A: {sol1}\n"
        f"B: {sol2}\n"
        "Which solution is correct? Answer with A or B:"
    )
    out, lp = generate(prompt, domain="chat", max_new=10)
    guess = extract_letter(out, ["A", "B"])
    if guess == truth:
        correct += 1
    loops.append(lp)
score = 100 * correct / SAMPLE
results["piqa"] = score
loop_totals["piqa"] = sum(loops) / len(loops)
print(f"  PIQA: {score:.1f}%  avg loops: {loop_totals['piqa']:.1f}\n")

# ── TASK 4: Winogrande ─────────────────────────────────────────────
print("[TASK 4/4] Winogrande")
ds = load_dataset("winogrande", "winogrande_xl", split="validation",
                  trust_remote_code=True)
correct = 0
loops = []
for item in tqdm(ds.select(range(SAMPLE)), desc="WinoGrande"):
    sentence = item["sentence"]
    opt1     = item["option1"]
    opt2     = item["option2"]
    truth    = "A" if item["answer"] == "1" else "B"
    prompt   = (
        f"[CHAT] Fill in the blank: {sentence}\n"
        f"A: {opt1}\n"
        f"B: {opt2}\n"
        "The correct word is option A or B:"
    )
    out, lp = generate(prompt, domain="chat", max_new=10)
    guess = extract_letter(out, ["A", "B"])
    if guess == truth:
        correct += 1
    loops.append(lp)
score = 100 * correct / SAMPLE
results["winogrande"] = score
loop_totals["winogrande"] = sum(loops) / len(loops)
print(f"  Winogrande: {score:.1f}%  avg loops: {loop_totals['winogrande']:.1f}\n")

# ── FINAL REPORT ───────────────────────────────────────────────────
BASELINES = {
    "arc_challenge": 40.4,
    "hellaswag":     55.5,
    "piqa":          75.2,
    "winogrande":    63.5,
}

print("=" * 62)
print("  FULL GENERATIVE BENCHMARK RESULTS")
print("  (Reasoning loops ENABLED via HaltingHead)")
print("=" * 62)
print(f"  {'Task':<20} {'Ours':>8} {'Base':>8} {'Delta':>8}  {'Loops':>6}")
print(f"  {'-'*54}")
for task in ["arc_challenge", "hellaswag", "piqa", "winogrande"]:
    s   = results[task]
    b   = BASELINES[task]
    d   = s - b
    lp  = loop_totals[task]
    mrk = "✅" if d >= 0 else "❌"
    print(f"  {task:<20} {s:>7.1f}% {b:>7.1f}% {d:>+7.1f}%  {lp:>5.1f}L  {mrk}")
print(f"  {'-'*54}")
avg_ours = sum(results.values()) / len(results)
avg_base = sum(BASELINES.values()) / len(BASELINES)
print(f"  {'AVERAGE':<20} {avg_ours:>7.1f}% {avg_base:>7.1f}% {avg_ours-avg_base:>+7.1f}%")
print("=" * 62)

# Save
with open("generative_benchmark_results.json", "w") as f:
    json.dump({"results": results, "loops": loop_totals,
               "baselines": BASELINES, "sample": SAMPLE}, f, indent=2)
print("\n  Results saved → generative_benchmark_results.json")
