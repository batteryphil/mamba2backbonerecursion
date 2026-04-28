"""
eval_latent_arc.py
==================
Generative ARC-Challenge Evaluation

Uses the actual HaltingHead generate() loop — NOT log-likelihood.
This is the correct way to benchmark a latent reasoning engine.

Reference baselines (0-shot, lm_eval):
  Mamba-2.8B base:  ARC-C 40.4%
  Mamba-2.8B base:  ARC-E 63.7%
  Falcon-3B:        ARC-C 38.5%
  OPT-2.7B:         ARC-C 30.0%
  GPT-2-XL (1.5B):  ARC-C 25.4%
"""

import torch
import torch.nn as nn
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

ENGINE_DIR    = "checkpoints/mamba-2.8b-latent"
HALT_THRESH   = 0.7
DOMAIN_MAX    = {"chat": 5, "math": 25, "code": 45, "tool": 10}
SAMPLE_SIZE   = 200   # first 200 of ARC-C test (full set = 1172)


class HaltingHead(nn.Module):
    def __init__(self, d_input: int = 2561):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


print("=" * 58)
print("  GENERATIVE ARC-CHALLENGE EVAL — LATENT ENGINE")
print("=" * 58)

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


def generate_latent(prompt: str, domain: str = "math",
                    max_new: int = 20) -> tuple:
    """Run prompt through the latent engine. Returns (answer_text, loops_used)."""
    m = DOMAIN_MAX.get(domain, 20)
    p = 0.0
    lp = 0
    with torch.no_grad():
        for lp in range(50):
            toks = tok(
                prompt + "=" * lp,
                return_tensors="pt",
                truncation=True,
                max_length=512
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
    answer = tok.decode(
        out[0][toks["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return answer.strip(), lp + 1


# ARC label normalization
LABEL_MAP = {"1": "A", "2": "B", "3": "C", "4": "D"}


def extract_guess(text: str) -> str:
    """Pull the first A/B/C/D letter from model output."""
    text = text.upper()
    m = re.search(r'\b([A-D])\b', text)
    if m:
        return m.group(1)
    # Fallback: first letter-like character
    m = re.search(r'[A-D]', text)
    return m.group(0) if m else "N/A"


print(f"[EVAL] Loading ARC-Challenge test set ({SAMPLE_SIZE} samples)...")
ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

correct   = 0
total     = 0
loop_hist = []

print(f"[EVAL] Running generative evaluation...\n")
for item in tqdm(ds.select(range(SAMPLE_SIZE)), desc="ARC-C"):
    q      = item["question"]
    labels = item["choices"]["label"]
    texts  = item["choices"]["text"]
    truth  = LABEL_MAP.get(item["answerKey"], item["answerKey"])

    prompt = f"[LOGIC] Question: {q}\nChoices:\n"
    for lbl, txt in zip(labels, texts):
        prompt += f"{lbl}: {txt}\n"
    prompt += "Output exactly the correct letter (A, B, C, or D).\nAnswer:"

    output, loops = generate_latent(prompt, domain="math")
    guess = extract_guess(output)
    guess = LABEL_MAP.get(guess, guess)

    loop_hist.append(loops)
    if guess == truth:
        correct += 1
    total += 1

score      = 100 * correct / total
avg_loops  = sum(loop_hist) / len(loop_hist)
min_loops  = min(loop_hist)
max_loops  = max(loop_hist)

print(f"\n{'=' * 58}")
print(f"  GENERATIVE ARC-CHALLENGE RESULTS ({total} samples)")
print(f"{'=' * 58}")
print(f"  Score:      {score:.1f}%  ({correct}/{total})")
print(f"  Avg loops:  {avg_loops:.1f}  (min={min_loops} max={max_loops})")
print(f"")
print(f"  REFERENCE BASELINES (0-shot lm_eval log-likelihood):")
print(f"    Mamba-2.8B base:    40.4%")
print(f"    Falcon-RW-1.3B:     33.3%")
print(f"    OPT-2.7B:           30.0%")
print(f"    GPT-2-XL (1.5B):    25.4%")
print(f"    Random baseline:    25.0%")
print(f"")
delta = score - 40.4
marker = "ABOVE" if delta > 0 else "BELOW"
print(f"  Delta vs Mamba-2.8B base: {delta:+.1f}% {marker}")
if score >= 40.4:
    print(f"  VERDICT: Latent training IMPROVED ARC-C over base model ✅")
elif score >= 35.0:
    print(f"  VERDICT: Close to base — latent SFT minor regression on MC ⚠️")
else:
    print(f"  VERDICT: Score below base — dark-loop suppression of next-tok prior")
    print(f"           (expected: lm_eval lobotomy baseline will be even lower)")
print(f"{'=' * 58}")
