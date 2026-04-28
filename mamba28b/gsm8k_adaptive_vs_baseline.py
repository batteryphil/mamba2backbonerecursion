#!/usr/bin/env python3
"""
gsm8k_adaptive_vs_baseline.py
==============================
Compares two inference modes on real GSM8K test problems:

  BASELINE  — hard-stop at loop 1 (single forward pass, no reasoning)
  ADAPTIVE  — HaltingHead chooses depth naturally (the engine as designed)

This directly answers: does adaptive compute improve over single-pass?
GSM8K is a standard benchmark, generative format, not in the SFT training data.
"""

import re
import json
import random
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

REPO_ID     = "batteryphil/mamba-2.8b-latent"
N_PROBLEMS  = 200
HALT_THRESH = 0.70
MAX_LOOPS   = 25       # math domain cap
MAX_NEW     = 80
SEED        = 43


class HaltingHead(nn.Module):
    """3-layer MLP halting probe."""
    def __init__(self, d_input=2561):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        """Return halt probability."""
        return self.net(x).squeeze(-1)


def load_engine():
    """Load model, tokenizer, and HaltingHead from HuggingFace."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32
    use_remote = (device == "cuda")

    print(f"Device: {device}")
    tok = AutoTokenizer.from_pretrained(REPO_ID, trust_remote_code=use_remote)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        REPO_ID, trust_remote_code=use_remote, dtype=dtype, device_map=device
    )
    model.eval()

    head_path = hf_hub_download(repo_id=REPO_ID, filename="halting_head.pt")
    ckpt = torch.load(head_path, weights_only=True, map_location="cpu")
    head = HaltingHead(ckpt["d_input"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()  # stays on CPU — works in all modes

    return tok, model, head, device


def extract_answer(solution: str) -> str:
    """Pull the numeric answer from GSM8K '#### 42' format."""
    m = re.search(r"####\s*([\d,]+)", solution)
    return m.group(1).replace(",", "") if m else ""


def check_answer(output: str, expected: str) -> bool:
    """Check if expected integer appears in model output."""
    expected = expected.strip()
    for n in re.findall(r"\d[\d,]*", output):
        if n.replace(",", "") == expected:
            return True
    return False


def run_baseline(prompt, tok, model, device):
    """Single forward pass — no reasoning loops."""
    with torch.no_grad():
        toks = tok(prompt, return_tensors="pt",
                   truncation=True, max_length=512).to(device)
        out  = model.generate(**toks, max_new_tokens=MAX_NEW,
                              do_sample=False, repetition_penalty=1.1)
    return tok.decode(out[0][toks["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


def run_adaptive(prompt, tok, model, head, device):
    """
    Let the HaltingHead choose depth naturally.
    Returns (output, loops_used, final_p_halt).
    """
    with torch.no_grad():
        loops_used = 1
        p_final    = 0.0
        for lp in range(MAX_LOOPS):
            toks = tok(
                prompt + "=" * lp,
                return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            out  = model(**toks, output_hidden_states=True)
            h    = out.hidden_states[-1][0, -1, :].float().cpu()
            ln   = torch.tensor([lp / MAX_LOOPS], dtype=torch.float32)
            feat = torch.cat([h, ln]).unsqueeze(0)
            p    = head(feat).item()
            if p >= HALT_THRESH:
                loops_used = lp + 1
                p_final    = p
                break

        gen_out = model.generate(
            **toks, max_new_tokens=MAX_NEW,
            do_sample=False, repetition_penalty=1.1
        )

    answer = tok.decode(
        gen_out[0][toks["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    return answer, loops_used, round(p_final, 3)


def main():
    """Run both modes on GSM8K and print comparison table."""
    print("Loading GSM8K test split...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    random.seed(SEED)
    indices  = random.sample(range(len(ds)), N_PROBLEMS)
    problems = [ds[i] for i in indices]
    print(f"Sampled {N_PROBLEMS} / {len(ds)} test problems\n")

    tok, model, head, device = load_engine()
    print()

    baseline_correct = []
    adaptive_correct = []
    adaptive_depths  = []

    print(f"{'#':>3}  {'Exp':>6}  {'B':>2}  {'A':>2}  {'Loops':>5}  "
          f"{'Baseline output (trunc)':40}  Adaptive output (trunc)")
    print("-" * 110)

    for i, prob in enumerate(problems):
        question = prob["question"].strip()
        expected = extract_answer(prob["answer"])
        prompt   = f"[LOGIC] {question}\nAnswer:"

        b_out = run_baseline(prompt, tok, model, device)
        a_out, loops, p = run_adaptive(prompt, tok, model, head, device)

        b_ok = check_answer(b_out, expected)
        a_ok = check_answer(a_out, expected)

        baseline_correct.append(b_ok)
        adaptive_correct.append(a_ok)
        adaptive_depths.append(loops)

        print(f"{i+1:>3}  {expected:>6}  "
              f"{'✅' if b_ok else '❌':>2}  {'✅' if a_ok else '❌':>2}  "
              f"{loops:>5}  "
              f"{b_out[:40]:40}  {a_out[:40]}")

    # ── Summary ──────────────────────────────────────────────────────────
    b_acc    = sum(baseline_correct) / N_PROBLEMS * 100
    a_acc    = sum(adaptive_correct) / N_PROBLEMS * 100
    avg_loop = sum(adaptive_depths)  / N_PROBLEMS
    delta    = a_acc - b_acc

    print("\n" + "=" * 75)
    print("RESULT: ADAPTIVE vs SINGLE-PASS on GSM8K (generative, OOD)")
    print("=" * 75)
    print(f"  Baseline  (1 loop, no HaltingHead):   {b_acc:.1f}%  ({sum(baseline_correct)}/{N_PROBLEMS})")
    print(f"  Adaptive  (HaltingHead chooses):       {a_acc:.1f}%  ({sum(adaptive_correct)}/{N_PROBLEMS})")
    print(f"  Delta:                                {'+' if delta>=0 else ''}{delta:.1f}%")
    print(f"  Avg loops used (adaptive):             {avg_loop:.1f}")
    print("=" * 75)

    print("\n\nREDDIT REPLY:")
    print("-" * 75)
    print(f"Re-ran on {N_PROBLEMS} randomly sampled GSM8K test problems — real benchmark,")
    print("generative format, not in the SFT training data.\n")
    print(f"The right comparison isn't loop-N vs loop-M. It's:")
    print(f"  Single-pass (1 loop, no HaltingHead):  {b_acc:.0f}%")
    print(f"  Adaptive    (HaltingHead chooses):      {a_acc:.0f}%  (avg {avg_loop:.1f} loops)")
    print(f"  Delta: {'+' if delta>=0 else ''}{delta:.0f}%\n")
    print("The model is being evaluated as designed: it decides when it's done.")

    with open("gsm8k_adaptive_results.json", "w") as f:
        json.dump({
            "n_problems":  N_PROBLEMS,
            "baseline_acc": b_acc,
            "adaptive_acc": a_acc,
            "delta":        delta,
            "avg_loops":    avg_loop,
            "depths":       adaptive_depths,
        }, f, indent=2)
    print("\nSaved: gsm8k_adaptive_results.json")


if __name__ == "__main__":
    main()
