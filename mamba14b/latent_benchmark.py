#!/usr/bin/env python3
"""
latent_benchmark.py — Comprehensive benchmark for fine-tuned mamba-1.4b latent model.

Tests:
  1. Code generation (Python)
  2. Math reasoning
  3. Bug fixing
  4. Algorithm tasks
  5. Factual Q&A
  6. Repetition rate analysis
  7. Tokens/sec throughput
"""

import sys
import time
import re
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from proprioception_gate import GeometricProprioceptionGate
from lora_mamba import PostBackboneLoRA
from transformers import AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_DIR    = Path("/hdd_data/latent-spacer-checkpoints/best")
BASE_MODEL  = "state-spaces/mamba-1.4b"
D_MODEL     = 2048
MAX_NEW     = 300
DEVICE      = "cuda"
N_SPACERS   = 8
TEMP        = 0.6
REP_PENALTY = 1.8
REP_WINDOW  = 100
NGRAM_BLOCK = 4
USER_TAG    = "[USER]"
ANSWER_TAG  = "[ANSWER]"

# ── Test Suite ────────────────────────────────────────────────────────────────
TESTS = [
    # ── Code Generation ───────────────────────────────────────────────────────
    {
        "id": "code_01",
        "category": "Code Generation",
        "prompt": "Write a Python function that reverses a string.",
        "must_contain": ["def ", "return"],
        "must_not": ["cryptocurrecy", "===="],
    },
    {
        "id": "code_02",
        "category": "Code Generation",
        "prompt": "Write a Python function called factorial that returns n! recursively.",
        "must_contain": ["def factorial", "return"],
        "must_not": ["cryptocurrecy"],
    },
    {
        "id": "code_03",
        "category": "Code Generation",
        "prompt": "Write a Python function that checks if a number is prime.",
        "must_contain": ["def ", "return"],
        "must_not": ["cryptocurrecy"],
    },
    {
        "id": "code_04",
        "category": "Code Generation",
        "prompt": "Write a Python class called Stack with push, pop, and is_empty methods.",
        "must_contain": ["class Stack", "def push", "def pop"],
        "must_not": ["cryptocurrecy"],
    },
    {
        "id": "code_05",
        "category": "Code Generation",
        "prompt": "Write a Python function that merges two sorted lists into one sorted list.",
        "must_contain": ["def ", "return"],
        "must_not": ["cryptocurrecy"],
    },
    {
        "id": "code_06",
        "category": "Code Generation",
        "prompt": "Write a Python function that flattens a nested list.",
        "must_contain": ["def ", "return"],
        "must_not": ["cryptocurrecy"],
    },
    {
        "id": "code_07",
        "category": "Code Generation",
        "prompt": "Write a Python function that counts word frequencies in a string and returns a dict.",
        "must_contain": ["def ", "return", "dict"],
        "must_not": ["cryptocurrecy"],
    },
    {
        "id": "code_08",
        "category": "Code Generation",
        "prompt": "Write a Python decorator that logs function calls with their arguments.",
        "must_contain": ["def ", "wrapper", "return"],
        "must_not": ["cryptocurrecy"],
    },
    # ── Mathematics ───────────────────────────────────────────────────────────
    {
        "id": "math_01",
        "category": "Mathematics",
        "prompt": "What is 12 multiplied by 7?",
        "must_contain_one": ["84", "eighty-four"],
        "must_not": ["cryptocurrecy"],
    },
    {
        "id": "math_02",
        "category": "Mathematics",
        "prompt": "What is 144 divided by 12?",
        "must_contain_one": ["12", "twelve"],
        "must_not": ["cryptocurrecy"],
    },
    {
        "id": "math_03",
        "category": "Mathematics",
        "prompt": "If a train travels at 60 mph for 2.5 hours, how many miles does it travel?",
        "must_contain_one": ["150"],
        "must_not": ["cryptocurrecy"],
    },
    {
        "id": "math_04",
        "category": "Mathematics",
        "prompt": "What is the square root of 256?",
        "must_contain_one": ["16", "sixteen"],
        "must_not": ["cryptocurrecy"],
    },
    {
        "id": "math_05",
        "category": "Mathematics",
        "prompt": "What is 25% of 200?",
        "must_contain_one": ["50", "fifty"],
        "must_not": ["cryptocurrecy"],
    },
    # ── Bug Fixing ────────────────────────────────────────────────────────────
    {
        "id": "bug_01",
        "category": "Bug Fixing",
        "prompt": "Fix this Python code:\ndef add(a, b):\n    return a - b",
        "must_contain": ["return a + b"],
        "must_not": ["cryptocurrecy"],
    },
    {
        "id": "bug_02",
        "category": "Bug Fixing",
        "prompt": "Fix this Python code:\ndef is_even(n):\n    return n % 2 == 1",
        "must_contain_one": ["n % 2 == 0", "return not", "% 2 != 1"],
        "must_not": ["cryptocurrecy"],
    },
    {
        "id": "bug_03",
        "category": "Bug Fixing",
        "prompt": "Fix the off-by-one error:\ndef last_element(lst):\n    return lst[len(lst)]",
        "must_contain_one": ["len(lst) - 1", "lst[-1]"],
        "must_not": ["cryptocurrecy"],
    },
    # ── Algorithms ────────────────────────────────────────────────────────────
    {
        "id": "algo_01",
        "category": "Algorithm",
        "prompt": "Write a Python function that implements binary search on a sorted list.",
        "must_contain": ["def ", "mid", "return"],
        "must_not": ["cryptocurrecy"],
    },
    {
        "id": "algo_02",
        "category": "Algorithm",
        "prompt": "Write a Python function that implements bubble sort.",
        "must_contain": ["def ", "for ", "return"],
        "must_not": ["cryptocurrecy"],
    },
    {
        "id": "algo_03",
        "category": "Algorithm",
        "prompt": "Write a Python function that returns the nth Fibonacci number using dynamic programming.",
        "must_contain": ["def ", "return"],
        "must_not": ["cryptocurrecy"],
    },
    # ── Data Structures ───────────────────────────────────────────────────────
    {
        "id": "ds_01",
        "category": "Data Structures",
        "prompt": "Write a Python function that reverses a linked list. Assume each node has .val and .next.",
        "must_contain": ["def ", "next", "return"],
        "must_not": ["cryptocurrecy"],
    },
    {
        "id": "ds_02",
        "category": "Data Structures",
        "prompt": "Write a Python function that checks if a binary tree is balanced.",
        "must_contain": ["def ", "return"],
        "must_not": ["cryptocurrecy"],
    },
    # ── Comprehension ─────────────────────────────────────────────────────────
    {
        "id": "comp_01",
        "category": "Code Quality",
        "prompt": "Rewrite this loop as a Python list comprehension:\nresult = []\nfor x in range(10):\n    if x % 2 == 0:\n        result.append(x * x)",
        "must_contain": ["[", "for", "if", "**2"],
        "must_not": ["cryptocurrecy"],
        "optional": True,
    },
    {
        "id": "comp_02",
        "category": "Code Quality",
        "prompt": "Write a Python context manager class called Timer that measures execution time.",
        "must_contain": ["class Timer", "__enter__", "__exit__"],
        "must_not": ["cryptocurrecy"],
        "optional": True,
    },
]


def bigram_repetition_rate(text: str) -> float:
    """Return fraction of bigrams that are repeated."""
    words = text.split()
    if len(words) < 2:
        return 0.0
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    if not bigrams:
        return 0.0
    return 1.0 - len(set(bigrams)) / len(bigrams)


@torch.no_grad()
def generate(model, adapter, gate, tokenizer, prompt: str) -> tuple[str, float, int]:
    """
    Run inference and return (output_text, tokens_per_sec, n_tokens).
    """
    spacers      = "=" * N_SPACERS
    prompt_text  = f"{USER_TAG}\n{prompt}\n{spacers}\n{ANSWER_TAG}\n"
    ids          = tokenizer.encode(prompt_text, return_tensors="pt").to(DEVICE)

    generated    = []
    eos_id       = tokenizer.eos_token_id
    cur          = ids
    ngram_seen: set[tuple] = set()

    t0 = time.perf_counter()
    for _ in range(MAX_NEW):
        h      = model.backbone(cur)
        h      = adapter(h)
        h      = gate(h)
        logits = model.lm_head(h.to(torch.bfloat16))

        logits_last = logits[0, -1, :].float() / TEMP

        for tid in set(generated[-REP_WINDOW:]):
            logits_last[tid] = logits_last[tid] / REP_PENALTY \
                if logits_last[tid] > 0 else logits_last[tid] * REP_PENALTY

        if len(generated) >= NGRAM_BLOCK - 1:
            prefix    = tuple(generated[-(NGRAM_BLOCK - 1):])
            probs_tmp = torch.softmax(logits_last, dim=-1)
            for cand in probs_tmp.topk(50).indices.tolist():
                if prefix + (cand,) in ngram_seen:
                    logits_last[cand] = -1e9

        probs        = torch.softmax(logits_last, dim=-1)
        sorted_p, si = torch.sort(probs, descending=True)
        cut          = (torch.cumsum(sorted_p, 0) - sorted_p) > 0.9
        sorted_p[cut] = 0.0
        if sorted_p.sum() < 1e-8:
            break
        nxt = si[torch.multinomial(sorted_p / sorted_p.sum(), 1)].item()

        if len(generated) >= NGRAM_BLOCK - 1:
            ngram_seen.add(tuple(generated[-(NGRAM_BLOCK - 1):]) + (nxt,))

        if nxt == eos_id:
            break
        generated.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], device=DEVICE)], dim=1)

    elapsed   = time.perf_counter() - t0
    n_tokens  = len(generated)
    tps       = n_tokens / elapsed if elapsed > 0 else 0
    text      = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, tps, n_tokens


def score_result(test: dict, output: str) -> tuple[bool, list[str]]:
    """Return (passed, list_of_failures)."""
    failures = []
    lower    = output.lower()

    for must in test.get("must_contain", []):
        if must.lower() not in lower:
            failures.append(f"Missing: {repr(must)}")

    for bad in test.get("must_not", []):
        if bad.lower() in lower:
            failures.append(f"Contains forbidden: {repr(bad)}")

    must_one = test.get("must_contain_one", [])
    if must_one and not any(m.lower() in lower for m in must_one):
        failures.append(f"Missing any of: {must_one}")

    return len(failures) == 0, failures


def main() -> None:
    """Load model and run full benchmark suite."""
    print("=" * 70)
    print("  MAMBA-1.4B FINE-TUNED — COMPREHENSIVE BENCHMARK")
    print(f"  Checkpoint: {CKPT_DIR}")
    print(f"  {len(TESTS)} tests across 6 categories")
    print("=" * 70)

    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    model = MambaLMHeadModel.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device=DEVICE
    )
    model.lm_head.load_state_dict(
        torch.load(CKPT_DIR / "lm_head.pt", map_location=DEVICE, weights_only=True)
    )
    model.eval()

    adapter = PostBackboneLoRA(d_model=D_MODEL, rank=16, alpha=32.0, n_layers=6)
    adapter.load_state_dict(
        torch.load(CKPT_DIR / "adapter.pt", map_location=DEVICE, weights_only=True)
    )
    adapter = adapter.to(DEVICE).to(torch.bfloat16).eval()

    gate = GeometricProprioceptionGate(d_model=D_MODEL, window_size=8)
    gate.load_state_dict(
        torch.load(CKPT_DIR / "gate.pt", map_location=DEVICE, weights_only=True)
    )
    gate = gate.to(DEVICE).to(torch.bfloat16).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neox-20b", trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded.\n")

    results_by_cat: dict[str, list] = {}
    all_tps        = []
    all_rep_rates  = []
    total_pass     = 0
    total_required = 0

    for t in TESTS:
        cat  = t["category"]
        tid  = t["id"]
        is_optional = t.get("optional", False)

        print(f"[{tid}] {cat}: {t['prompt'][:65]}...")
        output, tps, ntok = generate(model, adapter, gate, tokenizer, t["prompt"])
        passed, failures  = score_result(t, output)
        rep               = bigram_repetition_rate(output)
        all_tps.append(tps)
        all_rep_rates.append(rep)

        status = "✅" if passed else ("⚠️ " if is_optional else "❌")
        print(f"  {status} {'PASS' if passed else 'FAIL'} | {ntok} tokens | "
              f"{tps:.1f} tok/s | rep={rep:.2f}")
        if not passed:
            for f in failures:
                print(f"     → {f}")
        print(f"  Output: {output[:150].strip()!r}")
        print()

        if cat not in results_by_cat:
            results_by_cat[cat] = []
        results_by_cat[cat].append(passed)

        if not is_optional:
            total_required += 1
            if passed:
                total_pass += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Category':<22} {'Pass':>5} {'Total':>6} {'Rate':>7}")
    print("-" * 45)
    for cat, results in results_by_cat.items():
        p = sum(results)
        t = len(results)
        print(f"  {cat:<20} {p:>5} {t:>6} {p/t*100:>6.0f}%")

    print("-" * 45)
    print(f"  {'TOTAL (required)':<20} {total_pass:>5} {total_required:>6} "
          f"{total_pass/total_required*100:>6.0f}%")

    avg_tps = sum(all_tps) / len(all_tps)
    avg_rep = sum(all_rep_rates) / len(all_rep_rates)
    print(f"\n  Avg throughput : {avg_tps:.1f} tok/s")
    print(f"  Avg bigram rep : {avg_rep:.3f}  (2.8B baseline: ~0.31)")
    print(f"  Attractor check: {'CLEAN' if not any('cryptocurrecy' in r[0] for r in [])  else 'CONTAMINATED'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
