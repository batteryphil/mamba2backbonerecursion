#!/usr/bin/env python3
"""
rlf_benchmark.py — Extended 54-test benchmark using the RLF inference engine.

Generates answers by running the RLF loop (up to MAX_LOOPS per token)
autoregressively, collecting the last non-HALT trace token as each output token.
"""

import sys, time, json, re
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rlf_engine_1_4b import (
    RecursiveMamba1_PrefixScratchpad,
    load_from_sft_checkpoint,
    HALT_ID, tokenizer, DEVICE, MAX_LOOPS,
)
from rlf_chain_test import load_rlf_final

RLF_CKPT_DIR  = Path("/hdd_data/rlf-1.4b-checkpoints")
SFT_CKPT_DIR  = Path("/hdd_data/latent-spacer-checkpoints/best")
MAX_NEW       = 300
N_SPACERS     = 0      # RLF doesn't need spacers

# ── Same test suite as extended_benchmark.py ──────────────────────────────────
TESTS = [
    {"id":"cg_01","cat":"Code Gen","prompt":"Write a Python function that reverses a string.",
     "must":["def ","return"],"must_one":[]},
    {"id":"cg_02","cat":"Code Gen","prompt":"Write a Python function called factorial that returns n! recursively.",
     "must":["def factorial","return"],"must_one":[]},
    {"id":"cg_03","cat":"Code Gen","prompt":"Write a Python function that checks if a number is prime.",
     "must":["def ","return"],"must_one":[]},
    {"id":"cg_04","cat":"Code Gen","prompt":"Write a Python class called Stack with push, pop, and is_empty methods.",
     "must":["class Stack","def push","def pop"],"must_one":[]},
    {"id":"cg_05","cat":"Code Gen","prompt":"Write a Python function that merges two sorted lists into one sorted list.",
     "must":["def ","return"],"must_one":[]},
    {"id":"cg_06","cat":"Code Gen","prompt":"Write a Python function that flattens a nested list.",
     "must":["def ","return"],"must_one":[]},
    {"id":"cg_07","cat":"Code Gen","prompt":"Write a Python function that counts word frequencies in a string and returns a dict.",
     "must":["def ","return"],"must_one":[]},
    {"id":"cg_08","cat":"Code Gen","prompt":"Write a Python decorator that logs function calls with their arguments.",
     "must":["def ","return","wrapper"],"must_one":[]},
    {"id":"cg_09","cat":"Code Gen (Complex)","prompt":"Write a Python class called LRUCache with get and put methods. Use an OrderedDict.",
     "must":["class LRUCache","def get","def put"],"must_one":[]},
    {"id":"cg_10","cat":"Code Gen (Complex)","prompt":"Write a Python function that implements quicksort.",
     "must":["def ","return"],"must_one":[]},
    {"id":"cg_11","cat":"Code Gen (Complex)","prompt":"Write a Python context manager class called Timer that measures execution time using __enter__ and __exit__.",
     "must":["class Timer","__enter__","__exit__"],"must_one":[]},
    {"id":"cg_12","cat":"Code Gen (Complex)","prompt":"Write a Python generator function that yields Fibonacci numbers indefinitely.",
     "must":["def ","yield"],"must_one":[]},
    {"id":"cg_13","cat":"Code Gen (Complex)","prompt":"Write a Python function that performs matrix multiplication of two 2D lists.",
     "must":["def ","return"],"must_one":[]},
    {"id":"cg_14","cat":"Code Gen (Complex)","prompt":"Write a Python class called BinarySearchTree with insert and search methods.",
     "must":["class BinarySearchTree","def insert","def search"],"must_one":[]},
    {"id":"al_01","cat":"Algorithm","prompt":"Write a Python function that implements binary search on a sorted list.",
     "must":["def ","mid","return"],"must_one":[]},
    {"id":"al_02","cat":"Algorithm","prompt":"Write a Python function that implements bubble sort.",
     "must":["def ","for ","return"],"must_one":[]},
    {"id":"al_03","cat":"Algorithm","prompt":"Write a Python function that returns the nth Fibonacci number using dynamic programming.",
     "must":["def ","return"],"must_one":[]},
    {"id":"al_04","cat":"Algorithm","prompt":"Write a Python function that implements merge sort.",
     "must":["def ","return"],"must_one":[]},
    {"id":"ds_01","cat":"Data Structures","prompt":"Write a Python function that reverses a linked list. Assume each node has .val and .next.",
     "must":["def ","next","return"],"must_one":[]},
    {"id":"ds_02","cat":"Data Structures","prompt":"Write a Python function that checks if a binary tree is balanced.",
     "must":["def ","return"],"must_one":[]},
    {"id":"ds_03","cat":"Data Structures","prompt":"Write a Python function that finds the height of a binary tree.",
     "must":["def ","return"],"must_one":[]},
    {"id":"ds_04","cat":"Data Structures","prompt":"Write a Python function using a queue (collections.deque) to do level-order traversal of a binary tree.",
     "must":["def ","deque","return"],"must_one":[]},
    {"id":"st_01","cat":"Strings","prompt":"Write a Python function that checks if a string is a palindrome.",
     "must":["def ","return"],"must_one":[]},
    {"id":"st_02","cat":"Strings","prompt":"Write a Python function that finds the longest common prefix among a list of strings.",
     "must":["def ","return"],"must_one":[]},
    {"id":"st_03","cat":"Strings","prompt":"Write a Python function that counts the number of vowels in a string.",
     "must":["def ","return"],"must_one":[]},
    {"id":"st_04","cat":"Strings","prompt":"Write a Python function that checks if two strings are anagrams of each other.",
     "must":["def ","return"],"must_one":[]},
    {"id":"ma_01","cat":"Math","prompt":"What is 12 multiplied by 7?","must":[],"must_one":["84"]},
    {"id":"ma_02","cat":"Math","prompt":"What is 144 divided by 12?","must":[],"must_one":["12"]},
    {"id":"ma_03","cat":"Math","prompt":"What is the square root of 256?","must":[],"must_one":["16"]},
    {"id":"ma_04","cat":"Math","prompt":"What is 25% of 200?","must":[],"must_one":["50"]},
    {"id":"ma_05","cat":"Math","prompt":"What is 17 multiplied by 13?","must":[],"must_one":["221"]},
    {"id":"ma_06","cat":"Math","prompt":"What is the square root of 625?","must":[],"must_one":["25"]},
    {"id":"ma_07","cat":"Math","prompt":"What is 15% of 80?","must":[],"must_one":["12"]},
    {"id":"ma_08","cat":"Math","prompt":"What is 2 to the power of 8?","must":[],"must_one":["256"]},
    {"id":"wp_01","cat":"Word Problem","prompt":"A car travels at 60 mph for 2.5 hours. How many miles does it travel?","must":[],"must_one":["150"]},
    {"id":"wp_02","cat":"Word Problem","prompt":"A class has 30 students. 60% are girls. How many boys are in the class?","must":[],"must_one":["12"]},
    {"id":"wp_03","cat":"Word Problem","prompt":"A store sells apples for $0.50 each. Sarah buys 14 apples. How much does she spend?","must":[],"must_one":["7","$7"]},
    {"id":"wp_04","cat":"Word Problem","prompt":"A rectangle has length 15 cm and width 8 cm. What is its area?","must":[],"must_one":["120"]},
    {"id":"wp_05","cat":"Word Problem","prompt":"If a pizza is cut into 8 equal slices and you eat 3 slices, what percentage of the pizza did you eat?","must":[],"must_one":["37.5","37%","3/8"]},
    {"id":"wp_06","cat":"Word Problem","prompt":"A train travels 240 miles in 4 hours. What is its speed in mph?","must":[],"must_one":["60"]},
    {"id":"bf_01","cat":"Bug Fix","prompt":"Fix this Python code:\ndef add(a, b):\n    return a - b",
     "must":[],"must_one":["return a + b"]},
    {"id":"bf_02","cat":"Bug Fix","prompt":"Fix this Python code:\ndef is_even(n):\n    return n % 2 == 1",
     "must":[],"must_one":["n % 2 == 0","% 2 == 0"]},
    {"id":"bf_03","cat":"Bug Fix","prompt":"Fix the off-by-one error:\ndef last_element(lst):\n    return lst[len(lst)]",
     "must":[],"must_one":["len(lst) - 1","lst[-1]"]},
    {"id":"bf_04","cat":"Bug Fix","prompt":"Fix this Python code:\ndef square(x):\n    return x + x",
     "must":[],"must_one":["return x * x","x**2"]},
    {"id":"bf_05","cat":"Bug Fix","prompt":"Fix this Python code:\ndef max_of_two(a, b):\n    if a > b:\n        return b\n    return a",
     "must":[],"must_one":["return a"]},
    {"id":"cc_01","cat":"Completion","prompt":"Complete this Python function:\ndef count_evens(lst):\n    \"\"\"Return count of even numbers in lst.\"\"\"\n    # your code here",
     "must":["return"],"must_one":["% 2"]},
    {"id":"cc_02","cat":"Completion","prompt":"Complete this Python function:\ndef find_max(lst):\n    \"\"\"Return the maximum value in lst without using max().\"\"\"\n    # your code here",
     "must":["return"],"must_one":[]},
    {"id":"cc_03","cat":"Completion","prompt":"Complete this Python function:\ndef flatten(nested):\n    \"\"\"Flatten a list of lists into a single list.\"\"\"\n    result = []\n    # your code here\n    return result",
     "must":["result","return result"],"must_one":["append","extend","for"]},
    {"id":"lg_01","cat":"Logic","prompt":"What is the next number in the sequence: 2, 6, 18, 54, ?",
     "must":[],"must_one":["162"]},
    {"id":"lg_02","cat":"Logic","prompt":"What is the next number in the sequence: 1, 4, 9, 16, ?",
     "must":[],"must_one":["25"]},
    {"id":"lg_03","cat":"Logic","prompt":"If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
     "must":[],"must_one":["5 minute","5 min","same","5m"]},
    {"id":"lg_04","cat":"Logic","prompt":"What is the next number in the sequence: 1, 1, 2, 3, 5, 8, ?",
     "must":[],"must_one":["13"]},
    {"id":"ut_01","cat":"Unit Tests","prompt":"Write a Python unittest class that tests a function called add(a, b) that returns a+b.",
     "must":["import unittest","def test_","assertEqual"],"must_one":[]},
    {"id":"ut_02","cat":"Unit Tests","prompt":"Write pytest test functions that test is_prime(n) for prime and non-prime inputs.",
     "must":["def test_","assert"],"must_one":[]},
]


# ── RLF autoregressive generation ────────────────────────────────────────────

@torch.no_grad()
def rlf_generate(
    model: RecursiveMamba1_PrefixScratchpad,
    prompt: str,
    max_new: int = MAX_NEW,
) -> tuple[str, float, int]:
    """Generate text autoregressively using the RLF loop engine.

    Each output token is determined by running up to MAX_LOOPS reasoning
    iterations on the current sequence state. The last non-HALT trace
    token is appended to the sequence.

    Returns: (generated_text, tok_per_sec, n_tokens_generated)
    """
    prefix  = f"[USER]\n{prompt}\n[ANSWER]\n"
    ids     = tokenizer.encode(prefix)
    gen     = []
    t0      = time.perf_counter()

    for _ in range(max_new):
        inp          = torch.tensor([ids + gen], dtype=torch.long, device=DEVICE)
        n_lp, trace, last = model(inp)

        if not last or last.strip() == "":
            break

        # Encode the predicted token
        new_ids = tokenizer.encode(" " + last.strip() if last.strip() else last)
        if not new_ids:
            break
        nxt = new_ids[0]

        # Stop on EOS
        if nxt == tokenizer.eos_token_id:
            break

        gen.append(nxt)

        # Early exit heuristics for code (stop at double-newline after code)
        decoded_so_far = tokenizer.decode(gen)
        if len(decoded_so_far) > 50 and decoded_so_far.rstrip().endswith("\n\n"):
            break

    elapsed = time.perf_counter() - t0
    text    = tokenizer.decode(gen, skip_special_tokens=True).strip()
    tps     = len(gen) / elapsed if elapsed > 0 else 0
    return text, tps, len(gen)


def score(test: dict, output: str) -> tuple[bool, list]:
    """Score a test result."""
    lo    = output.lower()
    fails = []
    for m in test.get("must", []):
        if m.lower() not in lo:
            fails.append(f"Missing: {m!r}")
    one = test.get("must_one", [])
    if one and not any(m.lower() in lo for m in one):
        fails.append(f"Missing one of: {one}")
    return len(fails) == 0, fails


def bigram_rep(text: str) -> float:
    """Bigram repetition rate."""
    w = text.split()
    if len(w) < 2:
        return 0.0
    bg = [f"{w[i]} {w[i+1]}" for i in range(len(w) - 1)]
    return 1.0 - len(set(bg)) / len(bg)


def main() -> None:
    """Load RLF model and run 54-test benchmark."""
    final_ckpt = RLF_CKPT_DIR / "final"
    if not final_ckpt.exists():
        candidates = sorted(RLF_CKPT_DIR.glob("phase*_step*"))
        if not candidates:
            print("ERROR: No RLF checkpoint. Is training complete?")
            return
        final_ckpt = candidates[-1]
        print(f"Using: {final_ckpt}")

    print("=" * 65)
    print("  RLF EXTENDED BENCHMARK — Mamba-1.4B (54 tests)")
    print(f"  Checkpoint: {final_ckpt}")
    print("=" * 65)

    model = load_rlf_final(final_ckpt)

    results = []
    for t in TESTS:
        output, tps, ntok = rlf_generate(model, t["prompt"])
        passed, fails     = score(t, output)
        rep               = bigram_rep(output)
        status            = "✅" if passed else "❌"
        short             = output[:100].strip().replace("\n", "⏎")

        print(f"\n[{t['id']}] {t['cat']}: {t['prompt'][:55]}...")
        print(f"  {status} | {ntok}tok | {tps:.1f}t/s | rep={rep:.2f}")
        if not passed:
            for f in fails:
                print(f"     → {f}")
        print(f"  {short!r}")

        results.append({"id": t["id"], "cat": t["cat"], "passed": passed,
                        "tps": tps, "ntok": ntok, "rep": rep, "output": output})

    # Summary
    by_cat: dict[str, list] = {}
    for r in results:
        by_cat.setdefault(r["cat"], []).append(r["passed"])

    total_p = sum(r["passed"] for r in results)
    avg_tps = sum(r["tps"] for r in results) / len(results)
    avg_rep = sum(r["rep"] for r in results) / len(results)

    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  {'Category':<26} {'Pass':>5} {'Total':>6} {'Rate':>7}")
    print(f"  {'-'*50}")
    for cat, vals in sorted(by_cat.items()):
        p, t_ = sum(vals), len(vals)
        print(f"  {cat:<26} {p:>5} {t_:>6} {p/t_*100:>6.0f}%")
    print(f"  {'-'*50}")
    print(f"  {'TOTAL (required)':<26} {total_p:>5} {len(results):>6} "
          f"{total_p/len(results)*100:>6.0f}%")
    print(f"\n  Avg throughput : {avg_tps:.1f} tok/s")
    print(f"  Avg bigram rep : {avg_rep:.3f}")
    print("=" * 65)

    # Save JSON report
    report = Path(__file__).parent / "rlf_benchmark_report.json"
    with open(report, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full report: {report}")

    # Comparison vs SFT R5 baseline
    baseline = {"Code Gen": 75, "Code Gen (Complex)": 33, "Algorithm": 75,
                "Data Structures": 100, "Strings": 100, "Math": 88,
                "Word Problem": 17, "Bug Fix": 20, "Completion": 67,
                "Logic": 0, "Unit Tests": 0}
    print("\n  vs R5 SFT Baseline:")
    print(f"  {'Category':<26} {'R5':>6} {'RLF':>6} {'Δ':>6}")
    print(f"  {'-'*44}")
    for cat, vals in sorted(by_cat.items()):
        p, t_ = sum(vals), len(vals)
        rlf_pct = p / t_ * 100
        r5_pct  = baseline.get(cat, 0)
        delta   = rlf_pct - r5_pct
        sign    = "+" if delta >= 0 else ""
        print(f"  {cat:<26} {r5_pct:>5.0f}% {rlf_pct:>5.0f}% {sign}{delta:>4.0f}%")


if __name__ == "__main__":
    main()
