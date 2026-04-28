#!/usr/bin/env python3
"""
extended_benchmark.py
Full comparison across R3, R5 checkpoints.
50+ prompts, 600 max tokens, 12 categories.
Tests checkpoints: R3 (best val=0.110), R5 (hybrid, val=0.128)
"""

import sys, time, re, json
import torch
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent))
from proprioception_gate import GeometricProprioceptionGate
from lora_mamba import PostBackboneLoRA
from transformers import AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "state-spaces/mamba-1.4b"
D_MODEL      = 2048
MAX_NEW      = 300
DEVICE       = "cuda"
TEMP         = 0.6
REP_PENALTY  = 1.8
REP_WINDOW   = 100
NGRAM_BLOCK  = 4
N_SPACERS    = 8

CHECKPOINTS = {
    "R5 (val=0.128 hybrid)": Path("/hdd_data/latent-spacer-checkpoints/final"),
}

# ── Extended Test Suite ───────────────────────────────────────────────────────
TESTS = [
    # ── Code Generation: Basic ────────────────────────────────────────────────
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

    # ── Code Generation: Complex ──────────────────────────────────────────────
    {"id":"cg_09","cat":"Code Gen (Complex)",
     "prompt":"Write a Python class called LRUCache with get and put methods. Use an OrderedDict.",
     "must":["class LRUCache","def get","def put"],"must_one":[]},
    {"id":"cg_10","cat":"Code Gen (Complex)",
     "prompt":"Write a Python function that implements quicksort.",
     "must":["def ","return"],"must_one":[]},
    {"id":"cg_11","cat":"Code Gen (Complex)",
     "prompt":"Write a Python context manager class called Timer that measures execution time using __enter__ and __exit__.",
     "must":["class Timer","__enter__","__exit__"],"must_one":[]},
    {"id":"cg_12","cat":"Code Gen (Complex)",
     "prompt":"Write a Python generator function that yields Fibonacci numbers indefinitely.",
     "must":["def ","yield"],"must_one":[]},
    {"id":"cg_13","cat":"Code Gen (Complex)",
     "prompt":"Write a Python function that performs matrix multiplication of two 2D lists.",
     "must":["def ","return"],"must_one":[]},
    {"id":"cg_14","cat":"Code Gen (Complex)",
     "prompt":"Write a Python class called BinarySearchTree with insert and search methods.",
     "must":["class BinarySearchTree","def insert","def search"],"must_one":[]},

    # ── Algorithms ────────────────────────────────────────────────────────────
    {"id":"al_01","cat":"Algorithm","prompt":"Write a Python function that implements binary search on a sorted list.",
     "must":["def ","mid","return"],"must_one":[]},
    {"id":"al_02","cat":"Algorithm","prompt":"Write a Python function that implements bubble sort.",
     "must":["def ","for ","return"],"must_one":[]},
    {"id":"al_03","cat":"Algorithm","prompt":"Write a Python function that returns the nth Fibonacci number using dynamic programming.",
     "must":["def ","return"],"must_one":[]},
    {"id":"al_04","cat":"Algorithm","prompt":"Write a Python function that implements merge sort.",
     "must":["def ","return"],"must_one":[]},

    # ── Data Structures ───────────────────────────────────────────────────────
    {"id":"ds_01","cat":"Data Structures","prompt":"Write a Python function that reverses a linked list. Assume each node has .val and .next.",
     "must":["def ","next","return"],"must_one":[]},
    {"id":"ds_02","cat":"Data Structures","prompt":"Write a Python function that checks if a binary tree is balanced.",
     "must":["def ","return"],"must_one":[]},
    {"id":"ds_03","cat":"Data Structures","prompt":"Write a Python function that finds the height of a binary tree.",
     "must":["def ","return"],"must_one":[]},
    {"id":"ds_04","cat":"Data Structures","prompt":"Write a Python function using a queue (collections.deque) to do level-order traversal of a binary tree.",
     "must":["def ","deque","return"],"must_one":[]},

    # ── String Manipulation ───────────────────────────────────────────────────
    {"id":"st_01","cat":"Strings","prompt":"Write a Python function that checks if a string is a palindrome.",
     "must":["def ","return"],"must_one":[]},
    {"id":"st_02","cat":"Strings","prompt":"Write a Python function that finds the longest common prefix among a list of strings.",
     "must":["def ","return"],"must_one":[]},
    {"id":"st_03","cat":"Strings","prompt":"Write a Python function that counts the number of vowels in a string.",
     "must":["def ","return"],"must_one":[]},
    {"id":"st_04","cat":"Strings","prompt":"Write a Python function that checks if two strings are anagrams of each other.",
     "must":["def ","return"],"must_one":[]},

    # ── Mathematics: Exact Answers ────────────────────────────────────────────
    {"id":"ma_01","cat":"Math","prompt":"What is 12 multiplied by 7?","must":[],"must_one":["84"]},
    {"id":"ma_02","cat":"Math","prompt":"What is 144 divided by 12?","must":[],"must_one":["12"]},
    {"id":"ma_03","cat":"Math","prompt":"What is the square root of 256?","must":[],"must_one":["16"]},
    {"id":"ma_04","cat":"Math","prompt":"What is 25% of 200?","must":[],"must_one":["50"]},
    {"id":"ma_05","cat":"Math","prompt":"What is 17 multiplied by 13?","must":[],"must_one":["221"]},
    {"id":"ma_06","cat":"Math","prompt":"What is the square root of 625?","must":[],"must_one":["25"]},
    {"id":"ma_07","cat":"Math","prompt":"What is 15% of 80?","must":[],"must_one":["12"]},
    {"id":"ma_08","cat":"Math","prompt":"What is 2 to the power of 8?","must":[],"must_one":["256"]},

    # ── Word Problems ─────────────────────────────────────────────────────────
    {"id":"wp_01","cat":"Word Problem","prompt":"A car travels at 60 mph for 2.5 hours. How many miles does it travel?","must":[],"must_one":["150"]},
    {"id":"wp_02","cat":"Word Problem","prompt":"A class has 30 students. 60% are girls. How many boys are in the class?","must":[],"must_one":["12"]},
    {"id":"wp_03","cat":"Word Problem","prompt":"A store sells apples for $0.50 each. Sarah buys 14 apples. How much does she spend?","must":[],"must_one":["7","$7"]},
    {"id":"wp_04","cat":"Word Problem","prompt":"A rectangle has length 15 cm and width 8 cm. What is its area?","must":[],"must_one":["120"]},
    {"id":"wp_05","cat":"Word Problem","prompt":"If a pizza is cut into 8 equal slices and you eat 3 slices, what percentage of the pizza did you eat?","must":[],"must_one":["37.5","37%","3/8"]},
    {"id":"wp_06","cat":"Word Problem","prompt":"A train travels 240 miles in 4 hours. What is its speed in mph?","must":[],"must_one":["60"]},

    # ── Bug Fixing ────────────────────────────────────────────────────────────
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

    # ── Code Completion ───────────────────────────────────────────────────────
    {"id":"cc_01","cat":"Completion",
     "prompt":"Complete this Python function:\ndef count_evens(lst):\n    \"\"\"Return count of even numbers in lst.\"\"\"\n    # your code here",
     "must":["return"],"must_one":["% 2"]},
    {"id":"cc_02","cat":"Completion",
     "prompt":"Complete this Python function:\ndef find_max(lst):\n    \"\"\"Return the maximum value in lst without using max().\"\"\"\n    # your code here",
     "must":["return"],"must_one":[]},
    {"id":"cc_03","cat":"Completion",
     "prompt":"Complete this Python function:\ndef flatten(nested):\n    \"\"\"Flatten a list of lists into a single list.\"\"\"\n    result = []\n    # your code here\n    return result",
     "must":["result","return result"],"must_one":["append","extend","for"]},

    # ── Logic / Reasoning ─────────────────────────────────────────────────────
    {"id":"lg_01","cat":"Logic","prompt":"What is the next number in the sequence: 2, 6, 18, 54, ?",
     "must":[],"must_one":["162"]},
    {"id":"lg_02","cat":"Logic","prompt":"What is the next number in the sequence: 1, 4, 9, 16, ?",
     "must":[],"must_one":["25"]},
    {"id":"lg_03","cat":"Logic","prompt":"If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
     "must":[],"must_one":["5 minute","5 min","same","5m"]},
    {"id":"lg_04","cat":"Logic","prompt":"What is the next number in the sequence: 1, 1, 2, 3, 5, 8, ?",
     "must":[],"must_one":["13"]},

    # ── Unit Tests ────────────────────────────────────────────────────────────
    {"id":"ut_01","cat":"Unit Tests",
     "prompt":"Write a Python unittest class that tests a function called add(a, b) that returns a+b.",
     "must":["import unittest","def test_","assertEqual"],"must_one":[]},
    {"id":"ut_02","cat":"Unit Tests",
     "prompt":"Write pytest test functions that test is_prime(n) for prime and non-prime inputs.",
     "must":["def test_","assert"],"must_one":[]},
]

# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, adapter, gate, tokenizer, prompt: str) -> tuple[str,float,int]:
    """Run inference and return (text, tok/s, n_tokens)."""
    spacers  = "=" * N_SPACERS
    full_in  = f"[USER]\n{prompt}\n{spacers}\n[ANSWER]\n"
    ids      = tokenizer.encode(full_in, return_tensors="pt").to(DEVICE)
    cur      = ids
    gen      = []
    eos      = tokenizer.eos_token_id
    ngrams: set[tuple] = set()

    t0 = time.perf_counter()
    for _ in range(MAX_NEW):
        h      = model.backbone(cur)
        h      = adapter(h)
        h      = gate(h)
        logits = model.lm_head(h.to(torch.bfloat16))

        lg = logits[0,-1,:].float() / TEMP
        for tid in set(gen[-REP_WINDOW:]):
            lg[tid] = lg[tid]/REP_PENALTY if lg[tid]>0 else lg[tid]*REP_PENALTY
        if len(gen) >= NGRAM_BLOCK-1:
            pfx = tuple(gen[-(NGRAM_BLOCK-1):])
            for c in torch.softmax(lg,dim=-1).topk(50).indices.tolist():
                if pfx+(c,) in ngrams: lg[c]=-1e9
        p,si = torch.sort(torch.softmax(lg,dim=-1),descending=True)
        p[(torch.cumsum(p,0)-p)>0.9]=0.0
        if p.sum()<1e-8: break
        nxt=si[torch.multinomial(p/p.sum(),1)].item()
        if len(gen)>=NGRAM_BLOCK-1:
            ngrams.add(tuple(gen[-(NGRAM_BLOCK-1):])+(nxt,))
        if nxt==eos: break
        gen.append(nxt)
        cur=torch.cat([cur,torch.tensor([[nxt]],device=DEVICE)],dim=1)

    elapsed = time.perf_counter()-t0
    text    = tokenizer.decode(gen,skip_special_tokens=True).strip()
    return text, len(gen)/elapsed if elapsed>0 else 0, len(gen)


def score(test: dict, output: str) -> tuple[bool,list]:
    """Return (passed, failures)."""
    lo    = output.lower()
    fails = []
    for m in test.get("must",[]):
        if m.lower() not in lo: fails.append(f"Missing: {m!r}")
    one = test.get("must_one",[])
    if one and not any(m.lower() in lo for m in one):
        fails.append(f"Missing one of: {one}")
    return len(fails)==0, fails


def bigram_rep(text: str) -> float:
    """Bigram repetition rate."""
    w = text.split()
    if len(w)<2: return 0.0
    bg = [f"{w[i]} {w[i+1]}" for i in range(len(w)-1)]
    return 1.0 - len(set(bg))/len(bg)


def load_ckpt(ckpt_dir: Path, tokenizer):
    """Load model + adapter + gate from checkpoint dir."""
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    model = MambaLMHeadModel.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device=DEVICE
    )
    model.lm_head.load_state_dict(
        torch.load(ckpt_dir/"lm_head.pt", map_location=DEVICE, weights_only=True)
    )
    model.eval()

    adapter = PostBackboneLoRA(d_model=D_MODEL, rank=16, alpha=32.0, n_layers=6)
    adapter.load_state_dict(
        torch.load(ckpt_dir/"adapter.pt", map_location=DEVICE, weights_only=True)
    )
    adapter = adapter.to(DEVICE).to(torch.bfloat16).eval()

    gate = GeometricProprioceptionGate(d_model=D_MODEL, window_size=8)
    gate.load_state_dict(
        torch.load(ckpt_dir/"gate.pt", map_location=DEVICE, weights_only=True)
    )
    gate = gate.to(DEVICE).to(torch.bfloat16).eval()
    return model, adapter, gate


def run_checkpoint(name: str, ckpt_dir: Path, tokenizer) -> dict:
    """Run all tests for one checkpoint. Return results dict."""
    print(f"\n{'='*70}")
    print(f"  CHECKPOINT: {name}")
    print(f"  DIR: {ckpt_dir}")
    print(f"{'='*70}\n")

    model, adapter, gate = load_ckpt(ckpt_dir, tokenizer)

    results = []
    for t in TESTS:
        output, tps, ntok = generate(model, adapter, gate, tokenizer, t["prompt"])
        passed, fails     = score(t, output)
        rep               = bigram_rep(output)

        status = "✅" if passed else "❌"
        short  = output[:120].strip().replace("\n","⏎")
        print(f"[{t['id']}] {t['cat']}: {t['prompt'][:55]}...")
        print(f"  {status} | {ntok}tok | {tps:.1f}t/s | rep={rep:.2f}")
        if not passed:
            for f in fails: print(f"     → {f}")
        print(f"  {short!r}\n")

        results.append({
            "id":t["id"],"cat":t["cat"],"passed":passed,
            "tps":tps,"ntok":ntok,"rep":rep,"output":output,
        })

    del model, adapter, gate
    torch.cuda.empty_cache()
    return results


def print_summary(name: str, results: list) -> dict:
    """Print category breakdown and return stats dict."""
    by_cat: dict[str,list] = {}
    for r in results:
        by_cat.setdefault(r["cat"],[]).append(r["passed"])
    total_p = sum(r["passed"] for r in results)
    total   = len(results)
    avg_tps = sum(r["tps"] for r in results)/total
    avg_rep = sum(r["rep"] for r in results)/total

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  {'Category':<26} {'Pass':>5} {'Total':>6} {'Rate':>7}")
    print(f"  {'-'*48}")
    for cat,vals in sorted(by_cat.items()):
        p,t = sum(vals),len(vals)
        print(f"  {cat:<26} {p:>5} {t:>6} {p/t*100:>6.0f}%")
    print(f"  {'-'*48}")
    print(f"  {'TOTAL':<26} {total_p:>5} {total:>6} {total_p/total*100:>6.0f}%")
    print(f"\n  Avg throughput : {avg_tps:.1f} tok/s")
    print(f"  Avg bigram rep : {avg_rep:.3f}")
    return {"total_pass":total_p,"total":total,"pct":total_p/total*100,"tps":avg_tps,"rep":avg_rep,"by_cat":by_cat}


def main():
    """Load tokenizer once, run all checkpoints, print comparison."""
    print("="*70)
    print("  MAMBA-1.4B — EXTENDED BENCHMARK (50+ tests, 600 max tokens)")
    print(f"  {len(TESTS)} tests × {len(CHECKPOINTS)} checkpoint(s)")
    print("="*70)

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neox-20b", trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    all_stats  = {}
    all_res    = {}
    for name, ckpath in CHECKPOINTS.items():
        res                = run_checkpoint(name, ckpath, tokenizer)
        stats              = print_summary(name, res)
        all_stats[name]    = stats
        all_res[name]      = res

    # ── Final comparison table ─────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  FINAL COMPARISON")
    print(f"{'='*70}")
    cats = sorted({r["cat"] for r in list(all_res.values())[0]})
    header = f"  {'Category':<26}" + "".join(f" {'R'+n[-2:]:>9}" if 'R' in n else f" {n[:8]:>9}" for n in all_stats)
    print(header)
    print(f"  {'-'*65}")
    for cat in cats:
        row = f"  {cat:<26}"
        for stats in all_stats.values():
            vals = stats["by_cat"].get(cat,[])
            p,t  = sum(vals),len(vals)
            row += f" {p}/{t} {p/t*100:>3.0f}%" if t else "    —  "
        print(row)
    print(f"  {'-'*65}")
    row = f"  {'TOTAL':<26}"
    for stats in all_stats.values():
        row += f" {stats['total_pass']}/{stats['total']} {stats['pct']:>3.0f}%"
    print(row)
    row = f"  {'Throughput':<26}"
    for stats in all_stats.values():
        row += f" {stats['tps']:>9.1f}"
    print(row + " tok/s")
    row = f"  {'Bigram rep':<26}"
    for stats in all_stats.values():
        row += f" {stats['rep']:>9.3f}"
    print(row)
    print("="*70)

    # Save full report
    report_path = Path(__file__).parent / "extended_benchmark_report.json"
    with open(report_path,"w") as f:
        out = {}
        for name,res in all_res.items():
            out[name]=[{"id":r["id"],"cat":r["cat"],"passed":r["passed"],
                        "tps":r["tps"],"ntok":r["ntok"],"rep":r["rep"],
                        "output":r["output"][:300]} for r in res]
        json.dump(out, f, indent=2)
    print(f"\n  Full report saved: {report_path}")


if __name__ == "__main__":
    main()
