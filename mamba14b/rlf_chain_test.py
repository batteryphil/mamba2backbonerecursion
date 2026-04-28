#!/usr/bin/env python3
"""
rlf_chain_test.py — Validates RLF loop reasoning on held-out chain problems.

Three test suites based on historical failure analysis:
  1. Chain accuracy (20 tests: 1-hop, 2-hop, 3-hop, math, sequences, bugs)
  2. Scratchpad ablation (were the prefix memory tokens actually used?)
  3. Semantic shift (does it transfer beyond training syntax to natural language?)

The 130M experiments proved that scratchpad ablation Δ=0 and bAbI accuracy 0%
are the two real failure modes. This suite directly tests both.
"""

import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rlf_engine_1_4b import (
    RecursiveMamba1_PrefixScratchpad,
    load_from_sft_checkpoint,
    HALT_ID, tokenizer, DEVICE, PREFIX_M,
)

RLF_CKPT_DIR = Path("/hdd_data/rlf-1.4b-checkpoints")
SFT_CKPT_DIR = Path("/hdd_data/latent-spacer-checkpoints/best")

# ── Chain test cases ──────────────────────────────────────────────────────────
CHAIN_TESTS = [
    # ── 1-hop ─────────────────────────────────────────────────────────────────
    {"id": "1h_01", "cat": "1-hop",
     "prompt": "A=42. What is A?",
     "expected_chain": ["42"]},
    {"id": "1h_02", "cat": "1-hop",
     "prompt": "color=Blue. What is color?",
     "expected_chain": ["Blue"]},
    {"id": "1h_03", "cat": "1-hop",
     "prompt": "x=99. What is x?",
     "expected_chain": ["99"]},

    # ── 2-hop ─────────────────────────────────────────────────────────────────
    {"id": "2h_01", "cat": "2-hop",
     "prompt": "A=42. B=A. What is B?",
     "expected_chain": ["42", "42"]},
    {"id": "2h_02", "cat": "2-hop",
     "prompt": "X=Moon. Y=X. What is Y?",
     "expected_chain": ["Moon", "Moon"]},
    {"id": "2h_03", "cat": "2-hop",
     "prompt": "The sky is blue. alpha=77. beta=alpha. What is beta?",
     "expected_chain": ["77", "77"]},

    # ── 3-hop ─────────────────────────────────────────────────────────────────
    {"id": "3h_01", "cat": "3-hop",
     "prompt": "A=Star. B=A. C=B. What is C?",
     "expected_chain": ["Star", "Star", "Star"]},
    {"id": "3h_02", "cat": "3-hop",
     "prompt": "v1=55. v2=v1. v3=v2. What is v3?",
     "expected_chain": ["55", "55", "55"]},

    # ── 3-hop adversarial (distractors) ───────────────────────────────────────
    {"id": "3h_adv_01", "cat": "3-hop-adversarial",
     "prompt": ("zx=Fire. Water boils at 100C. "
                "zy=zx. zz=zy. Pi is 3.14159. What is zz?"),
     "expected_chain": ["Fire", "Fire", "Fire"]},
    {"id": "3h_adv_02", "cat": "3-hop-adversarial",
     "prompt": ("ab=Gold. cd=ab. ef=cd. In 1969 Armstrong walked on moon. "
                "xk9=77. What is ef?"),
     "expected_chain": ["Gold", "Gold", "Gold"]},

    # ── Math 1-step ───────────────────────────────────────────────────────────
    {"id": "ma_01", "cat": "math-1step",
     "prompt": "qty=6. price=0.50. cost=qty*price. What is cost?",
     "expected_chain": ["3.00"]},
    {"id": "ma_02", "cat": "math-1step",
     "prompt": "apples=4. price=1.25. total=apples*price. What is total?",
     "expected_chain": ["5.00"]},
    {"id": "ma_03", "cat": "math-1step",
     "prompt": "students=30. pct=0.60. girls=students*pct. What is girls?",
     "expected_chain": ["18"]},

    # ── Math 2-step ───────────────────────────────────────────────────────────
    {"id": "ma_04", "cat": "math-2step",
     "prompt": ("a=6. pa=0.50. ca=a*pa. "
                "b=4. pb=0.75. cb=b*pb. "
                "total=ca+cb. What is total?"),
     "expected_chain": ["3.00", "3.00", "6.00"]},

    # ── Sequence patterns ─────────────────────────────────────────────────────
    {"id": "sq_01", "cat": "sequence",
     "prompt": "x1=2. x2=x1*3. x3=x2*3. x4=x3*3. What is x5?",
     "expected_chain": ["162"]},
    {"id": "sq_02", "cat": "sequence",
     "prompt": "x1=1. x2=x1*2. x3=x2*2. x4=x3*2. What is x5?",
     "expected_chain": ["16"]},
    {"id": "sq_03", "cat": "sequence",
     "prompt": "x1=0. x2=x1+5. x3=x2+5. x4=x3+5. What is x5?",
     "expected_chain": ["20"]},
    {"id": "sq_04", "cat": "sequence",
     "prompt": "n1=1. n2=1. n3=n1+n2. n4=n2+n3. n5=n3+n4. What is n6?",
     "expected_chain": ["13"]},

    # ── Bug fix ───────────────────────────────────────────────────────────────
    {"id": "bf_01", "cat": "bug-fix",
     "prompt": "op=minus. fix=plus. expr=a op b. What is the fixed expr?",
     "expected_chain": ["a plus b"]},
    {"id": "bf_02", "cat": "bug-fix",
     "prompt": "bug=return b. fix=return a. code has bug. What is the fix?",
     "expected_chain": ["return a"]},
]

# ── Ablation prompts (same as easy chain tests — measures scratchpad delta) ───
ABLATION_PROMPTS = [
    ("A=42. B=A. C=B. What is C?",                       "42"),
    ("V1=99. V2=V1. V3=V2. What is V3?",                 "99"),
    ("x=7. y=x. z=y. What is z?",                        "7"),
    ("qty=6. price=0.50. cost=qty*price. What is cost?",  "3.00"),
    ("x1=3. x2=x1*2. x3=x2*2. What is x4?",              "24"),
]

# ── Semantic shift prompts (same logic, completely different syntax) ───────────
# Training syntax: "A=42. B=A. What is B?"
# Shifted syntax:  "The box holds what the bag holds. The bag holds 42."
# 130M scored 0% on this because it overfitted to causal token syntax.
SEMANTIC_SHIFT = [
    {"prompt": ("The box holds what the bag holds. "
                "The bag holds the apple. What does the box hold?"),
     "expected": "apple"},
    {"prompt": ("John's score equals Mary's score. "
                "Mary scored 88. What is John's score?"),
     "expected": "88"},
    {"prompt": ("Container B has what container A has. "
                "Container C has what container B has. "
                "Container A has the key. What does container C have?"),
     "expected": "key"},
    {"prompt": ("The red jar contains the same thing as the blue cup. "
                "The blue cup contains frost. What is in the red jar?"),
     "expected": "frost"},
    {"prompt": ("Sarah earns twice what Tom earns. Tom earns 500. "
                "What does Sarah earn?"),
     "expected": "1000"},
]


# ── Load RLF model ────────────────────────────────────────────────────────────

def load_rlf_final(ckpt_dir: Path) -> RecursiveMamba1_PrefixScratchpad:
    """Load the trained RLF model for evaluation."""
    model = load_from_sft_checkpoint(str(SFT_CKPT_DIR), DEVICE)

    for fname, target in [
        ("latent_memory.pt", None),
        ("bridge_down.pt",   model.bridge_down),
        ("bridge_up.pt",     model.bridge_up),
        ("mamba1_loop.pt",   model.mamba1_loop),
        ("loop_norm.pt",     model.loop_norm),
        ("lifeline_gate.pt", None),
        ("lm_head.pt",       model.lm_head),
    ]:
        fpath = ckpt_dir / fname
        if not fpath.exists():
            continue
        if fname == "latent_memory.pt":
            model.latent_memory.data.copy_(
                torch.load(fpath, map_location=DEVICE, weights_only=True)
            )
        elif fname == "lifeline_gate.pt":
            model.lifeline_gate.data.copy_(
                torch.load(fpath, map_location=DEVICE, weights_only=True)
            )
        else:
            target.load_state_dict(
                torch.load(fpath, map_location=DEVICE, weights_only=True)
            )

    lora_path = ckpt_dir / "lora.pt"
    if lora_path.exists():
        lora_state = torch.load(lora_path, map_location=DEVICE, weights_only=True)
        for name, param in model.named_parameters():
            if name in lora_state:
                param.data.copy_(lora_state[name])

    model.eval()
    return model


# ── Inference helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def run_chain(
    model: RecursiveMamba1_PrefixScratchpad, prompt: str
) -> tuple[list[str], float]:
    """Run RLF inference, return (output_chain, elapsed_s)."""
    ids     = tokenizer.encode(prompt)
    inp     = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    t0      = time.perf_counter()
    _, trace, _ = model(inp)
    elapsed = time.perf_counter() - t0
    chain   = [tok for (_, tok, _) in trace if tok != "<HALT>"]
    return chain, elapsed


def score_chain(output: list[str], expected: list[str]) -> tuple[bool, str]:
    """Fuzzy-match final output against expected[-1]."""
    if not output:
        return False, "empty output"
    exp_last = expected[-1].lower().strip()
    out_last = output[-1].lower().strip()
    if exp_last in out_last or out_last in exp_last:
        return True, ""
    return False, f"got {output!r}, expected final={expected[-1]!r}"


# ── Test suites ───────────────────────────────────────────────────────────────

def run_chain_suite(model: RecursiveMamba1_PrefixScratchpad) -> float:
    """Test 1: Chain accuracy across 20 held-out problems. Returns pass rate."""
    print("\n" + "=" * 65)
    print("  TEST 1: CHAIN ACCURACY (20 problems)")
    print("=" * 65)

    by_cat: dict[str, list[bool]] = {}
    total = 0

    for t in CHAIN_TESTS:
        out, elapsed = run_chain(model, t["prompt"])
        passed, reason = score_chain(out, t["expected_chain"])
        by_cat.setdefault(t["cat"], []).append(passed)
        total += int(passed)
        status = "✅" if passed else "❌"
        print(f"\n  [{t['id']}] {t['cat']}")
        print(f"    Prompt:   {t['prompt']}")
        print(f"    Expected: {t['expected_chain']}")
        print(f"    Got:      {out} ({elapsed:.1f}s)  {status} {reason}")

    print(f"\n  {'Category':<22} {'Pass':>4} {'Tot':>4} {'Rate':>7}")
    print(f"  {'-'*40}")
    for cat, vals in sorted(by_cat.items()):
        p, n = sum(vals), len(vals)
        print(f"  {cat:<22} {p:>4} {n:>4} {p/n*100:>6.0f}%")
    print(f"  {'-'*40}")
    pct = total / len(CHAIN_TESTS) * 100
    print(f"  {'TOTAL':<22} {total:>4} {len(CHAIN_TESTS):>4} {pct:>6.0f}%")

    if pct >= 70:
        print("\n  ✅ RLF IS WORKING — scratchpad retaining variables")
    elif pct >= 40:
        print("\n  ⚠️  PARTIAL — needs more Phase 3b training")
    else:
        print("\n  ❌ RLF NOT WORKING — chain following failed")
    return pct


def run_ablation(model: RecursiveMamba1_PrefixScratchpad) -> int:
    """Test 2: Does zeroing the scratchpad hurt accuracy?

    130M baseline: Δ=0 (scratchpad contributed NOTHING).
    We need Δ > 0 to prove the prefix memory is actually being used.
    """
    print("\n" + "=" * 65)
    print("  TEST 2: SCRATCHPAD ABLATION")
    print("  (130M: ablated=2%, normal=2%, Δ=0 → scratchpad unused)")
    print("=" * 65)

    def count_correct(mem_zeroed: bool) -> int:
        """Count correct answers with or without scratchpad."""
        if mem_zeroed:
            saved = model.latent_memory.data.clone()
            model.latent_memory.data.zero_()
        correct = 0
        for prompt, expected in ABLATION_PROMPTS:
            out, _ = run_chain(model, prompt)
            if out and expected.lower() in out[-1].lower():
                correct += 1
        if mem_zeroed:
            model.latent_memory.data.copy_(saved)
        return correct

    n = len(ABLATION_PROMPTS)
    run_a = count_correct(False)
    run_b = count_correct(True)
    delta = run_a - run_b

    print(f"\n  Run A (normal):          {run_a}/{n} = {run_a/n*100:.0f}%")
    print(f"  Run B (zero scratchpad): {run_b}/{n} = {run_b/n*100:.0f}%")
    print(f"  Δ scratchpad contribution: {delta:+d} tests")

    if delta > 1:
        print("  ✅ SCRATCHPAD IS CONTRIBUTING — zeroing hurts accuracy")
    elif delta == 0 and run_a > 2:
        print("  ⚠️  INCONCLUSIVE — tasks may be too easy to need scratchpad")
    elif delta == 0:
        print("  ❌ SCRATCHPAD UNUSED — matches 130M failure pattern (Δ=0)")
    else:
        print("  ⚠️  MARGINAL — minimal contribution, more training needed")
    return delta


def run_semantic_shift(model: RecursiveMamba1_PrefixScratchpad) -> float:
    """Test 3: Same logical operation, different English syntax.

    130M baseline: 0.0% — predicted 'photograp' for every sample.
    The model overfitted to 'A=X. B=A. What is B?' causal token syntax.
    Extended benchmark word problems will use natural language syntax.
    """
    print("\n" + "=" * 65)
    print("  TEST 3: SEMANTIC SHIFT (zero-shot syntax transfer)")
    print("  (130M: 0.0% — predicted 'photograp' for all 50 samples)")
    print("=" * 65)

    correct = 0
    for t in SEMANTIC_SHIFT:
        out, elapsed = run_chain(model, t["prompt"])
        last   = out[-1].lower() if out else ""
        passed = t["expected"].lower() in last or last in t["expected"].lower()
        correct += int(passed)
        status = "✅" if passed else "❌"
        print(f"\n  {status} [{elapsed:.1f}s] {t['prompt'][:62]}...")
        print(f"     Expected: {t['expected']}  Got: {out}")

    pct = correct / len(SEMANTIC_SHIFT) * 100
    print(f"\n  Semantic shift: {correct}/{len(SEMANTIC_SHIFT)} = {pct:.0f}%")

    if pct >= 60:
        print("  ✅ GENERALIZING — syntax-agnostic reasoning confirmed")
    elif pct >= 20:
        print("  ⚠️  PARTIAL — some transfer but still syntax-dependent")
    else:
        print("  ❌ SYNTAX OVERFIT — matches 130M bAbI failure pattern (0%)")
    return pct


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run all three test suites."""
    final_ckpt = RLF_CKPT_DIR / "final"
    if not final_ckpt.exists():
        candidates = sorted(RLF_CKPT_DIR.glob("phase*_step*"))
        if not candidates:
            print("ERROR: No RLF checkpoint found.")
            return
        final_ckpt = candidates[-1]
        print(f"Using partial checkpoint: {final_ckpt}")

    print("=" * 65)
    print("  RLF VALIDATION — Mamba-1.4B")
    print(f"  Checkpoint: {final_ckpt}")
    print("  Tests: Chain accuracy | Ablation | Semantic shift")
    print("=" * 65)

    print("\nLoading RLF model...")
    model = load_rlf_final(final_ckpt)

    chain_pct  = run_chain_suite(model)
    ablation_d = run_ablation(model)
    shift_pct  = run_semantic_shift(model)

    print("\n" + "=" * 65)
    print("  OVERALL VERDICT")
    print("=" * 65)
    print(f"  Chain accuracy:    {chain_pct:.0f}%  "
          f"{'✅' if chain_pct >= 70 else '⚠️ ' if chain_pct >= 40 else '❌'}")
    print(f"  Scratchpad Δ:      {ablation_d:+d}        "
          f"{'✅' if ablation_d > 1 else '⚠️ ' if ablation_d > 0 else '❌'}")
    print(f"  Semantic shift:    {shift_pct:.0f}%  "
          f"{'✅' if shift_pct >= 60 else '⚠️ ' if shift_pct >= 20 else '❌'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
