#!/usr/bin/env python3
"""
rlf_chain_test.py — Validates RLF loop reasoning on held-out chain problems.

Tests the core assumption: does the prefix scratchpad + loop engine actually
enable multi-hop reasoning that the plain SFT model fails at?

Covers:
  - Variable pointer chains (1-hop, 2-hop, 3-hop, adversarial)
  - Math chains (1-step, 2-step arithmetic)
  - Sequence patterns (×2, ×3, +5)
  - Bug fix chains
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


# ── Load RLF components ───────────────────────────────────────────────────────

def load_rlf_final(ckpt_dir: Path) -> RecursiveMamba1_PrefixScratchpad:
    """Load the trained RLF model for evaluation."""
    model = load_from_sft_checkpoint(str(SFT_CKPT_DIR), DEVICE)

    # Load trained RLF components
    component_map = [
        ("latent_memory.pt", None),
        ("bridge_down.pt",   model.bridge_down),
        ("bridge_up.pt",     model.bridge_up),
        ("mamba1_loop.pt",   model.mamba1_loop),
        ("loop_norm.pt",     model.loop_norm),
        ("lifeline_gate.pt", None),
        ("lm_head.pt",       model.lm_head),
    ]
    for fname, target in component_map:
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


# ── Chain evaluation ──────────────────────────────────────────────────────────

@torch.no_grad()
def run_chain(
    model: RecursiveMamba1_PrefixScratchpad,
    prompt: str,
) -> tuple[list[str], float]:
    """Run RLF inference on a prompt, return (output_chain, elapsed_s)."""
    ids  = tokenizer.encode(prompt)
    inp  = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    t0   = time.perf_counter()
    n_loops, trace, last = model(inp)
    elapsed = time.perf_counter() - t0

    # Extract non-HALT tokens from trace
    chain = [tok for (_, tok, _) in trace if tok != "<HALT>"]
    return chain, elapsed


def score_chain(output: list[str], expected: list[str]) -> tuple[bool, str]:
    """Check if the last non-HALT output matches the expected final answer."""
    if not output:
        return False, "empty output"
    # Fuzzy match: check if expected[-1] appears in last output token
    exp_last = expected[-1].lower().strip()
    out_last = output[-1].lower().strip() if output else ""
    if exp_last in out_last or out_last in exp_last:
        return True, ""
    return False, f"got {output!r}, expected final={expected[-1]!r}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run chain test suite and print results."""
    # Find the final RLF checkpoint
    final_ckpt = RLF_CKPT_DIR / "final"
    if not final_ckpt.exists():
        # Fall back to latest partial checkpoint
        candidates = sorted(RLF_CKPT_DIR.glob("phase*_step*"))
        if not candidates:
            print("ERROR: No RLF checkpoint found. Is training complete?")
            return
        final_ckpt = candidates[-1]
        print(f"Using partial checkpoint: {final_ckpt}")

    print("=" * 65)
    print("  RLF CHAIN TEST — Mamba-1.4B Latent Reasoning Validation")
    print(f"  Checkpoint: {final_ckpt}")
    print("=" * 65)

    print("\nLoading RLF model...")
    model = load_rlf_final(final_ckpt)

    results_by_cat: dict[str, list[bool]] = {}
    total_pass = 0

    for t in CHAIN_TESTS:
        chain_out, elapsed = run_chain(model, t["prompt"])
        passed, reason     = score_chain(chain_out, t["expected_chain"])
        status             = "✅" if passed else "❌"
        cat                = t["cat"]
        results_by_cat.setdefault(cat, []).append(passed)
        total_pass        += int(passed)

        print(f"\n[{t['id']}] {cat}")
        print(f"  Prompt:   {t['prompt']}")
        print(f"  Expected: {t['expected_chain']}")
        print(f"  Got:      {chain_out} ({elapsed:.1f}s)")
        print(f"  {status} {reason}")

    # Summary
    print("\n" + "=" * 65)
    print("  RESULTS BY CATEGORY")
    print("=" * 65)
    print(f"  {'Category':<22} {'Pass':>5} {'Total':>6} {'Rate':>7}")
    print(f"  {'-'*44}")
    for cat, vals in sorted(results_by_cat.items()):
        p, t_ = sum(vals), len(vals)
        print(f"  {cat:<22} {p:>5} {t_:>6} {p/t_*100:>6.0f}%")
    print(f"  {'-'*44}")
    print(f"  {'TOTAL':<22} {total_pass:>5} {len(CHAIN_TESTS):>6} "
          f"{total_pass/len(CHAIN_TESTS)*100:>6.0f}%")

    base_score = total_pass / len(CHAIN_TESTS) * 100
    if base_score >= 70:
        verdict = "✅ RLF IS WORKING — latent scratchpad is retaining variables"
    elif base_score >= 40:
        verdict = "⚠️  PARTIAL — scratchpad learning but needs more training"
    else:
        verdict = "❌ RLF NOT YET WORKING — may need Phase 3b longer"
    print(f"\n  {verdict}")
    print("=" * 65)


if __name__ == "__main__":
    main()
