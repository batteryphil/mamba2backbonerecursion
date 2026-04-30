#!/usr/bin/env python3
"""
phase_gate.py — Quick per-phase sanity check for V4 RLF training.

Run after each phase completes to verify that phase did its job
before investing GPU time in the next one.

Usage:
    python phase_gate.py --phase 3a   # after Phase 3a completes
    python phase_gate.py --phase 3b   # after Phase 3b completes
    python phase_gate.py --phase 3c   # after Phase 3c (full eval)
"""

import re
import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rlf_engine_1_4b import (
    RecursiveMamba1_PrefixScratchpad,
    load_from_sft_checkpoint,
    HALT_ID, tokenizer, DEVICE, PREFIX_M,
)

CKPT_DIR = Path("/hdd_data/rlf-1.4b-checkpoints")
SFT_CKPT = Path("/hdd_data/latent-spacer-checkpoints/best")

# Quick 1-hop probes for Phase 3a gate
PHASE_3A_PROBES = [
    ("A=42. What is A?",              "42"),
    ("color=Blue. What is color?",    "Blue"),
    ("x=99. What is x?",              "99"),
    ("num=7. What is num?",           "7"),
    ("val=512. What is val?",         "512"),
]

# Multi-hop probes for Phase 3b gate — held-out, never seen during training
PHASE_3B_PROBES = [
    ("A=42. B=A. What is B?",                                  "42"),
    ("X=Moon. Y=X. What is Y?",                                "Moon"),
    ("v1=55. v2=v1. v3=v2. What is v3?",                       "55"),
    ("qty=6. price=0.50. cost=qty*price. What is cost?",        "3.00"),
    ("A=Star. B=A. C=B. What is C?",                           "Star"),
]


def load_model(ckpt_dir: Path) -> RecursiveMamba1_PrefixScratchpad:
    """Load model from checkpoint directory."""
    model = load_from_sft_checkpoint(str(SFT_CKPT), DEVICE)

    for fname, target in [
        ("concept_perceptron.pt", model.concept_perceptron),
        ("bridge_down.pt",        model.bridge_down),
        ("bridge_up.pt",          model.bridge_up),
        ("mamba1_loop.pt",        model.mamba1_loop),
        ("loop_norm.pt",          model.loop_norm),
        ("lm_head.pt",            model.lm_head),
    ]:
        fpath = ckpt_dir / fname
        if fpath.exists():
            target.load_state_dict(
                torch.load(fpath, map_location=DEVICE, weights_only=True)
            )
    model.eval()
    return model


def check_inference_norm(model: RecursiveMamba1_PrefixScratchpad) -> float:
    """Measure the perceptron output norm at inference time.

    V4: F.normalize is baked into ConceptPerceptron so this is now a real,
    bounded measurement — no longer blows up at inference. Expected: close
    to model.concept_perceptron.scale (~1.0).
    """
    probe = "A=1. What is A?"
    ids   = torch.tensor([tokenizer.encode(probe)], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        x, _  = model._encode(ids)
        mem   = model.concept_perceptron(x)
        norm  = mem.norm(p=2, dim=-1).mean().item()
    return norm


def run_probe(model: RecursiveMamba1_PrefixScratchpad, prompt: str) -> str:
    """Run inference and return the model's final answer string."""
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        n_loops, trace, last_ans = model(ids)
    return last_ans.strip() if last_ans else ""


def gate_3a(ckpt_dir: Path) -> bool:
    """Phase 3a gate: verify ConceptPerceptron is alive at inference time.

    Pass criteria:
      1. Inference mem_norm is between 0.5 and 2.0
         (V4 F.normalize guarantees bounded; out of range = scale exploded)
      2. At least 1/5 simple 1-hop probes answered correctly
    """
    print("\n" + "="*60)
    print("  PHASE 3a GATE — ConceptPerceptron Health Check (V4)")
    print("="*60)

    model   = load_model(ckpt_dir)
    norm    = check_inference_norm(model)
    norm_ok = 0.5 <= norm <= 2.0
    scale   = float(model.concept_perceptron.scale.detach().cpu())
    print(f"\n  inference mem_norm = {norm:.3f}  "
          f"{'OK' if norm_ok else 'FAIL  (expected 0.5-2.0)'}"
          f"  |  scale param = {scale:.3f}")

    correct = 0
    for prompt, expected in PHASE_3A_PROBES:
        ans = run_probe(model, prompt)
        ok  = bool(ans) and expected.lower() in ans.lower()
        correct += int(ok)
        mark = "OK" if ok else "--"
        print(f"  [{mark}]  {prompt!r:50s}  -> {ans!r}")

    acc    = correct / len(PHASE_3A_PROBES)
    acc_ok = acc >= 0.20

    print(f"\n  1-hop accuracy: {correct}/{len(PHASE_3A_PROBES)} = {acc*100:.0f}%"
          f"  {'OK' if acc_ok else 'FAIL  (need >=20%)'}")

    passed = norm_ok and acc_ok
    print(f"\n  GATE RESULT: {'PASS -- proceed to Phase 3b' if passed else 'FAIL -- do not proceed'}")
    return passed


def gate_3b(ckpt_dir: Path) -> bool:
    """Phase 3b gate: verify multi-hop reasoning has activated.

    Pass criteria:
      1. Inference mem_norm is between 0.5 and 2.0
      2. At least 2/5 multi-hop probes answered correctly (40%)
    """
    print("\n" + "="*60)
    print("  PHASE 3b GATE -- Multi-Hop Reasoning Check")
    print("="*60)

    model   = load_model(ckpt_dir)
    norm    = check_inference_norm(model)
    norm_ok = 0.5 <= norm <= 2.0
    scale   = float(model.concept_perceptron.scale.detach().cpu())
    print(f"\n  inference mem_norm = {norm:.3f}  "
          f"{'OK' if norm_ok else 'FAIL'}"
          f"  |  scale = {scale:.3f}")

    correct = 0
    for prompt, expected in PHASE_3B_PROBES:
        ans = run_probe(model, prompt)
        ok  = bool(ans) and expected.lower() in ans.lower()
        correct += int(ok)
        mark = "OK" if ok else "--"
        print(f"  [{mark}]  {prompt!r:55s}  -> {ans!r}")

    acc    = correct / len(PHASE_3B_PROBES)
    acc_ok = acc >= 0.40

    print(f"\n  Multi-hop accuracy: {correct}/{len(PHASE_3B_PROBES)} = {acc*100:.0f}%"
          f"  {'OK' if acc_ok else 'FAIL  (need >=40%)'}")

    passed = norm_ok and acc_ok
    print(f"\n  GATE RESULT: {'PASS -- proceed to Phase 3c' if passed else 'FAIL -- extend 3b or abort'}")
    return passed


def gate_3c(ckpt_dir: Path) -> bool:
    """Phase 3c gate: run the full eval suite."""
    import os
    import subprocess
    print("\n" + "="*60)
    print("  PHASE 3c GATE -- Full Eval Suite")
    print("="*60)
    nccl = ("/home/phil/.gemini/antigravity/scratch/quill/.venv/lib/"
            "python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2")
    site = ("/home/phil/.local/share/mise/installs/python/3.14.3/lib/"
            "python3.14/site-packages")
    env  = os.environ.copy()
    env["LD_PRELOAD"] = f"{nccl}:{site}/nvidia/cu13/lib/libnvJitLink.so.13"
    env["TOKENIZERS_PARALLELISM"] = "false"
    script = Path(__file__).parent / "rlf_chain_test.py"
    result = subprocess.run(
        [sys.executable, "-B", str(script)],
        cwd=str(script.parent), env=env
    )
    return result.returncode == 0


def main() -> None:
    """Run the appropriate phase gate."""
    parser = argparse.ArgumentParser(description="Phase gate validator")
    parser.add_argument("--phase", required=True, choices=["3a", "3b", "3c"])
    args = parser.parse_args()

    candidates = sorted(CKPT_DIR.glob(f"phase{args.phase}_step*"))
    if candidates:
        ckpt = candidates[-1]
    elif (CKPT_DIR / "final").exists():
        ckpt = CKPT_DIR / "final"
    else:
        # Fall back to phase0 checkpoint for gate_3a after Phase 0 + 3a run
        phase0_ckpts = sorted(CKPT_DIR.glob("phase0_step*"))
        if phase0_ckpts and args.phase == "3a":
            ckpt = phase0_ckpts[-1]
        else:
            print(f"ERROR: No checkpoint found for phase {args.phase} in {CKPT_DIR}")
            sys.exit(1)

    print(f"  Checkpoint: {ckpt}")

    if args.phase == "3a":
        passed = gate_3a(ckpt)
    elif args.phase == "3b":
        passed = gate_3b(ckpt)
    else:
        passed = gate_3c(ckpt)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
