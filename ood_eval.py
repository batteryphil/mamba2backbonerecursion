"""
ood_eval.py — OOD Generalization Test for Mamba-130M RLF
=========================================================
Tests the trained checkpoint on held-out chains it has NEVER seen:
  - In-distribution (ID):  hops 2-8, same VALS vocab
  - OOD hop length:        hops 9-15 (never trained on)
  - OOD vocabulary:        new answer words not in VALS
  - Adversarial OOD:       hops 9-15 + distractors + novel vocab
"""

import torch
import random
import os
import sys
import string

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mamba_ssm import MambaLMHeadModel
from mamba1_engine import (
    RecursiveMamba1_PrefixScratchpad, MODEL_ID, tokenizer,
    HALT_ID as ENGINE_HALT_ID,
)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
HALT_ID  = ENGINE_HALT_ID
# Phase 2 checkpoint is the correct eval target:
# Phase 3 uses sparse reward with Lifeline OFF during dark loops, which breaks
# normal inference (Lifeline always fires in eval mode). Phase 2 dense = apples-to-apples.
CKPT     = "saved_weights/mamba130m_v3_phase2_best.pt"

# We now test Numeric Entropy generalization.
# "Seen" vocab in this context means numeric strings, which the tokenizer
# splits into BPE tokens. True OOD vocab would be testing words/colors,
# to see if the engine can route *any* token payload, even non-numeric ones.

TRAIN_VALS = [str(i) for i in range(1, 999_999 + 1, 137)] # Sparse subset of numbers

OOD_VALS = [
    "Quartz", "Zinc", "Hazel", "Monk", "Flux", "Vex", "Pyre",
    "Rune", "Dusk", "Gust", "Knot", "Wren", "Crest", "Plume",
]

_DISTRACTOR_KEYS = [
    "sys", "env", "tmp", "buf", "idx", "ptr", "cnt", "sum",
    "val", "key", "ref", "aux", "err", "bit", "reg", "mem",
]


def make_chain(
    hops: int,
    val: str,
    adversarial: bool = False,
    rng: random.Random = None,
) -> tuple[str, str]:
    """Build a chain prompt and return (prompt, answer)."""
    if rng is None:
        rng = random.Random()
    chain = [f"V1={val}."]
    for i in range(2, hops + 1):
        chain.append(f"V{i}=V{i-1}.")
    chain.append(f"What is V{hops}? Answer:")

    if adversarial:
        distractors = []
        for _ in range(rng.randint(2, 5)):
            dk = rng.choice(_DISTRACTOR_KEYS)
            dv = rng.choice(TRAIN_VALS)
            distractors.append(f"{dk}={dv}.")
        rng.shuffle(distractors)
        n_before = rng.randint(1, max(1, len(distractors) - 1))
        prompt = " ".join(
            distractors[:n_before] + chain + distractors[n_before:]
        )
    else:
        prompt = " ".join(chain)

    return prompt, val


def eval_suite(
    model: RecursiveMamba1_PrefixScratchpad,
    name: str,
    hops_range: tuple[int, int],
    vocab: list[str],
    n: int = 100,
    adversarial: bool = False,
    seed: int = 77777,
    n_dark_inference: int = 0,
) -> float:
    """Run n samples and return accuracy.

    Args:
        n_dark_inference: Number of Lifeline-off dark loops before the reward loops.
            Set to 0 for Phase 2 (dense) eval. Set to 3 for Phase 3 (sparse) eval.
    """
    # Must call .train() to trigger the 4-tuple return path in forward().
    # The training path checks `self.training and chain_targets is not None`.
    model.train()
    rng = random.Random(seed)
    correct = 0

    for _ in range(n):
        hops = rng.randint(*hops_range)
        val  = rng.choice(vocab)
        prompt, answer = make_chain(hops, val, adversarial=adversarial, rng=rng)

        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        ans_start = len(input_ids) - 1

        val_toks  = tokenizer.encode(" " + answer, add_special_tokens=False)
        ans_tok   = val_toks[0]
        target    = torch.tensor([[ans_tok, HALT_ID]], dtype=torch.long, device=DEVICE)
        inp       = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
        ans_start_t = torch.tensor([ans_start], dtype=torch.long, device=DEVICE)

        with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            loops, trace, predicted_ans = model(inp, n_dark_inference=n_dark_inference)
            
        # The engine trace is List[Tuple[str, str, float]] = [('L1', 'tok', prob), ...]
        # For sparse reward, the actual answer is at the loop just before <HALT>
        # (or the last loop if it didn't halt).
        # Find the loop just before HALT — that is always the answer prediction.
        # The model may run extra sub-token loops after the first prediction,
        # so we must find the HALT entry and take the entry immediately preceding it.
        halt_idx = next(
            (i for i, entry in enumerate(trace) if entry[1] == '<HALT>'), None
        )
        if halt_idx is not None and halt_idx > 0:
            pred_val = trace[halt_idx - 1][1]
        elif trace:
            pred_val = trace[-1][1]  # no HALT found, take last entry
        else:
            pred_val = ""

        target_str = tokenizer.decode([ans_tok]).strip()
        is_correct = (pred_val.strip() == target_str)
        correct += int(is_correct)

    acc = correct / n
    status = "✅" if acc >= 0.80 else ("⚠️ " if acc >= 0.50 else "❌")
    print(f"  {status} {name:<45} {correct:3d}/{n}  ({acc*100:.1f}%)")
    return acc


def main() -> None:
    """Load checkpoint and run all OOD suites."""
    print()
    print("═" * 65)
    print("  Mamba-130M RLF — OOD Generalization Eval")
    print(f"  Checkpoint: {CKPT}")
    print("═" * 65)
    print()

    if not os.path.exists(CKPT):
        print(f"[ERROR] Checkpoint not found: {CKPT}")
        sys.exit(1)

    print("[INIT] Loading model…")
    backbone = MambaLMHeadModel.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device=DEVICE
    )
    model = RecursiveMamba1_PrefixScratchpad(backbone, lora_rank=4).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()
    print(f"       Loaded {CKPT}\n")

    results = {}

    print("── In-Distribution (ID) ────────────────────────────────────────")
    results["ID 2-5 hops"] = eval_suite(
        model, "Hops 2-5, seen vocab", (2, 5), TRAIN_VALS, seed=11111)
    results["ID 2-8 hops"] = eval_suite(
        model, "Hops 2-8, seen vocab", (2, 8), TRAIN_VALS, seed=22222)
    results["ID adv 2-8"] = eval_suite(
        model, "Hops 2-8, adversarial, seen vocab", (2, 8), TRAIN_VALS,
        adversarial=True, seed=33333)

    print()
    print("── OOD: Longer Hop Chains ──────────────────────────────────────")
    results["OOD 9-11 hops"] = eval_suite(
        model, "Hops 9-11, seen vocab (OOD length)", (9, 11), TRAIN_VALS, seed=44444)
    results["OOD 12-15 hops"] = eval_suite(
        model, "Hops 12-15, seen vocab (OOD length)", (12, 15), TRAIN_VALS, seed=55555)

    print()
    print("── OOD: Novel Vocabulary ───────────────────────────────────────")
    results["OOD novel vocab 2-5"] = eval_suite(
        model, "Hops 2-5, novel vocab (OOD vocab)", (2, 5), OOD_VALS, seed=66666)
    results["OOD novel vocab 2-8"] = eval_suite(
        model, "Hops 2-8, novel vocab (OOD vocab)", (2, 8), OOD_VALS, seed=77777)

    print()
    print("── OOD: Hardest (long + novel + adversarial) ───────────────────")
    results["OOD hard"] = eval_suite(
        model, "Hops 9-12, novel vocab, adversarial", (9, 12), OOD_VALS,
        adversarial=True, seed=88888)

    # Summary
    id_avg  = sum(v for k, v in results.items() if k.startswith("ID"))  / 3
    ood_avg = sum(v for k, v in results.items() if k.startswith("OOD")) / 5

    print()
    print("─" * 65)
    print(f"  ID  average:  {id_avg*100:.1f}%")
    print(f"  OOD average:  {ood_avg*100:.1f}%")
    gap = (id_avg - ood_avg) * 100
    verdict = "✅ GENERALIZES" if gap < 15 else ("⚠️  MILD OVERFIT" if gap < 40 else "❌ OVERFIT")
    print(f"  ID→OOD gap:   {gap:.1f}%  →  {verdict}")
    print("─" * 65)
    print()


    # ── Phase 3 Dark Inference: Hard Suite ────────────────────────────
    # Load the Phase 3 sparse reward checkpoint and run the hardest eval suite
    # with n_dark_inference=3, exactly mirroring the Phase 3 training environment.
    P3_CKPT = "saved_weights/mamba130m_v5_phase5_best.pt"
    if os.path.exists(P3_CKPT):
        print()
        print("═" * 65)
        print("  Phase 3 Dark Inference — Sparse Reward Checkpoint")
        print(f"  Checkpoint: {P3_CKPT}")
        print(f"  n_dark_inference: 3  (mirrors Phase 3 training)")
        print("═" * 65)
        print()

        model_p3 = RecursiveMamba1_PrefixScratchpad(
            MambaLMHeadModel.from_pretrained(
                MODEL_ID, dtype=torch.bfloat16, device=DEVICE
            ),
            lora_rank=4,
        ).to(DEVICE)
        model_p3.load_state_dict(torch.load(P3_CKPT, map_location=DEVICE))
        model_p3.eval()
        print(f"       Loaded {P3_CKPT}\n")

        print("── Phase 3 Dark Inference Suites ───────────────────────────────")
        p3_results = {}
        p3_results["P3 ID 2-5"] = eval_suite(
            model_p3, "[dark] Hops 2-5, seen vocab",
            (2, 5), TRAIN_VALS, n_dark_inference=3, seed=11111)
        p3_results["P3 ID 2-8 adv"] = eval_suite(
            model_p3, "[dark] Hops 2-8, adversarial, seen vocab",
            (2, 8), TRAIN_VALS, adversarial=True, n_dark_inference=3, seed=33333)
        p3_results["P3 OOD novel 2-8"] = eval_suite(
            model_p3, "[dark] Hops 2-8, novel vocab",
            (2, 8), OOD_VALS, n_dark_inference=3, seed=77777)
        p3_results["P3 hard"] = eval_suite(
            model_p3, "[dark] Hops 9-12, novel vocab, adversarial",
            (9, 12), OOD_VALS, adversarial=True, n_dark_inference=3, seed=88888)

        p3_avg = sum(p3_results.values()) / len(p3_results)
        print()
        print("─" * 65)
        print(f"  Phase 3 dark-inference avg: {p3_avg*100:.1f}%")
        print("─" * 65)
        print()


if __name__ == "__main__":
    main()
