"""
evaluate_phase4.py — Full Phase 4 Engram Model Evaluation
==========================================================
Tests all 4 capability dimensions of the trained model:
  1. RLF Core Reasoning  — can it still chain logic correctly?
  2. Gate Discrimination — does it correctly accept/reject injections?
  3. Factual Accuracy    — does it use real facts when needed?
  4. OOD Generalization  — does it handle unseen hop counts and facts?
"""

import torch
import random
import os

from mamba_ssm import MambaLMHeadModel
from mamba1_engine import RecursiveMamba1_PrefixScratchpad, MODEL_ID, tokenizer

# ── CPU Engram Table (same as training) ───────────────────────────────────────
CPU_ENGRAM_TABLE = {
    "capital of france":       "Paris",
    "powerhouse of the cell":  "Mitochondria",
    "speed of light":          "299,792 km/s",
    "largest planet":          "Jupiter",
    "boiling point of water":  "100C",
}

# ── OOD facts the model has NEVER seen ────────────────────────────────────────
OOD_ENGRAM_TABLE = {
    "smallest planet":         "Mercury",
    "freezing point of water": "0C",
    "closest star to earth":   "Proxima Centauri",
}

POISON_FACTS = ["Mitochondria", "Jupiter", "Paris", "100C", "299,792 km/s"]
COLORS       = ["Red", "Quantum", "Void", "Titanium", "Azure", "Neon"]

DIVIDER = "=" * 70


def load_model(device):
    """Load the Phase 4 checkpoint."""
    backbone = MambaLMHeadModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device=device)
    model = RecursiveMamba1_PrefixScratchpad(backbone, lora_rank=4).to(device)

    ckpt = "saved_weights/mamba130m_phase4_engram_best.pt"
    if not os.path.exists(ckpt):
        ckpt = "saved_weights/mamba130m_phase4_engram_epoch1.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    model.eval()
    print(f"Loaded checkpoint: {ckpt}")
    return model


def gate_score(model, device, prompt: str, injection: str) -> float:
    """Return the gate value (0.0 to 1.0) for a given prompt + injection pair."""
    input_ids    = tokenizer.encode(prompt,    return_tensors="pt").to(device)
    injection_ids = tokenizer.encode(injection, return_tensors="pt").to(device)
    with torch.no_grad():
        _, gate_logit, gate = model.forward_with_engram(input_ids, injection_ids,
                                            chain_targets=None, ans_starts=None)
    return gate.item()


def rlf_infer(model, device, prompt: str):
    """Run the RLF inference loop and return (loops_used, answer_token)."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        loops, trace, answer = model(input_ids)
    return loops, trace, answer


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Gate Discrimination
# ─────────────────────────────────────────────────────────────────────────────
def test_gate_discrimination(model, device):
    """Check gate ACCEPTS real facts and REJECTS poison on diverse prompts."""
    print(f"\n{DIVIDER}")
    print("  TEST 1: Gate Discrimination")
    print(f"{DIVIDER}")
    correct = 0
    total = 0

    # Factual pairs — gate should ACCEPT (>0.5)
    for key, val in CPU_ENGRAM_TABLE.items():
        prompt    = f"Var_1 = The {key}. Var_2 = Var_1. What is Var_2? Answer:"
        injection = f" [ENGRAM: {val}]"
        gate = gate_score(model, device, prompt, injection)
        ok = gate > 0.5
        correct += int(ok)
        total += 1
        mark = "✅" if ok else "❌"
        print(f"  {mark} FACTUAL  | gate={gate:.3f} | '{key}' → '{val}'")

    # Poison pairs — gate should REJECT (<0.5)
    for col in COLORS[:5]:
        prompt    = f"Alpha = {col}. Beta = Alpha. What is Beta? Answer:"
        injection = f" [ENGRAM: {random.choice(POISON_FACTS)}]"
        gate = gate_score(model, device, prompt, injection)
        ok = gate < 0.5
        correct += int(ok)
        total += 1
        mark = "✅" if ok else "❌"
        print(f"  {mark} POISON   | gate={gate:.3f} | prompt='{col}' + poison injection")

    print(f"\n  Gate Discrimination Score: {correct}/{total} ({100*correct/total:.1f}%)")
    return correct / total


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: RLF Core Reasoning (clean logic chains, no injection)
# ─────────────────────────────────────────────────────────────────────────────
def test_rlf_reasoning(model, device):
    """Verify the RLF loop can still chain pure logic correctly."""
    print(f"\n{DIVIDER}")
    print("  TEST 2: RLF Core Reasoning (In-Distribution)")
    print(f"{DIVIDER}")

    chains = [
        ("A=Blue. B=A. C=B. What is C?",   "Blue"),
        ("X=7. Y=X. Z=Y. What is Z?",       "7"),
        ("P=Cat. Q=P. R=Q. S=R. What is S?", "Cat"),
        ("M=Sky. N=M. O=N. P=O. Q=P. What is Q?", "Sky"),
    ]
    correct = 0
    for prompt, expected in chains:
        loops, trace, answer = rlf_infer(model, device, prompt)
        ok = expected.lower() in answer.lower()
        correct += int(ok)
        mark = "✅" if ok else "❌"
        print(f"  {mark} '{prompt}' → got='{answer}' expected='{expected}' ({loops} loops)")

    print(f"\n  RLF Core Reasoning Score: {correct}/{len(chains)} ({100*correct/len(chains):.1f}%)")
    return correct / len(chains)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: Factual Accuracy (can it USE the injection to answer correctly?)
# ─────────────────────────────────────────────────────────────────────────────
def test_factual_accuracy(model, device):
    """Check whether high-gate injections lead to correct factual answers."""
    print(f"\n{DIVIDER}")
    print("  TEST 3: Factual Accuracy (Engram → Correct Answer)")
    print(f"{DIVIDER}")
    correct = 0
    total = 0
    for key, val in CPU_ENGRAM_TABLE.items():
        injection = f" [ENGRAM: {val}]"
        prompt = f"Var_1 = The {key}. Var_2 = Var_1. What is Var_2? Answer:{injection}"
        loops, trace, answer = rlf_infer(model, device, prompt)
        gate = gate_score(model, device,
                          f"Var_1 = The {key}. Var_2 = Var_1. What is Var_2? Answer:",
                          injection)
        ok = val.lower() in answer.lower()
        correct += int(ok)
        total += 1
        mark = "✅" if ok else "❌"
        print(f"  {mark} gate={gate:.2f} | Q='{key}' expected='{val}' got='{answer}'")

    print(f"\n  Factual Accuracy Score: {correct}/{total} ({100*correct/total:.1f}%)")
    return correct / total


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: OOD Generalization (unseen facts & longer chains)
# ─────────────────────────────────────────────────────────────────────────────
def test_ood_generalization(model, device):
    """Gate and reasoning on facts and chain lengths never seen during training."""
    print(f"\n{DIVIDER}")
    print("  TEST 4: OOD Generalization (Unseen Facts + Longer Chains)")
    print(f"{DIVIDER}")
    correct_gate = 0
    total = 0

    # OOD factual — gate should still ACCEPT (~>0.5)
    for key, val in OOD_ENGRAM_TABLE.items():
        prompt    = f"Var_1 = The {key}. What is Var_1? Answer:"
        injection = f" [ENGRAM: {val}]"
        gate = gate_score(model, device, prompt, injection)
        ok = gate > 0.5
        correct_gate += int(ok)
        total += 1
        mark = "✅" if ok else "❌"
        print(f"  {mark} OOD FACT | gate={gate:.3f} | '{key}' → '{val}'")

    # 8-hop chain (training was 5-hop)
    long_chains = [
        "A=1. B=A. C=B. D=C. E=D. F=E. G=F. H=G. What is H?",
        "Z=Dog. Y=Z. X=Y. W=X. V=W. U=V. T=U. S=T. What is S?",
    ]
    correct_chain = 0
    for prompt in long_chains:
        expected = prompt.split("=")[1].split(".")[0].strip()
        loops, trace, answer = rlf_infer(model, device, prompt)
        ok = expected.lower() in answer.lower()
        correct_chain += int(ok)
        mark = "✅" if ok else "❌"
        print(f"  {mark} 8-HOP    | expected='{expected}' got='{answer}' ({loops} loops)")

    total_ood = total + len(long_chains)
    total_correct = correct_gate + correct_chain
    print(f"\n  OOD Score: {total_correct}/{total_ood} ({100*total_correct/total_ood:.1f}%)")
    return total_correct / total_ood


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Run all evaluation tests and print final summary."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{DIVIDER}")
    print(f"  PHASE 4 FULL EVALUATION — Mamba-130M + Engram Gate")
    print(f"  Device: {device.upper()}")
    print(f"{DIVIDER}")

    model = load_model(device)

    with torch.no_grad():
        s1 = test_gate_discrimination(model, device)
        s2 = test_rlf_reasoning(model, device)
        s3 = test_factual_accuracy(model, device)
        s4 = test_ood_generalization(model, device)

    overall = (s1 + s2 + s3 + s4) / 4.0
    print(f"\n{DIVIDER}")
    print(f"  FINAL SUMMARY")
    print(f"{DIVIDER}")
    print(f"  1. Gate Discrimination:  {s1*100:.1f}%")
    print(f"  2. RLF Core Reasoning:   {s2*100:.1f}%")
    print(f"  3. Factual Accuracy:     {s3*100:.1f}%")
    print(f"  4. OOD Generalization:   {s4*100:.1f}%")
    print(f"  {'─'*40}")
    print(f"  Overall Score:           {overall*100:.1f}%")
    print(f"{DIVIDER}\n")


if __name__ == "__main__":
    main()
