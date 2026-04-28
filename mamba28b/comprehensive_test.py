"""
comprehensive_test.py — Full model capability evaluation
==========================================================
7 test categories:
  1.  Gate Discrimination (in-distribution)
  2.  Gate OOD Generalization
  3.  Gate Stress Test (adversarial edge cases)
  4.  RLF Reasoning — short chains (2-4 hop)
  5.  RLF Reasoning — long chains (5-8 hop)
  6.  RLF Reasoning — adversarial (noise, distractors)
  7.  Factual Injection (engram → correct token)
"""

import torch
import os
import glob

from mamba_ssm import MambaLMHeadModel
from mamba1_engine import RecursiveMamba1_PrefixScratchpad, MODEL_ID, tokenizer

DIV  = "=" * 70
SDIV = "-" * 70

# ── Factual hash table ────────────────────────────────────────────────────────
FACTS = {
    "capital of france":        "Paris",
    "powerhouse of the cell":   "Mitochondria",
    "speed of light":           "299,792 km/s",
    "largest planet":           "Jupiter",
    "boiling point of water":   "100C",
    "chemical symbol for gold": "Au",
    "inventor of telephone":    "Bell",
    "smallest country":         "Vatican",
    "hardest natural substance": "Diamond",
    "currency of japan":        "Yen",
}

OOD_FACTS = {
    "smallest planet":          "Mercury",
    "freezing point of water":  "0C",
    "closest star to earth":    "Proxima Centauri",
    "number of legs on a spider": "8",
    "atomic number of carbon":  "6",
}

POISON = list(FACTS.values())
COLORS = ["Red", "Quantum", "Void", "Titanium", "Azure", "Neon", "Plasma", "Ghost"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_model(device: str):
    """Load the best Phase 4 checkpoint."""
    backbone = MambaLMHeadModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device=device)
    model = RecursiveMamba1_PrefixScratchpad(backbone, lora_rank=4).to(device)
    ckpts = sorted(glob.glob("saved_weights/mamba130m_phase4_engram_best.pt"))
    ckpt  = ckpts[-1] if ckpts else "saved_weights/mamba130m_phase3_adversarial_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    model.eval()
    print(f"Loaded: {ckpt}\n")
    return model


def gate(model, device, prompt: str, injection: str) -> float:
    """Return sigmoid gate value for a given prompt+injection pair."""
    inp = tokenizer.encode(prompt,    return_tensors="pt").to(device)
    inj = tokenizer.encode(injection, return_tensors="pt").to(device)
    with torch.no_grad():
        _, gl, gv = model.forward_with_engram(inp, inj, chain_targets=None, ans_starts=None)
    return gv.item()


def rlf(model, device, prompt: str):
    """Run RLF inference, return (loops, answer_text)."""
    inp = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        loops, trace, answer = model(inp)
    return loops, answer


def score_section(label, correct, total):
    """Print a section score bar."""
    pct = 100 * correct / total if total else 0
    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    status = "✅ PASS" if pct >= 80 else ("⚠️  WARN" if pct >= 50 else "❌ FAIL")
    print(f"\n  {status}  {label}: {correct}/{total}  [{bar}]  {pct:.0f}%")
    return pct


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: Gate Discrimination — in-distribution
# ═════════════════════════════════════════════════════════════════════════════
def test_gate_indistribution(model, device):
    """Gate must ACCEPT known facts and REJECT poison on in-distribution prompts."""
    print(f"\n{DIV}\n  TEST 1: Gate Discrimination (In-Distribution)\n{DIV}")
    correct = 0
    total   = 0

    for key, val in list(FACTS.items()):
        prompt    = f"Var_1 = The {key}. Var_2 = Var_1. What is Var_2? Answer:"
        injection = f" [ENGRAM: {val}]"
        g = gate(model, device, prompt, injection)
        ok = g > 0.5
        correct += ok; total += 1
        print(f"  {'✅' if ok else '❌'} FACTUAL  gate={g:.3f}  '{key}' → '{val}'")

    import random
    random.seed(42)
    for col in random.sample(COLORS, min(6, len(COLORS))):
        prompt    = f"Alpha = {col}. Beta = Alpha. What is Beta? Answer:"
        injection = f" [ENGRAM: {random.choice(POISON)}]"
        g = gate(model, device, prompt, injection)
        ok = g < 0.5
        correct += ok; total += 1
        print(f"  {'✅' if ok else '❌'} POISON   gate={g:.3f}  prompt='{col}' + random fact")

    return score_section("Gate In-Distribution", correct, total)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: Gate OOD Generalization
# ═════════════════════════════════════════════════════════════════════════════
def test_gate_ood(model, device):
    """Gate must generalize to facts never seen during training."""
    print(f"\n{DIV}\n  TEST 2: Gate OOD Generalization\n{DIV}")
    correct = 0
    total   = 0

    for key, val in OOD_FACTS.items():
        prompt    = f"Var_1 = The {key}. What is Var_1? Answer:"
        injection = f" [ENGRAM: {val}]"
        g = gate(model, device, prompt, injection)
        ok = g > 0.5
        correct += ok; total += 1
        print(f"  {'✅' if ok else '❌'} OOD FACT  gate={g:.3f}  '{key}' → '{val}'")

    # OOD poison — new random words as the "query", known facts as poison
    for nonsense in ["Flibber = Zorp", "Glonk = Quaffle", "Bloop = Snorkel"]:
        prompt    = f"{nonsense}. What is the answer? Answer:"
        injection = f" [ENGRAM: {POISON[0]}]"
        g = gate(model, device, prompt, injection)
        ok = g < 0.5
        correct += ok; total += 1
        print(f"  {'✅' if ok else '❌'} OOD POISON gate={g:.3f}  '{nonsense}' + fact injection")

    return score_section("Gate OOD", correct, total)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: Gate Stress Test
# ═════════════════════════════════════════════════════════════════════════════
def test_gate_stress(model, device):
    """Edge cases: same injection on different prompts, empty-ish injection, etc."""
    print(f"\n{DIV}\n  TEST 3: Gate Stress Tests\n{DIV}")
    correct = 0
    total   = 0

    # Same fact, different question styles
    paris_inj = " [ENGRAM: Paris]"
    cases = [
        (f"The capital of France is located in Europe. What city? Answer:", True,  "direct factual"),
        (f"A=Red. B=A. What is B? Answer:",                                 False, "irrelevant logic"),
        (f"France capital = ? Answer:",                                      True,  "abbreviated factual"),
        (f"X=1. Y=2. Z=X+Y. What is Z? Answer:",                           False, "numeric logic"),
        (f"The largest city in France is the capital. What is it? Answer:", True,  "indirect factual"),
    ]
    for prompt, should_accept, label in cases:
        g = gate(model, device, prompt, paris_inj)
        ok = (g > 0.5) == should_accept
        correct += ok; total += 1
        exp = "ACCEPT" if should_accept else "REJECT"
        got = "ACCEPT" if g > 0.5 else "REJECT"
        print(f"  {'✅' if ok else '❌'} {label:28s}  gate={g:.3f}  expected={exp} got={got}")

    return score_section("Gate Stress", correct, total)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: RLF Reasoning — short chains
# ═════════════════════════════════════════════════════════════════════════════
def test_rlf_short(model, device):
    """2-4 hop variable chains."""
    print(f"\n{DIV}\n  TEST 4: RLF Reasoning — Short Chains (2-4 hops)\n{DIV}")
    chains = [
        ("A=Blue. B=A. What is B?",                     "Blue",  2),
        ("X=7. Y=X. What is Y?",                        "7",     2),
        ("P=Cat. Q=P. R=Q. What is R?",                 "Cat",   3),
        ("M=Sun. N=M. O=N. What is O?",                 "Sun",   3),
        ("A=42. B=A. C=B. D=C. What is D?",             "42",    4),
        ("V1=Moon. V2=V1. V3=V2. V4=V3. What is V4?",  "Moon",  4),
        ("Z=True. Y=Z. X=Y. What is X?",                "True",  3),
        ("Q=Apple. R=Q. S=R. What is S?",               "Apple", 3),
    ]
    correct = 0
    for prompt, expected, hops in chains:
        loops, answer = rlf(model, device, prompt)
        ok = expected.lower() in answer.lower()
        correct += ok
        print(f"  {'✅' if ok else '❌'} {hops}-hop  got='{answer}'  expected='{expected}'  ({loops} loops)")

    return score_section("RLF Short Chains", correct, len(chains))


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5: RLF Reasoning — long chains
# ═════════════════════════════════════════════════════════════════════════════
def test_rlf_long(model, device):
    """5-8+ hop chains — tests OOD length generalization."""
    print(f"\n{DIV}\n  TEST 5: RLF Reasoning — Long Chains (5-8 hops)\n{DIV}")
    chains = [
        ("A=Gamma. B=A. C=B. D=C. E=D. What is E?",                               "Gamma",  5),
        ("V1=Dog. V2=V1. V3=V2. V4=V3. V5=V4. What is V5?",                       "Dog",    5),
        ("X=99. Y=X. Z=Y. A=Z. B=A. C=B. What is C?",                            "99",     6),
        ("A=Fire. B=A. C=B. D=C. E=D. F=E. G=F. What is G?",                      "Fire",   7),
        ("V1=Key. V2=V1. V3=V2. V4=V3. V5=V4. V6=V5. V7=V6. V8=V7. What is V8?", "Key",   8),
    ]
    correct = 0
    for prompt, expected, hops in chains:
        loops, answer = rlf(model, device, prompt)
        ok = expected.lower() in answer.lower()
        correct += ok
        print(f"  {'✅' if ok else '❌'} {hops}-hop  got='{answer}'  expected='{expected}'  ({loops} loops)")

    return score_section("RLF Long Chains", correct, len(chains))


# ═════════════════════════════════════════════════════════════════════════════
# TEST 6: RLF Adversarial Reasoning
# ═════════════════════════════════════════════════════════════════════════════
def test_rlf_adversarial(model, device):
    """Noisy prompts, distractors, red herrings."""
    print(f"\n{DIV}\n  TEST 6: RLF Adversarial Reasoning\n{DIV}")
    chains = [
        # Distractor variable in the chain
        ("A=Red. DISTRACTOR=Blue. B=A. What is B?",         "Red",   "distractor variable"),
        # Misleading value shadowed by later reassignment
        ("A=Wrong. B=A. B=Correct. What is B?",             "Correct","shadowed variable"),
        # Number in string context
        ("X=Hello123. Y=X. What is Y?",                     "Hello123","alphanumeric value"),
        # Upper/lower case var names
        ("aa=lower. AA=aa. What is AA?",                    "lower", "case sensitivity"),
        # Long distractor sentence in middle
        ("A=Star. Note: this is irrelevant. B=A. What is B?","Star",  "inline noise"),
        # Multiple distractors
        ("A=Gold. B=Silver. C=A. What is C?",               "Gold",  "multi-var distractor"),
    ]
    correct = 0
    for prompt, expected, label in chains:
        loops, answer = rlf(model, device, prompt)
        ok = expected.lower() in answer.lower()
        correct += ok
        print(f"  {'✅' if ok else '❌'} {label:25s}  got='{answer}'  expected='{expected}'  ({loops} loops)")

    return score_section("RLF Adversarial", correct, len(chains))


# ═════════════════════════════════════════════════════════════════════════════
# TEST 7: Factual Injection (Engram → Answer token)
# ═════════════════════════════════════════════════════════════════════════════
def test_factual_injection(model, device):
    """Does the model correctly output the injected fact as the answer?"""
    print(f"\n{DIV}\n  TEST 7: Factual Injection (Engram → Correct Answer Token)\n{DIV}")
    correct = 0
    for key, val in list(FACTS.items())[:6]:
        injection = f" [ENGRAM: {val}]"
        prompt    = f"Var_1 = The {key}. What is Var_1? Answer:{injection}"
        g         = gate(model, device,
                         f"Var_1 = The {key}. What is Var_1? Answer:", injection)
        loops, answer = rlf(model, device, prompt)
        ok = val.lower() in answer.lower()
        correct += ok
        print(f"  {'✅' if ok else '❌'} gate={g:.2f}  Q='{key}'  expected='{val}'  got='{answer}'")

    return score_section("Factual Injection", correct, 6)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    """Run all comprehensive tests and print final report."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{DIV}")
    print(f"  COMPREHENSIVE EVALUATION — Mamba-130M RLF + Engram Gate")
    print(f"  Device: {device.upper()}")
    print(f"{DIV}")

    model = load_model(device)

    with torch.no_grad():
        s1 = test_gate_indistribution(model, device)
        s2 = test_gate_ood(model, device)
        s3 = test_gate_stress(model, device)
        s4 = test_rlf_short(model, device)
        s5 = test_rlf_long(model, device)
        s6 = test_rlf_adversarial(model, device)
        s7 = test_factual_injection(model, device)

    overall = (s1 + s2 + s3 + s4 + s5 + s6 + s7) / 7.0
    print(f"\n{DIV}")
    print(f"  COMPREHENSIVE REPORT")
    print(f"{DIV}")
    print(f"  {'Gate In-Distribution:':30s} {s1:.0f}%")
    print(f"  {'Gate OOD:':30s} {s2:.0f}%")
    print(f"  {'Gate Stress:':30s} {s3:.0f}%")
    print(f"  {'RLF Short Chains (2-4 hop):':30s} {s4:.0f}%")
    print(f"  {'RLF Long Chains (5-8 hop):':30s} {s5:.0f}%")
    print(f"  {'RLF Adversarial:':30s} {s6:.0f}%")
    print(f"  {'Factual Injection:':30s} {s7:.0f}%")
    print(f"  {SDIV[:50]}")
    print(f"  {'Overall Score:':30s} {overall:.0f}%")
    print(f"{DIV}\n")


if __name__ == "__main__":
    main()
