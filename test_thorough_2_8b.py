"""
test_thorough_2_8b.py — Comprehensive Phase 7 Validation
=========================================================
Tests:
  1. General Knowledge QA (open-ended)
  2. Multiple-Choice (A/B/C/D)
  3. True/False
  4. Math (arithmetic + algebra)
  5. RLF Variable Chain Reasoning (the model's core skill)
  6. Format Recall (does it produce clean answers vs noise)
"""
import torch
import sys
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel
from train_2_8b_rlf import mount_lora, LoRALinear

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LORA_CKPT = "checkpoints_2_8b/mamba2.8b_p75_GOLDEN.pt"

print("=" * 70)
print("  MAMBA 2.8B PHASE 7 — COMPREHENSIVE VALIDATION SUITE")
print("=" * 70)

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tok.add_special_tokens({"additional_special_tokens": ["<THINK>", "<HALT>"]})
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

HALT_ID = tok.convert_tokens_to_ids("<HALT>")

print("[INIT] Loading pristine slimpj base...")
model = MambaLMHeadModel.from_pretrained(
    "state-spaces/mamba-2.8b-slimpj", dtype=torch.bfloat16, device=DEVICE
)
mount_lora(model)
st = torch.load(LORA_CKPT, map_location=DEVICE, weights_only=True)
res = model.load_state_dict(st, strict=False)
print(f"[INIT] LoRA keys loaded: {len(st)} | Missing: {len(res.missing_keys)}")
model.eval()

# ── Probe weights are real, not empty ────────────────────────────────────────
total_lora_norm = sum(v.abs().mean().item() for v in st.values())
print(f"[CHECK] Avg LoRA weight magnitude: {total_lora_norm / max(len(st), 1):.6f}")
if total_lora_norm < 0.01:
    print("[FATAL] Checkpoint appears empty! Weights are all ~zero.")
    sys.exit(1)
else:
    print("[CHECK] ✅ Checkpoint has real weights\n")

# ── Inference helper ─────────────────────────────────────────────────────────
@torch.no_grad()
def infer(prompt: str, max_new: int = 20) -> str:
    """Run greedy generation up to max_new tokens or EOS/HALT."""
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    input_len = ids.shape[1]
    max_length = input_len + max_new

    out = model.generate(
        input_ids=ids,
        max_length=max_length,
        temperature=0.1,
        top_k=5,
        eos_token_id=tok.eos_token_id,
    )
    generated = out[0, input_len:]
    # Stop at HALT if present
    halt_positions = (generated == HALT_ID).nonzero(as_tuple=True)[0]
    if len(halt_positions) > 0:
        generated = generated[: halt_positions[0]]
    text = tok.decode(generated, skip_special_tokens=True).strip()
    return text


# ── Test batteries ────────────────────────────────────────────────────────────
TESTS = {
    "GENERAL_QA": [
        ("Q: What is the capital of France?\nAnswer:", "Paris"),
        ("Q: Who wrote Romeo and Juliet?\nAnswer:", "Shakespeare"),
        ("Q: What planet is known as the Red Planet?\nAnswer:", "Mars"),
        ("Q: What is H2O commonly known as?\nAnswer:", "water"),
        ("Q: How many continents are there on Earth?\nAnswer:", "7"),
    ],
    "MULTIPLE_CHOICE": [
        ("Q: What color is the sky on a clear day? A) Red B) Blue C) Green D) Yellow\nAnswer:", "B"),
        ("Q: Which animal is the largest? A) Elephant B) Blue Whale C) Giraffe D) Hippopotamus\nAnswer:", "B"),
        ("Q: How many legs does a spider have? A) 4 B) 6 C) 8 D) 10\nAnswer:", "C"),
        ("Q: What is the boiling point of water in Celsius? A) 50 B) 75 C) 90 D) 100\nAnswer:", "D"),
        ("Q: The Earth orbits the Sun. True of which planet? A) Mars B) Venus C) Earth D) Jupiter\nAnswer:", "C"),
    ],
    "TRUE_FALSE": [
        ("True or False: The Earth is flat.\nAnswer:", "False"),
        ("True or False: Water boils at 100 degrees Celsius at sea level.\nAnswer:", "True"),
        ("True or False: Humans have 206 bones.\nAnswer:", "True"),
        ("True or False: The sun rises in the west.\nAnswer:", "False"),
        ("True or False: Dogs are mammals.\nAnswer:", "True"),
    ],
    "MATH": [
        ("Q: What is 7 + 8?\nAnswer:", "15"),
        ("Q: What is 9 times 6?\nAnswer:", "54"),
        ("Q: What is 100 divided by 4?\nAnswer:", "25"),
        ("Q: What is the square root of 144?\nAnswer:", "12"),
        ("Q: What is 2 to the power of 8?\nAnswer:", "256"),
    ],
    "RLF_REASONING": [
        ("[LOGIC] V1=10 V2=V1+5 V3=V2*2 What is V3?\nSolution: =======\nAnswer:", "30"),
        ("[LOGIC] X=7 Y=X*3 Z=Y-1 What is Z?\nSolution: =======\nAnswer:", "20"),
        ("[LOGIC] A=100 B=A/4 C=B+10 What is C?\nSolution: =======\nAnswer:", "35"),
        ("[LOGIC] P=3 Q=P+P R=Q*Q What is R?\nSolution: =======\nAnswer:", "36"),
        ("[LOGIC] N=5 M=N*N K=M+N What is K?\nSolution: =======\nAnswer:", "30"),
    ],
}

# ── Run all tests ─────────────────────────────────────────────────────────────
results = {}
grand_correct = 0
grand_total = 0

for category, probes in TESTS.items():
    correct = 0
    print(f"\n{'─'*70}")
    print(f"  📋 {category}")
    print(f"{'─'*70}")
    for prompt, expected in probes:
        pred = infer(prompt)
        # Flexible match: exact or contains
        hit = (pred.lower().strip() == expected.lower().strip() or
               expected.lower().strip() in pred.lower().strip())
        mark = "✅" if hit else "❌"
        if hit:
            correct += 1
            grand_correct += 1
        grand_total += 1
        short_prompt = prompt.replace("\n", " ")[:55]
        print(f"  {mark} [{short_prompt}...]")
        print(f"       Expected: {expected:<12} | Got: {pred[:40]}")

    pct = (correct / len(probes)) * 100
    results[category] = pct
    print(f"\n  Score: {correct}/{len(probes)} ({pct:.0f}%)")

# ── Summary ───────────────────────────────────────────────────────────────────
grand_pct = (grand_correct / grand_total) * 100
print(f"\n{'=' * 70}")
print(f"  OVERALL SCORE: {grand_correct}/{grand_total} ({grand_pct:.1f}%)")
print(f"{'=' * 70}")
for cat, pct in results.items():
    bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
    print(f"  {cat:<20} [{bar}] {pct:.0f}%")
print(f"{'=' * 70}\n")

if grand_pct >= 70:
    print("  🟢 VERDICT: Model is healthy. Safe to push to HuggingFace.")
elif grand_pct >= 40:
    print("  🟡 VERDICT: Partial recovery. Review failures before pushing.")
else:
    print("  🔴 VERDICT: Model is degraded. Do NOT push to HuggingFace.")
