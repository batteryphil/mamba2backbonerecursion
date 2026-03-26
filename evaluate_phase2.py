"""
evaluate_phase2.py — OOD Stress Testing for Dual-Architecture Mamba
====================================================================
Loads the best Phase 2 checkpoint and runs 4 adversarial stress tests
to break the Latent Bridge, Temporal Attention, and O(1) constraints.
"""

import torch
import time
import gc
import sys
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel
from mamba_engine import (
    RecursiveMamba2_PrefixScratchpad, tokenizer, DEVICE, MODEL_ID,
    LORA_RANK, HALT_ID
)

CHECKPOINT_PATH = "mamba2_2.7b_phase2_joint_best.pt"

print(f"\n{'='*70}")
print("  INITIALIZING PHASE 2 EVALUATION ENGINE")
print(f"{'='*70}\n")

# ── 1. Load Model & Weights ───────────────────────────────────────────────
print(f"Loading Base Model: {MODEL_ID}...")
base_model = MambaLMHeadModel.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device=DEVICE
)

import torch.nn as nn
new_vocab = len(tokenizer)
old_vocab = base_model.backbone.embedding.weight.shape[0]
d_model = base_model.backbone.embedding.embedding_dim

if new_vocab > old_vocab:
    ne = nn.Embedding(new_vocab, d_model, dtype=torch.bfloat16)
    nn.init.normal_(ne.weight, std=0.02)
    ne.weight.data[:old_vocab] = base_model.backbone.embedding.weight.data
    base_model.backbone.embedding = ne
    nh = nn.Linear(d_model, new_vocab, bias=False, dtype=torch.bfloat16)
    nn.init.normal_(nh.weight, std=0.02)
    nh.weight.data[:old_vocab] = base_model.lm_head.weight.data
    base_model.lm_head = nh

for p in base_model.parameters():
    p.requires_grad = False

model = RecursiveMamba2_PrefixScratchpad(base_model, lora_rank=LORA_RANK).to(DEVICE)

print(f"Loading Phase 2 Checkpoint: {CHECKPOINT_PATH}...")
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
del ckpt
torch.cuda.empty_cache()

model.eval()
print(f"✅ Engine Ready. VRAM Usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB\n")


# ── 2. The Stress Tests ───────────────────────────────────────────────────
def generate_50_hop_marathon() -> str:
    """Generate a 50-hop abstract variable chain."""
    vars_l = [f"V{i}" for i in range(1, 52)]
    chain = f"{vars_l[0]} = Nebula. "
    for i in range(1, len(vars_l)):
        chain += f"{vars_l[i]} = {vars_l[i-1]}. "
    chain += f"What is {vars_l[-1]}?\nAnswer:"
    return chain


def generate_amnesia_test() -> str:
    """Generate a chain interrupted by a massive block of distractor text."""
    distractor = " The mitochondria is the powerhouse of the cell. " * 100
    return (f"Alpha = Titanium. {distractor}"
            f"Beta = Alpha. What is Beta?\nAnswer:")


tests = [
    {
        "name": "Test 1: Semantic Translation (Bridge Attack)",
        "prompt": ("John is the father of Michael. Michael is the brother of "
                   "Sarah. Sarah is the mother of David. David is the father "
                   "of Emma. Emma is the sister of Lucas. Who is John to "
                   "Lucas?\nAnswer:"),
        "expected": "great",
        "description": "Tests if the bridge can translate kinship→English",
    },
    {
        "name": "Test 2: Adversarial Distractor (Temporal Attention Attack)",
        "prompt": ("V1 = Quantum. V2 = V1. The weather in Chicago is usually "
                   "cold in the winter. V3 = V2. Did you know that the Eiffel "
                   "Tower grows in the summer? V4 = V3. I really need to buy "
                   "some milk today. V5 = V4. What is V5?\nAnswer:"),
        "expected": "Quantum",
        "description": "Tests if distractor text overwrites variable memory",
    },
    {
        "name": "Test 3: Infinite Extrapolation (O(1) VRAM Attack)",
        "prompt": generate_50_hop_marathon(),
        "expected": "Nebula",
        "description": "Tests O(1) memory + 50-hop compositionality",
    },
    {
        "name": "Test 4: Latent Amnesia (Compression Attack)",
        "prompt": generate_amnesia_test(),
        "expected": "Titanium",
        "description": "Tests if massive distractor overwrites the SSM state",
    },
]


# ── 3. Execution Engine ───────────────────────────────────────────────────
results = []
for test in tests:
    print(f"\n{'─'*70}")
    print(f"▶ {test['name']}")
    print(f"  {test['description']}")
    print(f"  Expected: {test['expected']}")
    print(f"{'─'*70}")

    input_ids = tokenizer.encode(
        test['prompt'], add_special_tokens=False, return_tensors="pt"
    ).to(DEVICE)

    print(f"  Input tokens: {input_ids.shape[1]}")

    start_time = time.time()
    vram_before = torch.cuda.memory_allocated() / 1e9

    with torch.no_grad():
        loops_taken, trace, answer = model(input_ids)

    vram_after = torch.cuda.memory_allocated() / 1e9
    run_time = time.time() - start_time

    hit = test['expected'].lower() in answer.lower()
    status = "✅ PASSED" if hit else f"❌ FAILED → {answer!r}"

    print(f"\n  Result: {status}")
    print(f"  Loops: {loops_taken} | Time: {run_time:.2f}s")
    print(f"  VRAM: {vram_before:.4f} → {vram_after:.4f} GB"
          f" | Delta: {vram_after - vram_before:+.4f} GB")

    # Print the gate routing trace
    print(f"\n  Attention Routing Trace:")
    for lbl, tok, prob in trace:
        marker = ""
        if tok.lower() == test['expected'].lower():
            marker = " ← ✅"
        elif tok == "<HALT>":
            marker = " ← HALT"
        print(f"    {lbl:5s}  {tok!r:15s}  p={prob:.4f}{marker}")

    results.append({
        "name": test["name"],
        "passed": hit,
        "answer": answer,
        "loops": loops_taken,
        "time": run_time,
        "vram_delta": vram_after - vram_before,
    })

    # Force GC for accurate VRAM on next test
    del input_ids
    torch.cuda.empty_cache()
    gc.collect()


# ── 4. Summary ────────────────────────────────────────────────────────────
print(f"\n\n{'='*70}")
print("  EVALUATION RESULTS")
print(f"{'='*70}\n")

passed = sum(1 for r in results if r["passed"])
total = len(results)

for r in results:
    icon = "✅" if r["passed"] else "❌"
    print(f"  {icon} {r['name']}")
    print(f"     Answer: {r['answer']!r} | Loops: {r['loops']}"
          f" | VRAM Δ: {r['vram_delta']:+.4f} GB")

print(f"\n  Score: {passed}/{total}")

# O(1) Memory verification
test3 = results[2]
if abs(test3["vram_delta"]) < 0.01:
    print(f"\n  🧠 O(1) MEMORY CONSTRAINT VERIFIED:"
          f" {test3['vram_delta']:+.4f} GB across {test3['loops']} loops")
else:
    print(f"\n  ⚠️  O(1) memory violation: {test3['vram_delta']:+.4f} GB")

print(f"\n{'='*70}\n")
