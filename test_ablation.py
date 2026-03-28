"""
Test 3: Scratchpad Ablation Proof
==================================
Mechanistically proves that the Prefix Latent Scratchpad and Latent Bridge 
are the load-bearing components driving the model's reasoning capability.
"""
import torch
import copy
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba1_engine import RecursiveMamba1_PrefixScratchpad
from ood_eval import eval_suite
import re

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "saved_weights/mamba130m_v5_phase5_best.pt"

TRAIN_VALS = [str(i) for i in range(1, 999_999 + 1, 137)]
with open("train_130m.py") as f: src = f.read()
import ast
for node in ast.parse(src).body:
    if isinstance(node, ast.Assign) and len(node.targets)==1 and getattr(node.targets[0], 'id', '') == 'TRAIN_VALS':
        pass
        break

def main():
    print("=" * 70)
    print("  Test 3: Scratchpad Ablation Proof")
    print(f"  Checkpoint: {CKPT}")
    print(f"  Device: {DEVICE.upper()}")
    print("=" * 70)

    print("\n[INIT] Loading model...")
    backbone = MambaLMHeadModel.from_pretrained(
        "state-spaces/mamba-130m", dtype=torch.bfloat16, device=DEVICE
    )
    model = RecursiveMamba1_PrefixScratchpad(backbone, lora_rank=4).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()
    print("  Model loaded OK\n")

    # ── Run A: Baseline ──────────────────────────────────────────────────
    print("Run A: Normal inference (trained architecture)...")
    acc_a = eval_suite(model, "Baseline", (2, 8), TRAIN_VALS, seed=22222, n=100) * 100
    
    # ── Run B: Zero Scratchpad ───────────────────────────────────────────
    print("\nRun B: latent_memory zeroed (ablated scratchpad)...")
    model_b = copy.deepcopy(model)
    with torch.no_grad():
        model_b.latent_memory.data.zero_()
    acc_b = eval_suite(model_b, "Ablated Scratchpad", (2, 8), TRAIN_VALS, seed=22222, n=100) * 100

    # ── Run C: Zero Bridge ───────────────────────────────────────────────
    print("\nRun C: Latent bridge weights zeroed (ablated bridge)...")
    model_c = copy.deepcopy(model)
    with torch.no_grad():
        model_c.bridge_down.weight.data.zero_()
        if model_c.bridge_down.bias is not None:
            model_c.bridge_down.bias.data.zero_()
        model_c.bridge_up.weight.data.zero_()
        if model_c.bridge_up.bias is not None:
            model_c.bridge_up.bias.data.zero_()
    acc_c = eval_suite(model_c, "Ablated Bridge", (2, 8), TRAIN_VALS, seed=22222, n=100) * 100

    # ── Verdict ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ABLATION VERDICT")
    print("=" * 70)
    print(f"\n  Run A — Normal:          {acc_a:>5.1f}%   (baseline)")
    print(f"  Run B — Zero scratchpad: {acc_b:>5.1f}%   (Δ {acc_b - acc_a:+.1f}pp)")
    print(f"  Run C — Zero bridge:     {acc_c:>5.1f}%   (Δ {acc_c - acc_a:+.1f}pp)\n")

    drop_b = acc_a - acc_b
    if drop_b >= acc_a * 0.5 and acc_a > 10.0:
        print(f"  ✅ Scratchpad ablation caused >{50}% relative drop ({drop_b:.1f}pp)")
        print("     MECHANISTIC PROOF: latent_memory is load-bearing")

    drop_c = acc_a - acc_c
    if drop_c >= acc_a * 0.5 and acc_a > 10.0:
        print(f"  ✅ Bridge ablation caused >{50}% relative drop ({drop_c:.1f}pp)")
        print("     MECHANISTIC PROOF: latent bridge is the routing conduit")

    print()

if __name__ == "__main__":
    main()
