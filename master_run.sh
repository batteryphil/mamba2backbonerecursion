#!/usr/bin/env bash

set -e

echo "======================================================================"
echo "  MAMBA-130M RLF: DEFINITIVE MASTER EXECUTION HARNESS"
echo "======================================================================"
echo "  This script executes the complete 7-Phase curriculum to compile"
echo "  and validate the final Turing-complete reasoning engine."
echo "======================================================================"

# ── Step 1: Execute Training (Phases 1 through 6) ───────────────────────────
echo ""
echo "[MASTER] Starting Phase 1-6 Automated Curriculum..."
python -u train_130m.py || { echo "[ERROR] Training pipeline failed."; exit 1; }

echo "[MASTER] Training completed successfully. Final weights generated."

# ── Step 2: Ensure test scripts points to Phase 6 weights ───────────────────
# Update the CKPT variable in the test scripts to use the final weights
echo "[MASTER] Wiring validation gauntlet to final weights..."
sed -i 's|saved_weights/.*_best\.pt|saved_weights/mamba130m_v6_best.pt|g' test_asymptotic.py
sed -i 's|saved_weights/.*_best\.pt|saved_weights/mamba130m_v6_best.pt|g' test_ablation.py
sed -i 's|saved_weights/.*_best\.pt|saved_weights/mamba130m_v6_best.pt|g' test_babi.py

# ── Step 3: Phase 7 Validation Gauntlet ─────────────────────────────────────
echo ""
echo "[MASTER] Commencing Phase 7 Validation Gauntlet..."

echo "--------------------------------------------------------"
echo "  Test 7a: Asymptotic Length (O(1) Memory Proof)"
echo "--------------------------------------------------------"
python -u test_asymptotic.py | tee test_asymptotic_final.txt

echo ""
echo "--------------------------------------------------------"
echo "  Test 7b: Mechanistic Ablation Proof"
echo "--------------------------------------------------------"
python -u test_ablation.py | tee test_ablation_final.txt

echo ""
echo "--------------------------------------------------------"
echo "  Test 7c: Semantic Syntax Shift (bAbI Evaluation)"
echo "--------------------------------------------------------"
python -u test_babi.py | tee test_babi_final.txt

echo ""
echo "======================================================================"
echo "  MASTER HARNESS COMPLETE"
echo "======================================================================"
echo "  Mamba-130M RLF v6 final baseline compiled and validated."
