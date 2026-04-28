#!/bin/bash
# auto_eval_rlf.sh — Waits for RLF training to finish, then runs full eval suite.
# Triggered automatically after rlf_trainer_1_4b.py completes.

set -euo pipefail

TRAINER_LOG="/home/phil/.gemini/antigravity/scratch/tiny-refinement/rlf_trainer.log"
EVAL_LOG="/home/phil/.gemini/antigravity/scratch/tiny-refinement/rlf_eval.log"
WORK_DIR="/home/phil/.gemini/antigravity/scratch/tiny-refinement"

echo "========================================" | tee -a "$EVAL_LOG"
echo "  RLF AUTO-EVAL — started $(date)" | tee -a "$EVAL_LOG"
echo "========================================" | tee -a "$EVAL_LOG"

# ── Wait for training to complete ────────────────────────────────────────────
echo "[wait] Polling for training completion..." | tee -a "$EVAL_LOG"
while true; do
    if grep -q "All phases complete" "$TRAINER_LOG" 2>/dev/null; then
        echo "[wait] Training complete at $(date)" | tee -a "$EVAL_LOG"
        break
    fi
    sleep 60
done

# Small buffer to ensure checkpoints are fully flushed to disk
sleep 30

# ── Phase 1: Chain accuracy test (primary RLF validation) ────────────────────
echo "" | tee -a "$EVAL_LOG"
echo "=======================================" | tee -a "$EVAL_LOG"
echo "  PHASE 1: RLF Chain Accuracy Test" | tee -a "$EVAL_LOG"
echo "=======================================" | tee -a "$EVAL_LOG"

cd "$WORK_DIR"
./run_env.sh rlf_chain_test.py 2>&1 | tee -a "$EVAL_LOG"

# ── Phase 2: Extended benchmark (54-test suite) ───────────────────────────────
echo "" | tee -a "$EVAL_LOG"
echo "=======================================" | tee -a "$EVAL_LOG"
echo "  PHASE 2: Extended Benchmark (54 tests)" | tee -a "$EVAL_LOG"
echo "=======================================" | tee -a "$EVAL_LOG"

# Temporarily repoint benchmark to use RLF model
# extended_benchmark uses its own generate() which loads backbone.
# For RLF eval, we run a separate rlf_benchmark wrapper.
./run_env.sh rlf_benchmark.py 2>&1 | tee -a "$EVAL_LOG"

echo "" | tee -a "$EVAL_LOG"
echo "=======================================" | tee -a "$EVAL_LOG"
echo "  AUTO-EVAL DONE — $(date)" | tee -a "$EVAL_LOG"
echo "  Results: $EVAL_LOG" | tee -a "$EVAL_LOG"
echo "=======================================" | tee -a "$EVAL_LOG"
