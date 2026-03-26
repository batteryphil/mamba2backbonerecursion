#!/bin/bash
# launch_phase2.sh — Monitor Phase 1 completion, then auto-start Phase 2
# ════════════════════════════════════════════════════════════════════════
# Usage: bash launch_phase2.sh
# This script polls for:
#   1. Phase 1 process to finish
#   2. Phase 1 checkpoint file to exist
# Then automatically launches Phase 2 joint training.

cd "/home/phil/Desktop/mambadiff/mambadiff llm tts" || exit 1

PHASE1_CKPT="mamba2_2.7b_phase1_scratchpad.pt"
PHASE2_SCRIPT="phase2_joint_training.py"
POLL_INTERVAL=30  # seconds

echo "═══════════════════════════════════════════════════════════"
echo "  Phase 2 Auto-Launcher"
echo "  Monitoring for Phase 1 checkpoint: $PHASE1_CKPT"
echo "  Poll interval: ${POLL_INTERVAL}s"
echo "═══════════════════════════════════════════════════════════"

# Wait for Phase 1 checkpoint to appear
while true; do
    if [ -f "$PHASE1_CKPT" ]; then
        echo ""
        echo "  ✅ Phase 1 checkpoint detected: $PHASE1_CKPT"
        echo "  File size: $(du -h "$PHASE1_CKPT" | cut -f1)"
        echo ""

        # Give Phase 1 a moment to finish its smoke test and exit cleanly
        echo "  Waiting 30s for Phase 1 to fully exit..."
        sleep 30

        echo "  🚀 Launching Phase 2 joint training..."
        echo "═══════════════════════════════════════════════════════════"
        echo ""

        python "$PHASE2_SCRIPT" 2>&1
        exit $?
    fi

    echo "  [$(date +%H:%M:%S)] Waiting for Phase 1... (${POLL_INTERVAL}s)"
    sleep "$POLL_INTERVAL"
done
