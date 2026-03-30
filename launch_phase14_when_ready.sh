#!/usr/bin/env bash
# =============================================================================
# Phase 14 Auto-Launch Daemon
# =============================================================================
# Monitors Phase 13 PID until it exits cleanly, then automatically launches
# the Phase 14 Inner-Loop Bypass trainer. Logs everything to training_p14.log.
# =============================================================================

set -euo pipefail

PHASE13_PID=127685
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P14_SCRIPT="$SCRIPT_DIR/phase14_inner_loop_bypass_trainer.py"
P14_LOG="$SCRIPT_DIR/training_p14.log"
P13_CHECKPOINT="$SCRIPT_DIR/checkpoints/mamba3_p13_universal_mastered.pt"

echo "============================================================"
echo "  MAMBA-3 PHASE 14 AUTO-LAUNCH DAEMON"
echo "  Watching Phase 13 PID: $PHASE13_PID"
echo "  Will launch: $P14_SCRIPT"
echo "  Output log: $P14_LOG"
echo "============================================================"

# Wait for Phase 13 to terminate
echo "[DAEMON] Blocking on PID $PHASE13_PID..."
while kill -0 "$PHASE13_PID" 2>/dev/null; do
    sleep 15
done

echo "[DAEMON] Phase 13 PID $PHASE13_PID has exited. Checking exit state..."

# Verify the Phase 13 mastered checkpoint was actually written
if [ ! -f "$P13_CHECKPOINT" ]; then
    echo "[DAEMON] FATAL: Phase 13 checkpoint not found at $P13_CHECKPOINT"
    echo "[DAEMON] Phase 13 may have crashed. Aborting Phase 14 launch."
    exit 1
fi

echo "[DAEMON] Phase 13 checkpoint confirmed. Igniting Phase 14..."
echo "[DAEMON] $(date) - Phase 14 launched."

cd "$SCRIPT_DIR"
python "$P14_SCRIPT" > "$P14_LOG" 2>&1 &

P14_PID=$!
echo "[DAEMON] Phase 14 PID: $P14_PID"
echo "[DAEMON] Monitor with: tail -f training_p14.log"
echo "[DAEMON] All systems nominal. Daemon exiting."
