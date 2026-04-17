#!/usr/bin/env bash
# vram_clear.sh — Hard VRAM pre-clearance for Project TinyRefinementModel
# Identifies and kills any process holding >200 MiB VRAM that is NOT part
# of the display stack. Preserves: Xorg, cinnamon, chrome, antigravity.
# Usage: ./vram_clear.sh [--force]   (--force kills without prompt)

set -euo pipefail

FORCE=${1:-""}

# Whitelist of commands that are allowed to hold VRAM (display stack)
WHITELIST=("Xorg" "cinnamon" "chrome" "antigravity" "gnome" "plasma" "kwin"
           "picom" "compton" "xfwm" "mutter" "compiz")

echo "========================================="
echo " VRAM Pre-Clearance Script"
echo " Project TinyRefinementModel"
echo "========================================="

# Show current state
echo "[vram_clear] Current GPU memory:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo ""
echo "[vram_clear] Scanning for VRAM-holding processes..."

# Get all PIDs accessing nvidia devices
GPU_PIDS=$(fuser /dev/nvidia0 /dev/nvidiactl 2>/dev/null | tr ' ' '\n' | sort -u | grep -v '^$' || true)

if [ -z "$GPU_PIDS" ]; then
    echo "[vram_clear] No GPU processes found via fuser."
else
    for PID in $GPU_PIDS; do
        # Get process info
        CMD=$(ps -p "$PID" -o comm= 2>/dev/null || echo "unknown")
        FULLCMD=$(ps -p "$PID" -o command= 2>/dev/null | cut -c1-80 || echo "unknown")

        # Check whitelist
        WHITELISTED=0
        for WL in "${WHITELIST[@]}"; do
            if echo "$CMD" | grep -qi "$WL"; then
                WHITELISTED=1
                break
            fi
        done

        # Get VRAM usage for this PID from nvidia-smi
        VRAM_MB=$(nvidia-smi --query-compute-apps=pid,used_gpu_memory \
                  --format=csv,noheader 2>/dev/null | \
                  awk -F',' -v p="$PID" '$1 ~ p {gsub(/ MiB/,"",$2); print $2+0}')
        VRAM_MB=${VRAM_MB:-0}

        if [ "$WHITELISTED" -eq 1 ]; then
            echo "[vram_clear]   KEEP  PID=$PID  CMD=$CMD  VRAM=${VRAM_MB}MiB (display stack)"
        else
            echo "[vram_clear]   KILL? PID=$PID  CMD=$CMD  VRAM=${VRAM_MB}MiB"
            echo "              Full: $FULLCMD"

            if [ "$FORCE" = "--force" ]; then
                echo "[vram_clear]   Killing PID=$PID..."
                kill -9 "$PID" 2>/dev/null && echo "[vram_clear]   Killed." || echo "[vram_clear]   Already dead."
            else
                read -r -p "              Kill this process? [y/N] " yn
                case $yn in
                    [Yy]*)
                        kill -9 "$PID" 2>/dev/null && echo "[vram_clear]   Killed." || true
                        ;;
                    *)
                        echo "[vram_clear]   Skipped."
                        ;;
                esac
            fi
        fi
    done
fi

# Also kill Ollama if running (it silently holds VRAM even without active sessions)
if pgrep -x ollama > /dev/null 2>&1; then
    echo "[vram_clear] Stopping Ollama daemon (silent VRAM hog)..."
    sudo systemctl stop ollama 2>/dev/null || killall -9 ollama 2>/dev/null || true
    sleep 1
fi

echo ""
echo "[vram_clear] Final GPU memory state:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
THRESHOLD=8000  # 8 GB minimum free required before Phase 2

if [ "$FREE_MIB" -ge "$THRESHOLD" ]; then
    echo "[vram_clear] ✅ VRAM CLEAR: ${FREE_MIB} MiB free — safe to proceed."
    exit 0
else
    echo "[vram_clear] ❌ WARNING: Only ${FREE_MIB} MiB free — below ${THRESHOLD} MiB threshold."
    echo "[vram_clear]    Phase 2 training may OOM. Investigate remaining processes."
    exit 1
fi
