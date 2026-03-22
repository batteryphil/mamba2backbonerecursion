#!/usr/bin/env bash
# ============================================================
#  DiM-LLM v4.0 – Clean Slate Protocol Launcher
#  Run from the project directory:
#    cd "/home/phil/Desktop/mambadiff/mambadiff llm tts"
#    bash start_training.sh
# ============================================================
# NOTE: Uses the custom MambaBlock (mamba_diffusion.py) with
#       pure-PyTorch CPU/GPU fallback. mamba_ssm is NOT required.
# ============================================================
set -uo pipefail
cd "$(dirname "$(realpath "$0")")"

echo "=================================================="
echo "  DiM-LLM v4.0 – Clean Slate Protocol (Linux)"
echo "  Backend: Custom MambaBlock (pure-PyTorch fallback)"
echo "=================================================="

# ── 1. Set LD_LIBRARY_PATH so PyTorch shared libs are visible ────────────────
TORCH_LIB=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')" 2>/dev/null || echo "")
if [ -n "$TORCH_LIB" ]; then
    export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
    echo "[✓] LD_LIBRARY_PATH set to include: $TORCH_LIB"
fi

# ── 2. Verify core dependencies ───────────────────────────────────────────────
echo ""
for pkg in torch transformers tqdm; do
    if python3 -c "import $pkg" 2>/dev/null; then
        echo "[✓] $pkg OK"
    else
        echo "[!] $pkg not found. Installing..."
        pip3 install $pkg
    fi
done
echo ""

# ── 3. GPU info ────────────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# ── 4. Check if mamba_scan custom extension works (optional, CUDA-only) ───────
python3 -W ignore -c "
import sys
import torch
sys.path.insert(0, '.')
try:
    import mamba_scan
    print('[✓] mamba_scan CUDA extension loaded (fast GPU path)')
except Exception as e:
    print(f'[~] mamba_scan unavailable ({type(e).__name__}): using pure-PyTorch fallback')
" 2>/dev/null

export PYTHONUNBUFFERED=1

echo ""
echo "[*] Starting training..."
# Suppress the GPT2 tokenizer sequence-length warning (cosmetic only)
python3 -u -W ignore::UserWarning train_llm.py 2>&1 | tee -a training.log
