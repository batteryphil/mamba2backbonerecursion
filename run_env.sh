#!/usr/bin/env bash
# run_env.sh — Environment shim for Project TinyRefinementModel
# Injects all required CUDA library paths and launches any Python script.
# Usage: ./run_env.sh phase1_distiller.py
#        ./run_env.sh phase2_sculptor_trainer.py

set -euo pipefail

PYTHON=/home/phil/.local/share/mise/installs/python/3.14.3/bin/python
SITE=/home/phil/.local/share/mise/installs/python/3.14.3/lib/python3.14/site-packages

export LD_PRELOAD="${SITE}/nvidia/nccl/lib/libnccl.so.2:${SITE}/nvidia/cu13/lib/libnvJitLink.so.13"
export PYTHONPATH="${PYTHONPATH:-}"

# Disable tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false
# Deterministic CUDA ops where possible
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "[run_env] LD_PRELOAD set for NCCL + nvJitLink"
echo "[run_env] Launching: $PYTHON $*"
exec "$PYTHON" "$@"
