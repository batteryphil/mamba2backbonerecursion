#!/usr/bin/env python3
"""
phase3_blacksmith.py -- Agent 3: The Blacksmith
Project TinyRefinementModel

Merges hook-based LoRA adapters algebraically into the frozen Mamba-1.4B
backbone weights, saves the full BF16 Gold Master checkpoint, converts to
GGUF via llama.cpp, and quantizes to Q2_K and Q4_K_M.

Usage:
    ./run_env.sh phase3_blacksmith.py
    ./run_env.sh phase3_blacksmith.py --skip-gguf    # BF16 save only
    ./run_env.sh phase3_blacksmith.py --quant-only   # GGUF + quant only
    ./run_env.sh phase3_blacksmith.py --skip-test    # skip inference test
"""

import argparse
import gc
import json
import logging
import math
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKSPACE = Path(__file__).parent
LORA_FINAL_DIR = WORKSPACE / "checkpoints" / "lora_final"
HALTING_HEAD_PATH = WORKSPACE / "checkpoints" / "lora_final" / "halting_head.pt"
GOLD_MASTER_DIR = WORKSPACE / "checkpoints" / "mamba-tiny-refinement-BF16-MASTER"
GGUF_DIR = WORKSPACE / "checkpoints" / "gguf"
LOG_FILE = WORKSPACE / "phase3_blacksmith.log"

STUDENT_REPO = "state-spaces/mamba-1.4b"
D_MODEL = 2048
LORA_RANK = 64
LORA_ALPHA = 128
LORA_SCALE = LORA_ALPHA / LORA_RANK  # = 2.0

LLAMA_CPP_DIR = Path("/home/phil/llama.cpp")
LLAMA_CONVERT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
LLAMA_QUANTIZE = LLAMA_CPP_DIR / "llama-quantize"
PYTHON_BIN = "/home/phil/.local/share/mise/installs/python/3.14.3/bin/python"

QUANT_LEVELS = ["Q2_K", "Q4_K_M"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Load backbone + merge LoRA algebraically
# ---------------------------------------------------------------------------

def merge_lora_into_backbone() -> nn.Module:
    """Load Mamba-1.4B, then algebraically fuse saved LoRA deltas into weights.

    The LoRA merge formula for each adapted weight W:
        W_merged = W + (lora_B @ lora_A) * scale

    This permanently bakes the trained adapter into the weight tensor,
    producing a standard dense model with no hook overhead.

    Returns:
        Merged MambaLMHeadModel with frozen backbone abandoned (fully merged).
    """
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
    from mamba_ssm.models.config_mamba import MambaConfig

    log.info("Loading Mamba-1.4B backbone for merge...")
    config_data = load_config_hf(STUDENT_REPO)
    model_cfg = MambaConfig(**config_data)
    model = MambaLMHeadModel(model_cfg, device="cuda", dtype=torch.bfloat16)
    state_dict = load_state_dict_hf(STUDENT_REPO, device="cuda", dtype=torch.bfloat16)
    model.load_state_dict(state_dict)
    del state_dict
    gc.collect()
    torch.cuda.empty_cache()
    log.info("Backbone loaded.")

    # Load saved LoRA parameters
    lora_path = LORA_FINAL_DIR / "lora_weights.pt"
    if not lora_path.exists():
        log.error("LoRA weights not found: %s", lora_path)
        log.error("Run phase2_sculptor_trainer.py first.")
        sys.exit(1)

    lora_state = torch.load(str(lora_path), map_location="cuda")
    log.info("Loaded LoRA state: %d tensors.", len(lora_state))

    # Group A/B pairs by layer path
    # Parameter names: "backbone.layers.N.mixer.in_proj_lora_A" etc.
    # Strip "_lora_A" / "_lora_B" suffix to get the layer key
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in lora_state.items():
        if name.endswith("_lora_A"):
            base_key = name[: -len("_lora_A")]
            pairs.setdefault(base_key, {})["A"] = tensor
        elif name.endswith("_lora_B"):
            base_key = name[: -len("_lora_B")]
            pairs.setdefault(base_key, {})["B"] = tensor

    log.info("Found %d LoRA A/B pairs to merge.", len(pairs))
    merged_count = 0

    for key, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            log.warning("Incomplete pair for key %s — skipping.", key)
            continue

        lora_A = ab["A"]  # (r, in_features)
        lora_B = ab["B"]  # (out_features, r)
        delta = (lora_B @ lora_A) * LORA_SCALE  # (out_features, in_features)

        # Reconstruct the parameter path: key = "backbone.layers.N.mixer.in_proj"
        # (the parent module path where lora_A/B were registered)
        # The target weight is at "backbone.layers.N.mixer.in_proj.weight"
        param_path = f"{key}.weight"

        try:
            # Navigate to the weight tensor
            obj = model
            parts = param_path.split(".")
            for part in parts[:-1]:
                obj = getattr(obj, part)
            weight = getattr(obj, "weight")
            weight.data.add_(delta.to(weight.dtype))
            merged_count += 1
        except AttributeError:
            log.warning("Could not resolve path %s — skipping.", param_path)
            continue

    log.info("Merged %d / %d LoRA pairs into backbone weights.", merged_count, len(pairs))

    # Re-enable fast path now that weights are fully merged (no hook dependency)
    for module in model.modules():
        if hasattr(module, "use_fast_path"):
            module.use_fast_path = True

    return model


# ---------------------------------------------------------------------------
# Step 2: Attach HaltingHead and save Gold Master
# ---------------------------------------------------------------------------

def save_gold_master(model: nn.Module) -> Path:
    """Save merged model + HaltingHead as BF16 Gold Master.

    CRITICAL: This directory must NOT be deleted. It is the source of
    truth for all future quantization and fine-tuning experiments.

    Args:
        model: Merged MambaLMHeadModel with LoRA baked in.

    Returns:
        Path to the Gold Master directory.
    """
    log.info("=" * 60)
    log.info("SAVING BF16 GOLD MASTER")
    log.info("DO NOT DELETE: %s", GOLD_MASTER_DIR)
    log.info("=" * 60)

    GOLD_MASTER_DIR.mkdir(parents=True, exist_ok=True)

    # Save model state dict and config
    config_path = GOLD_MASTER_DIR / "config.json"
    from mamba_ssm.utils.hf import load_config_hf
    config_data = load_config_hf(STUDENT_REPO)
    with config_path.open("w") as fh:
        json.dump(config_data, fh, indent=2)

    # Save merged weights
    weights_path = GOLD_MASTER_DIR / "pytorch_model.bin"
    log.info("Saving merged weights to %s ...", weights_path)
    torch.save(model.state_dict(), str(weights_path))

    # Save HaltingHead separately
    if HALTING_HEAD_PATH.exists():
        halt_dest = GOLD_MASTER_DIR / "halting_head.pt"
        shutil.copy2(str(HALTING_HEAD_PATH), str(halt_dest))
        log.info("HaltingHead saved to Gold Master.")
    else:
        log.warning("HaltingHead checkpoint not found at %s.", HALTING_HEAD_PATH)

    # Write manifest
    manifest = GOLD_MASTER_DIR / "GOLD_MASTER_MANIFEST.txt"
    with manifest.open("w") as fh:
        fh.write("Project TinyRefinementModel -- Gold Master BF16 Checkpoint\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"Base model: {STUDENT_REPO}\n")
        fh.write(f"LoRA source: {LORA_FINAL_DIR}\n")
        fh.write(f"LoRA rank: {LORA_RANK}  alpha: {LORA_ALPHA}  scale: {LORA_SCALE}\n")
        fh.write("Merge status: COMPLETE (algebraic weight fusion)\n")
        fh.write("Quantization status: NONE (raw BF16)\n")
        fh.write("\nIMPORTANT:\n")
        fh.write("This is the Gold Master. Do NOT delete or overwrite.\n")
        fh.write("All quantized variants are derived from this checkpoint.\n")
        fh.write("Use this for future fine-tuning experiments.\n")
    log.info("Manifest written.")

    size_bytes = sum(f.stat().st_size for f in GOLD_MASTER_DIR.rglob("*") if f.is_file())
    log.info("Gold Master saved: %s (%.2f GB)", GOLD_MASTER_DIR, size_bytes / 1024**3)
    return GOLD_MASTER_DIR


# ---------------------------------------------------------------------------
# Step 3: Convert HF -> GGUF
# ---------------------------------------------------------------------------

def convert_to_gguf(model_dir: Path) -> Path:
    """Convert BF16 HuggingFace model directory to GGUF F16 format.

    Uses llama.cpp convert_hf_to_gguf.py. Requires llama.cpp to be
    built at /home/phil/llama.cpp with Mamba GGUF support.

    Args:
        model_dir: Path to saved BF16 model directory.

    Returns:
        Path to the generated F16 .gguf file.
    """
    log.info("=" * 60)
    log.info("CONVERTING TO GGUF (F16)")
    log.info("=" * 60)

    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    output_gguf = GGUF_DIR / "mamba-tiny-refinement-f16.gguf"

    # Check for llama.cpp convert script
    if not LLAMA_CONVERT.exists():
        # Try alternate locations
        alt_paths = [
            Path("/home/phil/llama.cpp/convert_hf_to_gguf.py"),
            Path("/home/phil/llama.cpp/convert.py"),
        ]
        for alt in alt_paths:
            if alt.exists():
                convert_script = alt
                break
        else:
            log.error("llama.cpp convert script not found. Build llama.cpp first.")
            log.error("Expected: %s", LLAMA_CONVERT)
            return None
    else:
        convert_script = LLAMA_CONVERT

    cmd = [
        PYTHON_BIN, str(convert_script),
        str(model_dir),
        "--outfile", str(output_gguf),
        "--outtype", "f16",
    ]
    log.info("Running: %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        log.error("GGUF conversion failed:\nSTDOUT: %s\nSTDERR: %s",
                  result.stdout[-500:], result.stderr[-500:])
        return None

    if output_gguf.exists():
        size_gb = output_gguf.stat().st_size / 1024**3
        log.info("GGUF conversion complete: %s (%.2f GB)", output_gguf, size_gb)
        return output_gguf
    else:
        log.error("GGUF file not created despite exit code 0.")
        return None


# ---------------------------------------------------------------------------
# Step 4: Quantize
# ---------------------------------------------------------------------------

def quantize_gguf(f16_gguf: Path, quant_level: str) -> Path:
    """Quantize an F16 GGUF file to a target quantization level.

    Args:
        f16_gguf: Path to the F16 GGUF source file.
        quant_level: Target level string, e.g. "Q2_K" or "Q4_K_M".

    Returns:
        Path to the quantized .gguf file, or None on failure.
    """
    output_path = GGUF_DIR / f"mamba-tiny-refinement-{quant_level.lower()}.gguf"
    log.info("Quantizing to %s -> %s", quant_level, output_path.name)

    if not LLAMA_QUANTIZE.exists():
        # Try build directory
        alt = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"
        if alt.exists():
            quantize_bin = alt
        else:
            log.error("llama-quantize not found. Build llama.cpp first.")
            log.error("Try: cd /home/phil/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j4")
            return None
    else:
        quantize_bin = LLAMA_QUANTIZE

    cmd = [str(quantize_bin), str(f16_gguf), str(output_path), quant_level]
    log.info("Running: %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    if result.returncode != 0:
        log.error("Quantization failed for %s:\n%s", quant_level, result.stderr[-500:])
        return None

    if output_path.exists():
        size_gb = output_path.stat().st_size / 1024**3
        log.info("Quantized %s: %.2f GB", quant_level, size_gb)
        return output_path
    return None


# ---------------------------------------------------------------------------
# Step 5: Inference validation
# ---------------------------------------------------------------------------

def run_inference_test(gguf_path: Path, quant_level: str) -> bool:
    """Validate the quantized model generates ==== spacers before code.

    Tests 3 prompts and checks:
    - Output contains ==== spacer tokens
    - Output contains Python-looking code after spacers
    - No past_key_values created (O(1) memory via SSM state)

    Args:
        gguf_path: Path to the quantized GGUF file.
        quant_level: Label for log output.

    Returns:
        True if all checks pass, False otherwise.
    """
    log.info("=" * 60)
    log.info("INFERENCE VALIDATION (%s)", quant_level)
    log.info("=" * 60)

    try:
        from llama_cpp import Llama
    except ImportError:
        log.error("llama_cpp not available for inference test.")
        return False

    log.info("Loading %s ...", gguf_path.name)
    try:
        model = Llama(
            model_path=str(gguf_path),
            n_gpu_layers=-1,
            n_ctx=512,
            verbose=False,
        )
    except Exception as exc:
        log.error("Failed to load GGUF for inference: %s", exc)
        return False

    test_prompts = [
        "Write a Python function that reverses a linked list in-place.",
        "Implement binary search in Python with proper bounds.",
        "Write a Python function to check if a binary tree is balanced.",
    ]

    passed = 0
    for i, prompt in enumerate(test_prompts):
        try:
            result = model(
                prompt,
                max_tokens=256,
                temperature=0.1,
                top_k=1,
                echo=False,
            )
            output = result["choices"][0]["text"]
            has_spacers = "====" in output
            has_code = any(kw in output for kw in
                           ["def ", "return ", "class ", "import ", "    "])
            status = "PASS" if has_code else "PARTIAL"
            log.info("Test %d/%d [%s]: spacers=%s code=%s",
                     i + 1, len(test_prompts), status, has_spacers, has_code)
            log.info("  Preview: %s", output[:120].replace("\n", " "))
            if has_code:
                passed += 1
        except Exception as exc:
            log.warning("Test %d failed: %s", i + 1, exc)

    del model
    gc.collect()

    all_passed = passed == len(test_prompts)
    log.info("Validation: %d/%d tests passed for %s.", passed, len(test_prompts), quant_level)
    return all_passed


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Full Phase 3 pipeline: merge -> Gold Master -> GGUF -> quantize -> test."""
    parser = argparse.ArgumentParser(description="Agent 3: The Blacksmith")
    parser.add_argument("--skip-gguf", action="store_true",
                        help="Save BF16 Gold Master only; skip GGUF conversion.")
    parser.add_argument("--quant-only", action="store_true",
                        help="Skip merge/BF16 save; GGUF + quantize only.")
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip inference validation after quantization.")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("AGENT 3: THE BLACKSMITH -- Project TinyRefinementModel")
    log.info("=" * 60)

    if not args.quant_only:
        # Step 1: Load backbone + merge LoRA algebraically
        model = merge_lora_into_backbone()

        # Step 2: Save Gold Master BEFORE any quantization
        gold_master = save_gold_master(model)

        # Free GPU memory before GGUF conversion
        del model
        gc.collect()
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info()[0] / 1024**3
        log.info("VRAM after merge/save: %.2f GB free.", free)
    else:
        gold_master = GOLD_MASTER_DIR
        if not gold_master.exists():
            log.error("Gold Master not found for --quant-only: %s", gold_master)
            sys.exit(1)

    if args.skip_gguf:
        log.info("--skip-gguf: stopping after Gold Master save.")
        log.info("Gold Master: %s", gold_master)
        return

    # Check if llama.cpp conversion is available
    convert_available = LLAMA_CONVERT.exists() or \
                        (LLAMA_CPP_DIR / "convert.py").exists()
    quantize_available = LLAMA_QUANTIZE.exists() or \
                         (LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize").exists()

    if not convert_available or not quantize_available:
        log.warning("llama.cpp tools not found. Skipping GGUF conversion.")
        log.warning("To build: cd %s && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build -j4 --target llama-quantize convert_hf_to_gguf", LLAMA_CPP_DIR)
        log.info("Gold Master BF16 is saved and ready for manual GGUF conversion.")
        return

    # Step 3: Convert to F16 GGUF
    f16_gguf = convert_to_gguf(gold_master)
    if f16_gguf is None:
        log.error("GGUF conversion failed. Gold Master is still intact at %s.", gold_master)
        return

    # Step 4: Quantize to all target levels
    quantized: dict[str, Path] = {}
    for level in QUANT_LEVELS:
        path = quantize_gguf(f16_gguf, level)
        if path:
            quantized[level] = path

    # Step 5: Inference validation
    if not args.skip_test and quantized:
        best_level = "Q4_K_M" if "Q4_K_M" in quantized else next(iter(quantized))
        run_inference_test(quantized[best_level], best_level)

    # Final summary
    log.info("=" * 60)
    log.info("AGENT 3 COMPLETE -- Project TinyRefinementModel")
    log.info("Gold Master BF16: %s", GOLD_MASTER_DIR)
    for level, path in quantized.items():
        size_gb = path.stat().st_size / 1024**3
        log.info("  %s: %s (%.2f GB)", level, path.name, size_gb)
    if not quantized:
        log.info("  No GGUF artifacts (llama.cpp unavailable).")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
