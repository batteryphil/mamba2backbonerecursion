#!/usr/bin/env python3
"""
phase1_distiller.py -- Agent 1: The Distiller
Project TinyRefinementModel

Loads Qwen2.5-Coder-7B-Instruct-Q4_K_M via llama_cpp (GPU-accelerated),
generates Chain-of-Thought (CoT) trajectories on code/syntax problems,
strips English reasoning, counts tokens, and replaces them with an
equivalent number of ==== spacer tokens.

Output: training_data.jsonl (3,000 samples)

Usage:
    ./run_env.sh phase1_distiller.py
    ./run_env.sh phase1_distiller.py --dry-run
    ./run_env.sh phase1_distiller.py --resume
"""

import argparse
import gc
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKSPACE = Path(__file__).parent
MODEL_DIR = WORKSPACE / "models"
OUTPUT_FILE = WORKSPACE / "training_data.jsonl"
PROMPTS_FILE = WORKSPACE / "phase1_prompts.jsonl"
LOG_FILE = WORKSPACE / "phase1_distiller.log"

TEACHER_REPO = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
TEACHER_FILENAME = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
TEACHER_LOCAL = MODEL_DIR / TEACHER_FILENAME

N_GPU_LAYERS = -1      # All layers to GPU (~4.5 GB VRAM)
N_CTX = 4096
MAX_TOKENS = 1536
TEMPERATURE = 0.35     # Code mode: 0.3-0.5 per project rules
TOP_K = 40
TOP_P = 0.95
REPEAT_PENALTY = 1.1

TARGET_SAMPLES = 3000
SAMPLES_PER_PROMPT = 60   # 50 prompts x 60 modifiers = 3000
SPACER_TOKEN = "===="
VRAM_FREE_THRESHOLD_GB = 7.0

MODIFIERS = [
    "",
    " Make it iterative instead of recursive.",
    " Add comprehensive error handling and input validation.",
    " Optimize it for minimum memory usage.",
    " Add type hints for all arguments and return values.",
    " Make it work with very large inputs (N > 10^6).",
    " Write it in a purely functional style with no side effects.",
    " Add a generator version that yields results lazily.",
    " Write the solution using only the standard library.",
    " Include a complete test suite using unittest.",
    " Optimize it for cache efficiency.",
    " Handle all Unicode edge cases.",
    " Make the function thread-safe.",
    " Add logging at DEBUG level for each major step.",
    " Write a version that works in O(1) extra space.",
    " Implement it using an explicit stack instead of recursion.",
    " Include complexity analysis in the docstring.",
    " Make it configurable via a dataclass config object.",
    " Write a version that can be interrupted and resumed.",
    " Implement it without any loops -- use only recursion.",
    " Add a visualization function that prints the algorithm state.",
    " Write an async version using asyncio.",
    " Make it serializable -- state should be JSON-serializable.",
    " Add input fuzzing tests inside the docstring.",
    " Optimize for CPython interpreter internals.",
    " Use structural pattern matching (match/case) where appropriate.",
    " Implement it as a coroutine-based state machine.",
    " Make it numerically stable for floating point edge cases.",
    " Write a version that maintains a persistent cache.",
    " Use slots and __weakref__ to minimize memory overhead.",
    " Implement it as a context manager where applicable.",
    " Add a timeout mechanism that raises TimeoutError after N seconds.",
    " Write it to be picklable for multiprocessing use.",
    " Add support for numpy arrays as input.",
    " Implement it using bitwise operations wherever possible.",
    " Write it with explicit tail-call optimization via trampolining.",
    " Include a benchmark comparing to the naive approach.",
    " Write a streaming version for inputs larger than RAM.",
    " Implement with dependency injection for all I/O operations.",
    " Add a verbose mode explaining each decision.",
    " Return intermediate results as a Python generator.",
    " Write it compatible with both Python 3.10 and 3.12.",
    " Use dataclasses with __post_init__ validation.",
    " Implement with pluggable comparison functions (key= style).",
    " Add retry logic with exponential backoff.",
    " Write the core logic as a pure function, then wrap with I/O.",
    " Implement as a class with __iter__ support.",
    " Add circuit breaker logic.",
    " Use __init_subclass__ hooks for automatic registration.",
    " Write it with full Protocol/ABC compliance.",
    " Add a dry-run mode that validates inputs without executing.",
    " Write a version that emits structured JSON logs.",
    " Implement it with a configurable comparison predicate.",
    " Use contextlib for all resource management.",
    " Add property-based invariant checks inline as assertions.",
    " Write a version using only immutable data structures.",
    " Implement it with explicit memory pooling.",
    " Add instrumentation for profiling (cProfile-compatible).",
    " Write it using the builder pattern for complex configuration.",
    " Implement with short-circuit evaluation optimization.",
]


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
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrainingSample:
    """A single distilled training sample for the student model."""

    prompt: str
    cot_token_count: int
    spacer_sequence: str
    answer: str
    full_target: str
    source_id: int
    modifier_idx: int
    teacher_model: str = TEACHER_FILENAME
    generation_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def download_teacher_model() -> Path:
    """Download Qwen2.5-Coder-7B-Instruct Q4_K_M GGUF if not already cached.

    Returns:
        Path to the local GGUF file.
    """
    if TEACHER_LOCAL.exists():
        size_gb = TEACHER_LOCAL.stat().st_size / 1024 ** 3
        log.info("Teacher already present: %s (%.2f GB)", TEACHER_LOCAL, size_gb)
        return TEACHER_LOCAL

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s from %s ...", TEACHER_FILENAME, TEACHER_REPO)
    log.info("Expected size: ~4.7 GB. This is a one-time download.")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    local_path = hf_hub_download(
        repo_id=TEACHER_REPO,
        filename=TEACHER_FILENAME,
        local_dir=str(MODEL_DIR),
        resume_download=True,
    )
    log.info("Download complete: %s", local_path)
    return Path(local_path)


# ---------------------------------------------------------------------------
# Teacher loader
# ---------------------------------------------------------------------------

def load_teacher(model_path: Path):
    """Load Qwen2.5-Coder GGUF via llama_cpp with full GPU offload.

    Args:
        model_path: Path to the .gguf model file.

    Returns:
        A loaded Llama instance ready for inference.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        log.error("llama_cpp not installed.")
        sys.exit(1)

    log.info("Loading teacher: %s | gpu_layers=%d | ctx=%d",
             model_path.name, N_GPU_LAYERS, N_CTX)
    model = Llama(
        model_path=str(model_path),
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        n_threads=4,
        n_batch=512,
        flash_attn=True,
        verbose=False,
    )
    log.info("Teacher loaded successfully.")
    return model


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an elite Python programmer. For every coding problem, follow this "
    "EXACT format:\n\n"
    "<think>\n"
    "Your detailed step-by-step reasoning. Work through: algorithm selection, "
    "edge case analysis, data structure choices, complexity estimation, and "
    "implementation decisions. Reason thoroughly -- complex problems need more "
    "reasoning.\n"
    "</think>\n\n"
    "```python\n"
    "# Your complete, correct, production-quality Python solution here\n"
    "```\n\n"
    "CRITICAL: You MUST use <think></think> tags. Do NOT skip or abbreviate reasoning."
)


def build_chat_prompt(base_prompt: str, modifier: str) -> str:
    """Build a Qwen2.5 ChatML-formatted prompt.

    Args:
        base_prompt: Base coding problem description.
        modifier: Variation constraint appended to the problem.

    Returns:
        Full ChatML prompt string.
    """
    problem = f"{base_prompt}{modifier}"
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

def extract_cot_and_answer(
    raw_output: str,
) -> tuple[Optional[str], Optional[str]]:
    """Parse teacher output into (cot_reasoning, final_answer).

    Tries three strategies in priority order:
    1. Explicit <think>...</think> delimiters
    2. Heuristic split on the first ```python code block
    3. Last-paragraph fallback

    Args:
        raw_output: Raw string returned by the teacher model.

    Returns:
        Tuple (cot_text, answer_text), either may be None on failure.
    """
    # Strategy 1: Explicit think tags (ideal path)
    think_match = re.search(
        r"<think>(.*?)</think>(.*)",
        raw_output,
        re.DOTALL | re.IGNORECASE,
    )
    if think_match:
        cot = think_match.group(1).strip()
        answer_raw = think_match.group(2).strip()
        code_match = re.search(r"```(?:python)?\n?(.*?)```", answer_raw, re.DOTALL)
        answer = code_match.group(1).strip() if code_match else answer_raw
        if cot and answer and len(cot) > 30:
            return cot, answer

    # Strategy 2: Split at first code fence
    code_match = re.search(r"```(?:python)?\n?(.*?)```", raw_output, re.DOTALL)
    if code_match:
        code_start = raw_output.find(code_match.group(0))
        cot = raw_output[:code_start].strip()
        answer = code_match.group(1).strip()
        if cot and answer and len(cot) > 30:
            return cot, answer

    # Strategy 3: Last paragraph heuristic
    parts = raw_output.strip().split("\n\n")
    if len(parts) >= 2:
        answer = parts[-1].strip()
        cot = "\n\n".join(parts[:-1]).strip()
        if len(cot) > 30:
            return cot, answer

    return None, None


# ---------------------------------------------------------------------------
# Spacer token builder
# ---------------------------------------------------------------------------

def build_spacer_sequence(
    model, cot_text: str
) -> tuple[int, str]:
    """Count CoT tokens using the teacher tokenizer and build spacer string.

    Token count is clamped to [8, 512] to prevent degenerate samples.

    Args:
        model: Loaded Llama instance (used for its tokenizer).
        cot_text: The extracted chain-of-thought text.

    Returns:
        Tuple of (token_count, spacer_string).
    """
    raw_tokens = model.tokenize(cot_text.encode("utf-8"), add_bos=False)
    n = max(8, min(len(raw_tokens), 512))
    spacer_str = " ".join([SPACER_TOKEN] * n)
    return n, spacer_str


# ---------------------------------------------------------------------------
# Single sample generator
# ---------------------------------------------------------------------------

def generate_sample(
    model,
    prompt_str: str,
    base_problem: str,
    modifier: str,
    source_id: int,
    modifier_idx: int,
    dry_run: bool = False,
) -> Optional[TrainingSample]:
    """Run teacher inference and build one TrainingSample.

    Args:
        model: Loaded Llama instance.
        prompt_str: Full formatted ChatML prompt.
        base_problem: Original base problem text.
        modifier: Modifier string appended to problem.
        source_id: Seed prompt ID (0-49).
        modifier_idx: Modifier index used (0-59).
        dry_run: Skip actual inference if True.

    Returns:
        TrainingSample on success, None on parse failure.
    """
    if dry_run:
        dummy_cot = "Let me analyse this problem. " * 15
        dummy_answer = "def solution():\n    pass\n"
        n_tok, spacer = build_spacer_sequence(model, dummy_cot)
        return TrainingSample(
            prompt=f"{base_problem}{modifier}",
            cot_token_count=n_tok,
            spacer_sequence=spacer,
            answer=dummy_answer,
            full_target=f"{spacer}\n{dummy_answer}",
            source_id=source_id,
            modifier_idx=modifier_idx,
            generation_time_s=0.0,
        )

    t0 = time.time()
    result = model(
        prompt_str,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repeat_penalty=REPEAT_PENALTY,
        stop=["<|im_end|>", "<|im_start|>"],
        echo=False,
    )
    elapsed = time.time() - t0
    raw_output = result["choices"][0]["text"]

    cot_text, answer = extract_cot_and_answer(raw_output)
    if cot_text is None or answer is None:
        log.warning(
            "Parse failed for source_id=%d modifier=%d -- skipping.",
            source_id, modifier_idx
        )
        return None

    n_tok, spacer = build_spacer_sequence(model, cot_text)
    full_target = f"{spacer}\n{answer}"

    return TrainingSample(
        prompt=f"{base_problem}{modifier}",
        cot_token_count=n_tok,
        spacer_sequence=spacer,
        answer=answer,
        full_target=full_target,
        source_id=source_id,
        modifier_idx=modifier_idx,
        generation_time_s=round(elapsed, 2),
    )


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_completed_ids(output_file: Path) -> set[tuple[int, int]]:
    """Load set of (source_id, modifier_idx) pairs already in output file.

    Args:
        output_file: Path to training_data.jsonl.

    Returns:
        Set of completed (source_id, modifier_idx) tuples.
    """
    completed: set[tuple[int, int]] = set()
    if not output_file.exists():
        return completed
    with output_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
                completed.add((obj["source_id"], obj["modifier_idx"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


# ---------------------------------------------------------------------------
# VRAM teardown assertion
# ---------------------------------------------------------------------------

def assert_vram_cleared(threshold_gb: float = VRAM_FREE_THRESHOLD_GB) -> None:
    """Assert that VRAM is cleared below threshold after teacher teardown.

    Args:
        threshold_gb: Minimum free VRAM required (default 7.0 GB).

    Raises:
        RuntimeError: If free VRAM is below threshold.
    """
    try:
        import torch
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / 1024 ** 3
        total_gb = total_bytes / 1024 ** 3
        log.info("VRAM after teardown: %.2f GB free / %.2f GB total", free_gb, total_gb)
        if free_gb < threshold_gb:
            raise RuntimeError(
                f"VRAM not cleared! Only {free_gb:.2f} GB free, need {threshold_gb:.2f} GB. "
                "Run vram_clear.sh before Phase 2."
            )
        log.info("VRAM assertion passed: %.2f GB free >= %.2f GB threshold.", free_gb, threshold_gb)
    except ImportError:
        log.warning("torch not available for VRAM check -- skipping assertion.")


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

def validate_dataset(output_file: Path) -> bool:
    """Spot-check output JSONL for correctness.

    Validates:
    - File exists and has expected number of records
    - Random 10 samples have matching token count vs spacer length
    - No empty prompts or answers

    Args:
        output_file: Path to training_data.jsonl.

    Returns:
        True if validation passes, False otherwise.
    """
    import random as rng
    if not output_file.exists():
        log.error("Output file not found: %s", output_file)
        return False

    records = []
    with output_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    total = len(records)
    log.info("Dataset validation: %d records found.", total)
    if total < TARGET_SAMPLES:
        log.error("Too few records: %d < %d", total, TARGET_SAMPLES)
        return False

    # Spot-check 10 random samples
    spot = rng.sample(records, min(10, total))
    for rec in spot:
        spacer_count = len(rec["spacer_sequence"].split())
        tok_count = rec["cot_token_count"]
        if spacer_count != tok_count:
            log.error(
                "Mismatch source_id=%d: spacer_count=%d vs cot_token_count=%d",
                rec["source_id"], spacer_count, tok_count
            )
            return False
        if not rec["prompt"] or not rec["answer"]:
            log.error("Empty prompt or answer in source_id=%d", rec["source_id"])
            return False

    log.info("Dataset validation PASSED: %d records, spot-checks OK.", total)
    return True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main distillation pipeline: download -> load -> generate -> teardown."""
    parser = argparse.ArgumentParser(description="Agent 1: The Distiller")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip actual inference; write dummy samples to verify pipeline.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file, skipping completed samples.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit total samples generated (for testing).",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("AGENT 1: THE DISTILLER -- Project TinyRefinementModel")
    log.info("=" * 60)
    log.info("Target: %d samples | Dry-run: %s | Resume: %s",
             TARGET_SAMPLES, args.dry_run, args.resume)

    # ------------------------------------------------------------------
    # Step 1: Download teacher model
    # ------------------------------------------------------------------
    if not args.dry_run:
        model_path = download_teacher_model()
    else:
        model_path = TEACHER_LOCAL
        if not model_path.exists():
            log.info("Dry-run: teacher model not present, using dummy path.")
            model_path = Path("/dev/null")

    # ------------------------------------------------------------------
    # Step 2: Load seed prompts
    # ------------------------------------------------------------------
    if not PROMPTS_FILE.exists():
        log.error("Prompts file not found: %s", PROMPTS_FILE)
        sys.exit(1)

    seed_prompts: list[dict] = []
    with PROMPTS_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                seed_prompts.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    log.info("Loaded %d seed prompts.", len(seed_prompts))

    # ------------------------------------------------------------------
    # Step 3: Load resume state
    # ------------------------------------------------------------------
    completed_ids: set[tuple[int, int]] = set()
    if args.resume:
        completed_ids = load_completed_ids(OUTPUT_FILE)
        log.info("Resuming: %d samples already completed.", len(completed_ids))

    # ------------------------------------------------------------------
    # Step 4: Load teacher model into VRAM
    # ------------------------------------------------------------------
    if not args.dry_run:
        model = load_teacher(model_path)
    else:
        # For dry-run, load the smallest available GGUF for tokenizer
        fallback = Path("/home/phil/mamba-2.8b-latent/mamba-2.8b-Q2_K.gguf")
        if fallback.exists():
            from llama_cpp import Llama
            log.info("Dry-run: loading fallback tokenizer model...")
            model = Llama(
                model_path=str(fallback),
                n_gpu_layers=0,
                n_ctx=512,
                verbose=False,
            )
        else:
            log.error("Dry-run fallback model not found. Cannot tokenize.")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Step 5: Generation loop
    # ------------------------------------------------------------------
    output_file_handle = OUTPUT_FILE.open(
        "a" if args.resume else "w",
        encoding="utf-8",
    )

    total_written = len(completed_ids)
    total_failed = 0
    target = args.limit or TARGET_SAMPLES

    log.info("Starting generation loop. Target: %d samples.", target)

    try:
        for prompt_obj in seed_prompts:
            source_id = prompt_obj["id"] - 1   # 0-indexed
            base_problem = prompt_obj["prompt"]

            for mod_idx, modifier in enumerate(MODIFIERS):
                if total_written >= target:
                    break

                if (source_id, mod_idx) in completed_ids:
                    continue

                prompt_str = build_chat_prompt(base_problem, modifier)

                sample = generate_sample(
                    model=model,
                    prompt_str=prompt_str,
                    base_problem=base_problem,
                    modifier=modifier,
                    source_id=source_id,
                    modifier_idx=mod_idx,
                    dry_run=args.dry_run,
                )

                if sample is None:
                    total_failed += 1
                    continue

                output_file_handle.write(json.dumps(asdict(sample)) + "\n")
                output_file_handle.flush()
                total_written += 1

                if total_written % 50 == 0:
                    log.info(
                        "Progress: %d/%d samples (%.1f%%) | failed: %d | "
                        "last gen: %.1fs",
                        total_written, target,
                        100.0 * total_written / target,
                        total_failed,
                        sample.generation_time_s,
                    )

            if total_written >= target:
                break

    finally:
        output_file_handle.close()

    log.info("Generation complete: %d written, %d failed.", total_written, total_failed)

    # ------------------------------------------------------------------
    # Step 6: Teardown teacher and clear VRAM
    # ------------------------------------------------------------------
    log.info("Tearing down teacher model...")
    del model
    gc.collect()

    try:
        import torch
        torch.cuda.empty_cache()
        log.info("CUDA cache cleared.")
    except ImportError:
        pass

    assert_vram_cleared()

    # ------------------------------------------------------------------
    # Step 7: Validate dataset
    # ------------------------------------------------------------------
    if not args.dry_run:
        valid = validate_dataset(OUTPUT_FILE)
        if not valid:
            log.error("Dataset validation FAILED. Inspect %s.", OUTPUT_FILE)
            sys.exit(1)

    log.info("=" * 60)
    log.info("AGENT 1 COMPLETE: training_data.jsonl ready at %s", OUTPUT_FILE)
    log.info("HANDOFF to Agent 2 (The Sculptor) is now safe.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
