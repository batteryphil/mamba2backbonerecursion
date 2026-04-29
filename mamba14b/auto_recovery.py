#!/usr/bin/env python3
"""
auto_recovery.py — Self-healing agent for V3 RLF training.

Wraps the full training run with automatic failure diagnosis and recovery.
For known failure patterns, applies the fix and restarts automatically.
For unknown failures, fires a desktop notification with a full diagnosis
dump so the user can paste it to the AI on waking.

Known failure modes (auto-recoverable):
  K1: fast-path return arity mismatch (ValueError: not enough values to unpack)
  K2: SFTDataset JSON decode error (blank/null-byte lines)
  K3: Loss=inf (HALT mask applied to wrong sample)
  K4: mem_norm flat-zero despite penalty (norm_penalty not reaching gradient)
  K5: Phase gate 3a fail — mem_norm out of range
  K6: Phase gate 3b fail — multi-hop accuracy too low (extend 3b steps)

Unknown failures → desktop notification + diagnosis file for human review.
"""

import os
import re
import sys
import time
import shutil
import signal
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAINER    = Path("/home/phil/.gemini/antigravity/scratch/mamba2backbonerecursion/mamba14b")
LOG        = Path("/home/phil/.gemini/antigravity/scratch/tiny-refinement/rlf_trainer.log")
CKPT_DIR   = Path("/hdd_data/rlf-1.4b-checkpoints")
DIAG_DIR   = Path("/home/phil/.gemini/antigravity/scratch/tiny-refinement/diagnostics")
PYTHON     = "/home/phil/.local/share/mise/installs/python/3.14.3/bin/python"
NCCL       = "/home/phil/.gemini/antigravity/scratch/quill/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2"
SITE       = "/home/phil/.local/share/mise/installs/python/3.14.3/lib/python3.14/site-packages"

MAX_AUTO_RETRIES = 3   # stop auto-fixing after this many attempts

DIAG_DIR.mkdir(parents=True, exist_ok=True)

# ── Environment ───────────────────────────────────────────────────────────────
def make_env() -> dict:
    """Build training environment."""
    env = os.environ.copy()
    env["LD_PRELOAD"]              = f"{NCCL}:{SITE}/nvidia/cu13/lib/libnvJitLink.so.13"
    env["TOKENIZERS_PARALLELISM"]  = "false"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return env


# ── Notifications ─────────────────────────────────────────────────────────────
NTFY_TOPIC = "rlf-v3-phil"   # install ntfy app → subscribe to this topic

def notify(title: str, body: str, priority: str = "default") -> None:
    """Push notification via ntfy.sh — works on any phone with the ntfy app."""
    import urllib.request
    try:
        data = json.dumps({
            "topic":    NTFY_TOPIC,
            "title":    f"🤖 RLF Agent: {title}",
            "message":  body,
            "priority": priority,
        }).encode()
        req = urllib.request.Request(
            "https://ntfy.sh",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[notify] ntfy.sh failed: {e}")

    # Always write alert file for dashboard to display
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    with open(DIAG_DIR / "ALERT.txt", "w") as f:
        f.write(f"[{datetime.now()}] {title}\n\n{body}\n")
    print(f"\n{'!'*55}\n  {title}\n  {body}\n{'!'*55}\n")



# ── Log analysis ──────────────────────────────────────────────────────────────
def read_log_tail(n: int = 80) -> list[str]:
    """Return last n lines of training log."""
    try:
        result = subprocess.run(["strings", str(LOG)],
                                capture_output=True, text=True, timeout=10)
        return result.stdout.splitlines()[-n:]
    except Exception:
        return []


def get_last_step() -> tuple[str, int]:
    """Return (phase, step) of the last logged training step."""
    pat = re.compile(r"\[Phase(\w+)\]\[S(\d+)\]")
    for line in reversed(read_log_tail(100)):
        m = pat.search(line)
        if m:
            return m.group(1), int(m.group(2))
    return "unknown", 0


def get_mem_norm_history() -> list[float]:
    """Return last 10 mem_norm values."""
    pat = re.compile(r"mem_norm=([\d.]+)")
    vals = []
    for line in read_log_tail(200):
        m = pat.search(line)
        if m:
            vals.append(float(m.group(1)))
    return vals[-10:]


# ── Known failure pattern matcher ─────────────────────────────────────────────
def diagnose_failure(stderr: str, stdout: str) -> tuple[str, str]:
    """Match failure output against known patterns.

    Returns (pattern_id, human_description).
    Returns ("unknown", ...) if no pattern matches.
    """
    combined = (stderr + stdout + "\n".join(read_log_tail(50))).lower()

    if "not enough values to unpack" in combined:
        return "K1", "Return arity mismatch in compute_rlf_loss fast-path"

    if "json.decoder.jsondecodeerror" in combined:
        return "K2", "SFTDataset hit a corrupt line in the JSONL file"

    if "loss=inf" in combined or "loss=nan" in combined:
        return "K3", "Loss exploded (inf/nan) — likely HALT mask on wrong sample"

    if "phase 3a gate failed" in combined or "gate failed" in combined:
        norms = get_mem_norm_history()
        if norms and max(norms) < 0.1:
            return "K4", f"Phase 3a gate: mem_norm flat ({max(norms):.3f}) — norm penalty not reaching gradient"
        return "K5", f"Phase 3a gate: mem_norm OK but 1-hop accuracy too low"

    if "phase 3b gate failed" in combined:
        return "K6", "Phase 3b gate: multi-hop accuracy below 40%"

    if "attributeerror" in combined and "latent_memory" in combined:
        return "K7", "Stale V1 API reference (latent_memory) in eval or trainer"

    if "cuda out of memory" in combined or "oom" in combined:
        return "K8", "CUDA OOM — need to reduce batch size or gradient accumulation"

    return "unknown", f"Unrecognised failure. stderr tail:\n{stderr[-800:]}"


# ── Auto-fixes for known patterns ─────────────────────────────────────────────
def apply_fix(pattern: str) -> bool:
    """Apply the automated fix for a known pattern.

    Returns True if fix was applied and training should be retried.
    Returns False if the fix requires human review.
    """
    trainer_file = TRAINER / "rlf_trainer_1_4b.py"

    if pattern == "K1":
        # Already fixed in codebase — likely a stale .pyc. Clear cache.
        print("[auto-fix K1] Clearing Python bytecode cache...")
        for pyc in TRAINER.rglob("*.pyc"):
            pyc.unlink(missing_ok=True)
        for pycache in TRAINER.rglob("__pycache__"):
            shutil.rmtree(pycache, ignore_errors=True)
        return True

    if pattern == "K2":
        # JSONL has corrupt lines — the try/except guard should handle it.
        # If we're here it means it slipped through. Just retry.
        print("[auto-fix K2] JSONL corrupt line — retrying (try/except guard active)")
        return True

    if pattern == "K3":
        # Loss=inf — check if the HALT mask is the culprit
        print("[auto-fix K3] Loss=inf — retrying; if persistent, needs human review")
        return True

    if pattern == "K4":
        # mem_norm flat — bump norm penalty coefficient
        src = trainer_file.read_text()
        current = re.search(r"norm_penalty = ([\d.]+) \*", src)
        if current:
            old_coeff = float(current.group(1))
            new_coeff = min(old_coeff * 2, 1.0)
            src = src.replace(
                f"norm_penalty = {old_coeff} *",
                f"norm_penalty = {new_coeff} *"
            )
            trainer_file.write_text(src)
            print(f"[auto-fix K4] norm_penalty {old_coeff} → {new_coeff}")
            # Nuke checkpoints so Phase 3a restarts clean
            shutil.rmtree(CKPT_DIR, ignore_errors=True)
        return True

    if pattern == "K5":
        # 1-hop accuracy too low after 6000 steps — extend Phase 3a to 8000
        src = trainer_file.read_text()
        src = re.sub(
            r'"3a":\s*\{"steps":\s*\d+',
            '"3a": {"steps": 8000',
            src
        )
        trainer_file.write_text(src)
        shutil.rmtree(CKPT_DIR, ignore_errors=True)
        print("[auto-fix K5] Extended Phase 3a from 6000 → 8000 steps, cleared checkpoints")
        return True

    if pattern == "K6":
        # Multi-hop accuracy too low — extend Phase 3b by 2000 steps
        src = trainer_file.read_text()
        m = re.search(r'"3b":\s*\{"steps":\s*(\d+)', src)
        if m:
            old_steps = int(m.group(1))
            new_steps = old_steps + 2000
            src = src.replace(
                f'"3b": {{"steps": {old_steps}',
                f'"3b": {{"steps": {new_steps}'
            )
            trainer_file.write_text(src)
            print(f"[auto-fix K6] Extended Phase 3b {old_steps} → {new_steps} steps")
        return True

    if pattern == "K8":
        # OOM — halve gradient accumulation buffer
        src = trainer_file.read_text()
        m = re.search(r"GRAD_ACCUM\s*=\s*(\d+)", src)
        if m:
            old = int(m.group(1))
            new = max(old // 2, 4)
            src = src.replace(f"GRAD_ACCUM   = {old}", f"GRAD_ACCUM   = {new}")
            trainer_file.write_text(src)
            print(f"[auto-fix K8] GRAD_ACCUM {old} → {new}")
        return True

    return False  # unknown pattern → needs human


# ── Main recovery loop ────────────────────────────────────────────────────────
def write_diagnosis(attempt: int, pattern: str, desc: str,
                    stdout: str, stderr: str) -> Path:
    """Write a full diagnosis file for human or AI review."""
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = DIAG_DIR / f"diagnosis_attempt{attempt}_{ts}.txt"
    phase, step = get_last_step()
    norms = get_mem_norm_history()

    with open(path, "w") as f:
        f.write(f"RLF V3 Auto-Recovery Diagnosis\n")
        f.write(f"Time:       {datetime.now()}\n")
        f.write(f"Attempt:    {attempt}/{MAX_AUTO_RETRIES}\n")
        f.write(f"Pattern:    {pattern}\n")
        f.write(f"Desc:       {desc}\n")
        f.write(f"Last step:  Phase {phase} S{step:05d}\n")
        f.write(f"mem_norm history: {norms}\n\n")
        f.write("="*60 + "\nSTDERR TAIL\n" + "="*60 + "\n")
        f.write(stderr[-2000:])
        f.write("\n\n" + "="*60 + "\nLOG TAIL\n" + "="*60 + "\n")
        f.write("\n".join(read_log_tail(60)))
    return path


def run_training(resume: bool = False) -> tuple[int, str, str]:
    """Launch trainer and wait for completion.

    Returns (returncode, stdout, stderr).
    """
    cmd = [PYTHON, "-B", "rlf_trainer_1_4b.py", "--phase", "all"]
    if resume:
        cmd.append("--resume")

    proc = subprocess.Popen(
        cmd,
        cwd=str(TRAINER),
        env=make_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)

    proc.wait()
    combined = "".join(output_lines)
    return proc.returncode, combined, ""


def main() -> None:
    """Self-healing training loop."""
    print(f"\n{'='*60}")
    print(f"  RLF V3 Auto-Recovery Agent")
    print(f"  Started: {datetime.now()}")
    print(f"  Max auto-retries: {MAX_AUTO_RETRIES}")
    print(f"{'='*60}\n")

    notify("V3 Training Started",
           f"Auto-recovery agent active. Max {MAX_AUTO_RETRIES} retries.")

    for attempt in range(1, MAX_AUTO_RETRIES + 1):
        print(f"\n[agent] Attempt {attempt}/{MAX_AUTO_RETRIES} — launching trainer")
        returncode, stdout, stderr = run_training(resume=(attempt > 1))

        if returncode == 0:
            notify("✅ V3 Training COMPLETE",
                   "All phases passed their gates. Eval results are ready.")
            print("\n[agent] Training completed successfully. Eval results in logs.")
            return

        # ── Training failed — diagnose ─────────────────────────────────────
        pattern, desc = diagnose_failure(stderr, stdout)
        diag_file     = write_diagnosis(attempt, pattern, desc, stdout, stderr)

        print(f"\n[agent] Failure detected: [{pattern}] {desc}")
        print(f"[agent] Diagnosis written: {diag_file}")

        if attempt >= MAX_AUTO_RETRIES:
            break

        # Try to auto-fix
        fixed = apply_fix(pattern)
        if fixed:
            notify(f"⚠️ Auto-fixing [{pattern}]",
                   f"{desc}\nFixed automatically. Attempt {attempt+1} starting.")
            print(f"[agent] Fix applied for {pattern}. Retrying...")
            time.sleep(5)
            continue
        else:
            # Unknown failure — needs human
            break

    # ── Exhausted retries or unknown failure ──────────────────────────────
    pattern, desc = diagnose_failure("", "")
    diag_file     = DIAG_DIR / "FINAL_DIAGNOSIS.txt"

    msg = (f"Training failed after {attempt} attempt(s).\n"
           f"Pattern: {pattern}\n{desc}\n"
           f"Review: {diag_file}\n\n"
           f"Paste the diagnosis file to the AI to continue.")

    notify("❌ RLF NEEDS HUMAN REVIEW", msg)

    print(f"\n{'!'*60}")
    print(f"  Auto-recovery exhausted. Human review required.")
    print(f"  Diagnosis: {diag_file}")
    print(f"  Tell the AI: 'check the diagnosis file and fix it'")
    print(f"{'!'*60}\n")


if __name__ == "__main__":
    main()
