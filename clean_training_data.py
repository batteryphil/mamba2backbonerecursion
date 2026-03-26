"""
clean_training_data.py — 3-Stage Data Cleaning Pipeline
=========================================================
Cleans system2_logic_v1.json for Mamba2-130M training.

Stage 1: Eradicate arithmetic clamping — recalculate all math
Stage 2: Strip letter prior — neutralize A/B/C/D variable bias
Stage 3: Break single-token bottleneck — inject reasoning chains

Produces: system2_logic_v2_clean.json

Usage:
    python clean_training_data.py
    python clean_training_data.py --test
"""

import json
import os
import re
import sys
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1: Arithmetic Recalculation
# ═══════════════════════════════════════════════════════════════════════════════

def extract_arithmetic(text: str) -> Optional[dict]:
    """Extract arithmetic operation from a word problem.

    Supports patterns like:
      - "X has N items. X earns/gets/finds M items."  → addition
      - "X has N items. X spends/loses/gives away M." → subtraction
      - "X has N. Y gives M to X."                    → addition (recipient)

    Args:
        text: the prompt text

    Returns:
        dict with 'a', 'b', 'op', 'result' or None if not arithmetic
    """
    text_lower = text.lower()

    # Pattern 1: Addition via earning/getting/finding/receiving
    m = re.search(
        r'has (\d+)\b.*?'
        r'(?:earns?|gets?|finds?|receives?|gains?|collects?)\s+(\d+)',
        text_lower
    )
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return {"a": a, "b": b, "op": "+", "result": a + b}

    # Pattern 2: Addition via someone giving TO the subject
    # "X has N. Y gives M to X" — X gains M
    m = re.search(
        r'(\w+) has (\d+).*?'
        r'(?:\w+) gives? (\d+) \w+ to \1',
        text_lower
    )
    if m:
        a, b = int(m.group(2)), int(m.group(3))
        return {"a": a, "b": b, "op": "+", "result": a + b}

    # Pattern 3: Subtraction via spending/losing/giving away/eating
    m = re.search(
        r'has (\d+)\b.*?'
        r'(?:spends?|loses?|gives? away|eats?|drops?|breaks?)\s+(\d+)',
        text_lower
    )
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return {"a": a, "b": b, "op": "-", "result": a - b}

    # Pattern 4: Direct "N + M" or "N - M" expressions
    m = re.search(r'(\d+)\s*([+\-*/])\s*(\d+)', text)
    if m:
        a, b = int(m.group(1)), int(m.group(3))
        op = m.group(2)
        if op == '+':
            result = a + b
        elif op == '-':
            result = a - b
        elif op == '*':
            result = a * b
        elif op == '/':
            result = a / b if b != 0 else 0
            result = int(result) if result == int(result) else round(result, 2)
        else:
            return None
        return {"a": a, "b": b, "op": op, "result": result}

    return None


def fix_arithmetic(entry: dict) -> dict:
    """Recalculate arithmetic answers, fixing clamping bugs.

    Args:
        entry: dataset entry dict

    Returns:
        fixed entry (or original if not arithmetic)
    """
    text = entry["text"]
    answer = entry["answer"]

    arith = extract_arithmetic(text)
    if arith is None:
        return entry  # Not an arithmetic problem

    correct = str(arith["result"])

    # Check if the current answer matches
    try:
        current = str(int(answer))
    except ValueError:
        return entry  # Answer isn't numeric, skip

    if current != correct:
        entry = dict(entry)  # Don't mutate original
        # Fix: overwrite the answer
        old_ans = entry["answer"]
        entry["answer"] = correct
        # Also fix in the text field if answer is embedded
        if f"\nAnswer: {old_ans}" in entry["text"]:
            entry["text"] = entry["text"].replace(
                f"\nAnswer: {old_ans}", f"\nAnswer: {correct}"
            )
        entry["_arith_fixed"] = True
        entry["_arith_was"] = old_ans
        entry["_arith_eq"] = f"{arith['a']} {arith['op']} {arith['b']} = {correct}"

    return entry


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2: Strip Letter Prior
# ═══════════════════════════════════════════════════════════════════════════════

# Single-letter variable names that collide with MC options
LETTER_VARS = {"A", "B", "C", "D"}

# Replacement map: A→P, B→Q, C→R, D→S (no MC collision)
LETTER_REMAP = {"A": "P", "B": "Q", "C": "R", "D": "S"}


def strip_letter_prior(entry: dict) -> dict:
    """Rename single-letter variables A/B/C/D to prevent MC heuristic.

    Replaces variable names A→P, B→Q, C→R, D→S in both prompt and answer.
    Only replaces when the letter is used as a standalone variable
    (word boundary on both sides), not inside words.

    Also strips any explicit multiple-choice formatting like "A) value".

    Args:
        entry: dataset entry dict

    Returns:
        cleaned entry
    """
    text = entry["text"]
    answer = entry["answer"]
    modified = False

    # Step 1: Strip explicit MC formatting "A) value", "B. value"
    mc_match = re.search(r'([A-D])\)\s*(\S+)', text)
    if mc_match and answer in LETTER_VARS:
        # Answer is a letter choice — resolve to the actual value
        choices = {}
        for m in re.finditer(r'([A-D])\)\s*(\S+)', text):
            choices[m.group(1)] = m.group(2)
        if answer in choices:
            entry = dict(entry)
            entry["answer"] = choices[answer]
            # Remove MC formatting from text
            entry["text"] = re.sub(r'[A-D]\)\s*\S+\s*', '', entry["text"]).strip()
            modified = True

    # Step 2: Rename standalone A/B/C/D variables to P/Q/R/S
    for letter, replacement in LETTER_REMAP.items():
        # Match standalone variable usage:
        # "A = 5", "Set A", "What is A", "A copies", etc.
        # Use word boundaries to avoid replacing 'A' inside words like 'Alice'
        pattern = rf'\b{letter}\b'

        if re.search(pattern, text):
            # Check it's used as a variable (not a name like "Alice", "Bob")
            # Variable usage: appears near '=', 'is', 'copies', 'equals'
            is_var = bool(re.search(
                rf'\b{letter}\s*=|=\s*{letter}\b|'
                rf'\b{letter}\s+(?:copies|equals|points|is\b)|'
                rf'(?:Set|Let|Define|Given)\s+{letter}\b|'
                rf'What is {letter}\b|{letter}\s+is\?',
                text
            ))
            if is_var:
                entry = dict(entry)
                entry["text"] = re.sub(pattern, replacement, entry["text"])
                if entry["answer"] == letter:
                    entry["answer"] = replacement
                modified = True

    if modified:
        entry["_letter_fixed"] = True

    return entry


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3: Inject Reasoning Chain
# ═══════════════════════════════════════════════════════════════════════════════

def build_reasoning_chain(entry: dict) -> str:
    """Build a multi-token reasoning chain from the prompt.

    Format: [Reasoning] <steps> [Answer] <value> <HALT>

    Args:
        entry: dataset entry dict

    Returns:
        structured reasoning output string
    """
    text = entry["text"]
    answer = entry["answer"]
    steps = []

    # Split prompt at "Answer:" to get just the problem
    parts = text.split("\nAnswer:")
    problem = parts[0].strip() if parts else text

    # ── Arithmetic reasoning ──────────────────────────────────────────────
    arith = extract_arithmetic(problem)
    if arith:
        steps.append(f"Given: {arith['a']} {arith['op']} {arith['b']}")
        steps.append(f"Compute: {arith['a']} {arith['op']} {arith['b']} = {arith['result']}")
        return _format_chain(steps, str(arith["result"]))

    # ── Variable assignment chains ────────────────────────────────────────
    # Pattern: "X = blue. Y = X. Z = Y."
    assignments = re.findall(
        r'(?:Let |Set |Define |Given:? )?(\w+)\s*(?:=|is|copies|equals|points to|matched)\s+'
        r'(?:the same as |the value of )?(\w+)',
        problem, re.IGNORECASE
    )
    if assignments:
        resolved = {}
        for var, val in assignments:
            # Resolve chains
            actual = resolved.get(val, val)
            resolved[var] = actual
            steps.append(f"{var} ← {val}" + (f" = {actual}" if val != actual else ""))
        return _format_chain(steps, answer)

    # ── Containment chains ────────────────────────────────────────────────
    # Pattern: "card is in shelf. shelf is in closet."
    containment = re.findall(
        r'(\w+) is in (\w+)', problem, re.IGNORECASE
    )
    if containment:
        for item, container in containment:
            steps.append(f"{item} → {container}")
        return _format_chain(steps, answer)

    # ── Property chains ───────────────────────────────────────────────────
    # "X is the coin. The coin is blue."
    props = re.findall(
        r'(\w+) is (?:the )?(\w+)', problem, re.IGNORECASE
    )
    if props:
        for subj, prop in props:
            steps.append(f"{subj} = {prop}")
        return _format_chain(steps, answer)

    # ── Fallback: restate the problem as reasoning ────────────────────────
    sentences = [s.strip() for s in re.split(r'[.!?]', problem) if s.strip()]
    for i, s in enumerate(sentences[:-1]):  # Skip the question
        steps.append(f"Step {i+1}: {s}")

    return _format_chain(steps, answer)


def _format_chain(steps: list[str], answer: str) -> str:
    """Format reasoning steps into the target output template.

    Args:
        steps: list of reasoning step strings
        answer: final answer string

    Returns:
        formatted string: [Reasoning] ... [Answer] ... <HALT>
    """
    chain = "; ".join(steps)
    return f"[Reasoning] {chain} [Answer] {answer} <HALT>"


def inject_reasoning(entry: dict) -> dict:
    """Restructure entry's answer with reasoning chain.

    Prevents data leakage by ensuring the prompt is strictly the
    question. The text field ends at "Answer: " — the model must
    generate everything after that on its own.

    Args:
        entry: dataset entry dict

    Returns:
        entry with multi-token structured answer
    """
    chain = build_reasoning_chain(entry)

    entry = dict(entry)

    # 1. Strip the text field down to JUST the question, leaving it hanging.
    #    The model's job starts after "Answer: "
    parts = entry["text"].split("\nAnswer:")
    entry["text"] = parts[0].strip() + "\nAnswer: "

    # 2. The answer field contains the entire reasoning block
    #    for the loss calculation — NEVER appears in the prompt.
    entry["answer"] = chain

    return entry


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def clean_pipeline(
    input_path: str = "system2_logic_v1.json",
    output_path: str = "system2_logic_v2_clean.json",
) -> dict:
    """Run the full 3-stage cleaning pipeline.

    Args:
        input_path: path to raw dataset
        output_path: path to cleaned output

    Returns:
        stats dict with counts for each stage
    """
    print(f"\n{'='*60}")
    print(f"  DATA CLEANING PIPELINE")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    # Load
    with open(input_path) as f:
        data = json.load(f)
    total = len(data)
    print(f"  Loaded {total} entries\n")

    stats = {"total": total, "arith_fixed": 0, "arith_dropped": 0,
             "letter_fixed": 0, "reasoning_injected": 0}

    cleaned = []
    for entry in data:
        # ── Stage 1: Fix arithmetic ──
        entry = fix_arithmetic(entry)
        if entry.get("_arith_fixed"):
            stats["arith_fixed"] += 1

        # Drop entries where subtraction gives negative (nonsensical)
        arith = extract_arithmetic(entry["text"])
        if arith and arith["result"] < 0:
            stats["arith_dropped"] += 1
            continue

        # ── Stage 2: Strip letter prior ──
        entry = strip_letter_prior(entry)
        if entry.get("_letter_fixed"):
            stats["letter_fixed"] += 1

        # ── Stage 3: Inject reasoning chain ──
        entry = inject_reasoning(entry)
        stats["reasoning_injected"] += 1

        # Clean up internal markers
        for key in ["_arith_fixed", "_arith_was", "_arith_eq", "_letter_fixed"]:
            entry.pop(key, None)

        cleaned.append(entry)

    # Save
    with open(output_path, "w") as f:
        json.dump(cleaned, f, indent=2)

    output_size = os.path.getsize(output_path)

    print(f"  ┌─ Stage 1: Arithmetic Recalculation ───────────")
    print(f"  │  Fixed:   {stats['arith_fixed']}")
    print(f"  │  Dropped: {stats['arith_dropped']} (negative results)")
    print(f"  ├─ Stage 2: Letter Prior Stripping ─────────────")
    print(f"  │  Fixed:   {stats['letter_fixed']}")
    print(f"  ├─ Stage 3: Reasoning Chain Injection ──────────")
    print(f"  │  Injected: {stats['reasoning_injected']}")
    print(f"  ├─ Output ──────────────────────────────────────")
    print(f"  │  Entries: {len(cleaned)} / {total}")
    print(f"  │  Size:    {output_size / 1024:.1f} KB")
    print(f"  └──────────────────────────────────────────────\n")

    return stats


def demo_before_after() -> None:
    """Show a Before & After for a heavily corrupted example."""
    print("\n" + "=" * 64)
    print("  BEFORE & AFTER DEMO")
    print("=" * 64)

    # Construct a maximally corrupted example
    corrupted = {
        "text": "A = 5. B = A. Carol has 5 apples. Carol spends 5 apples. Carol now has?\nAnswer: 1",
        "answer": "1",
        "hops": 3,
    }

    print("\n  ── BEFORE (raw, corrupted) ──────────────────────")
    print(f"  text:   {corrupted['text']}")
    print(f"  answer: {corrupted['answer']}")
    print(f"  hops:   {corrupted['hops']}")

    # Stage 1: Fix arithmetic (5-5=0, not 1 from max(1,result))
    stage1 = fix_arithmetic(corrupted)
    print(f"\n  ── After Stage 1 (Arithmetic Fix) ──────────────")
    print(f"  answer: {stage1['answer']}  ← was '1', now '{stage1['answer']}'")
    if stage1.get("_arith_eq"):
        print(f"  equation: {stage1['_arith_eq']}")

    # Stage 2: Strip letter prior (A→P, B→Q)
    stage2 = strip_letter_prior(stage1)
    print(f"\n  ── After Stage 2 (Letter Prior Strip) ──────────")
    print(f"  text:   {stage2['text'].split(chr(10))[0]}")
    print(f"         A→P, B→Q to break MC heuristic")

    # Stage 3: Inject reasoning chain
    stage3 = inject_reasoning(stage2)
    print(f"\n  ── After Stage 3 (Reasoning Chain) ─────────────")
    parts = stage3["text"].split("\nAnswer: ")
    print(f"  prompt: {parts[0]}")
    print(f"  answer: {parts[1] if len(parts) > 1 else stage3['answer']}")

    print("\n  ── FINAL ENTRY ─────────────────────────────────")
    # Clean markers
    for key in ["_arith_fixed", "_arith_was", "_arith_eq", "_letter_fixed"]:
        stage3.pop(key, None)
    print(json.dumps(stage3, indent=4))
    print("=" * 64 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--test" in sys.argv:
        demo_before_after()
    else:
        if not os.path.exists("system2_logic_v1.json"):
            print("ERROR: system2_logic_v1.json not found")
            sys.exit(1)

        stats = clean_pipeline()
        demo_before_after()
        print("  ✓ Pipeline complete. Output: system2_logic_v2_clean.json\n")
