"""
data_builder_v2.py — Enhanced Training Data for RLF with Counterfactuals
==========================================================================
Extends system2_logic_builder.py with:
  - Counterfactual chains (overrides)
  - Distractor injection (irrelevant variables)
  - Multi-step arithmetic chains
  - Progressive curriculum (1-5 hops)
  - OOD validation set (6-10 hops, held out)

Output: system2_logic_v2.json (larger, more diverse dataset)
"""
import json
import random
from pathlib import Path
from typing import Optional

random.seed(42)

# ─── Token pools ──────────────────────────────────────────────────────────────

COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "white",
          "black", "pink", "brown"]
OBJECTS = ["ball", "cup", "box", "key", "book", "bag", "hat", "pen",
           "coin", "card"]
NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace",
         "Henry", "Ivy", "Jack"]
VARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = list(range(1, 10))


# ─── Core Generators ─────────────────────────────────────────────────────────

def var_chain(hops: int) -> dict:
    """Variable assignment chain: A = val. B = A. C = B. ... What is X?"""
    val = random.choice(COLORS)
    chain_vars = random.sample(VARS[:20], hops + 1)
    parts = [f"{chain_vars[0]} = {val}"]
    for i in range(1, len(chain_vars)):
        parts.append(f"{chain_vars[i]} = {chain_vars[i-1]}")
    text = ". ".join(parts) + f". What is {chain_vars[-1]}?\nAnswer: {val}"
    targets = [chain_vars[i] for i in range(hops - 1, -1, -1)] + [val, "<HALT>"]
    return {"text": text, "answer": val, "hops": hops,
            "chain_targets": targets}


def var_chain_counterfactual(hops: int) -> dict:
    """Chain with override: A = X. Override: A = Y. What is A? → Y."""
    val_orig = random.choice(COLORS)
    val_new = random.choice([c for c in COLORS if c != val_orig])
    chain_vars = random.sample(VARS[:15], max(2, hops + 1))

    parts = [f"{chain_vars[0]} = {val_orig}"]
    for i in range(1, len(chain_vars)):
        parts.append(f"{chain_vars[i]} = {chain_vars[i-1]}")

    # Override the base variable
    override_text = f"Override: {chain_vars[0]} = {val_new}"
    parts.append(override_text)
    text = ". ".join(parts) + f". What is {chain_vars[-1]}?\nAnswer: {val_new}"
    return {"text": text, "answer": val_new, "hops": hops,
            "type": "counterfactual",
            "chain_targets": [val_new, "<HALT>"]}


def var_chain_with_distractors(hops: int) -> dict:
    """Chain with irrelevant variables mixed in."""
    val = random.choice(COLORS)
    chain_vars = random.sample(VARS[:15], hops + 1)

    # Add 2-3 distractor variables
    n_distractors = random.randint(2, 3)
    used_vars = set(chain_vars)
    dist_vars = [v for v in VARS if v not in used_vars][:n_distractors]

    parts = []
    # Interleave chain and distractors
    parts.append(f"{chain_vars[0]} = {val}")
    for i in range(1, len(chain_vars)):
        if dist_vars and random.random() < 0.5:
            dv = dist_vars.pop()
            parts.append(f"{dv} = {random.choice(COLORS)}")
        parts.append(f"{chain_vars[i]} = {chain_vars[i-1]}")
    # Add remaining distractors
    for dv in dist_vars:
        parts.append(f"{dv} = {random.choice(COLORS)}")

    text = ". ".join(parts) + f". What is {chain_vars[-1]}?\nAnswer: {val}"
    targets = [chain_vars[i] for i in range(hops - 1, -1, -1)] + [val, "<HALT>"]
    return {"text": text, "answer": val, "hops": hops,
            "chain_targets": targets}


def math_chain(hops: int) -> Optional[dict]:
    """Multi-step arithmetic: start + op1 + op2 + ... = ?"""
    val = random.randint(2, 8)
    steps_text = [f"Start = {val}"]
    for _ in range(hops):
        op = random.choice(['+', '-'])
        delta = random.randint(1, 3)
        if op == '-' and val - delta < 0:
            op = '+'
        if op == '+' and val + delta > 15:
            op = '-'
        val = val + delta if op == '+' else val - delta
        steps_text.append(f"{op}{delta}")

    text = ". ".join(steps_text) + f". Result?\nAnswer: {val}"
    return {"text": text, "answer": str(val), "hops": hops,
            "chain_targets": [str(val), "<HALT>"]}


# ─── Build ────────────────────────────────────────────────────────────────────

def build() -> tuple[list, list]:
    """Build training and OOD validation sets."""
    train: list[dict] = []
    ood_val: list[dict] = []

    # Training data: 1-5 hops
    for hops in range(1, 6):
        n = {1: 2000, 2: 8000, 3: 8000, 4: 5000, 5: 3000}[hops]
        for _ in range(n):
            train.append(var_chain(hops))
        for _ in range(n // 3):
            train.append(var_chain_with_distractors(hops))
        if hops >= 2:
            for _ in range(n // 4):
                train.append(var_chain_counterfactual(hops))
            for _ in range(n // 5):
                s = math_chain(hops)
                if s:
                    train.append(s)

    # OOD validation: 6-10 hops (held out, never seen during training)
    for hops in range(6, 11):
        for _ in range(200):
            ood_val.append(var_chain(hops))
        for _ in range(100):
            ood_val.append(var_chain_with_distractors(hops))

    random.shuffle(train)
    random.shuffle(ood_val)

    # Dedup
    seen = set()
    unique_train = []
    for s in train:
        if s["text"] not in seen:
            seen.add(s["text"])
            unique_train.append(s)

    return unique_train, ood_val


def main() -> None:
    """Generate and save."""
    print("Building System 2 Logic Dataset V2 (Enhanced)...\n")
    print("Improvements over V1:")
    print("  - Counterfactual chains (override base variable)")
    print("  - Distractor injection (irrelevant variables)")
    print("  - Multi-step arithmetic chains")
    print("  - 1-5 hop curriculum + 6-10 hop OOD validation\n")

    train, ood = build()

    hop_dist: dict[int, int] = {}
    for s in train:
        h = s.get("hops", 0)
        hop_dist[h] = hop_dist.get(h, 0) + 1

    type_dist: dict[str, int] = {}
    for s in train:
        t = s.get("type", "chain")
        type_dist[t] = type_dist.get(t, 0) + 1

    print(f"Train: {len(train):,} samples")
    print(f"  Hop dist: {dict(sorted(hop_dist.items()))}")
    print(f"  Type dist: {type_dist}")
    print(f"OOD Val: {len(ood):,} samples (6-10 hops)")

    # Save training data
    out_train = Path("system2_logic_v2.json")
    with open(out_train, "w") as f:
        json.dump(train, f, indent=2)
    print(f"\nSaved → {out_train}")

    # Save OOD validation set
    out_ood = Path("system2_logic_v2_ood.json")
    with open(out_ood, "w") as f:
        json.dump(ood, f, indent=2)
    print(f"Saved → {out_ood}")

    # Examples
    print("\nExamples:")
    for s in random.sample(train, 4):
        t = s.get("type", "chain")
        print(f"  [{t}, {s['hops']}-hop] {s['text'][:80]}...")


if __name__ == "__main__":
    main()
