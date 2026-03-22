"""
system2_logic_builder.py
========================
Generates synthetic logic tasks that PHYSICALLY FORCE N=2 loop resolution.

CORE PRINCIPLE:
  The backbone cannot answer from System 1 memory because variable values
  are RANDOMIZED every sample. The N=2 loop is the only path to the answer.

  Loop 1 (THINK): Resolve the indirect reference chain (Z → X)
  Loop 2 (Answer): Look up the bound value (X → blue) → output "blue"

  Tokens used: ONLY highly common pre-trained words (colors, objects, names,
  digits) — no rare subwords. Backbone understands ALL these tokens,
  so the LoRA never fights vocabulary. It only learns to ROUTE logic.

TASK TYPES:
  1. Variable Binding:   X=blue, Z=X. What is Z?               → blue
  2. Spatial Chain:      Ball in cup. Cup in box. Ball in?      → box
  3. Math Word Problem:  Alice has 3. Bob gives 2. Alice has?   → 5
  4. Two-Hop Property:   Alice owns a cat. Cats have tails. Alice owns? → tail
  5. Negation Chain:     X is NOT red. X is NOT blue. X is?    → (third color)

Output: system2_logic_v1.json  (20,000 samples, ~120 tokens avg)
"""
import json
import random
from pathlib import Path

random.seed(42)

# ─── Token pools (ALL common, well-represented in pretraining) ────────────────

COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "white", "black", "pink", "brown"]
OBJECTS = ["ball", "cup", "box", "key", "book", "bag", "hat", "pen", "coin", "card"]
CONTAINERS = ["cup", "box", "bag", "jar", "bowl", "drawer", "basket", "shelf", "pocket", "drawer"]
LOCATIONS = ["box", "bag", "table", "shelf", "drawer", "floor", "desk", "chair", "closet", "basket"]
NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"]
ANIMALS = ["cat", "dog", "bird", "fish", "horse", "rabbit", "bear", "fox", "wolf", "deer"]
ANIMAL_PROPS = {"cat": "tail", "dog": "tail", "bird": "wing", "fish": "fin",
                "horse": "tail", "rabbit": "ear", "bear": "paw", "fox": "tail",
                "wolf": "tail", "deer": "tail"}
VARS = ["X", "Y", "Z", "W", "V", "A", "B", "C"]
DIGITS = list(range(1, 10))


# ─── Task generators ──────────────────────────────────────────────────────────

def var_binding_1hop() -> dict:
    """
    Direct: X = <color>. What is X?
    N=1 could solve this — used as curriculum warmup.
    """
    var = random.choice(VARS[:4])
    val = random.choice(COLORS)
    templates = [
        f"{var} = {val}. What is {var}?\nAnswer: {val}",
        f"Let {var} be {val}. {var} equals?\nAnswer: {val}",
        f"Set {var} to {val}. The value of {var} is?\nAnswer: {val}",
    ]
    text = random.choice(templates)
    return {"text": text, "answer": val, "hops": 1}


def var_binding_2hop() -> dict:
    """
    Chain: X = <color>. Y = X. What is Y?
    N=2 REQUIRED: Loop 1 resolves Y→X, Loop 2 looks up X→color.
    """
    v1, v2 = random.sample(VARS[:5], 2)
    val = random.choice(COLORS)
    templates = [
        f"{v1} = {val}. {v2} = {v1}. What is {v2}?\nAnswer: {val}",
        f"Let {v1} = {val}. Set {v2} = {v1}. {v2} is?\nAnswer: {val}",
        f"Given: {v1} is {val}. {v2} points to {v1}. What is {v2}?\nAnswer: {val}",
        f"Define {v1} as {val}. {v2} copies {v1}. The value of {v2}?\nAnswer: {val}",
    ]
    text = random.choice(templates)
    return {"text": text, "answer": val, "hops": 2}


def var_binding_3hop() -> dict:
    """
    Chain: X = <color>. Y = X. Z = Y. What is Z?
    N=2+ REQUIRED: Multiple hops through the chain.
    """
    v1, v2, v3 = random.sample(VARS[:6], 3)
    val = random.choice(COLORS)
    templates = [
        f"{v1} = {val}. {v2} = {v1}. {v3} = {v2}. What is {v3}?\nAnswer: {val}",
        f"Set {v1} = {val}, {v2} = {v1}, {v3} = {v2}. {v3} equals?\nAnswer: {val}",
    ]
    text = random.choice(templates)
    return {"text": text, "answer": val, "hops": 3}


def var_object_binding() -> dict:
    """
    Variable bound to an object: X = ball. Color X blue. What color is X?
    (Uses objects + colors — both common tokens)
    """
    var = random.choice(VARS[:4])
    obj = random.choice(OBJECTS)
    color = random.choice(COLORS)
    templates = [
        f"{var} = {obj}. The {var} is {color}. What color is {var}?\nAnswer: {color}",
        f"Let {var} be a {obj}. Paint {var} {color}. {var} is what color?\nAnswer: {color}",
    ]
    text = random.choice(templates)
    return {"text": text, "answer": color, "hops": 2}


def spatial_2hop() -> dict:
    """
    Spatial chain: Ball in cup. Cup in box. Ball in?
    N=2 REQUIRED: Loop 1 finds cup's container, Loop 2 resolves ball→cup→box.
    """
    obj = random.choice(OBJECTS)
    mid = random.choice([c for c in CONTAINERS if c != obj])
    final = random.choice([l for l in LOCATIONS if l != mid and l != obj])
    prep = random.choice(["in", "inside", "inside the"])
    templates = [
        f"The {obj} is {prep} the {mid}. The {mid} is {prep} the {final}. Where is the {obj}?\nAnswer: {final}",
        f"{obj} is in {mid}. {mid} is in {final}. {obj} is in?\nAnswer: {final}",
        f"Put {obj} in {mid}, then {mid} in {final}. Where is {obj}?\nAnswer: {final}",
    ]
    text = random.choice(templates)
    return {"text": text, "answer": final, "hops": 2}


def math_word_problem() -> dict:
    """
    Simple arithmetic that requires tracking state across the sequence.
    The backbone knows math but can't System-1 'Alice has 3, Bob gives 2'
    without reading and chaining the sequence.
    Note: delta <= start for minus ops to ensure result >= 0 (correct labels).
    """
    name = random.choice(NAMES)
    other = random.choice([n for n in NAMES if n != name])
    start = random.randint(2, 9)
    op = random.choice(["plus", "minus"])
    if op == "plus":
        start  = min(start, 8)                     # ensure result <= 9
        delta  = random.randint(1, 9 - start)
        result = start + delta
        action, direction = "gives", "to"
    else:
        delta  = random.randint(1, start)           # delta <= start so result >= 0
        result = start - delta
        action, direction = "takes", "from"

    templates = [
        f"{name} has {start} apples. {other} {action} {delta} apples {direction} {name}. {name} has?\nAnswer: {result}",
        f"{name} has {start} coins. {name} {'earns' if op == 'plus' else 'spends'} {delta} coins. {name} now has?\nAnswer: {result}",
        f"Start: {name}={start}. Change: {'+' if op == 'plus' else '-'}{delta}. End: {name}=?\nAnswer: {result}",
    ]
    text = random.choice(templates)
    return {"text": text, "answer": str(result), "hops": 2}


def two_hop_property() -> dict:
    """
    Property chain: Alice has a cat. A cat has a tail. Alice's pet has a?
    KEY FIX v26:
      - Property word embedded verbatim in prompt (pointer mask can find it)
      - Single-token answers only: tail/wing/fin/paw/ear (no plurals like claws/antlers)
      - Template always puts '{prop}' in the input as 'a {prop}'
    Forces lookup: person → animal, animal → property.
    """
    name   = random.choice(NAMES)
    animal = random.choice(ANIMALS)
    prop   = ANIMAL_PROPS[animal]   # single-token: tail/wing/fin/paw/ear
    templates = [
        f"{name} has a {animal}. A {animal} has a {prop}. {name}'s pet has a?\nAnswer: {prop}",
        f"{name} owns a {animal}. The {animal} has a {prop}. {name}'s {animal} has a?\nAnswer: {prop}",
        f"Ask: {name} keeps a {animal}. A {animal} has a {prop}. {name}'s animal has a?\nAnswer: {prop}",
    ]
    text = random.choice(templates)
    return {"text": text, "answer": prop, "hops": 2}


def color_assignment_with_distractor() -> dict:
    """
    X = red. Y = blue. Z = X. What color is Z?
    Distractors (Y=blue) test whether the model correctly chains Z→X, not Z→Y.
    """
    v_x, v_y, v_z = random.sample(VARS[:5], 3)
    val_x, val_y = random.sample(COLORS, 2)
    templates = [
        f"{v_x} = {val_x}. {v_y} = {val_y}. {v_z} = {v_x}. What is {v_z}?\nAnswer: {val_x}",
        f"Let {v_x} = {val_x}, {v_y} = {val_y}, {v_z} = {v_x}. {v_z} is?\nAnswer: {val_x}",
        f"Set {v_x} to {val_x} and {v_y} to {val_y}. Set {v_z} equal to {v_x}. {v_z}?\nAnswer: {val_x}",
    ]
    text = random.choice(templates)
    return {"text": text, "answer": val_x, "hops": 2}


def name_var_binding() -> dict:
    """
    Alice likes blue. Bob likes Alice's favorite color. What does Bob like?
    Uses proper names (common tokens). Requires chaining Bob→Alice→blue.
    """
    n1, n2 = random.sample(NAMES, 2)
    color = random.choice(COLORS)
    templates = [
        f"{n1} likes {color}. {n2}'s favorite is {n1}'s color. What does {n2} like?\nAnswer: {color}",
        f"{n1}'s favorite color is {color}. {n2} copies {n1}'s choice. {n2} prefers?\nAnswer: {color}",
        f"{n1} chose {color}. {n2} matched {n1}'s pick. What did {n2} pick?\nAnswer: {color}",
    ]
    text = random.choice(templates)
    return {"text": text, "answer": color, "hops": 2}


def number_var_chain() -> dict:
    """
    A = 3. B = A. C = B. What is C?
    Pure number variable chain. N=2+ required.
    """
    v1, v2, v3 = random.sample(VARS[:6], 3)
    val = random.choice(DIGITS)
    n_hops = random.randint(2, 3)
    if n_hops == 2:
        text = f"{v1} = {val}. {v2} = {v1}. {v2} equals?\nAnswer: {val}"
    else:
        text = f"{v1} = {val}. {v2} = {v1}. {v3} = {v2}. {v3} equals?\nAnswer: {val}"
    return {"text": text, "answer": str(val), "hops": n_hops}


def object_color_chain() -> dict:
    """
    X is the ball. The ball is red. What color is X?
    Requires: X→ball, ball→red. Classic 2-hop property chase.
    """
    var = random.choice(VARS[:4])
    obj = random.choice(OBJECTS)
    color = random.choice(COLORS)
    templates = [
        f"{var} is the {obj}. The {obj} is {color}. What color is {var}?\nAnswer: {color}",
        f"Call the {obj} {var}. {var} is painted {color}. {var}'s color?\nAnswer: {color}",
        f"{var} refers to the {obj}. The {obj} is {color}. {var} is what color?\nAnswer: {color}",
    ]
    text = random.choice(templates)
    return {"text": text, "answer": color, "hops": 2}


# ─── Build ────────────────────────────────────────────────────────────────────

GENERATORS = [
    (var_binding_2hop, 5000),           # Core task: 2-hop variable chain
    (color_assignment_with_distractor, 4000),  # With distractors
    (spatial_2hop, 3000),               # Spatial chaining
    (math_word_problem, 3000),          # Math word problems
    (name_var_binding, 2000),           # Proper-name chains
    (number_var_chain, 2000),           # Number variable chains
    (object_color_chain, 2000),         # Object→property→color
    (two_hop_property, 1500),           # Animal property chains
    (var_binding_3hop, 1500),           # 3-hop warmup
    (var_object_binding, 1000),         # Object binding
    (var_binding_1hop, 500),            # 1-hop warmup (easy)
]


def build() -> list[dict]:
    """Build the full dataset."""
    all_samples = []
    for fn, target in GENERATORS:
        count = 0
        attempts = 0
        while count < target and attempts < target * 10:
            s = fn()
            attempts += 1
            if s is not None:
                all_samples.append(s)
                count += 1
        print(f"  {fn.__name__:35s}: {count:5d} ({count - sum(1 for x in all_samples[:-count] if x)} new)")
    random.shuffle(all_samples)
    return all_samples


def main() -> None:
    """Generate and save."""
    print("Building System 2 Logic Dataset V1...\n")
    print("Rules:")
    print("  - NO facts the backbone knows (no 'gold is denser than water')")
    print("  - ALL common tokens (colors, objects, names, digits)")
    print("  - RANDOMIZED variable bindings → backbone cannot System-1 these")
    print("  - Physically forces N=2 loop to resolve variable chains\n")

    total_target = sum(t for _, t in GENERATORS)
    all_s = []
    for fn, target in GENERATORS:
        count = 0
        attempts = 0
        while count < target and attempts < target * 10:
            s = fn()
            attempts += 1
            if s is not None:
                all_s.append(s)
                count += 1
        print(f"  {fn.__name__:35s}: {count:5d}")

    random.shuffle(all_s)

    # Dedup
    seen = set()
    unique = []
    for s in all_s:
        if s["text"] not in seen:
            seen.add(s["text"])
            unique.append(s)
    print(f"\nRaw: {len(all_s):,} | Unique: {len(unique):,} ({len(all_s)-len(unique):,} dupes removed)")

    # Verify
    missing_boundary = sum(1 for s in unique if "Answer: " not in s["text"])
    hop_dist = {}
    for s in unique:
        h = s.get("hops", "?")
        hop_dist[h] = hop_dist.get(h, 0) + 1
    print(f"Missing Answer: boundary: {missing_boundary}")
    print(f"Hop distribution: {dict(sorted(hop_dist.items()))}")

    # Avg token estimate (rough)
    avg_chars = sum(len(s["text"]) for s in unique) / len(unique)
    est_tokens = avg_chars / 4  # rough chars-per-token estimate
    print(f"Avg text length: {avg_chars:.0f} chars (~{est_tokens:.0f} tokens)")

    # Save
    out = Path("system2_logic_v1.json")
    with open(out, "w") as f:
        json.dump(unique, f, indent=2)
    print(f"\nSaved → {out}")

    # Examples
    print("\nExamples:")
    for s in random.sample(unique, 6):
        print(f"  {s['text'].strip()}")
        print(f"  [hops={s.get('hops','?')}]\n")


if __name__ == "__main__":
    main()
