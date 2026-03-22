"""
logic_v5_generator.py — Extended Reasoning Dataset Generator

Adds 4 new reasoning types missing from logic_v4.json:
  1. Negation          — "Alice is NOT taller than Bob. Who is taller?"
  2. Multi-hop         — "A > B, B > C. Who is smallest?"
  3. Contradiction     — "Alice > Bob, Bob > Alice. Is this consistent?"
  4. Numerical         — "Alice is 5cm taller than Bob who is 170cm. How tall is Alice?"

Generates 2,500 samples per category = 10,000 new samples total.
Merges with existing logic_v4.json → logic_v5.json
"""
import json, random

random.seed(42)

NAMES    = ["Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace", "Hank",
            "Iris", "Jack", "Kate", "Leo", "Mia", "Nick", "Olivia", "Paul"]
OBJECTS  = ["rock", "feather", "brick", "coin", "pebble", "stone", "leaf",
            "hammer", "book", "phone", "pencil", "key", "bottle", "cup"]
ATTRS = {
    "height":  ("tall",   "short",  "taller",  "shorter", "tallest",  "shortest",  "cm tall"),
    "weight":  ("heavy",  "light",  "heavier", "lighter", "heaviest", "lightest",  "kg"),
    "speed":   ("fast",   "slow",   "faster",  "slower",  "fastest",  "slowest",   "mph"),
    "age":     ("old",    "young",  "older",   "younger", "oldest",   "youngest",  "years old"),
    "strength":("strong", "weak",   "stronger","weaker",  "strongest","weakest",   "kg lifted"),
}

samples = []


# ── 1. Negation ───────────────────────────────────────────────────────────────
def gen_negation(n=2500):
    """Alice is NOT taller than Bob. Who is taller? Answer: Bob."""
    results = []
    for _ in range(n):
        a, b = random.sample(NAMES, 2)
        attr, (pos, neg, comp_pos, comp_neg, sup_pos, sup_neg, unit) = random.choice(list(ATTRS.items()))
        # Negation: a is NOT comp_pos than b → b is comp_pos than a
        templates = [
            f"{a} is NOT {comp_pos} than {b}. Who is {comp_pos}? Answer: {b}.",
            f"{a} is NOT {comp_pos} than {b}. So who is {sup_pos} between them? Answer: {b}.",
            f"It is false that {a} is {comp_pos} than {b}. Who is {comp_neg}? Answer: {a}.",
            f"{a} is not {pos}er than {b}. Who is {neg}er? Answer: {a}.",
            f"{a} is NOT more {pos} than {b}. Who is {comp_pos}? Answer: {b}.",
        ]
        results.append({"text": random.choice(templates), "type": "negation"})
    return results


# ── 2. Multi-hop transitivity ─────────────────────────────────────────────────
def gen_multihop(n=2500):
    """A > B, B > C. Who is smallest? Answer: C."""
    results = []
    for _ in range(n):
        entities = random.sample(NAMES + OBJECTS, 4)
        attr, (pos, neg, comp_pos, comp_neg, sup_pos, sup_neg, unit) = random.choice(list(ATTRS.items()))
        chain_len = random.choice([3, 4])
        chain = entities[:chain_len]
        # chain[0] > chain[1] > ... > chain[-1]
        facts = [f"{chain[i]} is {comp_pos} than {chain[i+1]}" for i in range(chain_len - 1)]
        premise = ". ".join(facts) + "."
        smallest = chain[-1]
        largest  = chain[0]
        mid_opts = chain[1:-1]

        q_type = random.randint(0, 3)
        if q_type == 0:
            q = f"{premise} Who is {sup_neg}? Answer: {smallest}."
        elif q_type == 1:
            q = f"{premise} Who is {sup_pos}? Answer: {largest}."
        elif q_type == 2 and mid_opts:
            mid = random.choice(mid_opts)
            idx = chain.index(mid)
            above = chain[idx-1]
            below = chain[idx+1]
            q = f"{premise} Is {mid} {comp_pos} than {below}? Answer: Yes."
        else:
            q = f"{premise} Rank from {sup_pos} to {sup_neg}: Answer: {' > '.join(chain)}."
        results.append({"text": q, "type": "multihop"})
    return results


# ── 3. Contradiction detection ────────────────────────────────────────────────
def gen_contradiction(n=2500):
    """Alice > Bob and Bob > Alice. Is this consistent? Answer: No."""
    results = []
    for _ in range(n):
        a, b, c = random.sample(NAMES, 3)
        attr, (pos, neg, comp_pos, comp_neg, sup_pos, sup_neg, unit) = random.choice(list(ATTRS.items()))

        # Direct contradiction
        contradict_templates = [
            f"{a} is {comp_pos} than {b}. {b} is {comp_pos} than {a}. Is this consistent? Answer: No, this is a contradiction.",
            f"{a} is {comp_pos} than {b} and {b} is {comp_pos} than {a}. Can both be true? Answer: No.",
            f"Claim 1: {a} is {comp_pos} than {b}. Claim 2: {b} is {comp_pos} than {a}. Are these compatible? Answer: No, they contradict each other.",
        ]
        # Transitive contradiction: A > B, B > C, C > A
        transitive_templates = [
            f"{a} is {comp_pos} than {b}. {b} is {comp_pos} than {c}. {c} is {comp_pos} than {a}. Is this possible? Answer: No, this creates a circular contradiction.",
            f"If {a} is {comp_pos} than {b} and {b} is {comp_pos} than {c}, can {c} be {comp_pos} than {a}? Answer: No.",
        ]
        # Consistent case
        consistent_templates = [
            f"{a} is {comp_pos} than {b}. {b} is {comp_pos} than {c}. Is this consistent? Answer: Yes.",
            f"{a} is {comp_pos} than {b} and {b} is {comp_pos} than {c}. Can both be true? Answer: Yes.",
        ]

        roll = random.random()
        if roll < 0.4:
            results.append({"text": random.choice(contradict_templates), "type": "contradiction"})
        elif roll < 0.7:
            results.append({"text": random.choice(transitive_templates), "type": "contradiction_transitive"})
        else:
            results.append({"text": random.choice(consistent_templates), "type": "consistent"})
    return results


# ── 4. Numerical reasoning ────────────────────────────────────────────────────
def gen_numerical(n=2500):
    """Alice is 5cm taller than Bob who is 170cm tall. How tall is Alice? Answer: 175cm."""
    results = []
    for _ in range(n):
        a, b, c = random.sample(NAMES, 3)
        attr, (pos, neg, comp_pos, comp_neg, sup_pos, sup_neg, unit) = random.choice(list(ATTRS.items()))

        base    = random.randint(100, 220)
        diff1   = random.randint(1, 30)
        diff2   = random.randint(1, 20)
        a_val   = base + diff1
        b_val   = base
        c_val   = base - diff2

        q_type = random.randint(0, 3)
        if q_type == 0:
            t = (f"{a} is {diff1} {unit} {comp_pos} than {b}, who is {b_val} {unit}. "
                 f"How {pos} is {a}? Answer: {a_val} {unit}.")
        elif q_type == 1:
            t = (f"{a} is {a_val} {unit}. {b} is {diff1} {unit} {comp_neg} than {a}. "
                 f"How {pos} is {b}? Answer: {b_val} {unit}.")
        elif q_type == 2:
            t = (f"{a} is {a_val} {unit}, {b} is {b_val} {unit}. "
                 f"By how much is {a} {comp_pos} than {b}? Answer: {diff1} {unit}.")
        else:
            t = (f"{a} is {a_val} {unit}, {b} is {b_val} {unit}, {c} is {c_val} {unit}. "
                 f"Who is {sup_neg}? Answer: {c}.")
        results.append({"text": t, "type": "numerical"})
    return results


# ── Generate all ─────────────────────────────────────────────────────────────
print("Generating extended reasoning dataset...")
new_samples = []
new_samples += gen_negation(2500);      print(f"  Negation:       2,500 samples")
new_samples += gen_multihop(2500);      print(f"  Multi-hop:      2,500 samples")
new_samples += gen_contradiction(2500); print(f"  Contradiction:  2,500 samples")
new_samples += gen_numerical(2500);     print(f"  Numerical:      2,500 samples")

random.shuffle(new_samples)
print(f"  New total:      {len(new_samples):,} samples\n")

# ── Merge with logic_v4.json ─────────────────────────────────────────────────
try:
    existing = json.load(open("logic_v4.json"))
    print(f"Loaded logic_v4.json: {len(existing):,} existing samples")
    merged = existing + new_samples
except Exception as e:
    print(f"Could not load logic_v4.json ({e}) — saving new samples only")
    merged = new_samples

random.shuffle(merged)
json.dump(merged, open("logic_v5.json", "w"), indent=2)
print(f"Saved logic_v5.json: {len(merged):,} total samples")
print(f"Done.")
