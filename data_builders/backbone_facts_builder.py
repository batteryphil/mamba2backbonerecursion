"""
backbone_facts_builder.py
=========================
Generates training data using REAL WORLD FACTS that the frozen mamba-130m backbone
already knows from pretraining. This is fundamentally different from the synthetic
entity-name approach — the backbone cooperates instead of fighting us.

KEY INSIGHT:
  - Frozen backbone has pretraining knowledge (science, geography, math, etc.)
  - LoRA just needs to learn: "use N=2 loops to retrieve → reason → answer"
  - Single-letter answers (A/B/C/D) avoid the multi-token generation problem

LOOP TRAINING TARGET:
  Loop 1 (THINK): Model retrieves the relevant fact from backbone weights
  Loop 2 (Answer): LoRA resolves the comparison → outputs single letter

Output: backbone_facts_v1.json   (25,000 samples, balanced A/B/C/D distribution)
"""
import json
import random
from pathlib import Path

random.seed(42)

# ─── Fact tables ──────────────────────────────────────────────────────────────

DENSITIES = {  # g/cm³
    "gold": 19.3, "platinum": 21.4, "lead": 11.3, "silver": 10.5,
    "copper": 8.9, "iron": 7.9, "steel": 7.8, "zinc": 7.1,
    "tin": 7.3, "aluminum": 2.7, "glass": 2.5, "water": 1.0,
    "ice": 0.92, "wood (oak)": 0.77, "cork": 0.12, "air": 0.0013,
}

SPEEDS_KMH = {  # km/h
    "light": 1_080_000_000, "sound (air)": 1235, "space shuttle": 28000,
    "commercial jet": 900, "bullet train": 320, "cheetah": 112,
    "highway car": 100, "falcon": 89, "greyhound": 72,
    "human sprint": 45, "horse": 48, "bicycle": 25, "walking": 5,
}

DISTANCES_KM = {  # distance from Earth or size in km
    "Sun": 149_600_000, "Moon": 384_400, "Mars": 225_000_000,
    "Jupiter": 778_000_000, "Saturn": 1_430_000_000,
    "Mount Everest (height)": 8.849, "Mariana Trench (depth)": 10.935,
    "Amazon River (length)": 6400, "Nile River (length)": 6650,
    "Pacific Ocean (width)": 12_300,
}

POPULATIONS = {  # approximate, in millions
    "China": 1412, "India": 1417, "USA": 335, "Indonesia": 277,
    "Pakistan": 231, "Brazil": 215, "Nigeria": 220, "Bangladesh": 170,
    "Russia": 144, "Mexico": 130, "Japan": 125, "Germany": 84,
    "UK": 68, "France": 68, "Australia": 26, "Canada": 38,
}

MELTING_POINTS_C = {  # °C
    "tungsten": 3422, "iron": 1538, "gold": 1064, "silver": 962,
    "aluminum": 660, "tin": 232, "lead": 327, "ice": 0,
    "oxygen (liquid)": -218, "nitrogen (liquid)": -196,
    "hydrogen (liquid)": -259,
}

CAPITALS = {
    "France": "Paris", "Germany": "Berlin", "Japan": "Tokyo",
    "Australia": "Canberra", "Canada": "Ottawa", "Brazil": "Brasilia",
    "Russia": "Moscow", "China": "Beijing", "India": "New Delhi",
    "USA": "Washington D.C.", "UK": "London", "Mexico": "Mexico City",
    "Spain": "Madrid", "Italy": "Rome", "South Korea": "Seoul",
    "Egypt": "Cairo", "Argentina": "Buenos Aires", "Nigeria": "Abuja",
    "Saudi Arabia": "Riyadh", "Turkey": "Ankara",
}

WRONG_CAPITALS = [
    "London", "Tokyo", "Paris", "Berlin", "Moscow", "Beijing",
    "New York", "Los Angeles", "Sydney", "Toronto", "Dubai",
    "Shanghai", "Mumbai", "Lagos", "Cape Town",
]

MATH_FACTS = [
    (2, 3, "+", 5), (7, 4, "+", 11), (15, 8, "+", 23), (100, 37, "+", 137),
    (9, 3, "*", 27), (7, 8, "*", 56), (12, 6, "*", 72), (5, 9, "*", 45),
    (20, 4, "/", 5), (36, 6, "/", 6), (100, 5, "/", 20), (48, 8, "/", 6),
    (10, 3, "-", 7), (25, 8, "-", 17), (100, 47, "-", 53), (50, 23, "-", 27),
]

SCIENCE_TF = [  # (statement, is_true)
    ("Hot air rises", True), ("Water boils at 100°C at sea level", True),
    ("Diamonds are made of carbon", True), ("Sound travels faster than light", False),
    ("The Earth orbits the Sun", True), ("The Moon orbits the Earth", True),
    ("Plants produce oxygen", True), ("Humans have 23 pairs of chromosomes", True),
    ("Gold is more dense than water", True), ("Ice is less dense than liquid water", True),
    ("Lightning is hotter than the Sun's surface", True),
    ("Sound can travel through a vacuum", False),
    ("The Earth is flat", False), ("Human body is mostly carbon", False),
    ("Bats are blind", False), ("Penguins live in the Arctic", False),
]


# ─── Sample generators ────────────────────────────────────────────────────────

def ab_prompt(question: str, a_text: str, b_text: str, correct: str) -> dict:
    """Generate A/B choice prompt. correct must be 'A' or 'B'."""
    text = f"{question}\nA. {a_text}\nB. {b_text}\nAnswer: {correct}"
    return {"text": text, "answer": correct}


def abcd_prompt(question: str, options: list[str], correct_idx: int) -> dict:
    """Generate A/B/C/D prompt. correct_idx is 0-3."""
    letters = ["A", "B", "C", "D"]
    option_text = "\n".join(f"{letters[i]}. {options[i]}" for i in range(4))
    correct = letters[correct_idx]
    text = f"{question}\n{option_text}\nAnswer: {correct}"
    return {"text": text, "answer": correct}


def make_density_sample() -> dict:
    items = random.sample(list(DENSITIES.keys()), 2)
    a, b = items
    da, db = DENSITIES[a], DENSITIES[b]
    if da == db:
        return None
    question_types = [
        (f"Which is denser: {a} or {b}?",
         a if da > db else b, b if da > db else a,
         "A" if da > db else "B"),
        (f"Is {a} denser than {b}?",
         "Yes", "No",
         "A" if da > db else "B"),
        (f"Which has greater density?",
         a.capitalize(), b.capitalize(),
         "A" if da > db else "B"),
    ]
    q, a_opt, b_opt, ans = random.choice(question_types)
    return ab_prompt(q, a_opt, b_opt, ans)


def make_speed_sample() -> dict:
    items = random.sample(list(SPEEDS_KMH.keys()), 2)
    a, b = items
    sa, sb = SPEEDS_KMH[a], SPEEDS_KMH[b]
    if sa == sb:
        return None
    q_types = [
        (f"Which travels faster: {a} or {b}?",
         a, b, "A" if sa > sb else "B"),
        (f"Is {a} faster than {b}?",
         "Yes", "No", "A" if sa > sb else "B"),
    ]
    q, ao, bo, ans = random.choice(q_types)
    return ab_prompt(q, ao, bo, ans)


def make_population_sample() -> dict:
    items = random.sample(list(POPULATIONS.keys()), 2)
    a, b = items
    pa, pb = POPULATIONS[a], POPULATIONS[b]
    if pa == pb:
        return None
    q_types = [
        (f"Which country has a larger population: {a} or {b}?",
         a, b, "A" if pa > pb else "B"),
        (f"Does {a} have more people than {b}?",
         "Yes", "No", "A" if pa > pb else "B"),
    ]
    q, ao, bo, ans = random.choice(q_types)
    return ab_prompt(q, ao, bo, ans)


def make_melting_sample() -> dict:
    items = random.sample(list(MELTING_POINTS_C.keys()), 2)
    a, b = items
    ma, mb = MELTING_POINTS_C[a], MELTING_POINTS_C[b]
    if ma == mb:
        return None
    q_types = [
        (f"Which has a higher melting point: {a} or {b}?",
         a, b, "A" if ma > mb else "B"),
        (f"Does {a} melt at a higher temperature than {b}?",
         "Yes", "No", "A" if ma > mb else "B"),
    ]
    q, ao, bo, ans = random.choice(q_types)
    return ab_prompt(q, ao, bo, ans)


def make_capital_sample() -> dict:
    country = random.choice(list(CAPITALS.keys()))
    correct_capital = CAPITALS[country]
    # Generate 3 wrong capitals
    wrongs = [w for w in WRONG_CAPITALS if w != correct_capital]
    wrongs = random.sample(wrongs, 3)
    options = [correct_capital] + wrongs
    random.shuffle(options)
    correct_idx = options.index(correct_capital)
    q = f"What is the capital city of {country}?"
    return abcd_prompt(q, options, correct_idx)


def make_math_sample() -> dict:
    a, b, op, result = random.choice(MATH_FACTS)
    op_symbol = {"+" : "+", "*": "×", "/": "÷", "-": "-"}[op]
    op_word = {"+": "plus", "*": "times", "/": "divided by", "-": "minus"}[op]

    # Make wrong answer variants
    delta = random.choice([1, 2, 3, 5, 10])
    wrong = result + random.choice([-delta, delta])
    if wrong == result:
        wrong = result + 1

    q_types = [
        (f"What is {a} {op_symbol} {b}?",
         [str(result), str(wrong), str(result + 7), str(abs(result - 4))]),
        (f"{a} {op_word} {b} equals what?",
         [str(result), str(wrong), str(result * 2), str(result - 3)]),
    ]
    q, opts = random.choice(q_types)
    # Shuffle and find correct index
    random.shuffle(opts)
    # Ensure result is unique in opts
    while opts.count(str(result)) > 1:
        opts[opts.index(str(result), 1)] = str(int(opts[opts.index(str(result), 1)]) + 13)
    correct_idx = opts.index(str(result))
    return abcd_prompt(q, opts, correct_idx)


def make_science_tf_sample() -> dict:
    statement, is_true = random.choice(SCIENCE_TF)
    q = f"True or False: {statement}"
    return ab_prompt(q, "True", "False", "A" if is_true else "B")


def make_transitive_sample() -> dict:
    """
    Multi-step transitive reasoning over REAL quantities the model knows.
    E.g.: Gold is denser than silver. Silver is denser than aluminum.
    Therefore, which is least dense?
    """
    items = random.sample(list(DENSITIES.keys()), 3)
    a, b, c = sorted(items, key=lambda x: DENSITIES[x], reverse=True)
    # a > b > c (by density)

    question_types = [
        (f"{a} is denser than {b}.\n{b} is denser than {c}.\nWhich material is the least dense?",
         a, b, c, "C"),
        (f"{a} is denser than {b}.\n{b} is denser than {c}.\nWhich material is the most dense?",
         a, b, c, "A"),
        (f"If {a} > {b} > {c} in density, which has the middle density?",
         a, b, c, "B"),
    ]
    q, ao, bo, co, ans = random.choice(question_types)
    opts = [ao.capitalize(), bo.capitalize(), co.capitalize(), "None of the above"]
    idx = {"A": 0, "B": 1, "C": 2}[ans]
    return abcd_prompt(q, opts, idx)


def make_arithmetic_chain() -> dict:
    """Two-step arithmetic the model can verify."""
    a = random.randint(2, 20)
    b = random.randint(2, 10)
    c = random.randint(1, 5)
    result = a + b - c
    wrong = result + random.choice([-2, -1, 1, 2, 3])
    if wrong == result:
        wrong += 1
    q = f"If x = {a} + {b}, then x - {c} = ?"
    opts = [str(result), str(wrong), str(a + b), str(a - c)]
    random.shuffle(opts)
    while opts.count(str(result)) > 1:
        for i in range(len(opts)):
            if opts[i] == str(result) and i != opts.index(str(result)):
                opts[i] = str(int(opts[i]) + 17)
                break
    correct_idx = opts.index(str(result))
    return abcd_prompt(q, opts, correct_idx)


# ─── Dataset builder ──────────────────────────────────────────────────────────

GENERATORS = [
    (make_density_sample, 5000),
    (make_speed_sample, 3000),
    (make_population_sample, 3000),
    (make_melting_sample, 3000),
    (make_capital_sample, 4000),
    (make_math_sample, 4000),
    (make_science_tf_sample, 2000),
    (make_transitive_sample, 3000),
    (make_arithmetic_chain, 3000),
]


def build_dataset() -> list[dict]:
    """Build the full dataset."""
    all_samples = []
    for fn, target in GENERATORS:
        count = 0
        attempts = 0
        while count < target and attempts < target * 10:
            sample = fn()
            attempts += 1
            if sample is not None:
                all_samples.append(sample)
                count += 1
        print(f"  {fn.__name__:30s}: {count:5d} samples")
    random.shuffle(all_samples)
    return all_samples


def verify_balance(samples: list[dict]) -> None:
    """Verify answer letter distribution is balanced."""
    from collections import Counter
    dist = Counter(s["answer"] for s in samples)
    total = len(samples)
    print(f"\nAnswer distribution ({total} total):")
    for letter in sorted(dist):
        pct = dist[letter] / total * 100
        print(f"  {letter}: {dist[letter]:5d} ({pct:.1f}%)")


def main() -> None:
    """Generate and save the backbone-knowledge dataset."""
    print("Building backbone-knowledge dataset...")
    print("(Using real facts the frozen mamba-130m already knows)\n")

    samples = build_dataset()
    verify_balance(samples)

    # Verify Answer: boundary is always present
    missing = sum(1 for s in samples if "Answer: " not in s["text"])
    print(f"\nMissing 'Answer: ' boundary: {missing}")
    print(f"Total samples: {len(samples)}")

    # Show examples
    print("\nExamples:")
    for s in random.sample(samples, 5):
        print(f"  {repr(s['text'][:100])}")

    out = Path("backbone_facts_v1.json")
    with open(out, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
