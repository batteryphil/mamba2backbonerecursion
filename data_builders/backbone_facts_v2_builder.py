"""
backbone_facts_v2_builder.py
============================
V2: Uses CONTENT-WORD answers instead of letters.

The key insight: the frozen mamba-130m backbone was trained on books/Wikipedia.
After "Answer:" it naturally wants to output CONTENT WORDS ("gold", "Paris"),
not abstract letters ("A", "B"). Training with letter answers forces the LoRA
to fight the backbone's prior — causing the 12-15% accuracy ceiling.

By matching the backbone's natural output format:
- Frozen backbone: cooperates fully (already knows "gold is denser than water")
- LoRA: just needs to learn the N=2 loop routing, not fight the output distribution

TASK DESIGNS:
The answer is always a SHORT, SINGLE-TOKEN word that the backbone naturally produces.
The THINK loop teaches the model to retrieve the relevant fact first.

Format:
  Question: Which is denser: gold or water?
  Answer: gold

The answer is 1-3 tokens max. The backbone ALREADY wants to output this after "Answer:".

Output: backbone_facts_v2.json (25,000 samples)
"""
import json
import random
from pathlib import Path

random.seed(42)

# ─── Fact tables ──────────────────────────────────────────────────────────────

DENSITIES = {
    "gold": 19.3, "platinum": 21.4, "lead": 11.3, "silver": 10.5,
    "copper": 8.9, "iron": 7.9, "aluminum": 2.7, "water": 1.0,
    "ice": 0.92, "cork": 0.12, "air": 0.0013, "glass": 2.5,
    "wood": 0.6, "tin": 7.3, "zinc": 7.1, "diamond": 3.5,
}

SPEEDS_KMH = {
    "light": 1_080_000_000, "sound": 1235, "rocket": 28000,
    "jet": 900, "cheetah": 112, "car": 100, "horse": 48,
    "bicycle": 25, "walking": 5, "snail": 0.05,
}

MELTING_POINTS_C = {
    "tungsten": 3422, "iron": 1538, "gold": 1064, "silver": 962,
    "aluminum": 660, "lead": 327, "tin": 232, "ice": 0,
    "nitrogen": -196, "oxygen": -218,
}

POPULATIONS_M = {
    "China": 1412, "India": 1417, "USA": 335, "Indonesia": 277,
    "Brazil": 215, "Russia": 144, "Japan": 125, "Germany": 84,
    "UK": 68, "France": 68, "Australia": 26, "Canada": 38,
}

CAPITALS = {
    "France": "Paris", "Germany": "Berlin", "Japan": "Tokyo",
    "Australia": "Canberra", "Canada": "Ottawa", "Brazil": "Brasilia",
    "Russia": "Moscow", "China": "Beijing", "India": "Delhi",
    "Spain": "Madrid", "Italy": "Rome", "Egypt": "Cairo",
    "Turkey": "Ankara", "Mexico": "Mexico City", "Argentina": "Buenos Aires",
}

CURRENCIES = {
    "USA": "dollar", "UK": "pound", "Japan": "yen", "Europe": "euro",
    "China": "yuan", "India": "rupee", "Russia": "ruble",
    "Brazil": "real", "Australia": "dollar", "Canada": "dollar",
}

MATH_PAIRS = [
    (2, 3, "plus", 5), (7, 4, "plus", 11), (9, 3, "times", 27),
    (8, 7, "times", 56), (20, 4, "divided by", 5), (36, 6, "divided by", 6),
    (15, 8, "minus", 7), (25, 6, "minus", 19), (12, 4, "times", 48),
    (100, 4, "divided by", 25), (6, 6, "times", 36), (50, 25, "minus", 25),
]

SCIENCE_FACTS = [
    ("Water is made of hydrogen and", "oxygen"),
    ("The chemical symbol for gold is", "Au"),
    ("The chemical symbol for iron is", "Fe"),
    ("The chemical symbol for silver is", "Ag"),
    ("The powerhouse of the cell is the", "mitochondria"),
    ("Light travels at approximately 300,000 kilometers per", "second"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("The closest star to Earth is the", "Sun"),
    ("Water freezes at", "zero"),
    ("Diamonds are made of", "carbon"),
    ("The Earth has", "one"),  # moon
    ("The speed of sound in air is approximately 343 meters per", "second"),
]

COMPARATIVES = [
    ("heavier", "lighter"), ("denser", "less dense"), ("faster", "slower"),
    ("hotter", "cooler"), ("larger", "smaller"), ("older", "younger"),
]


# ─── Sample generators ────────────────────────────────────────────────────────

def density_comparison() -> dict:
    """Which of two materials is denser? Answer is the denser one."""
    items = random.sample(list(DENSITIES.keys()), 2)
    a, b = items
    da, db = DENSITIES[a], DENSITIES[b]
    if da == db:
        return None
    denser = a if da > db else b
    lighter = b if da > db else a

    q_types = [
        f"Which is denser: {a} or {b}?\nAnswer: {denser}",
        f"Between {a} and {b}, which is heavier per unit volume?\nAnswer: {denser}",
        f"Which sinks deeper: {a} or {b}?\nAnswer: {denser}",
        f"Which would float on top: {a} or {b}?\nAnswer: {lighter}",
        f"Is {a} denser than {b}?\nAnswer: {'yes' if da > db else 'no'}",
    ]
    text = random.choice(q_types)
    answer = text.split("Answer: ")[1].strip()
    return {"text": text, "answer": answer}


def speed_comparison() -> dict:
    """Which of two things is faster? Answer is the faster one."""
    items = random.sample(list(SPEEDS_KMH.keys()), 2)
    a, b = items
    sa, sb = SPEEDS_KMH[a], SPEEDS_KMH[b]
    if sa == sb:
        return None
    faster = a if sa > sb else b
    slower = b if sa > sb else a

    q_types = [
        f"Which is faster: {a} or {b}?\nAnswer: {faster}",
        f"Is {a} faster than {b}?\nAnswer: {'yes' if sa > sb else 'no'}",
        f"Which travels at higher speed: {a} or {b}?\nAnswer: {faster}",
    ]
    text = random.choice(q_types)
    answer = text.split("Answer: ")[1].strip()
    return {"text": text, "answer": answer}


def melting_comparison() -> dict:
    """Which material has a higher melting point?"""
    items = random.sample(list(MELTING_POINTS_C.keys()), 2)
    a, b = items
    ma, mb = MELTING_POINTS_C[a], MELTING_POINTS_C[b]
    if ma == mb:
        return None
    hotter = a if ma > mb else b

    q_types = [
        f"Which has a higher melting point: {a} or {b}?\nAnswer: {hotter}",
        f"Which requires more heat to melt: {a} or {b}?\nAnswer: {hotter}",
        f"Does {a} melt at a higher temperature than {b}?\nAnswer: {'yes' if ma > mb else 'no'}",
    ]
    text = random.choice(q_types)
    answer = text.split("Answer: ")[1].strip()
    return {"text": text, "answer": answer}


def population_comparison() -> dict:
    """Which country has more people?"""
    items = random.sample(list(POPULATIONS_M.keys()), 2)
    a, b = items
    if POPULATIONS_M[a] == POPULATIONS_M[b]:
        return None
    bigger = a if POPULATIONS_M[a] > POPULATIONS_M[b] else b

    q_types = [
        f"Which country has more people: {a} or {b}?\nAnswer: {bigger}",
        f"Is {a}'s population larger than {b}'s?\nAnswer: {'yes' if POPULATIONS_M[a] > POPULATIONS_M[b] else 'no'}",
    ]
    text = random.choice(q_types)
    answer = text.split("Answer: ")[1].strip()
    return {"text": text, "answer": answer}


def capital_sample() -> dict:
    """What is the capital of X?"""
    country = random.choice(list(CAPITALS.keys()))
    capital = CAPITALS[country]
    text = f"What is the capital of {country}?\nAnswer: {capital}"
    return {"text": text, "answer": capital}


def currency_sample() -> dict:
    """What currency does X use?"""
    country = random.choice(list(CURRENCIES.keys()))
    currency = CURRENCIES[country]
    text = f"What currency is used in {country}?\nAnswer: {currency}"
    return {"text": text, "answer": currency}


def math_sample() -> dict:
    """Simple arithmetic with numeric answer."""
    a, b, op, result = random.choice(MATH_PAIRS)
    templates = [
        f"What is {a} {op} {b}?\nAnswer: {result}",
        f"{a} {op} {b} equals?\nAnswer: {result}",
        f"Calculate: {a} {op} {b}\nAnswer: {result}",
    ]
    text = random.choice(templates)
    answer = str(result)
    return {"text": text, "answer": answer}


def science_completion() -> dict:
    """Complete the science fact."""
    prefix, answer = random.choice(SCIENCE_FACTS)
    templates = [
        f"{prefix} ___.\nAnswer: {answer}",
        f"Complete: {prefix} what?\nAnswer: {answer}",
    ]
    text = random.choice(templates)
    return {"text": text, "answer": answer}


def transitive_chain() -> dict:
    """3-way transitive: A > B > C. Which is [most/least]?"""
    items = random.sample(list(DENSITIES.keys()), 3)
    a, b, c = sorted(items, key=lambda x: DENSITIES[x], reverse=True)
    # a > b > c

    q_types = [
        (f"{a} is denser than {b}, and {b} is denser than {c}.\nWhich is the least dense?\nAnswer: {c}", c),
        (f"{a} is denser than {b}, and {b} is denser than {c}.\nWhich is the most dense?\nAnswer: {a}", a),
        (f"If {a} > {b} > {c} in density, what has the middle density?\nAnswer: {b}", b),
    ]
    text, answer = random.choice(q_types)
    return {"text": text, "answer": answer}


def yes_no_density() -> dict:
    """Simple yes/no about density facts."""
    items = random.sample(list(DENSITIES.keys()), 2)
    a, b = items
    da, db = DENSITIES[a], DENSITIES[b]
    a_denser = da > db
    q = random.choice([
        (f"Is {a} denser than {b}?", "yes" if a_denser else "no"),
        (f"Does {a} float on {b}?", "yes" if not a_denser else "no"),
        (f"Is {b} lighter than {a}?", "yes" if db < da else "no"),
    ])
    text = f"{q[0]}\nAnswer: {q[1]}"
    return {"text": text, "answer": q[1]}


# ─── Dataset builder ──────────────────────────────────────────────────────────

GENERATORS = [
    (density_comparison, 7000),
    (speed_comparison, 3000),
    (melting_comparison, 3000),
    (population_comparison, 2000),
    (capital_sample, 3000),
    (currency_sample, 1000),
    (math_sample, 3000),
    (science_completion, 1000),
    (transitive_chain, 4000),
    (yes_no_density, 4000),
]


def build() -> list[dict]:
    """Build the dataset."""
    all_samples = []
    for fn, target in GENERATORS:
        count = 0
        attempts = 0
        while count < target and attempts < target * 20:
            sample = fn()
            attempts += 1
            if sample is not None:
                all_samples.append(sample)
                count += 1
        print(f"  {fn.__name__:28s}: {count:5d} samples")
    random.shuffle(all_samples)
    return all_samples


def verify(samples: list[dict]) -> None:
    """Verify dataset properties."""
    from collections import Counter
    answers = Counter(s["answer"] for s in samples)
    total = len(samples)
    print(f"\nTop 10 answers: {answers.most_common(10)}")
    print(f"Unique answers: {len(answers)}")
    yes_no = sum(1 for s in samples if s["answer"] in ("yes", "no"))
    print(f"Yes/No questions: {yes_no} ({yes_no/total*100:.1f}%)")
    print(f"Missing Answer: boundary: {sum(1 for s in samples if 'Answer: ' not in s['text'])}")
    # Check short answers (LoRA target should be 1-3 tokens)
    long_ans = [s["answer"] for s in samples if len(s["answer"].split()) > 3]
    print(f"Long answers (>3 words): {len(long_ans)}")


def main() -> None:
    """Generate and save."""
    print("Building backbone-facts V2 dataset...")
    print("(Content-word answers matching backbone natural output)\n")
    samples = build()
    # Dedup
    seen = set()
    unique = []
    for s in samples:
        if s["text"] not in seen:
            seen.add(s["text"])
            unique.append(s)
    print(f"\nRaw: {len(samples)} | Unique: {len(unique)} ({len(samples)-len(unique)} dupes removed)")
    verify(unique)

    out = Path("backbone_facts_v2.json")
    with open(out, "w") as f:
        json.dump(unique, f, indent=2)
    print(f"\nSaved → {out} ({len(unique)} samples)")

    # Show examples
    print("\nExamples:")
    for s in random.sample(unique, 5):
        print(f"  Q: {s['text'][:80].strip()!r}")
        print(f"  A: {s['answer']!r}\n")


if __name__ == "__main__":
    main()
