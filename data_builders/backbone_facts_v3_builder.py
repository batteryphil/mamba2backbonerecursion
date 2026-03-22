"""
backbone_facts_v3_builder.py
============================
V3: Expanded fact tables for 15k+ unique samples.
Content-word answers matching backbone's natural output distribution.
"""
import json, random
from itertools import combinations
from pathlib import Path

random.seed(42)

# ─── LARGE fact tables ────────────────────────────────────────────────────────

# Density in g/cm³ — 30 materials
DENSITIES = {
    "osmium": 22.6, "iridium": 22.4, "platinum": 21.4, "gold": 19.3,
    "tungsten": 19.3, "uranium": 19.1, "lead": 11.3, "silver": 10.5,
    "copper": 8.9, "iron": 7.9, "steel": 7.8, "nickel": 8.9,
    "tin": 7.3, "zinc": 7.1, "chrome": 7.2, "titanium": 4.5,
    "aluminum": 2.7, "glass": 2.5, "diamond": 3.5, "concrete": 2.3,
    "water": 1.0, "seawater": 1.03, "ice": 0.92, "wood": 0.6,
    "oil": 0.85, "ethanol": 0.79, "cork": 0.12, "foam": 0.06,
    "air": 0.0013, "hydrogen": 0.00009,
}

# Speed in km/h — 20 items
SPEEDS = {
    "light": 1_080_000_000, "electron beam": 900_000_000,
    "solar wind": 1_440_000, "rocket": 28000, "meteor": 72000,
    "space shuttle": 28000, "supersonic jet": 2450, "commercial jet": 900,
    "bullet": 1800, "sound (air)": 1235, "Formula 1 car": 360,
    "bullet train": 320, "hawk": 240, "cheetah": 112, "car": 100,
    "horse": 48, "human sprint": 45, "bicycle": 25,
    "walking": 5, "snail": 0.05,
}

# Melting points in °C — 20 items
MELTING = {
    "tungsten": 3422, "carbon": 3527, "osmium": 3033, "rhenium": 3186,
    "platinum": 1768, "iron": 1538, "nickel": 1455, "gold": 1064,
    "silver": 962, "copper": 1085, "aluminum": 660, "zinc": 419,
    "lead": 327, "tin": 232, "sodium": 98, "sulfur": 113,
    "ice": 0, "mercury": -39, "oxygen": -218, "nitrogen": -196,
}

# Boiling points in °C — 15 items
BOILING = {
    "tungsten": 5555, "iron": 2862, "gold": 2856, "copper": 2562,
    "silver": 2162, "aluminum": 2519, "lead": 1749, "tin": 2602,
    "zinc": 907, "sulfur": 445, "water": 100, "ethanol": 78,
    "chlorine": -34, "nitrogen": -196, "oxygen": -183,
}

# Atomic numbers — 25 elements
ATOMIC_NUMBERS = {
    "hydrogen": 1, "helium": 2, "carbon": 6, "nitrogen": 7,
    "oxygen": 8, "sodium": 11, "magnesium": 12, "aluminum": 13,
    "silicon": 14, "sulfur": 16, "chlorine": 17, "calcium": 20,
    "iron": 26, "copper": 29, "zinc": 30, "silver": 47,
    "tin": 50, "gold": 79, "mercury": 80, "lead": 82,
    "uranium": 92, "plutonium": 94, "helium": 2, "neon": 10, "argon": 18,
}

# Capitals — 40 countries
CAPITALS = {
    "France": "Paris", "Germany": "Berlin", "Japan": "Tokyo",
    "China": "Beijing", "Russia": "Moscow", "USA": "Washington",
    "UK": "London", "India": "Delhi", "Brazil": "Brasilia",
    "Australia": "Canberra", "Canada": "Ottawa", "Italy": "Rome",
    "Spain": "Madrid", "Mexico": "Mexico City", "Argentina": "Buenos Aires",
    "Egypt": "Cairo", "Nigeria": "Abuja", "South Africa": "Pretoria",
    "Turkey": "Ankara", "Saudi Arabia": "Riyadh", "Iran": "Tehran",
    "Pakistan": "Islamabad", "Bangladesh": "Dhaka", "South Korea": "Seoul",
    "Thailand": "Bangkok", "Vietnam": "Hanoi", "Indonesia": "Jakarta",
    "Philippines": "Manila", "Malaysia": "Kuala Lumpur", "Sweden": "Stockholm",
    "Norway": "Oslo", "Denmark": "Copenhagen", "Finland": "Helsinki",
    "Poland": "Warsaw", "Netherlands": "Amsterdam", "Belgium": "Brussels",
    "Switzerland": "Bern", "Austria": "Vienna", "Greece": "Athens",
    "Portugal": "Lisbon",
}

# Math problems — generating many variants
def gen_math_samples(n: int = 4000) -> list[dict]:
    """Generate arithmetic problems with numeric answers."""
    samples = []
    ops = [
        ("+", "plus", lambda a, b: a + b),
        ("-", "minus", lambda a, b: a - b),
        ("*", "times", lambda a, b: a * b),
    ]
    for _ in range(n * 5):
        a = random.randint(2, 20)
        b = random.randint(2, 10)
        op_sym, op_word, fn = random.choice(ops)
        result = fn(a, b)
        if result < 0 or result > 200:
            continue
        templates = [
            f"What is {a} {op_sym} {b}?\nAnswer: {result}",
            f"{a} {op_word} {b} equals?\nAnswer: {result}",
            f"What does {a} {op_word} {b} make?\nAnswer: {result}",
            f"Calculate {a} {op_sym} {b}.\nAnswer: {result}",
        ]
        samples.append({"text": random.choice(templates), "answer": str(result)})
        if len(samples) >= n:
            break
    return samples

# Planet data — sizes and distances
PLANET_SIZES_KM = {
    "Jupiter": 139820, "Saturn": 116460, "Uranus": 50724,
    "Neptune": 49244, "Earth": 12742, "Venus": 12104,
    "Mars": 6779, "Mercury": 4879,
}
PLANET_DISTANCES_M_KM = {  # million km from Sun
    "Mercury": 57.9, "Venus": 108, "Earth": 150, "Mars": 228,
    "Jupiter": 779, "Saturn": 1430, "Uranus": 2870, "Neptune": 4500,
}

# Chemical symbols
CHEM_SYMBOLS = {
    "gold": "Au", "silver": "Ag", "iron": "Fe", "copper": "Cu",
    "lead": "Pb", "potassium": "K", "sodium": "Na", "tungsten": "W",
    "tin": "Sn", "mercury": "Hg", "water": "H2O", "oxygen": "O",
    "nitrogen": "N", "hydrogen": "H", "carbon": "C", "helium": "He",
}

# Science yes/no
SCIENCE_YN = [
    ("Does ice float on water?", "yes"),
    ("Is gold magnetic?", "no"),
    ("Is iron magnetic?", "yes"),
    ("Does sound travel faster than light?", "no"),
    ("Is water denser than oil?", "yes"),
    ("Does hot air rise?", "yes"),
    ("Is the Earth round?", "yes"),
    ("Do plants produce oxygen?", "yes"),
    ("Is helium lighter than air?", "yes"),
    ("Does salt dissolve in water?", "yes"),
    ("Is diamond the hardest natural substance?", "yes"),
    ("Does electricity flow through rubber?", "no"),
    ("Is the Sun a star?", "yes"),
    ("Is Venus closer to the Sun than Earth?", "yes"),
    ("Does copper conduct electricity?", "yes"),
    ("Is aluminum lighter than iron?", "yes"),
    ("Are there more atoms in the universe than grains of sand on Earth?", "yes"),
    ("Does wood float on water?", "yes"),
    ("Is mercury a liquid at room temperature?", "yes"),
    ("Is the Moon larger than Earth?", "no"),
    ("Does nitrogen gas make up most of the Earth's atmosphere?", "yes"),
    ("Is gold softer than iron?", "yes"),
    ("Do electrons have a negative charge?", "yes"),
    ("Is the speed of sound faster in water than air?", "yes"),
    ("Is carbon dioxide heavier than air?", "yes"),
    ("Is platinum rarer than gold?", "yes"),
    ("Do all metals conduct electricity?", "yes"),
    ("Is seawater denser than freshwater?", "yes"),
    ("Does copper rust?", "no"),
    ("Is the Earth's core made of iron?", "yes"),
]


# ─── Sample generators ────────────────────────────────────────────────────────

def density_pair() -> dict | None:
    a, b = random.sample(list(DENSITIES.keys()), 2)
    da, db = DENSITIES[a], DENSITIES[b]
    if da == db:
        return None
    denser = a if da > db else b
    lighter = b if da > db else a
    templates = [
        f"Which is denser: {a} or {b}?\nAnswer: {denser}",
        f"Which is lighter: {a} or {b}?\nAnswer: {lighter}",
        f"Is {a} denser than {b}?\nAnswer: {'yes' if da > db else 'no'}",
        f"Between {a} and {b}, which floats higher?\nAnswer: {lighter}",
        f"Which has more mass per cubic centimeter: {a} or {b}?\nAnswer: {denser}",
    ]
    text = random.choice(templates)
    return {"text": text, "answer": text.split("Answer: ")[1].strip()}


def speed_pair() -> dict | None:
    a, b = random.sample(list(SPEEDS.keys()), 2)
    sa, sb = SPEEDS[a], SPEEDS[b]
    if sa == sb:
        return None
    faster = a if sa > sb else b
    slower = b if sa > sb else a
    templates = [
        f"Which travels faster: {a} or {b}?\nAnswer: {faster}",
        f"Which is slower: {a} or {b}?\nAnswer: {slower}",
        f"Is {a} faster than {b}?\nAnswer: {'yes' if sa > sb else 'no'}",
    ]
    text = random.choice(templates)
    return {"text": text, "answer": text.split("Answer: ")[1].strip()}


def melting_pair() -> dict | None:
    a, b = random.sample(list(MELTING.keys()), 2)
    ma, mb = MELTING[a], MELTING[b]
    if ma == mb:
        return None
    hotter = a if ma > mb else b
    cooler = b if ma > mb else a
    templates = [
        f"Which has a higher melting point: {a} or {b}?\nAnswer: {hotter}",
        f"Which melts at a lower temperature: {a} or {b}?\nAnswer: {cooler}",
        f"Does {a} melt before {b} when heated?\nAnswer: {'yes' if ma < mb else 'no'}",
        f"Which requires more heat to melt: {a} or {b}?\nAnswer: {hotter}",
    ]
    text = random.choice(templates)
    return {"text": text, "answer": text.split("Answer: ")[1].strip()}


def atomic_pair() -> dict | None:
    a, b = random.sample(list(ATOMIC_NUMBERS.keys()), 2)
    na, nb = ATOMIC_NUMBERS[a], ATOMIC_NUMBERS[b]
    if na == nb:
        return None
    higher = a if na > nb else b
    lower = b if na > nb else a
    templates = [
        f"Which element has a higher atomic number: {a} or {b}?\nAnswer: {higher}",
        f"Which element has a lower atomic number: {a} or {b}?\nAnswer: {lower}",
        f"Does {a} have more protons than {b}?\nAnswer: {'yes' if na > nb else 'no'}",
    ]
    text = random.choice(templates)
    return {"text": text, "answer": text.split("Answer: ")[1].strip()}


def capital_q() -> dict:
    country = random.choice(list(CAPITALS.keys()))
    templates = [
        f"What is the capital of {country}?\nAnswer: {CAPITALS[country]}",
        f"Name the capital city of {country}.\nAnswer: {CAPITALS[country]}",
    ]
    text = random.choice(templates)
    return {"text": text, "answer": CAPITALS[country]}


def planet_size_pair() -> dict | None:
    a, b = random.sample(list(PLANET_SIZES_KM.keys()), 2)
    if PLANET_SIZES_KM[a] == PLANET_SIZES_KM[b]:
        return None
    larger = a if PLANET_SIZES_KM[a] > PLANET_SIZES_KM[b] else b
    smaller = b if PLANET_SIZES_KM[a] > PLANET_SIZES_KM[b] else a
    templates = [
        f"Which planet is larger: {a} or {b}?\nAnswer: {larger}",
        f"Which planet is smaller: {a} or {b}?\nAnswer: {smaller}",
        f"Is {a} bigger than {b}?\nAnswer: {'yes' if PLANET_SIZES_KM[a] > PLANET_SIZES_KM[b] else 'no'}",
    ]
    text = random.choice(templates)
    return {"text": text, "answer": text.split("Answer: ")[1].strip()}


def planet_distance_pair() -> dict | None:
    a, b = random.sample(list(PLANET_DISTANCES_M_KM.keys()), 2)
    da, db = PLANET_DISTANCES_M_KM[a], PLANET_DISTANCES_M_KM[b]
    if da == db:
        return None
    closer = a if da < db else b
    farther = b if da < db else a
    templates = [
        f"Which planet is closer to the Sun: {a} or {b}?\nAnswer: {closer}",
        f"Which planet is farther from the Sun: {a} or {b}?\nAnswer: {farther}",
        f"Is {a} closer to the Sun than {b}?\nAnswer: {'yes' if da < db else 'no'}",
    ]
    text = random.choice(templates)
    return {"text": text, "answer": text.split("Answer: ")[1].strip()}


def chem_symbol_q() -> dict:
    element = random.choice(list(CHEM_SYMBOLS.keys()))
    symbol = CHEM_SYMBOLS[element]
    templates = [
        f"What is the chemical symbol for {element}?\nAnswer: {symbol}",
        f"Write the chemical symbol of {element}.\nAnswer: {symbol}",
    ]
    text = random.choice(templates)
    return {"text": text, "answer": symbol}


def science_yn() -> dict:
    q, a = random.choice(SCIENCE_YN)
    return {"text": f"{q}\nAnswer: {a}", "answer": a}


def transitive_density() -> dict | None:
    items = random.sample(list(DENSITIES.keys()), 3)
    a, b, c = sorted(items, key=lambda x: DENSITIES[x], reverse=True)
    templates = [
        (f"{a} is denser than {b}, and {b} is denser than {c}.\nWhich is least dense?\nAnswer: {c}", c),
        (f"{a} is denser than {b}, and {b} is denser than {c}.\nWhich is most dense?\nAnswer: {a}", a),
        (f"If {a} > {b} > {c} in density, which is in the middle?\nAnswer: {b}", b),
        (f"{a} sinks below {b}, and {b} sinks below {c}.\nWhich floats highest?\nAnswer: {c}", c),
    ]
    text, answer = random.choice(templates)
    return {"text": text, "answer": answer}


def boiling_pair() -> dict | None:
    a, b = random.sample(list(BOILING.keys()), 2)
    ba, bb = BOILING[a], BOILING[b]
    if ba == bb:
        return None
    higher = a if ba > bb else b
    templates = [
        f"Which has a higher boiling point: {a} or {b}?\nAnswer: {higher}",
        f"Does {a} boil at a higher temperature than {b}?\nAnswer: {'yes' if ba > bb else 'no'}",
    ]
    text = random.choice(templates)
    return {"text": text, "answer": text.split("Answer: ")[1].strip()}


# ─── Build ────────────────────────────────────────────────────────────────────

GENERATORS = [
    (density_pair, 6000),
    (speed_pair, 2000),
    (melting_pair, 2000),
    (atomic_pair, 2000),
    (capital_q, 2000),
    (planet_size_pair, 1000),
    (planet_distance_pair, 1000),
    (chem_symbol_q, 1000),
    (science_yn, 1000),
    (transitive_density, 3000),
    (boiling_pair, 1000),
]


def build() -> list[dict]:
    all_samples = []
    for fn, target in GENERATORS:
        count = 0
        attempts = 0
        while count < target and attempts < target * 50:
            s = fn()
            attempts += 1
            if s is not None:
                all_samples.append(s)
                count += 1
        print(f"  {fn.__name__:25s}: {count:5d}")

    # Add math
    math = gen_math_samples(3000)
    all_samples.extend(math)
    print(f"  {'gen_math_samples':25s}: {len(math):5d}")

    random.shuffle(all_samples)
    return all_samples


def main() -> None:
    print("Building backbone-facts V3 dataset...\n")
    samples = build()

    # Dedup
    seen = set()
    unique = []
    for s in samples:
        if s["text"] not in seen:
            seen.add(s["text"])
            unique.append(s)
    print(f"\nRaw: {len(samples)} | Unique: {len(unique)} ({len(samples)-len(unique)} dupes)")

    missing = sum(1 for s in unique if "Answer: " not in s["text"])
    print(f"Missing Answer: boundary: {missing}")

    out = Path("backbone_facts_v3.json")
    with open(out, "w") as f:
        json.dump(unique, f, indent=2)
    print(f"Saved → {out} ({len(unique)} samples)")

    print("\nExamples:")
    for s in random.sample(unique, 5):
        print(f"  {s['text'].strip()!r}")
        print(f"  → {s['answer']!r}\n")


if __name__ == "__main__":
    main()
