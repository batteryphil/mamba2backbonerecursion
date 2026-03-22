"""
logic_v7_generator.py
====================
Generates a large, maximally diverse reasoning dataset (logic_v7.json).
Targets all failure modes identified in v6 deep-dive:
  - Inversion (pick object not subject)
  - Passive voice ("B is outweighed by A")
  - OOD vocabulary (sci-fi / fantasy / international names)
  - 3-var and 4-var chains with scrambled premise order
  - Multiple domains (height, weight, speed, cost, age, temp, brightness, …)
  - QA extraction (context-based factual retrieval)
  - Negation ("not as tall as")
  - Multi-hop QA
  - Generic English regularization sentences (prevents domain squeeze)

Target: ~75,000 unique samples written to logic_v7.json
"""

import json
import random
import itertools

random.seed(42)

# ── Name Pools ────────────────────────────────────────────────────────────────
STANDARD_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank",
    "Iris", "Jack", "Karen", "Leo", "Mia", "Nick", "Olivia", "Paul",
    "Quinn", "Rachel", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xander",
    "Yara", "Zoe", "Aaron", "Beth", "Carl", "Dana", "Earl", "Fiona",
]
SCIFI_NAMES = [
    "Zorblax", "Quibble", "Xenthor", "Vela", "Drix", "Krynn", "Nurev",
    "Thex", "Omnix", "Phalaz", "Quasar", "Relvon", "Sylvex", "Talon",
    "Umbra", "Vorax", "Wraith", "Xarion", "Yendor", "Zephon", "Aeko",
    "Belrix", "Cyrex", "Delara", "Elox", "Fyron", "Galvex", "Hydra",
]
FANTASY_NAMES = [
    "Elindra", "Torgen", "Seraphine", "Malachor", "Nyxara", "Draveth",
    "Caelan", "Lysara", "Oberon", "Pyreth", "Quilara", "Ryndor",
    "Sylvara", "Thaleon", "Ulvara", "Vexis", "Wyndrel", "Xylara",
    "Ystara", "Zondrel", "Aelindra", "Borveth", "Calyra", "Daelion",
]
INTERNATIONAL_NAMES = [
    "Hiroshi", "Amara", "Sven", "Kofi", "Priya", "Mateus", "Yuki",
    "Ibrahim", "Fatima", "Luca", "Sonja", "Tariq", "Mei", "Andrei",
    "Nadia", "Olusegun", "Hana", "Dmitri", "Aisha", "Kwame", "Ingrid",
    "Rodrigo", "Sakura", "Tsegay", "Valentina", "Wanjiru", "Xiulan",
]
OBJECT_NAMES = [
    "Box A", "Box B", "Box C", "Box D",
    "Item A", "Item B", "Item C", "Item D",
    "Package 1", "Package 2", "Package 3", "Package 4",
    "Block X", "Block Y", "Block Z",
    "Sphere P", "Sphere Q", "Sphere R",
]
PLANET_NAMES = [
    "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune",
    "Proxima", "Kepler", "Andor", "Tatooine", "Vulcan", "Rigel",
]
ALL_NAME_POOLS = [STANDARD_NAMES, SCIFI_NAMES, FANTASY_NAMES,
                  INTERNATIONAL_NAMES, OBJECT_NAMES, PLANET_NAMES]

def sample_names(n: int) -> list[str]:
    """Pick n unique names from a random pool."""
    pool = random.choice(ALL_NAME_POOLS)
    return random.sample(pool, min(n, len(pool)))


# ── Domains ───────────────────────────────────────────────────────────────────
# Each domain: (adjective_more, adjective_less, noun_form, passive_verb)
DOMAINS = [
    ("taller",       "shorter",     "height",      "surpassed in height by"),
    ("heavier",      "lighter",     "weight",      "outweighed by"),
    ("faster",       "slower",      "speed",       "outpaced by"),
    ("older",        "younger",     "age",         "outlived in seniority by"),
    ("smarter",      "less smart",  "intelligence","outranked in IQ by"),
    ("stronger",     "weaker",      "strength",    "overpowered by"),
    ("richer",       "poorer",      "wealth",      "outearned by"),
    ("brighter",     "dimmer",      "brightness",  "outshone by"),
    ("louder",       "quieter",     "volume",      "outdecibeled by"),
    ("hotter",       "cooler",      "temperature", "outtemped by"),
    ("longer",       "shorter",     "length",      "surpassed in length by"),
    ("larger",       "smaller",     "size",        "exceeded in size by"),
    ("more expensive","cheaper",    "cost",        "undercut in expense by"),
    ("more popular", "less popular","popularity",  "outranked in fame by"),
    ("more distant", "closer",      "distance",    "outflanked in distance by"),
]


# ── Template helpers ──────────────────────────────────────────────────────────
def _direct_2var(a: str, b: str, more: str, less: str, **_) -> list[dict]:
    """A is MORE than B. Who is LESS? → B  AND  Who is MORE? → A"""
    samples = []
    templates_less = [
        f"{a} is {more} than {b}. Who is {less}? Answer: {b}",
        f"Given that {a} is {more} than {b}, who is the {less} one? Answer: {b}",
        f"{a} beats {b} in {_['noun']}. Who has less {_['noun']}? Answer: {b}",
        f"Between {a} and {b}, {a} is {more}. Who is {less}? Answer: {b}",
        f"Comparison: {a} > {b} ({_['noun']}). Who ranks lower? Answer: {b}",
        f"In terms of {_['noun']}, {a} exceeds {b}. Who is {less}? Answer: {b}",
        f"Ranking by {_['noun']}: {a} first, {b} second. Who is {less}? Answer: {b}",
    ]
    templates_more = [
        f"{a} is {more} than {b}. Who is {more}? Answer: {a}",
        f"Given that {a} is {more} than {b}, who is the {more} one? Answer: {a}",
        f"{a} beats {b} in {_['noun']}. Who has more {_['noun']}? Answer: {a}",
        f"Between {a} and {b}, {b} is {less}. Who is {more}? Answer: {a}",
        f"Ranking by {_['noun']}: {a} first, {b} second. Who is {more}? Answer: {a}",
    ]
    samples += [{"text": t} for t in templates_less + templates_more]
    return samples


def _passive_2var(a: str, b: str, more: str, less: str, passive: str, noun: str, **_) -> list[dict]:
    """Passive voice: B is outweighed by A. Who is heavier? → A"""
    templates = [
        f"{b} is {passive} {a}. Who is {more}? Answer: {a}",
        f"{b} is {passive} {a}. Who is {less}? Answer: {b}",
        f"In terms of {noun}, {b} is bested by {a}. Who wins? Answer: {a}",
        f"{b} is beaten by {a} in {noun}. Which one is {more}? Answer: {a}",
        f"When it comes to {noun}, {a} dominates {b}. Who is {less}? Answer: {b}",
    ]
    return [{"text": t} for t in templates]


def _negation_2var(a: str, b: str, more: str, less: str, noun: str, **_) -> list[dict]:
    """Negation: B is not as tall as A. Who is taller? → A"""
    templates = [
        f"{b} is not as {more[:-2] if more.endswith('er') else more} as {a}. Who is {more}? Answer: {a}",
        f"{b} is not {more} than {a}. Who is {more}? Answer: {a}",
        f"It is false that {b} is {more} than {a}. Who is {more}? Answer: {a}",
        f"{a} is not {less} than {b}. Who is {less}? Answer: {b}",
        f"In {noun}, {b} does not surpass {a}. Who is {more}? Answer: {a}",
    ]
    return [{"text": t} for t in templates]


def _3var_chain(a: str, b: str, c: str, more: str, less: str, noun: str, **_) -> list[dict]:
    """a > b > c chain with many surface forms and question variants."""
    templates = [
        # Inline chain
        f"{a} is {more} than {b}, and {b} is {more} than {c}. Who is {less}? Answer: {c}",
        f"{a} is {more} than {b}, and {b} is {more} than {c}. Who is {more}? Answer: {a}",
        f"{a} > {b} > {c} in {noun}. Who is {less}? Answer: {c}",
        f"{a} > {b} > {c} in {noun}. Who is {more}? Answer: {a}",
        f"Ranked by {noun}: {a}, {b}, {c}. Who ranks lowest? Answer: {c}",
        f"Ranked by {noun}: {a}, {b}, {c}. Who ranks highest? Answer: {a}",
        # Scrambled premises
        f"{b} is {more} than {c}. Also, {a} is {more} than {b}. Who is {less}? Answer: {c}",
        f"{b} is {more} than {c}. {a} beats {b} in {noun}. Who is {more}? Answer: {a}",
        f"We know: {a} > {b} in {noun}, and {b} > {c} in {noun}. Who is least {noun}? Answer: {c}",
        # Middle entity
        f"{a} is {more} than {b}, and {b} is {more} than {c}. Who is in the middle? Answer: {b}",
        f"Ranked by {noun}: {a}, {b}, {c}. Who is second? Answer: {b}",
        # Inversion questions on chain
        f"{a} is {more} than {b}, and {b} is {more} than {c}. Is {c} the {less}? Answer: Yes",
        f"{a} is {more} than {b}, and {b} is {more} than {c}. Is {a} the {less}? Answer: No",
    ]
    return [{"text": t} for t in templates]


def _4var_chain(a: str, b: str, c: str, d: str, more: str, less: str, noun: str, **_) -> list[dict]:
    """a > b > c > d chain."""
    templates = [
        f"{a} > {b} > {c} > {d} in {noun}. Who is {less}? Answer: {d}",
        f"{a} > {b} > {c} > {d} in {noun}. Who is {more}? Answer: {a}",
        f"Ranked by {noun}: {a}, {b}, {c}, {d}. Who ranks last? Answer: {d}",
        f"Ranked by {noun}: {a}, {b}, {c}, {d}. Who ranks first? Answer: {a}",
        f"Ranked by {noun}: {a}, {b}, {c}, {d}. Who is third? Answer: {c}",
        f"{a} is {more} than {b}. {b} is {more} than {c}. {c} is {more} than {d}. Who is {less}? Answer: {d}",
        f"{b} is {more} than {c}. {c} is {more} than {d}. {a} is {more} than {b}. Who is {more}? Answer: {a}",
        f"Among {a}, {b}, {c}, {d} ranked by {noun} descending, who is at the bottom? Answer: {d}",
    ]
    return [{"text": t} for t in templates]


# ── QA Extraction ─────────────────────────────────────────────────────────────
QA_TEMPLATES = [
    ("password",   "The password is {val}.",          "What is the password?",            "{val}"),
    ("code",       "The secret code is {val}.",        "What is the secret code?",         "{val}"),
    ("capital",    "The capital city is {val}.",       "What is the capital?",             "{val}"),
    ("color",      "The correct color is {val}.",      "What is the correct color?",       "{val}"),
    ("number",     "The lucky number is {val}.",       "What is the lucky number?",        "{val}"),
    ("key",        "The encryption key is {val}.",     "What is the encryption key?",      "{val}"),
    ("name",       "The winner's name is {val}.",      "What is the winner's name?",       "{val}"),
    ("location",   "The meeting location is {val}.",   "Where is the meeting?",            "{val}"),
    ("date",       "The event date is {val}.",         "When is the event?",               "{val}"),
    ("formula",    "The compound formula is {val}.",   "What is the formula?",             "{val}"),
]
QA_VALUES = [
    "Zephyr-7", "Omega-9", "Delta-3", "Phoenix-1", "Titan-X",
    "Paris", "Tokyo", "Cairo", "Oslo", "Lima",
    "Red", "Blue", "Emerald", "Crimson", "Violet",
    "42", "17", "99", "256", "1024",
    "Alpha-7", "Nexus-4", "Sigma-3",
    "H2O", "CO2", "NaCl", "C6H12O6",
    "Sunday", "March 7th", "2099", "Midnight",
]
QA_CTX_TEMPLATES = [
    "Context: [{fact}] Question: {question} Answer: {answer}",
    "System: [{fact}] Query: {question} Answer: {answer}",
    "[INFO: {fact}] Q: {question} A: {answer}",
    "Given: [{fact}] — {question} Answer: {answer}",
    "Memo: {fact} | Question: {question} | Answer: {answer}",
    "Background: {fact}. Now answer: {question} Answer: {answer}",
]

def _qa_samples(n: int = 3000) -> list[dict]:
    samples = []
    for _ in range(n):
        label, fact_tpl, q, ans_tpl = random.choice(QA_TEMPLATES)
        val = random.choice(QA_VALUES)
        fact = fact_tpl.format(val=val)
        answer = ans_tpl.format(val=val)
        ctx_tpl = random.choice(QA_CTX_TEMPLATES)
        text = ctx_tpl.format(fact=fact, question=q, answer=answer)
        samples.append({"text": text})
    return samples


# ── Multi-hop QA ──────────────────────────────────────────────────────────────
def _multihop_samples(n: int = 1500) -> list[dict]:
    samples = []
    for _ in range(n):
        names = sample_names(3)
        a, b, c = names[0], names[1], names[2]
        val = random.choice(QA_VALUES)
        templates = [
            f"Context: [{a}'s code is {val}. {b} uses {a}'s code.] Question: What is {b}'s code? Answer: {val}",
            f"Context: [{a} owns key {val}. {b} inherited {a}'s key.] Question: What key does {b} have? Answer: {val}",
            f"[{a} knows {val}. {b} learned it from {a}. {c} learned it from {b}.] What does {c} know? Answer: {val}",
            f"System: [{a}'s password is {val}. {b}'s password matches {a}'s.] What is {b}'s password? Answer: {val}",
        ]
        samples.append({"text": random.choice(templates)})
    return samples


# ── Generic English Regularization ───────────────────────────────────────────
GENERIC_QA = [
    "The mitochondria is the powerhouse of the cell. What generates cellular energy? Answer: Mitochondria",
    "Water boils at 100 degrees Celsius at sea level. At what temperature does water boil? Answer: 100 degrees Celsius",
    "The speed of light is approximately 299,792 kilometers per second. How fast does light travel? Answer: 299,792 km/s",
    "DNA carries genetic information in living organisms. What molecule stores genetic data? Answer: DNA",
    "Photosynthesis converts sunlight into chemical energy in plants. What process do plants use to make food? Answer: Photosynthesis",
    "The Amazon River is the largest river by discharge volume. Which river has the greatest discharge? Answer: Amazon River",
    "Gravity pulls objects toward the Earth's center. What force attracts objects to Earth? Answer: Gravity",
    "Neurons transmit signals in the nervous system. What cells carry nerve signals? Answer: Neurons",
    "The Pacific Ocean is the largest ocean on Earth. Which ocean covers the most area? Answer: The Pacific Ocean",
    "Hydrogen is the lightest element on the periodic table. What is the lightest element? Answer: Hydrogen",
    "The heart pumps blood through the circulatory system. What organ circulates blood? Answer: The heart",
    "Sound travels faster in water than in air. In which medium does sound travel faster? Answer: Water",
    "The moon orbits Earth approximately every 27 days. How long does it take the moon to orbit Earth? Answer: About 27 days",
    "Rome is the capital of Italy. What is the capital of Italy? Answer: Rome",
    "Diamonds are the hardest natural substance known. What is the hardest natural material? Answer: Diamond",
    "The Earth rotates on its axis once every 24 hours. How long is one full Earth rotation? Answer: 24 hours",
    "Plants absorb carbon dioxide and release oxygen. What gas do plants release during photosynthesis? Answer: Oxygen",
    "The Great Wall of China stretches over 13,000 miles. How long is the Great Wall of China? Answer: Over 13,000 miles",
    "Isaac Newton formulated the laws of motion and gravity. Who developed the laws of motion? Answer: Isaac Newton",
    "The human body has 206 bones. How many bones does the adult human body have? Answer: 206",
    "Gold has the chemical symbol Au. What is the symbol for gold? Answer: Au",
    "The brain controls all bodily functions. Which organ is responsible for bodily control? Answer: The brain",
    "Mercury is the closest planet to the Sun. Which planet orbits nearest to the Sun? Answer: Mercury",
    "Volcanoes form at tectonic plate boundaries. Where do most volcanoes form? Answer: At tectonic plate boundaries",
    "The human heart beats about 60-100 times per minute. How often does the heart beat per minute? Answer: 60-100 times",
]

def _generic_samples(n: int = 3000) -> list[dict]:
    out = []
    for _ in range(n):
        base = random.choice(GENERIC_QA)
        # Add surface variation
        variations = [
            base,
            base,   # Duplicate weight — generic text is rare
        ]
        out.append({"text": random.choice(variations)})
    return out


# ── Main generation ───────────────────────────────────────────────────────────
def generate_dataset(target: int = 75000) -> list[dict]:
    """Generate a maximally diverse reasoning dataset."""
    all_samples: list[dict] = []

    print("Generating 2-var comparisons (direct, passive, negation)...")
    for domain in DOMAINS:
        more, less, noun, passive = domain
        for _ in range(200):
            a, b = sample_names(2)
            kwargs = dict(a=a, b=b, more=more, less=less, noun=noun, passive=passive)
            all_samples += _direct_2var(**kwargs)
            all_samples += _passive_2var(**kwargs)
            all_samples += _negation_2var(**kwargs)
    print(f"  After 2-var: {len(all_samples):,}")

    print("Generating 3-var chains...")
    for domain in DOMAINS:
        more, less, noun, passive = domain
        for _ in range(180):
            a, b, c = sample_names(3)
            all_samples += _3var_chain(a=a, b=b, c=c,
                                       more=more, less=less, noun=noun, passive=passive)
    print(f"  After 3-var: {len(all_samples):,}")

    print("Generating 4-var chains...")
    for domain in DOMAINS:
        more, less, noun, passive = domain
        for _ in range(120):
            a, b, c, d = sample_names(4)
            all_samples += _4var_chain(a=a, b=b, c=c, d=d,
                                       more=more, less=less, noun=noun, passive=passive)
    print(f"  After 4-var: {len(all_samples):,}")

    print("Generating QA extraction...")
    all_samples += _qa_samples(5000)
    print(f"  After QA: {len(all_samples):,}")

    print("Generating multi-hop QA...")
    all_samples += _multihop_samples(2500)
    print(f"  After multi-hop: {len(all_samples):,}")

    print("Generating generic English regularization...")
    all_samples += _generic_samples(4000)
    print(f"  After generic: {len(all_samples):,}")

    # Deduplicate
    seen: set = set()
    unique: list = []
    for s in all_samples:
        key = s["text"].strip()
        if key not in seen:
            seen.add(key)
            unique.append(s)
    print(f"\n  Raw: {len(all_samples):,}  →  Unique: {len(unique):,} "
          f"(removed {len(all_samples)-len(unique):,} dupes)")

    random.shuffle(unique)

    if len(unique) > target:
        unique = unique[:target]
        print(f"  Trimmed to target: {len(unique):,}")

    return unique


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  Logic v7 Dataset Generator — Maximum Diversity Edition")
    print(f"{'='*60}\n")

    dataset = generate_dataset()

    out_path = "/home/phil/Desktop/mambadiff/mambadiff llm tts/logic_v7.json"
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=None)

    print(f"\n✅ Saved {len(dataset):,} samples → {out_path}")
    print(f"   File size: {__import__('os').path.getsize(out_path) / 1e6:.1f} MB")

    # Quick sanity check - sample 10 random
    print(f"\n  Sample entries:")
    for item in random.sample(dataset, 10):
        print(f"    {repr(item['text'][:90])}")
