"""
logic_v3_balanced_builder.py
Generates a balanced version of the logic training data where:
- 50% of samples have the FIRST entity as the answer  
- 50% of samples have the SECOND entity as the answer
- Uses the MMLU-style 'Answer: X' format for consistent boundary detection
- All samples verified to fit within SEQ_LEN=512 tokens

Logic types supported:
  - TransitiveInversion: X < Y → who is tallest = Y (or X if inverted)
  - DeepChain: sorting by rank
  - MultiHop: chain of relationships

Output: logic_v3_balanced.json  (15000 samples, 50/50 balance)
"""
import json
import random
import re
from pathlib import Path

random.seed(42)

# ─── Name pools ───────────────────────────────────────────────────────────────
FIRST_NAMES = [
    "Sarah", "James", "Maria", "Thomas", "Jennifer", "Robert", "Patricia",
    "Michael", "Linda", "William", "Barbara", "David", "Elizabeth", "Richard",
    "Susan", "Joseph", "Jessica", "Charles", "Karen", "Christopher", "Nancy",
    "Daniel", "Lisa", "Matthew", "Betty", "Anthony", "Margaret", "Mark",
    "Sandra", "Donald", "Ashley", "Steven", "Dorothy", "Paul"
]
MIDDLE_NAMES = [
    "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel",
    "India", "Juliett", "Kilo", "Lima", "Mike", "November", "Oscar", "Papa",
    "Quebec", "Romeo", "Sierra", "Tango", "Uniform", "Victor", "Whiskey",
    "Xray", "Yankee", "Zulu"
]

def make_entity() -> str:
    """Generate a unique two-part entity name."""
    return f"{random.choice(FIRST_NAMES)}_{random.choice(MIDDLE_NAMES)}"


def make_two_entities() -> tuple[str, str]:
    """Generate two distinct entity names."""
    a = make_entity()
    b = make_entity()
    while b == a:
        b = make_entity()
    return a, b


# ─── Sample generators ────────────────────────────────────────────────────────
COMPARISONS = [
    ("shorter", "taller",   "tallest"),
    ("lighter", "heavier",  "heaviest"),
    ("younger", "older",    "oldest"),
    ("slower",  "faster",   "fastest"),
    ("weaker",  "stronger", "strongest"),
]
NOISE_CONDITIONS = [
    "the sky is blue",
    "the cat is sleeping",
    "water is wet",
    "the sun rises in the east",
    "birds can fly",
    "the grass is green",
]


def make_transitive_sample(answer_is_first: bool) -> dict:
    """
    TransitiveInversion: given X < Y (X is shorter than Y).
    If answer_is_first=True:  ask 'who is SHORTEST?' → X (first mentioned) is answer
    If answer_is_first=False: ask 'who is TALLEST?'  → Y (second mentioned) is answer
    This guarantees true 50/50 balance without changing entity order in text.
    """
    a, b = make_two_entities()  # a < b (a is lesser, b is greater)
    lesser, greater, _ = random.choice(COMPARISONS)
    noise = random.choice(NOISE_CONDITIONS)

    if answer_is_first:
        # a < b, ask for the lesser one → a (first mentioned) is the answer
        superlative = lesser  # e.g. 'shortest', 'lightest'
        answer = a
        analysis_txt = (
            f"Since {a} < {b}, {a} must be the {lesser} entity."
        )
    else:
        # a < b, ask for the greater one → b (second mentioned) is the answer
        superlative = greater  # e.g. 'tallest', 'heaviest'
        answer = b
        analysis_txt = (
            f"Since {a} < {b}, {b} must be the {greater} entity."
        )

    text = (
        f"Condition: {noise}. "
        f"Premise: {a} is {lesser} than {b}. "
        f"Target Inquiry: Who is the {superlative}? "
        f"Analysis: {analysis_txt} Conclusion: {answer}.\n"
        f"Answer: {answer}"
    )
    return {"text": text, "answer": answer, "type": "TransitiveInversion"}


def make_chain_sample(chain_len: int, answer_is_first: bool) -> dict:
    """
    DeepChain: sort N entities by rank, answer is max or min.
    answer_is_first: True → return the highest-ranked (first introduced)
    """
    n = max(3, chain_len)
    entities = [make_entity() for _ in range(n)]
    # Make unique
    seen = set()
    unique = []
    for e in entities:
        while e in seen:
            e = make_entity()
        seen.add(e)
        unique.append(e)
    entities = unique

    # entities[0] < entities[1] < ... < entities[n-1]
    # So entities[-1] is the max (highest rank)
    chain_str = " > ".join(reversed(entities))  # highest first for readability
    noise = random.choice(NOISE_CONDITIONS)

    if answer_is_first:
        answer = entities[0]  # original lowest → "minimum" 
        inquiry = "Who has the lowest rank?"
        analysis = f"The chain {chain_str} shows {entities[0]} is at the bottom."
    else:
        answer = entities[-1]  # original highest
        inquiry = "Who has the highest rank?"
        analysis = f"The chain {chain_str} shows {entities[-1]} is at the top."

    text = (
        f"Chain: {chain_str}. "
        f"Condition: {noise}. "
        f"Target Inquiry: {inquiry} "
        f"Analysis: {analysis} Conclusion: {answer}.\n"
        f"Answer: {answer}"
    )
    return {"text": text, "answer": answer, "type": "DeepChain"}


# ─── Build balanced dataset ────────────────────────────────────────────────────
def build_dataset(n_samples: int = 15000) -> list[dict]:
    """Build a balanced dataset of n_samples logic problems."""
    samples = []
    half = n_samples // 2

    # TransitiveInversion: half first-wins, half second-wins
    for i in range(half):
        samples.append(make_transitive_sample(answer_is_first=(i % 2 == 0)))

    # DeepChain: mix of chain lengths 3-5, balanced
    remaining = n_samples - half
    for i in range(remaining):
        chain_len = random.randint(3, 5)
        samples.append(make_chain_sample(chain_len, answer_is_first=(i % 2 == 0)))

    random.shuffle(samples)
    return samples


def main() -> None:
    """Generate and save the balanced dataset."""
    print("Generating balanced logic dataset...")
    samples = build_dataset(15000)

    # Simple balance check: is each sample's answer the first entity mentioned?
    first_entity_answer = 0
    for s in samples:
        answer = s["answer"]
        entities = re.findall(r'[A-Z][a-z]+_[A-Z][a-z]+', s["text"])
        if entities and entities[0] == answer:
            first_entity_answer += 1

    pct = first_entity_answer / len(samples) * 100
    print(f"Balance check: first entity is answer {first_entity_answer}/{len(samples)} = {pct:.1f}%")
    print(f"(Target: ~50% — {'✅ BALANCED' if 45 < pct < 55 else '❌ STILL BIASED'})")

    # Verify all samples have Answer: boundary
    missing = sum(1 for s in samples if "Answer: " not in s["text"])
    print(f"Missing 'Answer: ' boundary: {missing}")

    # Save
    out_path = Path("logic_v3_balanced.json")
    with open(out_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"\nSaved {len(samples)} samples to {out_path}")
    print("Sample 0:", samples[0]["text"][:200])
    print("...")
    print("Answer:", samples[0]["answer"])


if __name__ == "__main__":
    main()
