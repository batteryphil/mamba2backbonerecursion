"""
generate_phase3_data.py — Adversarial Curriculum Generator
====================================================================
Generates mixed-format out-of-distribution reasoning chains to force
the Mamba Dual-Architecture to generalize its Latent Bridge.
Formats: Variable Chaos, Semantic Prose, and Distractor Injection.
"""

import json
import random

COLORS = [
    "red", "blue", "quantum", "nebula", "titanium", "void",
    "crimson", "cipher", "omega", "sigma",
]
NAMES = [
    "Alice", "Bob", "Charlie", "David", "Eve", "Frank",
    "Grace", "Heidi", "Ivan", "Judy", "Mallory",
]
DISTRACTORS = [
    "The mitochondria is the powerhouse of the cell.",
    "Did you know that the Eiffel Tower grows in the summer?",
    "I really need to buy some milk and eggs today.",
    "Chicago is known for its deep-dish pizza and harsh winters.",
    "The stock market experienced a slight dip on Tuesday.",
    "Quantum entanglement allows particles to sync instantaneously.",
    "In 1969, humanity took its first steps on the lunar surface.",
]


def generate_variable_chaos(hops: int) -> dict:
    """Format 1: Destroys the single-letter dependency (V1, User_12, foo).

    Args:
        hops: Number of chain hops to generate

    Returns:
        Training sample dict with text, answer, type, and hops
    """
    val = random.choice(COLORS)
    var_names = [f"Var_{random.randint(10, 99)}" for _ in range(hops + 1)]

    # Mix up the assignment operators
    operators = ["=", "<-", "is set to", "equals"]

    text = f"{var_names[0]} {random.choice(operators)} {val}. "
    reasoning_steps: list[str] = []

    for i in range(1, len(var_names)):
        op = random.choice(operators)
        text += f"{var_names[i]} {op} {var_names[i-1]}. "
        reasoning_steps.append(f"{var_names[i]} ← {var_names[i-1]}")

    text += f"What is the value of {var_names[-1]}?\nAnswer: "
    reasoning = "; ".join(reasoning_steps)
    answer = f" [Reasoning] {reasoning} [Answer] {val} <HALT>"

    return {
        "text": text + answer,
        "answer": answer,
        "type": "variable_chaos",
        "hops": hops,
    }


def generate_semantic_prose(hops: int) -> dict:
    """Format 2: Destroys math syntax. Forces logic tracking in English prose.

    Args:
        hops: Number of chain hops to generate

    Returns:
        Training sample dict with text, answer, type, and hops
    """
    val = random.choice(COLORS)
    agents = random.sample(NAMES, min(hops + 1, len(NAMES)))

    # If we need more hops than we have names, generate Agent_X
    while len(agents) < hops + 1:
        agents.append(f"Agent_{random.randint(100, 999)}")

    text = (f"The secret vault password is '{val}'. "
            f"{agents[0]} memorizes the password. ")
    reasoning_steps: list[str] = []

    verbs = [
        "whispers it to", "texts it to", "passes the intel to",
        "briefs", "securely messages",
    ]

    for i in range(1, len(agents)):
        verb = random.choice(verbs)
        text += f"{agents[i-1]} {verb} {agents[i]}. "
        reasoning_steps.append(f"{agents[i]} ← {agents[i-1]}")

    text += (f"What is the password that {agents[-1]} received?\n"
             f"Answer: ")
    reasoning = "; ".join(reasoning_steps)
    answer = f" [Reasoning] {reasoning} [Answer] {val} <HALT>"

    return {
        "text": text + answer,
        "answer": answer,
        "type": "semantic_prose",
        "hops": hops,
    }


def generate_distractor_chain(hops: int) -> dict:
    """Format 3: Context Stretcher. Injects noise to train long-context.

    Args:
        hops: Number of chain hops to generate

    Returns:
        Training sample dict with text, answer, type, and hops
    """
    val = random.choice(COLORS)
    vars_l = [chr(65 + i) for i in range(hops + 1)]  # A, B, C...

    text = f"{vars_l[0]} = {val}. "
    reasoning_steps: list[str] = []

    for i in range(1, len(vars_l)):
        # Inject a distractor 30% of the time
        if random.random() < 0.30:
            text += random.choice(DISTRACTORS) + " "

        text += f"{vars_l[i]} = {vars_l[i-1]}. "
        reasoning_steps.append(f"{vars_l[i]} ← {vars_l[i-1]}")

    text += f"What is {vars_l[-1]}?\nAnswer: "
    reasoning = "; ".join(reasoning_steps)
    answer = f" [Reasoning] {reasoning} [Answer] {val} <HALT>"

    return {
        "text": text + answer,
        "answer": answer,
        "type": "distractor_chain",
        "hops": hops,
    }


def main() -> None:
    """Generate the Phase 3 adversarial curriculum."""
    dataset: list[dict] = []
    n_samples_per_format = 1000
    hop_range = range(5, 21)  # Train on 5 to 20 hops

    print("Generating Phase 3 Adversarial Curriculum...")

    for _ in range(n_samples_per_format):
        hops = random.choice(list(hop_range))
        dataset.append(generate_variable_chaos(hops))
        dataset.append(generate_semantic_prose(hops))
        dataset.append(generate_distractor_chain(hops))

    # Shuffle so the optimizer can't guess the format
    random.shuffle(dataset)

    output_file = "phase3_adversarial_curriculum.json"
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"✅ Generated {len(dataset)} samples saved to {output_file}")

    # Quick diagnostics
    print("\nDataset Composition:")
    print(f"  Variable Chaos:    {n_samples_per_format}")
    print(f"  Semantic Prose:    {n_samples_per_format}")
    print(f"  Distractor Chains: {n_samples_per_format}")
    print(f"  Hop Range: 5-20")

    # Token length stats
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    lengths = [len(tok.encode(s["text"], add_special_tokens=False))
               for s in dataset]
    print(f"\n  Token lengths:")
    print(f"    Min: {min(lengths)}")
    print(f"    Max: {max(lengths)}")
    print(f"    Mean: {sum(lengths)/len(lengths):.0f}")
    print(f"    Median: {sorted(lengths)[len(lengths)//2]}")
    n_over_128 = sum(1 for l in lengths if l > 128)
    n_over_256 = sum(1 for l in lengths if l > 256)
    n_over_512 = sum(1 for l in lengths if l > 512)
    print(f"    >128 tokens: {n_over_128} ({n_over_128*100/len(lengths):.1f}%)")
    print(f"    >256 tokens: {n_over_256} ({n_over_256*100/len(lengths):.1f}%)")
    print(f"    >512 tokens: {n_over_512} ({n_over_512*100/len(lengths):.1f}%)")


if __name__ == "__main__":
    main()
