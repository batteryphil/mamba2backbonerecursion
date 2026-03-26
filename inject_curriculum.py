"""
inject_curriculum.py — High-Hop RoPE Anchoring
================================================
Injects 6-to-10 hop reasoning chains into the clean dataset
to force the Mamba-2 RoPE embeddings to generalize to deeper loops.

Input:  system2_logic_v2_clean.json
Output: system2_logic_v3_curriculum.json
"""

import json
import random
import os

# Safe variables (Avoiding A, B, C, D to respect Stage 2 cleaning)
VARS = ["P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        "M", "N", "K", "L", "H", "J"]
VALUES = ["red", "blue", "green", "yellow", "purple", "orange",
          "black", "white", "apple", "grape", "lemon", "cherry",
          "iron", "gold", "silver"]


def generate_deep_chain(hops: int) -> dict:
    """Generate a shuffled pointer-tracking task for deep latent routing.

    Example (3-hop): "Z = Y. X = blue. Y = X. What is Z?"

    Args:
        hops: number of variable hops

    Returns:
        dict with text, answer, hops fields
    """
    # Grab enough unique variables for the requested hop count
    selected_vars = random.sample(VARS, hops + 1)
    start_var = selected_vars[0]
    target_val = random.choice(VALUES)

    # 1. Create the assignment statements
    statements = [f"{start_var} = {target_val}."]
    for i in range(1, len(selected_vars)):
        statements.append(f"{selected_vars[i]} = {selected_vars[i-1]}.")

    # 2. Shuffle the statements
    # We scramble the input so the model CANNOT solve it sequentially.
    # It must pull the statements into its latent space and recursively sort them.
    random.shuffle(statements)

    question = f"What is {selected_vars[-1]}?"
    prompt = " ".join(statements) + f" {question}"

    # 3. Build the strict reasoning chain format
    steps = [f"{start_var} = {target_val}"]
    for i in range(1, len(selected_vars)):
        steps.append(f"{selected_vars[i]} ← {selected_vars[i-1]} = {target_val}")

    chain_str = "; ".join(steps)
    answer_block = f"[Reasoning] {chain_str} [Answer] {target_val} <HALT>"

    # 4. Package exactly like v2_clean to prevent data leakage
    return {
        "text": f"{prompt}\nAnswer: ",
        "answer": answer_block,
        "hops": hops,
        "_is_synthetic_curriculum": True,
    }


def run_injection(
    input_path: str = "system2_logic_v2_clean.json",
    output_path: str = "system2_logic_v3_curriculum.json",
) -> None:
    """Run the curriculum injection pipeline.

    Args:
        input_path: path to cleaned v2 dataset
        output_path: path for augmented v3 dataset
    """
    print(f"\n{'='*50}")
    print("  ROPE CURRICULUM INJECTION PIPELINE")
    print(f"{'='*50}\n")

    # Load existing clean data
    if not os.path.exists(input_path):
        print(f"[!] ERROR: Could not find {input_path}")
        return

    with open(input_path, "r") as f:
        data = json.load(f)

    base_count = len(data)
    print(f"[*] Loaded {base_count} existing entries from {input_path}")

    # Generate the high-hop curriculum
    # (200 of each depth to gently anchor the embeddings)
    hop_targets = [6, 7, 8, 9, 10]
    entries_per_hop = 200

    new_entries = []
    for hops in hop_targets:
        for _ in range(entries_per_hop):
            new_entries.append(generate_deep_chain(hops))

    print(f"[*] Synthesized {len(new_entries)} deep-reasoning chains (6-10 hops).")

    # Shuffle the synthetic data smoothly into the existing dataset
    combined_data = data + new_entries
    random.shuffle(combined_data)

    # Save
    with open(output_path, "w") as f:
        json.dump(combined_data, f, indent=2)

    print(f"[*] Saved {len(combined_data)} total entries to {output_path}")
    print(f"    - Base: {base_count}")
    print(f"    - Injected: {len(new_entries)}")

    # Show hop distribution
    hop_dist: dict[int, int] = {}
    for e in combined_data:
        h = e.get("hops", 0)
        hop_dist[h] = hop_dist.get(h, 0) + 1
    print(f"    - Hop dist: {dict(sorted(hop_dist.items()))}")

    print(f"\n[+] Injection Complete. Ready for training.\n")


if __name__ == "__main__":
    run_injection()
