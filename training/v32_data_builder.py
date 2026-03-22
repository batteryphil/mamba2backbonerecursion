"""
v32_data_builder.py — Random-Vocabulary Chain Data Generator
=============================================================
Generates chain training data using RANDOM, RARE words from the full 50k
tokenizer vocabulary. This forces the model to learn a universal COPY
operation rather than relying on semantic priors.

Key design decisions (from v31 failure analysis):
  1. Random vocab: uses words from ~5k-40k token range (diverse, not top-100)
  2. Flat hop distribution: exactly equal counts of 1/2/3/4/5-hop chains
  3. Values are drawn from a pool that changes every sample — no repetition bias
  4. Output: system2_logic_v32.json  (replaces system2_logic_v1.json for v32)
"""
import json
import random
import re
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>", "<HALT>"]})

TOTAL_SAMPLES    = 30_000   # 6k per hop level × 5 levels
HOPS_PER_LEVEL   = TOTAL_SAMPLES // 5  # 6,000 each
OUTPUT_FILE      = "system2_logic_v32.json"
SEED             = 42

# Variable names for chains — single uppercase letters + 2-letter combos
VAR_POOL = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [
    "AA", "BB", "CC", "DD", "EE", "FF", "GG", "HH", "II", "JJ",
    "KK", "LL", "MM", "NN", "PP", "QQ", "RR", "SS", "TT", "UU"]

def build_word_pool(tokenizer, min_id: int = 5000, max_id: int = 45000,
                    min_len: int = 3, max_len: int = 12,
                    pool_size: int = 8000) -> list[str]:
    """
    Sample clean single-token words from the middle of the vocab.
    The middle range (5k-45k) has real words without being dominated by
    ultra-common tokens (the/a/is) or garbage byte sequences.
    """
    words = []
    ids   = list(range(min_id, max_id))
    random.shuffle(ids)
    for tid in ids:
        tok = tokenizer.decode([tid])
        # Keep only clean alphabetic words (no BPE symbols, no spaces)
        clean = tok.strip()
        if (clean.isalpha()
                and min_len <= len(clean) <= max_len
                and clean.isascii()):
            words.append(clean)
        if len(words) >= pool_size:
            break
    print(f"  Word pool size: {len(words)} (ids {min_id}-{max_id})")
    return words


def make_chain(vars_: list[str], value: str, n_hops: int) -> str:
    """
    Build a chain prompt string.
    Example (3 hops): 'X = carburetor. Y = X. Z = Y. What is Z?\nAnswer:'
    """
    chain_vars = vars_[:n_hops + 1]  # need n_hops + 1 variables
    # First var gets the value, each subsequent points to the previous
    parts = [f"{chain_vars[0]} = {value}."]
    for i in range(1, len(chain_vars)):
        parts.append(f"{chain_vars[i]} = {chain_vars[i-1]}.")
    question = f"What is {chain_vars[-1]}?"
    return " ".join(parts) + f" {question}\nAnswer:"


def build_dataset(word_pool: list[str], hops_per_level: int,
                  rng: random.Random) -> list[dict]:
    """Generate hops_per_level samples for each of 1/2/3/4/5 hops."""
    samples = []
    for n_hops in range(1, 6):
        for _ in range(hops_per_level):
            # Pick a random value from the pool — rare word
            value = rng.choice(word_pool)
            # Pick n_hops+1 distinct variable names
            vars_ = rng.sample(VAR_POOL, n_hops + 1)
            text  = make_chain(vars_, value, n_hops)
            samples.append({
                "text":   text,
                "answer": value,
                "n_hops": n_hops,
            })
    rng.shuffle(samples)
    return samples


def verify_sample(sample: dict) -> bool:
    """Sanity check: answer appears in prompt and is tokenizable."""
    text   = sample["text"]
    answer = sample["answer"]
    if answer not in text:
        return False
    # Verify answer tokenizes to a single clean token (space-prefixed)
    ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    return len(ids) >= 1


if __name__ == "__main__":
    rng = random.Random(SEED)

    print(f"Building v32 random-vocab chain dataset...")
    print(f"  Target: {TOTAL_SAMPLES:,} samples ({HOPS_PER_LEVEL:,} per hop level)")
    print(f"  Hop levels: 1, 2, 3, 4, 5 (flat distribution)")
    print(f"  Seed: {SEED}")

    word_pool = build_word_pool(tokenizer)
    samples   = build_dataset(word_pool, HOPS_PER_LEVEL, rng)

    # Verify and report
    valid  = [s for s in samples if verify_sample(s)]
    bad    = len(samples) - len(valid)
    print(f"\n  Generated: {len(samples):,}")
    print(f"  Valid:     {len(valid):,}")
    print(f"  Filtered:  {bad:,} (answer not clean single token)")

    hop_dist = {}
    for s in valid:
        hop_dist[s["n_hops"]] = hop_dist.get(s["n_hops"], 0) + 1
    print(f"  Hop dist:  {dict(sorted(hop_dist.items()))}")

    # Show examples
    print(f"\n  Examples:")
    for n in range(1, 6):
        ex = next(s for s in valid if s["n_hops"] == n)
        print(f"    {n}-hop: {ex['text'][:80]!r}  → {ex['answer']!r}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(valid, f, indent=2)
    print(f"\n  ✅ Saved {len(valid):,} samples → {OUTPUT_FILE}")
