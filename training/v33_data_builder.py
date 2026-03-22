"""
v33_data_builder.py — Single-Token Random Vocab + Reality Override Counterfactuals
====================================================================================
Three key upgrades from v32 analysis:

1. SINGLE-TOKEN FILTER: Only use words that tokenize to exactly ONE token.
   This eliminates the "carb" instead of "carburetor" issue and makes the
   model's success/failure unambiguous — no more partial credit edge cases.

2. FLAT HOP DISTRIBUTION: Equal counts 1/2/3/4/5 hops (same as v32).

3. REALITY OVERRIDE COUNTERFACTUALS: 3,000 procedurally generated counterfactual
   examples (e.g. "Fire is icy cold. Bob touched fire. What did Bob feel? cold")
   These give the model a gradient incentive to override its pretrained priors.
"""
import json
import random
import re
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>", "<HALT>"]})

CHAIN_SAMPLES    = 30_000   # 6k per hop × 5 levels
HOPS_PER_LEVEL   = CHAIN_SAMPLES // 5
OVERRIDE_SAMPLES = 3_000    # reality override counterfactuals
OUTPUT_FILE      = "system2_logic_v33.json"
SEED             = 42

VAR_POOL = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [
    "AA","BB","CC","DD","EE","FF","GG","HH","II","JJ",
    "KK","LL","MM","NN","PP","QQ","RR","SS","TT","UU",
    "X1","X2","X3","X4","X5","Y1","Y2","Z1","Z2","W1"]


def build_single_token_pool(tokenizer, min_id: int = 3000, max_id: int = 48000,
                             min_len: int = 3, max_len: int = 14,
                             pool_size: int = 12000) -> list[str]:
    """
    Build pool of words that tokenize to EXACTLY ONE token.
    This is the critical filter — eliminates carburetor→['carb','uretor'] splits.
    Uses space-prefixed encoding (real usage context for GPT-NeoX tokenizer).
    """
    words = []
    ids   = list(range(min_id, max_id))
    random.shuffle(ids)
    for tid in ids:
        raw = tokenizer.decode([tid])
        clean = raw.strip()
        if not (clean.isalpha() and min_len <= len(clean) <= max_len and clean.isascii()):
            continue
        # Verify single-token round-trip with space prefix (real usage)
        re_enc = tokenizer.encode(" " + clean, add_special_tokens=False)
        if len(re_enc) == 1 and re_enc[0] == tid:
            words.append(clean)
        if len(words) >= pool_size:
            break
    print(f"  Single-token word pool: {len(words)} words "
          f"(filtered from ids {min_id}-{max_id})")
    return words


def make_chain(vars_: list[str], value: str, n_hops: int) -> str:
    """Build chain prompt. E.g.: 'X = democracy. Y = X. Z = Y. What is Z?\nAnswer:'"""
    chain_vars = vars_[:n_hops + 1]
    parts = [f"{chain_vars[0]} = {value}."]
    for i in range(1, len(chain_vars)):
        parts.append(f"{chain_vars[i]} = {chain_vars[i-1]}.")
    return " ".join(parts) + f" What is {chain_vars[-1]}?\nAnswer:"


def build_chain_dataset(word_pool: list[str], hops_per_level: int,
                        rng: random.Random) -> list[dict]:
    """Generate equal counts per hop level with single-token random values."""
    samples = []
    for n_hops in range(1, 6):
        for _ in range(hops_per_level):
            value = rng.choice(word_pool)
            vars_ = rng.sample(VAR_POOL, n_hops + 1)
            text  = make_chain(vars_, value, n_hops)
            samples.append({"text": text, "answer": value,
                             "n_hops": n_hops, "type": "chain"})
    return samples


# ── Reality Override Template Engine ──────────────────────────────────────────
# Each template has: (premise pattern, question pattern, answer)
# Values substituted procedurally to get diversity

PROPERTY_TEMPLATES = [
    # Substance → property counterfactuals
    ("{subst} is {adj}. {name} touched {subst}. What did {name} feel?", "{adj}"),
    ("{subst} is {adj}. {name} tasted {subst}. How did it taste?",      "{adj}"),
    ("{subst} is {adj}. {name} smelled {subst}. How did it smell?",     "{adj}"),
]

MOTION_TEMPLATES = [
    ("In this world things fall {dir1}. {name} dropped a ball. Which direction did it fall?", "{dir1}"),
    ("Gravity pushes {dir1} here. {name} let go of a rock. Where did it go?", "{dir1}"),
]

SOUND_TEMPLATES = [
    ("In this world {animal} make {sound} sounds. {name} has a {animal}. What sound does it make?", "{sound}"),
]

COLOR_TEMPLATES = [
    ("Here {plant} is {color}. {name} picked a {plant}. What color was it?", "{color}"),
    ("In this world {sky} is {color}. {name} looked up. What color did {name} see?", "{color}"),
]

SUBSTANCES  = ["fire","ice","water","metal","wood","glass","stone","steel","leather","cotton"]
ADJECTIVES  = ["icy","boiling","freezing","rough","smooth","sharp","dry","wet","soft","hard",
               "sticky","slippery","bitter","salty","sweet","sour","loud","silent","bright","dark"]
NAMES       = ["Alice","Bob","Carol","Dave","Eve","Frank","Grace","Hank","Iris","Jack",
               "Kate","Leo","Mia","Ned","Ona","Pete","Quinn","Rose","Sam","Tina"]
ANIMALS     = ["dog","cat","bird","fish","horse","rabbit","lion","bear","wolf","fox"]
SOUNDS      = ["bark","meow","chirp","roar","buzz","hiss","moo","oink","neigh","quack"]
PLANTS      = ["grass","leaf","flower","tree","cactus","fern","moss","vine","bush","reed"]
COLORS      = ["red","blue","green","purple","orange","yellow","pink","cyan","black","white"]
DIRECTIONS  = ["up","sideways","backward","inward","outward","diagonal","circular"]
SKY_THINGS  = ["the sky","clouds","stars","the sun","the moon","rain","snow"]


def build_override_dataset(n: int, rng: random.Random) -> list[dict]:
    """Procedurally generate counterfactual reality override examples."""
    samples = []
    for _ in range(n):
        kind = rng.randint(0, 3)
        if kind == 0:
            tmpl, ans_tmpl = rng.choice(PROPERTY_TEMPLATES)
            subst = rng.choice(SUBSTANCES); adj = rng.choice(ADJECTIVES)
            name  = rng.choice(NAMES)
            text  = tmpl.format(subst=subst, adj=adj, name=name) + "\nAnswer:"
            ans   = ans_tmpl.format(adj=adj)
        elif kind == 1:
            tmpl, ans_tmpl = rng.choice(MOTION_TEMPLATES)
            dir1  = rng.choice(DIRECTIONS); name = rng.choice(NAMES)
            text  = tmpl.format(dir1=dir1, name=name) + "\nAnswer:"
            ans   = ans_tmpl.format(dir1=dir1)
        elif kind == 2:
            tmpl, ans_tmpl = rng.choice(SOUND_TEMPLATES)
            animal = rng.choice(ANIMALS); sound = rng.choice(SOUNDS)
            name   = rng.choice(NAMES)
            text   = tmpl.format(animal=animal, sound=sound, name=name) + "\nAnswer:"
            ans    = ans_tmpl.format(sound=sound)
        else:
            tmpl, ans_tmpl = rng.choice(COLOR_TEMPLATES)
            plant = rng.choice(PLANTS + SKY_THINGS); color = rng.choice(COLORS)
            name  = rng.choice(NAMES)
            text  = tmpl.format(plant=plant, sky=plant, color=color, name=name) + "\nAnswer:"
            ans   = ans_tmpl.format(color=color)

        # Verify answer is single token
        enc = tokenizer.encode(" " + ans, add_special_tokens=False)
        if len(enc) == 1:
            samples.append({"text": text, "answer": ans,
                             "n_hops": 1, "type": "override"})
    return samples


if __name__ == "__main__":
    rng = random.Random(SEED)

    print(f"Building v33 dataset...")
    print(f"  Chain samples:    {CHAIN_SAMPLES:,} ({HOPS_PER_LEVEL:,} per hop, 1-5 hops)")
    print(f"  Override samples: {OVERRIDE_SAMPLES:,} counterfactuals")
    print(f"  CRITICAL: single-token filter on all chain values\n")

    word_pool  = build_single_token_pool(tokenizer)
    chains     = build_chain_dataset(word_pool, HOPS_PER_LEVEL, rng)
    overrides  = build_override_dataset(OVERRIDE_SAMPLES * 2, rng)  # oversample, filter below

    # Keep only valid overrides (single-token answer already filtered above)
    overrides  = overrides[:OVERRIDE_SAMPLES]

    all_samples = chains + overrides
    rng.shuffle(all_samples)

    # Stats
    hop_dist  = {}; type_dist = {}
    for s in all_samples:
        hop_dist[s["n_hops"]] = hop_dist.get(s["n_hops"], 0) + 1
        type_dist[s["type"]]  = type_dist.get(s["type"], 0) + 1

    print(f"\n  Total:     {len(all_samples):,}")
    print(f"  By type:   {type_dist}")
    print(f"  Hop dist:  {dict(sorted(hop_dist.items()))}")

    print(f"\n  Chain examples:")
    for n in range(1, 6):
        ex = next(s for s in chains if s["n_hops"] == n)
        print(f"    {n}-hop: {ex['text'][:80]!r}  → {ex['answer']!r}")

    print(f"\n  Override examples:")
    for ex in overrides[:4]:
        print(f"    {ex['text'][:80]!r}  → {ex['answer']!r}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_samples, f, indent=2)
    print(f"\n  ✅ Saved {len(all_samples):,} samples → {OUTPUT_FILE}")
    print(f"  Pool purity: all chain values are SINGLE TOKENS — no more BPE splits")
