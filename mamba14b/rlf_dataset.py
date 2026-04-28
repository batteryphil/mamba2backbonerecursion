"""
rlf_dataset.py — RLF Training Data for Mamba-1.4B
====================================================
Port of batteryphil/mamba2backbonerecursion/dataset_rlf.py,
extended with:
  - Math variable chains (word problems as variable assignments)
  - Sequence pattern chains (fixes 0% logic score)
  - Bug variable chains (fixes 20% bug fix score)

All chains produce a sequence of intermediate tokens ending in § (HALT).
The model must predict each intermediate step, then halt.
"""

import torch
import random
import string
from torch.utils.data import Dataset, DataLoader
from rlf_engine_1_4b import tokenizer, HALT_ID


# ── Distractor corpus ─────────────────────────────────────────────────────────
PROSE_DISTRACTORS = [
    "The quick brown fox jumps over the lazy dog.",
    "Water boils at 100 degrees Celsius under standard atmospheric pressure.",
    "In 1969, Neil Armstrong became the first person to walk on the moon.",
    "Photosynthesis converts sunlight, water, and CO2 into oxygen and sugar.",
    "The capital of France is Paris.",
    "Shakespeare wrote Romeo and Juliet early in his career.",
    "The mitochondria is the powerhouse of the cell.",
    "Pi is approximately equal to 3.14159.",
]


# ── Random entity generators ──────────────────────────────────────────────────

def rand_var(rng: random.Random, n: int = 4) -> str:
    """Generate a short random variable name."""
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


def rand_val(rng: random.Random) -> str:
    """Generate a short random value (digit string)."""
    return "".join(rng.choice(string.digits) for _ in range(rng.randint(2, 3)))


def rand_word(rng: random.Random) -> str:
    """Generate a random word-like value."""
    words = ["Blue", "Red", "Cat", "Dog", "Sun", "Moon", "Fire",
             "Star", "Gold", "Ice", "Sky", "Sea", "Oak", "Fox",
             "True", "False", "Alpha", "Beta", "Gamma"]
    return rng.choice(words)


# ── Chain generators ──────────────────────────────────────────────────────────

def make_var_chain(
    rng: random.Random, hops: int, mode: str = "clean"
) -> tuple[str, list[str]]:
    """Variable pointer chain: V1=X, V2=V1, ... What is Vn?

    Returns (prompt, chain_answers) where chain_answers ends with '§'.
    """
    val = rand_word(rng)
    entities = [rand_var(rng) for _ in range(hops + 1)]
    facts = [f"{entities[0]}={val}."] + [
        f"{entities[i]}={entities[i-1]}."
        for i in range(1, hops + 1)
    ]
    query = entities[-1]
    chain: list[str] = [val] * hops + ["§"]

    if mode == "adversarial":
        # Insert distractor facts + prose
        for _ in range(rng.randint(1, 3)):
            k, v = rand_var(rng), rand_val(rng)
            facts.append(f"{k}={v}.")
        for _ in range(rng.randint(0, 2)):
            facts.append(rng.choice(PROSE_DISTRACTORS))
        for _ in range(rng.randint(0, 2)):
            n = rng.randint(5, 15)
            facts.append("".join(rng.choice(string.ascii_letters + string.digits)
                                 for _ in range(n)))

    rng.shuffle(facts)
    prompt = " ".join(facts) + f" What is {query}?"
    return prompt, chain


def make_math_chain(
    rng: random.Random, n_steps: int = 2
) -> tuple[str, list[str]]:
    """Arithmetic word problem as variable chain.

    E.g. n_steps=2:
      apples=6. price=0.50. cost=apples*price. What is cost?
      chain: ["3.00", "§"]

    n_steps=3:
      a=6. pa=0.50. ca=a*pa. b=4. pb=0.75. cb=b*pb. total=ca+cb. What is total?
      chain: ["3.00", "3.00", "6.00", "§"]
    """
    steps_data = []
    for _ in range(n_steps):
        qty   = rng.randint(1, 12)
        price = rng.choice([25, 50, 75, 100, 150, 200]) / 100.0
        cost  = round(qty * price, 2)
        steps_data.append((qty, price, cost))

    total = round(sum(s[2] for s in steps_data), 2)

    # Build variable names
    qty_vars  = [rand_var(rng, 3) for _ in range(n_steps)]
    prc_vars  = [rand_var(rng, 3) for _ in range(n_steps)]
    cst_vars  = [rand_var(rng, 3) for _ in range(n_steps)]
    tot_var   = rand_var(rng, 3)

    facts = []
    chain_vals = []
    for i, (qty, price, cost) in enumerate(steps_data):
        facts.append(f"{qty_vars[i]}={qty}.")
        facts.append(f"{prc_vars[i]}={price:.2f}.")
        facts.append(f"{cst_vars[i]}={qty_vars[i]}*{prc_vars[i]}.")
        chain_vals.append(f"{cost:.2f}")

    if n_steps > 1:
        sum_expr = "+".join(cst_vars)
        facts.append(f"{tot_var}={sum_expr}.")
        chain_vals.append(f"{total:.2f}")
        query = tot_var
    else:
        query = cst_vars[0]

    chain_vals.append("§")
    rng.shuffle(facts)
    prompt = " ".join(facts) + f" What is {query}?"
    return prompt, chain_vals


def make_sequence_chain(
    rng: random.Random,
) -> tuple[str, list[str]]:
    """Sequence pattern chain. model must predict each term then halt.

    Patterns: ×3, ×2, +5, Fibonacci, squares.
    E.g. ×3: x1=2. x2=x1*3. x3=x2*3. x4=x3*3. What is x5?
         chain: ["162", "§"]   (one step: compute x5 = 54*3)
    """
    pattern = rng.choice(["mul3", "mul2", "add5", "fib", "squares"])
    length  = rng.randint(3, 5)  # show this many terms

    if pattern == "mul3":
        start  = rng.randint(1, 5)
        terms  = [start * (3 ** i) for i in range(length + 1)]
        ops    = [f"x{i+1}=x{i}*3" for i in range(1, length + 1)]
    elif pattern == "mul2":
        start  = rng.randint(1, 8)
        terms  = [start * (2 ** i) for i in range(length + 1)]
        ops    = [f"x{i+1}=x{i}*2" for i in range(1, length + 1)]
    elif pattern == "add5":
        start  = rng.randint(0, 10)
        terms  = [start + 5 * i for i in range(length + 1)]
        ops    = [f"x{i+1}=x{i}+5" for i in range(1, length + 1)]
    elif pattern == "fib":
        terms  = [1, 1]
        while len(terms) <= length:
            terms.append(terms[-1] + terms[-2])
        ops = ([f"x1=1", f"x2=1"] +
               [f"x{i+1}=x{i}+x{i-1}" for i in range(2, length + 1)])
    else:  # squares
        terms  = [(i + 1) ** 2 for i in range(length + 1)]
        ops    = [f"x{i+1}=(i+1)^2" for i in range(1, length + 1)]

    # Build prompt: show terms[0..length-1], ask for terms[length]
    facts = [f"x1={terms[0]}."] + [f"{op}." for op in ops[:length - 1]]
    query = f"x{length + 1}"
    answer = str(terms[length])
    chain = [answer, "§"]

    rng.shuffle(facts)
    prompt = " ".join(facts) + f" What is {query}?"
    return prompt, chain


def make_bug_chain(rng: random.Random) -> tuple[str, list[str]]:
    """Bug fix as a variable substitution chain.

    E.g.: op=minus. fix=plus. expr=a op b. What is the fixed expr?
    chain: ["a plus b", "§"]
    """
    bugs = [
        ("minus", "plus",   "a minus b",   "a plus b"),
        ("plus",  "minus",  "a plus b",    "a minus b"),
        ("times", "divide", "a times b",   "a divide b"),
        ("==1",   "==0",    "n%2 ==1",     "n%2 ==0"),
        ("+x",    "-x",     "return n +x", "return n -x"),
    ]
    bug, fix, expr, fixed = rng.choice(bugs)
    var_b = rand_var(rng, 3)
    var_f = rand_var(rng, 3)
    prompt = f"{var_b}={bug}. {var_f}={fix}. expr=a {var_b} b. What is the fixed expr?"
    chain  = [fixed, "§"]
    return prompt, chain


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class RLFDataset(Dataset):
    """Mixed RLF training dataset: var chains, math chains, sequences, bugs.

    Curriculum mix (configurable):
      - 50% variable pointer chains (1-3 hops, clean + adversarial)
      - 30% math variable chains (arithmetic word problems)
      - 15% sequence pattern chains
      -  5% bug fix chains
    """

    def __init__(
        self,
        size: int = 15000,
        seq_len: int = 256,
        seed: int = 42,
        adversarial_prob: float = 0.4,
    ) -> None:
        """Init dataset.

        Args:
            size:             number of samples to generate
            seq_len:          max token length (truncated from left)
            seed:             random seed for reproducibility
            adversarial_prob: fraction of var-chain samples with distractors
        """
        self.size      = size
        self.seq_len   = seq_len
        self.rng       = random.Random(seed)
        self.adv_prob  = adversarial_prob
        self.pad_id    = tokenizer.eos_token_id

    def __len__(self) -> int:
        """Return dataset size."""
        return self.size

    def __getitem__(self, idx: int) -> dict:
        """Generate one sample deterministically from idx.

        Returns dict with:
            input_ids:      [seq_len] token ids
            chain_targets:  list of target token ids (chain + HALT_ID)
            ans_start:      index in input_ids where the answer begins
        """
        self.rng.seed(idx + 314159)

        roll = self.rng.random()
        if roll < 0.50:
            # Variable pointer chain
            hops = self.rng.choice([1, 2, 3])
            mode = "adversarial" if self.rng.random() < self.adv_prob else "clean"
            prompt, chain = make_var_chain(self.rng, hops, mode)
        elif roll < 0.80:
            # Math chain
            n_steps = self.rng.choice([1, 2, 3])
            prompt, chain = make_math_chain(self.rng, n_steps)
        elif roll < 0.95:
            # Sequence chain
            prompt, chain = make_sequence_chain(self.rng)
        else:
            # Bug fix chain
            prompt, chain = make_bug_chain(self.rng)

        # Tokenize prompt
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > self.seq_len:
            input_ids = input_ids[-self.seq_len :]   # truncate left
        ans_start = len(input_ids)

        # Pad
        pad_len   = self.seq_len - len(input_ids)
        input_ids = input_ids + [self.pad_id] * pad_len

        # Tokenize chain targets (first token of each word)
        chain_ids: list[int] = []
        for step in chain:
            if step == "§":
                chain_ids.append(HALT_ID)
            else:
                toks = tokenizer.encode(" " + step)
                chain_ids.append(toks[0])

        return {
            "input_ids":     torch.tensor(input_ids, dtype=torch.long),
            "chain_targets": chain_ids,
            "ans_start":     ans_start,
        }


def collate_rlf(batch: list[dict]) -> tuple:
    """Collate a batch of RLF samples.

    Returns:
        input_ids:     [B, seq_len]
        chain_targets: list of B lists
        ans_starts:    list of B ints
    """
    input_ids    = torch.stack([item["input_ids"] for item in batch])
    chain_targets= [item["chain_targets"] for item in batch]
    ans_starts   = [item["ans_start"] for item in batch]
    return input_ids, chain_targets, ans_starts


if __name__ == "__main__":
    ds = RLFDataset(size=10, seq_len=128, adversarial_prob=0.5)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_rlf)
    for ids, targets, starts in loader:
        print("Input shape:", ids.shape)
        for i in range(len(targets)):
            decoded = tokenizer.decode(ids[i, : starts[i]])
            chain   = [
                tokenizer.decode([t]) if t != HALT_ID else "<HALT>"
                for t in targets[i]
            ]
            print(f"  Sample {i}: {decoded!r}")
            print(f"  Chain:    {chain}")
        break
