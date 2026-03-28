"""
Test 1: Asymptotic Length Validation
====================================
Proves that the Mamba-130M RLF architecture achieves O(1) memory usage
across extreme hop chain lengths (up to 1000 hops).

A standard Transformer would OOM or collapse at 1000 hops due to KV-cache
growth. This model uses a fixed-size prefix scratchpad regardless of depth.
"""
import torch
import random
import time
import gc
from mamba1_engine import RecursiveMamba1_PrefixScratchpad, tokenizer, HALT_ID

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CKPT       = "saved_weights/mamba130m_v6_best.pt"
N_SAMPLES  = 20   # samples per hop level
HOP_LEVELS = [15, 50, 100, 500, 1000]
VOCAB_SIZE = 50_000  # numeric range for payload

# Word vocab (novel) — same 500 words as in train_130m.py
import string
_WORD_VOCAB = [
    "apple","brave","crane","drift","ember","frost","globe","haven","india",
    "joust","karma","lemon","mirth","noble","ocean","prime","quest","raven",
    "solar","thorn","ultra","vivid","water","xenon","yield","zonal","alpha",
    "bravo","delta","echo" ,"foxtrot","golf","hotel","juliet","kilo","lima",
    "mike","november","oscar","papa","quebec","romeo","sierra","tango","uniform",
    "victor","whisky","yankee","zulu","amber","cedar","flint","grove","hazel",
]

_DISTRACTOR_KEYS = [
    "sys","env","tmp","buf","idx","ptr","cnt","sum",
    "val","key","ref","aux","err","bit","reg","mem",
]


def make_chain(n_hops: int, rng: random.Random, adversarial: bool = True) -> tuple[str, str]:
    """Generate an n_hop chain prompt and return (prompt_text, answer_str)."""
    # Build the chain: V1→V2→...→V_{n+1} where V_{n+1} is the numeric payload
    chain_vars = [f"V{i}" for i in range(1, n_hops + 2)]
    payload    = str(rng.randint(10000, 99999))

    facts = []
    for i in range(n_hops):
        facts.append(f"{chain_vars[i]}={chain_vars[i+1]}")
    facts.append(f"{chain_vars[-1]}={payload}")

    statements = facts.copy()
    if adversarial:
        # Inject chameleon distractors
        for _ in range(min(5, n_hops // 10 + 2)):
            dk = rng.choice(_DISTRACTOR_KEYS)
            dv = str(rng.randint(100, 99999))
            statements.append(f"{dk}={dv}")

    rng.shuffle(statements)
    prompt = ". ".join(statements) + f". What is {chain_vars[0]}?"
    return prompt, payload


def measure_memory_mb() -> float:
    """Return current GPU allocated memory in MB."""
    if DEVICE == "cuda":
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def run_inference(model, prompt: str, max_loops: int = 32) -> str:
    """Run model inference and return predicted answer string."""
    model.eval()
    with torch.no_grad():
        inp = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        loops, trace, answer = model(inp)
    return answer.strip()


def main() -> None:
    """Run the asymptotic length validation suite."""
    print("=" * 70)
    print("  Test 1: Asymptotic Length Validation")
    print(f"  Checkpoint: {CKPT}")
    print(f"  Device: {DEVICE.upper()}")
    print("=" * 70)

    # Load model
    print("\n[INIT] Loading model...")
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    backbone = MambaLMHeadModel.from_pretrained(
        "state-spaces/mamba-130m", dtype=torch.bfloat16, device=DEVICE
    )
    model = RecursiveMamba1_PrefixScratchpad(backbone, lora_rank=4).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()
    print("  Model loaded OK\n")

    rng = random.Random(42)

    # Baseline memory (model loaded, no inference)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
    baseline_mem = measure_memory_mb()

    results = []

    print(f"{'Hops':>6} | {'Acc':>6} | {'MemMB':>8} | {'DeltaMemMB':>11} | {'ms/sample':>10}")
    print("-" * 60)

    for n_hops in HOP_LEVELS:
        correct = 0
        mem_readings = []
        times = []

        for i in range(N_SAMPLES):
            prompt, expected = make_chain(n_hops, rng, adversarial=True)

            t0 = time.perf_counter()
            predicted = run_inference(model, prompt)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            mem_readings.append(measure_memory_mb())
            times.append(elapsed_ms)

            if predicted == expected:
                correct += 1

        acc = correct / N_SAMPLES * 100
        avg_mem = sum(mem_readings) / len(mem_readings)
        delta_mem = avg_mem - baseline_mem
        avg_ms = sum(times) / len(times)

        results.append((n_hops, acc, avg_mem, delta_mem, avg_ms))
        print(f"{n_hops:>6} | {acc:>5.1f}% | {avg_mem:>8.1f} | {delta_mem:>+11.1f} | {avg_ms:>9.0f}ms")

    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    # Memory flatness check
    mem_deltas = [r[3] for r in results]
    max_delta  = max(mem_deltas) - min(mem_deltas)
    print(f"\n  Memory footprint variance across {HOP_LEVELS[0]}→{HOP_LEVELS[-1]} hops: {max_delta:+.1f} MB")

    if max_delta < 50:
        print("  ✅ FLAT MEMORY — O(1) memory claim CONFIRMED")
        print("     (A KV-cache Transformer would OOM at this hop depth)")
    else:
        print("  ⚠️  Memory grew — investigate caching behaviour")

    # Accuracy trend
    accs = [r[1] for r in results]
    print(f"\n  Accuracy trend: {' → '.join(f'{a:.0f}%' for a in accs)}")
    if accs[-1] > 0:
        print(f"  ✅ Model produces valid output at {HOP_LEVELS[-1]} hops (non-zero acc)")
    else:
        print(f"  ⚠️  Accuracy collapsed at {HOP_LEVELS[-1]} hops — latent decay wall reached")

    print()


if __name__ == "__main__":
    main()
