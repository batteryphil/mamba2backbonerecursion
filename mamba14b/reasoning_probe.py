#!/usr/bin/env python3
"""
reasoning_probe.py
Tests whether the latent spacer reasoning is actually doing anything —
i.e. does adding more spacer ticks improve accuracy on hard multi-step problems?

Tests:
  1. Raw answer (0 spacers) vs 4 vs 8 vs 16 spacers — does accuracy improve?
  2. Multi-step math (should need reasoning)
  3. Logic puzzles
  4. Show the raw token-by-token output so we can see if the latent state
     is being used as scratchpad
"""

import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from proprioception_gate import GeometricProprioceptionGate
from lora_mamba import PostBackboneLoRA
from transformers import AutoTokenizer

CKPT_DIR   = Path("/hdd_data/latent-spacer-checkpoints/best")
BASE_MODEL  = "state-spaces/mamba-1.4b"
D_MODEL     = 2048
MAX_NEW     = 200
DEVICE      = "cuda"
TEMP        = 0.6
REP_PENALTY = 1.8
REP_WINDOW  = 100
NGRAM_BLOCK = 4

# Multi-step problems that require actual reasoning
HARD_PROMPTS = [
    {
        "prompt": "A store sells apples for $0.50 each and oranges for $0.75 each. If Sarah buys 6 apples and 4 oranges, how much does she spend in total?",
        "expected": "6.00",  # 6*0.5 + 4*0.75 = 3 + 3 = 6
        "steps": "6 × $0.50 = $3.00, 4 × $0.75 = $3.00, total = $6.00"
    },
    {
        "prompt": "If a rectangle has a length of 12 cm and a width of 8 cm, what is its area and perimeter?",
        "expected": "96",   # area = 96
        "steps": "Area = 12 × 8 = 96 cm², Perimeter = 2(12+8) = 40 cm"
    },
    {
        "prompt": "A class has 30 students. 60% are girls. How many boys are in the class?",
        "expected": "12",   # 30 * 0.4 = 12
        "steps": "Girls = 60% of 30 = 18, Boys = 30 - 18 = 12"
    },
    {
        "prompt": "What is the next number in the sequence: 2, 6, 18, 54, ?",
        "expected": "162",  # multiply by 3
        "steps": "Pattern: ×3 each time. 54 × 3 = 162"
    },
    {
        "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
        "expected": "5",    # classic trick question — 5 minutes
        "steps": "Each machine makes 1 widget in 5 minutes. 100 machines make 100 widgets in 5 minutes."
    },
]


@torch.no_grad()
def run(model, adapter, gate, tokenizer, prompt: str, n_spacers: int) -> tuple[str, float]:
    """Generate with specified number of spacer ticks."""
    spacers     = "=" * n_spacers
    full_prompt = f"[USER]\n{prompt}\n{spacers}\n[ANSWER]\n"
    ids         = tokenizer.encode(full_prompt, return_tensors="pt").to(DEVICE)

    generated  = []
    eos_id     = tokenizer.eos_token_id
    cur        = ids
    ngrams: set[tuple] = set()

    t0 = time.perf_counter()
    for _ in range(MAX_NEW):
        h      = model.backbone(cur)
        h      = adapter(h)
        h      = gate(h)
        logits = model.lm_head(h.to(torch.bfloat16))

        lgt = logits[0, -1, :].float() / TEMP
        for tid in set(generated[-REP_WINDOW:]):
            lgt[tid] = lgt[tid] / REP_PENALTY if lgt[tid] > 0 else lgt[tid] * REP_PENALTY
        if len(generated) >= NGRAM_BLOCK - 1:
            pfx = tuple(generated[-(NGRAM_BLOCK - 1):])
            for c in torch.softmax(lgt, dim=-1).topk(50).indices.tolist():
                if pfx + (c,) in ngrams:
                    lgt[c] = -1e9

        p, si = torch.sort(torch.softmax(lgt, dim=-1), descending=True)
        p[(torch.cumsum(p, 0) - p) > 0.9] = 0.0
        if p.sum() < 1e-8:
            break
        nxt = si[torch.multinomial(p / p.sum(), 1)].item()

        if len(generated) >= NGRAM_BLOCK - 1:
            ngrams.add(tuple(generated[-(NGRAM_BLOCK - 1):]) + (nxt,))
        if nxt == eos_id:
            break
        generated.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], device=DEVICE)], dim=1)

    elapsed = time.perf_counter() - t0
    text    = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, elapsed


def main() -> None:
    """Load model and probe latent reasoning."""
    print("=" * 70)
    print("  LATENT REASONING PROBE")
    print("  Does spacer depth improve multi-step reasoning accuracy?")
    print("=" * 70)

    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    model = MambaLMHeadModel.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device=DEVICE
    )
    model.lm_head.load_state_dict(
        torch.load(CKPT_DIR / "lm_head.pt", map_location=DEVICE, weights_only=True)
    )
    model.eval()

    adapter = PostBackboneLoRA(d_model=D_MODEL, rank=16, alpha=32.0, n_layers=6)
    adapter.load_state_dict(
        torch.load(CKPT_DIR / "adapter.pt", map_location=DEVICE, weights_only=True)
    )
    adapter = adapter.to(DEVICE).to(torch.bfloat16).eval()

    gate = GeometricProprioceptionGate(d_model=D_MODEL, window_size=8)
    gate.load_state_dict(
        torch.load(CKPT_DIR / "gate.pt", map_location=DEVICE, weights_only=True)
    )
    gate = gate.to(DEVICE).to(torch.bfloat16).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neox-20b", trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded.\n")

    tick_counts  = [0, 4, 8, 16]
    score_table  = {t: 0 for t in tick_counts}
    total        = len(HARD_PROMPTS)

    for i, item in enumerate(HARD_PROMPTS, 1):
        print(f"\n{'─'*70}")
        print(f"[Q{i}] {item['prompt']}")
        print(f"  Expected answer contains: {item['expected']!r}")
        print(f"  Correct reasoning: {item['steps']}")
        print()

        for n in tick_counts:
            output, elapsed = run(model, adapter, gate, tokenizer,
                                  item["prompt"], n)
            hit = item["expected"] in output
            if hit:
                score_table[n] += 1
            status = "✅" if hit else "❌"
            label  = f"{n:2d} spacers"
            print(f"  {status} [{label}] ({elapsed:.1f}s): {output[:120].strip()!r}")

    # Summary table
    print(f"\n{'='*70}")
    print("  SPACER DEPTH vs ACCURACY — Does latent reasoning help?")
    print(f"{'='*70}")
    print(f"\n  {'Spacers':>10} {'Correct':>10} {'Rate':>8}")
    print(f"  {'-'*32}")
    for n in tick_counts:
        s = score_table[n]
        print(f"  {n:>10}   {s:>5}/{total}   {s/total*100:>6.0f}%")

    # Verdict
    best    = max(tick_counts, key=lambda n: score_table[n])
    worst   = min(tick_counts, key=lambda n: score_table[n])
    delta   = score_table[best] - score_table[worst]
    working = delta > 0

    print(f"\n  Reasoning {'IS' if working else 'IS NOT'} helping.")
    print(f"  Best spacer count: {best} ticks → {score_table[best]}/{total} correct")
    if working:
        print(f"  Latent 'thinking' adds +{delta} correct answers vs no spacers")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
