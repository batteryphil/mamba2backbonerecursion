"""
quick_test.py — Phase 10 evaluation.
Matches trainer exactly: embed-only forward for void prefix,
then token-accumulating greedy decode.
"""
import re
import torch
from mamba_ssm import MambaLMHeadModel
from transformers import AutoTokenizer

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
N_LOOPS    = 10
CHECKPOINT = "checkpoints/mamba3_p10_g6200.pt"
MAX_NEW    = 60
D_MODEL    = 768

PROBLEMS = [
    ("What is 5 + 7?",    "12"),
    ("What is 48 - 6?",   "42"),
    ("What is 4 * 9?",    "36"),
    ("What is 100 + 23?", "123"),
    ("What is 99 - 9?",   "90"),
    ("What is 3 * 7?",    "21"),
    ("What is 50 + 50?",  "100"),
    ("What is 81 - 37?",  "44"),
]

ONE_SHOT = (
    "Problem: What is 2 + 3?\n"
    "Solution: <answer>5</answer>.\n\n"
)


def embed_forward(model, embeds):
    """Run a raw embedding tensor through all 24 Mamba layers, return last-position logits."""
    h, res = embeds, None
    for layer in model.backbone.layers:
        h, res = layer(h, residual=res)
    h = model.backbone.norm_f(h + res) if res is not None else model.backbone.norm_f(h)
    return model.lm_head(h.to(torch.bfloat16))[:, -1, :]   # [1, vocab]


def generate_answer(model, tokenizer, problem: str) -> str:
    """
    Inference path that matches the Phase 10 trainer exactly:
      1. Embed the prompt tokens
      2. Concatenate N_LOOPS zero-vector void tokens
      3. Run one dense forward pass → get logits at last position
      4. Greedy-decode token by token, appending each new token embed
    No token IDs are ever used to represent void positions.
    """
    prompt     = ONE_SHOT + f"Problem: {problem}\nSolution:"
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    void = torch.zeros(1, N_LOOPS, D_MODEL, device=DEVICE, dtype=torch.bfloat16)

    model.eval()
    with torch.no_grad():
        # ── Step 1: Build prefix embeds = prompt + void ───────────────────
        prompt_embeds   = model.backbone.embedding(prompt_ids)        # [1, L, d]
        combined_embeds = torch.cat([prompt_embeds, void], dim=1)     # [1, L+N, d]

        # ── Step 2: Dense pass through combined prefix ────────────────────
        next_logits = embed_forward(model, combined_embeds)            # [1, vocab]
        running_embeds = combined_embeds                               # accumulate here

        # ── Step 3: Greedy decode — append embed of each new token ────────
        generated = []
        for _ in range(MAX_NEW):
            probs    = torch.softmax(next_logits / 0.7, dim=-1)
            next_tok = torch.argmax(probs, dim=-1)                    # [1]
            tok_id   = next_tok.item()

            if tok_id == tokenizer.eos_token_id:
                break
            generated.append(tok_id)

            # Embed the new token and append — no void IDs involved
            tok_embed      = model.backbone.embedding(next_tok.unsqueeze(0).unsqueeze(0))  # [1,1,d]
            running_embeds = torch.cat([running_embeds, tok_embed], dim=1)
            next_logits    = embed_forward(model, running_embeds)

            decoded = tokenizer.decode(generated)
            if "</answer>" in decoded:
                break

    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    """Run arithmetic evaluation on Phase 10 checkpoint."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading: {CHECKPOINT}")
    model = MambaLMHeadModel.from_pretrained(
        "state-spaces/mamba-130m", device=DEVICE, dtype=torch.bfloat16
    )
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    vram = torch.cuda.memory_reserved(DEVICE) / (1024 ** 3)
    print(f"VRAM: {vram:.2f} GB\n")
    print("=" * 60)
    print(f"  PHASE 10 EVAL  [g6200 | N_LOOPS={N_LOOPS} | zero-void]")
    print("=" * 60)

    correct = 0
    for problem, expected in PROBLEMS:
        print(f"\n[Q]: {problem}")
        output = generate_answer(model, tokenizer, problem)
        m = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
        extracted = m.group(1).strip() if m else "—"
        hit = "✅" if extracted == expected else "❌"
        print(f"[OUT]: {output[:150]}")
        print(f"[ANS]: '{extracted}'  (expected: '{expected}')  {hit}")
        if extracted == expected:
            correct += 1

    print(f"\n{'='*60}")
    print(f"SCORE: {correct} / {len(PROBLEMS)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
