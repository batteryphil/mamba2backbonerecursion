"""
phase7_general_recovery_v2.py — Fixed Phase 7 General Recovery Trainer
=======================================================================
Corrected version — uses MambaLMHeadModel directly (NOT the RLF engine)
with the native Phase 14 format:
  [LOGIC] {question}\nSolution: <answer>{answer}</answer>

Previous attempt FAILED because it used RecursiveMamba1_PrefixScratchpad
with [answer_tok, HALT_ID] targets — a format mamba3_p15 was never trained on.
All 2000 steps had 0% accuracy as a result.

This version:
  - Uses MambaLMHeadModel.load_state_dict() directly
  - Format: [LOGIC] Q:\nSolution: <answer>X</answer>  (CE on answer only)
  - 50% general format (MC, T/F, fill-blank, QA) + 50% RLF chain problems
  - p_drop not applicable here — uses ROM re-injection from Phase 14 instead
  - Same 3-tier LR groups as Phase 14: core < head < (halting_head n/a)

Usage:
    python phase7_general_recovery_v2.py
    python phase7_general_recovery_v2.py --ckpt checkpoints/mamba3_p15_conversational_thoughts.pt
    python phase7_general_recovery_v2.py --steps 500   # quick test
"""

import os
import sys
import time
import random
import argparse
import torch
import torch.nn as nn
from mamba_ssm import MambaLMHeadModel
from transformers import AutoTokenizer

# ── Config ───────────────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID     = "state-spaces/mamba-130m"
CKPT_DIR     = "checkpoints"
INPUT_CKPT   = f"{CKPT_DIR}/mamba3_p15_conversational_thoughts.pt"
OUTPUT_CKPT  = f"{CKPT_DIR}/mamba3_p7v3_general_recovery_best.pt"
LOG_PATH     = "training_phase7v3.log"

STEPS        = 2000
BATCH        = 4
LR_CORE      = 5e-7     # backbone — nearly frozen, preserve Phase 15
LR_HEAD      = 5e-5     # LM head — needs substantial movement to learn new format
STOP_ACC     = 0.15     # roll(100) token accuracy — realistic for sporadic batch hits
STOP_AFTER   = 400
LOG_EVERY    = 25
CKPT_EVERY   = 200

ROMI_PERIOD  = 5        # ROM re-injection every N loops (from Phase 14)
GENERAL_RATIO = 0.50    # 50% general format, 50% RLF chains

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token

ANSWER_OPEN  = "<answer>"
ANSWER_CLOSE = "</answer>"

# ── General-format QA pools — full-text answers matching model output style ──
# The model was trained Phase 14/15 style and outputs full words, not A/B/C codes.
_MC = [
    ("What color is the sky on a clear day?",         "Blue"),
    ("How many sides does a triangle have?",           "3"),
    ("Which planet is closest to the sun?",            "Mercury"),
    ("What is 2 plus 2?",                              "4"),
    ("What is the largest ocean on Earth?",            "Pacific"),
    ("At what temperature in Celsius does water boil?", "100"),
    ("How many days are in a week?",                   "7"),
    ("What is the capital of France?",                 "Paris"),
    ("What is the chemical formula for water?",        "H2O"),
    ("How many continents are on Earth?",              "7"),
    ("What is the square root of 16?",                 "4"),
    ("How many seconds are in one minute?",            "60"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("How many legs does a spider have?",              "8"),
    ("What is the capital of Japan?",                  "Tokyo"),
]
_TF = [
    ("The sun is a star.",                             "True"),
    ("Fish can breathe underwater.",                   "True"),
    ("The Earth is flat.",                             "False"),
    ("Ice is hotter than steam.",                      "False"),
    ("The moon produces its own light.",               "False"),
    ("Gold is a metal.",                               "True"),
    ("Sound travels faster than light.",               "False"),
    ("Dogs are mammals.",                              "True"),
    ("Penguins live at the North Pole.",               "False"),
    ("The heart pumps blood.",                         "True"),
]
_FILL = [
    ("The capital of Japan is ___.",                   "Tokyo"),
    ("Water freezes at ___ degrees Celsius.",          "0"),
    ("A dozen equals ___ items.",                      "12"),
    ("A triangle has ___ sides.",                      "3"),
    ("One minute has ___ seconds.",                    "60"),
    ("The square root of 9 is ___.",                   "3"),
]
_QA = [
    ("What is 7 plus 5?",                              "12"),
    ("How many legs does a spider have?",              "8"),
    ("What is 4 times 4?",                             "16"),
    ("How many hours are in a day?",                   "24"),
    ("What is 10 minus 3?",                            "7"),
    ("What is the first letter of the alphabet?",      "A"),
]

# ── RLF chain samples (same variable-chain format as Phase 7) ─────────────────
def _make_chain(rng: random.Random) -> tuple[str, str]:
    """Generate a variable-hop chain problem and its answer.

    Args:
        rng: Seeded random instance.

    Returns:
        Tuple of (question_text, answer_string).
    """
    hops = rng.randint(2, 5)
    val  = str(rng.randint(1, 9999))
    parts = [f"V1={val}."]
    for i in range(2, hops + 1):
        parts.append(f"V{i}=V{i-1}.")
    parts.append(f"What is V{hops}?")
    return " ".join(parts), val


def make_sample(idx: int, general_ratio: float) -> tuple[str, str]:
    """Generate a training (prompt, answer) pair in the Phase 14 native format.

    Args:
        idx: Sample index for seeding.
        general_ratio: Fraction of samples that should be general-format QA.

    Returns:
        Tuple of (prompt_prefix, answer_string) where prompt ends right before answer.
    """
    rng = random.Random(idx * 31337 + 7)
    if rng.random() < general_ratio:
        fmt = rng.randint(0, 3)
        if fmt == 0:
            q, a = rng.choice(_MC)
            return f"[LOGIC] {q}\nSolution: ", a
        elif fmt == 1:
            stmt, a = rng.choice(_TF)
            return f"[LOGIC] True or False: {stmt}\nSolution: ", a
        elif fmt == 2:
            tmpl, a = rng.choice(_FILL)
            return f"[LOGIC] Complete the following: {tmpl}\nSolution: ", a
        else:
            q, a = rng.choice(_QA)
            return f"[LOGIC] {q}\nSolution: ", a
    else:
        q, a = _make_chain(rng)
        return f"[LOGIC] {q}\nSolution: ", a


def build_ids(
    prompt: str,
    answer: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build input_ids and labels for CE training.

    Input:  [LOGIC] prompt\nSolution: <answer>X</answer>
    Labels: -100 for prompt, token ids for <answer>X</answer>

    Args:
        prompt: Prompt text ending before the answer.
        answer: The correct answer string.

    Returns:
        Tuple of (input_ids [1, T], labels [1, T]).
    """
    answer_text = f"{ANSWER_OPEN}{answer}{ANSWER_CLOSE}"
    full_text   = prompt + answer_text

    p_ids  = tokenizer.encode(prompt,      add_special_tokens=False)
    a_ids  = tokenizer.encode(answer_text, add_special_tokens=False)
    full   = p_ids + a_ids

    input_ids = torch.tensor([full], dtype=torch.long, device=DEVICE)
    labels    = input_ids.clone()
    labels[0, :len(p_ids)] = -100   # mask prompt — supervise answer only
    return input_ids, labels


def forward_pass(
    model: MambaLMHeadModel,
    input_ids: torch.Tensor,
    rom_embedding: torch.Tensor,
    n_rom_loops: int = 1,
) -> torch.Tensor:
    """Run Phase 14 style forward: full pass + ROM re-injection loops.

    Args:
        model: The MambaLMHeadModel.
        input_ids: [1, T] token ids.
        rom_embedding: [1, T, d_model] frozen prompt embedding.
        n_rom_loops: How many extra ROM re-injection loops to run.

    Returns:
        logits: [1, T, vocab].
    """
    hidden = model.backbone.embedding(input_ids)  # [1, T, d]
    residual = None
    for layer in model.backbone.layers:
        hidden, residual = layer(hidden, residual=residual)

    for loop_i in range(n_rom_loops):
        if loop_i % ROMI_PERIOD == 0:
            rom_pooled = rom_embedding.mean(dim=1, keepdim=True).to(hidden.dtype)
            hidden = hidden + rom_pooled
        for layer in model.backbone.layers:
            hidden, residual = layer(hidden, residual=residual)

    if residual is not None:
        final = model.backbone.norm_f(hidden + residual)
    else:
        final = model.backbone.norm_f(hidden)

    return model.lm_head(final.to(torch.bfloat16))


def main() -> None:
    """Load checkpoint, run fixed Phase 7, save best checkpoint."""
    parser = argparse.ArgumentParser(description="Phase 7 v2 General Recovery (fixed format)")
    parser.add_argument("--ckpt",  type=str, default=INPUT_CKPT)
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()

    print(f"\n{'='*68}")
    print(f"  PHASE 7 v2 — General Recovery (Fixed Format)")
    print(f"  Device:      {DEVICE.upper()}")
    print(f"  Input ckpt:  {args.ckpt}")
    print(f"  Output:      {OUTPUT_CKPT}")
    print(f"  Format:      [LOGIC] prompt\\nSolution: <answer>X</answer>")
    print(f"  Mix:         {100*(1-GENERAL_RATIO):.0f}% RLF chains  +  {100*GENERAL_RATIO:.0f}% general QA")
    print(f"{'='*68}\n")

    os.makedirs(CKPT_DIR, exist_ok=True)

    # Load model
    print("[INIT] Loading backbone…")
    model = MambaLMHeadModel.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device=DEVICE
    )
    if os.path.exists(args.ckpt):
        state = torch.load(args.ckpt, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
        print(f"[INIT] Loaded checkpoint: {args.ckpt}")
    else:
        print(f"[WARN] Checkpoint not found: {args.ckpt} — using pretrained base")

    model.train()

    # 3-tier optimizer — protect Phase 15 knowledge, gently nudge format recall
    head_params = list(model.lm_head.parameters())
    head_ids    = {id(p) for p in head_params}
    core_params = [p for p in model.backbone.parameters() if id(p) not in head_ids]

    optimizer = torch.optim.AdamW([
        {"params": core_params, "lr": LR_CORE},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=0.01)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_acc  = 0.0
    rolling   = []
    step      = 0
    rng       = random.Random(42)

    print(f"\n{'='*68}")
    print(f"  Training started — logs flush every {LOG_EVERY} steps")
    print(f"{'='*68}\n")

    with open(LOG_PATH, "a") as log:
        log.write(f"\n=== PHASE 7 v2 START {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log.write(f"Input: {args.ckpt}\n")
        t_start = time.time()

        for step in range(args.steps):
            # Sample a batch of BATCH problems
            batch_loss = torch.tensor(0.0, device=DEVICE)
            batch_correct = 0
            batch_valid   = 0

            for b in range(BATCH):
                sample_idx = step * BATCH + b
                prompt, answer = make_sample(sample_idx, GENERAL_RATIO)

                input_ids, labels = build_ids(prompt, answer)

                if input_ids.shape[1] < 2:
                    continue

                with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                    out    = model(input_ids)
                    logits = out.logits
                    shifted_logits = logits[:, :-1, :].contiguous()
                    shifted_labels = labels[:, 1:].contiguous()
                    loss = criterion(
                        shifted_logits.view(-1, shifted_logits.size(-1)),
                        shifted_labels.view(-1),
                    )

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                batch_loss   = batch_loss + loss
                batch_valid  += 1

                # Accuracy: check first non-masked prediction is the answer token.
                # The label sequence is: ...<answer>X</answer>
                # We want the token AFTER the opening <answer> tag — that's the answer.
                # Encode just the tag so we know how many tokens it takes.
                mask_pos = (shifted_labels[0] != -100).nonzero(as_tuple=True)[0]
                if len(mask_pos) > 0:
                    tag_toks = tokenizer.encode(ANSWER_OPEN, add_special_tokens=False)
                    tag_len  = len(tag_toks)
                    # First supervised position is <answer>; answer token is tag_len positions later
                    ans_pos  = mask_pos[0].item() + tag_len
                    if ans_pos < shifted_logits.shape[1]:
                        pred_tok = shifted_logits[0, ans_pos].argmax().item()
                        ans_toks = tokenizer.encode(answer, add_special_tokens=False)
                        if ans_toks and pred_tok == ans_toks[0]:
                            batch_correct += 1

            if batch_valid == 0:
                continue

            avg_batch_loss = batch_loss / batch_valid
            avg_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            step_acc = batch_correct / max(batch_valid, 1)
            rolling.append(step_acc)
            if len(rolling) > 100:
                rolling.pop(0)
            roll_avg = sum(rolling) / len(rolling)

            if step_acc > best_acc:
                best_acc = step_acc
                torch.save(model.state_dict(), OUTPUT_CKPT)

            if step % LOG_EVERY == 0:
                elapsed = time.time() - t_start
                line = (
                    f"P7v2 Step {step:5d} | Loss {avg_batch_loss.item():.4f} | "
                    f"Acc {step_acc:.2f} | Roll {roll_avg:.2f} | "
                    f"Best {best_acc:.2f} | {elapsed:.0f}s"
                )
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()

            if step > 0 and step % CKPT_EVERY == 0:
                ckpt = f"{CKPT_DIR}/mamba3_p7v3_step{step}.pt"
                torch.save(model.state_dict(), ckpt)
                print(f"  [CKPT] {ckpt}", flush=True)

            if step >= STOP_AFTER and roll_avg >= STOP_ACC:
                msg = f"  ✅ Early stop at step {step} — Roll(100) acc {roll_avg:.3f} >= {STOP_ACC}"
                print(msg, flush=True)
                log.write(msg + "\n")
                break

        log.write(f"\nPhase 7 v2 done. Best acc: {best_acc:.3f} → {OUTPUT_CKPT}\n")
        log.write(f"=== PHASE 7 v2 END {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    print(f"\n🏁 Phase 7 v2 complete → {OUTPUT_CKPT}")
    print(f"\nValidate with:")
    print(f"  python eval/format_recall_test.py --ckpt {OUTPUT_CKPT}")


if __name__ == "__main__":
    main()
