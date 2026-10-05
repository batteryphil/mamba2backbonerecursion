"""
train_mamba_gist.py — End-to-End Training and Evaluation for Mamba-Gist Hybrid
=============================================================================
Replaces the failed Prefix-RLF with Topological Associative Thought Reconstruction (Gist).
Freezes the Mamba backbone and trains only the Gist projections to bind facts
into associative matrix M and recall them across recurrent reasoning loops.
"""

import os
import sys
import time
import math
import random
import string
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Environment setup for ROCm / MIOpen and offline mode
MIO_DIR = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.makedirs(MIO_DIR, exist_ok=True)
os.environ["MIOPEN_USER_DB_PATH"] = MIO_DIR
os.environ["MIOPEN_CACHE_DIR"] = MIO_DIR
os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"
os.environ["AMD_SERIALIZE_KERNEL"] = "3"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_gist_engine import MambaGistModel


# ─────────────────────────────────────────────────────────────────────────────
# Task & Data Generation: Multi-Hop Associative Variable Chains
# ─────────────────────────────────────────────────────────────────────────────

WORDS = [
    "Red", "Blue", "Green", "Gold", "Silver", "Iron", "Wolf", "Hawk",
    "Lion", "Bear", "Sun", "Moon", "Star", "Fire", "Ice", "Rock",
    "Alpha", "Beta", "Gamma", "Delta", "Echo", "Fox", "Oak", "Pine"
]

PROSE_DISTRACTORS = [
    "The quick brown fox jumps over the lazy dog.",
    "Water boils at 100 degrees Celsius under standard atmospheric pressure.",
    "Neil Armstrong was the first person to walk on the moon in 1969.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into oxygen.",
    "Paris is the capital and largest city of France.",
    "The mitochondria is considered the powerhouse of the biological cell.",
    "Pi is approximately equal to 3.14159265.",
    "Jupiter is the largest planet in the solar system.",
]


def generate_variable_chain(
    rng: random.Random,
    hops: int = 2,
    with_distractors: bool = True,
    reverse_order: bool = False,
) -> Tuple[str, str]:
    """
    Generates a variable assignment chain:
      x0 = Val
      x1 = x0
      ...
      xn = x(n-1)
      Query: What is xn? Answer: Val
      
    If reverse_order=True, pointers appear in reverse causal order:
      xn = x(n-1)
      ...
      x1 = x0
      x0 = Val
    This strictly requires recurrent memory (Loop 1) to resolve,
    as single-pass causal scan cannot see forward into the definitions.
    """
    val = rng.choice(WORDS)
    var_len = rng.randint(2, 4)
    vars = ["".join(rng.choice(string.ascii_lowercase) for _ in range(var_len)) for _ in range(hops + 1)]
    # Ensure distinct variable names
    while len(set(vars)) < len(vars):
        vars = ["".join(rng.choice(string.ascii_lowercase) for _ in range(var_len)) for _ in range(hops + 1)]

    # Ground fact: vars[0] = val
    facts = [f"{vars[0]}={val}."]
    # Pointer chains: vars[i] = vars[i-1]
    for i in range(1, hops + 1):
        facts.append(f"{vars[i]}={vars[i-1]}.")

    distractor_facts = []
    if with_distractors:
        num_distractors = rng.randint(1, 3)
        for _ in range(num_distractors):
            d_var = "".join(rng.choice(string.ascii_lowercase) for _ in range(var_len))
            d_val = rng.choice(WORDS)
            if d_var not in vars:
                distractor_facts.append(f"{d_var}={d_val}.")
        
        if rng.random() > 0.5:
            distractor_facts.append(rng.choice(PROSE_DISTRACTORS))

    if reverse_order:
        # Pointers come first, definition comes last
        facts = list(reversed(facts))
        if distractor_facts:
            facts = distractor_facts[:1] + facts + distractor_facts[1:]
    else:
        facts = facts + distractor_facts
        rng.shuffle(facts)

    query_var = vars[-1]
    prompt = " ".join(facts) + f" What is {query_var}? Answer:"
    return prompt, val


def prepare_sample(
    model: MambaGistModel,
    min_hops: int = 1,
    max_hops: int = 3,
    with_distractors: bool = True,
    reverse_order: bool = False,
    rng: Optional[random.Random] = None,
) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
    """
    Creates an exact unpadded single sample [1, L].
    Prevents Mamba ROCm causal scan kernel memory misalignment from variable padding.
    """
    if rng is None:
        rng = random.Random()

    hops = rng.randint(min_hops, max_hops)
    prompt, val = generate_variable_chain(
        rng, hops=hops, with_distractors=with_distractors, reverse_order=reverse_order
    )

    # Encode prompt unpadded
    enc = model.tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)

    # Target token
    target_toks = model.tokenizer.encode(f" {val}", add_special_tokens=False)
    target_tensor = torch.tensor([target_toks[0]], dtype=torch.long, device=model.device)

    return input_ids, target_tensor, prompt, val


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Routine
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
@torch.no_grad()
def evaluate_model(
    model: MambaGistModel,
    num_samples: int = 30,
    min_hops: int = 2,
    max_hops: int = 3,
    with_distractors: bool = True,
    reverse_order: bool = False,
    seed: int = 999,
) -> Dict[str, float]:
    """
    Evaluates:
      1. Loop 0 accuracy (single causal pass)
      2. Loop 1 accuracy (with Gist associative memory passed from Loop 0)
      3. Loop 1 Ablated accuracy (with Gist associative memory M zeroed out)
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.eval()
    rng = random.Random(seed)

    correct_loop0 = 0
    correct_loop1 = 0
    correct_ablated = 0

    for _ in range(num_samples):
        input_ids, targets, _, _ = prepare_sample(
            model,
            min_hops=min_hops,
            max_hops=max_hops,
            with_distractors=with_distractors,
            reverse_order=reverse_order,
            rng=rng,
        )

        # Loop 0: Initial pass, collect Gist states
        logits_0, states_0 = model(input_ids, state=None, return_state=True)
        preds_0 = logits_0[:, -1, :].argmax(dim=-1)
        correct_loop0 += (preds_0 == targets).sum().item()

        # Loop 1: Recurrent pass initialized with state from Loop 0
        logits_1, _ = model(input_ids, state=states_0, return_state=False)
        preds_1 = logits_1[:, -1, :].argmax(dim=-1)
        correct_loop1 += (preds_1 == targets).sum().item()

        # Loop 1 Ablated: Zero out associative memory M
        states_ablated = {}
        for idx, (M, Z) in states_0.items():
            states_ablated[idx] = (torch.zeros_like(M), torch.zeros_like(Z))
        logits_ablated, _ = model(input_ids, state=states_ablated, return_state=False)
        preds_ablated = logits_ablated[:, -1, :].argmax(dim=-1)
        correct_ablated += (preds_ablated == targets).sum().item()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    acc_0 = (correct_loop0 / num_samples) * 100.0
    acc_1 = (correct_loop1 / num_samples) * 100.0
    acc_ab = (correct_ablated / num_samples) * 100.0
    delta = acc_1 - acc_ab

    return {
        "samples": num_samples,
        "acc_loop0": acc_0,
        "acc_loop1": acc_1,
        "acc_ablated": acc_ab,
        "delta": delta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model_name: str = "state-spaces/mamba-130m-hf",
    gist_layers: Optional[List[int]] = None,
    num_steps: int = 40,
    batch_size: int = 4,
    lr: float = 3e-4,
    eval_interval: int = 20,
    save_path: Optional[str] = None,
):
    print("=" * 65)
    print("  TRAINING MAMBA-GIST RECURRENT ASSOCIATIVE MEMORY")
    print("=" * 65)
    print(f"[*] Base model: {model_name}")
    print(f"[*] Target steps: {num_steps} | Effective batch size (grad accum): {batch_size} | LR: {lr}")

    model = MambaGistModel(
        base_model_name_or_path=model_name,
        gist_layer_indices=gist_layers,
        d_map=32,
    )
    model.freeze_backbone()

    # AdamW on Gist parameters only
    optimizer = torch.optim.AdamW(
        [p for p in model.gist_modules.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    # Initial baseline evaluation
    print("\n[*] Initial Zero-Shot Evaluation (Shuffled Chains):")
    init_eval = evaluate_model(model, num_samples=25, min_hops=2, max_hops=3, reverse_order=False)
    print(f"    Loop 0 Acc: {init_eval['acc_loop0']:.1f}% | "
          f"Loop 1 Acc: {init_eval['acc_loop1']:.1f}% | "
          f"Ablated Acc: {init_eval['acc_ablated']:.1f}% | "
          f"Memory Delta: {init_eval['delta']:+.1f}%")

    print("\n[*] Starting Training...")
    model.train()
    start_time = time.time()
    rng = random.Random(42)

    grad_accum = batch_size

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        step_loss = 0.0

        for _ in range(grad_accum):
            rev = rng.random() > 0.5
            input_ids, targets, _, _ = prepare_sample(
                model,
                min_hops=1,
                max_hops=3,
                with_distractors=True,
                reverse_order=rev,
                rng=rng,
            )

            # Step A: Loop 0 encodes sequence, generates associative memory state S_0
            logits_0, states_0 = model(input_ids, state=None, return_state=True)

            # Step B: Loop 1 re-reads context with truncated associative memory state S_0
            states_detached = {k: (m.detach(), z.detach()) for k, (m, z) in states_0.items()}
            logits_1, _ = model(input_ids, state=states_detached, return_state=False)

            loss_loop1 = F.cross_entropy(logits_1[:, -1, :], targets)
            loss_loop0 = F.cross_entropy(logits_0[:, -1, :], targets)
            sample_loss = (loss_loop1 + 0.5 * loss_loop0) / grad_accum

            sample_loss.backward()
            step_loss += sample_loss.item()

        torch.nn.utils.clip_grad_norm_(model.gist_modules.parameters(), 1.0)
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            if step % 5 == 0:
                torch.cuda.empty_cache()

        if step % 10 == 0 or step == 1:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            print(f"  Step [{step:3d}/{num_steps:3d}] | "
                  f"Loss: {step_loss:.4f} | "
                  f"Speed: {steps_per_sec:.2f} step/s")

        if step % eval_interval == 0 or step == num_steps:
            print(f"\n--- Evaluation at Step {step} ---")
            eval_shuf = evaluate_model(model, num_samples=25, min_hops=2, max_hops=3, reverse_order=False)
            eval_rev = evaluate_model(model, num_samples=25, min_hops=2, max_hops=3, reverse_order=True)
            print(f"  [Shuffled] Loop 0: {eval_shuf['acc_loop0']:.1f}% | "
                  f"Loop 1: {eval_shuf['acc_loop1']:.1f}% | "
                  f"Ablated: {eval_shuf['acc_ablated']:.1f}% | "
                  f"Delta: {eval_shuf['delta']:+.1f}%")
            print(f"  [Reverse ] Loop 0: {eval_rev['acc_loop0']:.1f}% | "
                  f"Loop 1: {eval_rev['acc_loop1']:.1f}% | "
                  f"Ablated: {eval_rev['acc_ablated']:.1f}% | "
                  f"Delta: {eval_rev['delta']:+.1f}%\n")
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.gist_modules.state_dict(), save_path)
            model.train()

    total_time = time.time() - start_time
    print(f"\n[+] Training finished in {total_time:.1f}s.")

    # Final rigorous evaluation
    print("\n" + "=" * 65)
    print("  FINAL BENCHMARK: REASONING ACROSS DISTRACTORS")
    print("=" * 65)
    final_shuf = evaluate_model(model, num_samples=40, min_hops=2, max_hops=3, reverse_order=False)
    final_rev = evaluate_model(model, num_samples=40, min_hops=2, max_hops=3, reverse_order=True)
    print(f"  [Shuffled Chains]  Loop 0: {final_shuf['acc_loop0']:.1f}% | Loop 1: {final_shuf['acc_loop1']:.1f}% | Delta: {final_shuf['delta']:+.1f}%")
    print(f"  [Reverse Chains]   Loop 0: {final_rev['acc_loop0']:.1f}% | Loop 1: {final_rev['acc_loop1']:.1f}% | Delta: {final_rev['delta']:+.1f}%")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.gist_modules.state_dict(), save_path)
        print(f"\n[+] Trained Gist associative memory weights saved to: {save_path}")

    return model, final_shuf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mamba-Gist Recurrent Memory")
    parser.add_argument("--model", type=str, default="state-spaces/mamba-130m-hf")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--save", type=str, default="checkpoints/mamba_gist_weights.pt")
    args = parser.parse_args()

    train(
        model_name=args.model,
        num_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_interval=args.eval_interval,
        save_path=args.save,
    )
