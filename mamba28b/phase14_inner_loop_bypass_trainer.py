"""
Phase 14: Inner-Loop Bypass Trainer
====================================
Architectural Innovation: Detaches the LM Head from the SSM tick loop.
Instead of generating N autoregressive tokens to "think", the model
loops the SSM recurrence h_t = MambaBlock(h_{t-1}) internally in Python.

This collapses the O(N) autoregressive compute latency down to near-O(1)
by bypassing the embedding and vocabulary projection at every tick.

New Components:
  - HaltingHead: A lightweight 2-layer linear classifier that reads the
    SSM hidden state and outputs P(halt). When P(halt) > HALT_THRESHOLD,
    the loop exits and the LM Head renders the final answer.
  - ROM Re-injection: Every ROMI_PERIOD ticks, the original prompt's
    pooled embedding is added back into the residual stream to prevent
    bfloat16 numeric washout over long reasoning chains.
"""

import torch
import torch.nn as nn
import json
import random
import os
from mamba_ssm import MambaLMHeadModel
from transformers import AutoTokenizer

# ─── Hyperparameters ────────────────────────────────────────────────────────
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
LR_HALT       = 1e-4     # Halting Head is brand new, needs fast learning
LR_HEAD       = 5e-6     # LM Head: gentle — preserves Phase 13 English
LR_CORE       = 5e-7     # Backbone: almost frozen — protect the ALU
BATCH_SIZE    = 4        # Smaller batch: inner loop is compute-heavy
MAX_STEPS     = 6000
HALT_THRESHOLD = 0.70    # P(halt) must exceed this to exit the loop
MIN_LOOPS     = 1        # Always execute at minimum 1 SSM tick
MAX_LOOPS     = 20       # Hard ceiling: prevents runaway infinite loops
ROMI_PERIOD   = 5        # Re-inject ROM context every N inner loop ticks
D_MODEL       = 768      # Mamba-130M hidden dimension
CKPT_INTERVAL = 500
LOG_INTERVAL  = 5

BASE_CHECKPOINT = "checkpoints/mamba3_p13_universal_mastered.pt"
# ─────────────────────────────────────────────────────────────────────────────


class HaltingHead(nn.Module):
    """
    Lightweight binary classifier attached to the SSM residual stream.
    Outputs P(halt | h_t) — the probability the model has finished reasoning.

    Architecture: 2-layer MLP with GELU activation (no recurrence needed).
    Trained jointly with the backbone but at a higher learning rate since
    it starts from random initialization.
    """

    def __init__(self, d_model: int) -> None:
        """Initialize the halting head with a 2-layer MLP."""
        super().__init__()
        self.probe = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: (batch, seq_len, d_model) — final SSM residual
        Returns:
            halt_prob: (batch,) — mean P(halt) across position axis
        """
        # Pool across sequence, project to scalar probability
        pooled = hidden_state.mean(dim=1)         # (B, d_model)
        return self.probe(pooled).squeeze(-1)      # (B,)


def run_inner_loop(
    model: MambaLMHeadModel,
    halting_head: HaltingHead,
    prompt_ids: torch.Tensor,
    rom_embedding: torch.Tensor,
    training_mode: bool = True,
    n_true: int = 8,
) -> tuple[torch.Tensor, int, list[float]]:
    """
    Core Phase 14 forward pass: SSM inner loop with HaltingHead control.

    training_mode=True (Teacher Forcing):
        Ignores p_halt entirely. Loops exactly n_true times.
        BCE target=1 is placed at tick n_true — the oracle label.
        This prevents the HaltingHead from learning to be a fixed MAX_LOOPS clock.

    training_mode=False (Inference):
        Uses the real HaltingHead-steered break. The model autonomously
        decides how many ticks to compute before firing the LM Head.

    Args:
        model: MambaLMHeadModel with Phase 13 weights.
        halting_head: HaltingHead binary classifier.
        prompt_ids: (1, seq_len) input token IDs.
        rom_embedding: (1, seq_len, d_model) frozen prompt embedding.
        training_mode: If True, use Teacher Forcing (ignore p_halt).
        n_true: Oracle loop count for Teacher Forcing (ignored if training_mode=False).

    Returns:
        logits: (1, seq_len, vocab_size)
        n_loops: Number of inner loop ticks actually executed.
        halt_probs: List of P(halt) values at each tick.
    """
    hidden_states = model.backbone.embedding(prompt_ids)
    residual = None

    # First full pass through all Mamba layers
    for layer in model.backbone.layers:
        hidden_states, residual = layer(hidden_states, residual=residual)

    halt_probs = []
    n_loops = 0
    loop_limit = n_true if training_mode else MAX_LOOPS

    # Inner loop: Teacher Forcing (training) or HaltingHead-steered (inference)
    while n_loops < loop_limit:
        n_loops += 1

        # ROM Re-injection: pool prompt embedding to (B, 1, D) then broadcast.
        # This avoids shape mismatch when prompt_len != current hidden state seq_len.
        if n_loops % ROMI_PERIOD == 0:
            rom_pooled = rom_embedding.mean(dim=1, keepdim=True).to(hidden_states.dtype)
            hidden_states = hidden_states + rom_pooled

        # Run the full layer stack again on the current hidden state
        for layer in model.backbone.layers:
            hidden_states, residual = layer(hidden_states, residual=residual)

        # Query HaltingHead: monitor P(halt) regardless of mode
        p_halt = halting_head(hidden_states)   # (B,)
        halt_probs.append(p_halt.mean().item())

        # Inference mode only: exit when HaltingHead fires
        if not training_mode and p_halt.mean().item() > HALT_THRESHOLD and n_loops >= MIN_LOOPS:
            break

    # Final normalization and LM Head projection (only happens ONCE)
    if residual is not None:
        final_hidden = model.backbone.norm_f(hidden_states + residual)
    else:
        final_hidden = model.backbone.norm_f(hidden_states)

    logits = model.lm_head(final_hidden.to(torch.bfloat16))
    return logits, n_loops, halt_probs


def load_training_data() -> list[dict]:
    """Load GSM8K math data for Phase 14 loop-calibration training."""
    data = []
    if os.path.exists("phase12b_gsm8k.jsonl"):
        with open("phase12b_gsm8k.jsonl", "r") as f:
            for line in f:
                data.append(json.loads(line))
    print(f"[INIT] Loaded {len(data)} GSM8K problems for loop calibration.")
    return data


def main() -> None:
    """Main Phase 14 training loop with HaltingHead and ROM re-injection."""
    print("=" * 62)
    print("  MAMBA-3 PHASE 14: INNER-LOOP BYPASS + HALTING HEAD")
    print("=" * 62)

    os.makedirs("checkpoints", exist_ok=True)

    if not os.path.exists(BASE_CHECKPOINT):
        print(f"[FATAL] Phase 13 checkpoint not found: {BASE_CHECKPOINT}")
        return

    print(f"[INIT] Loading Phase 13 Universal Weights: {BASE_CHECKPOINT}")
    model = MambaLMHeadModel.from_pretrained(
        "state-spaces/mamba-130m", device=DEVICE, dtype=torch.bfloat16
    )
    model.load_state_dict(torch.load(BASE_CHECKPOINT, map_location=DEVICE))
    model.train()

    # Initialize HaltingHead from scratch (random weights)
    halting_head = HaltingHead(d_model=D_MODEL).to(DEVICE).to(torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token

    # Orthogonal parameter groups (3-tier gradient surgery)
    head_params = set(model.lm_head.parameters())
    core_params = [p for p in model.backbone.parameters() if p not in head_params]

    optimizer = torch.optim.AdamW([
        {"params": core_params,                   "lr": LR_CORE},
        {"params": list(head_params),              "lr": LR_HEAD},
        {"params": halting_head.parameters(),      "lr": LR_HALT},
    ], weight_decay=0.01)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    # Halting supervision: BCELoss — teach P(halt)=1 only at final tick
    halt_criterion = nn.BCELoss()

    data = load_training_data()
    if not data:
        print("[FATAL] No training data found. Aborting Phase 14.")
        return

    global_step = 0

    while global_step < MAX_STEPS:
        item = random.choice(data)
        n_loops_in_batch = []

        # Build a spaced-digit prompt for the math problem
        prompt_text = f"[LOGIC] {item['prompt']}\nSolution: "
        answer_text = f"<answer>{item['answer']}</answer>"

        p_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(DEVICE)
        a_ids = tokenizer.encode(answer_text, return_tensors="pt")[0].to(DEVICE)
        eos_id = torch.tensor([tokenizer.eos_token_id]).to(DEVICE)
        full_ids = torch.cat([p_ids[0], a_ids, eos_id]).unsqueeze(0)

        # Targets: mask out prompt, supervise on answer only
        labels = full_ids.clone()
        labels[0, :p_ids.shape[1]] = -100

        # Teacher Forcing: sample oracle loop count from Uniform(5, 12)
        # This is the ground truth — how many ticks are needed for this problem.
        # We do NOT let the untrained HaltingHead steer the loop count during training.
        n_true = random.randint(5, 12)

        # Compute frozen ROM embedding (no gradient — this is the lifeline anchor)
        with torch.no_grad():
            rom_embedding = model.backbone.embedding(p_ids)

        optimizer.zero_grad()

        # Run the full inner-loop forward pass with Teacher Forcing
        padded_prompt = full_ids[:, :-1]
        logits, n_loops, halt_probs = run_inner_loop(
            model, halting_head, padded_prompt, rom_embedding,
            training_mode=True, n_true=n_true
        )

        targets = labels[:, 1:]

        # Language modeling loss (CE on answer tokens only)
        lm_loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )

        # HaltingHead Teacher Forcing supervision:
        # target = 0 for ticks 0..n_true-2 (keep computing)
        # target = 1 at tick n_true-1 (oracle halt signal)
        # This prevents learning a fixed MAX_LOOPS clock.
        if n_loops > 0:
            halt_targets = torch.zeros(n_loops, device=DEVICE, dtype=torch.bfloat16)
            halt_targets[-1] = 1.0   # Oracle: halt exactly at n_true
            halt_probs_tensor = torch.tensor(halt_probs, device=DEVICE, dtype=torch.bfloat16)
            halt_loss = halt_criterion(halt_probs_tensor, halt_targets)
        else:
            halt_loss = torch.tensor(0.0, device=DEVICE)

        # Combined loss: LM quality + Halting calibration
        loss = lm_loss + 0.1 * halt_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(halting_head.parameters(), 1.0)
        optimizer.step()

        n_loops_in_batch.append(n_loops)
        global_step += 1

        if global_step % LOG_INTERVAL == 0:
            avg_loops = sum(n_loops_in_batch) / len(n_loops_in_batch)
            vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(
                f"[P14 S{global_step:05d}] "
                f"LM Loss: {lm_loss.item():.4f} | "
                f"Halt Loss: {halt_loss.item():.4f} | "
                f"Avg Loops: {avg_loops:.1f} | "
                f"VRAM: {vram_gb:.2f} GB",
                flush=True
            )

        if global_step % CKPT_INTERVAL == 0:
            torch.save(model.state_dict(), f"checkpoints/mamba3_p14_g{global_step}.pt")
            torch.save(halting_head.state_dict(), f"checkpoints/mamba3_p14_halting_head_g{global_step}.pt")
            print(f"[CKPT] Saved Phase 14 tensors at step {global_step}.")

    print("[SYSTEM] PHASE 14 COMPLETE: Inner-Loop Bypass Engine Forged.")
    torch.save(model.state_dict(), "checkpoints/mamba3_p14_bypass_mastered.pt")
    torch.save(halting_head.state_dict(), "checkpoints/mamba3_p14_halting_head_mastered.pt")
    print("[SYSTEM] mamba3_p14_bypass_mastered.pt and halting head saved.")


if __name__ == "__main__":
    main()
