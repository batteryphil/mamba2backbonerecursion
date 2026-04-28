"""
phase5_rlf_recovery.py — RLF Reasoning Recovery Fine-Tune
==========================================================
Short recovery run at LR=1e-5 on clean chain data to restore
RLF reasoning accuracy lost during Phase 4 Engram training.
Preserves the gate weights by using a frozen engram_gate_head.
"""

import torch
import os
import random
from torch.utils.data import Dataset, DataLoader
from mamba_ssm import MambaLMHeadModel
from mamba1_engine import RecursiveMamba1_PrefixScratchpad, MODEL_ID, tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Curriculum: clean chain data matching Phases 1-3 format ──────────────────
VALS = (
    ["Blue", "Red", "Cat", "Dog", "Sun", "Moon", "Fire", "Star", "Gold", "Ice",
     "Sky", "Sea", "Oak", "Elm", "Ash", "Fox", "Owl", "Bat", "Bee", "Ant"]
    + [str(n) for n in range(1, 50)]
    + ["True", "False", "Alpha", "Beta", "Gamma", "Delta"]
)


def make_chain(hops: int) -> tuple[str, str]:
    """Generate a variable-chain prompt and expected answer."""
    val = random.choice(VALS)
    chain = f"V1={val}."
    for i in range(2, hops + 1):
        chain += f" V{i}=V{i-1}."
    chain += f" What is V{hops}? Answer:"
    return chain, val


def generate_recovery_data(size: int = 8000) -> list[tuple[str, str]]:
    """Generate clean chain data across 2-6 hop lengths."""
    data = []
    for _ in range(size):
        hops = random.randint(2, 6)
        prompt, answer = make_chain(hops)
        data.append((prompt, answer))
    return data


# ── Dataset ───────────────────────────────────────────────────────────────────
class RecoveryDataset(Dataset):
    """Wraps (prompt, answer) pairs into tensors."""

    def __init__(self, data: list[tuple[str, str]]) -> None:
        """Initialize dataset."""
        self.data = data

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Return tokenized sample."""
        prompt, answer = self.data[idx]
        full_text = prompt + " " + answer + " <HALT>"
        input_ids = tokenizer.encode(prompt,    return_tensors="pt")[0]
        target_ids = tokenizer.encode(full_text, return_tensors="pt")[0]
        ans_start  = len(input_ids) - 1
        return {"input_ids": input_ids, "target_ids": target_ids, "ans_start": ans_start}


def recovery_collate(batch: list[dict]) -> tuple:
    """Pad batch and return tensors."""
    input_ids  = [b["input_ids"]  for b in batch]
    target_ids = [b["target_ids"] for b in batch]
    ans_starts = torch.tensor([b["ans_start"] for b in batch])

    max_in  = max(len(x) for x in input_ids)
    max_tgt = max(len(x) for x in target_ids)
    pad_id  = tokenizer.pad_token_id or 0

    input_ids  = torch.stack([torch.nn.functional.pad(x, (0, max_in  - len(x)), value=pad_id) for x in input_ids])
    target_ids = torch.stack([torch.nn.functional.pad(x, (0, max_tgt - len(x)), value=pad_id) for x in target_ids])
    return input_ids, target_ids, ans_starts


# ── Training Loop ─────────────────────────────────────────────────────────────
def phase5_recovery() -> None:
    """Run the Phase 5 RLF recovery fine-tune."""
    print("\n" + "=" * 70)
    print("  PHASE 5: RLF REASONING RECOVERY")
    print("=" * 70)

    # Load model
    backbone = MambaLMHeadModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device=DEVICE)
    model    = RecursiveMamba1_PrefixScratchpad(backbone, lora_rank=4).to(DEVICE)

    ckpt = "saved_weights/mamba130m_phase4_engram_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)
    print(f"Loaded Phase 4 checkpoint: {ckpt}")

    # ── Freeze the gate head — preserve Phase 4 gate weights ─────────────────
    for name, param in model.named_parameters():
        param.requires_grad = "engram_gate_head" not in name  # freeze gate
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable (gate frozen): {trainable:,}\n")

    # ─ Optimizer: recovery LR = 1e-5 per training rules ─────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.01
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    data    = generate_recovery_data(size=8000)
    dataset = RecoveryDataset(data)
    loader  = DataLoader(dataset, batch_size=4, shuffle=True,
                         collate_fn=recovery_collate, drop_last=True)

    # ── Loop ─────────────────────────────────────────────────────────────────
    best_acc = 0.0
    model.train()
    step = 0
    total_steps = 2000
    recent_acc  = []

    while step < total_steps:
        for input_ids, target_ids, ans_starts in loader:
            if step >= total_steps:
                break

            input_ids  = input_ids.to(DEVICE)
            target_ids = target_ids.to(DEVICE)
            ans_starts = ans_starts.to(DEVICE)

            optimizer.zero_grad()
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                loss, acc, ans_acc, halt_acc = model(
                    input_ids, chain_targets=target_ids, ans_starts=ans_starts
                )
                if torch.isnan(loss):
                    step += 1
                    continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            recent_acc.append(acc)

            if step % 50 == 0:
                rolling = sum(recent_acc[-50:]) / max(len(recent_acc[-50:]), 1)
                print(f"  Step {step:4d} | Loss: {loss.item():.4f} | "
                      f"RLF Acc: {acc:.2f} | Rolling(50): {rolling:.2f}")

            if step > 0 and step % 500 == 0:
                torch.save(model.state_dict(),
                           f"saved_weights/mamba130m_phase5_recovery_step{step}.pt")

            # Save best
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(),
                           "saved_weights/mamba130m_phase5_recovery_best.pt")

            step += 1

    # Final save
    torch.save(model.state_dict(), "saved_weights/mamba130m_phase5_recovery_best.pt")
    print("\nPhase 5 Complete → saved_weights/mamba130m_phase5_recovery_best.pt")
    print(f"Best RLF Acc: {best_acc:.2f}")


if __name__ == "__main__":
    phase5_recovery()
