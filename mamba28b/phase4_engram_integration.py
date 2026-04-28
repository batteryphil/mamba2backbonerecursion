"""
phase4_engram_integration.py — Factual Offload & Context Gating
====================================================================
Implements a DeepSeek-style CPU "Engram" hash table. 
Trains the Mamba Latent Bridge to selectively gate external memory 
injections without breaking its continuous reasoning loop.
"""

import torch
import random
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel

from mamba1_engine import RecursiveMamba1_PrefixScratchpad, MODEL_ID, tokenizer
from dataset_rlf import collate_rlf

# Simulated CPU-side Engram Factual Table (In reality, this would be a massive DB on your NVMe)
CPU_ENGRAM_TABLE = {
    "capital of france": "Paris",
    "powerhouse of the cell": "Mitochondria",
    "speed of light": "299,792 km/s",
    "largest planet": "Jupiter",
    "boiling point of water": "100C"
}

def generate_engram_curriculum(size=2000, hops=5):
    """
    Generates two types of prompts:
    1. Factual Need (70%): The model MUST accept the Engram injection to solve it.
    2. Poisoned Logic (30%): The model MUST reject the Engram injection to solve it.

    70/30 split forces the gate to learn bidirectional polarization.
    Only clean binary labels (0.0 or 1.0) — no ambiguous 0.5 mixed batches.
    """
    dataset = []
    n_factual = int(size * 0.60)
    n_poison  = size - n_factual

    for _ in range(n_factual):
        # 1. Factual Need — gate target = 1.0
        fact_key, fact_val = random.choice(list(CPU_ENGRAM_TABLE.items()))
        text = f"Var_1 = The {fact_key}. Var_2 = Var_1. "
        for i in range(3, hops + 1):
            text += f"Var_{i} = Var_{i-1}. "
        text += f"What is Var_{hops}? Answer:"
        cpu_injection    = f" [ENGRAM: {fact_val}]"
        target_reasoning = f" {fact_val} <HALT>"
        dataset.append((text, target_reasoning, cpu_injection, 1.0))

    colors = ["Red", "Quantum", "Void", "Titanium", "Azure", "Neon"]
    for _ in range(n_poison):
        # 2. Poisoned Logic — gate target = 0.0
        val = random.choice(colors)
        text = f"Alpha = {val}. Beta = Alpha. "
        for i in range(3, hops + 1):
            text += f"Var_{i} = Var_{i-1}. "
        text += f"What is Var_{hops}? Answer:"
        cpu_injection    = f" [ENGRAM: {random.choice(list(CPU_ENGRAM_TABLE.values()))}]"
        target_reasoning = f" {val} <HALT>"
        dataset.append((text, target_reasoning, cpu_injection, 0.0))

    return dataset


class Phase4Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.pad_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, target_reasoning, cpu_injection, must_accept = self.data[idx]
        
        input_ids = tokenizer.encode(text, return_tensors="pt")[0]
        injection_ids = tokenizer.encode(cpu_injection, return_tensors="pt")[0]
        
        # Target logic formatting (for the mamba1_engine to compute RLF accuracy)
        target_ids = tokenizer.encode(target_reasoning, return_tensors="pt")[0]
        
        # Determine where the answer officially starts (for ans_acc)
        ans_start = 0 
        
        return {
            "input_ids": input_ids,
            "injection_ids": injection_ids,
            "target_ids": target_ids,
            "ans_start": len(input_ids) - 1,  # last valid token position (avoid out-of-bounds)
            "must_accept": must_accept
        }

def phase4_collate(batch):
    pad_id = tokenizer.eos_token_id
    
    # Collect lengths
    # We must pad everything dynamically to seq_len
    input_ids = [b["input_ids"] for b in batch]
    injection_ids = [b["injection_ids"] for b in batch]
    target_ids = [b["target_ids"] for b in batch]
    ans_starts = torch.tensor([b["ans_start"] for b in batch])
    must_accepts = torch.tensor([b["must_accept"] for b in batch], dtype=torch.float32)
    
    # Pad input_ids
    max_in = max(len(x) for x in input_ids)
    in_padded = torch.zeros(len(batch), max_in, dtype=torch.long) + pad_id
    for i, x in enumerate(input_ids):
        in_padded[i, :len(x)] = x
        
    # Pad injection_ids
    max_inj = max(len(x) for x in injection_ids)
    inj_padded = torch.zeros(len(batch), max_inj, dtype=torch.long) + pad_id
    for i, x in enumerate(injection_ids):
        inj_padded[i, :len(x)] = x
        
    # Pad target_ids
    max_tgt = max(len(x) for x in target_ids)
    tgt_padded = torch.zeros(len(batch), max_tgt, dtype=torch.long) - 100
    for i, x in enumerate(target_ids):
        tgt_padded[i, :len(x)] = x
        
    return in_padded, inj_padded, tgt_padded, ans_starts, must_accepts


# ── Pseudo-code for the Phase 4 Training Loop ─────────────────────────────
def phase4_training_loop():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print("  INITIALIZING PHASE 4: ENGRAM GATING")
    print(f"{'='*70}\n")
    
    # Load Model
    backbone = MambaLMHeadModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device=device)
    model = RecursiveMamba1_PrefixScratchpad(backbone, lora_rank=4).to(device)
    
    # Load latest Phase 4 epoch checkpoint if available, else fall back to Phase 3
    import glob
    p4_ckpts = sorted(glob.glob("saved_weights/mamba130m_phase4_engram_epoch*.pt"))
    p3_ckpt = "saved_weights/mamba130m_phase3_adversarial_best.pt"
    if p4_ckpts:
        ckpt = p4_ckpts[-1]
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        print(f"Resuming Phase 4 from: {ckpt}")
    elif os.path.exists(p3_ckpt):
        model.load_state_dict(torch.load(p3_ckpt, map_location=device), strict=False)
        print(f"Successfully loaded Phase 3 checkpoint: {p3_ckpt}")
    else:
        print(f"ERROR: No checkpoint found. Starting from scratch.")
            
    # Unfreeze LoRA
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
            
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "engram_gate_head" not in n], "lr": 5e-5, "weight_decay": 0.01},
        {"params": list(model.engram_gate_head.parameters()), "lr": 2e-4, "weight_decay": 0.01},  # Gate head polarizes faster than backbone
    ])
    
    phase4_data = generate_engram_curriculum(size=10000, hops=5)
    random.shuffle(phase4_data)
    dataset = Phase4Dataset(phase4_data)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=phase4_collate, drop_last=True)

    print(f"Generated {len(phase4_data)} Engram Gating Scenarios.")
    model.train()
    step = 0
    epoch = 0
    steps_per_epoch = 3000
    recent_gate_correct = []

    def gate_goal_achieved():
        """Returns True if 50-batch rolling gate accuracy is >= 80%."""
        if len(recent_gate_correct) < 50:
            return False
        return sum(recent_gate_correct[-50:]) / 50 >= 0.80

    while True:
        epoch += 1
        epoch_end = epoch * steps_per_epoch
        print(f"\n[Epoch {epoch}] Running steps {step} → {epoch_end}...")

        for input_ids, injection_ids, target_ids, ans_starts, must_accepts in loader:
            if step >= epoch_end:
                break

            input_ids = input_ids.to(device)
            injection_ids = injection_ids.to(device)
            target_ids = target_ids.to(device)
            ans_starts = ans_starts.to(device)
            must_accepts = must_accepts.to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                # ── Gate-only pass: compute gate from injection semantics ─────
                _, gate_logit, gate_value = model.forward_with_engram(
                    input_ids,
                    injection_ids,
                    chain_targets=None,
                    ans_starts=None
                )

                # ── RLF pass: train on prompt-only (stable, no concat NaN) ──
                base_loss, acc, ans_acc, halt_acc = model(
                    input_ids,
                    chain_targets=target_ids,
                    ans_starts=ans_starts
                )

                # NaN guard
                if torch.isnan(base_loss):
                    step += 1
                    continue

                # 4. Phase 4 Gate Loss — per-sample BCEWithLogitsLoss
                # batch_size=1 guarantees target is always pure 0.0 or 1.0
                target_gate = must_accepts[0].to(gate_logit.dtype)  # scalar, no mean needed
                gate_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    gate_logit.unsqueeze(0), target_gate.unsqueeze(0)
                )
                total_loss = base_loss + (1.0 * gate_loss)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track rolling gate accuracy
            gate_correct = (gate_value.item() > 0.5) == (target_gate.item() > 0.5)
            recent_gate_correct.append(float(gate_correct))

            if step % 20 == 0:
                action = "ACCEPTED" if gate_value.item() > 0.5 else "REJECTED"
                correct_action = "✅" if gate_correct else "❌"
                rolling = sum(recent_gate_correct[-50:]) / max(len(recent_gate_correct[-50:]), 1)
                print(f"Phase 4 Step {step} | Loss: {total_loss.item():.4f} | RLF Acc {acc:.2f} | Gate: {gate_value.item():.2f} ({action}) {correct_action} | Target: {target_gate.item():.1f} | GateAcc(50): {rolling:.2f}")

            if step > 0 and step % 500 == 0:
                torch.save(model.state_dict(), f"saved_weights/mamba130m_phase4_engram_step{step}.pt")

            step += 1

        # End of epoch — check if goal achieved
        torch.save(model.state_dict(), f"saved_weights/mamba130m_phase4_engram_epoch{epoch}.pt")
        if gate_goal_achieved():
            print(f"\n✅ Gate goal achieved at step {step} (epoch {epoch})! GateAcc(50) >= 80%")
            break
        else:
            rolling = sum(recent_gate_correct[-50:]) / max(len(recent_gate_correct[-50:]), 1)
            print(f"\n⚠️  Gate goal not yet achieved at step {step} (GateAcc={rolling:.2f}). Extending by {steps_per_epoch} more steps...")

    torch.save(model.state_dict(), "saved_weights/mamba130m_phase4_engram_best.pt")
    print("Phase 4 Complete -> saved_weights/mamba130m_phase4_engram_best.pt")


if __name__ == "__main__":
    phase4_training_loop()
