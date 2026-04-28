import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel
import os

from mamba1_engine import (
    RecursiveMamba1_PrefixScratchpad, 
    freeze_for_phase1, 
    get_phase1_optimizer,
    MODEL_ID,
    tokenizer
)
from dataset_rlf import RLFAdversarialDataset, collate_rlf

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Mamba-130m RLF Training on {device}")
    
    os.makedirs("saved_weights", exist_ok=True)
    
    # Load Mamba-1 model
    print(f"Loading backbone {MODEL_ID}...")
    backbone = MambaLMHeadModel.from_pretrained(
        MODEL_ID, 
        dtype=torch.bfloat16, 
        device=device
    )
    
    # Wrap in our Mamba-1 Engine
    model = RecursiveMamba1_PrefixScratchpad(backbone, lora_rank=4).to(device)
    
    # Training configurations
    dataset = RLFAdversarialDataset(size=20000, seq_len=512)
    loader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=collate_rlf, 
        drop_last=True
    )
    
    # ── PHASE 1: Warmup ───────────────────────────────────────────────────────
    print("\nStarting PHASE 1: Warmup (Training Scratchpad + Bridge only)")
    freeze_for_phase1(model)
    optimizer = get_phase1_optimizer(model)
    
    model.train()
    step = 0
    phase1_steps = 500
    
    for inputs, targets, starts in loader:
        if step >= phase1_steps:
            break
            
        inputs = inputs.to(device)
        
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            loss, acc, ans_acc, halt_acc = model(inputs, chain_targets=targets, ans_starts=starts)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 10 == 0:
            print(f"P1 Step {step} | Loss {loss.item():.4f} | RLF Acc {acc:.2f} | Ans Acc {ans_acc:.2f} | Halt Acc {halt_acc:.2f}")
            
        step += 1
        
    torch.save(model.state_dict(), "saved_weights/mamba130m_rlf_phase1_best.pt")
    
    # ── PHASE 2: Joint Training ───────────────────────────────────────────────
    print("\nStarting PHASE 2: Joint Training (LoRA + Bridge + Scratchpad)")
    
    # Unfreeze LoRA
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
            
    # New optimizer for joint phase
    params = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad], "lr": 1e-4, "weight_decay": 0.01}
    ]
    optimizer = torch.optim.AdamW(params)
    
    step = 0
    phase2_steps = 2500
    
    for inputs, targets, starts in loader:
        if step >= phase2_steps:
            break
            
        inputs = inputs.to(device)
        
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            loss, acc, ans_acc, halt_acc = model(inputs, chain_targets=targets, ans_starts=starts)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 10 == 0:
            print(f"P2 Step {step} | Loss {loss.item():.4f} | RLF Acc {acc:.2f} | Ans Acc {ans_acc:.2f} | Halt Acc {halt_acc:.2f}")
            
        # Checkpoint every 50 steps
        if step > 0 and step % 50 == 0:
            torch.save(model.state_dict(), f"saved_weights/mamba130m_rlf_step{step}.pt")
            print(f"Checkpoint saved at step {step}")
            
        step += 1
        
    print("Training Complete!")

if __name__ == "__main__":
    train()
