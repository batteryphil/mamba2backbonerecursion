import torch
import os
from torch.utils.data import DataLoader
from mamba_ssm import MambaLMHeadModel

from mamba1_engine import (
    RecursiveMamba1_PrefixScratchpad, 
    freeze_for_phase1, 
    get_phase1_optimizer,
    MODEL_ID
)
from dataset_rlf import RLFAdversarialDataset, collate_rlf

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Mamba-130m Phase 1: Warmup on {device}")
    
    os.makedirs("saved_weights", exist_ok=True)
    
    # Load Base
    backbone = MambaLMHeadModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device=device)
    model = RecursiveMamba1_PrefixScratchpad(backbone, lora_rank=4).to(device)
    
    # Check if checkpoint exists to resume
    ckpt_path = "saved_weights/mamba130m_phase1_scratchpad.pt"
    start_step = 0
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Resuming from {ckpt_path}")
        start_step = 1000
    
    # Phase 1 uses CLEAN data
    dataset = RLFAdversarialDataset(size=4000, seq_len=512, mode="clean")
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_rlf, drop_last=True)
    
    # Phase 1 Freeze
    freeze_for_phase1(model)
    optimizer = get_phase1_optimizer(model)
    
    model.train()
    step = start_step
    total_steps = 10000 # Let auto-stop catch it
    recent_accs = []
    
    while step < total_steps:
        for inputs, targets, starts in loader:
            if step >= total_steps:
                break
                
            inputs = inputs.to(device)
            optimizer.zero_grad()
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                loss, acc, ans_acc, halt_acc = model(inputs, chain_targets=targets, ans_starts=starts)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if step % 20 == 0:
                print(f"Phase 1 Step {step} | Loss {loss.item():.4f} | RLF Acc {acc:.2f} | Ans Acc {ans_acc:.2f} | Halt Acc {halt_acc:.2f}")
                
            if step > 0 and step % 500 == 0:
                torch.save(model.state_dict(), f"saved_weights/mamba130m_phase1_step{step}.pt")
                
            recent_accs.append(acc)
            if len(recent_accs) > 50:
                recent_accs.pop(0)
                
            avg_acc = sum(recent_accs) / len(recent_accs)
            if step > start_step + 100 and avg_acc >= 0.97:
                print(f"Early stopping at step {step}! Moving average RLF Acc reached {avg_acc:.3f} >= 0.97")
                break
                
            step += 1
            
    torch.save(model.state_dict(), "saved_weights/mamba130m_phase1_scratchpad.pt")
    print("Phase 1 Complete -> saved_weights/mamba130m_phase1_scratchpad.pt")

if __name__ == "__main__":
    train()
