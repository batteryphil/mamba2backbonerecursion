import torch
import os
from torch.utils.data import DataLoader
from mamba_ssm import MambaLMHeadModel

from mamba1_engine import RecursiveMamba1_PrefixScratchpad, MODEL_ID
from dataset_rlf import RLFAdversarialDataset, collate_rlf

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Mamba-130m Phase 3: Adversarial Generalization on {device}")
    
    os.makedirs("saved_weights", exist_ok=True)
    
    # Load Base
    backbone = MambaLMHeadModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device=device)
    model = RecursiveMamba1_PrefixScratchpad(backbone, lora_rank=4).to(device)
    
    # Load Phase 2 checkpoint
    p2_ckpt = "saved_weights/mamba130m_phase2_joint_best.pt"
    if os.path.exists(p2_ckpt):
        model.load_state_dict(torch.load(p2_ckpt, map_location=device))
        print(f"Successfully loaded Phase 2 checkpoint: {p2_ckpt}")
    else:
        print(f"ERROR: Could not find {p2_ckpt}. Please run Phase 2 first.")
        return

    # Unfreeze LoRA (same frozen/unfrozen status as Phase 2)
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
            
    # Optimizer for adversarial phase
    params = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad], "lr": 5e-5, "weight_decay": 0.01}
    ]
    optimizer = torch.optim.AdamW(params)
    
    # Phase 3 uses ADVERSARIAL data (variable chaos and semantic prose)
    dataset = RLFAdversarialDataset(size=12000, seq_len=512, mode="adversarial")
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_rlf, drop_last=True)
    
    model.train()
    step = 0
    total_steps = 3000
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
                print(f"Phase 3 Step {step} | Loss {loss.item():.4f} | RLF Acc {acc:.2f} | Ans Acc {ans_acc:.2f} | Halt Acc {halt_acc:.2f}")
                
            if step > 0 and step % 500 == 0:
                torch.save(model.state_dict(), f"saved_weights/mamba130m_phase3_adversarial_step{step}.pt")
                
            recent_accs.append(acc)
            if len(recent_accs) > 50:
                recent_accs.pop(0)
                
            avg_acc = sum(recent_accs) / len(recent_accs)
            if step > 500 and avg_acc >= 0.97:
                print(f"Early stopping at step {step}! Moving average RLF Acc reached {avg_acc:.3f} >= 0.97")
                break
                
            step += 1
            
    torch.save(model.state_dict(), "saved_weights/mamba130m_phase3_adversarial_best.pt")
    print("Phase 3 Complete -> saved_weights/mamba130m_phase3_adversarial_best.pt")

if __name__ == "__main__":
    train()
