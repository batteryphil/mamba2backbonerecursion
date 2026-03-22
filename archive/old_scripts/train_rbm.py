"""
DiM-LLM v4.0 Evolution: Recurrent Bidirectional Mamba (RBM)
==========================================================
Next-Token Prediction Training Script with internal reasoning loops.
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import time
import copy
import random
import glob
from transformers import GPT2Tokenizer
from mamba_rbm import RecursiveMambaLM, Config

# ── Linux-specific performance flags ──────────────────────────────────────────
_phys_cores = max(1, os.cpu_count() // 2)
os.environ.setdefault("OMP_NUM_THREADS", str(_phys_cores))
os.environ.setdefault("MKL_NUM_THREADS", str(_phys_cores))
torch.set_num_threads(_phys_cores)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

import argparse

# ── Hyper-parameters ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data",       type=str,   default="logic_v2.json")
parser.add_argument("--lr",         type=float, default=1e-4)
parser.add_argument("--epochs",     type=int,   default=20)
parser.add_argument("--batch_size", type=int,   default=2)
parser.add_argument("--seq_len",    type=int,   default=1024)
args = parser.parse_args()

DSR_FILE       = args.data
BATCH_SIZE     = args.batch_size
EPOCHS         = args.epochs
SEQ_LEN        = 1024 
LR             = args.lr
EARLY_STOP_PAT = 5
ACCUM_STEPS    = 4

# ── Helpers ───────────────────────────────────────────────────────────────────

def pick_latest_checkpoint() -> str | None:
    for candidate in ["rbm_v1_checkpoint.pt", "rbm_v1_best.pt"]:
        if os.path.exists(candidate):
            return candidate
    return None

def build_dsr_chunks(tokenizer: GPT2Tokenizer, seq_len: int) -> list:
    if not os.path.exists(DSR_FILE):
        return []
    with open(DSR_FILE, "r") as f:
        dsr_list = json.load(f)
    
    all_ids = []
    for item in dsr_list:
        text = item.get("text", "") if isinstance(item, dict) else item
        u_ids = tokenizer.encode(text + " " + tokenizer.eos_token, add_special_tokens=False)
        all_ids.extend(u_ids)
    
    num_chunks = len(all_ids) // seq_len
    return [torch.tensor(all_ids[i*seq_len : (i+1)*seq_len]) for i in range(num_chunks)]

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    config = Config(vocab_size=len(tokenizer), d_model=1024, n_layers=8, seq_len=SEQ_LEN)
    model = RecursiveMambaLM(config).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    ckpt_path = pick_latest_checkpoint()
    start_epoch = 0
    if ckpt_path:
        print(f"🛠️ Resuming RBM from {ckpt_path}...")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        
        # 🧪 HOT-PATCH: Override Weight Decay after loading state
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = 0.05
        print("ADJUSTMENT: Weight Decay tightened to 0.05 for Token Separation.")
        
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = state.get("epoch", 0)

    train_chunks = build_dsr_chunks(tokenizer, SEQ_LEN)
    print(f"🚀 RBM Manifest: {len(train_chunks)} chunks | LR={LR}")

    best_val_loss = float('inf')
    patience_counter = 0
    stats = {
        "train_loss": [],
        "val_loss": [],
        "salads": [],
        "tps": 0.0,
        "step": 0,
        "diverged": False
    }

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        random.shuffle(train_chunks)
        epoch_loss = 0
        start_time = time.time()
        
        for i in range(0, len(train_chunks), BATCH_SIZE):
            batch = torch.stack(train_chunks[i:i+BATCH_SIZE]).to(device)
            
            # --- AR Objective: Next Token Prediction ---
            inputs = batch[:, :-1].contiguous()
            targets = batch[:, 1:].contiguous()
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(inputs)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
            
            (loss / ACCUM_STEPS).backward()
            
            if (i // BATCH_SIZE + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            stats["step"] += 1
            
            if (i // BATCH_SIZE) % 5 == 0:
                elapsed = time.time() - start_time
                tps = (i + BATCH_SIZE) * SEQ_LEN / (elapsed + 1e-6)
                stats["tps"] = tps
                print(f"Epoch {epoch+1} | Step {i//BATCH_SIZE} | Loss: {loss.item():.4f} | TPS: {tps:.0f}")
                with open("training_stats.json", "w") as f:
                    json.dump(stats, f)

        avg_loss = epoch_loss / (len(train_chunks) / BATCH_SIZE)
        stats["train_loss"].append(avg_loss)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        
        # --- Diagnostic Sampling (The "Salad") ---
        model.eval()
        test_prompt = "What is the capital of France?"
        prompt_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(prompt_ids, max_new_tokens=64)
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
        stats["salads"].append([{
            "type": "REASONING_PROBE",
            "prompt": test_prompt,
            "response": response,
            "steps": config.n_reasoning,
            "mode": "RBM_AR"
        }])

        # Save check
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(), "epoch": epoch}, "rbm_v1_best.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            
        scheduler.step()
        with open("training_stats.json", "w") as f:
            json.dump(stats, f)

        if patience_counter >= EARLY_STOP_PAT:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    train()
