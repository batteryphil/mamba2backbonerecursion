import torch
import random
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel

from mamba2_8b_engine import (
    RecursiveMamba1_PrefixScratchpad,
    freeze_for_phase1,
    get_phase1_optimizer,
    MODEL_ID,
    MAX_LOOPS,
    HALT_ID
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 40
BATCH = 2
ACCUM = 2
BASE_REPO = "state-spaces/mamba-2.8b-slimpj"

# Load Linguistic Dataset from Phase 8 setup
from train_phase8_mixed import Phase8Dataset, _MC_QUESTIONS
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tok.pad_token = tok.eos_token

class MemoryWarmupDataset(Dataset):
    def __init__(self, size=8000):
        self.size = size
        self.rng = random.Random(42)
    def __len__(self): return self.size
    def __getitem__(self, idx):
        # We exclusively use the Multiple Choice Strings to bind Text Variables
        q, a = self.rng.choice(_MC_QUESTIONS)
        prompt = f"[QA] Q: {q}\nSolution: "
        
        p_ids = tok.encode(prompt, add_special_tokens=False)
        a_ids = tok.encode(a, add_special_tokens=False)
        return p_ids, a_ids

def collate_engine(batch):
    inputs, chain_targets, ans_starts = [], [], []
    for p_ids, a_ids in batch:
        n_loops = random.randint(2, max(2, MAX_LOOPS))
        
        # In the engine, the forward pass explicitly generates loops natively via run_one_loop.
        # It takes input_ids to create the prompt embed, and expects chain_targets for supervision.
        # So we only feed the prompt into input_ids.
        
        # ans_starts indicates where the LM head calculates loss (at the end of prompt).
        # We append a pad to inputs so it passes the engine's strict `as_ < max_len` bounding check
        inputs.append(p_ids + [tok.pad_token_id])
        chain_targets.append(a_ids + [HALT_ID])
        ans_starts.append(len(p_ids))
    
    # Pad inputs
    max_len = max(len(x) for x in inputs)
    padded = [torch.nn.functional.pad(torch.tensor(x), (0, max_len - len(x)), value=tok.pad_token_id) for x in inputs]
    
    return torch.stack(padded), chain_targets, ans_starts

def train():
    print("[INIT] Loading Mamba-2.8B Backbone...")
    backbone = MambaLMHeadModel.from_pretrained(BASE_REPO, dtype=torch.bfloat16, device=DEVICE)
    model = RecursiveMamba1_PrefixScratchpad(backbone, lora_rank=8).to(DEVICE)
    
    print("[INIT] Attaching Phase 7.5 GOLDEN LoRA Matrix for native logic preservation...")
    st = torch.load("checkpoints_2_8b/mamba2.8b_p75_GOLDEN.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(st, strict=False)
    
    print("[INIT] Activating Gradient Surgery: Phase 1 Warmup (Prefix + Bridge only)...")
    freeze_for_phase1(model)
    opt = get_phase1_optimizer(model)
    
    ds = MemoryWarmupDataset()
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate_engine)
    
    model.train()
    step = 0
    os.makedirs("checkpoints_phase1_engine", exist_ok=True)
    
    recent_loss = []
    accum_loss = 0.0
    
    while step < STEPS:
        for inputs, targets, starts in loader:
            if step >= STEPS: break
            
            inputs = inputs.to(DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                # We use sparse_reward=True so we provide progressive gradients across loops
                # loss_weights array sets gradient strength per loop
                weights = [0.1] * MAX_LOOPS + [1.0, 1.0] # weak supervision on dark loops, max on answer and halt
                loss, avg_acc, ans_acc, halt_acc = model(
                    inputs, chain_targets=targets, ans_starts=starts, 
                    sparse_reward=True, loss_weights=weights, n_dark_loops=MAX_LOOPS-2
                )
                if loss is None or torch.isnan(loss):
                    continue
                loss = loss / ACCUM
            
            loss.backward()
            accum_loss += loss.item()
            
            if (step + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                
                recent_loss.append(accum_loss)
                if len(recent_loss) > 20: recent_loss.pop(0)
                
                print(f"Step {step+1:4d} | Batch Loss: {accum_loss:.4f} | AvgAcc: {avg_acc:.2f} | AnsAcc: {ans_acc:.2f} | HaltAcc: {halt_acc:.2f}")
                accum_loss = 0.0
            
            step += 1

    torch.save(model.state_dict(), "checkpoints_phase1_engine/mamba2.8b_engine_warmup.pt")
    print("[DONE] Weights saved to checkpoints_phase1_engine/mamba2.8b_engine_warmup.pt")

if __name__ == "__main__":
    train()
