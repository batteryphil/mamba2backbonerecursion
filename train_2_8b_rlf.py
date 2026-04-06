"""
train_2_8b_rlf.py — VRAM-Conscious RLF Training for Native Mamba-2.8B
=====================================================================
Uses ItsMick's observation that Mamba natively handles O(1) loop state
over sequence time. We bypass custom engines entirely and train the 
base model on sequences filled with spacer tokens `=` to simulate latent loops.

Designed to fit in 12GB VRAM by:
- Loading the base Mamba-2.8B weights natively onto GPU.
- Freezing the bottom 32 layers.
- Using LoRA Rank 8 on the top 32 layers.
- Using Batch Size 1 with Gradient Accumulation.
"""

import os
import sys
import time
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import safetensors.torch
from mamba_ssm import MambaLMHeadModel
from mamba_ssm.models.mixer_seq_simple import MambaConfig
from transformers import AutoTokenizer

# ── Config ───────────────────────────────────────────────────────────────────
SCND_CKPT_DIR = "checkpoints_2_8b"
BASE_WEIGHTS  = "/home/phil/Desktop/mamba_7b_engine/checkpoints/mamba-2.8b-latent-final/model.safetensors"
LOG_PATH      = "training_2_8b.log"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

STEPS         = 2000
BATCH         = 1
ACCUM_STEPS   = 4
LR            = 1e-4

BASE_SPLIT    = 32
LORA_RANK     = 8
MAX_LOOPS     = 10

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>", "<HALT>"]})
HALT_ID   = tokenizer.convert_tokens_to_ids("<HALT>")
SPACER_ID = tokenizer.convert_tokens_to_ids("=")

# ── LoRA Linear Implementation (for training) ────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.bias = linear.bias
        d_out, d_in = linear.weight.shape
        dtype = linear.weight.dtype
        device = linear.weight.device
        self.register_buffer("base_weight", linear.weight.data.clone())
        self.lora_A = nn.Parameter(torch.empty(rank, d_in, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, dtype=dtype, device=device))
        self.scale = float(alpha) / rank
        nn.init.kaiming_uniform_(self.lora_A)

    @property
    def weight(self) -> torch.Tensor:
        return self.base_weight + self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

def mount_lora(model: MambaLMHeadModel):
    # Freeze lower layers
    for layer in model.backbone.layers[:BASE_SPLIT]:
        for p in layer.parameters():
            p.requires_grad = False
    
    # Mount LoRA on top layers
    for layer in model.backbone.layers[BASE_SPLIT:]:
        mx = layer.mixer
        for attr in ("in_proj", "out_proj"):
            if hasattr(mx, attr):
                setattr(mx, attr, LoRALinear(getattr(mx, attr), rank=LORA_RANK, alpha=LORA_RANK * 2.0))
                
# ── Dataset (Phase 7 General Recovery Mixed) ─────────────────────────────────
_MC_QUESTIONS = [
    ("Q: What is the largest ocean? A) Atlantic B) Indian C) Pacific D) Arctic", "C"),
    ("Q: Which planet is closest to the Sun? A) Earth B) Mars C) Mercury D) Venus", "C"),
    ("Q: What gas do plants absorb? A) Oxygen B) CO2 C) Nitrogen D) Helium", "B"),
    ("Q: In what year did WWII end? A) 1940 B) 1945 C) 1950 D) 1939", "B"),
    ("Q: What is the hardest natural substance? A) Gold B) Iron C) Diamond D) Quartz", "C"),
]
_TF_QUESTIONS = [
    ("True or False: The Earth is flat.", "False"),
    ("True or False: Water boils at 100 degrees Celsius at sea level.", "True"),
    ("True or False: Spiders are insects.", "False"),
    ("True or False: The capital of Japan is Tokyo.", "True"),
    ("True or False: Sound travels faster in a vacuum.", "False"),
]
_FILLBLANK_QUESTIONS = [
    ("Complete: The primary colors are red, yellow, and ___.", "blue"),
    ("Complete: The chemical symbol for water is ___.", "H2O"),
    ("Complete: A triangle has ___ sides.", "3"),
    ("Complete: The opposite of hot is ___.", "cold"),
    ("Complete: ___ is the process by which plants make food.", "photosynthesis"),
]
_DIRECT_QA = [
    ("Q: What is 4 times 4?", "16"),
    ("Q: Who wrote Hamlet?", "Shakespeare"),
    ("Q: What is the capital of France?", "Paris"),
    ("Q: How many millimeters are in one centimeter?", "10"),
    ("Q: What is the square root of 81?", "9"),
]

_JSON_TOOL_TASKS = [
    ("System Tool Execution. User query: Calculate 7 + 8. Route to JSON schema: {\"status\": \"success\", \"computation\": X}", "```json\n{\n  \"status\": \"success\",\n  \"computation\": 15\n}\n```"),
    ("System Tool Execution. User query: Multiply 12 by 4. Route to JSON schema: {\"result_type\": \"math\", \"value\": X}", "```json\n{\n  \"result_type\": \"math\",\n  \"value\": 48\n}\n```"),
    ("System Tool Execution. User query: Compute the difference between 100 and 45. Route to JSON schema: {\"action\": \"subtract\", \"answer\": X}", "```json\n{\n  \"action\": \"subtract\",\n  \"answer\": 55\n}\n```"),
    ("System Tool Execution. User query: What is 5 squared? Route to JSON schema: {\"operation\": \"pow\", \"output\": X}", "```json\n{\n  \"operation\": \"pow\",\n  \"output\": 25\n}\n```"),
    ("System Tool Execution. User query: Add 15, 10, and 5. Route to JSON schema: {\"sum\": X}", "```json\n{\n  \"sum\": 30\n}\n```"),
    ("System Tool Execution. User query: A user wants you to calculate the monthly cost of a $1200 yearly subscription. Return ONLY structural JSON: {\"monthly_cost\": int}.", "```json\n{\n  \"monthly_cost\": 100\n}\n```")
]

def generate_chain(rng: random.Random) -> tuple[str, str]:
    hops = rng.randint(2, 5)
    val  = str(rng.randint(1, 9999))
    parts = [f"V1={val}."]
    for i in range(2, hops + 1):
        parts.append(f"V{i}=V{i-1}.")
    parts.append(f"What is V{hops}?")
    return " ".join(parts), val

class ScaleUpDataset(torch.utils.data.Dataset):
    def __init__(self, size=2000, mix_ratio=0.5):
        self.size = size
        self.rng = random.Random(42)
        self.mix = mix_ratio
        self.general_pools = _MC_QUESTIONS + _TF_QUESTIONS + _FILLBLANK_QUESTIONS + _DIRECT_QA

    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # 50% RLF Math Chains, 50% General Formatting Recall
        rand = self.rng.random()
        if rand > 0.50:
            q, a = generate_chain(self.rng)
            prompt = f"[LOGIC] {q}\nSolution: "
            is_reasoning = True
        else:
            q, a = self.rng.choice(self.general_pools)
            prompt = f"[QA] {q}\nAnswer: "
            is_reasoning = False
            
        return prompt, a, is_reasoning

def collate_fn(batch):
    prompts = [item[0] for item in batch]
    answers = [item[1] for item in batch]
    reasoning_flags = [item[2] for item in batch]
    
    input_ids_list = []
    labels_list    = []

    for prompt, ans, is_reason in zip(prompts, answers, reasoning_flags):
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        a_ids = tokenizer.encode(ans, add_special_tokens=False)
        
        if is_reason:
            n_loops = random.randint(2, MAX_LOOPS)
            ans_prefix_ids = tokenizer.encode("\\nAnswer: ", add_special_tokens=False)
            spacer_ids = [SPACER_ID] * n_loops + ans_prefix_ids
        else:
            n_loops = random.randint(0, 1) # Recall fast-path
            spacer_ids = [SPACER_ID] * n_loops
        
        full_seq = p_ids + spacer_ids + a_ids + [HALT_ID]
        # Labels: -100 for prompt, predict spacer/answer/halt
        lbl = [-100] * len(p_ids) + spacer_ids + a_ids + [HALT_ID]
        
        input_ids_list.append(full_seq)
        labels_list.append(lbl)
    
    max_len = max(len(ids) for ids in input_ids_list)
    padded_inputs, padded_labels = [], []
    for seq, lbl in zip(input_ids_list, labels_list):
        pad_len = max_len - len(seq)
        padded_inputs.append(seq + [tokenizer.eos_token_id] * pad_len)
        padded_labels.append(lbl + [-100] * pad_len)
    
    return torch.tensor(padded_inputs, dtype=torch.long), torch.tensor(padded_labels, dtype=torch.long)

def main():
    print(f"\n{'='*70}")
    print(f"  2.8B NATIVE O(1) RLF SEQUENCE TRAINING (ItsMick Protocol)")
    print(f"  Device:     {DEVICE.upper()}")
    print(f"  Base Model: {BASE_WEIGHTS}")
    print(f"  Batch:      {BATCH} x {ACCUM_STEPS} accum")
    print(f"{'='*70}\n")
    
    os.makedirs(SCND_CKPT_DIR, exist_ok=True)
    
    print("[INIT] Loading Pristine state-spaces/mamba-2.8b-slimpj Base Backbone Native to GPU...")
    model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-2.8b-slimpj", dtype=torch.bfloat16, device=DEVICE)
    
    # 1. EXPLICITLY FREEZE 100% OF THE PRISTINE BASE MODEL MATRICES
    for p in model.parameters():
        p.requires_grad = False

    print("[INIT] Mounting LoRA Parameters...")
    mount_lora(model)
    model.train()
        
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    dataset = ScaleUpDataset(size=STEPS * BATCH)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, collate_fn=collate_fn, drop_last=True)
    
    print("\n[START] Native Sequence Training Loop:")
    step = 0
    running_loss = 0.0
    optimizer.zero_grad()
    t_start = time.time()
    
    for inputs, labels in loader:
        if step >= STEPS: break
        
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # MambaLMHeadModel forward
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            out = model(inputs)
            logits = out.logits
            
            # Standard causal shift
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            loss = criterion(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [WARN] Step {step} — NaN Loss, skipping.")
            step += 1
            continue
            
        scaled_loss = loss / ACCUM_STEPS
        scaled_loss.backward()
        running_loss += loss.item()
        
        if (step + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            avg_running = running_loss / ACCUM_STEPS
            elapsed = time.time() - t_start
            
            print(f"Seq Step {step+1:4d} | Loss {avg_running:.4f} | {elapsed:.0f}s")
            running_loss = 0.0
            
            if (step + 1) % 200 == 0:
                ckpt = f"{SCND_CKPT_DIR}/mamba2.8b_seq_step{step+1}.pt"
                # Map trainable keys explicitly since state_dict tensors drop requires_grad
                trainable_keys = {name for name, p in model.named_parameters() if p.requires_grad}
                trainable = {k: v for k, v in model.state_dict().items() if k in trainable_keys}
                torch.save(trainable, ckpt)

        step += 1

    print("\n🏁 Native Sequence Training Complete")
    final_ckpt = f"{SCND_CKPT_DIR}/mamba2.8b_seq_final.pt"
    trainable_keys = {name for name, p in model.named_parameters() if p.requires_grad}
    trainable = {k: v for k, v in model.state_dict().items() if k in trainable_keys}
    torch.save(trainable, final_ckpt)
    print(f"Saved -> {final_ckpt}")

if __name__ == "__main__":
    main()
