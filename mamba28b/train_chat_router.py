import torch
import random
import os
import time
from torch.utils.data import Dataset, DataLoader
from mamba_ssm import MambaLMHeadModel
from mamba1_engine import MODEL_ID, tokenizer, RecursiveMamba1_PrefixScratchpad

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use a weird token as THINK, e.g., <|endoftext|> is already 0, let's just use the word "<THINK>"
# or better yet, we can use a dedicated special token if the tokenizer supports it.
# We'll just define the text string "<THINK>" and check if it generates it verbatim.
THINK_TEXT = "<THINK>"
THINK_IDS = tokenizer.encode(THINK_TEXT, add_special_tokens=False)

PAD_ID = tokenizer.eos_token_id

# ── Dummy Conversational Dataset ──────────────────────────────────────────────
# A fast synthetic dataset to teach basic English generation and prevent
# catastrophic forgetting of natural language syntax.
QA_PAIRS = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("How are you today?", "I am functioning optimally as an AI assistant."),
    ("What is 2+2?", "2+2 is 4."),
    ("Tell me a greeting.", "Hello! How can I assist you?"),
    ("What color is the sky?", "The sky is typically blue during the day."),
    ("Who wrote Hamlet?", "William Shakespeare wrote Hamlet."),
    ("What is the speed of light?", "Approximately 299,792 kilometers per second."),
    ("Are you an AI?", "Yes, I am a neural network reasoning engine."),
    ("What is water?", "Water is a transparent fluid which forms the world's streams, lakes, oceans and rain."),
    ("Name a planet.", "Mars is a planet in our solar system."),
]

# Numeric payload range for synthetic routing triggers
NUM_MIN = 1
NUM_MAX = 999_999

def _rand_num(rng: random.Random) -> str:
    return str(rng.randint(NUM_MIN, NUM_MAX))

# ── Dataset ───────────────────────────────────────────────────────────────────
class ChatRouterDataset(Dataset):
    """
    Mixes conversational Q&A with synthetic mathematical variables.
    If the prompt is conversational, the target is the English answer.
    If the prompt is a logic chain (e.g. V1=... V2=...), the target is <THINK>.
    """
    def __init__(self, size: int, seed: int = 42):
        self.size = size
        self.seed = seed

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        rng = random.Random(self.seed + idx * 31337)
        
        # 50% chance of conversational QA, 50% chance of a Logic Chain Trigger
        if rng.random() < 0.50:
            q, a = rng.choice(QA_PAIRS)
            prompt = f"User: {q}\nAI: {a}{tokenizer.eos_token}"
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            # Find the start of the AI's response to only calculate loss on the answer
            ai_str = "AI: "
            ans_idx = prompt.find(ai_str) + len(ai_str)
            ans_token_idx = len(tokenizer.encode(prompt[:ans_idx], add_special_tokens=False))
            
            target_ids = input_ids.copy()
            target_ids[:ans_token_idx] = [-100] * ans_token_idx # Ignore prompt in loss
            
        else:
            # Logic Chain Routing Trigger
            hops = rng.randint(2, 6)
            val = _rand_num(rng)
            chain_parts = [f"V1={val}."]
            for i in range(2, hops + 1):
                chain_parts.append(f"V{i}=V{i-1}.")
            chain_parts.append(f"What is V{hops}? Answer:")
            prompt_str = " ".join(chain_parts) + f" {THINK_TEXT}{tokenizer.eos_token}"
            
            input_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
            ans_token_idx = len(tokenizer.encode(" ".join(chain_parts), add_special_tokens=False))
            
            target_ids = input_ids.copy()
            target_ids[:ans_token_idx] = [-100] * ans_token_idx
            
        # Standard causal language modeling setup: shift targets left
        # We'll just return input_ids and let the collate function shift them out
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long)
        }

def collate_fn(batch):
    input_ids = [b["input_ids"][:-1] for b in batch]   # Remove last token from input
    target_ids = [b["target_ids"][1:] for b in batch]  # Shift targets left 1
    
    max_len = max(len(x) for x in input_ids)
    
    inp_pad = torch.stack([
        torch.nn.functional.pad(x, (0, max_len - len(x)), value=PAD_ID)
        for x in input_ids
    ])
    tgt_pad = torch.stack([
        torch.nn.functional.pad(x, (0, max_len - len(x)), value=-100)
        for x in target_ids
    ])
    return inp_pad, tgt_pad


def main():
    print("=" * 70)
    print("  PHASE 3: TRAINING LORA_CHAT ROUTER")
    print("======================================================================")

    steps = 1000
    batch_size = 8
    lr = 5e-4
    
    print("[INIT] Loading Base Mamba-130M backbone...")
    # NOTE: We initialize standard LoRA but NO Latent Bridge since this is just the Chat Router
    # For simplicity, we can load the base model and add standard PEFT LoRA, 
    # OR we can just use our Recursive engine but keep n_dark_inference=0 and never train the bridge.
    # To ensure 100% decoupling from the RLF engine mechanics, let's load a raw backbone.
    # Wait, our RLF architecture wraps the backbone and uses its embedding layer. 
    # If we use `RecursiveMamba1_PrefixScratchpad` and zero out the bridge entirely, it works.
    # But it's cleaner to just use the engine block we already built, and ONLY authorize gradients on LoRA.
    backbone = MambaLMHeadModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device=DEVICE)
    
    # We instantiate the RLF engine just for the LoRA adapters it injects,
    # but we will NEVER execute dark loops or use the bridge here.
    model = RecursiveMamba1_PrefixScratchpad(backbone, lora_rank=4).to(DEVICE)
    
    # Freeze everything except LoRA parameters
    for name, param in model.named_parameters():
        param.requires_grad = "lora" in name.lower()
        
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA_Chat Trainable params: {trainable:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
    )

    dataset = ChatRouterDataset(size=5000)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True
    )

    model.train()
    step = 0
    t_start = time.time()
    
    # Causal LM Loss
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    print("\\n[TRAIN] Beginning Chat Router optimization...")
    while step < steps:
        for inp, tgt in loader:
            if step >= steps:
                break
                
            inp = inp.to(DEVICE)
            tgt = tgt.to(DEVICE)
            
            optimizer.zero_grad()
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                # Standard forward pass without any dark loops or sparse routing
                # Because we hijacked `RecursiveMamba1_PrefixScratchpad.forward` to return a 4-tuple during training,
                # we must extract the logits directly from the backbone to do standard causal LM!
                hidden_states = model.backbone(inp)
                logits = model.lm_head(hidden_states)
                
                # Standard Causal LM loss
                loss = loss_fct(logits.view(-1, logits.size(-1)), tgt.view(-1))
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if step % 50 == 0:
                elapsed = time.time() - t_start
                print(f"Router Step {step:4d} | Causal Loss: {loss.item():.4f} | {elapsed:.0f}s")
                
            step += 1

    os.makedirs("saved_weights", exist_ok=True)
    out_path = "saved_weights/mamba130m_lora_chat.pt"
    # We only save the LoRA keys to avoid bloated adapter files!
    # Wait, the RLF code saves the full state dict. Let's just save the full state dict for compatibility.
    torch.save(model.state_dict(), out_path)
    print(f"\\n[DONE] LoRA_Chat Router saved to {out_path}")

if __name__ == "__main__":
    main()
