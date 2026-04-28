import torch
import torch.nn as nn
from mamba_ssm import MambaLMHeadModel
import json
import random
import os
from transformers import AutoTokenizer
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR_HEAD = 2.0e-5    # Fast learning rate for English vocabulary output
LR_CORE = 1.0e-6    # Extremely frozen learning rate to protect the ALU geometry
BATCH_SIZE = 8
MAX_STEPS = 5000

# Base Checkpoint: The Final Boss GRPO Mathematical Singularity
BASE_CHECKPOINT = "checkpoints/mamba3_p12_mastered.pt"

def load_unified_curriculum():
    print("[INIT] Forging Phase 13 Universal Replay Buffer...")
    
    # 1. Math Routing (To preserve the Phase 12-C RAM Geometry)
    math_data = []
    if os.path.exists("phase12b_gsm8k.jsonl"):
        with open("phase12b_gsm8k.jsonl", "r") as f:
            for line in f:
                math_data.append(json.loads(line))
        print(f" -> Preserved GSM8K Routing Logic: {len(math_data)}")

    # 2. Conversational Re-Anchoring (To recover English generation)
    chat_data = []
    print("[INIT] Downloading HuggingFaceH4/ultrachat_200k...")
    try:
        chat_dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:5000]")
        for item in chat_dataset:
            # Simple conversational extraction
            messages = item['messages']
            if len(messages) >= 2 and messages[0]['role'] == 'user' and messages[1]['role'] == 'assistant':
                chat_data.append({
                    "prompt": messages[0]['content'],
                    "answer": messages[1]['content']
                })
        print(f" -> Recovered English Conversational Priors: {len(chat_data)}")
    except Exception as e:
        print(f"[FATAL] Failed to download UltraChat dataset: {e}")
        
    return math_data, chat_data

def get_mixed_batch(math_data, chat_data, batch_size):
    """80% Math Latent Routing, 20% Conversational Generation"""
    batch = []
    for _ in range(batch_size):
        if random.random() < 0.8 and math_data:  # System 2 Bias (80%)
            item = random.choice(math_data)
            n_loops = random.randint(5, 12)
            dark_loops = "=" * n_loops
            ans = f"<answer>{item['answer']}</answer>"
            # System 2 Routing Prefix
            batch.append((f"[LOGIC] {item['prompt']}", dark_loops, ans))
        elif chat_data:  # System 1 Bias (20%)
            item = random.choice(chat_data)
            # System 1 Routing Prefix
            batch.append((f"[CHAT] {item['prompt']}", "", item['answer']))
        else:
            item = random.choice(math_data)
            batch.append((f"[LOGIC] {item['prompt']}", "=======", f"<answer>{item['answer']}</answer>"))
    return batch

def main():
    print("==========================================================")
    print("  MAMBA-3 PHASE 13: CONVERSATIONAL RE-ANCHORING SFT")
    print("==========================================================")
    
    os.makedirs("checkpoints", exist_ok=True)
    
    if not os.path.exists(BASE_CHECKPOINT):
        print(f"[FATAL] Mamba-3 Phase 12-C Mastery checkpoint missing: {BASE_CHECKPOINT}. Waiting for Phase 12 to finish cooking.")
        return
        
    print(f"[INIT] Loading GRPO Mastered Weights: {BASE_CHECKPOINT}")
    model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m", device=DEVICE, dtype=torch.bfloat16)
    model.load_state_dict(torch.load(BASE_CHECKPOINT, map_location=DEVICE))
    model.train()
    
    # Gradient Surgery: Orthogonal Learning Rates
    # Handle tied embeddings (lm_head.weight == backbone.embedding.weight)
    head_params_set = set(model.lm_head.parameters())
    core_params_list = [p for p in model.backbone.parameters() if p not in head_params_set]
    
    optimizer = torch.optim.AdamW([
        {'params': core_params_list, 'lr': LR_CORE},
        {'params': list(head_params_set), 'lr': LR_HEAD}
    ], weight_decay=0.01)
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    criterion = nn.CrossEntropyLoss(ignore_index=-100) 
    
    math_data, chat_data = load_unified_curriculum()
    
    global_step = 0
    
    while global_step < MAX_STEPS:
        raw_items = get_mixed_batch(math_data, chat_data, BATCH_SIZE)
        
        batch_input_ids = []
        batch_labels = []
        
        for prompt_text, dark_loops, answer_text in raw_items:
            # The prompt_text already contains the [CHAT] or [LOGIC] dynamic router prefix
            formatted_prompt = f"{prompt_text}\nSolution: "
            
            p_ids = tokenizer.encode(formatted_prompt, return_tensors="pt")[0]
            d_ids = tokenizer.encode(dark_loops, return_tensors="pt")[0] if dark_loops else torch.tensor([], dtype=torch.long)
            a_ids = tokenizer.encode(answer_text, return_tensors="pt")[0]
            eos_id = torch.tensor([tokenizer.eos_token_id])
            
            full_ids = torch.cat([p_ids, d_ids, a_ids, eos_id])
            
            # Target Masking: Mask out the Prompt and the Dark Loops.
            # We ONLY calculate Cross-Entropy loss on the generated Answer!
            # If it's a conversation, the model learns to generate English answers.
            # If it's math, it learns to hold the format.
            labels = full_ids.clone()
            labels[:len(p_ids) + len(d_ids)] = -100
            
            max_len = 512 # Expand sequence length for longer conversational English
            if len(full_ids) > max_len:
                full_ids = full_ids[:max_len]
                labels = labels[:max_len]
                
            batch_input_ids.append(full_ids)
            batch_labels.append(labels)
            
        # Pad batch dynamically
        max_batch_seq_len = max(len(seq) for seq in batch_input_ids)
        padded_inputs = torch.full((BATCH_SIZE, max_batch_seq_len), tokenizer.pad_token_id, dtype=torch.long)
        padded_labels = torch.full((BATCH_SIZE, max_batch_seq_len), -100, dtype=torch.long)
        
        for i in range(BATCH_SIZE):
            seq_len = len(batch_input_ids[i])
            padded_inputs[i, :seq_len] = batch_input_ids[i]
            padded_labels[i, :seq_len] = batch_labels[i]
            
        padded_inputs = padded_inputs.to(DEVICE)
        padded_labels = padded_labels.to(DEVICE)
        
        inputs = padded_inputs[:, :-1]
        targets = padded_labels[:, 1:]
        
        optimizer.zero_grad()
        
        hidden_states = model.backbone.embedding(inputs)
        residual = None
        for layer in model.backbone.layers:
            hidden_states, residual = layer(hidden_states, residual=residual)
            
        final_hidden = model.backbone.norm_f(hidden_states + residual) if residual is not None else model.backbone.norm_f(hidden_states)
        logits = model.lm_head(final_hidden.to(torch.bfloat16))
        
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        global_step += 1
        
        if global_step % 50 == 0:
            print(f"[PHASE 13 S{global_step:04d}] Universal Target Masked Loss: {loss.item():.4f}")
            
        if global_step % 1000 == 0:
            torch.save(model.state_dict(), f"checkpoints/mamba3_p13_conversational_g{global_step}.pt")

    print("[SYSTEM] TURING OVERRIDE: Phase 13 Universal Instruct Model Created.")
    torch.save(model.state_dict(), "checkpoints/mamba3_p13_universal_mastered.pt")

if __name__ == "__main__":
    main()
