import torch
import json
import os
from transformers import GPT2Tokenizer
from mamba_llm_diffusion import DiM_LLM, MaskedDiffusionEngine, Config

def process_dataset_standalone(data_list, tokenizer):
    all_ids = []
    all_masks = []
    for item in data_list:
        if isinstance(item, dict):
            text = item.get('text', '')
        else:
            text = item
        parts = text.split(" | Assistant: ")
        if len(parts) == 2:
            u_ids = tokenizer.encode(parts[0] + " | Assistant: ", add_special_tokens=False)
            a_ids = tokenizer.encode(parts[1] + " " + tokenizer.eos_token + " ", add_special_tokens=False)
            all_ids.extend(u_ids)
            all_masks.extend([0] * len(u_ids))
            all_ids.extend(a_ids)
            all_masks.extend([1] * len(a_ids))
        else:
            ids = tokenizer.encode(text + " " + tokenizer.eos_token + " ", add_special_tokens=False)
            all_ids.extend(ids)
            all_masks.extend([1] * len(ids))
    return torch.tensor(all_ids, dtype=torch.long), torch.tensor(all_masks, dtype=torch.uint8)

def check_loss():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    config = Config(vocab_size=len(tokenizer), d_model=1024, n_layers=11, seq_len=1024)
    model = DiM_LLM(config).to(device)
    
    ckpt_path = "/home/phil/Desktop/mambadiff/mambadiff llm tts/dim_llm_epoch002.pt"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    m_state = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
    print(f"Ckpt embedding shape: {m_state['token_embed.weight'].shape}")
    model.load_state_dict(m_state)
    model.eval()
    
    # Load gold samples
    gold_path = "/home/phil/Desktop/mambadiff/mambadiff llm tts/balanced_distillation.json"
    with open(gold_path, "r") as f:
        gold_data = json.load(f)
    
    # Load normal samples
    train_path = "/home/phil/Desktop/mambadiff/mambadiff llm tts/train_data.json"
    with open(train_path, "r") as f:
        train_data = json.load(f)
    
    engine = MaskedDiffusionEngine(model, config, device=device)
    
    # 1. Gold Loss
    ids_gold, masks_gold = process_dataset_standalone(gold_data, tokenizer)
    ids_gold = ids_gold[:1024].unsqueeze(0).to(device)
    masks_gold = masks_gold[:1024].unsqueeze(0).to(device)
    
    # 2. Normal Loss
    ids_norm, masks_norm = process_dataset_standalone(train_data[:5], tokenizer)
    ids_norm = ids_norm[:1024].unsqueeze(0).to(device)
    masks_norm = masks_norm[:1024].unsqueeze(0).to(device)
    
    # Load history (dummy context or real)
    # For a simple check, just use zero context or similar
    context_tokens = torch.full((1, 1024), engine.mask_id, dtype=torch.long, device=device)
    with torch.no_grad():
        context_embeddings = model.token_embed(context_tokens)
        context_bank = context_embeddings.mean(dim=1, keepdim=True)

    with torch.no_grad():
        loss_gold = engine.forward_process(ids_gold, context_bank=context_bank, loss_mask=masks_gold)
        loss_norm = engine.forward_process(ids_norm, context_bank=context_bank, loss_mask=masks_norm)
        
    print(f"Model: {ckpt_path}")
    print(f"Loss on Gold samples: {loss_gold.item():.4f}")
    print(f"Loss on Normal samples (avg of first 5): {loss_norm.item():.4f}")

if __name__ == "__main__":
    check_loss()
