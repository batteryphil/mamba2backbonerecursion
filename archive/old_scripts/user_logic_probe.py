import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from mamba_rbm import RecursiveMambaLM, Config
import os

def top_p_filtering(logits, top_p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = -float('Inf')
    return logits

def run_user_probe():
    device = "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # 📈 FOUR PASS MODE: Testing "Deep Thinking" beyond training depth
    config = Config(
        vocab_size=50257, 
        d_model=1024, 
        n_layers=8, 
        seq_len=1024,
        n_reasoning=4 # 4 sweeps per layer (Deep deliberation)
    )
    
    model = RecursiveMambaLM(config).to(device)
    
    ckpt = "latest_checkpoint.pt"
    if os.path.exists(ckpt):
        print(f"🛠️ Loading Manifold State (Step 12000+): {ckpt}")
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state"])
    else:
        print("❌ latest_checkpoint.pt not found.")
        return

    model.eval()
    
    prompt = "Alpha is taller than Beta. Question: Who is shorter? Answer:"

    print("\n" + "="*50)
    print("📉 1-PASS REASONING TEST (No Deliberation)")
    print("="*50)
    print(f"PROMPT: {prompt}")

    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    current_ids = ids
    
    max_new_tokens = 40
    temperature = 0.5 # Lower T to see "Pure" 1-pass signal
    top_p = 0.9

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(current_ids[:, -config.seq_len:])
            next_token_logits = logits[0, -1, :] / (temperature + 1e-8)
            
            # Apply Top-P (Nucleus) Filtering
            next_token_logits = top_p_filtering(next_token_logits, top_p=top_p)
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
        response_full = tokenizer.decode(current_ids[0], skip_special_tokens=False)
        print(f"\nFULL RESPONSE:\n{repr(response_full)}")
        print("="*50)

if __name__ == "__main__":
    run_user_probe()
