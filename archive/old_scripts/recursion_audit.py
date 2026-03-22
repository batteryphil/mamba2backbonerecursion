import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from mamba_rbm import RecursiveMambaLM, Config
import os

def top_p_filtering(logits, top_p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = -float('Inf')
    return logits

def run_multi_pass_audit():
    device = "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prompt = "Alpha is taller than Beta. Question: Who is shorter? Answer:"
    
    # 🕵️ Previous Results (Step 12,300 Manifold)
    history = {
        1: ":::::::::::::::::::::::::::::::::::::::::",
        2: ":::::::::::::::::::::::::::::::::::::::::",
        3: "of of of of the the",
        4: "a a a, they, they they to to to, they they they they they to to to"
    }

    print("\n" + "="*60)
    print("🧠 RBM RECURSION AUDIT (Step 12,300 Manifold)")
    print("="*60)
    for p, res in history.items():
        print(f"N={p}: {repr(res)}")
    print("-" * 60)

    # 🚀 Run the 10-Pass Test
    config = Config(
        vocab_size=50257,
        d_model=1024,
        n_layers=8,
        seq_len=1024,
        n_reasoning=10 # Extreme deliberation
    )
    
    model = RecursiveMambaLM(config).to(device)
    ckpt = "latest_checkpoint.pt"
    if os.path.exists(ckpt):
        print(f"🛠️ Loading Manifold State: {ckpt}")
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state"])
    else:
        print("❌ latest_checkpoint.pt not found.")
        return

    model.eval()
    print(f"PROMPT: {prompt}")

    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    current_ids = ids
    
    max_new_tokens = 40
    temperature = 0.5 
    top_p = 0.9

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(current_ids[:, -config.seq_len:])
            next_token_logits = logits[0, -1, :] / (temperature + 1e-8)
            next_token_logits = top_p_filtering(next_token_logits, top_p=top_p)
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            current_ids = torch.cat([current_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
                
        response_full = tokenizer.decode(current_ids[0], skip_special_tokens=False)
        print(f"\nN=10 RESULT:\n{repr(response_full)}")
        print("="*60)

if __name__ == "__main__":
    run_multi_pass_audit()
