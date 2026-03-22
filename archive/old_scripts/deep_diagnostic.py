import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from mamba_rbm import RecursiveMambaLM, Config
import matplotlib.pyplot as plt
import numpy as np
import os
import json

def deep_diagnostic():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    ckpt_path = "latest_checkpoint.pt"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return

    print(f"🔍 Starting Deep Diagnostic on {ckpt_path}...")
    
    # Configuration (matching training)
    config = Config(
        vocab_size=50257, 
        d_model=1024, 
        n_layers=8, 
        seq_len=1024,
        n_reasoning=3
    )
    
    model = RecursiveMambaLM(config).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    results = {}

    # 1. ⚖️ Weight Audit
    print("\n--- 1. Weight Audit ---")
    param_stats = []
    total_params = 0
    nan_count = 0
    inf_count = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        p_data = param.detach().cpu().float().numpy()
        nans = np.isnan(p_data).sum()
        infs = np.isinf(p_data).sum()
        nan_count += nans
        inf_count += infs
        
        param_stats.append({
            "name": name,
            "shape": list(param.shape),
            "mean": float(p_data.mean()),
            "std": float(p_data.std()),
            "max": float(p_data.max()),
            "min": float(p_data.min())
        })
    
    nan_count = int(nan_count)
    inf_count = int(inf_count)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"NaNs detected: {nan_count}")
    print(f"Infs detected: {inf_count}")
    
    results["weight_audit"] = {
        "total_params": int(total_params),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "param_details": param_stats[:10] # Subset for brevity
    }

    # 2. 🌀 Recursive Sensitivity (The "Deliberation Curve")
    print("\n--- 2. Recursive Sensitivity ---")
    prompt = "### LOGIC MANIFOLD v5.0\n--- RAW TEXT ---\nPremise: Small is less than Big. Target: Which is greater? Answer:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    deliberation_results = []
    for n in range(1, 11): # Test from 1 to 10 recursions
        model.config.n_reasoning = n
        with torch.no_grad():
            logits = model(input_ids)
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            top_prob, top_idx = torch.max(probs, dim=-1)
            top_token = tokenizer.decode([top_idx.item()])
            
            deliberation_results.append({
                "n": n,
                "entropy": entropy,
                "top_prob": top_prob.item(),
                "top_token": top_token
            })
            print(f"N={n} | Entropy: {entropy:.4f} | Top Token: '{top_token}' ({top_prob.item():.4f})")
    
    results["recursive_sensitivity"] = deliberation_results

    # 3. 📊 Logit Distribution Analysis
    print("\n--- 3. Logit Distribution (Head Bias) ---")
    model.config.n_reasoning = 3
    with torch.no_grad():
        logits = model(input_ids)
        last_logits = logits[0, -1, :].cpu().float().numpy()
        
        # Calculate skewness/kurtosis or just simple stats
        res_logits = {
            "mean": float(last_logits.mean()),
            "std": float(last_logits.std()),
            "median": float(np.median(last_logits)),
            "high_tokens": []
        }
        
        # Get top 20 tokens
        sorted_indices = np.argsort(last_logits)[::-1]
        for i in range(20):
            idx = sorted_indices[i]
            res_logits["high_tokens"].append({
                "token": tokenizer.decode([int(idx)]),
                "logit": float(last_logits[idx])
            })
            
        print("Top 5 tokens in distribution:")
        for t in res_logits["high_tokens"][:5]:
            print(f"  '{t['token']}': {t['logit']}")

    results["logit_stats"] = res_logits

    # 4. 🧠 Memory/Coherence Probe (Sequence Consistency)
    print("\n--- 4. Sequence Consistency Probe ---")
    long_prompt = "User: Tell me a story about a robot named Mamba. | Assistant: Once upon a time, in a digital garden, there lived a robot named Mamba."
    input_ids_long = tokenizer.encode(long_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        logits_long = model(input_ids_long)
        # Check if model "recognizes" its own prompt tokens
        # Cross-Entropy at each step
        shift_logits = logits_long[..., :-1, :].contiguous()
        shift_labels = input_ids_long[..., 1:].contiguous()
        loss_per_token = F.cross_entropy(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1), reduction='none')
        loss_per_token = loss_per_token.view(shift_labels.shape).cpu().numpy()[0]
        
        avg_loss = float(loss_per_token.mean())
        print(f"Internal Coherence (Loss on prompt): {avg_loss:.4f}")
        
    results["coherence"] = {
        "avg_prompt_loss": avg_loss,
        "token_losses": loss_per_token.tolist()
    }

    # Save Diagnostic Report
    with open("deep_diagnostic_report.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Diagnostic Complete. Report saved to deep_diagnostic_report.json")

if __name__ == "__main__":
    deep_diagnostic()
