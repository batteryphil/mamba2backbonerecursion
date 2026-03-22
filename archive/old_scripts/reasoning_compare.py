import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from mamba_rbm import RecursiveMambaLM, Config
import os
import time

def evaluate_logic(model, tokenizer, device, n_depths=[1, 2, 5]):
    test_cases = [
        "If x = 10 and y = x * 2, then y is",
        "Alice is taller than Bob. Bob is taller than Charlie. Therefore, the tallest person is",
        "Question: If a=1, b=2, then a+b=",
        "Rule: All birds can fly. Tweety is a bird. Question: Can Tweety fly? Answer:",
    ]
    
    results = {}
    for n in n_depths:
        model.config.n_reasoning = n
        n_results = []
        for prompt in test_cases:
            ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                # Greedy search for stability
                out = model.generate(ids, max_new_tokens=10, temperature=0.1, top_k=1)
            response = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
            n_results.append({"prompt": prompt, "response": response})
        results[n] = n_results
    return results

def deep_dive():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    checkpoints = {
        "Morning (Epoch 1)": "rbm_hybrid_epoch_1.pt",
        "Latest (Step 2900+)": "latest_checkpoint.pt"
    }
    
    config = Config(
        vocab_size=len(tokenizer), 
        d_model=1024, 
        n_layers=8, 
        seq_len=1024,
        n_reasoning=1
    )
    
    report = []
    
    for name, path in checkpoints.items():
        if not os.path.exists(path):
            print(f"Skipping {name} (not found)")
            continue
            
        print(f"🔬 Auditing {name}...")
        model = RecursiveMambaLM(config).to(device)
        state = torch.load(path, map_location=device)
        if "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)
        model.eval()
        
        res = evaluate_logic(model, tokenizer, device)
        report.append({"name": name, "results": res})
        
    # Print Comparison
    print("\n" + "="*80)
    print("📈 REASONING IMPROVEMENT COMPARISON")
    print("="*80)
    
    for n in [1, 5]:
        print(f"\n--- Depth N={n} ---")
        for i in range(len(report[0]["results"][n])):
            prompt = report[0]["results"][n][i]["prompt"]
            print(f"Prompt: {prompt}")
            for r in report:
                print(f"  [{r['name']}]: {r['results'][n][i]['response']}")
                
if __name__ == "__main__":
    deep_dive()
