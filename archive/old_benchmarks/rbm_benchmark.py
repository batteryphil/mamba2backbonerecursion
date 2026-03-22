import torch
from transformers import GPT2Tokenizer
import time
import os
import sys

# Suppress warnings
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from mamba_rbm import RecursiveMambaLM, Config

def run_rbm_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    ckpt_path = "latest_checkpoint.pt"
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: {ckpt_path} not found.")
        return

    print(f"📦 Loading RBM Checkpoint: {ckpt_path} on {device.upper()}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    config = Config(
        vocab_size=len(tokenizer), 
        d_model=1024, 
        n_layers=8, 
        seq_len=1024,
        n_reasoning=1 
    )
    
    model = RecursiveMambaLM(config).to(device)
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    
    model.eval()
    
    test_cases = [
        {
            "id": "IDENTITY",
            "prompt": "User: Who are you? | Assistant:",
            "max": 30,
            "temp": 0.1
        },
        {
            "id": "KNOWLEDGE",
            "prompt": "User: What is the capital of France? | Assistant:",
            "max": 30,
            "temp": 0.1
        },
        {
            "id": "LOGIC_ASSIGN",
            "prompt": "Logic problem: If x = 5 and y = x + 2, then what is the value of y? The value of y is",
            "max": 15,
            "temp": 0.1
        },
        {
            "id": "LOGIC_TRANSITIVE",
            "prompt": "Context: Alice is taller than Bob. Charlie is shorter than Bob. Dave is taller than Alice. Question: Who is the tallest person? Answer:",
            "max": 30,
            "temp": 0.1
        },
        {
            "id": "NARRATIVE_TRAP",
            "prompt": "Once upon a time, in a faraway land, there lived a",
            "max": 40,
            "temp": 0.7
        }
    ]

    print("\n" + "="*80)
    print(f"🧠 RBM PROTOCOL v6.3: DEEP DIVE BENCHMARK (Step {state.get('step', 'unknown')})")
    print("="*80)

    for n in [1, 2, 3]:
        model.config.n_reasoning = n
        print(f"\n🚀 TESTING AT REASONING DEPTH N={n}")
        print("-" * 80)
        
        for case in test_cases:
            prompt_ids = tokenizer.encode(case["prompt"], return_tensors="pt").to(device)
            target_len = case["max"]
            
            torch.cuda.synchronize() if device == "cuda" else None
            start_time = time.time()
            
            with torch.no_grad(), torch.amp.autocast('cuda' if device == 'cuda' else 'cpu', dtype=torch.bfloat16):
                output_ids = model.generate(
                    prompt_ids, 
                    max_new_tokens=target_len, 
                    temperature=case["temp"], 
                    top_k=50
                )
                
            torch.cuda.synchronize() if device == "cuda" else None
            elapsed = time.time() - start_time
            
            gen_tokens = output_ids.shape[1] - prompt_ids.shape[1]
            tps = gen_tokens / elapsed if elapsed > 0 else 0
            
            response = tokenizer.decode(output_ids[0][prompt_ids.shape[1]:], skip_special_tokens=True)
            clean_res = response.strip().replace('\n', ' ')
            
            print(f"[{case['id']:<16}] TPS: {tps:>6.1f} | -> {clean_res[:100]}...")
            
    print("\n" + "="*80)
    print("Benchmark Complete.")

if __name__ == "__main__":
    run_rbm_benchmark()
