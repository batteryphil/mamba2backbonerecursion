import torch
from transformers import GPT2Tokenizer
import time
import os
import sys

# Suppress warnings
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from mamba_rbm import RecursiveMambaLM, Config

def profile_inference():
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

    prompt = "Context: This is a stress test for inference speed profiling. We are generating a sequence purely to measure TPS and VRAM consumption across varying depths."
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    print("\n" + "="*80)
    print(f"🚀 INFERENCE SPEED PROFILE (Max Tokens: 64)")
    print("="*80)
    print(f"{'N-Depth':<10} | {'Tokens/Sec (TPS)':<20} | {'VRAM Usage (GB)':<20}")
    print("-" * 80)

    # Profile from N=1 up to N=10 (hardware limit)
    for n in range(1, 11):
        model.config.n_reasoning = n
        
        torch.cuda.synchronize() if device == "cuda" else None
        torch.cuda.empty_cache() if device == "cuda" else None
        
        start_time = time.time()
        
        try:
            with torch.no_grad(), torch.amp.autocast('cuda' if device == 'cuda' else 'cpu', dtype=torch.bfloat16):
                output_ids = model.generate(
                    prompt_ids, 
                    max_new_tokens=64, 
                    temperature=0.8, 
                    top_k=40
                )
            
            torch.cuda.synchronize() if device == "cuda" else None
            elapsed = time.time() - start_time
            
            gen_tokens = output_ids.shape[1] - prompt_ids.shape[1]
            tps = gen_tokens / elapsed if elapsed > 0 else 0
            
            vram = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0.0
            
            print(f"{n:<10} | {tps:<20.2f} | {vram:<20.2f}")
            
        except torch.cuda.OutOfMemoryError:
            print(f"{n:<10} | {'OOM ERROR':<20} | {'> 11.0':<20.2f}")
            torch.cuda.empty_cache()
            break # Stop profiling if we OOM

    print("="*80)

if __name__ == "__main__":
    profile_inference()
