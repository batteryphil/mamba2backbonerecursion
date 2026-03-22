import torch
from transformers import GPT2Tokenizer
import time
import os
import sys

# Protocol v6.3: Suppress expandable segments warning for clean output
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from mamba_rbm import RecursiveMambaLM, Config

def run_diagnostic():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔬 INITIALIZING N=3 DIAGNOSTIC PROBE ON {device.upper()}")
    
    ckpt_path = "latest_checkpoint.pt"
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: {ckpt_path} not found.")
        return

    # Load Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Base Config (N will be mutated during the test)
    config = Config(
        vocab_size=len(tokenizer), 
        d_model=1024, 
        n_layers=8, 
        seq_len=1024,
        n_reasoning=1 
    )
    
    model = RecursiveMambaLM(config).to(device)
    model.eval()

    print(f"📂 Loading Manifold: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    step = state.get("step", "Unknown")
    print(f"✅ Weights Loaded. Current Training Step: {step}")
    
    print("\n" + "="*60)
    print("🧠 DIAGNOSTIC PROBE: MULTI-STEP VARIABLE ASSIGNMENT")
    print("Goal: Test resolution of 'Abstract Fog' via reasoning depth.")
    print("="*60)

    # The Prompt (designed to require holding variables across steps)
    prompt_text = "Logic problem: If x = 5 and y = x + 2, then what is the value of y? The value of y is"
    print(f"\n[PROMPT]: \"{prompt_text}\"")
    
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

    # Test N=1 (Intuition Only / Baseline)
    # Test N=2 (Current Anchor State)
    # Test N=3 (The Target Resolution Depth)
    
    test_depths = [1, 2, 3]
    
    for n in test_depths:
        model.config.n_reasoning = n
        print(f"\n⚙️ Running with N={n} (Depth Iterations)...")
        
        start_time = time.time()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output_ids = model.generate(
                prompt_ids, 
                max_new_tokens=20, # Keep it short to see the immediate answer
                temperature=0.1,   # Low temp for deterministic logic testing
                top_k=5
            )
        elapsed = time.time() - start_time
        
        response = tokenizer.decode(output_ids[0][prompt_ids.shape[1]:], skip_special_tokens=True)
        
        # Clean up response formats (sometimes it generates newlines or extra spaces)
        clean_response = response.strip().split('\n')[0]
        
        print(f"  [OUTPUT]: {clean_response}")
        print(f"  [TIME]:   {elapsed:.2f}s")
        
        # Simple heuristic check
        if "7" in clean_response:
            print("  [STATUS]: ✅ SUCCESS (Resolved)")
        else:
            print("  [STATUS]: ❌ FAIL (Abstract Fog)")

    print("\n" + "="*60)
    print("🔬 DIAGNOSTIC COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_diagnostic()
