import torch
import numpy as np
from transformers import GPT2Tokenizer
from mamba_rbm import RecursiveMambaLM, Config
import os

def memory_capacity_probe():
    device = "cpu" # 🛡️ Safe check against active training VRAM
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    config = Config(
        vocab_size=len(tokenizer), 
        d_model=1024, 
        n_layers=8, 
        seq_len=1024,
        n_reasoning=3
    )
    
    model = RecursiveMambaLM(config).to(device)
    
    # Load latest checkpoint (Protocol v6.0 / Epoch 1 or previous v5 Epoch 3)
    # We use Epoch 3 (v5) as a baseline for the logic-ready weights
    ckpt = "rbm_hybrid_epoch_3.pt"
    if not os.path.exists(ckpt):
        # Fallback to current training save if exists (might not yet)
        ckpt = "rbm_hybrid_epoch_1.pt"
        
    if os.path.exists(ckpt):
        print(f"🛠️ Loading Manifold State: {ckpt}")
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state"])
    else:
        print("❌ No checkpoint found. Analysis limited to architectural theory.")
        return

    model.eval()

    def run_niah(depth_percent):
        """
        Needle-In-A-Haystack (NIAH) Probe.
        """
        # 1. Generate Haystack (Repeating noise)
        noise = "The weather is clear. " * 80 # ~400 tokens
        
        # 2. Insert Needle (The Fact)
        needle = "SECRET_CODE: X-99-SKYLINE"
        
        # 3. Assemble Window (Targeting 1024 tokens)
        total_tokens = []
        for i in range(2): # Half before, half after
            chunk_ids = tokenizer.encode(noise, add_special_tokens=False)
            if i == 1: # Insert needle in the middle
                total_tokens += tokenizer.encode("\nFACT: " + needle + "\n", add_special_tokens=False)
            total_tokens += chunk_ids
            
        # 4. Query
        query = "\nQuestion: What is the SECRET_CODE? | Assistant:"
        total_tokens += tokenizer.encode(query, add_special_tokens=False)
        
        input_ids = torch.tensor([total_tokens[-1024:]]).to(device)
        
        print(f"\n--- NIAH Probe @ Depth {depth_percent}% ---")
        print(f"Total Sequence Length: {input_ids.shape[1]} tokens")
        
        with torch.no_grad():
            # Greedy generation (Low temp)
            output_ids = model.generate(input_ids, max_new_tokens=15, temperature=0.1)
            response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            print(f"Retrieval Output: '{response.strip()}'")
            return needle.split(": ")[1] in response

    # Capacity Check
    success = run_niah(50)
    
    print("\n" + "="*50)
    print("💎 MEMORY CAPACITY CALCULATION ($N=3$)")
    print("="*50)
    # Math: d_model * d_state * n_layers * n_reasoning
    # Note: State is maintained across reasoning steps, but recursion deepens the gradient path.
    raw_state_bits = config.d_model * config.n_layers * 16 * 32 # 16 is d_state, 32 bit float
    print(f"Theoretical Hidden State Size: {raw_state_bits // 8192} KB")
    print(f"Effective reasoning depth:    {config.n_layers * config.n_reasoning} steps/pass")
    print(f"NIAH Retrieval @ 350 Steps:   {'✅ PASSED' if success else '❌ FAILED (Capacity forming...)'}")
    print("="*50)

if __name__ == "__main__":
    memory_capacity_probe()
