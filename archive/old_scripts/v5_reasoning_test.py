import torch
from transformers import GPT2Tokenizer
from mamba_rbm import RecursiveMambaLM, Config
import os

def run_test():
    device = "cpu" # 🛡️ Use CPU to avoid VRAM conflict with training
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    config = Config(
        vocab_size=len(tokenizer), 
        d_model=1024, 
        n_layers=8, 
        seq_len=1024,
        n_reasoning=3
    )
    
    model = RecursiveMambaLM(config).to(device)
    
    ckpt = "rbm_hybrid_epoch_3.pt"
    if os.path.exists(ckpt):
        print(f"🛠️ Loading Epoch 3 State: {ckpt}")
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state"])
    else:
        print("❌ Checkpoint not found.")
        return

    model.eval()
    
    prompts = [
        # 0. Basic Completion
        "The sky is",
        
        # 1. Overfitting Check (Direct from train_data.json)
        "User: Write a Python script that accepts a string of text as input and calculates the frequency of each character in the string. The output should be a dictionary where keys are characters and values",
        
        # 2. Logic Probe
        "### LOGIC MANIFOLD v5.0\n--- RAW TEXT ---\nPremise: Alice is shorter than Bob. Who is the tallest individual? | Assistant:",
    ]

    print("\n" + "="*50)
    print("🧠 RBM v5.0 REASONING PROBE (Greedy)")
    print("="*50)

    for p in prompts:
        print(f"\nPROMPT: {repr(p)}")
        ids = tokenizer.encode(p, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(ids, max_new_tokens=20, temperature=0.1) # low temp approx greedy
            response_full = tokenizer.decode(output_ids[0], skip_special_tokens=False)
            print(f"FULL RESPONSE (incl prompt):\n{repr(response_full)}")
            print("-" * 30)

if __name__ == "__main__":
    run_test()
