import torch
from transformers import GPT2Tokenizer
from mamba_rbm import RecursiveMambaLM, Config
import time
import os

def run_quick_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Check for the latest checkpoint
    checkpoint_options = [
        "manual_v6_1_step14200.pt",
        "latest_checkpoint.pt"
    ]
    
    ckpt_path = None
    for opt in checkpoint_options:
        if os.path.exists(opt):
            ckpt_path = opt
            break
            
    if not ckpt_path:
        print("Error: No RBM checkpoint found.")
        return

    print(f"📦 Loading checkpoint from: {ckpt_path}")

    # Configuration 
    config = Config(
        vocab_size=50257, 
        d_model=1024, 
        n_layers=8, 
        seq_len=1024,
        n_reasoning=1  # testing with one recursion
    )
    
    print(f"🧠 RBM QUICK TEST (Reasoning Depth: N={config.n_reasoning})")
    model = RecursiveMambaLM(config).to(device)
    
    state = torch.load(ckpt_path, map_location=device)
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
        
    model.eval()

    test_cases = [
        {
            "id": "IDENTITY",
            "prompt": "User: Who are you? | Assistant:",
            "meta": "Identity check (Using training separator '|')"
        },
        {
            "id": "LOGIC_INVERSION",
            "prompt": "### LOGIC MANIFOLD v5.0\n--- RAW TEXT ---\nPremise: Alice is shorter than Bob. Target Inquiry: Who is the tallest? Analysis:",
            "meta": "Logic format check"
        },
        {
            "id": "CAPITAL",
            "prompt": "User: What is the capital of France? | Assistant:",
            "meta": "General knowledge"
        }
    ]

    for case in test_cases:
        print(f"\n--- {case['id']} ---")
        print(f"Metadata: {case['meta']}")
        print(f"Prompt: {case['prompt']}")
        
        prompt_ids = tokenizer.encode(case['prompt'], return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                prompt_ids, 
                max_new_tokens=128, 
                temperature=1.0, 
                top_k=40
            )
        
        response = tokenizer.decode(output_ids[0][prompt_ids.shape[1]:], skip_special_tokens=True)
        print(f"RESULT: {response.strip()}")
        print("-" * 30)

if __name__ == "__main__":
    run_quick_test()
