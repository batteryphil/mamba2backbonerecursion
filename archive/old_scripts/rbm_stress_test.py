import torch
from transformers import GPT2Tokenizer
from mamba_rbm import RecursiveMambaLM, Config
import time
import os

def run_stress_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    ckpt_path = "rbm_v1_best.pt"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return

    # Configuration for deep reasoning
    config = Config(
        vocab_size=len(tokenizer), 
        d_model=1024, 
        n_layers=8, 
        seq_len=1024,
        n_reasoning=5  # Optimal deliberation depth for Epoch 4
    )
    
    print(f"🧠 RBM LOGIC STRESS-TEST (Manifold Depth: N={config.n_reasoning})")
    model = RecursiveMambaLM(config).to(device)
    
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    test_cases = [
        {
            "id": "L1_INVERSION",
            "prompt": "--- RAW TEXT ---\nFact: Kevin is shorter than Riley. Question: Who is the tallest between them? Reasoning:",
            "meta": "Target: Riley (Logic Inversion check)"
        },
        {
            "id": "L2_CHAIN",
            "prompt": "--- RAW TEXT ---\nPremises: Bob is taller than Alice. Dave is taller than Bob. Charlie is shorter than Alice. Question: Who is the tallest among Alice, Bob, Charlie, and Dave? Answer:",
            "meta": "Target: Dave (4-Entity Chain)"
        },
        {
            "id": "L3_NEGATION",
            "prompt": "--- RAW TEXT ---\nInformation: Steve is NOT shorter than Tanya. Assuming they have different heights, who is taller? Answer:",
            "meta": "Target: Steve (Logical Negation)"
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
                max_new_tokens=64, 
                temperature=0.1, # Keep it deterministic for logic
                top_k=1
            )
        
        response = tokenizer.decode(output_ids[0][prompt_ids.shape[1]:], skip_special_tokens=True)
        print(f"RESULT: {response.strip()}")
        print("-" * 30)

if __name__ == "__main__":
    run_stress_test()
