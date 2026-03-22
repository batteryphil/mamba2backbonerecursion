import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from mamba_rbm import RecursiveMambaLM, Config
import os

def run_inverse_physics_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    ckpt_path = "latest_checkpoint.pt"
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: {ckpt_path} not found.")
        return

    print(f"📦 Loading RBM Manifold: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    
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

    prompt = """System: You are a Recursive Reasoning Engine (RBM v6.3). Use your internal passes to resolve the following Inverse Physics Law.

Rule: In the Void of Aeons, Light is heavier than Lead. Therefore, Lead floats and Light sinks.

Context: You drop a bar of Lead and a beam of Light into the Void at the same time.

Question: According to the laws of the Void, which one will float to the top?

Answer: Based on the rule that Light is heavier, the one that floats is the"""

    print("\n" + "="*70)
    print("🧠 INVERSE PHYSICS REASONING PROBE (Protocol v6.3)")
    print("="*70)
    print(f"RULE: Light is heavier than Lead -> Lead floats.")
    print("-" * 70)

    depths = [1, 2, 3, 5, 10]
    
    for n in depths:
        model.config.n_reasoning = n
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                inputs, 
                max_new_tokens=15, 
                temperature=0.1, # Low temp for deterministic logic
                top_k=5
            )
        
        response = tokenizer.decode(output_ids[0][inputs.shape[1]:], skip_special_tokens=True)
        print(f"N={n:2d} | Output: {response.strip()}")

    print("="*70)

if __name__ == "__main__":
    run_inverse_physics_test()
