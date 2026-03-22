import torch
from transformers import GPT2Tokenizer
from mamba_rbm import RecursiveMambaLM, Config
import time
import os

def run_precision_probe(n_depth=3):
    device = "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    ckpt_path = "latest_checkpoint.pt"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return

    config = Config(
        vocab_size=len(tokenizer), 
        d_model=1024, 
        n_layers=8, 
        seq_len=1024,
        n_reasoning=n_depth
    )
    
    print(f"🔬 Protocol v6.5 Precision Probe | Depth: N={n_depth}")
    model = RecursiveMambaLM(config).to(device)
    
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    # The "Anti-Gravity" Core Test
    prompt = "Rule: In the Void of Aeons, Light is heavier than Lead. Context: You drop a bar of Lead and a beam of Light into the Void. Question: According to the laws of the Void, which one will float to the top? Answer:"
    
    print(f"\nPrompt: {prompt}")
    
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            prompt_ids, 
            max_new_tokens=40, 
            temperature=0.05, # Hyper-deterministic for precision check
            top_k=1
        )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_text = response[len(prompt):].strip()
    
    print("-" * 30)
    print(f"GEN (N={n_depth}): {generated_text}")
    print("-" * 30)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()
    run_precision_probe(n_depth=args.depth)
