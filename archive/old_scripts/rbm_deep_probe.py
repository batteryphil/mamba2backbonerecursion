import torch
from transformers import GPT2Tokenizer
from mamba_rbm import RecursiveMambaLM, Config
import time

def run_deep_reasoning_probe():
    device = "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load the best checkpoint
    ckpt_path = "latest_checkpoint.pt"
    import os
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found. Ensure training has saved at least one best checkpoint.")
        return

    # Configuration for deep reasoning
    config = Config(
        vocab_size=len(tokenizer), 
        d_model=1024, 
        n_layers=8, 
        seq_len=1024,
        n_reasoning=3  # 🚀 Testing our actual target depth
    )
    
    print(f"🧠 Initializing RBM Deep Probe (N={config.n_reasoning})")
    model = RecursiveMambaLM(config).to(device)
    
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    # Complex Logic/Math Prompt
    # Testing Transitive Property Reasoning
    prompt = "Context: Alice is taller than Bob. Charlie is shorter than Bob. Dave is taller than Alice. Question: Who is the tallest person among Alice, Bob, Charlie, and Dave? Logical sequence provided below. \nAnswer:"
    
    print(f"Prompt: {prompt}")
    print("Thinking...")
    
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            prompt_ids, 
            max_new_tokens=128, 
            temperature=0.7, 
            top_k=50
        )
    end_time = time.time()
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print("\n" + "="*50)
    print("🚀 DEEP REASONING PROBE RESULT")
    print("="*50)
    print(f"Response: {response}")
    print("="*50)
    print(f"Inference Time: {end_time - start_time:.2f}s")
    print(f"Effective Reasoning Steps: {config.n_layers * config.n_reasoning} bidirectional sweeps per token.")

if __name__ == "__main__":
    run_deep_reasoning_probe()
