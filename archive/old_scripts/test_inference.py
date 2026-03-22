import torch
import json
import os
from transformers import GPT2Tokenizer
from mamba_llm_diffusion import DiM_LLM, MaskedDiffusionEngine, Config

def test_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing Jarvis Inference (GPT-2 BPE) on {device}...")

    # 1. Initialize GPT-2 Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    
    config = Config(
        vocab_size=len(tokenizer),
        d_model=1024,
        n_layers=11,
        seq_len=256
    )

    # 2. Setup Model
    model = DiM_LLM(config)
    checkpoint_path = "dim_llm_ema_checkpoint.pt" if os.path.exists("dim_llm_ema_checkpoint.pt") else "dim_llm_checkpoint.pt"
    
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("Warning: No checkpoint found!")

        
    engine = MaskedDiffusionEngine(model, config, device=device)
    model.eval()

    # 3. Test Sampling with Prompt
    if os.path.exists("system_prompt.txt"):
        with open("system_prompt.txt", "r") as f:
            system_prompt = f.read()
    else:
        system_prompt = "You are Jarvis, a helpful assistant."

    prompt = system_prompt + "\nUser: Hello Jarvis, are you functional?\nAssistant: "
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    print("Generating response via Masked Diffusion (32 steps)...")
    with torch.no_grad():
        output_tokens = engine.sample(n_samples=1, steps=32, prompt_ids=input_ids)
    
    # Decode using GPT-2
    response = tokenizer.decode(output_tokens[0].tolist(), skip_special_tokens=False)
    print(f"\nPrompt: {prompt}")
    print(f"Jarvis (Full Seq): {response}")

if __name__ == "__main__":
    test_inference()
