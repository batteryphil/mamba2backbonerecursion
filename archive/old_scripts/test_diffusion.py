import torch
import json
import os
from transformers import GPT2Tokenizer
from mamba_llm_diffusion import DiM_LLM, MaskedDiffusionEngine, Config

def test_diffusion_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing Jarvis Diffusion Inference (GPT-2 BPE) on {device}...")

    # 1. Initialize GPT-2 Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    
    config = Config(
        vocab_size=len(tokenizer),
        d_model=1024,
        n_layers=11,
        seq_len=1024 # Matching training
    )

    # 2. Setup Model
    model = DiM_LLM(config)
    checkpoint_path = "backup_checkpoints/dim_llm_checkpoint.pt"
    
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        # Check if it's a full checkpoint or just model state
        if "model_state" in state_dict:
            model.load_state_dict(state_dict["model_state"])
        else:
            model.load_state_dict(state_dict)
    else:
        print(f"Warning: {checkpoint_path} not found!")
        return

        
    engine = MaskedDiffusionEngine(model, config, device=device)
    model.eval()

    # 3. Test Sampling with Prompt
    prompt = "User: Hello Jarvis, are you functional? | Assistant:"
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    print("Generating response via Masked Diffusion (64 steps)...")
    with torch.no_grad():
        output_tokens = engine.sample(n_samples=1, steps=64, prompt_ids=input_ids)
    
    # Decode using GPT-2
    response = tokenizer.decode(output_tokens[0].tolist(), skip_special_tokens=False)
    print(f"\nPrompt: {prompt}")
    print(f"Jarvis (Full Seq): {response}")

if __name__ == "__main__":
    test_diffusion_inference()
