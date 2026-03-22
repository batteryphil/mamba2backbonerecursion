import torch
import json
from mamba_llm_diffusion import Tokenizer, DiM_LLM, LlmTrainer
import os

def run_chat():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading MambaDiffusion Chatbot on {device}...")

    # 1. Load context
    with open("train_data.json", "r") as f:
        train_data = json.load(f)
    
    tokenizer = Tokenizer(vocab_size=300)
    tokenizer.build_vocab(train_data)
    
    # 2. Setup Model
    model = DiM_LLM(vocab_size=len(tokenizer.vocab), d_model=128)
    if os.path.exists("dim_llm_checkpoint.pt"):
        print("Loading trained weights...")
        model.load_state_dict(torch.load("dim_llm_checkpoint.pt", map_location=device))
    else:
        print("Warning: No checkpoint found. Model will output noise.")
        
    trainer = LlmTrainer(model, device=device)
    model.eval()

    print("\n--- MambaDiffusion Interactive Interface ---")
    print("Type 'exit' to quit. This model uses Mamba-based Diffusion to denoise its response.")
    
    while True:
        prompt = input("\nUser -> ")
        if prompt.lower() == 'exit':
            break
            
        # Encode
        input_tokens = tokenizer.encode(prompt, max_len=64).unsqueeze(0).to(device)
        
        # Geberate using Diffusion process
        print("Thinking (Denoising via Mamba)...")
        with torch.no_grad():
            output_tokens = trainer.sample(input_tokens, n_gen_steps=100)
            
        response = tokenizer.decode(output_tokens[0])
        print(f"Assistant -> {response}")

if __name__ == "__main__":
    run_chat()
