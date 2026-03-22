import torch
import sys
import time
import os
from transformers import GPT2Tokenizer
from mamba_rbm import RecursiveMambaLM, Config
import warnings
warnings.filterwarnings("ignore")

def load_model(ckpt_path="latest_checkpoint.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Model State from {ckpt_path} onto {device.upper()}...")
    
    if not os.path.exists(ckpt_path):
        print(f"\n[ERROR] Checkpoint '{ckpt_path}' not found.")
        print("Please ensure the training script has finished or saved at least once.")
        sys.exit(1)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Build the 150M parameter architecture
    config = Config(
        vocab_size=len(tokenizer), 
        d_model=1024, 
        n_layers=8, 
        seq_len=1024,
        n_reasoning=3 # Default to N=3
    )
    
    model = RecursiveMambaLM(config).to(device)
    
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    
    return model, tokenizer, device

def chat_loop():
    print("\n" + "="*50)
    print("🧠 Recursive Mamba (150M) - Interactive Inference")
    print("="*50)
    
    model, tokenizer, device = load_model()
    
    print("\nReady! Special Commands:")
    print("  /depth N    -> Change recursive reasoning depth (default 3)")
    print("  /temp T     -> Change randomness/temperature (default 0.3)")
    print("  /quit       -> Exit chat")
    print("="*50 + "\n")

    current_depth = 3
    current_temp = 0.3

    while True:
        try:
            prompt = input("\n👤 You: ")
        except (KeyboardInterrupt, EOFError):
            break
            
        if not prompt.strip():
            continue
            
        if prompt.strip().lower() in ['/quit', '/exit']:
            print("Shutting down... 👋")
            break
            
        # Command Handling
        if prompt.startswith("/depth"):
            try:
                current_depth = int(prompt.split()[1])
                print(f"⚙️ Reasoning depth set to N={current_depth}")
                continue
            except:
                print("Usage: /depth [number]")
                continue
                
        if prompt.startswith("/temp"):
            try:
                current_temp = float(prompt.split()[1])
                print(f"⚙️ Temperature set to {current_temp}")
                continue
            except:
                print("Usage: /temp [number]")
                continue

        # Inference
        model.config.n_reasoning = current_depth
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        print(f"\n🤖 System (Thinking at N={current_depth}...)\n")
        
        start_time = time.time()
        
        with torch.no_grad():
            output_ids = model.generate(
                prompt_ids, 
                max_new_tokens=80, 
                temperature=current_temp,
                top_k=50 
            )
            
        end_time = time.time()
        
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Strip out the prompt text to only show what the model generated
        generated_text = response[len(prompt):].strip()
        
        # Stream the output artificially for aesthetics
        for char in generated_text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.01)
            
        print(f"\n\n[⏱️ Inference Time: {end_time - start_time:.2f}s]")
        print("-" * 50)

if __name__ == "__main__":
    chat_loop()
