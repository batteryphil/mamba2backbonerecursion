import torch
from transformers import GPT2Tokenizer
from datasets import load_dataset
from mamba_rbm import RecursiveMambaLM, Config
import os
import warnings
warnings.filterwarnings("ignore")

def load_v6_model():
    device = "cpu" # 🛡️ Training is using 10.6GB VRAM, forcing CPU for inference test
    ckpt_path = "latest_checkpoint.pt"
    
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return None, None, None

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    config = Config(
        vocab_size=len(tokenizer), 
        d_model=1024, 
        n_layers=8, 
        seq_len=1024,
        n_reasoning=3 # 🚀 Locked at N=3 for testing reasoning
    )
    
    model = RecursiveMambaLM(config).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    
    return model, tokenizer, device

def test_generative_reasoning(model, tokenizer, device):
    print("\n" + "="*60)
    print("🔬 ADVANCED REASONING PROBE (Generative Analysis)")
    print("="*60)
    print(f"Model Depth: N={model.config.n_reasoning} | Loading advanced prompts...")
    
    # GSM8K - Grade School Math (Step-by-step logic)
    print("\n[Test 1: GSM8K Math Logic]")
    gsm_prompt = "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? \nStep-by-step Answer:"
    run_and_print(model, tokenizer, device, gsm_prompt)
    
    # ARC-Challenge - AI2 Reasoning Challenge (Science/Logic)
    print("\n[Test 2: ARC Challenge (Science/Logic)]")
    arc_prompt = "Question: Which of the following is the best example of a chemical change? (A) rubbing alcohol evaporating (B) a cake baking (C) an ice cube melting (D) freezing water. \nReasoning:"
    run_and_print(model, tokenizer, device, arc_prompt)
    
    # LogiQA - Deductive Reasoning
    print("\n[Test 3: LogiQA (Deductive Constraint)]")
    logiqa_prompt = "Rule: All squares are rectangles. Some rectangles are rhombuses. Therefore, some squares are rhombuses. \nQuestion: Is this logical deduction true or false? \nAnalysis:"
    run_and_print(model, tokenizer, device, logiqa_prompt)
    
def run_and_print(model, tokenizer, device, prompt):
    print("-" * 60)
    print(f"Prompt:\n{prompt}")
    print("\nModel Output (N=3 Thinking...):")
    
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            prompt_ids, 
            max_new_tokens=60, 
            temperature=0.3, # Low temp for deterministic logic
            top_k=10
        )
        
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_text = response[len(prompt):].strip()
    
    print(f"\n>> {generated_text}")
    print("-" * 60)

if __name__ == "__main__":
    model, tokenizer, device = load_v6_model()
    if model:
        test_generative_reasoning(model, tokenizer, device)
