import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from mamba_rbm import RecursiveMambaLM, Config
import os
import gc

def load_v6_model():
    # Force CPU to prevent CUDA OOM since training is running in the background on the GPU.
    device = "cpu"
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
        n_reasoning=3
    )
    
    model = RecursiveMambaLM(config).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    
    return model, tokenizer, device

def build_haystack(tokenizer, total_tk_target=800):
    """ Builds a generic paragraph structure of repeating/semi-random tokens to fill up the context. """
    filler_sentence = "The foundation of the organization heavily relied upon standard operating procedures developed during the late industrial period. System architecture documentation requires extensive maintenance to ensure compliance with modern regulatory frameworks. Managers frequently held long meetings discussing resource utilization and productivity metrics. Employees commuted through dense urban traffic every morning to reach the corporate headquarters located downtown. In recent years, technological advancements have significantly altered communication paradigms within the enterprise. Annual reports highlighted continuous growth despite volatile market conditions and unexpected supply chain disruptions. "
    
    haystack_text = ""
    current_tokens = 0
    
    # Keep appending the filler until we hit the target context size
    while current_tokens < total_tk_target:
        haystack_text += filler_sentence
        current_tokens = len(tokenizer.encode(haystack_text))
        
    return haystack_text

def test_needle(model, tokenizer, device, haystack, needle_depth=0.5, n_reasoning=3):
    needle_sentence = " The secret access code to the mainframe is Protocol-7. "
    question = "\nQuestion: What is the secret access code to the mainframe? Answer:"
    target_answer = " Protocol-7."
    
    # Convert haystack to list of words to easily inject the needle at a specific depth
    words = haystack.split()
    insert_idx = int(len(words) * needle_depth)
    
    # Inject the needle
    words.insert(insert_idx, needle_sentence)
    
    # Reassemble and format the final prompt
    prompt = " ".join(words) + question
    
    # Ensure it's not exceeding the physical 1024 engine limit
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt")
    if prompt_ids.shape[1] > 1000:
        # Trim from the START so we don't accidentally cut off the question/target at the end
        excess = prompt_ids.shape[1] - 1000
        prompt_ids = prompt_ids[:, excess:]
        prompt = tokenizer.decode(prompt_ids[0])
    
    # Prepare text for loss calculation
    full_text = prompt + target_answer
    inputs = tokenizer.encode(full_text, return_tensors="pt").to(device)
    prompt_len = tokenizer.encode(prompt, return_tensors="pt").shape[1]
    
    print(f"\n--- Testing Needle at Depth {needle_depth*100:.0f}% (N={n_reasoning}) ---")
    print(f"Context Length: {inputs.shape[1]} tokens")
    
    # Temporarily override the config's n_reasoning value
    original_n = model.config.n_reasoning
    model.config.n_reasoning = n_reasoning
    
    try:
        with torch.no_grad():
            logits = model(inputs)
            
            # Shift alignment
            shift_logits = logits[0, prompt_len-1:-1, :].contiguous()
            shift_labels = inputs[0, prompt_len:].contiguous()
            
            # 1. Calculate Perplexity/Loss on the correct target string
            loss = F.cross_entropy(shift_logits, shift_labels, reduction='mean').item()
            
            # 2. Grab the actual generated textual response
            # Start at the prompt length and generate exactly the length of the target answer
            gen_ids = inputs[:, :prompt_len]
            target_len = shift_labels.shape[0]
            
            generated_text = ""
            for _ in range(target_len):
                next_logits = model(gen_ids)
                next_token = torch.argmax(next_logits[0, -1, :], dim=-1).unsqueeze(0).unsqueeze(0)
                gen_ids = torch.cat([gen_ids, next_token], dim=1)
                
            generated_text = tokenizer.decode(gen_ids[0, prompt_len:])
            
            print(f"Target Loss (Lower is better): {loss:.4f}")
            print(f"Model Generated: '{generated_text}'")
            print(f"Expected Target: '{target_answer}'")
            
    finally:
         # Restore config
         model.config.n_reasoning = original_n
         
    return loss

if __name__ == "__main__":
    print("==================================================")
    print("Needle In a Haystack (Context Retention) Probe")
    print("==================================================")
    
    model, tokenizer, device = load_v6_model()
    if model is None:
        exit(1)
        
    print(f"Model loaded on {device}. Building ~800 token haystack...")
    haystack_base = build_haystack(tokenizer, total_tk_target=800)
    
    depths = [0.0, 0.5, 0.9]
    n_sweeps = [1, 2, 3]
    
    results = {}
    
    for depth in depths:
        results[depth] = {}
        for n in n_sweeps:
            loss = test_needle(model, tokenizer, device, haystack_base, needle_depth=depth, n_reasoning=n)
            results[depth][n] = loss
            
            # Prevent RAM ballooning since we are on CPU
            gc.collect()

    print("\n--- SUMMARY OF CROSS-ENTROPY LOSS ON TARGET ---")
    print(f"{'Depth':<10} | {'N=1':<10} | {'N=2':<10} | {'N=3':<10}")
    print("-" * 45)
    for depth in depths:
        r = results[depth]
        print(f"{depth*100:<9.0f}% | {r[1]:<10.4f} | {r[2]:<10.4f} | {r[3]:<10.4f}")
    
    print("\nTest Complete.")
