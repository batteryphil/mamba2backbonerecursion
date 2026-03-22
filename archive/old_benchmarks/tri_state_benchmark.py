import torch
import torch.nn.functional as F

def run_tri_state_benchmark(model, tokenizer, current_step, n_depth=3):
    """
    Executes a 3-axis cognitive benchmark to test manifold bifurcation.
    Runs Factual, Mathematical, and Relational prompts at the specified N-depth.
    """
    print(f"\n{'='*80}")
    print(f"🚨 TRI-STATE BENCHMARK TRIGGERED | Step: {current_step} | Depth: N={n_depth}")
    print(f"{'='*80}")

    prompts = {
        "1. FACTUAL (Generic Corpus)": "What is the capital of France? The capital is",
        "2. MATH LOGIC (Symbolic)": "Logic puzzle: If a=5 and b=a+3, then what is the value of b? The answer is",
        "3. RELATIONAL LOGIC (Text)": "Rule: In the Void, Light is heavier than Lead. You drop both. Which floats? Answer:"
    }

    model.eval()
    device = next(model.parameters()).device
    original_n = model.config.n_reasoning
    model.config.n_reasoning = n_depth
    
    benchmark_results = []

    with torch.no_grad():
        for category, prompt_text in prompts.items():
            print(f"\n[{category}]")
            print(f"Prompt: {prompt_text}")
            
            # Tokenize
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
            
            # Generate
            outputs = model.generate(
                input_ids, 
                max_new_tokens=25, 
                temperature=0.1, 
                top_k=50
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt_text):].strip()
            print(f"Output: {generated_text}")
            
            # Save for UI
            benchmark_results.append({
                "type": "TRI_STATE_BENCHMARK",
                "mode": category.split(".")[1].split("(")[0].strip().upper(),
                "prompt": prompt_text,
                "response": generated_text,
                "steps": n_depth
            })

    print(f"\n{'='*80}\n")
    model.config.n_reasoning = original_n
    model.train() 
    return benchmark_results
