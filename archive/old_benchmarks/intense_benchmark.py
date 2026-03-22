import torch
import torch.nn.functional as F
import time
import os
import json
import psutil
from transformers import GPT2Tokenizer
from mamba_rbm import RecursiveMambaLM, Config

def get_stats(tensor):
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "max": tensor.max().item(),
        "min": tensor.min().item()
    }

def calculate_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-9)).item()

def run_intense_benchmark():
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
    
    report = {
        "step": state.get("step", "unknown"),
        "weight_stats": {},
        "generative_results": [],
        "telemetry": []
    }

    # 1. Weight Audit
    print("🔬 Auditing Model Manifold Weights...")
    weight_stats = {}
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() > 1:
            weight_stats[name] = get_stats(param)
    report["weight_stats"] = weight_stats

    # 2. Generative Probe
    test_cases = [
        {"id": "IDENTITY", "prompt": "User: Who are you? | Assistant:"},
        {"id": "LOGIC", "prompt": "Premise: Alice is taller than Bob. Question: Who is shorter? Answer:"},
        {"id": "KNOWLEDGE", "prompt": "The largest planet in our solar system is"},
        {"id": "CREATIVE", "prompt": "In a galaxy far, far away,"},
        {"id": "INVERSE_PHYSICS", "prompt": "Rule: Light is heavier than Lead. Lead floats. Question: Which one floats? Answer:"}
    ]

    depth_metrics = []
    depths = [1, 2, 3, 5, 10]
    
    for n in depths:
        model.config.n_reasoning = n
        print(f"\n🚀 PROBING DEPTH N={n}...")
        depth_results = {"n": n, "cases": []}
        
        for case in test_cases:
            inputs = tokenizer.encode(case["prompt"], return_tensors="pt").to(device)
            start_gen = time.time()
            
            with torch.no_grad():
                # Get first token entropy
                logits = model(inputs)[:, -1, :]
                entropy = calculate_entropy(logits)
                
                # Generate sequence
                output_ids = model.generate(
                    inputs, 
                    max_new_tokens=40, 
                    temperature=0.7, 
                    top_k=50
                )
            
            gen_time = time.time() - start_gen
            response = tokenizer.decode(output_ids[0][inputs.shape[1]:], skip_special_tokens=True)
            
            # Hardware check during generation
            cpu_temp = "N/A"
            try:
                temps = psutil.sensors_temperatures()
                if 'coretemp' in temps:
                    cpu_temp = max(t.current for t in temps['coretemp'])
            except: pass

            case_res = {
                "id": case["id"],
                "prompt": case["prompt"],
                "response": response.strip(),
                "entropy": entropy,
                "latency_sec": gen_time,
                "cpu_temp": cpu_temp
            }
            depth_results["cases"].append(case_res)
            print(f"  [{case['id']}] -> {response.strip()[:60]}... (Entropy: {entropy:.2f})")
            
        depth_metrics.append(depth_results)

    report["generative_results"] = depth_metrics

    # Save Report
    with open("benchmark_results.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n✅ Intensity Benchmark Complete. Report saved to benchmark_results.json")

if __name__ == "__main__":
    run_intense_benchmark()
