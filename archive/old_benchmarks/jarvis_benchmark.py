import time
import torch
import os
import glob
import copy
import json
from transformers import GPT2Tokenizer
from mamba_llm_diffusion import DiM_LLM, MaskedDiffusionEngine, Config

def pick_latest_checkpoint():
    # Use absolute paths or handle relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        "dim_llm_ema_checkpoint.pt",
        "dim_llm_ema_best.pt",
        "dim_llm_checkpoint.pt",
        *sorted(glob.glob("dim_llm_ema_epoch*.pt"), reverse=True),
        *sorted(glob.glob("dim_llm_epoch*.pt"), reverse=True),
    ]
    for candidate in candidates:
        full_path = os.path.join(script_dir, candidate)
        if os.path.exists(full_path):
            return full_path
    return None

def run_benchmark(engine, tokenizer):
    test_cases = [
        {
            "category": "MATH (GSM8K)",
            "prompt": "[Reasoning] Question: If Bob has 3 RTX 3060s and buys 2 more for $200 each, how many GPUs does he have and what was the cost of the new ones? | Assistant:",
            "expected_logic": "Addition (3+2) and Multiplication (2*200)"
        },
        {
            "category": "TOOL-USE (eBay)",
            "prompt": "[Tool Call] User: Find an MSI Gaming Laptop on eBay for under $800. | Assistant:",
            "expected_format": "<tool>search_ebay(query='MSI Gaming Laptop', max_price=800)</tool>"
        },
        {
            "category": "INSTINCT (Greeting)",
            "prompt": "[Greeting] User: Hello there, Jarvis! How are you today? | Assistant:",
            "expected_mode": "FACT (32 Steps)"
        }
    ]

    print(f"\n🚀 Starting DiM-LLM v3.2 Benchmark | Context: 1024 | Hardware: RTX 3060\n" + "-"*60)
    
    for case in test_cases:
        print(f"CATEGORY: {case['category']}")
        print(f"PROMPT:   {case['prompt'][:70]}...")
        
        start_time = time.time()
        
        # Prepare inputs
        prompt_ids = tokenizer.encode(case["prompt"], return_tensors="pt").to(engine.device)
        
        # Triggering Adaptive Denoising Logic
        # adaptive_sample returns (output_ids, steps_taken, final_entropy)
        output_ids, steps, final_entropy = engine.adaptive_sample(
            n_samples=1,
            prompt_ids=prompt_ids,
            max_steps=1000,
            entropy_threshold=0.2 # As requested by the Arkansas Architect
        )
        
        duration = time.time() - start_time
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Calculate tokens generated (total seq_len)
        tokens_count = output_ids.shape[1]
        tps = tokens_count / duration if duration > 0 else 0
        
        print(f"STEPS:    {steps}/1000")
        print(f"ENTROPY:  {final_entropy:.4f}")
        print(f"TPS (inf): {tps:.2f}")
        print(f"OUTPUT:   {output_text[:500]}...")
        print("-" * 30)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Performance flags for RTX 3060
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'mask_token': '[MASK]'})
    
    SEQ_LEN = 1024
    config = Config(
        vocab_size=50258, # Hardcoded to match checkpoint
        d_model=1024, 
        n_layers=11, 
        seq_len=SEQ_LEN
    )
    
    model = DiM_LLM(config).to(device)
    ema_model = copy.deepcopy(model).to(device)
    
    ckpt_path = pick_latest_checkpoint()
    if ckpt_path:
        print(f"[*] Resuming from latest checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])
            ema_model.load_state_dict(state["ema_state"])
        else:
            model.load_state_dict(state)
            ema_model.load_state_dict(state)
        print("[✓] Weights loaded successfully.")
    else:
        print("[!] Fatal: No checkpoint found in directory.")
        exit(1)
        
    engine = MaskedDiffusionEngine(model, config, device=device)
    engine.ema_model = ema_model
    
    run_benchmark(engine, tokenizer)
