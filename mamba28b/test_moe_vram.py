import torch
import sys
import time
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel
from mamba1_engine import MODEL_ID, RecursiveMamba1_PrefixScratchpad, tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THINK_TEXT = "<THINK>"

print("======================================================================")
print("  Test 1: MoE VRAM Fragmentation Stress Test (500 Swaps)")
print("======================================================================")

print("[INIT] Loading tokenizers and base backbone...")
backbone = MambaLMHeadModel.from_pretrained(MODEL_ID, device=DEVICE, dtype=torch.bfloat16)
model = RecursiveMamba1_PrefixScratchpad(backbone, lora_rank=4).to(DEVICE)
model.eval()

print("[INIT] Loading LoRA adapters into host memory...")
chat_weights = torch.load("saved_weights/mamba130m_lora_chat.pt", map_location="cpu")
rlf_weights = torch.load("saved_weights/mamba130m_v6_best.pt", map_location="cpu")

think_token_id = tokenizer.encode(THINK_TEXT, add_special_tokens=False)[0]
eos_token_id = tokenizer.eos_token_id

def chat_generate(prompt: str, max_tokens: int = 10):
    """Simulated chat generation limited to 10 tokens for speed."""
    input_ids = tokenizer.encode(f"User: {prompt}\nAI: ", return_tensors="pt").to(DEVICE)
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        for _ in range(max_tokens):
            hidden_states = model.backbone(input_ids)
            logits = model.lm_head(hidden_states)
            next_token_id = logits[0, -1, :].argmax().item()
            if next_token_id == think_token_id:
                return True
            if next_token_id == eos_token_id:
                break
            next_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=DEVICE)
            input_ids = torch.cat([input_ids, next_tensor], dim=1)
    return False

def format_mb(bytes_val):
    return f"{bytes_val / (1024 * 1024):.1f} MB"

def main():
    if not torch.cuda.is_available():
        print("[ERROR] CUDA required for VRAM fragmentation test.")
        sys.exit(1)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start_mem = torch.cuda.memory_allocated()
    print(f"\\n[START] Initial VRAM Allocated: {format_mb(start_mem)}")
    
    n_cycles = 500
    mems = []
    
    start_t = time.time()
    for i in range(1, n_cycles + 1):
        # 1. Chat Prompt
        model.load_state_dict(chat_weights, strict=False)
        chat_generate("What is the capital of France?", max_tokens=5)
        
        # simulated logic intercept (force swap)
        model.load_state_dict(rlf_weights, strict=False)
        
        # 2. Logic Coprocessor execution
        rlf_prompt = "V1=123. V2=V1. What is V2? Answer:"
        input_ids = tokenizer.encode(rlf_prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            model(input_ids, n_dark_inference=2)
            
        # Record memory after cycle
        current_mem = torch.cuda.memory_allocated()
        mems.append(current_mem)
        
        if i % 50 == 0:
            elapsed = time.time() - start_t
            peak = torch.cuda.max_memory_allocated()
            delta = current_mem - start_mem
            print(f"Cycle {i:3d}/{n_cycles} | Current VRAM: {format_mb(current_mem)} "
                  f"| Peak: {format_mb(peak)} | Delta: {format_mb(delta)} | {elapsed:.1f}s")

    end_mem = mems[-1]
    net_leak = end_mem - start_mem
    
    print("\\n======================================================================")
    print("  VERDICT: VRAM FRAGMENTATION TARGET")
    print("======================================================================")
    if net_leak > 1024 * 1024: # > 1 MB leak
        print(f"  ❌ FAIL: VRAM leaked {format_mb(net_leak)} over {n_cycles} cycles.")
    else:
        print(f"  ✅ PASS: VRAM isolated perfectly. Final Variance: {format_mb(net_leak)}")
        
if __name__ == "__main__":
    main()
