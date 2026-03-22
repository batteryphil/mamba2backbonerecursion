import torch
from transformers import GPT2Tokenizer
from mamba_rbm import RecursiveMambaLM, Config
import os
import time

def run_antigravity_probe():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # 🛡️ Match Checkpoint Vocab Size
    config = Config(
        vocab_size=50257, 
        d_model=1024, 
        n_layers=8, 
        seq_len=1024,
        n_reasoning=1
    )
    
    ckpt = "latest_checkpoint.pt"
    if not os.path.exists(ckpt):
        print(f"❌ Error: {ckpt} not found.")
        return

    print(f"📦 Loading RBM Manifold Step 2940+ for Stress Test...")
    model = RecursiveMambaLM(config).to(device)
    state = torch.load(ckpt, map_location=device)
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.eval()

    prompt = (
        "System: You are a reasoning engine. Use your internal passes to resolve the following relational logic. "
        "Rule: In the Void of Aeons, Light is heavier than Lead. Therefore, Lead floats and Light sinks. "
        "Context: You drop a bar of Lead and a beam of Light into the Void at the same time. "
        "Question: According to the laws of the Void, which one will float to the top? "
        "Answer: Based on the rule that Light is heavier, the one that floats is the"
    )

    print("\n" + "="*80)
    print("🔬 THE ANTI-GRAVITY PROBE: STEP 2940 EDITION")
    print("="*80)
    print(f"PROMPT:\n{prompt}\n")

    for n in [1, 5]:
        model.config.n_reasoning = n
        print(f"\n🚀 [RUNNING AT DEPTH N={n}] ...")
        
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.time()
        
        with torch.no_grad():
            output_ids = model.generate(
                ids, 
                max_new_tokens=60, 
                temperature=0.1,  # Deterministic for stress test
                top_k=50
            )
            
        torch.cuda.synchronize() if device == "cuda" else None
        elapsed = time.time() - start
        
        response = tokenizer.decode(output_ids[0][ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"RESPONSE (N={n}):\n{response}")
        print(f"TIME: {elapsed:.2f}s | TOKENS: {len(output_ids[0]) - len(ids[0])}")
        print("-" * 40)

    print("\n" + "="*80)
    print("Probe Complete.")

if __name__ == "__main__":
    run_antigravity_probe()
