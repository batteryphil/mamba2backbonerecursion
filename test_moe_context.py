import torch
import sys
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel
from mamba1_engine import MODEL_ID, RecursiveMamba1_PrefixScratchpad, tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THINK_TEXT = "<THINK>"

print("======================================================================")
print("  Test 3: Context-Bleed Test (State Isolation)")
print("======================================================================")

print("[INIT] Loading tokenizers and base backbone...")
backbone = MambaLMHeadModel.from_pretrained(MODEL_ID, device=DEVICE, dtype=torch.bfloat16)
model = RecursiveMamba1_PrefixScratchpad(backbone, lora_rank=4).to(DEVICE)
model.eval()

print("[INIT] Loading LoRA adapters...")
chat_weights = torch.load("saved_weights/mamba130m_lora_chat.pt", map_location="cpu")
rlf_weights = torch.load("saved_weights/mamba130m_v6_best.pt", map_location="cpu")

eos_token_id = tokenizer.eos_token_id

def main():
    if not torch.cuda.is_available():
        print("[ERROR] CUDA required.")
        sys.exit(1)

    print("\\n[TESTING STATE ISOLATION]")
    
    # Simulate the exact orchestrator logic across a multi-turn conversation
    conversation_history = "User: V1=99. V2=V1. What is V2? Answer:\n"
    
    # -- TURN 1: Trigger RLF --
    model.load_state_dict(rlf_weights, strict=False)
    input_ids = tokenizer.encode(conversation_history, return_tensors="pt").to(DEVICE)
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        loops, trace, predicted_ans = model(input_ids, n_dark_inference=2)
    
    print(f"\\nTurn 1 (Logic) -> RLF Engine outputs: {predicted_ans}")
    
    # Append the RLF output precisely as the orchestrator would
    conversation_history += f"AI: <THINK> ... The answer is {predicted_ans}.\n"
    conversation_history += "User: Great. What is water?\nAI: "
    
    print(f"\\nTurn 2 (Chat) -> Passing full context to LoRA_Chat:")
    print("-" * 40)
    print(conversation_history, end="")
    
    # -- TURN 2: Generate with Chat --
    model.load_state_dict(chat_weights, strict=False)
    input_ids = tokenizer.encode(conversation_history, return_tensors="pt").to(DEVICE)
    
    generated_text = ""
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        for _ in range(15):
            hidden_states = model.backbone(input_ids)
            logits = model.lm_head(hidden_states)
            next_token_id = logits[0, -1, :].argmax().item()
            
            if next_token_id == eos_token_id:
                break
                
            token_str = tokenizer.decode([next_token_id])
            generated_text += token_str
            print(token_str, end="", flush=True)
            
            next_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=DEVICE)
            input_ids = torch.cat([input_ids, next_tensor], dim=1)
            
    print()
    print("-" * 40)
    
    print("\\n======================================================================")
    print("  VERDICT: CONTEXT-BLEED TARGET")
    print("======================================================================")
    
    gen_lower = generated_text.lower()
    if "variable" in gen_lower or "v1" in gen_lower or "99" in gen_lower or "answer" in gen_lower:
        print("  ❌ FAIL: Context Bleed Detected. Generation contains mathematical or recurrent artifact logic.")
    else:
        print("  ✅ PASS: State Isolation Perfect. Chat model answered conversationally with no Latent Bridge corruption.")
        
if __name__ == "__main__":
    main()
