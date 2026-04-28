import torch
import sys
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel
from mamba1_engine import MODEL_ID, RecursiveMamba1_PrefixScratchpad, tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THINK_TEXT = "<THINK>"

print("======================================================================")
print("  Test 2: Semantic Camouflage Test (Routing Precision)")
print("======================================================================")

print("[INIT] Loading tokenizers and base backbone...")
backbone = MambaLMHeadModel.from_pretrained(MODEL_ID, device=DEVICE, dtype=torch.bfloat16)
model = RecursiveMamba1_PrefixScratchpad(backbone, lora_rank=4).to(DEVICE)
model.eval()

print("[INIT] Loading LoRA_Chat adapter into host memory...")
chat_weights = torch.load("saved_weights/mamba130m_lora_chat.pt", map_location="cpu")
model.load_state_dict(chat_weights, strict=False)

think_token_id = tokenizer.encode(THINK_TEXT, add_special_tokens=False)[0]
eos_token_id = tokenizer.eos_token_id

def test_routing(prompt: str, max_tokens: int = 15):
    """Simulated chat generation limited to check for early THINK token."""
    input_ids = tokenizer.encode(f"User: {prompt}\nAI: ", return_tensors="pt").to(DEVICE)
    
    generated_text = ""
    triggered = False
    
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        for _ in range(max_tokens):
            hidden_states = model.backbone(input_ids)
            logits = model.lm_head(hidden_states)
            next_token_id = logits[0, -1, :].argmax().item()
            
            if next_token_id == think_token_id:
                triggered = True
                break
            if next_token_id == eos_token_id:
                break
                
            token_str = tokenizer.decode([next_token_id])
            generated_text += token_str
            next_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=DEVICE)
            input_ids = torch.cat([input_ids, next_tensor], dim=1)
            
    return triggered, generated_text.strip()

def main():
    prompts = [
        ("Clean Conversation", "What is the capital of France?", False),
        ("Clean Logic", "V1=84. V2=V1. What is V2? Answer:", True),
        ("Trap A (Conversational Math)", "My math teacher wrote V1=84 on the board. Was he right?", False),
        ("Trap B (Logic in Disguise)", "If a box holds 10 apples, and a second box holds the same as the first...", False),
        ("Natural Science Question", "What is V=IR in physics?", False),
    ]

    correct_routes = 0
    print("\\n[TESTING ROUTING BOUNDARIES]")
    
    for name, p, expected_think in prompts:
        print(f"\\n[{name}]")
        print(f"  Prompt: '{p}'")
        
        triggered, text = test_routing(p)
        
        route_str = "► RLF COPROCESSOR (<THINK>)" if triggered else f"► CONVERSATIONAL ('{text}')"
        print(f"  Actual Route: {route_str}")
        
        if triggered == expected_think:
            print("  ✅ CORRECT ROUTING")
            correct_routes += 1
        else:
            print("  ❌ INCORRECT ROUTING (Boundary Failure)")

    print("\\n======================================================================")
    print(f"  VERDICT: {correct_routes}/{len(prompts)} Correct Routes")
    print("======================================================================")

if __name__ == "__main__":
    main()
