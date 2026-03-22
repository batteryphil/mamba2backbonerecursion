import torch
import json
import re
import os
from transformers import GPT2Tokenizer
from mamba_llm_diffusion import DiM_LLM, MaskedDiffusionEngine, Config
import tools_jarvis

def parse_xml_tag(text, tag_name):
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def run_chat():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading 'Jarvis' (DiM-LLM v3 / GPT-2 BPE) on {device}...")

    # 1. Initialize GPT-2 Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    
    config = Config(
        vocab_size=len(tokenizer),
        d_model=1024,
        n_layers=11,
        seq_len=256
    )

    # 2. Setup Model
    model = DiM_LLM(config)
    checkpoint_path = "dim_llm_ema_checkpoint.pt" if os.path.exists("dim_llm_ema_checkpoint.pt") else "dim_llm_checkpoint.pt"
    
    if os.path.exists(checkpoint_path):
        print(f"Restoring Jarvis Weights from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("Warning: Jarvis has not been trained yet. Outputs will be stochastic.")

        
    engine = MaskedDiffusionEngine(model, config, device=device)
    model.eval()

    if os.path.exists("system_prompt.txt"):
        with open("system_prompt.txt", "r") as f:
            system_prompt = f.read()
    else:
        system_prompt = "You are Jarvis."

    print("\n--- JARVIS INTERFACE (DiM-LLM v3 / Masked Diffusion) ---")
    print("Welcome back, Philip. GPT-2 BPE Tokenizer active.")
    
    conversation_history = system_prompt + "\n"

    while True:
        user_input = input("\nPhilip -> ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        conversation_history += f"User: {user_input}\nAssistant: "
        
        thinking_complete = False
        iteration = 0
        
        print("Jarvis is thinking (Denoising)...")
        while not thinking_complete and iteration < 3:
            iteration += 1
            input_ids = torch.tensor(tokenizer.encode(conversation_history), dtype=torch.long).unsqueeze(0).to(device)
            
            # 🧠 Hippocampus Memory Bank: Dense Context Pooling
            # If conversation history exceeds 50% of the sequence length, 
            # compress the "old" part into a dense vector via mean-pooling.
            if input_ids.shape[1] > (config.seq_len // 2):
                split_idx = input_ids.shape[1] - (config.seq_len // 2)
                old_context_ids = input_ids[:, :split_idx]
                recent_ids = input_ids[:, split_idx:]
                
                with torch.no_grad():
                    # Map IDs to continuous embeddings and mean-pool
                    old_embeddings = model.token_embed(old_context_ids) # (1, L_old, D)
                    context_bank = old_embeddings.mean(dim=1, keepdim=True) # (1, 1, D)
                
                input_ids = recent_ids
                print(f"   [Hippocampus] Compressed {split_idx} tokens into dense context.")
            else:
                context_bank = None

            with torch.no_grad():
                output_tokens = engine.sample(
                    n_samples=1, 
                    steps=32, 
                    prompt_ids=input_ids, 
                    context_bank=context_bank
                )
            
            # Decode the segment AFTER the conversation history
            full_decoded = tokenizer.decode(output_tokens[0].tolist(), skip_special_tokens=False)
            
            # Extract just the newly generated part (Assistant's response)
            # This is tricky with Masked Diffusion because it fills from left to right usually.
            # But here prompt is fixed at the start.
            prompt_decoded = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
            response_segment = full_decoded[len(prompt_decoded):].strip()
            
            # --- Check for TOOL Use (Robust Parser) ---
            tool_name = None
            params = {}
            tool_content = parse_xml_tag(response_segment, "tool")
            
            if tool_content:
                print(f"   [JARVIS TOOL EXEC] -> {tool_content}")
                try:
                    # Pattern 1: <tool name="terminal" params='{"command": "..."}'>...</tool>
                    name_match = re.search(r'name=[\"\']([^\"\']+)[\"\']', response_segment)
                    params_match = re.search(r'params=[\"\']?({.*?})[\"\']?', response_segment, re.DOTALL)
                    
                    if name_match:
                        tool_name = name_match.group(1)
                    
                    if params_match:
                        # Sanitize before JSON parse
                        raw_params = params_match.group(1).replace("\\'", "'")
                        params = json.loads(raw_params)
                    elif tool_content.strip():
                        # Pattern 2: bare expression like add(45, 12) or nvidia-smi
                        # Try to call terminal as fallback
                        tool_name = tool_name or "terminal"
                        params = {"command": tool_content.strip()}
                    
                    result = tools_jarvis.call_tool(tool_name or "terminal", params)
                    
                except json.JSONDecodeError:
                    result = {"error": "Jarvis generated malformed tool params. Skipping."}
                except Exception as e:
                    result = {"error": f"Tool execution failed: {str(e)}"}
                
                observation = f"\n<observation>{json.dumps(result)}</observation>\n"
                conversation_history += f"{response_segment}{observation}"
                print(f"Assistant -> {response_segment}")
            else:
                print(f"Assistant -> {response_segment}")
                conversation_history += f"{response_segment}\n"
                thinking_complete = True

                
        if iteration >= 3:
            print("   [Jarvis Memo] Reached maximum reasoning steps.")

if __name__ == "__main__":
    run_chat()

