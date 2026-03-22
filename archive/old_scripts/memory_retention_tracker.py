import torch
from transformers import GPT2Tokenizer
import os
import sys

# Suppress warnings
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from mamba_rbm import RecursiveMambaLM, Config

def track_hidden_state_norm():
    """
    Tracks the L2 norm of the hidden state for a specific token across recursive passes.
    Adapted for the Dual-Path Mamba Architecture (RBM v6.3).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    ckpt_path = "latest_checkpoint.pt"
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: {ckpt_path} not found.")
        return

    print(f"📦 Loading RBM Checkpoint: {ckpt_path} on {device.upper()}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    n_passes = 3
    config = Config(
        vocab_size=len(tokenizer), 
        d_model=1024, 
        n_layers=8, 
        seq_len=1024,
        n_reasoning=n_passes 
    )
    
    model = RecursiveMambaLM(config).to(device)
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    
    model.eval()

    prompt = "Logic problem: If x = 5 and y = x + 2, then what is the value of y? The value of y is"
    target_word = "5"
    
    print(f"\n[PROMPT]: \"{prompt}\"")
    
    # Tokenize input and move to device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # Find the sequence index by checking the decoded text
    target_idx = -1
    for i in range(len(input_ids[0])):
        if target_word in tokenizer.decode([input_ids[0][i]]):
            target_idx = i
            break
            
    if target_idx == -1:
        print(f"Target token '{target_word}' not found in the prompt sequence.")
        # Debug print
        print("Sequence tokens: ", [tokenizer.decode([t]) for t in input_ids[0]])
        return
        
    print(f"Tracking hidden state for '{target_word}' at sequence index {target_idx}")

    # Storage for the L2 norms across passes. 
    # In RBM v6.3, the recursion happens *inside* the DualCausalMambaBlock.
    # The provided script assumes n_reasoning is passed to the forward pass, 
    # but in our architecture, it's looping internally in RecursiveMambaLM.forward
    
    l2_norms = []
    
    # We need to hook into the logic path specifically.
    # The logic path is calculated in `layer["mamba"](...)` where `use_logic=True`.
    # Let's hook the output of the mamba block and filter for logic passes.

    # Counter to track which internal reasoning step we are on for a given forward pass
    reasoning_step_counter = 0

    def hook_fn(module, input_tuple, output):
        nonlocal reasoning_step_counter
        # The input arguments to DualCausalMambaBlock forward are (u, use_logic=False)
        # Hooks don't easily give kwargs, but we can deduce it from the execution flow
        # In RecursiveMambaLM.forward:
        # 1. intuition_out = mamba(normed_x, use_logic=False)
        # 2. for _ in range(n_reasoning):
        # 3.     logic_out = mamba(norm(logic_out), use_logic=True)
        
        # We only care about the outputs during the logic loops
        # We can track the calls. First call per layer is intuition, next N are logic.
        # But an easier way is to just let the loop run normally, and tracking logic.
        
        # Let's simplify: hook the LayerNorm inside the loop? No, mamba output is better.
        # Because we want to see the retention *after* the logic block processes it.
        pass

    # A more robust hook for the RBM v6.3 architecture:
    # We will hook the `mamba` module of the LAST layer.
    
    call_count = 0
    
    def rbm_hook(module, inputs, output):
        nonlocal call_count
        # inputs is a tuple: (normed_x,)  -- kwargs like use_logic are not in this tuple sadly in standard PyTorch hooks
        # However, we know the exact call sequence per token generation forward pass:
        # For a single forward pass through the whole network:
        # Layer 0: 1 intuition call, N logic calls
        # ...
        # Layer 7: 1 intuition call, N logic calls
        
        # So for the last layer (Layer 7), call 0 is intuition, calls 1 to N are the reasoning passes!
        
        if call_count > 0 and call_count <= n_passes:
            # This is a logic pass!
            hidden_states = output
            token_state = hidden_states[0, target_idx, :]
            norm = torch.norm(token_state, p=2).item()
            l2_norms.append(norm)
            
        call_count += 1
        
        # Reset counter after a full forward pass of the layer
        if call_count > n_passes:
            call_count = 0

    handle = model.layers[-1]["mamba"].register_forward_hook(rbm_hook)
    
    with torch.no_grad(), torch.amp.autocast('cuda' if device == 'cuda' else 'cpu', dtype=torch.bfloat16):
        # We only do a single forward pass for the prompt to see the initial state retention
        model(input_ids)
        
    handle.remove()
    
    print("\n--- L2 Norm Tracking Results for '5' at Layer 8 ---")
    if not l2_norms:
        print("No logic passes were intercepted. Check architecture configuration.")
        return

    for i, norm in enumerate(l2_norms):
        print(f"Logic Iteration {i+1} (N={n_passes}): {norm:.4f}")
        
    if len(l2_norms) > 1:
        retention_rate = (l2_norms[-1] / l2_norms[0]) * 100
        print(f"\nState Retention from Logic Pass 1 to Pass {len(l2_norms)}: {retention_rate:.2f}%")
        if retention_rate < 10.0:
            print("Status: ❌ DECAY DETECTED. The model is dropping the variable state.")
        else:
            print("Status: ✅ STABLE. The model is actively holding the variable in memory.")

if __name__ == "__main__":
    track_hidden_state_norm()
