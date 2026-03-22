import torch
from mamba_rbm import RecursiveMambaLM, Config
import os

def audit_weight_tying():
    # Load model with config
    config = Config()
    model = RecursiveMambaLM(config)
    
    ckpt = "latest_checkpoint.pt"
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model_state"])
        print(f"✅ Loaded checkpoint: {ckpt}")
    
    # Check 1: Pointer Equality
    ptr_embed = model.token_embed.weight.data_ptr()
    ptr_head = model.lm_head.weight.data_ptr()
    pointer_match = "MATCED" if ptr_embed == ptr_head else "FAILED"
    
    # Check 2: Cosine Similarity (should be 1.0)
    cos = torch.nn.functional.cosine_similarity(model.token_embed.weight, model.lm_head.weight, dim=1).mean().item()
    
    print("\n" + "="*50)
    print("🔬 WEIGHT TYING AUDIT RESULTS")
    print("="*50)
    print(f"Pointer Equality:   {ptr_embed} == {ptr_head} -> {pointer_match}")
    print(f"Cosine Similarity:  {cos:.4f}")
    print(f"Grad Sync Status:   {'ENABLED' if model.token_embed.weight.grad is model.lm_head.weight.grad else 'INDIRECT (Weight Tied)'}")
    print("="*50)

if __name__ == "__main__":
    audit_weight_tying()
