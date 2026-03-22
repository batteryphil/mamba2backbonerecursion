import torch
import torch.nn as nn
from mamba_rbm import RecursiveMambaLM, Config
import os

def audit_weights():
    ckpt_path = "latest_checkpoint.pt"
    if not os.path.exists(ckpt_path):
        print("❌ Checkpoint not found.")
        return

    print(f"🔬 Auditing Manifold: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model_state = state["model_state"] if "model_state" in state else state

    # Protocol v6.3 Topology
    # d_model=1024, n_layers=8
    
    stats = []
    has_nans = False
    dead_neurons = 0
    total_params = 0

    print(f"{'Layer Name':<50} | {'Mean':<8} | {'Std':<8} | {'Max':<8} | {'Min':<8}")
    print("-" * 90)

    for name, param in model_state.items():
        if "weight" not in name or param.dim() < 1:
            continue
            
        total_params += param.numel()
        
        # Check for Numerical Corruption
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"🚨 CRITICAL: {name} contains NaNs or Infs!")
            has_nans = True
            
        # Check for Dead Neurons (Approximate collapse)
        dead_mask = (param.abs() < 1e-8)
        dead_neurons += dead_mask.sum().item()

        p_mean = param.mean().item()
        p_std = param.std().item()
        p_max = param.max().item()
        p_min = param.min().item()
        
        # Print key layers only for brevity
        if any(x in name for x in ["embedding", "mamba.A_log", "mamba.D", "recurrent"]):
            print(f"{name:<50} | {p_mean:8.4f} | {p_std:8.4f} | {p_max:8.4f} | {p_min:8.4f}")

    print("-" * 90)
    print(f"📊 Manifold Summary:")
    print(f"  - Total Parameters: {total_params / 1e6:.2f}M")
    print(f"  - Numerical Integrity: {'✅ PASS' if not has_nans else '❌ FAIL'}")
    print(f"  - Model Sparsity: {(dead_neurons / total_params)*100:.4f}%")
    print(f"  - Training Step: {state.get('step', 'unknown')}")

if __name__ == "__main__":
    audit_weights()
