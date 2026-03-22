import torch
import torch.nn as nn
import os
from typing import Dict, Any

def check_numerical_integrity(model_state: Dict[str, torch.Tensor]) -> bool:
    """
    Checks for NaNs or Infs in the model's state dictionary.
    
    Args:
        model_state: The state dictionary of the model.
        
    Returns:
        bool: True if no NaNs or Infs are found, False otherwise.
    """
    has_corruption = False
    for name, param in model_state.items():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"🚨 CRITICAL: {name} contains NaNs or Infs!")
            has_corruption = True
    return not has_corruption

def audit_layer_statistics(model_state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Calculates and prints statistics for key layers in the model.
    
    Args:
        model_state: The state dictionary of the model.
        
    Returns:
        Dict: A summary of the audit including total params and dead neurons.
    """
    print(f"{'Layer Name':<50} | {'Mean':<8} | {'Std':<8} | {'Max':<8} | {'Min':<8}")
    print("-" * 90)
    
    total_params = 0
    dead_neurons = 0
    
    # Filter for interesting layers: Embeddings, Mamba A/D parameters, Norms
    targets = ["embed", "A_log", "D", "norm", "head"]
    
    for name, param in model_state.items():
        if param.dim() < 1:
            continue
            
        total_params += param.numel()
        
        # Dead neuron detection (approximate collapse)
        dead_mask = (param.abs() < 1e-8)
        dead_neurons += dead_mask.sum().item()
        
        if any(t in name for t in targets):
            p_mean = param.mean().item()
            p_std = param.std().item()
            p_max = param.max().item()
            p_min = param.min().item()
            print(f"{name:<50} | {p_mean:8.4f} | {p_std:8.4f} | {p_max:8.4f} | {p_min:8.4f}")
            
    return {"total_params": total_params, "dead_neurons": dead_neurons}

def check_weight_tying(model_state: Dict[str, torch.Tensor]) -> None:
    """
    Verifies if weight tying is preserved between embedding and head.
    
    Args:
        model_state: The state dictionary of the model.
    """
    embed_key = "token_embed.weight"
    head_key = "lm_head.weight"
    
    if embed_key in model_state and head_key in model_state:
        embed_w = model_state[embed_key]
        head_w = model_state[head_key]
        
        # Check pointer equality (if they were loaded into a model)
        # Note: In a state_dict, they are separate tensors unless specifically handled.
        # But we can check if the values are identical.
        are_identical = torch.equal(embed_w, head_w)
        
        # Manual cosine similarity to avoid functional.cosine_similarity oddities with large vectors
        dot_product = torch.dot(embed_w.view(-1), head_w.view(-1))
        norm_embed = torch.norm(embed_w.view(-1))
        norm_head = torch.norm(head_w.view(-1))
        cos_sim = (dot_product / (norm_embed * norm_head)).item()
        
        print("-" * 90)
        print(f"🔗 Weight Tying Audit:")
        print(f"  - Identical Values: {'✅ YES' if are_identical else '❌ NO'}")
        print(f"  - Cosine Similarity: {cos_sim:.10f}")
        
        if cos_sim > 0.999999:
            print("✅ Weight tying is strictly maintained.")
        else:
            print("⚠️ Warning: Weight tying might be drifting or desynced.")
    else:
        print("❌ Weight tying check failed: Keys not found in state dict.")

def perform_weight_audit(ckpt_path: str = "latest_checkpoint.pt") -> None:
    """
    Main entry point for the weight audit protocol.
    
    Args:
        ckpt_path: Path to the checkpoint file.
    """
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found at {ckpt_path}")
        return

    print(f"🔬 INITIALIZING WEIGHT AUDIT: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_state = state["model_state"] if "model_state" in state else state
    
    integrity_pass = check_numerical_integrity(model_state)
    summary = audit_layer_statistics(model_state)
    check_weight_tying(model_state)
    
    total_params = summary["total_params"]
    dead_neurons = summary["dead_neurons"]
    sparsity = (dead_neurons / total_params) * 100 if total_params > 0 else 0
    
    print("-" * 90)
    print(f"📊 Manifold Summary:")
    print(f"  - Total Parameters: {total_params / 1e6:.2f}M")
    print(f"  - Numerical Integrity: {'✅ PASS' if integrity_pass else '❌ FAIL'}")
    print(f"  - Model Sparsity: {sparsity:.6f}%")
    print(f"  - Training Step: {state.get('step', 'unknown')}")
    print(f"  - Vocab Size: {model_state.get('token_embed.weight', torch.zeros(0)).shape[0]}")
    print("-" * 90)

if __name__ == "__main__":
    perform_weight_audit()
