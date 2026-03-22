import torch
import os

def migrate_state_dict(state_dict):
    """Migrates from Separate Mamba (v2) to Fused BiMamba (v3)."""
    new_state_dict = {}
    
    # Track which projections we've already migrated (since they are now shared)
    migrated_projs = set()

    for k, v in state_dict.items():
        if ".mamba." not in k:
            new_state_dict[k] = v
            continue
        
        # k looks like: layers.0.mamba.mamba_fwd.in_proj.weight
        parts = k.split(".")
        # ['layers', '0', 'mamba', 'mamba_fwd', 'in_proj', 'weight']
        prefix = ".".join(parts[:3]) # layers.0.mamba
        direction = parts[3]         # mamba_fwd or mamba_bwd
        suffix = ".".join(parts[4:]) # in_proj.weight, etc.

        if suffix in ["in_proj.weight", "out_proj.weight"]:
            target_key = f"{prefix}.{suffix}"
            if target_key not in migrated_projs:
                if direction == "mamba_fwd":
                    new_state_dict[target_key] = v.clone()
                    migrated_projs.add(target_key)
                else:
                    # If bwd comes first (unlikely), or if fwd missing, use bwd
                    pass 
            continue

        # Remap inner SSM parameters
        new_dir = "fwd" if direction == "mamba_fwd" else "bwd"
        
        # Mamba v2 internal names vs BiMambaBlock names
        map_table = {
            "conv1d.weight": f"{new_dir}_conv1d.weight",
            "conv1d.bias":   f"{new_dir}_conv1d.bias",
            "x_proj.weight": f"{new_dir}_x_proj.weight",
            "dt_proj.weight": f"{new_dir}_dt_proj.weight",
            "dt_proj.bias":   f"{new_dir}_dt_proj.bias",
            "A_log":          f"{new_dir}_A_log",
            "D":              f"{new_dir}_D"
        }
        
        if suffix in map_table:
            new_state_dict[f"{prefix}.{map_table[suffix]}"] = v.clone()
        else:
            print(f"Warning: Unmapped suffix {suffix} in {k}")

    return new_state_dict

def migrate_checkpoint(input_path, output_path):
    print(f"Migrating {input_path}...")
    checkpoint = torch.load(input_path, map_location="cpu")
    
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            checkpoint["model_state"] = migrate_state_dict(checkpoint["model_state"])
        if "ema_state" in checkpoint:
            checkpoint["ema_state"] = migrate_state_dict(checkpoint["ema_state"])
        
        # We MUST invalidate optimizer because projection shapes/names changed
        if "optimizer_state" in checkpoint: del checkpoint["optimizer_state"]
        if "scheduler_state" in checkpoint: del checkpoint["scheduler_state"]
        
        if "model_state" not in checkpoint and "ema_state" not in checkpoint:
            checkpoint = migrate_state_dict(checkpoint)
    else:
        checkpoint = migrate_state_dict(checkpoint)
        
    torch.save(checkpoint, output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    # Remove previous legacy markers to avoid confusion
    for f in ["dim_llm_checkpoint.pt", "dim_llm_ema_best.pt", "dim_llm_ema_checkpoint.pt"]:
        legacy = f.replace(".pt", "_legacy.pt")
        if os.path.exists(legacy):
            os.remove(legacy)

    targets = ["dim_llm_checkpoint.pt", "dim_llm_ema_best.pt", "dim_llm_ema_checkpoint.pt"]
    for t in targets:
        if os.path.exists(t):
            legacy_name = t.replace(".pt", "_v2_legacy.pt")
            migrate_checkpoint(t, t + ".v3")
            os.rename(t, legacy_name)
            os.rename(t + ".v3", t)
            print(f"Migrated {t} to v3 (Fused BiMamba)")
