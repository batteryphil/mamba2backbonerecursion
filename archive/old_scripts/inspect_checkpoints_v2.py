import torch
import os

files = ["latest_checkpoint.pt", "panic_checkpoint.pt", "rbm_hybrid_epoch_1.pt"]

for f in files:
    path = f"/home/phil/Desktop/mambadiff/mambadiff llm tts/{f}"
    if os.path.exists(path):
        print(f"\n--- Inspecting {f} ---")
        try:
            # We use map_location="cpu" to avoid GPU OOM during inspection
            state = torch.load(path, map_location="cpu")
            if isinstance(state, dict):
                print(f"Keys: {list(state.keys())}")
                if "step" in state: print(f"Step: {state['step']}")
                if "epoch" in state: print(f"Epoch: {state['epoch']}")
                if "model_state" in state:
                    num_params = sum(p.numel() for p in state["model_state"].values())
                    print(f"Model parameters: {num_params:,}")
                if "optimizer_state" in state:
                    print("Optimizer state found.")
            else:
                print("Not a dictionary.")
        except Exception as e:
            print(f"Error loading {f}: {e}")
    else:
        print(f"File not found: {f}")
