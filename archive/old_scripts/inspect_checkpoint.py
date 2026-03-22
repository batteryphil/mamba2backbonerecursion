import torch
import os

path = "/home/phil/Desktop/mambadiff/mambadiff llm tts/dim_llm_epoch002.pt"
if os.path.exists(path):
    print(f"Inspecting {path}...")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        print(f"Keys: {state.keys()}")
        if "epoch" in state: print(f"Epoch: {state['epoch']}")
        if "step" in state: print(f"Step: {state['step']}")
    else:
        print("Not a dictionary.")
else:
    print(f"File not found: {path}")
