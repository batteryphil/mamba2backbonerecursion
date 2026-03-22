import torch
from mamba_ssm import MambaLMHeadModel

try:
    print("Loading Mamba-2-130m...")
    model = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-130m")
    mixer = model.backbone.layers[0].mixer
    print("Mamba 2 Mixer Attributes:")
    for name, module in mixer.named_children():
        print(f" - {name}: {module}")
        
    print("\nMamba 2 Mixer __dict__ keys:")
    for key in mixer.__dict__.keys():
        if "proj" in key:
            print(f" - {key}")
except Exception as e:
    print(f"Error: {e}")
