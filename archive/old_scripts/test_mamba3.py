from transformers import AutoModelForCausalLM
from mamba_ssm import MambaLMHeadModel

models_to_try = [
    "state-spaces/mamba3-130m",
    "state-spaces/mamba-3-130m",
    "state-spaces/mamba-3",
    "togethercomputer/mamba-3-130m",
    "togethercomputer/mamba3-130m"
]

for model_name in models_to_try:
    try:
        print(f"Trying {model_name}...")
        MambaLMHeadModel.from_pretrained(model_name)
        print(f"✅ SUCCESS: {model_name}")
        break
    except Exception as e:
        print(f"❌ FAILED: {model_name} - {str(e)[:100]}")
