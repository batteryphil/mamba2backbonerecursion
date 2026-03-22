import json
import random
from transformers import GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm

# Initialize Tokenizer to enforce RTX 3060 VRAM limits
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"mask_token": "[MASK]"})
MAX_TOKENS = 250 # Leave room for formatting

dataset_mixture = []

def add_to_mixture(prompt, response):
    """Formats and checks token length before adding to the global dataset."""
    formatted_str = f"User: {prompt} | Assistant: {response}"
    tokens = tokenizer.encode(formatted_str, add_special_tokens=False)
    if len(tokens) <= MAX_TOKENS:
        dataset_mixture.append(formatted_str)

print("🚀 Booting Data Harvester for DiM-LLM v3...")

# 1. Harvest OpenHermes 2.5 (General Benchmark Intelligence)
print("📥 Downloading OpenHermes 2.5 subset...")
hermes = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
hermes_iter = iter(hermes)
hermes_count = 0
while hermes_count < 15000:
    try:
        row = next(hermes_iter)
        # Only grab standard Q&A (2 turns)
        if len(row['conversations']) >= 2 and row['conversations'][0]['from'] == 'human':
            prompt = row['conversations'][0]['value'].replace('\n', ' ').strip()
            response = row['conversations'][1]['value'].replace('\n', ' ').strip()
            add_to_mixture(prompt, response)
            hermes_count += 1
    except StopIteration:
        break

# 2. Harvest SmolTalk (Math & Coding Benchmarks)
print("📥 Downloading SmolTalk (metamathqa-50k subset)...")
smol = load_dataset("HuggingFaceTB/smoltalk", "metamathqa-50k", split="train", streaming=True)
smol_iter = iter(smol)
smol_count = 0
while smol_count < 10000:
    try:
        row = next(smol_iter)
        if len(row['messages']) >= 2:
            prompt = row['messages'][0]['content'].replace('\n', ' ').strip()
            response = row['messages'][1]['content'].replace('\n', ' ').strip()
            add_to_mixture(prompt, response)
            smol_count += 1
    except StopIteration:
        break

# 3. Harvest Glaive (Agentic / Tool Calling Benchmarks)
print("📥 Downloading Glaive Function Calling v2...")
glaive = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)
glaive_iter = iter(glaive)
glaive_count = 0
while glaive_count < 10000:
    try:
        row = next(glaive_iter)
        # Glaive v2 uses a specific system prompt for tools
        if "USER:" in row['chat'] and "ASSISTANT:" in row['chat']:
            # Simplified parsing for the prototype
            parts = row['chat'].split("ASSISTANT:")
            prompt = parts[0].replace("USER:", "").replace('\n', ' ').strip()
            response = parts[1].replace('\n', ' ').strip()
            # Convert <call:name>{params}</call> to Jarvis XML if needed, 
            # but Glaive often already has readable formats or we can just use as is.
            # Jarvis likes <tool name="..." params='{}'></tool>
            if "<call:" in response:
                # Basic regex-like transform
                import re
                match = re.search(r"<call:(\w+)>(.*)</call>", response)
                if match:
                    t_name = match.group(1)
                    t_params = match.group(2)
                    response = f"<tool name=\"{t_name}\" params='{t_params}'></tool>"
            add_to_mixture(prompt, response)
            glaive_count += 1
    except StopIteration:
        break

print(f"✅ Harvester Complete. Total valid 3060-safe sequences: {len(dataset_mixture)}")

# Shuffle to prevent catastrophic forgetting of specific datasets
random.shuffle(dataset_mixture)

# Split 90/10 for Train and Validation
split_idx = int(0.9 * len(dataset_mixture))
train_data = dataset_mixture[:split_idx]
val_data = dataset_mixture[split_idx:]

with open("train_data.json", "w") as f:
    json.dump(train_data, f)
with open("val_data.json", "w") as f:
    json.dump(val_data, f)

print(f"💾 Saved {len(train_data)} examples to train_data.json")
print(f"💾 Saved {len(val_data)} examples to val_data.json")
