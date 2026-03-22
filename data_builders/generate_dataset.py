import json
import random

# Synthetic Dataset for Coding & Tool Use (Small demo scale)
# This mimics the patterns of a chatbot capable of tool invocation and code generation.

def generate_tool_data(num_samples=500):
    prefixes = [
        "How can I calculate ",
        "Can you add ",
        "What is the sum of ",
        "Help me add these: "
    ]
    
    data = []
    for _ in range(num_samples):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        prefix = random.choice(prefixes)
        
        prompt = f"{prefix} {a} and {b}?"
        tool_call = f"<tool>add({a}, {b})</tool>"
        result = f"<result>{a+b}</result>"
        response = f"The result is {a+b}."
        
        # Format: [Prompt] [Tool Call] [Result] [Final Response]
        data.append(f"{prompt} | {tool_call} | {result} | {response}")
        
    return data

def generate_code_data(num_samples=500):
    snippets = [
        ("Write a print statement for ", "print('{v}')"),
        ("Generate a list of size ", "list(range({v}))"),
        ("Define a function called ", "def {v}(): pass"),
        ("How do I import ", "import {v}")
    ]
    
    values = ["hello", "world", "data", "math", "test_func", "main", "os", "sys"]
    
    data = []
    for _ in range(num_samples):
        instr, template = random.choice(snippets)
        v = random.choice(values)
        if "{v}" in template:
            code = template.format(v=v)
        else:
            code = template
            
        prompt = f"{instr} '{v}'"
        data.append(f"{prompt} | <code>{code}</code> | Output: None")
        
    return data

def generate_jarvis_data(num_samples=500):
    jarvis_examples = [
        ("How's the 3060 v2 rig doing?", "<tool name=\"terminal\" params='{\"command\": \"nvidia-smi\"}'></tool>"),
        ("Find a used P40 GPU for my server.", "<tool name=\"ebay_search\" params='{\"query\": \"Tesla P40 24GB\", \"category\": \"Graphics Card\"}'></tool>"),
        ("Analyze the coordinates 12.345, -90.123 in Maynard.", "<tool name=\"property_analysis\" params='{\"coordinates\": \"12.345, -90.123\", \"layer\": \"lidar\"}'></tool>"),
        ("Check the status of my training run.", "<tool name=\"terminal\" params='{\"command\": \"type training_stats.json\"}'></tool>"),
        ("Write a new Jarvis script to scratch.", "<tool name=\"file_io\" params='{\"action\": \"write\", \"path\": \"new_script.py\", \"content\": \"print(\\'Hello Jarvis\\')\"}'></tool>")
    ]
    
    data = []
    for _ in range(num_samples):
        instr, tool_call = random.choice(jarvis_examples)
        data.append(f"User: {instr} | Assistant: {tool_call} | <observation>OK</observation>")
        
    return data

def main():
    print("Generating synthetic coding, tool-use, and Jarvis-specific dataset...")
    tool_data = generate_tool_data(1000)
    code_data = generate_code_data(1000)
    jarvis_data = generate_jarvis_data(1000)
    
    full_dataset = tool_data + code_data + jarvis_data
    random.shuffle(full_dataset)
    
    # Split
    split_idx = int(0.9 * len(full_dataset))
    train_data = full_dataset[:split_idx]
    val_data = full_dataset[split_idx:]
    
    with open("train_data.json", "w") as f:
        json.dump(train_data, f)
    with open("val_data.json", "w") as f:
        json.dump(val_data, f)
        
    print(f"Dataset ready. Train size: {len(train_data)}, Val size: {len(val_data)}")

if __name__ == "__main__":
    main()
