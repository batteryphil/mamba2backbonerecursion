import json
import random
import os

def generate_logic_shock():
    names = ["Alice", "Bob", "Charlie", "Dave", "Edward", "Fiona", "George", "Hannah", "Isaac", "Jasmine", 
             "Kevin", "Laura", "Mike", "Nina", "Oscar", "Penny", "Quinn", "Riley", "Steve", "Tanya"]
    
    entities_pool = names
    data = []

    # 1. The Inversion Set (500 samples): "X shorter than Y" -> "Who is tallest?"
    for _ in range(500):
        n1, n2 = random.sample(entities_pool, 2)
        
        # Multimodal Generation
        raw_text = f"Fact: {n1} is shorter than {n2}. Question: Who is the tallest between them? Reasoning: If {n1} < {n2}, then {n2} is the tallest. Answer: {n2}."
        
        json_fmt = json.dumps({
            "observation": f"{n1} is shorter than {n2}",
            "goal": "identify tallest",
            "relationship": f"{n1} < {n2}",
            "result": n2
        }, indent=2)
        
        python_fmt = f"def logic_check():\n    # {n1} shorter than {n2}\n    heights = {{'{n1}': 160, '{n2}': 180}}\n    return max(heights, key=heights.get)\n# result: {n2}"
        
        combined_text = f"--- RAW TEXT ---\n{raw_text}\n\n--- JSON DATA ---\n{json_fmt}\n\n--- PYTHON LOGIC ---\n{python_fmt}"
        data.append({"text": combined_text, "type": "logic_shock_inversion"})

    # 2. The Chain Set (500 samples): 4-5 entities complex transitive logic
    for _ in range(500):
        num_ent = random.randint(4, 5)
        current = random.sample(entities_pool, num_ent)
        # Truth Order: current[0] > current[1] > ...
        
        premises = []
        for i in range(num_ent - 1):
            if random.random() > 0.5:
                # Taller syntax
                premises.append(f"{current[i]} is taller than {current[i+1]}")
            else:
                # Shorter syntax
                premises.append(f"{current[i+1]} is shorter than {current[i]}")
        
        random.shuffle(premises)
        
        raw_text = f"Premises: {'. '.join(premises)}. Question: Who is the tallest among {', '.join(current)}? Answer: {current[0]}."
        
        json_fmt = json.dumps({
            "premises": premises,
            "chain": " > ".join(current),
            "answer": current[0]
        }, indent=2)
        
        python_fmt = f"def resolve_chain():\n    nodes = {current}\n    # Logic: " + " > ".join(current) + f"\n    return nodes[0]\n# result: {current[0]}"
        
        combined_text = f"--- RAW TEXT ---\n{raw_text}\n\n--- JSON DATA ---\n{json_fmt}\n\n--- PYTHON LOGIC ---\n{python_fmt}"
        data.append({"text": combined_text, "type": "logic_shock_chain"})

    # 3. The Negation Set (500 samples): "Alice is NOT shorter than Bob"
    for _ in range(500):
        n1, n2 = random.sample(entities_pool, 2)
        # "X is NOT shorter than Y" -> X >= Y (we assume discrete distinct heights for training simplicity, so X > Y)
        
        raw_text = f"Information: {n1} is NOT shorter than {n2}. Assuming they have different heights, who is taller? Solution: Not shorter implies taller. Answer: {n1}."
        
        json_fmt = json.dumps({
            "negation": f"NOT shorter({n1}, {n2})",
            "inference": f"taller({n1}, {n2})",
            "answer": n1
        }, indent=2)
        
        python_fmt = f"def evaluate_negation():\n    is_shorter = False  # {n1} is NOT shorter than {n2}\n    if not is_shorter:\n        return '{n1}'\n# result: {n1}"
        
        combined_text = f"--- RAW TEXT ---\n{raw_text}\n\n--- JSON DATA ---\n{json_fmt}\n\n--- PYTHON LOGIC ---\n{python_fmt}"
        data.append({"text": combined_text, "type": "logic_shock_negation"})

    # Append to silver_v4_logic.json
    target_logic = "silver_v4_logic.json"
    
    # If silver_v4_data.json exists, we incorporate its existing logic too
    master_data = []
    if os.path.exists("silver_v4_data.json"):
        with open("silver_v4_data.json", "r") as f:
            master_data = json.load(f)
            
    final_data = master_data + data
    
    with open(target_logic, "w") as f:
        json.dump(final_data, f, indent=2)
    
    print(f"✅ LOGIC SHOCK COMPLETE: 1,500 new high-entropy samples generated.")
    print(f"Total logic corpus size in {target_logic}: {len(final_data)} samples.")
    
    # Update Training Manifest (train_rbm.py)
    print("🛠️ Updating Training Manifest (train_rbm.py)...")
    with open("train_rbm.py", "r") as f:
        script_content = f.read()
    
    # Update the default data file to silver_v4_logic.json
    new_script = script_content.replace('default="silver_v4_data.json"', 'default="silver_v4_logic.json"')
    
    with open("train_rbm.py", "w") as f:
        f.write(new_script)
    
    print("🚀 Training manifest updated. RBM will target the Logic Corpus on next run.")

if __name__ == "__main__":
    generate_logic_shock()
