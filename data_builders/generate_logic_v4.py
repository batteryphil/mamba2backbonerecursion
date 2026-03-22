import json
import random
import os

def generate_logic_v4():
    print("🚀 Initializing Logic V4 Generator (Deep Recursion Expanded Vocabulary)...")
    
    # --- New Domain Vocabularies ---
    metals = ["Lead", "Gold", "Silver", "Iron", "Copper", "Platinum", "Titanium", "Zinc", "Nickel", "Cobalt", "Tungsten", "Mercury", "Brass", "Bronze", "Steel"]
    planets = ["Mars", "Jupiter", "Venus", "Saturn", "Neptune", "Uranus", "Mercury", "Earth", "Pluto", "Ceres", "Eris", "Makemake", "Haumea", "Titan", "Europa"]
    objects = ["Cube", "Sphere", "Pyramid", "Cylinder", "Prism", "Cone", "Torus", "Rectangle", "Square", "Circle", "Triangle", "Hexagon", "Octagon", "Polygon", "Rhombus"]
    names = ["Alice", "Bob", "Charlie", "Dave", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy", "Mallory", "Nina", "Oscar", "Peggy", "Victor"]
    
    # --- Relational Properties ---
    properties = {
        "metals": ("heavier than", "lighter than", "Heaviest", "Lightest"),
        "planets": ("hotter than", "colder than", "Hottest", "Coldest"),
        "objects": ("larger than", "smaller than", "Largest", "Smallest"),
        "names": ("older than", "younger than", "Oldest", "Youngest")
    }
    
    domains = {
        "metals": metals,
        "planets": planets,
        "objects": objects,
        "names": names
    }

    data = []

    def format_logic_prompt(raw_text, json_fmt, tag):
        # We drop the python_logic block to make the prompt lighter and focus strictly on RAW TEXT -> JSON reasoning
        combined = f"### LOGIC MANIFOLD v6.0\n--- RAW TEXT ---\n{raw_text}\n\n--- JSON DATA ---\n{json_fmt}"
        return {"text": combined, "type": tag}

    # Generate 10,000 multi-variable chain questions
    print("Generating 10,000 deep-chain relational problems (4-5 variables)...")
    for _ in range(10000):
        # 1. Pick a domain
        domain_key = random.choice(list(domains.keys()))
        pool = domains[domain_key]
        rel_greater, rel_lesser, target_max, target_min = properties[domain_key]
        
        # 2. Pick 4 to 5 unique entities
        num_ent = random.randint(4, 5)
        current = random.sample(pool, num_ent)
        
        # Absolute ground truth order: current[0] > current[1] > current[2] > current[3] ...
        # (e.g. 0 is heavier than 1, 1 is heavier than 2...)
        
        premises = []
        for i in range(num_ent - 1):
            if random.random() > 0.5:
                # Greater relationship
                premises.append(f"{current[i]} is {rel_greater} {current[i+1]}")
            else:
                # Lesser relationship
                premises.append(f"{current[i+1]} is {rel_lesser} {current[i]}")
                
        # Shuffle premises to obfuscate the geometric pattern
        random.shuffle(premises)
        
        # 3. Determine target query (Max or Min)
        find_max = random.random() > 0.5
        target_word = target_max if find_max else target_min
        correct_answer = current[0] if find_max else current[-1]
        
        # Build the Text
        raw_text = f"Observations: [{', '.join(premises)}]. Systematic Query: Identify the {target_word} entity. Answer: {correct_answer}."
        
        # Build the geometric JSON structure expected by the N=3 reasoning circuit
        json_chain = f" > ".join(current) if find_max else f" < ".join(reversed(current))
        json_obj = {
            "manifold": "DeepChainV4",
            "domain": domain_key.upper(),
            "chain": json_chain,
            "target": target_word,
            "result": correct_answer
        }
        json_fmt = json.dumps(json_obj)
        
        data.append(format_logic_prompt(raw_text, json_fmt, f"logic_v4_{domain_key}"))

    # Generate 5,000 Direct Inversions (A < B, Who is A>B?)
    print("Generating 5,000 direct logical inversions...")
    for _ in range(5000):
        domain_key = random.choice(list(domains.keys()))
        pool = domains[domain_key]
        rel_greater, rel_lesser, target_max, target_min = properties[domain_key]
        
        e1, e2 = random.sample(pool, 2)
        
        # Truth: e1 > e2
        if random.random() > 0.5:
            # Show lesser, ask for max
            premise = f"{e2} is {rel_lesser} {e1}"
            raw_text = f"Premise: {premise}. Target Inquiry: Which is {target_max}? Analysis: Since {e2} is {rel_lesser}, {e1} must be the {target_max}. Conclusion: {e1}."
            json_obj = {"manifold": "Inversion", "logic": f"{e2} < {e1}", "result": e1}
        else:
            # Show greater, ask for min
            premise = f"{e1} is {rel_greater} {e2}"
            raw_text = f"Premise: {premise}. Target Inquiry: Which is {target_min}? Analysis: Since {e1} is {rel_greater}, {e2} must be the {target_min}. Conclusion: {e2}."
            json_obj = {"manifold": "Inversion", "logic": f"{e1} > {e2}", "result": e2}
            
        json_fmt = json.dumps(json_obj)
        data.append(format_logic_prompt(raw_text, json_fmt, f"logic_v4_inversion"))

    # Save to logic_v4.json
    output_file = "logic_v4.json"
    print(f"Shuffling and saving {len(data)} logical architectures to {output_file}...")
    random.shuffle(data)
    
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ V4 Dataset Complete! File Size: {os.path.getsize(output_file) // (1024*1024)} MB")

if __name__ == "__main__":
    generate_logic_v4()
