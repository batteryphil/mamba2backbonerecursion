import json
import random
import os

def generate_logic_v3():
    # 🚀 MASSIVE NAME POOL (1,000 unique identifiers)
    # We combine standard names with Greek letters and NATO phonetics to hit 1k+
    base_names = [
        "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda", "David", "Elizabeth",
        "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Christopher", "Karen",
        "Charles", "Lisa", "Daniel", "Nancy", "Matthew", "Betty", "Anthony", "Sandra", "Mark", "Margaret",
        "Donald", "Ashley", "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
        "Kenneth", "Dorothy", "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa", "Timothy", "Deborah",
        "Ronald", "Stephanie", "Edward", "Rebecca", "Jason", "Sharon", "Jeffrey", "Laura", "Ryan", "Cynthia"
    ]
    
    nato = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India", "Juliett", 
            "Kilo", "Lima", "Mike", "November", "Oscar", "Papa", "Quebec", "Romeo", "Sierra", "Tango", 
            "Uniform", "Victor", "Whiskey", "Xray", "Yankee", "Zulu"]
    
    greek = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa", 
             "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi", "Rho", "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega"]

    full_names = []
    for bn in base_names:
        for n in nato:
            full_names.append(f"{bn}_{n}")
    
    # Trim to exactly 1,000 for consistency
    entities = full_names[:1000]
    random.shuffle(entities)
    
    data = []

    def get_fmt(text, json_data, python_logic, tag):
        # Protocol v5.0 uses a strict multi-modal triple
        combined = f"### LOGIC MANIFOLD v5.0\n--- RAW TEXT ---\n{text}\n\n--- JSON DATA ---\n{json_data}\n\n--- PYTHON LOGIC ---\n{python_logic}"
        return {"text": combined, "type": tag}

    # 1. High-Entropy Inversions (5,000 samples)
    for _ in range(5000):
        n1, n2 = random.sample(entities, 2)
        noise = random.choice(["the sun is bright", "the cat is sleeping", "it's raining outside", "the gears are turning"])
        
        raw_text = f"Condition: {noise}. Premise: {n1} is shorter than {n2}. Target Inquiry: Who is the tallest? Analysis: Since {n1} < {n2}, {n2} must be the taller entity. Conclusion: {n2}."
        
        json_fmt = json.dumps({"manifold": "TransitiveInversion", "logic": f"{n1} < {n2}", "noise": noise, "target": "Tallest", "result": n2})
        python_fmt = f"def resolve_logic():\n    e = {{'{n1}': 0, '{n2}': 1}}\n    return max(e, key=e.get)\n# -> {n2}"
        
        data.append(get_fmt(raw_text, json_fmt, python_fmt, "logic_v3_inversion"))

    # 2. Deep Transitive Chains (5,000 samples)
    for _ in range(5000):
        num_ent = random.randint(4, 7) # Upping depth to 7 entities
        current = random.sample(entities, num_ent)
        # Truth: current[0] > current[1] > ...
        
        premises = []
        for i in range(num_ent - 1):
            if random.random() > 0.5:
                premises.append(f"{current[i]} is taller than {current[i+1]}")
            else:
                premises.append(f"{current[i+1]} is shorter than {current[i]}")
        
        random.shuffle(premises)
        
        raw_text = f"Observations: [{', '.join(premises)}]. Systematic Query: Rank the tallest individual. Answer: {current[0]}."
        
        json_fmt = json.dumps({"manifold": "DeepChain", "chain": " > ".join(current), "entities": current, "result": current[0]})
        python_fmt = f"def sort_manifold():\n    h_map = {list(reversed(current))}\n    return h_map[-1]\n# -> {current[0]}"
        
        data.append(get_fmt(raw_text, json_fmt, python_fmt, "logic_v3_chain"))

    # 3. Negated Conditionals (5,000 samples)
    for _ in range(5000):
        n1, n2 = random.sample(entities, 2)
        raw_text = f"Logic State: NOT({n1} taller than {n2}). Distinct Heights assumed. Identification: Who has the greater height? Resolution: {n2}."
        
        json_fmt = json.dumps({"manifold": "Negation", "input": f"!({n1} > {n2})", "inference": f"{n2} > {n1}", "result": n2})
        python_fmt = f"def logic_negation():\n    if not ({n1}_height > {n2}_height):\n        return '{n2}'\n# -> {n2}"
        
        data.append(get_fmt(raw_text, json_fmt, python_fmt, "logic_v3_negation"))

    # Save to logic_v3.json
    output_file = "logic_v3.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"💎 LOGIC MANIFOLD v5.0 GENERATED: 15,000 high-entropy samples.")
    print(f"Source Pool: 1,000 unique entity IDs.")
    print(f"Training File Size: {os.path.getsize(output_file) // (1024*1024)} MB")

if __name__ == "__main__":
    generate_logic_v3()
