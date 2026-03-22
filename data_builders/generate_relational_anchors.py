import json
import random

def generate_transitive():
    names = random.sample(["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Hank"], 3)
    traits = [
        ("taller", "shorter"), ("older", "younger"), 
        ("faster", "slower"), ("heavier", "lighter")
    ]
    trait, anti_trait = random.choice(traits)
    
    # A > B and B > C. 
    premise = f"{names[0]} is {trait} than {names[1]}. {names[1]} is {trait} than {names[2]}."
    
    # Ask about the extremes
    if random.choice([True, False]):
        question = f"Who is the {trait}?"
        answer = names[0]
    else:
        question = f"Who is the {anti_trait}?"
        answer = names[2]
        
    return f"Premise: {premise} Question: {question} Answer: {answer}"

def generate_spatial():
    objects = random.sample(["keys", "coin", "ring", "watch", "pen", "marble"], 1)
    all_containers = ["cup", "box", "drawer", "safe", "jar", "pocket", "bag"]
    containers = random.sample(all_containers, 2)
    
    obj = objects[0]
    c1 = containers[0]
    c2 = containers[1]
    
    premise = f"The {obj} is inside the {c1}. The {c1} is placed inside the {c2}."
    
    if random.choice([True, False]):
        question = f"Is the {obj} inside the {c2}?"
        answer = "Yes."
    else:
        # Generate a negative case
        c3 = random.choice([c for c in all_containers if c not in [c1, c2]])
        question = f"Is the {obj} inside the {c3}?"
        answer = "No."
        
    return f"Premise: {premise} Question: {question} Answer: {answer}"

def generate_substitution():
    elements = [("fire", "freezes"), ("ice", "boils"), ("water", "burns"), ("wind", "crushes")]
    rule_pair = random.choice(elements)
    
    element, action = rule_pair
    
    premise = f"In the mirrored realm, {element} {action} things."
    
    if element == "fire":
        question = "You place a steak in the fire. Does it freeze or cook?"
        answer = "It freezes."
    elif element == "ice":
        question = "You put a pot of soup on the ice. Does it boil or chill?"
        answer = "It boils."
    elif element == "water":
        question = "You splash water on the paper. Does it burn or get wet?"
        answer = "It burns."
    else: # wind
        question = "The wind hits the glass window. Does it crush it or blow past?"
        answer = "It crushes it."
        
    return f"Premise: {premise} Question: {question} Answer: {answer}"

def generate_dataset(num_samples=500, filename="relational_anchors.jsonl"):
    generators = [generate_transitive, generate_spatial, generate_substitution]
    
    with open(filename, 'w') as f:
        for _ in range(num_samples):
            # Randomly select one of the logic topologies
            gen_func = random.choice(generators)
            text = gen_func()
            
            # Format as standard causal LM text
            json_line = json.dumps({"text": text})
            f.write(json_line + '\n')
            
    print(f"✅ Generated {num_samples} relational anchors and saved to {filename}")

if __name__ == "__main__":
    generate_dataset(500)
