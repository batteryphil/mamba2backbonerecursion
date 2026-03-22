import json
import random
import os

def generate_qa_dataset(num_samples=10000, output_file="qa_anchors.json"):
    """
    Generates a synthetic dataset for Instruction Tuning: Context Extraction.
    Creates paragraphs of random noise and buries a specific fact inside it,
    followed by a direct Question -> Answer pair.
    """
    
    # Vocabularies for the "Needles"
    entities = ["Alice", "Bob", "Charlie", "Dave", "Eve", "Frank", "Grace", "Heidi", 
                "Ivan", "Judy", "Mallory", "Nina", "Oscar", "Peggy", "Victor",
                "Titanium", "Gold", "Silver", "Platinum", "Lead", "Tungsten",
                "Mars", "Venus", "Saturn", "Jupiter", "Neptune", "Uranus",
                "Square", "Circle", "Triangle", "Pyramid", "Cube", "Sphere"]
                
    properties = ["the CEO", "a spy", "a doctor", "a lawyer", "an engineer", 
                  "the heaviest metal", "the lightest metal", "a toxic substance",
                  "the hottest planet", "a gas giant", "a dwarf planet",
                  "a 2D shape", "a 3D shape", "a polygon", "a polyhedron",
                  "the secret password", "the launch code", "the security clearance"]
                  
    # Distractor text templates
    distractor_starts = [
        "The committee met on Tuesday to discuss the ongoing budget constraints.",
        "Recent advancements in synthetic biology have revolutionized the field.",
        "During the medieval period, knights would often participate in jousting tournaments.",
        "The local library announced expanded hours to accommodate students studying for final exams.",
        "Global supply chains continue to face unprecedented disruption due to severe weather.",
        "Many species of deep-sea fish have evolved bioluminescent properties to attract prey.",
        "The architecture of the new museum blends modern glass structures with classic brick.",
        "Quantum computing theories suggest that a state can exist in multiple superpositions simultaneously.",
        "The ancient ruins were finally excavated after decades of bureaucratic delays."
    ]
    
    distractor_ends = [
        "Furthermore, analysts predict a significant market correction by the end of the fiscal year.",
        "It is imperative that all outstanding invoices are processed before the quarterly audit.",
        "The indigenous population relied heavily on seasonal migration patterns for survival.",
        "Therefore, the hypothesis must be re-evaluated under stricter laboratory conditions.",
        "In conclusion, the socio-economic impacts of this policy will ripple for decades.",
        "The spacecraft successfully established a stable orbit around the moon.",
        "All personnel must attend the mandatory cybersecurity training session on Friday.",
        "The painting is universally regarded as a masterpiece of the Renaissance era.",
        "He decided to take the scenic route home, enjoying the sunset over the valley."
    ]

    dataset = []

    print(f"Generating {num_samples} Instruction/Extraction Logic Samples...")
    
    for _ in range(num_samples):
        # Pick the needle
        e = random.choice(entities)
        p = random.choice(properties)
        
        # Format the needle fact
        fact = f"{e} is {p}."
        
        # Build the Haystack
        num_start = random.randint(1, 4)
        num_end = random.randint(1, 4)
        
        s_distractors = random.sample(distractor_starts, num_start)
        e_distractors = random.sample(distractor_ends, num_end)
        
        context = " ".join(s_distractors) + " " + fact + " " + " ".join(e_distractors)
        
        # Some variation in the prompt
        q_variations = [
            f"Question: Who is {p}?",
            f"Question: What is {e}?",
            f"Systematic Query: Identify {p}.",
            f"Task: Extract the identity of {p}."
        ]
        
        q = random.choice(q_variations)
        
        if "Who is" in q or "Identify" in q or "identity" in q:
            a = e
        else:
            a = p
            
        # The exact string format expected by the model's loss calculation
        full_text = f"Context: [{context}] {q} Answer: {a}."
        
        # Store for serialization
        json_obj = {
            "manifold": "ContextExtractionV1",
            "entity": e,
            "property": p,
            "context": context,
            "question": q,
            "answer": a
        }
        
        dataset.append({
            "text": full_text,
            "type": "qa_instruction",
            "json": json_obj
        })

    # Save to disk
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved to {output_file}. File Size: {os.path.getsize(output_file)/(1024*1024):.2f} MB")

if __name__ == "__main__":
    generate_qa_dataset(num_samples=10000)
