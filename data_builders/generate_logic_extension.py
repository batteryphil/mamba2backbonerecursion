import json
import random
import os

def generate_contrastive_logic():
    names = ["Alice", "Bob", "Charlie", "Dave", "Edward", "Fiona", "George", "Hannah", "Isaac", "Jasmine"]
    data = []

    # 1. Inversion Samples (shorter than -> Who is taller?)
    for _ in range(500):
        n1, n2 = random.sample(names, 2)
        # Format: Bullet/JSON/Text randomized
        fmt = random.choice(["bullets", "json", "text"])
        
        if fmt == "bullets":
            text = f"* Premise: {n1} is shorter than {n2}.\n* Question: Who is taller?\n* Reasoning: If {n1} < {n2}, then {n2} is taller.\n* Final Answer: {n2}"
        elif fmt == "json":
            text = json.dumps({
                "premise": f"{n1} is shorter than {n2}",
                "question": "Who is taller?",
                "reasoning": f"If {n1} < {n2}, then {n2} is taller than {n1}.",
                "answer": n2
            }, indent=2)
        else:
            text = f"Fact: {n1} is shorter than {n2}. Tell me, who is the taller of the two? Since {n1} is shorter, {n2} must be the taller person. The answer is {n2}."
            
        data.append({"text": text, "format": fmt, "type": "contrastive_inversion"})

    # 2. Multi-Step Logic (4-5 entities)
    for _ in range(500):
        num_entities = random.randint(4, 5)
        current_names = random.sample(names, num_entities)
        
        # Sort them internally to create a ground truth
        # Let's say indices in current_names define the order: 0 > 1 > 2 > 3 > 4
        # We will present premises out of order
        premises = []
        for i in range(num_entities - 1):
            # i is taller than i + 1
            if random.random() > 0.5:
                premises.append(f"{current_names[i]} is taller than {current_names[i+1]}")
            else:
                premises.append(f"{current_names[i+1]} is shorter than {current_names[i]}")
        
        random.shuffle(premises)
        
        fmt = random.choice(["bullets", "json", "text"])
        question = f"Who is the tallest among {', '.join(current_names[:-1])} and {current_names[-1]}?"
        reasoning = " -> ".join(current_names) + " (Taller to Shorter)"
        answer = current_names[0]

        if fmt == "bullets":
            text = "\n".join([f"* {p}" for p in premises]) + f"\n* Question: {question}\n* Reasoning: Sorting the entities: {reasoning}. Therefore, {answer} is at the top.\n* Final Answer: {answer}"
        elif fmt == "json":
            text = json.dumps({
                "premises": premises,
                "question": question,
                "chain": reasoning,
                "answer": answer
            }, indent=2)
        else:
            text = f"Consider these facts: {'. '.join(premises)}. Now, {question} By following the transitive property and ordering them ({reasoning}), we find that {answer} is the tallest."

        data.append({"text": text, "format": fmt, "type": "multi_step_logic"})

    # Save/Append to silver_v4_data.json
    target_file = "silver_v4_data.json"
    if os.path.exists(target_file):
        with open(target_file, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = []
        
    final_data = existing_data + data
    
    with open(target_file, "w") as f:
        json.dump(final_data, f, indent=2)
    
    print(f"🚀 Contrastive Logic Extension Complete. Added 1000 samples to {target_file}.")
    print(f"New total dataset size: {len(final_data)} samples.")

if __name__ == "__main__":
    generate_contrastive_logic()
    # Trigger training restart/refresh
    print("Dataset refreshed. Training manifold should pick up changes on next epoch shuffle.")
    # No need to restart if training is running and shuffles every epoch
