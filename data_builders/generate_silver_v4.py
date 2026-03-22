import json
import random
from typing import List, Dict, Any

def generate_reasoning_logic() -> Dict[str, Any]:
    """Generates a synthetic logic problem involving transitive height relations."""
    names = ["Alice", "Bob", "Charlie", "Diana", "Edward"]
    random.shuffle(names)
    
    # Problem: A > B, B > C. Who is tallest?
    a, b, c = names[:3]
    problem = {
        "premise_1": f"{a} is taller than {b}.",
        "premise_2": f"{b} is taller than {c}.",
        "question": f"Who is the tallest among {a}, {b}, and {c}?",
        "reasoning": f"If {a} > {b} and {b} > {c}, then {a} > {c}. Therefore, {a} is the tallest.",
        "answer": a
    }
    return problem

def format_json(data: Dict[str, Any]) -> str:
    """Formats the logic problem as a JSON structure."""
    return json.dumps(data, indent=2)

def format_python(data: Dict[str, Any]) -> str:
    """Formats the logic problem as a Python dictionary string."""
    return f"logic_problem = {{\n    'premises': ['{data['premise_1']}', '{data['premise_2']}'],\n    'query': '{data['question']}',\n    'chain_of_thought': '{data['reasoning']}',\n    'conclusion': '{data['answer']}'\n}}"

def format_bullet_list(data: Dict[str, Any]) -> str:
    """Formats the logic problem as a bulleted list."""
    return (
        f"* Premise 1: {data['premise_1']}\n"
        f"* Premise 2: {data['premise_2']}\n"
        f"* Question: {data['question']}\n"
        f"* Reasoning: {data['reasoning']}\n"
        f"* Final Answer: {data['answer']}"
    )

def format_plain_text(data: Dict[str, Any]) -> str:
    """Formats the logic problem as plain conversational text."""
    return (
        f"Consider the following: {data['premise_1']} Also, {data['premise_2']} "
        f"{data['question']} Well, following the logic: {data['reasoning']} "
        f"So the answer is {data['answer']}."
    )

def main():
    """Main execution block to generate silver-tier synthetic data for V4."""
    num_samples = 250 # Generates 1000 total sequences (250 * 4 formats)
    silver_data = []

    print(f"[*] Generating {num_samples} logic seeds...")
    
    for i in range(num_samples):
        grain = generate_reasoning_logic()
        
        # Phase 2: Apply the V4 Decoupling Law
        silver_data.append({"text": format_json(grain), "format": "json"})
        silver_data.append({"text": format_python(grain), "format": "python"})
        silver_data.append({"text": format_bullet_list(grain), "format": "bullets"})
        silver_data.append({"text": format_plain_text(grain), "format": "text"})

    output_file = "silver_v4_data.json"
    with open(output_file, "w") as f:
        json.dump(silver_data, f, indent=2)
    
    print(f"[✓] Successfully exported {len(silver_data)} formatted logic samples to {output_file}.")

if __name__ == "__main__":
    main()
