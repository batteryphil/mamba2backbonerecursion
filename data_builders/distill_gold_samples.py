import json

# The "Gold" samples from your successful benchmark
gold_samples = [
    {"text": "[Greeting] User: Hello there, Jarvis! | Assistant: Hello! I am ready to assist with your technical queries. [Step: 32 | Mode: FACT]"},
    {"text": "[Tool Call] User: Find an RTX 3060 on eBay for under $250. | Assistant: <tool>search_ebay(query='RTX 3060', max_price=250)</tool>"},
    {"text": "[Reasoning] User: Calculate the torque for a 6.7 Powerstroke cam bolt. | Assistant: The torque spec is 24 lb-ft (33 Nm). \\boxed{24}"}
]

def generate_distillation_set(samples, output_file="gold_distillation.json"):
    distill_data = [s["text"] for s in samples]
    
    with open(output_file, 'w') as f:
        json.dump(distill_data, f, indent=4)
    print(f"✅ Created {output_file} with Gold logic formatting.")

if __name__ == "__main__":
    generate_distillation_set(gold_samples)
