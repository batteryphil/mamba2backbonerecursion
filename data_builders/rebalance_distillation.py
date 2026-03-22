import json

# We keep the same gold samples, but we lower the intensity
gold_samples = [
    {"prompt": "Hello there, Jarvis!", "completion": "Assistant: Hello! I am ready to assist with your technical queries. [Step: 32 | Mode: FACT]"},
    {"prompt": "Find an RTX 3060 on eBay for under $250.", "completion": "<tool>search_ebay(query='RTX 3060', max_price=250)</tool>"},
    {"prompt": "Calculate the torque for a 6.7 Powerstroke cam bolt.", "completion": "The torque spec is 24 lb-ft (33 Nm). \\boxed{24}"}
]

def generate_balanced_distillation(samples, output_file="balanced_distillation.json"):
    distill_data = []
    for s in samples:
        # The 1.25x Anchor - firm but not overwhelming
        distill_data.append({
            "text": f"User: {s['prompt']} | {s['completion']}",
            "loss_weight": 1.25
        })
    
    with open(output_file, "w") as f:
        json.dump(distill_data, f, indent=4)
    print(f"[✓] Balanced Distillation Set created: {output_file}")

if __name__ == "__main__":
    generate_balanced_distillation(gold_samples)
