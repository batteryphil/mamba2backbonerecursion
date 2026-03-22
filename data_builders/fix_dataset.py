import json
import os

def clean_file(filename):
    if not os.path.exists(filename): return
    print(f"Cleaning {filename}...")
    with open(filename, 'r') as f:
        data = json.load(f)
    
    cleaned = 0
    for i in range(len(data)):
        if "####" in data[i]:
            data[i] = data[i].replace("####", "\\boxed{}")
            cleaned += 1
        if '{"lyrics": \\"}' in data[i]:
            data[i] = data[i].replace('{"lyrics": \\"}', '{"lyrics": ""}')
            cleaned += 1
            
    with open(filename, 'w') as f:
        json.dump(data, f)
        
    print(f"Fixed {cleaned} issues in {filename}")

clean_file("train_data.json")
clean_file("val_data.json")
