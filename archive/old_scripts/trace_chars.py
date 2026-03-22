
import json
with open("train_data.json", "r") as f:
    data = json.load(f)

for n in [10, 50, 100, 200, 500, 1000, 2000, 3000]:
    chars = set()
    for line in data[:min(n, len(data))]:
        for char in line:
            chars.add(char)
    # Plus special tokens (not in text mostly)
    print(f"n={n}: {len(chars)} unique chars (Raw: {chars})")
