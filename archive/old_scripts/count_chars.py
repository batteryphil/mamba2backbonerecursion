
import json
with open("train_data.json", "r") as f:
    data = json.load(f)
chars = set()
for line in data:
    for char in line:
        chars.add(char)
print(f"Total unique chars: {len(chars)}")
# Also print common chars to see if they are what we expect
print(sorted(list(chars)))
