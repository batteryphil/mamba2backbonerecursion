
import json
import random

def generate_tool_data(num_samples=500):
    prefixes = ["How can I calculate ", "Can you add ", "What is the sum of ", "Help me add these: "]
    data = []
    for _ in range(num_samples):
        a = random.randint(1, 100); b = random.randint(1, 100); prefix = random.choice(prefixes)
        prompt = f"{prefix} {a} and {b}?"; tool_call = f"<tool>add({a}, {b})</tool>"; result = f"<result>{a+b}</result>"; response = f"The result is {a+b}."
        data.append(f"{prompt} | {tool_call} | {result} | {response}")
    return data

def generate_code_data(num_samples=500):
    snippets = [("Write a print statement for ", "print('{v}')"), ("Generate a list of size ", "list(range({v}))"), ("Define a function called ", "def {v}(): pass"), ("How do I import ", "import {v}")]
    values = ["hello", "world", "data", "math", "test_func", "main", "os", "sys"]
    data = []
    for _ in range(num_samples):
        instr, template = random.choice(snippets); v = random.choice(values)
        code = template.format(v=v); prompt = f"{instr} '{v}'"
        data.append(f"{prompt} | <code>{code}</code> | Output: None")
    return data

td = generate_tool_data(500)
cd = generate_code_data(500)
chars = set()
for line in td + cd:
    for char in line: chars.add(char)
print(f"Unique chars in Tool+Code: {len(chars)}")
print(sorted(list(chars)))
