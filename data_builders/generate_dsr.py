import json
import random

def generate_cot_data(num_samples=2000):
    data = []
    
    # 1. Simple Math
    for _ in range(num_samples // 3):
        a = random.randint(1, 50)
        b = random.randint(1, 50)
        op = random.choice(["+", "-", "*"])
        if op == "+":
            ans = a + b
            text = f"User: What is {a} + {b}? | Assistant: Let's think step by step. First, we take the starting number, which is {a}. Then, we add {b} to it. {a} plus {b} equals {ans}. Therefore, the result is {ans}."
        elif op == "-":
            ans = a - b
            text = f"User: Calculate {a} - {b}. | Assistant: Let's think step by step. We start with {a}. We need to subtract {b}. {a} minus {b} is {ans}. Therefore, the result is {ans}."
        elif op == "*":
            ans = a * b
            text = f"User: Multiply {a} and {b}. | Assistant: Let's think step by step. We have {a} groups of {b}. {a} multiplied by {b} equals {ans}. Therefore, the result is {ans}."
        data.append(text)
        
    # 2. Word Problems
    items = ["apples", "books", "chairs", "coins", "cars"]
    names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"]
    for _ in range(num_samples // 3):
        name = random.choice(names)
        item = random.choice(items)
        start = random.randint(5, 50)
        give = random.randint(1, start - 1)
        ans = start - give
        text = f"User: {name} has {start} {item}. They give away {give} {item}. How many do they have left? | Assistant: Let's think step by step. First, we note the starting amount: {name} begins with {start} {item}. Then, {name} gives away {give} {item}, which is a subtraction. We calculate {start} - {give} = {ans}. Therefore, the result is {ans}."
        data.append(text)
        
    # 3. Logic/Reasoning
    colors = ["red", "blue", "green", "yellow", "purple"]
    for _ in range(num_samples // 3 + 1):
        c1 = random.choice(colors)
        c2 = random.choice([c for c in colors if c != c1])
        text = f"User: I have a {c1} ball and a {c2} box. If I put the ball in the box, what color is the object inside the box? | Assistant: Let's think step by step. The question asks for the color of the object inside the box. The object being placed inside is the ball. The ball's color is {c1}. The box itself is {c2}, but the item inside is the ball. Therefore, the result is {c1}."
        data.append(text)
        
    random.shuffle(data)
    return data[:num_samples]

if __name__ == "__main__":
    dsr_data = generate_cot_data(2000)
    with open("synthetic_dsr_data.json", "w") as f:
        json.dump(dsr_data, f, indent=2)
    print(f"Generated {len(dsr_data)} DSR examples in synthetic_dsr_data.json")
