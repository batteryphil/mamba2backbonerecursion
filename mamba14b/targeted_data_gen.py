#!/usr/bin/env python3
"""
targeted_data_gen.py — v3 HYBRID
Key insight from R4 evaluation:
  - CoT answers HELP reasoning (probe went from 20% to 60%)
  - CoT answers HURT code benchmark (model narrates instead of coding)
Strategy:
  - Math / word problems / bug diagnosis → CoT step-by-step answers
  - Code generation / bug fixes → DIRECT code answers (no preamble)
  - Comprehensions / decorators → direct code answers
This hybrid should get R3 benchmark accuracy (76%) AND R4 reasoning (60%+).
"""

import json
import math
import random
from pathlib import Path

OUT = Path("/home/phil/.gemini/antigravity/scratch/tiny-refinement/targeted_samples.jsonl")
random.seed(42)
samples = []


def add(prompt: str, answer: str) -> None:
    """Append one training sample."""
    samples.append({"prompt": prompt.strip(), "answer": answer.strip()})


# ===========================================================================
# 1. ARITHMETIC — CoT for math, direct for code-adjacent
# ===========================================================================
print("Generating arithmetic CoT samples...")

for _ in range(600):
    a, b = random.randint(2, 50), random.randint(2, 50)
    r = a * b
    add(f"What is {a} multiplied by {b}?",
        f"Step 1: Multiply {a} by {b}.\nStep 2: {a} × {b} = {r}\nAnswer: {r}")
    add(f"Calculate {a} × {b}.",
        f"Step 1: {a} × {b}\nStep 2: = {r}\nAnswer: {r}")

for _ in range(300):
    b = random.randint(2, 20)
    a = b * random.randint(2, 30)
    r = a // b
    add(f"What is {a} divided by {b}?",
        f"Step 1: Divide {a} by {b}.\nStep 2: {a} ÷ {b} = {r}\nAnswer: {r}")

for _ in range(300):
    a, b = random.randint(10, 999), random.randint(10, 999)
    add(f"What is {a} plus {b}?",
        f"Step 1: Add {a} and {b}.\nStep 2: {a} + {b} = {a+b}\nAnswer: {a+b}")
    add(f"What is {a} minus {b}?",
        f"Step 1: Subtract {b} from {a}.\nStep 2: {a} - {b} = {a-b}\nAnswer: {a-b}")

# Square roots — CoT
for n in [4,9,16,25,36,49,64,81,100,121,144,169,196,225,256,
          289,324,361,400,441,484,529,576,625,676,729,784,841,900]:
    root = int(math.sqrt(n))
    for _ in range(10):
        add(f"What is the square root of {n}?",
            f"Step 1: Find r such that r × r = {n}.\n"
            f"Step 2: {root} × {root} = {n}\nAnswer: {root}")
        add(f"Calculate √{n}.",
            f"Step 1: √{n} = ?\nStep 2: {root} × {root} = {n}\nAnswer: {root}")

# Powers — CoT
for base in range(2, 12):
    for exp in range(2, 5):
        result = base ** exp
        steps  = " × ".join([str(base)] * exp)
        add(f"What is {base} to the power of {exp}?",
            f"Step 1: {base}^{exp} = {steps}\nStep 2: = {result}\nAnswer: {result}")

# ===========================================================================
# 2. PERCENTAGES — CoT
# ===========================================================================
print("Generating percentage CoT samples...")

for _ in range(500):
    pct  = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80])
    base = random.randint(2, 40) * 10
    result = int(base * pct / 100)
    add(f"What is {pct}% of {base}?",
        f"Step 1: Convert {pct}% to decimal: {pct} ÷ 100 = {pct/100}\n"
        f"Step 2: Multiply: {base} × {pct/100} = {result}\nAnswer: {result}")
    add(f"Calculate {pct} percent of {base}.",
        f"Step 1: ({pct} / 100) × {base}\nStep 2: = {pct/100} × {base} = {result}\nAnswer: {result}")

# ===========================================================================
# 3. WORD PROBLEMS — CoT
# ===========================================================================
print("Generating word problem CoT samples...")

for _ in range(400):
    speed = random.choice([30, 40, 50, 60, 70, 80, 90, 100])
    hours = random.choice([1, 1.5, 2, 2.5, 3, 4, 5])
    dist  = speed * hours
    add(f"A car travels at {speed} mph for {hours} hours. How many miles does it travel?",
        f"Step 1: Formula: Distance = Speed × Time\n"
        f"Step 2: Distance = {speed} × {hours} = {dist:.0f} miles\nAnswer: {dist:.0f} miles")

for _ in range(300):
    pa, pb = random.choice([25,50,75,100,150,200]), random.choice([10,20,30,40,50])
    qa, qb = random.randint(1,10), random.randint(1,10)
    total  = pa*qa + pb*qb
    add(f"A store sells item A for ${pa} and item B for ${pb}. "
        f"If you buy {qa} of item A and {qb} of item B, what is the total cost?",
        f"Step 1: Cost of item A: {qa} × ${pa} = ${qa*pa}\n"
        f"Step 2: Cost of item B: {qb} × ${pb} = ${qb*pb}\n"
        f"Step 3: Total = ${qa*pa} + ${qb*pb} = ${total}\nAnswer: ${total}")

for _ in range(200):
    principal = random.choice([100,200,500,1000,2000])
    rate      = random.choice([5,10,15,20])
    years     = random.choice([1,2,3,5])
    interest  = int(principal * rate * years / 100)
    add(f"Calculate simple interest on ${principal} at {rate}% per year for {years} year(s).",
        f"Step 1: Interest = (Principal × Rate × Time) / 100\n"
        f"Step 2: = ({principal} × {rate} × {years}) / 100 = {principal*rate*years} / 100\n"
        f"Step 3: = ${interest}\nAnswer: ${interest}")

for _ in range(300):
    total = random.choice([20,25,30,40,50])
    pct   = random.choice([20,25,40,50,60,75,80])
    ga    = int(total * pct / 100)
    gb    = total - ga
    add(f"A class has {total} students. {pct}% are girls. How many boys are in the class?",
        f"Step 1: Girls = {pct}% of {total} = {pct}/100 × {total} = {ga}\n"
        f"Step 2: Boys = {total} - {ga} = {gb}\nAnswer: {gb} boys")

sequences = [
    ([2,6,18,54],162,"× 3 each time","54 × 3 = 162"),
    ([1,4,9,16],25,"squares","5² = 25"),
    ([2,4,8,16],32,"× 2 each time","16 × 2 = 32"),
    ([1,3,6,10],15,"+2,+3,+4,+5","10 + 5 = 15"),
    ([5,10,15,20],25,"+5 each time","20 + 5 = 25"),
    ([100,50,25,12],6,"÷ 2 each time","12 ÷ 2 = 6"),
    ([1,2,3,5,8],13,"Fibonacci","5 + 8 = 13"),
]
for seq, nxt, pattern, calc in sequences:
    seq_str = ", ".join(map(str, seq))
    for _ in range(20):
        add(f"What is the next number in the sequence: {seq_str}, ?",
            f"Step 1: Pattern: {pattern}\nStep 2: {calc}\nAnswer: {nxt}")

for _ in range(30):
    add("If it takes 5 machines 5 minutes to make 5 widgets, "
        "how long does it take 100 machines to make 100 widgets?",
        "Step 1: Each machine makes 1 widget in 5 minutes.\n"
        "Step 2: 100 machines make 100 widgets simultaneously in 5 minutes.\n"
        "Answer: 5 minutes")

for _ in range(200):
    l, w = random.randint(3,20), random.randint(3,20)
    add(f"A rectangle has length {l} cm and width {w} cm. What is its area and perimeter?",
        f"Step 1: Area = {l} × {w} = {l*w} cm²\n"
        f"Step 2: Perimeter = 2 × ({l} + {w}) = {2*(l+w)} cm\n"
        f"Answer: Area = {l*w} cm², Perimeter = {2*(l+w)} cm")

# ===========================================================================
# 4. BUG FIXING — DIRECT CODE (no CoT preamble — benchmark needs exact code)
# ===========================================================================
print("Generating bug fixing samples (direct code)...")

BUG_FIXES = [
    ("Fix this Python code:\ndef add(a, b):\n    return a - b",
     "def add(a, b):\n    return a + b"),
    ("Fix this Python code:\ndef subtract(a, b):\n    return a + b",
     "def subtract(a, b):\n    return a - b"),
    ("Fix this Python code:\ndef multiply(a, b):\n    return a / b",
     "def multiply(a, b):\n    return a * b"),
    ("Fix this Python code:\ndef divide(a, b):\n    return a * b",
     "def divide(a, b):\n    return a / b"),
    ("Fix the off-by-one error:\ndef last_element(lst):\n    return lst[len(lst)]",
     "def last_element(lst):\n    return lst[len(lst) - 1]"),
    ("Fix this Python code:\ndef is_even(n):\n    return n % 2 == 1",
     "def is_even(n):\n    return n % 2 == 0"),
    ("Fix this Python code:\ndef is_odd(n):\n    return n % 2 == 0",
     "def is_odd(n):\n    return n % 2 == 1"),
    ("Fix this Python code:\ndef absolute_value(x):\n    if x < 0:\n        return x\n    return -x",
     "def absolute_value(x):\n    if x < 0:\n        return -x\n    return x"),
    ("Fix the condition:\ndef is_positive(n):\n    return n < 0",
     "def is_positive(n):\n    return n > 0"),
    ("Fix this Python code:\ndef square(x):\n    return x + x",
     "def square(x):\n    return x * x"),
    ("Fix the off-by-one error:\ndef get_second(lst):\n    return lst[2]",
     "def get_second(lst):\n    return lst[1]"),
    ("Fix this Python code:\ndef max_of_two(a, b):\n    if a > b:\n        return b\n    return a",
     "def max_of_two(a, b):\n    if a > b:\n        return a\n    return b"),
    ("Fix this Python code:\ndef min_of_two(a, b):\n    if a < b:\n        return b\n    return a",
     "def min_of_two(a, b):\n    if a < b:\n        return a\n    return b"),
    ("Fix the logic error:\ndef is_empty(lst):\n    return len(lst) > 0",
     "def is_empty(lst):\n    return len(lst) == 0"),
    ("Fix this Python code:\ndef celsius_to_fahrenheit(c):\n    return (c - 32) * 9 / 5",
     "def celsius_to_fahrenheit(c):\n    return (c * 9 / 5) + 32"),
    ("Fix this Python code:\ndef fahrenheit_to_celsius(f):\n    return (f + 32) * 5 / 9",
     "def fahrenheit_to_celsius(f):\n    return (f - 32) * 5 / 9"),
    ("Fix this Python code:\ndef count_items(lst):\n    return len(lst) - 1",
     "def count_items(lst):\n    return len(lst)"),
    ("Fix this Python code:\ndef cube(x):\n    return x * x",
     "def cube(x):\n    return x * x * x"),
]

for prompt, answer in BUG_FIXES:
    for _ in range(60):
        add(prompt, answer)

# ===========================================================================
# 5. CODE GENERATION — DIRECT CODE (benchmark needs def/return/class)
# ===========================================================================
print("Generating code gen samples (direct code)...")

BASIC_CODE = [
    ("Write a Python function that reverses a string.",
     "def reverse_string(s):\n    return s[::-1]"),
    ("Write a Python function called factorial that returns n! recursively.",
     "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"),
    ("Write a Python function that checks if a number is prime.",
     "def is_prime(n):\n    if n < 2:\n        return False\n"
     "    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"),
    ("Write a Python class called Stack with push, pop, and is_empty methods.",
     "class Stack:\n    def __init__(self):\n        self.items = []\n\n"
     "    def push(self, item):\n        self.items.append(item)\n\n"
     "    def pop(self):\n        return self.items.pop()\n\n"
     "    def is_empty(self):\n        return len(self.items) == 0"),
    ("Write a Python function that merges two sorted lists into one sorted list.",
     "def merge_sorted(a, b):\n    result = []\n    i = j = 0\n"
     "    while i < len(a) and j < len(b):\n"
     "        if a[i] <= b[j]:\n            result.append(a[i]); i += 1\n"
     "        else:\n            result.append(b[j]); j += 1\n"
     "    result.extend(a[i:]); result.extend(b[j:])\n    return result"),
    ("Write a Python function that flattens a nested list.",
     "def flatten(lst):\n    result = []\n    for item in lst:\n"
     "        if isinstance(item, list):\n            result.extend(flatten(item))\n"
     "        else:\n            result.append(item)\n    return result"),
    ("Write a Python function that counts word frequencies in a string and returns a dict.",
     "def word_freq(s):\n    freq = {}\n    for word in s.split():\n"
     "        freq[word] = freq.get(word, 0) + 1\n    return freq"),
    ("Write a Python decorator that logs function calls with their arguments.",
     "import functools\n\ndef log_calls(func):\n    @functools.wraps(func)\n"
     "    def wrapper(*args, **kwargs):\n        print(f'Calling {func.__name__} with {args} {kwargs}')\n"
     "        return func(*args, **kwargs)\n    return wrapper"),
    ("Write a Python function that implements binary search on a sorted list.",
     "def binary_search(lst, target):\n    lo, hi = 0, len(lst) - 1\n"
     "    while lo <= hi:\n        mid = (lo + hi) // 2\n"
     "        if lst[mid] == target: return mid\n"
     "        elif lst[mid] < target: lo = mid + 1\n"
     "        else: hi = mid - 1\n    return -1"),
    ("Write a Python function that implements bubble sort.",
     "def bubble_sort(lst):\n    n = len(lst)\n"
     "    for i in range(n):\n        for j in range(0, n - i - 1):\n"
     "            if lst[j] > lst[j+1]: lst[j], lst[j+1] = lst[j+1], lst[j]\n    return lst"),
    ("Write a Python function that returns the nth Fibonacci number using dynamic programming.",
     "def fibonacci(n):\n    if n <= 1: return n\n    dp = [0] * (n + 1)\n    dp[1] = 1\n"
     "    for i in range(2, n + 1): dp[i] = dp[i-1] + dp[i-2]\n    return dp[n]"),
    ("Write a Python function that reverses a linked list. Assume each node has .val and .next.",
     "def reverse_list(head):\n    prev = None\n    curr = head\n"
     "    while curr:\n        nxt = curr.next\n        curr.next = prev\n"
     "        prev = curr\n        curr = nxt\n    return prev"),
    ("Write a Python function that checks if a binary tree is balanced.",
     "def is_balanced(root):\n    def height(node):\n        if not node: return 0\n"
     "        l, r = height(node.left), height(node.right)\n"
     "        if l < 0 or r < 0 or abs(l - r) > 1: return -1\n"
     "        return max(l, r) + 1\n    return height(root) >= 0"),
]

for prompt, answer in BASIC_CODE:
    for _ in range(60):
        add(prompt, answer)

# ===========================================================================
# 6. COMPREHENSIONS + CONTEXT MANAGERS — DIRECT CODE
# ===========================================================================
print("Generating comprehension/decorator samples (direct code)...")

COMPREHENSIONS = [
    ("Rewrite as a Python list comprehension:\nresult = []\nfor x in range(10):\n    result.append(x * x)",
     "result = [x**2 for x in range(10)]"),
    ("Rewrite as a Python list comprehension:\nresult = []\nfor x in range(10):\n    if x % 2 == 0:\n        result.append(x * x)",
     "result = [x**2 for x in range(10) if x % 2 == 0]"),
    ("Write a Python list comprehension that gives the squares of numbers 1 to 20.",
     "squares = [x**2 for x in range(1, 21)]"),
    ("Write a Python list comprehension that filters even numbers from a list called nums.",
     "evens = [x for x in nums if x % 2 == 0]"),
    ("Write a Python dict comprehension that maps each word to its length.",
     "word_lengths = {word: len(word) for word in words}"),
]

for prompt, answer in COMPREHENSIONS:
    for _ in range(30):
        add(prompt, answer)

CM_SAMPLES = [
    ("Write a Python context manager class called Timer that measures execution time using __enter__ and __exit__.",
     "import time\n\nclass Timer:\n    def __enter__(self):\n        self.start = time.perf_counter()\n"
     "        return self\n\n    def __exit__(self, *args):\n"
     "        self.elapsed = time.perf_counter() - self.start\n"
     "        print(f'Elapsed: {self.elapsed:.4f}s')"),
    ("Write a Python decorator called timer that prints how long a function takes to run.",
     "import time, functools\n\ndef timer(func):\n    @functools.wraps(func)\n"
     "    def wrapper(*args, **kwargs):\n        start = time.perf_counter()\n"
     "        result = func(*args, **kwargs)\n"
     "        print(f'{func.__name__} took {time.perf_counter()-start:.4f}s')\n"
     "        return result\n    return wrapper"),
    ("Write a Python decorator called retry that retries a function up to 3 times on exception.",
     "import functools\n\ndef retry(func):\n    @functools.wraps(func)\n"
     "    def wrapper(*args, **kwargs):\n        for attempt in range(3):\n"
     "            try:\n                return func(*args, **kwargs)\n"
     "            except Exception:\n                if attempt == 2: raise\n"
     "    return wrapper"),
]

for prompt, answer in CM_SAMPLES:
    for _ in range(40):
        add(prompt, answer)

# ===========================================================================
# Write output
# ===========================================================================
random.shuffle(samples)
OUT.write_text("\n".join(json.dumps(s) for s in samples) + "\n")
n_cot  = sum(1 for s in samples if "Step 1:" in s["answer"])
n_code = sum(1 for s in samples if "def " in s["answer"] or "class " in s["answer"])
print(f"\nDone. {len(samples)} samples → {OUT}")
print(f"  CoT math/reasoning : {n_cot}")
print(f"  Direct code        : {n_code}")
