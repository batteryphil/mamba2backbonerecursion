import json
import random
import os

def generate_logic_shock_v2():
    # 🚀 EXPANDED NAME POOL (500+ unique identifiers)
    names = [
        "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda", "David", "Elizabeth",
        "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Christopher", "Karen",
        "Charles", "Lisa", "Daniel", "Nancy", "Matthew", "Betty", "Anthony", "Sandra", "Mark", "Margaret",
        "Donald", "Ashley", "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
        "Kenneth", "Dorothy", "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa", "Timothy", "Deborah",
        "Ronald", "Stephanie", "Edward", "Rebecca", "Jason", "Sharon", "Jeffrey", "Laura", "Ryan", "Cynthia",
        "Jacob", "Kathleen", "Gary", "Amy", "Nicholas", "Shirley", "Eric", "Angela", "Jonathan", "Helen",
        "Stephen", "Anna", "Larry", "Brenda", "Justin", "Pamela", "Scott", "Nicole", "Brandon", "Emma",
        "Benjamin", "Samantha", "Samuel", "Katherine", "Gregory", "Christine", "Alexander", "Debra", "Patrick", "Rachel",
        "Frank", "Catherine", "Raymond", "Carolyn", "Jack", "Janet", "Dennis", "Ruth", "Jerry", "Maria",
        "Tyler", "Heather", "Aaron", "Diane", "Jose", "Virginia", "Adam", "Julie", "Nathan", "Joyce",
        "Henry", "Victoria", "Douglas", "Olivia", "Zachary", "Kelly", "Peter", "Christina", "Kyle", "Lauren",
        "Ethan", "Joan", "Walter", "Evelyn", "Harold", "Judith", "Jeremy", "Megan", "Christian", "Cheryl",
        "Keith", "Andrea", "Roger", "Hannah", "Terry", "Martha", "Gerald", "Jacqueline", "Lawrence", "Frances",
        "Sean", "Gloria", "Arthur", "Ann", "Austin", "Teresa", "Noah", "Kathryn", "Carl", "Sara",
        "Lawrence", "Janice", "Dylan", "Jean", "Jesse", "Alice", "Jordan", "Madison", "Bryan", "Doris",
        "Billy", "Abigail", "Joe", "Julia", "Bruce", "Judy", "Gabriel", "Grace", "Logan", "Amber",
        "Albert", "Alice", "Willie", "Denise", "Alan", "Beverly", "Juan", "Amber", "Wayne", "Danielle",
        "Elijah", "Theresa", "Randy", "Sophia", "Dylan", "Marie", "Howard", "Diana", "Vincent", "Brittany",
        "Roy", "Natalie", "Eugene", "Isabella", "Gabriel", "Charlotte", "Louis", "Rose", "Bobby", "Alexis",
        "Harry", "Kayla"
    ]
    # Add some random unique strings to prevent lexical lock-on
    for i in range(300):
        names.append(f"Entity_{i:03d}")
    
    data = []

    def get_fmt(text, json_data, python_logic, tag):
        combined = f"--- RAW TEXT ---\n{text}\n\n--- JSON DATA ---\n{json_data}\n\n--- PYTHON LOGIC ---\n{python_logic}"
        return {"text": combined, "type": tag}

    # 1. Logic Shock V2: Inversion (2,500 samples)
    for _ in range(2500):
        n1, n2 = random.sample(names, 2)
        distractor_entity = random.choice(names)
        while distractor_entity in [n1, n2]: distractor_entity = random.choice(names)
        
        raw_text = f"Fact: {distractor_entity} is wearing a hat. Fact: {n1} is shorter than {n2}. Who is the tallest person mentioned? Answer: {n2}."
        
        json_fmt = json.dumps({"facts": [f"{n1} < {n2}", f"wearing_hat({distractor_entity})"], "answer": n2})
        python_fmt = f"def logic():\n    h = {{'{n1}': 1, '{n2}': 2}}\n    return max(h, key=h.get)\n# -> {n2}"
        
        data.append(get_fmt(raw_text, json_fmt, python_fmt, "logic_shock_v2_inversion"))

    # 2. Logic Shock V2: Deep Chains (2,500 samples)
    for _ in range(2500):
        num_ent = random.randint(4, 6)
        current = random.sample(names, num_ent)
        premises = []
        for i in range(num_ent - 1):
            if random.random() > 0.5:
                premises.append(f"{current[i]} is taller than {current[i+1]}")
            else:
                premises.append(f"{current[i+1]} is shorter than {current[i]}")
        
        # Add irrelevant noise
        premises.insert(random.randint(0, len(premises)), f"{random.choice(names)} prefers coffee over tea.")
        
        random.shuffle(premises)
        raw_text = f"Contextual Data: {'. '.join(premises)}. Question: Of the people whose heights were mentioned, who is the tallest? Answer: {current[0]}."
        
        json_fmt = json.dumps({"chain": " > ".join(current), "answer": current[0]})
        python_fmt = f"def sort_h():\n    order = {current}\n    return order[0]\n# -> {current[0]}"
        
        data.append(get_fmt(raw_text, json_fmt, python_fmt, "logic_shock_v2_chain"))

    # 3. Logic Shock V2: Negated Conditionals (2,500 samples)
    for _ in range(2500):
        n1, n2 = random.sample(names, 2)
        raw_text = f"Observation: It is false that {n1} is taller than {n2}. Assuming they have different heights, who is taller? Answer: {n2}."
        json_fmt = json.dumps({"negation": f"NOT taller({n1}, {n2})", "answer": n2})
        python_fmt = f"def check():\n    t_n1_n2 = False\n    return '{n2}' if not t_n1_n2 else '{n1}'"
        
        data.append(get_fmt(raw_text, json_fmt, python_fmt, "logic_shock_v2_negation"))

    # 4. Standard Reasoning (2,500 samples)
    for _ in range(2500):
        n1, n2 = random.sample(names, 2)
        raw_text = f"Context: {n1} is older than {n2}. Who is older? Answer: {n1}."
        json_fmt = json.dumps({"fact": f"age({n1}) > age({n2})", "answer": n1})
        python_fmt = f"def age_test(): return '{n1}'"
        data.append(get_fmt(raw_text, json_fmt, python_fmt, "logic_shock_v2_standard"))

    # Save to logic_v2.json
    with open("logic_v2.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ LOGIC SHOCK V2 COMPLETE: 10,000 samples generated with distractor noise.")
    print(f"Name Pool Size: {len(names)} unique entities.")

if __name__ == "__main__":
    generate_logic_shock_v2()
