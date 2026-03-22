import json
import random

def generate_rft_data(num_samples=2000):
    data = []
    
    # 1. Acoustic Prosody Logic
    prosody_rules = [
        ("exclamation", "rise", "excitement or emphasis"),
        ("question", "rise", "inquiry or uncertainty"),
        ("statement", "fall", "completion or certainty"),
        ("comma", "stay flat", "continuation or pause"),
        ("whisper", "drop in volume", "secrecy or quietness")
    ]
    for _ in range(num_samples // 4):
        rule = random.choice(prosody_rules)
        sentence = f"The man walked down the street{'.' if rule[0] == 'statement' else '!' if rule[0] == 'exclamation' else '?' if rule[0] == 'question' else ',' if rule[0] == 'comma' else ' (whispering)'}"
        text = f"User: Analyze the prosody for the following sentence based on its punctuation: \"{sentence}\" | Assistant: Let's think step by step. First, we identify the punctuation or style marker at the end of the sentence. The sentence ends with a {rule[0]} marker. Second, we recall the phonetic rule associated with a {rule[0]}. The rule states that the pitch should {rule[1]} to convey {rule[2]}. Finally, we apply this rule to the acoustic generation parameters. Therefore, the prosody must {rule[1]} at the end. The answer is: \\boxed{{pitch_{rule[1]}}}"
        data.append(text)
        
    # 2. Hardware Control (RTX 3060 / nvidia-smi)
    for _ in range(num_samples // 4):
        temp = random.randint(40, 95)
        vram_used = random.randint(1000, 12000)
        action = "throttle_workload" if temp > 85 else "free_vram" if vram_used > 11000 else "continue_execution"
        text = f"User: Check the hardware status. The current GPU temperature is {temp}C and the VRAM usage is {vram_used}MB on the RTX 3060. Determine the next action. | Assistant: Let's think step by step. First, we analyze the GPU temperature. The current temperature is {temp}C. The critical threshold is 85C. Since {temp} {'is greater than' if temp > 85 else 'is less than or equal to'} 85, the thermal status is {'critical' if temp > 85 else 'safe'}. Second, we analyze the VRAM usage. The RTX 3060 has 12288MB of VRAM. The current usage is {vram_used}MB. The safe threshold is 11000MB. Since {vram_used} {'is greater than' if vram_used > 11000 else 'is less than or equal to'} 11000, the memory status is {'critical' if vram_used > 11000 else 'safe'}. Finally, based on these two states, we determine the control action. The action is {action}. The answer is: \\boxed{{{action}}}"
        data.append(text)

    # 3. Code Syntax (Selective Scans / List Comprehensions)
    code_topics = [
        ("List Comprehension", "squaring even numbers", "We iterate through the original list `xs`. For each element `x`, we check the condition `x % 2 == 0`. If true, we apply the expression `x**2`. The resulting list contains the squared even numbers.", "`[x**2 for x in xs if x % 2 == 0]`"),
        ("Mamba Selective Scan", "updating the hidden state", "We receive the current input token `x_t` and the previous hidden state `h_{t-1}`. We calculate the data-dependent step size `dt`. We discretize the continuous parameters `A` and `B` using `dt` to get `A_bar` and `B_bar`. We update the hidden state: `h_t = A_bar * h_{t-1} + B_bar * x_t`.", "`h_t = A_bar * h_{t-1} + B_bar * x_t`")
    ]
    for _ in range(num_samples // 4):
        topic = random.choice(code_topics)
        text = f"User: Explain the step-by-step logic for {topic[0]} when {topic[1]}. | Assistant: Let's think step by step. First, we define the objective: {topic[1]} using {topic[0]}. Second, we break down the fundamental mechanism. {topic[2]} Finally, we synthesize this logic into the final syntax. The answer is: \\boxed{{{topic[3]}}}"
        data.append(text)

    # 4. Agentic Search (eBay GPU scraping)
    gpu_models = ["Tesla P40 24GB", "Tesla M40 24GB", "RTX 3060 12GB"]
    for _ in range(num_samples // 4 + (num_samples % 4)):
        gpu = random.choice(gpu_models)
        budget = random.randint(50, 300)
        text = f"User: I need to upgrade my server for LLM inference. Find me a {gpu} on eBay for under ${budget}. | Assistant: Let's think step by step. First, I identify the target item from the user's request. The item is `{gpu}`. Second, I identify the constraints. The budget constraint is `< ${budget}`. Third, I formulate the search query parameters required by the `ebay_search` tool. The query string needs to combine the item and the budget logic. The exact tool call is `<tool name=\"ebay_search\" params='{{\"query\": \"{gpu}\", \"max_price\": {budget}}}'></tool>`. Finally, I emit the structured command to execute the search. The answer is: \\boxed{{<tool name=\"ebay_search\" params='{{\"query\": \"{gpu}\", \"max_price\": {budget}}}'></tool>}}"
        data.append(text)

    random.shuffle(data)
    return data[:num_samples]

if __name__ == "__main__":
    rft_data = generate_rft_data(2000)
    with open("synthetic_dsr_data.json", "w") as f:
        json.dump(rft_data, f, indent=2)
    print(f"Generated {len(rft_data)} RFT examples in synthetic_dsr_data.json")
