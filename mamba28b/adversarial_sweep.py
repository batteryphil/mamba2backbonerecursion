#!/usr/bin/env python3
"""
adversarial_sweep.py
====================
Adversarial 10-hop variable tracking with distractors.
Compares:
 1. Base Mamba (no SFT, no extra loops)
 2. Latent Engine Baseline (1 loop)
 3. Latent Engine Adaptive (HaltingHead chooses depth)
"""

import re
import json
import random
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import gc

ENGINE_REPO = "batteryphil/mamba-2.8b-latent"
BASE_REPO   = "state-spaces/mamba-2.8b-hf"

N_PROBLEMS  = 30
HOPS        = 10
DISTRACTORS = 5
MAX_NEW     = 60
HALT_THRESH = 0.70
TARGET_LOOPS= 25

def generate_adversarial_problem(seed=None):
    if seed is not None: random.seed(seed)
    vars_avail = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    random.shuffle(vars_avail)
    
    curr_var = vars_avail.pop(0)
    current_val = random.randint(2, 6)
    
    steps = [f"{curr_var}={current_val}"]
    active_vars = [curr_var]
    
    d_left = DISTRACTORS
    
    for _ in range(HOPS):
        next_var = vars_avail.pop(0)
        op = random.choice(['+', '-', '*'])
        operand = random.randint(2, 4)
        
        if op == '+': current_val += operand
        elif op == '-': current_val -= operand
        elif op == '*': current_val *= operand
        
        steps.append(f"{next_var}={curr_var}{op}{operand}")
        curr_var = next_var
        active_vars.append(curr_var)
        
        # Chance to add a distractor here
        if d_left > 0 and random.random() < 0.4:
            dist_var = vars_avail.pop(0)
            base_var = random.choice(active_vars)
            dist_op = random.choice(['+', '-', '*'])
            dist_operand = random.randint(2, 4)
            steps.append(f"{dist_var}={base_var}{dist_op}{dist_operand}")
            active_vars.append(dist_var)
            d_left -= 1
            
    # dump remaining distractors
    while d_left > 0:
        dist_var = vars_avail.pop(0)
        base_var = random.choice(active_vars)
        dist_op = random.choice(['+', '-', '*'])
        dist_operand = random.randint(2, 4)
        steps.append(f"{dist_var}={base_var}{dist_op}{dist_operand}")
        active_vars.append(dist_var)
        d_left -= 1
        
    prompt = f"[LOGIC] " + ". ".join(steps) + f". Output {curr_var}."
    return prompt, current_val

def check_answer(output: str, expected: int) -> bool:
    expected_str = str(expected)
    for n in re.findall(r"-?\d+", output):
        if n == expected_str: return True
    return False

class HaltingHead(nn.Module):
    def __init__(self, d_input=2561):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).squeeze(-1)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    print(f"Generating {N_PROBLEMS} adversarial problems ({HOPS} hops, {DISTRACTORS} distractors)...")
    problems = [generate_adversarial_problem(seed=42+i) for i in range(N_PROBLEMS)]
    
    results_base = []
    
    # === 1. Evaluate BASE MAMBA ===
    print("\n--- LOADING BASE MAMBA ---")
    try:
        base_tok = AutoTokenizer.from_pretrained(BASE_REPO)
        if base_tok.pad_token is None: base_tok.pad_token = base_tok.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(BASE_REPO, dtype=dtype, device_map=device)
        base_model.eval()
        
        print(f"Running BASE inference ({N_PROBLEMS} problems)...")
        for i, (prompt, expected) in enumerate(problems):
            with torch.no_grad():
                inputs = base_tok(prompt + " Answer:", return_tensors="pt").to(device)
                out = base_model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
                ans = base_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            correct = check_answer(ans, expected)
            results_base.append(correct)
            print(f" [Base] {i+1:2d} | {'✅' if correct else '❌'} Exp: {expected:5d} | Out: {ans[:40]}")
            
        del base_model
        del base_tok
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Failed to load/run base mamba: {e}")
        print("Skipping base comparison.")
        results_base = [False] * N_PROBLEMS
        
    
    # === 2. Evaluate LATENT ENGINE ===
    print("\n--- LOADING LATENT ENGINE ---")
    tok = AutoTokenizer.from_pretrained(ENGINE_REPO, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(ENGINE_REPO, trust_remote_code=True, dtype=dtype, device_map=device)
    model.eval()
    
    head_path = hf_hub_download(repo_id=ENGINE_REPO, filename="halting_head.pt")
    ckpt = torch.load(head_path, weights_only=True, map_location="cpu")
    head = HaltingHead(ckpt["d_input"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    
    results_baseline = []
    results_adaptive = []
    depths = []
    
    print(f"Running ENGINE inference ({N_PROBLEMS} problems)...")
    for i, (prompt, expected) in enumerate(problems):
        # single pass
        with torch.no_grad():
            inputs = tok(prompt + " Answer:", return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
            ans_b = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        b_ok = check_answer(ans_b, expected)
        results_baseline.append(b_ok)
        
        # adaptive pass
        with torch.no_grad():
            lp_used = 1
            for lp in range(TARGET_LOOPS):
                inputs = tok(prompt + "=" * lp, return_tensors="pt").to(device)
                out = model(**inputs, output_hidden_states=True)
                h = out.hidden_states[-1][0, -1, :].float().cpu()
                ln = torch.tensor([lp/TARGET_LOOPS], dtype=torch.float32)
                p = head(torch.cat([h, ln]).unsqueeze(0)).item()
                if p >= HALT_THRESH:
                    lp_used = lp + 1
                    break
            gen = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
            ans_a = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            
        a_ok = check_answer(ans_a, expected)
        results_adaptive.append(a_ok)
        depths.append(lp_used)
        
        print(f" [Eng]  {i+1:2d} | B={'✅' if b_ok else '❌'}  A={'✅' if a_ok else '❌'}  L={lp_used:2d} | Exp: {expected:5d}")
        print(f"        B_out: {ans_b[:40]}")
        print(f"        A_out: {ans_a[:40]}")


    # === SUMMARY ===
    b_acc = sum(results_base) / N_PROBLEMS * 100 if len(results_base) else 0
    e1_acc = sum(results_baseline) / N_PROBLEMS * 100
    ea_acc = sum(results_adaptive) / N_PROBLEMS * 100
    avg_l = sum(depths)/N_PROBLEMS
    
    print("\n" + "="*70)
    print("RESULTS: ADVERSARIAL VARIABLE TRACKING (10 hops + 5 distractors)")
    print("="*70)
    print(f" 1. Base Mamba (no SFT, no loops):      {b_acc:5.1f}%")
    print(f" 2. Latent Engine (1 loop, baseline):   {e1_acc:5.1f}%")
    print(f" 3. Latent Engine (Adaptive loops):     {ea_acc:5.1f}%  (avg {avg_l:.1f} loops)")
    print("="*70)
    
    with open("adv_sweep.json", "w") as f:
        json.dump({
            "base": b_acc, "engine_1": e1_acc, "engine_a": ea_acc, "avg_loops": avg_l
        }, f)
        
if __name__ == "__main__":
    main()
