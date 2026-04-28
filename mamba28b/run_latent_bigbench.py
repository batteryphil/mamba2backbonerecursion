import os
import sys
import re
import torch
import json
import re
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel
from datasets import load_dataset
from train_2_8b_rlf import mount_lora
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LORA_CKPT = "checkpoints_phase8/mamba2.8b_phase8_logic.pt"
HALT_STR = "<HALT>"
BASE_REPO = "state-spaces/mamba-2.8b-slimpj"

print("=" * 70)
print("  🚀 NATIVE LATENT BIGBENCH EVALUATOR (100-SHOT CAP) 🚀  ")
print("=" * 70)

print(f"[INIT] Mapping pristine Phase 8 '{LORA_CKPT}' directly onto state-spaces base...")
model = MambaLMHeadModel.from_pretrained(BASE_REPO, dtype=torch.bfloat16, device=DEVICE)
mount_lora(model)
st = torch.load(LORA_CKPT, map_location=DEVICE, weights_only=True)
model.load_state_dict(st, strict=False)
model.eval()

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tok.pad_token = tok.eos_token
if "<HALT>" not in tok.get_vocab():
    tok.add_special_tokens({"additional_special_tokens": ["<THINK>", "<HALT>"]})
HALT_ID = tok.convert_tokens_to_ids("<HALT>")

def infer(prompt: str) -> str:
    p_ids = tok.encode(prompt, return_tensors='pt').to(DEVICE)
    raw_res = ""
    for _ in range(64):
        with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            out = model.generate(input_ids=p_ids, max_length=p_ids.shape[1] + 1, cg=True)
            new_id = out[0][-1].item()
            if new_id == HALT_ID: break
            raw_res += tok.decode([new_id])
            p_ids = out
            if raw_res.count("=") > 30: break
    
    ans_match = re.search(r'Answer:\s*(.*)', raw_res, re.IGNORECASE)
    if ans_match: return ans_match.group(1).strip()
    return raw_res.strip()

def run_subset(subset_name: str, n_samples=100):
    print(f"\n[EVAL] Fetching '{subset_name}' dataset...")
    try:
        # bigbench sets are available under tasksource/bigbench
        ds = load_dataset("tasksource/bigbench", subset_name, split="train")
    except Exception as e:
        print(f"[ERROR] Could not load dataset {subset_name}: {e}")
        return

    samples = list(ds)
    random.seed(42)
    random.shuffle(samples)
    samples = samples[:n_samples]
    
    correct = 0
    print(f"\n[EVAL] Running {len(samples)} generation probes for {subset_name}...")
    
    for i, s in enumerate(samples):
        q = s['inputs']
        targets_str = s.get('multiple_choice_targets', [])
        scores = s.get('multiple_choice_scores', [])
        correct_idx = -1
        
        if scores and targets_str:
            try:
                correct_idx = scores.index(1)
                expected = targets_str[correct_idx]
            except ValueError:
                expected = s.get('targets', [''])[0]
        else:
            expected = s.get('targets', [''])[0]
            if expected.lower() in ['yes', 'no']:
                targets_str = ['Yes', 'No']
                correct_idx = 0 if expected.lower() == 'yes' else 1

        options_str = " ".join([f"{chr(65+j)}) {t}" for j, t in enumerate(targets_str)]) if targets_str else ""
        prompt = f"[QA] Q: {q.strip()}"
        if options_str:
            prompt += f" {options_str}"
        prompt += "\nSolution: "
        
        raw_res = infer(prompt)
        extracted = raw_res
        
        ans_match = re.search(r'Answer:\s*(.*)', raw_res, re.IGNORECASE)
        if ans_match:
            extracted = ans_match.group(1).strip()
            
        # Match either exact string or the letter option
        hit = (expected.lower().strip() in extracted.lower().strip())
        if not hit and correct_idx >= 0:
            exp_letter = chr(65+correct_idx).lower()
            hit = (extracted.lower().strip().startswith(exp_letter))
            
        if hit:
            correct += 1
            
        mark = "✅" if hit else "❌"
        res_display = extracted.replace('\n', ' ')
        if len(res_display) > 35: res_display = res_display[:32] + "..."
        expected_display = expected[:15]
        
        print(f"  [{i+1}/{len(samples)}] Exp: {expected_display:<15} | Got: {res_display:<35} | {mark}")
        
    score = (correct / len(samples)) * 100
    print(f"\n[METRIC] {subset_name.upper()} SCORE: {score:.1f}% ({correct}/{len(samples)})")

if __name__ == "__main__":
    run_subset("logical_deduction", 100)
    run_subset("strategyqa", 100)
