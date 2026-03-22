import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from datasets import load_dataset
from mamba_rbm import RecursiveMambaLM, Config
import os
import time
import warnings

def load_v6_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "latest_checkpoint.pt"
    
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return None, None, None

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    config = Config(
        vocab_size=len(tokenizer), 
        d_model=1024, 
        n_layers=8, 
        seq_len=1024,
        n_reasoning=3 # 🚀 Hard lock at N=3 for benchmark parity
    )
    
    model = RecursiveMambaLM(config).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    
    return model, tokenizer, device

def get_choice_loss(model, tokenizer, device, prompt, choice_text):
    text = prompt + " " + choice_text
    inputs = tokenizer.encode(text, return_tensors="pt").to(device)
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = prompt_ids.shape[1]
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(inputs)
            
    # Shift logits and labels aligned to the choice payload
    shift_logits = logits[0, prompt_len-1:-1, :].contiguous()
    shift_labels = inputs[0, prompt_len:].contiguous()
    
    loss = F.cross_entropy(shift_logits, shift_labels, reduction='mean').item()
    return loss

def evaluate_piqa(model, tokenizer, device, num_samples=100):
    print(f"\n[1/2] Loading PIQA Dataset (Testing Physical Logic & Common Sense)")
    try:
        ds = load_dataset("piqa", split="validation", trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load PIQA: {e}")
        return 0

    total = min(num_samples, len(ds))
    correct = 0
    
    print(f"Evaluating {total} samples at N=3...")
    start_time = time.time()
    
    for i in range(total):
        item = ds[i]
        prompt = f"Goal: {item['goal']}\nSolution:"
        
        loss1 = get_choice_loss(model, tokenizer, device, prompt, item["sol1"])
        loss2 = get_choice_loss(model, tokenizer, device, prompt, item["sol2"])
        
        pred = 0 if loss1 < loss2 else 1
        if pred == item["label"]:
            correct += 1
            
        if (i+1) % 25 == 0:
            print(f"  Progress: {i+1}/{total} | Current Acc: {correct/(i+1)*100:.1f}%")

    end_time = time.time()
    acc = correct / total
    print(f"✅ PIQA Final Accuracy: {acc*100:.1f}% ({correct}/{total}) in {end_time - start_time:.1f}s")
    return acc

def evaluate_boolq(model, tokenizer, device, num_samples=100):
    print(f"\n[2/2] Loading BoolQ Dataset (Testing Reading Comprehension & Boolean Logic)")
    try:
        ds = load_dataset("boolq", split="validation", trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load BoolQ: {e}")
        return 0

    total = min(num_samples, len(ds))
    correct = 0
    
    print(f"Evaluating {total} samples at N=3...")
    start_time = time.time()
    
    for i in range(total):
        item = ds[i]
        prompt = f"Passage: {item['passage']}\nQuestion: {item['question']}?\nAnswer:"
        
        loss_true = get_choice_loss(model, tokenizer, device, prompt, "Yes")
        loss_false = get_choice_loss(model, tokenizer, device, prompt, "No")
        
        pred = True if loss_true < loss_false else False
        if pred == item["answer"]:
            correct += 1
            
        if (i+1) % 25 == 0:
            print(f"  Progress: {i+1}/{total} | Current Acc: {correct/(i+1)*100:.1f}%")

    end_time = time.time()
    acc = correct / total
    print(f"✅ BoolQ Final Accuracy: {acc*100:.1f}% ({correct}/{total}) in {end_time - start_time:.1f}s")
    return acc

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print("==================================================")
    print("HF Benchmark Script for Recursive Dual-Path Mamba")
    print("==================================================")
    
    model, tokenizer, device = load_v6_model()
    if model is None:
        exit(1)
        
    evaluate_piqa(model, tokenizer, device, num_samples=100)
    evaluate_boolq(model, tokenizer, device, num_samples=100)
    
    print("\nBenchmark Complete.")
