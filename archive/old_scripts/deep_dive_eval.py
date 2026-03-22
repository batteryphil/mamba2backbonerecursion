"""
deep_dive_eval.py — Comprehensive RBM v6 GPU Evaluation Suite
Tests: Weight Health, Logit Distribution, Recursive Depth, Logic Battery,
       QA Extraction, Needle in Haystack, OOD Vocabulary, Gradient Sensitivity
"""
import torch
import torch.nn.functional as F
import numpy as np
import time
import os
import json
from transformers import GPT2Tokenizer
from mamba_rbm import RecursiveMambaLM, Config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "latest_checkpoint.pt"
SEP = "=" * 60

def load_model(n_reasoning=3):
    """Load model from latest checkpoint onto GPU."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    config = Config(vocab_size=len(tokenizer), d_model=1024, n_layers=8,
                    seq_len=1024, n_reasoning=n_reasoning)
    model = RecursiveMambaLM(config).to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(state["model_state"])
    model.eval()
    step = state.get("step", 0)
    print(f"  Loaded checkpoint at Step {step} | Device: {DEVICE}")
    return model, tokenizer, step

def generate(model, tokenizer, prompt, max_new=30, temperature=0.7):
    """Generate tokens from a prompt."""
    ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(ids)
            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, 1).unsqueeze(0)
            ids = torch.cat([ids, next_tok], dim=1)
    return tokenizer.decode(ids[0, tokenizer.encode(prompt, return_tensors="pt").shape[1]:])

def score_answer(model, tokenizer, prompt, answer):
    """Compute cross-entropy loss of a model generating an answer following a prompt."""
    full = tokenizer.encode(prompt + answer, return_tensors="pt").to(DEVICE)
    p_len = tokenizer.encode(prompt, return_tensors="pt").shape[1]
    with torch.no_grad():
        logits = model(full)
    shift_logits = logits[0, p_len-1:-1,:].contiguous()
    shift_labels = full[0, p_len:].contiguous()
    return F.cross_entropy(shift_logits, shift_labels).item()


# ──────────────────────────────────────────────────────────────────────────────
def test_weight_health(model):
    """Check all parameters for NaN, Inf, dead weights, and weight distribution."""
    print(f"\n{SEP}\n[TEST 1] WEIGHT HEALTH\n{SEP}")
    total, nan_count, inf_count, near_zero = 0, 0, 0, 0
    param_norms = []
    for name, p in model.named_parameters():
        d = p.data
        total += d.numel()
        nan_count += d.isnan().sum().item()
        inf_count += d.isinf().sum().item()
        near_zero += (d.abs() < 1e-8).sum().item()
        param_norms.append((name, d.norm().item(), d.mean().item(), d.std().item()))

    print(f"  Total Parameters:  {total:,}")
    print(f"  NaN Values:        {nan_count}  {'✅' if nan_count==0 else '🚨 CRITICAL'}")
    print(f"  Inf Values:        {inf_count}  {'✅' if inf_count==0 else '🚨 CRITICAL'}")
    print(f"  Near-Zero (<1e-8): {near_zero:,} ({near_zero/total*100:.2f}%) {'✅' if near_zero/total < 0.05 else '⚠️ HIGH'}")
    print(f"\n  Top 5 layers by L2 norm:")
    param_norms.sort(key=lambda x: x[1], reverse=True)
    for name, norm, mean, std in param_norms[:5]:
        print(f"    {name:<50} norm={norm:.3f}  mean={mean:.4f}  std={std:.4f}")
    return nan_count == 0 and inf_count == 0


def test_logit_distribution(model, tokenizer):
    """Probe the head bias: entropy, top-k diversity, and repetition tendency."""
    print(f"\n{SEP}\n[TEST 2] LOGIT DISTRIBUTION & HEAD BIAS\n{SEP}")
    prompts = [
        "The capital of France is",
        "Alice is taller than Bob. Who is tallest?",
        "def fibonacci(n):",
        "Once upon a time in a land far away,",
        "Gold is heavier than Silver. Silver is heavier than Iron. Who is lightest?"
    ]
    for prompt in prompts:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(ids)
        last = logits[0, -1, :]
        probs = F.softmax(last, dim=-1)
        entropy = -(probs * (probs + 1e-9).log()).sum().item()
        top5 = torch.topk(probs, 5)
        top5_tokens = [tokenizer.decode([i.item()]).strip() for i in top5.indices]
        top5_probs  = [f"{p.item()*100:.1f}%" for p in top5.values]
        print(f"\n  Prompt: '{prompt[:55]}'")
        print(f"  Entropy:  {entropy:.3f} (max~10.8 for uniform | low=biased)")
        print(f"  Top-5:    {list(zip(top5_tokens, top5_probs))}")


def test_recursion_depth_comparison(model, tokenizer):
    """Compare loss and speed across N=1, 2, 3 on the same logic prompts."""
    print(f"\n{SEP}\n[TEST 3] RECURSIVE DEPTH COMPARISON (N=1 vs N=2 vs N=3)\n{SEP}")
    probes = [
        ("Simple 2-var logic",
         "Alice is taller than Bob. Who is shortest?",
         " Bob."),
        ("3-var chain logic",
         "Mars is hotter than Pluto. Pluto is hotter than Uranus. Who is coldest?",
         " Uranus."),
        ("4-var chain logic",
         "Gold is heavier than Silver. Silver is heavier than Copper. Copper is heavier than Zinc. Who is lightest?",
         " Zinc."),
        ("5-var deep chain",
         "Titanium is heavier than Iron. Iron is heavier than Bronze. Bronze is heavier than Brass. Brass is heavier than Nickel. Who is lightest?",
         " Nickel."),
        ("QA extraction",
         "Context: [The system password is Delta-9. The weather is cloudy.] Question: What is the system password? Answer:",
         " Delta-9."),
    ]
    original_n = model.config.n_reasoning
    results = {}
    header = f"  {'Probe':<25} | {'N=1 Loss':>9} | {'N=2 Loss':>9} | {'N=3 Loss':>9} | {'Best N':>6}"
    print(header)
    print("  " + "-" * 65)
    for label, prompt, target in probes:
        row = {}
        for n in [1, 2, 3]:
            model.config.n_reasoning = n
            loss = score_answer(model, tokenizer, prompt, target)
            row[n] = loss
        best = min(row, key=row.get)
        print(f"  {label:<25} | {row[1]:>9.4f} | {row[2]:>9.4f} | {row[3]:>9.4f} | {'N='+str(best):>6}")
        results[label] = row
    model.config.n_reasoning = original_n
    return results


def test_logic_battery(model, tokenizer):
    """Fire a diverse battery of logic prompts and capture generated answers."""
    print(f"\n{SEP}\n[TEST 4] LOGIC GENERATION BATTERY (N=3, T=0.1)\n{SEP}")
    cases = [
        ("2-var names",     "Alice is older than Bob. Who is youngest?"),
        ("2-var metals",    "Gold is heavier than Silver. Who is lightest?"),
        ("2-var planets",   "Mars is hotter than Pluto. Which is coldest?"),
        ("3-var names",     "Charlie is taller than Alice. Alice is taller than Bob. Who is shortest?"),
        ("3-var metals",    "Platinum is heavier than Gold. Gold is heavier than Silver. Who is lightest?"),
        ("3-var planets",   "Venus is hotter than Earth. Earth is hotter than Mars. Which is coldest?"),
        ("4-var chain",     "Titanium > Iron > Bronze > Brass (heavier). Who is lightest?"),
        ("Inversion",       "Alice is shorter than Bob. Who is tallest?"),
        ("OOD vocabulary",  "Zorblax is heavier than Quibble. Quibble is heavier than Floop. Who is lightest?"),
        ("Math",            "What is 7 multiplied by 8?"),
        ("General knowledge","What planet is closest to the Sun?"),
        ("Code",            "def add(a, b):\n    # Returns the sum\n    return"),
    ]
    for label, prompt in cases:
        out = generate(model, tokenizer, prompt, max_new=15, temperature=0.1)
        print(f"  [{label:<20}] → '{out.strip()[:60]}'")


def test_needle_gpu(model, tokenizer):
    """Re-run Needle in a Haystack at all depths on the GPU for speed/quality comparison."""
    print(f"\n{SEP}\n[TEST 5] NEEDLE IN A HAYSTACK (GPU, N=1/2/3)\n{SEP}")
    filler = "The committee discussed budget constraints. Recent advances in biology transformed the field. Medieval knights participated in jousting. The library expanded its hours for students. Supply chains faced unprecedented disruption. " * 4
    needle = " The secret code is Protocol-7. "
    question = "\nQuestion: What is the secret code? Answer:"
    target = " Protocol-7."
    original_n = model.config.n_reasoning
    header = f"  {'Depth':<8} | {'N=1':>8} | {'N=2':>8} | {'N=3':>8}"
    print(header)
    print("  " + "-" * 35)
    for depth_frac in [0.0, 0.5, 0.9]:
        row = {}
        words = filler.split()
        idx = int(len(words) * depth_frac)
        words.insert(idx, needle)
        prompt = " ".join(words) + question
        tok = tokenizer.encode(prompt, return_tensors="pt")
        if tok.shape[1] > 1000:
            tok = tok[:, -(1000):]
            prompt = tokenizer.decode(tok[0])
        for n in [1, 2, 3]:
            model.config.n_reasoning = n
            loss = score_answer(model, tokenizer, prompt, target)
            row[n] = loss
        print(f"  {depth_frac*100:<7.0f}% | {row[1]:>8.4f} | {row[2]:>8.4f} | {row[3]:>8.4f}")
    model.config.n_reasoning = original_n


def test_inference_speed(model, tokenizer):
    """Profile TPS at different input lengths and N depths."""
    print(f"\n{SEP}\n[TEST 6] INFERENCE SPEED PROFILE\n{SEP}")
    base = "Alice is taller than Bob. " * 40  # ~200 tokens repeated
    original_n = model.config.n_reasoning
    print(f"  {'N':>3} | {'Seq Len':>8} | {'ms/tok':>8} | {'TPS':>8} | {'VRAM MB':>9}")
    print("  " + "-" * 45)
    for n in [1, 2, 3]:
        for seq_l in [64, 256, 512]:
            model.config.n_reasoning = n
            ids = tokenizer.encode(base, return_tensors="pt")[:, :seq_l].to(DEVICE)
            # Warmup
            with torch.no_grad():
                _ = model(ids)
            torch.cuda.synchronize()
            t0 = time.time()
            RUNS = 5
            with torch.no_grad():
                for _ in range(RUNS):
                    _ = model(ids)
            torch.cuda.synchronize()
            elapsed = (time.time() - t0) / RUNS
            tps = ids.shape[1] / elapsed
            vram = torch.cuda.memory_allocated() / 1e6
            print(f"  {n:>3} | {seq_l:>8} | {elapsed*1000/seq_l:>8.2f} | {tps:>8.0f} | {vram:>9.1f}")
    model.config.n_reasoning = original_n


def test_gradient_sensitivity(model, tokenizer):
    """Measure model output sensitivity to single token perturbations."""
    print(f"\n{SEP}\n[TEST 7] GRADIENT SENSITIVITY (Input Perturbation)\n{SEP}")
    prompt = "Alice is older than Bob. Who is youngest?"
    ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        base_logits = model(ids)[0, -1, :].cpu()

    print(f"  Base prompt: '{prompt}'")
    print(f"  Testing sensitivity by flipping each token one at a time...")
    max_diff = 0
    max_diff_pos = 0
    diffs = []
    for i in range(ids.shape[1]):
        perturbed = ids.clone()
        perturbed[0, i] = (perturbed[0, i] + 100) % len(tokenizer)  # shift token
        with torch.no_grad():
            pert_logits = model(perturbed)[0, -1, :].cpu()
        diff = (base_logits - pert_logits).abs().mean().item()
        diffs.append(diff)
        if diff > max_diff:
            max_diff = diff
            max_diff_pos = i

    avg_diff = np.mean(diffs)
    print(f"  Avg logit shift per token flip:   {avg_diff:.5f}")
    print(f"  Max logit shift:                  {max_diff:.5f}  (at token pos {max_diff_pos}: '{tokenizer.decode([ids[0,max_diff_pos].item()])}')")
    print(f"  Stability Rating: {'✅ STABLE' if avg_diff < 0.02 else '⚠️ SENSITIVE' if avg_diff < 0.1 else '🚨 UNSTABLE'}")


def test_qa_extraction(model, tokenizer):
    """Test QA context extraction across multiple question types now that qa_anchors were added."""
    print(f"\n{SEP}\n[TEST 8] QA EXTRACTION PROBE (Post QA-Anchor Training)\n{SEP}")
    cases = [
        ("Short context",
         "Context: [The server password is Banana-42. The sky is blue.] Question: What is the server password? Answer:",
         " Banana-42."),
        ("Medium context w/ distractor",
         "Context: [Paris is the capital of France. The emergency code is Echo-7. Dogs are mammals.] Question: What is the emergency code? Answer:",
         " Echo-7."),
        ("Name extraction",
         "Context: [The project lead is Alice. The deadline is Friday. The budget is $500.] Question: Who is the project lead? Answer:",
         " Alice."),
        ("OOD entity extraction",
         "Context: [Zorblax is the oldest. Quibble is younger than Zorblax. Floop is youngest.] Question: Who is youngest? Answer:",
         " Floop."),
    ]
    original_n = model.config.n_reasoning
    print(f"  {'Case':<25} | {'N=1 Loss':>9} | {'N=3 Loss':>9} | Generated (N=3)")
    print("  " + "-" * 80)
    for label, prompt, target in cases:
        model.config.n_reasoning = 1
        loss_n1 = score_answer(model, tokenizer, prompt, target)
        model.config.n_reasoning = 3
        loss_n3 = score_answer(model, tokenizer, prompt, target)
        gen = generate(model, tokenizer, prompt, max_new=10, temperature=0.1)
        print(f"  {label:<25} | {loss_n1:>9.4f} | {loss_n3:>9.4f} | '{gen.strip()[:30]}'")
    model.config.n_reasoning = original_n


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    print(f"\n{'#'*60}")
    print("  RBM v6 — DEEP DIVE EVALUATION SUITE")
    print(f"  Device: {DEVICE} | Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    model, tokenizer, step = load_model(n_reasoning=3)
    torch.cuda.empty_cache()

    t_start = time.time()

    okay = test_weight_health(model)
    test_logit_distribution(model, tokenizer)
    test_recursion_depth_comparison(model, tokenizer)
    test_logic_battery(model, tokenizer)
    test_needle_gpu(model, tokenizer)
    test_inference_speed(model, tokenizer)
    test_gradient_sensitivity(model, tokenizer)
    test_qa_extraction(model, tokenizer)

    elapsed = time.time() - t_start
    print(f"\n{'#'*60}")
    print(f"  EVALUATION COMPLETE in {elapsed:.1f}s")
    print(f"  Weight Health: {'PASS ✅' if okay else 'FAIL 🚨'}")
    print(f"{'#'*60}\n")
