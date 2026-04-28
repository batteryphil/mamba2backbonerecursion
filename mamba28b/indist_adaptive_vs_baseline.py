#!/usr/bin/env python3
"""
indist_adaptive_vs_baseline.py
================================
Adaptive vs baseline on in-distribution variable-tracking problems —
the exact format the model was SFT trained on.

Tests both simple (2-step) and complex (5-7 step) chains.
"""

import re
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

REPO_ID     = "batteryphil/mamba-2.8b-latent"
HALT_THRESH = 0.70
MAX_LOOPS   = 25
MAX_NEW     = 80


class HaltingHead(nn.Module):
    """3-layer MLP halting probe."""
    def __init__(self, d_input=2561):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        """Return halt probability."""
        return self.net(x).squeeze(-1)


# In-distribution problems — same [LOGIC] X=... format as SFT training
PROBLEMS = [
    # SIMPLE (2-step)
    {"id": "s1",  "tier": "SIMPLE",
     "prompt": "[LOGIC] X=4. Y=X+3. Output Y.",        "answer": 7},
    {"id": "s2",  "tier": "SIMPLE",
     "prompt": "[LOGIC] A=10. B=A-6. Output B.",        "answer": 4},
    {"id": "s3",  "tier": "SIMPLE",
     "prompt": "[LOGIC] P=3. Q=P*2. Output Q.",          "answer": 6},
    {"id": "s4",  "tier": "SIMPLE",
     "prompt": "[LOGIC] M=15. N=M/3. Output N.",         "answer": 5},
    {"id": "s5",  "tier": "SIMPLE",
     "prompt": "[LOGIC] K=7. L=K+K. Output L.",          "answer": 14},

    # MEDIUM (3-4 step)
    {"id": "m1",  "tier": "MEDIUM",
     "prompt": "[LOGIC] X=5. Y=X*2. Z=Y+3. Output Z.",  "answer": 13},
    {"id": "m2",  "tier": "MEDIUM",
     "prompt": "[LOGIC] A=3. B=A+7. C=B*2. Output C.",   "answer": 20},
    {"id": "m3",  "tier": "MEDIUM",
     "prompt": "[LOGIC] P=8. Q=P-3. R=Q*4. Output R.",   "answer": 20},
    {"id": "m4",  "tier": "MEDIUM",
     "prompt": "[LOGIC] X=4. Y=X*X. Z=Y+X. Output Z.",   "answer": 20},
    {"id": "m5",  "tier": "MEDIUM",
     "prompt": "[LOGIC] A=6. B=A+4. C=B*2. D=C-5. Output D.", "answer": 15},

    # HARD (5-7 step, cross-reference)
    {"id": "h1",  "tier": "HARD",
     "prompt": "[LOGIC] X=5. Y=X*2. Z=Y+3. W=Z-X. Output W.",           "answer": 8},
    {"id": "h2",  "tier": "HARD",
     "prompt": "[LOGIC] A=4. B=A+2. C=B*A. D=C-B. E=D/A. Output E.",    "answer": 4},
    {"id": "h3",  "tier": "HARD",
     "prompt": "[LOGIC] P=3. Q=P*P. R=Q+P. S=R*2. T=S-Q. Output T.",    "answer": 15},
    {"id": "h4",  "tier": "HARD",
     "prompt": "[LOGIC] K=2. L=K*5. M=L+3. N=M*2. O=N-L. Output O.",    "answer": 16},
    {"id": "h5",  "tier": "HARD",
     "prompt": "[LOGIC] X=7. Y=X-3. Z=Y*X. W=Z+Y. V=W/Y. Output V.",    "answer": 14},
    {"id": "h6",  "tier": "HARD",
     "prompt": "[LOGIC] A=5. B=A+1. C=B*A. D=C-A. E=D+B. F=E-C. Output F.", "answer": 1},
    {"id": "h7",  "tier": "HARD",
     "prompt": "[LOGIC] M=6. N=M-2. O=N*M. P=O+N. Q=P/2. R=Q-M. Output R.", "answer": 10},
]


def load_engine():
    """Load model, tokenizer, HaltingHead."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32
    use_remote = (device == "cuda")
    print(f"Device: {device}")

    tok = AutoTokenizer.from_pretrained(REPO_ID, trust_remote_code=use_remote)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        REPO_ID, trust_remote_code=use_remote, dtype=dtype, device_map=device
    )
    model.eval()

    head_path = hf_hub_download(repo_id=REPO_ID, filename="halting_head.pt")
    ckpt = torch.load(head_path, weights_only=True, map_location="cpu")
    head = HaltingHead(ckpt["d_input"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    return tok, model, head, device


def check(output, expected):
    """Check if expected number appears in output."""
    for n in re.findall(r"-?\d+\.?\d*", output):
        try:
            if float(n) == float(expected):
                return True
        except ValueError:
            pass
    return False


def run_baseline(prompt, tok, model, device):
    """Single forward pass."""
    with torch.no_grad():
        toks = tok(prompt, return_tensors="pt",
                   truncation=True, max_length=256).to(device)
        out  = model.generate(**toks, max_new_tokens=MAX_NEW,
                              do_sample=False, repetition_penalty=1.1)
    return tok.decode(out[0][toks["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


def run_adaptive(prompt, tok, model, head, device):
    """HaltingHead chooses depth naturally."""
    with torch.no_grad():
        loops = 1
        p_val = 0.0
        for lp in range(MAX_LOOPS):
            toks = tok(prompt + "=" * lp, return_tensors="pt",
                       truncation=True, max_length=256).to(device)
            out  = model(**toks, output_hidden_states=True)
            h    = out.hidden_states[-1][0, -1, :].float().cpu()
            ln   = torch.tensor([lp / MAX_LOOPS], dtype=torch.float32)
            feat = torch.cat([h, ln]).unsqueeze(0)
            p    = head(feat).item()
            if p >= HALT_THRESH:
                loops = lp + 1
                p_val = p
                break

        gen = model.generate(**toks, max_new_tokens=MAX_NEW,
                             do_sample=False, repetition_penalty=1.1)
    answer = tok.decode(gen[0][toks["input_ids"].shape[1]:],
                        skip_special_tokens=True).strip()
    return answer, loops, round(p_val, 3)


def main():
    """Run sweep and print full answer text for qualitative analysis."""
    tok, model, head, device = load_engine()
    results = []

    print("\n" + "=" * 80)
    print("IN-DISTRIBUTION TEST: [LOGIC] X=... variable tracking")
    print("Adaptive (HaltingHead) vs Single-pass baseline")
    print("=" * 80)

    tiers = {"SIMPLE": [], "MEDIUM": [], "HARD": []}

    for prob in PROBLEMS:
        b_out = run_baseline(prob["prompt"], tok, model, device)
        a_out, loops, p = run_adaptive(prob["prompt"], tok, model, head, device)

        b_ok = check(b_out, prob["answer"])
        a_ok = check(a_out, prob["answer"])
        tiers[prob["tier"]].append((b_ok, a_ok))

        print(f"\n{'─'*80}")
        print(f"[{prob['tier']:6}] {prob['id']}  Expected: {prob['answer']}")
        print(f"  Prompt:   {prob['prompt']}")
        print(f"  Baseline: {'✅' if b_ok else '❌'}  →  {b_out}")
        print(f"  Adaptive: {'✅' if a_ok else '❌'}  ({loops} loops, P={p})  →  {a_out}")

        results.append({
            "id": prob["id"], "tier": prob["tier"],
            "expected": prob["answer"],
            "baseline_correct": b_ok, "baseline_output": b_out,
            "adaptive_correct": a_ok, "adaptive_output": a_out,
            "loops": loops
        })

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for tier, scores in tiers.items():
        b_acc = sum(b for b, a in scores) / len(scores) * 100
        a_acc = sum(a for b, a in scores) / len(scores) * 100
        print(f"  {tier:6}  Baseline: {b_acc:.0f}%   Adaptive: {a_acc:.0f}%   "
              f"Delta: {'+' if a_acc-b_acc >= 0 else ''}{a_acc-b_acc:.0f}%")

    all_b = sum(b for t in tiers.values() for b, a in t)
    all_a = sum(a for t in tiers.values() for b, a in t)
    n     = sum(len(t) for t in tiers.values())
    print(f"  {'OVERALL':6}  Baseline: {all_b/n*100:.0f}%   "
          f"Adaptive: {all_a/n*100:.0f}%   "
          f"Delta: {'+' if (all_a-all_b)/n*100>=0 else ''}{(all_a-all_b)/n*100:.0f}%")

    with open("indist_adaptive_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: indist_adaptive_results.json")


if __name__ == "__main__":
    main()
