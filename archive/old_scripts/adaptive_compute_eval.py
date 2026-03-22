"""
adaptive_compute_eval.py — Adaptive Computation / Dynamic Halting Benchmark
Evaluates Mamba-130M v17 on Test A (Lobotomized) and Test B (Unleashed).

Test A: INFER_MAX_LOOPS = 0. Forces 1 pass of the reasoning layers.
Test B: Unleashed. Dynamic loops stopping at CONFIDENCE_THR = 0.85
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel
import json
import random
import os

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS     = "mamba130m_finetuned_v20_step2800.pt"

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token

# V21: Add the <THINK> token
tokenizer.add_special_tokens({'additional_special_tokens': ['<THINK>']})
THINK_TOKEN_ID = tokenizer.convert_tokens_to_ids('<THINK>')
ALLOWED_CORE_TOKENS = [tokenizer.eos_token_id, THINK_TOKEN_ID] + tokenizer.encode(" A B C D", add_special_tokens=False)

# ── LoRA Linear ───────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.bias  = linear.bias
        d_out, d_in = linear.weight.shape
        self.register_buffer("base_weight", linear.weight.data.clone())
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.scale  = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A)

    @property
    def weight(self) -> torch.Tensor:
        return self.base_weight + self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

# ── Recursive Mamba Wrapper ───────────────────────────────────────────────────
class RecursiveMamba130M(nn.Module):
    MAX_LOOPS: int = 10

    def __init__(self, backbone_model: MambaLMHeadModel, lora_rank: int = 8):
        super().__init__()
        self.backbone   = backbone_model.backbone
        self.lm_head    = backbone_model.lm_head
        self.top_layers = nn.ModuleList([backbone_model.backbone.layers[i] for i in range(6, 24)])
        self.norm       = backbone_model.backbone.norm_f
        d_model         = backbone_model.backbone.embedding.embedding_dim

        ALPHA = lora_rank * 2.0
        for layer in self.top_layers:
            mx = layer.mixer
            for attr in ("in_proj", "x_proj", "dt_proj"):
                if hasattr(mx, attr):
                    setattr(mx, attr, LoRALinear(getattr(mx, attr), rank=lora_rank, alpha=ALPHA))

        self.step_emb = nn.Embedding(self.MAX_LOOPS, d_model)
        nn.init.normal_(self.step_emb.weight, std=0.01)
        self.loop_norm = nn.RMSNorm(d_model)

    def forward(self, input_ids: torch.Tensor, lobotomized=False) -> tuple:
        x = self.backbone.embedding(input_ids)
        residual = None

        for layer in self.backbone.layers[:6]:
            x, residual = layer(x, residual)

        base_features = x.clone()
        loops_taken = self.MAX_LOOPS
        
        # Test A: Lobotomized (exactly 1 loop iteration: L0)
        # We simulate this by stopping immediately after loop step 0.
        
        max_limit = 1 if lobotomized else self.MAX_LOOPS
        
        # ── V21: Dynamic Pointer Masking (Blueprint 2) ──
        # Build mask once outside loop since input doesn't change
        vocab_size = len(tokenizer)
        mask = torch.full((vocab_size,), float('-inf'), device=x.device)
        unique_input_ids = torch.unique(input_ids[0])
        allowed_indices = torch.cat([unique_input_ids, torch.tensor(ALLOWED_CORE_TOKENS, device=x.device)]).unique()
        mask[allowed_indices] = 0.0
        
        trace = []
        for step_i in range(max_limit):
            step_vec = self.step_emb(torch.tensor(step_i, device=x.device))
            x = x + step_vec
            for layer in self.top_layers:
                x, residual = layer(x, residual)
            x = x + base_features
            x = self.loop_norm(x)

            logits_tmp = self.lm_head(self.norm(x, residual, prenorm=False))
            
            # Apply dynamic pointer mask
            logits_tmp[0, -1, :] = logits_tmp[0, -1, :] + mask
            
            p = torch.softmax(logits_tmp[0, -1, :], dim=-1)
            
            max_prob = p.max().item()
            entropy  = -(p * (p + 1e-12).log()).sum().item()
            top_tok_id = p.argmax().item()
            trace.append((step_i+1, max_prob, tokenizer.decode([top_tok_id]).strip()))

            # ── V21: The Native ACT / Natural Halting (Blueprint 1) ──
            # Stop organically when the model decides to decode anything other than <THINK>
            if not lobotomized and top_tok_id != THINK_TOKEN_ID:
                loops_taken = step_i + 1
                break
                
        if lobotomized:
            loops_taken = 1

        x = self.norm(x, residual, prenorm=False)
        return self.lm_head(x), loops_taken, trace

# ── Load Model ────────────────────────────────────────────────────────────────
def load_model():
    print("Loading base mamba-130m...")
    base_model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m", dtype=torch.float32, device=DEVICE)
    
    # V21: Manually resize model embeddings for the <THINK> token
    new_vocab_size = len(tokenizer)
    old_vocab_size = base_model.backbone.embedding.weight.shape[0]
    d_model = base_model.backbone.embedding.embedding_dim

    if new_vocab_size > old_vocab_size:
        new_emb = nn.Embedding(new_vocab_size, d_model)
        new_emb.weight.data[:old_vocab_size] = base_model.backbone.embedding.weight.data
        base_model.backbone.embedding = new_emb
        
        new_head = nn.Linear(d_model, new_vocab_size, bias=False)
        new_head.weight.data[:old_vocab_size] = base_model.lm_head.weight.data
        base_model.lm_head = new_head
        
    for p in base_model.parameters(): p.requires_grad = False
    
    print("Wrapping with Recursive reasoning head...")
    model = RecursiveMamba130M(base_model, lora_rank=8).to(DEVICE)
    model.eval()
    
    if os.path.exists(WEIGHTS):
        print(f"Loading finetuned weights: {WEIGHTS}")
        ckpt = torch.load(WEIGHTS, map_location=DEVICE)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
    else:
        print(f"⚠️ Warning: Could not find {WEIGHTS}, running untrained!")
        
    return model

# ── Evaluator ──────────────────────────────────────────────────────────────────
def evaluate():
    model = load_model()
    
    # 1. We will generate our own test data to guarantee logical depths
    
    # 1-hop (Easy)
    easy_data = [
        ("Alice is taller than Bob. Who is taller? Answer:", "Alice"),
        ("X > Y > Z in height. Who is tallest? Answer:", "X"),
        ("Mars is colder than Venus. Which planet is warmer? Answer:", "Venus"),
        ("Package 1 is heavier than Package 2. Who is lighter? Answer:", "Package 2"),
        ("Tatooine > Rigel in distance. Who is closer? Answer:", "Rigel"),
    ]
    
    # 4-hop (Hard)
    hard_data = [
        ("A is bigger than B, B is bigger than C, C is bigger than D, D is bigger than E. Who is smallest? Answer:", "E"),
        ("Wraith > Zephon > Hydra > Vela > Kael. Who is tallest? Answer:", "Wraith"),
        ("Box 1 is smaller than Box 2. Box 2 is smaller than Box 3. Box 3 is smaller than Box 4. Who is largest? Answer:", "Box 4"),
        ("Block X < Block Y. Block Y < Block Z. Block Z < Block W. Who is smallest? Answer:", "Block X"),
        ("Mercury is hotter than Mars. Mars is hotter than Earth. Earth is hotter than Neptune. Who is coldest? Answer:", "Neptune"),
    ]

    print("\n" + "="*80)
    print("  ADAPTIVE COMPUTATION & LOGIC EVALUATION")
    print("="*80)

    # ────────────────────────── TEST A: LOBOTOMIZED ───────────────────────────
    print("\n▶ TEST A: LOBOTOMIZED (INFER_MAX_LOOPS = 1)")
    print("  (Forcing directly to 1 pass without confidence metrics)")
    
    test_a_correct = 0
    total = len(easy_data) + len(hard_data)
    
    for (prompt, answer) in easy_data + hard_data:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits, loops, trace = model(ids, lobotomized=True)
            pred = tokenizer.decode([logits[0, -1, :].argmax().item()]).strip()
            
            if pred.lower() == answer.lower(): test_a_correct += 1

    print(f"  Accuracy: {test_a_correct}/{total} ({(test_a_correct/total)*100:.1f}%)")

    # ────────────────────────── TEST B: UNLEASHED ─────────────────────────────
    print("\n▶ TEST B: UNLEASHED (Native Halting until token != <THINK>)")
    
    test_b_correct = 0
    
    easy_loops = []
    print("\n  [Easy 1-Hop Questions]")
    for (prompt, answer) in easy_data:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits, loops, trace = model(ids, lobotomized=False)
            pred = tokenizer.decode([logits[0, -1, :].argmax().item()]).strip()
            
            easy_loops.append(loops)
            is_correct = (pred.lower() == answer.lower())
            if is_correct: test_b_correct += 1
            print(f"    {'✅' if is_correct else '❌'} [{loops} loops] {prompt[-40:]!r} → {pred!r} (expected {answer})")

    hard_loops = []
    print("\n  [Hard 4-Hop Questions]")
    for (prompt, answer) in hard_data:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits, loops, trace = model(ids, lobotomized=False)
            pred = tokenizer.decode([logits[0, -1, :].argmax().item()]).strip()
            
            hard_loops.append(loops)
            is_correct = (pred.lower() == answer.lower())
            if is_correct: test_b_correct += 1
            print(f"    {'✅' if is_correct else '❌'} [{loops} loops] {prompt[-40:]!r} → {pred!r} (expected {answer})")

    avg_easy = sum(easy_loops)/len(easy_loops)
    avg_hard = sum(hard_loops)/len(hard_loops)

    print("\n" + "="*80)
    print("  RESULTS & DELTAS")
    print("="*80)
    print(f"  Test A (Lobotomized) Accuracy:   {(test_a_correct/total)*100:.1f}%")
    print(f"  Test B (Unleashed) Accuracy:     {(test_b_correct/total)*100:.1f}%")
    
    delta = ((test_b_correct - test_a_correct) / total)*100
    print(f"  Delta:                           +{delta:.1f}%")

    print("\n  Adaptive Computation Time (ACT) Correlation:")
    print(f"  Avg Loops for Easy 1-Hop:  {avg_easy:.1f}")
    print(f"  Avg Loops for Hard 4-Hop:  {avg_hard:.1f}")
    
    if avg_hard > avg_easy + 0.5:
        print("  ✅ PASS: Significant latency correlation with logical depth! Cognitive mapping successful.")
    else:
        print("  ⚠️ FAIL: Loop count does not scale effectively with logical depth.")
    print("="*80 + "\n")

if __name__ == "__main__":
    evaluate()
