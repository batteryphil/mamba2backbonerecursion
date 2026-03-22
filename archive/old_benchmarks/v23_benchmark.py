"""
v21_benchmark.py — The Native ACT Inference Engine
Evaluates Mamba-130M v21 using perfectly native loop halting based on the <THINK> token.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel
import os

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_INFER_LOOPS = 10

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

    @property
    def weight(self) -> torch.Tensor:
        return self.base_weight + self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

# ── Recursive Mamba Wrapper ───────────────────────────────────────────────────
class RecursiveMamba130M(nn.Module):
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
            for attr in ("in_proj", "out_proj"):
                if hasattr(mx, attr):
                    setattr(mx, attr, LoRALinear(getattr(mx, attr), rank=lora_rank, alpha=ALPHA))

        self.step_emb = nn.Embedding(MAX_INFER_LOOPS, d_model)
        self.loop_norm = nn.RMSNorm(d_model)

    def forward(self, input_ids: torch.Tensor, max_loops_override: int = None) -> tuple:
        x = self.backbone.embedding(input_ids)
        residual = None

        for layer in self.backbone.layers[:6]:
            x, residual = layer(x, residual)

        base_features = x.clone()
        loops_taken = 0
        trajectory = []
        final_answer = "<THINK>"
        
        # ── V21 DYNAMIC POINTER MASKING ──
        vocab_size = self.lm_head.weight.shape[0]
        mask = torch.full((vocab_size,), float('-inf'), device=x.device)
        unique_input_ids = torch.unique(input_ids[0])
        allowed_indices = torch.cat([unique_input_ids, torch.tensor(ALLOWED_CORE_TOKENS, device=x.device)]).unique()
        mask[allowed_indices] = 0.0
        
        limit = max_loops_override if max_loops_override is not None else MAX_INFER_LOOPS
        for step_i in range(limit):
            # 1. Inject Clock Cycle
            step_vec = self.step_emb(torch.tensor(step_i, device=x.device))
            x = x + step_vec
            
            # 2. Forward pass through LoRA logic blocks
            for layer in self.top_layers:
                x, residual = layer(x, residual)
                
            # 3. Re-Anchor and Normalize
            x = x + base_features
            x = self.loop_norm(x)

            # 4. Decode to LM Head
            xn = self.norm(x, residual, prenorm=False)
            logits = self.lm_head(xn)
            
            # 5. DYNAMIC POINTER MASKING
            masked_logits = logits[0, -1, :] + mask
            
            # 6. Strict Greedy Decoding
            predicted_token_id = masked_logits.argmax().item()
            loops_taken += 1
            
            # 7. THE NATIVE HALT CONDITION
            if predicted_token_id == THINK_TOKEN_ID:
                trajectory.append("<THINK>")
                continue # Model decided it needs more time!
            else:
                # The model actively chose an answer!
                final_answer = tokenizer.decode([predicted_token_id]).strip()
                trajectory.append(final_answer)
                break 
                
        return final_answer, loops_taken, trajectory

# ── Load Model ────────────────────────────────────────────────────────────────
def load_model(weights_path):
    print("Loading base mamba2-130m...")
    base_model = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-130m", dtype=torch.float32, device=DEVICE)
    
    # Resizing token embeddings for <THINK>
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
    
    print("Wrapping with Native ACT head...")
    model = RecursiveMamba130M(base_model, lora_rank=8).to(DEVICE)
    model.eval()
    
    if os.path.exists(weights_path):
        print(f"Loading finetuned weights: {weights_path}")
        ckpt = torch.load(weights_path, map_location=DEVICE)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
    else:
        print(f"⚠️ Warning: Could not find {weights_path}, running untrained!")
        
    return model

# ── Evaluator ──────────────────────────────────────────────────────────────────
def run_benchmark(weights_path):
    model = load_model(weights_path)
    
    hard_data = [
        ("A is bigger than B, B is bigger than C, C is bigger than D, D is bigger than E. Who is smallest? Answer:", "E"),
        ("Wraith > Zephon > Hydra > Vela > Kael. Who is tallest? Answer:", "Wraith"),
        ("Box 1 is smaller than Box 2. Box 2 is smaller than Box 3. Box 3 is smaller than Box 4. Who is largest? Answer:", "Box 4"),
        ("Block X < Block Y. Block Y < Block Z. Block Z < Block W. Who is smallest? Answer:", "Block X"),
        ("Mercury is hotter than Mars. Mars is hotter than Earth. Earth is hotter than Neptune. Who is coldest? Answer:", "Neptune"),
    ]

    print("\n" + "="*80)
    print("  🚀 V21 NATIVE ACT BENCHMARK 🚀")
    print("="*80)

    for n_override in range(10, 0, -1):
        print(f"\n>>>> ABLATION TESTING MAX LOOPS: {n_override} <<<<")
        for (prompt, expected) in hard_data:
            ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                final_answer, loops, config_trace = model(ids, max_loops_override=n_override)
                
                is_correct = (final_answer.lower() == expected.lower())
                print(f"[{'✅' if is_correct else '❌'}] Loops: {loops} | {config_trace} (Exp: {expected})")

if __name__ == "__main__":
    ckpts = ["mamba2_finetuned_v23.pt"]
    for ckpt in ckpts:
        print(f"\n========================================")
        print(f"  EVALUATING {ckpt}")
        print(f"========================================")
        run_benchmark(ckpt)
