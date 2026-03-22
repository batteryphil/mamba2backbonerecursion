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
MAX_INFER_LOOPS = 4

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token

# V21: Add the <THINK> token
tokenizer.add_special_tokens({'additional_special_tokens': ['<THINK>']})
THINK_TOKEN_ID = tokenizer.convert_tokens_to_ids('<THINK>')
ALLOWED_CORE_TOKENS = [THINK_TOKEN_ID] + tokenizer.encode(" A B C D E X Y W Z", add_special_tokens=False)

# ── LoRA Linear ───────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.bias  = linear.bias
        d_out, d_in = linear.weight.shape
        self.register_buffer("base_weight", linear.weight.data.clone())
        self.lora_A = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(rank, d_in), a=5**0.5))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.scale  = alpha / rank

    @property
    def weight(self) -> torch.Tensor:
        return self.base_weight + self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

# ── JIT Fused MIMO Core ───────────────────────────────────────────────────────
@torch.jit.script
def fused_mamba3_mimo_core(
    x_in: torch.Tensor,
    real_state: torch.Tensor,
    imag_state: torch.Tensor,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    B_real: torch.Tensor,
    B_imag: torch.Tensor,
    C_real: torch.Tensor,
    C_imag: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B_r = B_real.unsqueeze(0).unsqueeze(0)
    B_i = B_imag.unsqueeze(0).unsqueeze(0)
    C_r = C_real.unsqueeze(0).unsqueeze(0)
    C_i = C_imag.unsqueeze(0).unsqueeze(0)
    
    bx_real = B_r * x_in
    bx_imag = B_i * x_in
    new_real = (cos_t * real_state - sin_t * imag_state) + bx_real
    new_imag = (sin_t * real_state + cos_t * imag_state) + bx_imag
    
    y_real = (C_r * new_real) - (C_i * new_imag)
    y_real_sum = y_real.sum(dim=-1)
    return y_real_sum, new_real, new_imag

# ── Recursive Mamba Wrapper ───────────────────────────────────────────────────
class Mamba3ReasoningBlock(nn.Module):
    def __init__(self, d_model: int, n_channels: int = 2, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.n_channels = n_channels
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, n_channels * d_model, bias=False)
        self.A_theta = nn.Parameter(torch.randn(n_channels, d_model, d_state) * 0.1)
        self.B_real = nn.Parameter(torch.randn(n_channels, d_model, d_state) * 0.1)
        self.B_imag = nn.Parameter(torch.randn(n_channels, d_model, d_state) * 0.1)
        self.C_real = nn.Parameter(torch.randn(n_channels, d_model, d_state) * 0.1)
        self.C_imag = nn.Parameter(torch.randn(n_channels, d_model, d_state) * 0.1)
        self.out_proj = nn.Linear(n_channels * d_model, d_model, bias=False)
        self.mixer_norm = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor, real_state: torch.Tensor = None, imag_state: torch.Tensor = None, cos_t: torch.Tensor = None, sin_t: torch.Tensor = None) -> tuple:
        B, L, _ = x.shape
        x_in = self.in_proj(x)
        x_in = x_in.view(B, L, self.n_channels, self.d_model).unsqueeze(-1)
        
        if real_state is None:
            real_state = torch.zeros(B, L, self.n_channels, self.d_model, self.d_state, device=x.device)
            imag_state = torch.zeros(B, L, self.n_channels, self.d_model, self.d_state, device=x.device)
            
        if cos_t is None or sin_t is None:
            cos_t = torch.cos(self.A_theta).unsqueeze(0).unsqueeze(0)
            sin_t = torch.sin(self.A_theta).unsqueeze(0).unsqueeze(0)
            
        y_real_sum, new_real, new_imag = fused_mamba3_mimo_core(
            x_in, real_state, imag_state, 
            cos_t, sin_t, self.B_real, self.B_imag, self.C_real, self.C_imag
        )
        
        y_flat = y_real_sum.view(B, L, self.n_channels * self.d_model)
        out = self.out_proj(y_flat)
        out = self.mixer_norm(out)
        return x + out, new_real, new_imag

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
            for attr in ("in_proj", "x_proj", "dt_proj"):
                if hasattr(mx, attr):
                    setattr(mx, attr, LoRALinear(getattr(mx, attr), rank=lora_rank, alpha=ALPHA))

        self.step_emb = nn.Embedding(MAX_INFER_LOOPS, d_model)
        self.loop_norm = nn.RMSNorm(d_model)
        self.mamba3_core = Mamba3ReasoningBlock(d_model=d_model, n_channels=2, d_state=16)

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
        real_state = None
        imag_state = None
        
        cos_t = torch.cos(self.mamba3_core.A_theta).unsqueeze(0).unsqueeze(0)
        sin_t = torch.sin(self.mamba3_core.A_theta).unsqueeze(0).unsqueeze(0)
        
        for step_i in range(limit):
            # 1. Inject Clock Cycle
            step_vec = self.step_emb(torch.tensor(step_i, device=x.device))
            x = x + step_vec
            
            # 2. Forward pass through LoRA logic blocks
            for layer in self.top_layers:
                x, residual = layer(x, residual)
                
            x, real_state, imag_state = self.mamba3_core(x, real_state, imag_state, cos_t=cos_t, sin_t=sin_t)
                
            # 3. Re-Anchor and Normalize
            x = self.loop_norm(x)

            # 4. Decode to LM Head
            xn = self.norm(x, residual, prenorm=False)
            logits = self.lm_head(xn)
            
            # 5. DYNAMIC POINTER MASKING
            masked_logits = logits[0, -1, :] + mask
            
            # 6. Latent Logic Probe
            probs = F.softmax(masked_logits, dim=-1)
            predicted_token_id = masked_logits.argmax().item()
            conf = probs[predicted_token_id].item() * 100.0
            token_str = tokenizer.decode([predicted_token_id])
            
            loops_taken += 1
            trajectory.append(f"'{token_str}' ({conf:.1f}%)")
            
            # 7. THE NATIVE HALT CONDITION
            if predicted_token_id == THINK_TOKEN_ID:
                continue # Model decided it needs more time!
            else:
                # The model actively chose an answer!
                final_answer = token_str.strip()
                break 
                
        return final_answer, loops_taken, trajectory

# ── Load Model ────────────────────────────────────────────────────────────────
def load_model(weights_path):
    print("Loading base mamba-130m...")
    base_model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m", dtype=torch.float32, device=DEVICE)
    
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
        
        # Safely align step_emb dimensions natively
        if "step_emb.weight" in state:
            emb_w = state["step_emb.weight"]
            if emb_w.shape[0] < MAX_INFER_LOOPS:
                new_emb = torch.zeros(MAX_INFER_LOOPS, emb_w.shape[1], device=DEVICE)
                new_emb[:emb_w.shape[0]] = emb_w
                state["step_emb.weight"] = new_emb
                
        model.load_state_dict(state, strict=False)
    else:
        print(f"⚠️ Warning: Could not find {weights_path}, running untrained!")
        
    return model

# ── Evaluator ──────────────────────────────────────────────────────────────────
def run_benchmark(weights_path):
    model = load_model(weights_path)
    
    hard_data = [
        # Base 1-hop
        ("A is bigger than B. Who is smallest? Answer:", "B"),
        # Test 1: "Dirty Fuel" Injection (Noise Resistance)
        ("A is bigger than B. The sky in Arkansas is blue today, and my Silverado needs an oil change. B is bigger than C. Who is smallest? Answer:", "C"),
        # Test 2: "Over-Rev" Zero-Shot Extrapolation 4-hop
        ("A is bigger than B, B is bigger than C, C is bigger than D, D is bigger than E. Who is smallest? Answer:", "E"),
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
    ckpts = ["mamba3_finetuned_v24_MaxN_2.pt"]
    for ckpt in ckpts:
        print(f"\n========================================")
        print(f"  EVALUATING {ckpt}")
        print(f"========================================")
        run_benchmark(ckpt)
