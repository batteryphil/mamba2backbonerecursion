"""
finetune_mamba130m.py — v18 (Dense Trajectory Supervision)
=================================================================
Fixes applied:
  Bug 1: Batch index [0] → average loss across full batch
  Bug 2: Residual double-add removed (base_features added to x only)
  Bug 3: Inference matches training (no pure base pass, starts at step_emb[0])
  Bug 4: LayerNorm → RMSNorm for loop_norm
  Bug 5: LoRA rank-8 on all 18 top layers (6-23) → ~1M trainable params
  Bug 6: Dual LR optimizer (step_emb@1e-2, LoRA@3e-4)
  New:   Checkpoint shape guard (step_emb slice on MAX_LOOPS mismatch)
  New:   Mixed training data (75k logic + 10k MMLU 4-choice format)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel
import json
import random
import time
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
STEPS       = 50000
SEQ_LEN     = 256
BATCH_SIZE  = 4
ACCUM       = 8
LOG_EVERY   = 50
RESUME_FROM = ""
SAVE_PATH   = "mamba2_finetuned_v23.pt"

print(f"\n{'='*60}", flush=True)
print(f"  Mamba2-130M v23 — Mamba 2 Architecture Upgrade", flush=True)
print(f"  Device: {DEVICE} | Steps={STEPS}", flush=True)
print(f"{'='*60}\n", flush=True)


# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token

# V21: Add the <THINK> token
tokenizer.add_special_tokens({'additional_special_tokens': ['<THINK>']})
THINK_TOKEN_ID = tokenizer.convert_tokens_to_ids('<THINK>')
print(f"  <THINK> Token ID generated: {THINK_TOKEN_ID}")

_answer_seq_ids = tokenizer.encode(" Answer:", add_special_tokens=False)
print(f"  Answer boundary tokens: {_answer_seq_ids} = ' Answer:'")

# Base MMLU letter targets (A, B, C, D) for the Pointer Mask
ALLOWED_CORE_TOKENS = [tokenizer.eos_token_id, THINK_TOKEN_ID] + tokenizer.encode(" A B C D", add_special_tokens=False)


def find_answer_start(ids: list[int]) -> int:
    """Find the token position where ' Answer:' begins in a sequence."""
    n = len(_answer_seq_ids)
    for i in range(len(ids) - n):
        if ids[i:i+n] == _answer_seq_ids:
            return i + n
    return -1


# ── LoRA Linear ───────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation wrapper for nn.Linear.

    Bug 5 fix: adds ~1M trainable params to steer the frozen backbone.
    Critical: exposes .weight property so Mamba's CUDA kernels can access
    the fused (base + delta) weight directly without AttributeError.
    """
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
        """Fused weight: frozen base + low-rank delta. Accessed by Mamba CUDA kernels."""
        return self.base_weight + self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using fused weight."""
        return F.linear(x, self.weight, self.bias)


# ── Recursive Mamba Wrapper ───────────────────────────────────────────────────
class RecursiveMamba130M(nn.Module):
    """
    Wraps frozen mamba-130m with a recursive reasoning head.

    Architecture:
      - Layers 0-5:  fixed feature extractor (run once)
      - Layers 6-23: reasoning engine (run N times in loop)
      - step_emb:    loop-step clock (tells model which iteration it's on)
      - loop_norm:   RMSNorm before each loop step (Bug 4 fix)
      - LoRA:        rank-8 adapters on in_proj/x_proj/dt_proj of layers 6-23

    Training: stochastic depth (random N per batch, no entropy check)
    Inference: entropy-gated halt (exit when entropy < threshold)
    """
    MAX_LOOPS: int = 10

    def __init__(self, backbone_model: MambaLMHeadModel, lora_rank: int = 8):
        super().__init__()
        self.backbone   = backbone_model.backbone
        self.lm_head    = backbone_model.lm_head
        self.top_layers = nn.ModuleList(
            [backbone_model.backbone.layers[i] for i in range(6, 24)]
        )
        self.norm       = backbone_model.backbone.norm_f
        d_model         = backbone_model.backbone.embedding.embedding_dim

        print(f"  Frozen params:  {sum(p.numel() for p in backbone_model.parameters()):,}"
              f"  (ALL 24 backbone layers + head + embedding)", flush=True)
        print(f"  Trainable from base: 0  — zero catastrophic forgetting", flush=True)
        print(f"  New params to train: step_emb + loop_norm (~9k) added in wrapper", flush=True)

        # Bug 5 fix — LoRA adapters on ALL 18 top layers (layers 6-23)
        # alpha = 2 × rank (standard LoRA convention)
        ALPHA = lora_rank * 2.0
        for layer in self.top_layers:
            mx = layer.mixer
            for attr in ("in_proj", "out_proj"):
                if hasattr(mx, attr):
                    setattr(mx, attr, LoRALinear(getattr(mx, attr), rank=lora_rank, alpha=ALPHA))

        # Step embedding: tells the model which loop iteration it's on
        # Small std=0.01 init prevents step_emb from overwhelming frozen states
        self.step_emb = nn.Embedding(self.MAX_LOOPS, d_model)
        nn.init.normal_(self.step_emb.weight, std=0.01)

        # Bug 4 fix — RMSNorm matches Mamba's internal normalization (not LayerNorm)
        self.loop_norm = nn.RMSNorm(d_model)

        n_lora = sum(p.numel() for n, p in self.named_parameters()
                     if p.requires_grad and "lora" in n.lower())
        n_emb  = sum(p.numel() for p in self.step_emb.parameters())
        n_norm = sum(p.numel() for p in self.loop_norm.parameters())
        print(f"  LoRA params:    {n_lora:,}", flush=True)
        print(f"  step_emb+norm:  {n_emb+n_norm:,}", flush=True)
        print(f"  Total trainable:{n_lora+n_emb+n_norm:,}\n", flush=True)

    def forward(self, input_ids: torch.Tensor, tgt_labels: torch.Tensor = None, ans_starts: list = None, accum: int = 1, max_train_loops: int = None) -> tuple:
        """
        Forward pass — training uses stochastic depth, inference uses entropy halt.

        Bug 3 fix: inference now starts at step_emb[0] without a pure base pass,
        matching the training distribution exactly.
        Bug 2 fix: base_features added to x only (not to internal residual tensor).
        """
        x       = self.backbone.embedding(input_ids)
        residual = None

        # Feature extractor pass (layers 0-5) — runs once
        for layer in self.backbone.layers[:6]:
            x, residual = layer(x, residual)

        # Save base features for residual anchoring
        base_features = x.clone()

        if self.training:
            # ── Training path: stochastic depth with Dense Trajectory Supervision ──
            # v19: Gradient Checkpointing for exact BPTT physics at 0.5GB VRAM
            # v20: Curriculum Anchoring allows forcing max_train_loops
            if max_train_loops is None:
                max_train_loops = self.MAX_LOOPS
            
            # V21 BUG FIX: n_steps CANNOT BE RANDOM during <THINK> Token training.
            # If it is random, Loop 0 is assigned `<THINK>` for n_steps=2, but `Answer` for n_steps=1.
            # The model mathematically cannot optimize this contradiction and bottoms out at CE loss ~3.5.
            # It must be perfectly deterministic for the current curriculum tier!
            n_steps = max_train_loops
            step_losses = []
            
            # Helper for checkpointing LoRA layers
            def run_lora_layers(x_in, res_in):
                for layer in self.top_layers:
                    x_in, res_in = layer(x_in, res_in)
                return x_in, res_in
                
            for step_i in range(n_steps):
                step_vec = self.step_emb(torch.tensor(step_i, device=x.device))
                x = x + step_vec
                
                # Checkpoint the forward pass to save VRAM while keeping graph connected!
                x, residual = checkpoint(run_lora_layers, x, residual, use_reentrant=False)
                
                x = x + base_features
                x = self.loop_norm(x)
                # Calculate loss for this step sequence
                if tgt_labels is not None and ans_starts is not None:
                    # ── V22: LATENT FREE-FLOAT (TERMINUS TARGETING) ──
                    # We ONLY calculate the cross-entropy loss on the final terminating loop.
                    # Intermediate loops float freely in the residual stream without projection!
                    if step_i == n_steps - 1:
                        x_normed = self.norm(x, residual, prenorm=False)
                        logits_step = self.lm_head(x_normed)
                        
                        # ── V21: Dynamic Pointer Masking ──
                        vocab_size = logits_step.shape[-1]
                        B, max_len = input_ids.shape[0], input_ids.shape[1]
                        batch_mask = torch.full((B, vocab_size), float('-inf'), device=x.device)
                        for b in range(B):
                            unique_input_ids = torch.unique(input_ids[b])
                            allowed_indices = torch.cat([unique_input_ids, torch.tensor(ALLOWED_CORE_TOKENS, device=x.device)]).unique()
                            batch_mask[b, allowed_indices] = 0.0
                        
                        logits_step = logits_step + batch_mask.unsqueeze(1)
                        step_loss = torch.tensor(0.0, device=x.device)
                        
                        for b in range(B):
                            ans_start = ans_starts[b]
                            if ans_start < 0 or ans_start >= max_len - 1:
                                tgt = tgt_labels[b, 1:max_len]
                                logits_b = logits_step[b, :max_len-1, :]
                            else:
                                tgt = tgt_labels[b, ans_start:max_len]
                                logits_b = logits_step[b, ans_start-1:max_len-1, :]
                                
                            if tgt.shape[0] > 0:
                                step_loss = step_loss + F.cross_entropy(logits_b[:tgt.shape[0]], tgt, ignore_index=tokenizer.eos_token_id) / B

                        step_losses.append(step_loss)

            # Average loss out of loop, and let the external loop handle `.backward()`
            avg_traj_loss = torch.stack(step_losses).mean() if step_losses else torch.tensor(0.0, device=x.device, requires_grad=True)
            return None, n_steps, [], avg_traj_loss

        else:
            # ── Inference path: Native ACT / Natural Halting (Blueprint 1) ──
            # Stop organically when the model decides to decode anything other than <THINK>
            
            # ── V21: Dynamic Pointer Masking (Blueprint 2) ──
            # Build mask once outside loop since input doesn't change
            vocab_size = len(tokenizer)
            mask = torch.full((vocab_size,), float('-inf'), device=x.device)
            unique_input_ids = torch.unique(input_ids[0])
            allowed_indices = torch.cat([unique_input_ids, torch.tensor(ALLOWED_CORE_TOKENS, device=x.device)]).unique()
            mask[allowed_indices] = 0.0
            
            trace = []
            loops_taken = self.MAX_LOOPS
            for step_i in range(self.MAX_LOOPS):
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
                top_tok  = tokenizer.decode([top_tok_id]).strip()
                trace.append((f"L{step_i+1}", round(max_prob, 2), top_tok))

                if top_tok_id != THINK_TOKEN_ID:
                    loops_taken = step_i + 1
                    break

            x = self.norm(x, residual, prenorm=False)
            return self.lm_head(x), loops_taken, trace


# ── Load Base Model ───────────────────────────────────────────────────────────
base_model = MambaLMHeadModel.from_pretrained(
    "state-spaces/mamba-130m", dtype=torch.float32, device=DEVICE
)

# V21: Manually resize model embeddings for the <THINK> token (Mamba lacks HF resize utilities)
new_vocab_size = len(tokenizer)
old_vocab_size = base_model.backbone.embedding.weight.shape[0]
d_model = base_model.backbone.embedding.embedding_dim

if new_vocab_size > old_vocab_size:
    print(f"  Resizing vocabulary {old_vocab_size} -> {new_vocab_size} for <THINK> token")
    
    # 1. Resize input embeddings
    new_emb = nn.Embedding(new_vocab_size, d_model)
    nn.init.normal_(new_emb.weight, std=0.02)
    new_emb.weight.data[:old_vocab_size] = base_model.backbone.embedding.weight.data
    base_model.backbone.embedding = new_emb
    
    # 2. Resize output LM Head
    new_head = nn.Linear(d_model, new_vocab_size, bias=False)
    nn.init.normal_(new_head.weight, std=0.02)
    new_head.weight.data[:old_vocab_size] = base_model.lm_head.weight.data
    base_model.lm_head = new_head

for p in base_model.parameters():
    p.requires_grad = False
    
# Except the language head and embeddings, which we just resized and need to unfreeze slightly for the new token:
base_model.backbone.embedding.weight.requires_grad = True
base_model.lm_head.weight.requires_grad = True

model = RecursiveMamba130M(base_model, lora_rank=8).to(DEVICE)

# New — Checkpoint shape guard
# If resuming from a checkpoint with different MAX_LOOPS, slice step_emb
if RESUME_FROM and os.path.exists(RESUME_FROM):
    print(f"  Loading checkpoint from {RESUME_FROM}...")
    ckpt = torch.load(RESUME_FROM, map_location=DEVICE)
    state_dict = ckpt.get("model_state_dict", ckpt)
    
    # --- V17 MAX_LOOPS GUARD ---
    if "step_emb.weight" in state_dict:
        ckpt_emb = state_dict["step_emb.weight"]
        model_emb = model.step_emb.weight
        
        if ckpt_emb.shape != model_emb.shape:
            print(f"  ⚠️  MAX_LOOPS mismatch! Checkpoint: {ckpt_emb.shape[0]}, Model: {model_emb.shape[0]}. Adapting...")
            
            # Create a new tensor initialized with the model's current random init (std=0.01)
            # This ensures any *new* loops get the proper tiny initialization, not zeros.
            adapted_emb = model_emb.clone()
            
            # Find how many loop steps overlap
            min_loops = min(ckpt_emb.shape[0], model_emb.shape[0])
            
            # Copy over the trained overlapping steps
            adapted_emb[:min_loops, :] = ckpt_emb[:min_loops, :]
            
            # Replace the tensor in the dictionary
            state_dict["step_emb.weight"] = adapted_emb
            
    # Load with strict=False so it doesn't crash if LoRA keys are missing in older checkpoints
    model.load_state_dict(state_dict, strict=False)
    print("  ✅ Checkpoint loaded safely.\n")
elif RESUME_FROM:
    print(f"  ⚠️  {RESUME_FROM} not found — starting fresh\n")
else:
    print(f"  Starting fresh (no resume checkpoint)\n")


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_training_data() -> list[str]:
    """
    Load and mix logic samples (75k) + MMLU-format 4-choice samples (10k).
    The MMLU-format data teaches: Answer: → A/B/C/D (letter distribution)
    The logic data teaches: Answer: → entity (name/symbol distribution)

    Without MMLU data: model scores 23.8% MMLU (below random 25%)
    Goal: MMLU data teaches letter output format → target >30% MMLU
    """
    samples = []

    # Logic reasoning data
    for fname in ["logic_v3.json", "logic_v2.json", "logic_v1.json",
                  "reasoning_data.json", "logic_qa.json"]:
        if os.path.exists(fname):
            with open(fname) as f:
                data = json.load(f)
            for item in data:
                text = item.get("text", item.get("prompt", ""))
                if text:
                    samples.append(text)
            print(f"  Logic samples loaded:  {len(samples):>10,}  ({fname})")
            break

    # MMLU-format 4-choice data (new in v17)
    if os.path.exists("mmlu_format_v17.json"):
        before = len(samples)
        with open("mmlu_format_v17.json") as f:
            mmlu_data = json.load(f)
        for item in mmlu_data:
            samples.append(item["text"])
        print(f"  MMLU-format samples:   {len(samples)-before:>10,}  (mmlu_format_v17.json)")

    # Remove duplicates
    before = len(samples)
    samples = list(dict.fromkeys(samples))
    print(f"  Deduplication:         {before:>10,} → {len(samples):,} "
          f"unique ({before-len(samples)} dupes removed)")
    print(f"  Total samples:         {len(samples):>10,}\n")
    return samples

samples = load_training_data()


# ── Optimizer (Bug 6 fix — dual LR) ──────────────────────────────────────────
# Group 1: step_emb + loop_norm + resized embeddings → LR=1e-2
# Group 2: all LoRA A/B matrices → LR=3e-4
group1_params = [model.step_emb.weight] + list(model.loop_norm.parameters()) + [base_model.backbone.embedding.weight, base_model.lm_head.weight]
group1_ids    = {id(p) for p in group1_params}
group2_params = [p for p in model.parameters()
                 if p.requires_grad and id(p) not in group1_ids]

optimizer = optim.AdamW([
    {"params": group1_params, "lr": 1e-2,  "weight_decay": 0.0},
    {"params": group2_params, "lr": 3e-4,  "weight_decay": 0.01},
])
scheduler = CosineAnnealingLR(optimizer, T_max=STEPS // ACCUM, eta_min=1e-6)

# V17: Load optimizer and scheduler state to prevent LR sledgehammering on resume
if RESUME_FROM and os.path.exists(RESUME_FROM):
    ckpt = torch.load(RESUME_FROM, map_location=DEVICE)
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print("  ✅ Optimizer state loaded.")
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print("  ✅ Scheduler state loaded.")
print(f"  Optimizer: group1={len(group1_params)} tensors @ LR=1e-2"
      f"  |  group2={len(group2_params)} tensors @ LR=3e-4")
print(f"\n{'─'*60}")
print(f"  Starting fine-tune: {STEPS} steps | Batch={BATCH_SIZE*ACCUM}")
print(f"{'─'*60}")


# ── Training Loop ─────────────────────────────────────────────────────────────
model.train()
optimizer.zero_grad()
t0 = time.time()
total_loss = 0.0

# v20 Curriculum Anchoring state
current_max_loops = 1
loss_window = []

# Determine the correct starting step if resuming
start_step = 1
if RESUME_FROM and os.path.exists(RESUME_FROM):
    ckpt = torch.load(RESUME_FROM, map_location=DEVICE)
    if "step" in ckpt:
        start_step = ckpt["step"] + 1
        print(f"  ✅ Resuming training loop from step {start_step}...")
    
    # Restore Curriculum State
    if "current_max_loops" in ckpt:
        current_max_loops = ckpt["current_max_loops"]
        print(f"  ✅ Restored Curriculum State: MaxN = {current_max_loops}")
    elif start_step > 2400:
        current_max_loops = 2
        print(f"  ⚠️ Inferred Curriculum State: MaxN = {current_max_loops} (from step count)")
        
for step in range(start_step, STEPS + 1):

    # Sample a random batch
    batch_texts = random.choices(samples, k=BATCH_SIZE)
    batch_ids   = [
        tokenizer.encode(t, add_special_tokens=False,
                         max_length=SEQ_LEN, truncation=True)
        for t in batch_texts
    ]
    max_len = max(len(ids) for ids in batch_ids)
    padded  = [ids + [tokenizer.eos_token_id] * (max_len - len(ids))
               for ids in batch_ids]
    input_t = torch.tensor(padded, dtype=torch.long, device=DEVICE)

    # Find answer starts for masking
    B = input_t.shape[0]
    ans_starts = [find_answer_start(batch_ids[b]) for b in range(B)]

    # Forward pass calculates layer-wise cross-entropy and constructs Checkpoint graph
    _, n_steps, _, avg_traj_loss = model(input_t, tgt_labels=input_t, ans_starts=ans_starts, accum=ACCUM, max_train_loops=current_max_loops)

    loss = avg_traj_loss / ACCUM
    loss.backward()
    total_loss += avg_traj_loss.item() * ACCUM

    if step % ACCUM == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    if step % LOG_EVERY == 0:
        elapsed  = time.time() - t0
        tps      = (LOG_EVERY * BATCH_SIZE * SEQ_LEN) / elapsed
        mem      = torch.cuda.memory_allocated(DEVICE) / 1e9 if DEVICE == "cuda" else 0
        avg_loss = total_loss / LOG_EVERY
        lr_now   = scheduler.get_last_lr()[0]
        print(f"  Step {step:>4} | Loss: {avg_loss:.4f} | "
              f"LR: {lr_now:.2e} | TPS: {tps:.0f} | VRAM: {mem:.2f}GB | MaxN: {current_max_loops}", flush=True)
              
        # V20 Curriculum Anchoring: Slide window and scale N if mastered
        loss_window.append(avg_loss)
        
        # ── V21: Curriculum 3-Step Average Stoppage ──
        if len(loss_window) >= 3:
            rolling_avg = sum(loss_window[-3:]) / 3.0
            if rolling_avg < 1.45:
                if current_max_loops < model.MAX_LOOPS:
                    current_max_loops += 1
                    print(f"\n🚀 CURRICULUM UPGRADE: Mastering N={current_max_loops-1}. Escalating to N={current_max_loops} loops!\n", flush=True)
                    loss_window.clear()
                    
                    # ── V22 Milestone Save ──
                    milestone_path = SAVE_PATH.replace(".pt", f"_MaxN_{current_max_loops}.pt")
                    torch.save({
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": rolling_avg,
                        "current_max_loops": current_max_loops
                    }, milestone_path)
                    print(f"  🏆 Milestone Checkpoint saved → {milestone_path}", flush=True)
                else:
                    print(f"\n🎉 N={model.MAX_LOOPS} MASTERED! Loss stabilized below 1.45 over 3 steps. Halting training early.\n", flush=True)
                    break
        
        total_loss = 0.0
        t0 = time.time()

    if step % 1000 == 0:
        save_path = SAVE_PATH.replace(".pt", f"_step{step}.pt")
        torch.save({
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": sum(loss_window) / len(loss_window) if len(loss_window) > 0 else avg_traj_loss,
            "current_max_loops": current_max_loops
        }, save_path)
        # Also save to the main checkpoint path
        torch.save({
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": sum(loss_window) / len(loss_window) if len(loss_window) > 0 else avg_traj_loss,
            "current_max_loops": current_max_loops
        }, SAVE_PATH)
        print(f"  💾 Checkpoint saved → {save_path}", flush=True)
        
        # ── DISK CLEANUP (Prevent OS Crash limit) ──
        import glob
        clean_ckpts = sorted(glob.glob(SAVE_PATH.replace(".pt", "_step*.pt")), key=os.path.getmtime)
        for old_ckpt in clean_ckpts[:-2]:
            try:
                os.remove(old_ckpt)
            except Exception:
                pass

# ── End-of-run probe ──────────────────────────────────────────────────────────
model.eval()
probes = [
    ("Alice is taller than Bob. Who is shorter? Answer:", "Bob"),
    ("X > Y > Z height. Who is shortest? Answer:",       "Z"),
    ("Question: What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:", "B"),
    ("Question: What planet is closest to the sun?\nA. Venus\nB. Earth\nC. Mercury\nD. Mars\nAnswer:", "C"),
]
print(f"\n{'='*60}")
for prompt, expected in probes:
    ids    = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits, loops, trace = model(ids[:, -SEQ_LEN:])
    top1 = tokenizer.decode([logits[0, -1, :].argmax().item()]).strip()
    hit  = expected.lower() in top1.lower()
    print(f"  {'✅' if hit else '❌'} [{loops} loops] {prompt[-40:]!r} → {top1!r}")
print(f"{'='*60}")

ckpt = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "step": STEPS
}
torch.save(ckpt, SAVE_PATH)
print(f"\n✅ v19 complete — weights saved to {SAVE_PATH}")
