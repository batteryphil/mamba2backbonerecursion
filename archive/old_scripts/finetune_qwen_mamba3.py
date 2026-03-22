"""
finetune_qwen_mamba3.py
========================
Mamba3 recursive loop architecture grafted onto Qwen2.5-1.5B-Instruct.

Core insight (KV cache cure):
  Traditional CoT:  input(L) → generate_tokens(+N) → KV grows to L+N
  This approach:    input(L, FIXED) → N loops → KV stays at L forever

Architecture:
  Layers 0-13  (frozen) : feature extraction, runs ONCE per request
  Layers 14-27 (LoRA)   : reasoning engine, runs N times on fixed L
  loop_state             : accumulated thought-state carried between loops
  step_emb               : loop-position clock injected before each pass

Critical fixes from review:
  1. DRIVESHAFT: loop_state carries reasoning forward (NOT reset to base_features)
  2. BPTT VRAM:  gradient checkpointing on top-layer forward (14×6=84 eff. layers)
"""

import os, json, random, sys, threading, datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, Qwen2ForCausalLM, AutoConfig

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Constants ─────────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID    = "Qwen/Qwen2.5-1.5B-Instruct"
STEPS       = 100_000
BATCH_SIZE  = 4
ACCUM       = 4          # effective batch = 16
LOG_EVERY   = 50
SEQ_LEN     = 384        # reduced from 512 to cut VRAM during backward
MAX_LOOPS   = 6
BASE_SPLIT  = 14         # layers 0-13 frozen feature extractor, 14-27 looped
LORA_RANK   = 8
LR_LORA     = 3e-4
LR_EMB      = 1e-3
RESUME_FROM = ""
SAVE_PATH   = "qwen_mamba3_v1.pt"

print(f"\n{'='*60}")
print(f"  Qwen2.5-1.5B + Mamba3 Recursive Loops")
print(f"  KV-Cache cure: reason depth N ≠ context growth")
print(f"{'='*60}\n", flush=True)

# ── Tokenizer + THINK token ────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

THINK_STR = "<THINK>"
if THINK_STR not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({"additional_special_tokens": [THINK_STR]})
    print(f"  Added {THINK_STR} to tokenizer. New vocab: {len(tokenizer)}")

THINK_TOKEN_ID = tokenizer.convert_tokens_to_ids(THINK_STR)
EOS_TOKEN_ID   = tokenizer.eos_token_id
print(f"  <THINK> ID : {THINK_TOKEN_ID}")
print(f"  EOS ID     : {EOS_TOKEN_ID}")

# Answer boundary tokens for pointer mask
_ANS_IDS = tokenizer.encode("\nAnswer:", add_special_tokens=False)
print(f"  Answer boundary: {_ANS_IDS} = {tokenizer.decode(_ANS_IDS)!r}")

ALLOWED_CORE = [EOS_TOKEN_ID, THINK_TOKEN_ID]
# NOTE: A/B/C/D intentionally excluded — accessible via MMLU input tokens only

def find_answer_start(ids: list[int]) -> int:
    """Return index of first non-Answer-boundary token after 'Answer:'."""
    for i in range(len(ids) - len(_ANS_IDS)):
        if ids[i:i+len(_ANS_IDS)] == _ANS_IDS:
            pos = i + len(_ANS_IDS)
            return pos if pos < len(ids) else -1
    return -1


# ── LoRA ───────────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """Low-rank adapter for a frozen nn.Linear."""
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base   = base
        self.rank   = rank
        self.scale  = alpha / rank
        r           = rank
        d_in, d_out = base.in_features, base.out_features
        self.A = nn.Parameter(torch.randn(r, d_in,  device=base.weight.device,
                                          dtype=base.weight.dtype) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, r, device=base.weight.device,
                                          dtype=base.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


# ── Recursive Qwen wrapper ──────────────────────────────────────────────────────
class RecursiveQwen(nn.Module):
    """
    Qwen2.5-1.5B with Mamba3 recursive loop grafted onto the top half.

    Layers 0-13  : frozen feature extractor (runs once)
    Layers 14-27 : LoRA reasoning engine (runs MAX_LOOPS times)

    Loop state design (FIX 1 — the driveshaft):
      loop_state starts as zeros and ACCUMULATES across loops.
      Each loop sees: base_features + loop_state + step_emb(step_i)
      After top layers: loop_state = new_x  ← output CARRIED FORWARD

    BPTT budget (FIX 2 — the VRAM trap):
      gradient_checkpoint(run_top_layers) compresses 14×6=84 eff. layers
      back to checkpoint-able segments.
    """
    MAX_LOOPS: int = 6

    def __init__(self, base_model: Qwen2ForCausalLM):
        super().__init__()
        m = base_model.model
        self.embed_tokens  = m.embed_tokens
        self.rotary_emb    = m.rotary_emb    # needed for position_embeddings in Transformers 5.x
        self.base_layers   = nn.ModuleList(m.layers[:BASE_SPLIT])   # frozen
        self.top_layers    = nn.ModuleList(m.layers[BASE_SPLIT:])   # LoRA
        self.norm          = m.norm
        self.lm_head       = base_model.lm_head
        d_model            = m.embed_tokens.embedding_dim           # 1536

        # Freeze everything
        for p in base_model.parameters():
            p.requires_grad_(False)

        # Inject LoRA on q_proj and v_proj of top layers
        # Note: keep LoRA params in fp32 for stable gradients under AMP
        for layer in self.top_layers:
            attn = layer.self_attn
            attn.q_proj = LoRALinear(attn.q_proj, rank=LORA_RANK, alpha=LORA_RANK*2)
            attn.v_proj = LoRALinear(attn.v_proj, rank=LORA_RANK, alpha=LORA_RANK*2)

        # step_emb and loop_norm: fp32 params, cast to bf16 in forward
        self.step_emb = nn.Embedding(MAX_LOOPS, d_model)
        nn.init.normal_(self.step_emb.weight, std=0.01)
        self.loop_norm = nn.RMSNorm(d_model, eps=1e-6)

        n_lora = sum(p.numel() for n, p in self.named_parameters()
                     if p.requires_grad and "lora" in n.lower() or
                     "A" in n.split(".")[-1] or "B" in n.split(".")[-1])
        n_new  = (sum(p.numel() for p in self.step_emb.parameters()) +
                  sum(p.numel() for p in self.loop_norm.parameters()))
        # More accurate count
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"  Frozen params   : {frozen:,}")
        print(f"  Trainable params: {trainable:,}  (LoRA + step_emb + loop_norm)")
        print(f"  LoRA split      : layers {BASE_SPLIT}-27 ({len(self.top_layers)} layers)")

    # ── position helper ───────────────────────────────────────────────────
    @staticmethod
    def _make_position_ids(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.arange(seq_len, device=device).unsqueeze(0)

    # ── run top layers (one loop pass) ─────────────────────────────────────
    def _run_top_layers(self, x: torch.Tensor,
                        pos_ids: torch.Tensor,
                        position_embeddings: tuple) -> torch.Tensor:
        """Run all top layers (for gradient checkpointing wrapper).

        Pass attention_mask=None — Qwen2's SDPA attention handles
        causal masking internally via is_causal=True.
        """
        for layer in self.top_layers:
            raw = layer(x, attention_mask=None, position_ids=pos_ids,
                        position_embeddings=position_embeddings)
            x = raw[0] if isinstance(raw, tuple) else raw
        return x

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, input_ids: torch.Tensor,
                tgt_labels: torch.Tensor = None,
                ans_starts: list = None,
                accum: int = 1,
                max_train_loops: int = None) -> tuple:
        """
        Training:  tgt_labels provided → returns (loss, acc)
        Inference: tgt_labels=None     → returns (logits, loops_taken, trace)
        """
        B, L = input_ids.shape

        # Positional infrastructure (same for all loops)
        pos_ids = self._make_position_ids(L, input_ids.device)

        # ── Base pass (layers 0-13, runs ONCE) ──────────────────────────────
        x = self.embed_tokens(input_ids)
        # Rotary embeddings only depend on position, not content — compute once
        position_embeddings = self.rotary_emb(x, pos_ids)
        for layer in self.base_layers:
            # attention_mask=None → Qwen SDPA uses is_causal=True internally
            raw = layer(x, attention_mask=None, position_ids=pos_ids,
                        position_embeddings=position_embeddings)
            x = raw[0] if isinstance(raw, tuple) else raw
        base_features = x.detach()   # detach: base layers are frozen, no grad needed

        # ── Training path ────────────────────────────────────────────────────
        if self.training:
            if max_train_loops is None:
                max_train_loops = self.MAX_LOOPS

            # FIX 1 (DRIVESHAFT): loop_state carries reasoning forward.
            # It is NOT reset to base_features each loop — that would
            # discard all accumulated thought and reduce to N parallel passes.
            loop_state = torch.zeros_like(base_features)

            step_losses  = []
            answer_losses= []

            for step_i in range(max_train_loops):
                step_vec = self.step_emb(
                    torch.tensor(step_i, device=input_ids.device)
                ).to(base_features.dtype)  # cast to match bf16 base features
                ln_state = self.loop_norm(loop_state.float()).to(base_features.dtype)
                x = base_features + ln_state + step_vec

                # gradient_checkpoint with use_reentrant=True is more stable
                # with frozen base weights + LoRA trainable params.
                x = checkpoint(
                    self._run_top_layers, x, pos_ids,
                    position_embeddings,
                    use_reentrant=True
                )

                # DRIVESHAFT: save output as the thought-state for next loop
                loop_state = x

                # Compute logits + pointer mask + loss
                # BUG FIX: clone() before masking — in-place mutation of a
                # gradient-checkpointed tensor causes NaN in backprop.
                logits_raw = self.lm_head(self.norm(x))  # (B, L, V)
                V = logits_raw.shape[-1]

                # Per-sample pointer mask (only tokens present in input + core)
                is_answer_step = (step_i == max_train_loops - 1)
                step_targets   = []
                masked_logits  = []  # build per-sample, then stack

                for b in range(B):
                    # Pointer mask for this sample (out-of-place)
                    unique_ids  = torch.unique(input_ids[b])
                    allowed     = torch.cat([
                        unique_ids,
                        torch.tensor(ALLOWED_CORE, device=input_ids.device)
                    ]).unique()
                    pmask = torch.full((V,), float("-inf"), device=input_ids.device,
                                      dtype=logits_raw.dtype)
                    pmask[allowed] = 0.0
                    # Out-of-place: logits_raw[b] + pmask creates a new tensor
                    masked_logits.append(logits_raw[b] + pmask)

                    tgt = torch.full((L,), -100, dtype=torch.long,
                                     device=input_ids.device)
                    ans_pos = ans_starts[b] if ans_starts else -1
                    if is_answer_step:
                        # Final loop: supervise the actual answer token
                        if 0 < ans_pos < L:
                            tgt[ans_pos - 1] = tgt_labels[b, ans_pos]
                    else:
                        # Intermediate loops: supervise THINK token
                        if 0 < ans_pos < L:
                            tgt[ans_pos - 1] = THINK_TOKEN_ID
                    step_targets.append(tgt)

                if tgt_labels is not None:
                    logits_masked = torch.stack(masked_logits)  # (B, L, V)
                    tgt_tensor    = torch.stack(step_targets)
                    # CRITICAL: cast to fp32 before loss — fp16 lm_head logits
                    # over 151k vocab can overflow (~65504 max) causing NaN.
                    # clamp prevents extreme values before softmax.
                    logits_f32 = logits_masked.float().clamp(-100.0, 100.0)
                    raw_loss = F.cross_entropy(
                        logits_f32.view(-1, V), tgt_tensor.view(-1),
                        ignore_index=-100
                    ) / accum
                    # NaN guard: skip step if loss blew up
                    if torch.isnan(raw_loss) or torch.isinf(raw_loss):
                        print(f"  ⚠️  NaN/Inf loss at step_i={step_i}, skipping",
                              flush=True)
                    else:
                        step_losses.append(raw_loss)
                        if is_answer_step:
                            answer_losses.append(raw_loss.detach())

            # Return raw total_loss tensor (no backward here) and scalar acc.
            # The training loop will call scaler.scale(loss).backward().
            combined_loss = sum(step_losses) if step_losses else torch.tensor(0.0, device=input_ids.device, requires_grad=True)
            avg_loss_val  = sum(l.item() for l in answer_losses) / max(1, len(answer_losses))

            # Accuracy on final loop answer positions
            with torch.no_grad():
                final_logits = self.lm_head(self.norm(loop_state)).float()
                correct = total_tok = 0
                for b in range(B):
                    ans_pos = ans_starts[b] if ans_starts else -1
                    if 0 < ans_pos < L:
                        pred = final_logits[b, ans_pos - 1, :].argmax().item()
                        gold = tgt_labels[b, ans_pos].item()
                        if gold != -100:
                            correct  += (pred == gold)
                            total_tok += 1
                acc = correct / max(1, total_tok) * 100

            return combined_loss, avg_loss_val, acc

        # ── Inference path ────────────────────────────────────────────────────
        vocab_size  = self.lm_head.weight.shape[0]
        dtype       = next(self.embed_tokens.parameters()).dtype
        # Pointer mask (built once — input is FIXED throughout all loops)
        pmask = torch.full((vocab_size,), float("-inf"), device=input_ids.device,
                           dtype=dtype)
        unique_ids = torch.unique(input_ids[0])
        allowed    = torch.cat([
            unique_ids,
            torch.tensor(ALLOWED_CORE, device=input_ids.device)
        ]).unique()
        pmask[allowed] = 0.0

        trace       = []
        loops_taken = self.MAX_LOOPS
        loop_state  = torch.zeros_like(base_features)  # scratchpad starts blank

        for step_i in range(self.MAX_LOOPS):
            step_vec   = self.step_emb(
                torch.tensor(step_i, device=input_ids.device)
            ).to(base_features.dtype)
            ln_state   = self.loop_norm(loop_state.float()).to(base_features.dtype)
            x = base_features + ln_state + step_vec

            with torch.no_grad():
                x = self._run_top_layers(x, pos_ids, position_embeddings)

            loop_state = x   # carry state forward

            logits_last = self.lm_head(self.norm(x))[0, -1, :] + pmask
            p           = torch.softmax(logits_last, dim=-1)
            top_tok_id  = p.argmax().item()
            top_tok_str = tokenizer.decode([top_tok_id]).strip()
            max_prob    = p.max().item()
            entropy     = -(p * (p + 1e-12).log()).sum().item()
            trace.append((f"L{step_i+1}", round(max_prob, 2), top_tok_str))

            if top_tok_id != THINK_TOKEN_ID:
                loops_taken = step_i + 1
                break

        return self.lm_head(self.norm(x)), loops_taken, trace


# ── Load base model ───────────────────────────────────────────────────────────
print("  Loading Qwen2.5-1.5B-Instruct (bfloat16)...", flush=True)
_base = Qwen2ForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
_base.resize_token_embeddings(len(tokenizer))   # accommodate <THINK>
model = RecursiveQwen(_base).to(DEVICE)
model.train()


# ── Data loading ──────────────────────────────────────────────────────────────
def load_training_data() -> list[dict]:
    """Load system2_logic + MMLU-format samples, return {text, hops} dicts."""
    samples = []
    if Path("system2_logic_v1.json").exists():
        with open("system2_logic_v1.json") as f:
            raw = json.load(f)
        for s in raw:
            samples.append({"text": s["text"], "hops": s.get("hops", 2)})
        print(f"  Logic samples  : {len(samples):,}  (system2_logic_v1.json)")
    if Path("mmlu_format_v17.json").exists():
        with open("mmlu_format_v17.json") as f:
            raw = json.load(f)
        for s in raw[:10_000]:
            samples.append({"text": s.get("text", s.get("prompt", "")), "hops": 1})
        print(f"  MMLU samples   : {min(10000,len(raw)):,}  (mmlu_format_v17.json)")
    # Deduplicate
    seen, unique = set(), []
    for s in samples:
        if s["text"] not in seen:
            seen.add(s["text"])
            unique.append(s)
    print(f"  Total unique   : {len(unique):,}")
    return unique

data = load_training_data()


def make_batch(pool: list[dict], max_loops: int, seed: int) -> tuple:
    """Sample a batch gated by curriculum max_loops."""
    rng     = random.Random(seed)
    eligible = [s for s in pool if s.get("hops", 1) <= max_loops]
    if len(eligible) < BATCH_SIZE:
        eligible = pool
    batch   = rng.sample(eligible, BATCH_SIZE)
    texts   = [s["text"] for s in batch]

    enc = tokenizer(
        texts, max_length=SEQ_LEN, truncation=True,
        padding="max_length", return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(DEVICE)
    labels    = input_ids.clone()

    ans_starts = []
    for b in range(BATCH_SIZE):
        ids_list = input_ids[b].tolist()
        pos = find_answer_start(ids_list)
        ans_starts.append(pos)

    return input_ids, labels, ans_starts


# ── Optimizer ─────────────────────────────────────────────────────────────────
emb_params  = list(model.step_emb.parameters()) + list(model.loop_norm.parameters())
lora_params = [p for n, p in model.named_parameters()
               if p.requires_grad and p not in set(emb_params)]

optimizer = torch.optim.AdamW([
    {"params": emb_params,  "lr": LR_EMB,  "weight_decay": 0.0},
    {"params": lora_params, "lr": LR_LORA, "weight_decay": 0.01},
], betas=(0.9, 0.999))

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=1e-5)


# ── Checkpoint loading ────────────────────────────────────────────────────────
_loops_expanded = False
if RESUME_FROM and Path(RESUME_FROM).exists():
    ckpt = torch.load(RESUME_FROM, map_location=DEVICE)
    ckpt_emb = ckpt.get("model_state_dict", {}).get("step_emb.weight")
    if ckpt_emb is not None and ckpt_emb.shape[0] != MAX_LOOPS:
        sd = ckpt["model_state_dict"]
        adapted = model.step_emb.weight.data.clone()
        min_l   = min(ckpt_emb.shape[0], MAX_LOOPS)
        adapted[:min_l] = ckpt_emb[:min_l]
        sd["step_emb.weight"] = adapted
        _loops_expanded = True
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    if not _loops_expanded:
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"  Resumed from {RESUME_FROM}")
else:
    print(f"  Starting fresh (no resume checkpoint)")


# ── Training loop ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    current_max_loops = 2
    rolling_acc       = 0.0
    rolling_window    = []
    start_step        = 0

    if RESUME_FROM and Path(RESUME_FROM).exists():
        ckpt2 = torch.load(RESUME_FROM, map_location="cpu")
        start_step        = ckpt2.get("step", 0)
        current_max_loops = ckpt2.get("current_max_loops", 2)

    print(f"\n{'─'*60}")
    print(f"  Training: {STEPS} steps | Batch={BATCH_SIZE*ACCUM} (×{ACCUM} accum)")
    print(f"  Curriculum resuming at MaxN={current_max_loops}")
    print(f"{'─'*60}\n", flush=True)

    model.train()
    optimizer.zero_grad()
    total_loss = total_acc = 0.0
    log_count  = 0

    for step in range(start_step, STEPS):
        for accum_i in range(ACCUM):
            input_ids, labels, ans_starts = make_batch(data, current_max_loops,
                                                       seed=step * ACCUM + accum_i)
            loss_tensor, avg_loss_val, acc = model(
                input_ids, tgt_labels=labels, ans_starts=ans_starts,
                accum=ACCUM, max_train_loops=current_max_loops
            )
            loss_tensor.backward()
            total_loss += avg_loss_val
            total_acc  += acc
            log_count  += 1

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # ── Logging ────────────────────────────────────────────────────────
        if (step + 1) % LOG_EVERY == 0:
            avg_loss = total_loss / log_count if log_count else 0.0
            avg_acc  = total_acc  / log_count if log_count else 0.0
            total_loss = total_acc = log_count = 0.0

            vram = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0
            lr_e = optimizer.param_groups[0]["lr"]
            lr_l = optimizer.param_groups[1]["lr"]

            print(
                f"  Step {step+1:5d} | Loss: {avg_loss:.4f} | Acc: {avg_acc:5.1f}% "
                f"| LR(emb): {lr_e:.2e} LR(lora): {lr_l:.2e} "
                f"| VRAM: {vram:.2f}GB | MaxN: {current_max_loops}",
                flush=True
            )

            # Rolling window for curriculum graduation
            rolling_window.append(avg_acc)
            if len(rolling_window) > 5:
                rolling_window.pop(0)
            rolling_acc = sum(rolling_window) / len(rolling_window)

            # Graduate to next loop depth
            if len(rolling_window) == 5 and rolling_acc > 85.0:
                if current_max_loops < MAX_LOOPS:
                    current_max_loops += 1
                    rolling_window.clear()
                    print(f"\n  🚀 CURRICULUM → N={current_max_loops} "
                          f"(rolling {rolling_acc:.1f}%)\n", flush=True)
                    # Milestone checkpoint
                    milestone = SAVE_PATH.replace(".pt", f"_MaxN_{current_max_loops}.pt")
                    torch.save({
                        "model_state_dict"   : model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "step"               : step + 1,
                        "current_max_loops"  : current_max_loops,
                        "acc"                : rolling_acc,
                    }, milestone)
                    print(f"  🏆 Checkpoint → {milestone}\n", flush=True)

                elif rolling_acc > 95.0:
                    print(f"\n  🎉 N={MAX_LOOPS} MASTERED! Halting.\n", flush=True)
                    break

        # Periodic checkpoint
        if (step + 1) % 200 == 0:
            torch.save({
                "model_state_dict"   : model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "step"               : step + 1,
                "current_max_loops"  : current_max_loops,
            }, SAVE_PATH.replace(".pt", f"_step{step+1}.pt"))
            print(f"  💾 Checkpoint → {SAVE_PATH.replace('.pt', f'_step{step+1}.pt')}",
                  flush=True)

    # Final save
    torch.save({
        "model_state_dict": model.state_dict(),
        "step"            : STEPS,
    }, SAVE_PATH)
    print(f"\n  ✅ Done — weights saved to {SAVE_PATH}")

    # ── Quick inference test ────────────────────────────────────────────────
    model.eval()
    test_prompts = [
        "X = blue. Y = X. What is Y?\nAnswer:",
        "The book is in the bag. The bag is on the shelf. Where is the book?\nAnswer:",
        "Alice has 4 apples. Alice earns 3 apples. Alice now has?\nAnswer:",
        "A = red. B = A. C = B. What is C?\nAnswer:",
    ]
    print("\n" + "="*60)
    for prompt in test_prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            _, loops, trace = model(ids)
        trace_str = " → ".join(f"{t[0]}:{t[2]}" for t in trace)
        print(f"  {trace_str}")
        print(f"  Q: {repr(prompt.strip())[:60]}  A: {trace[-1][2]!r}\n")
    print("="*60)
