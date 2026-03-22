"""
DiM-LLM v4.0 – Clean Slate Protocol Training Script
================================================
Changes vs. the original Windows build:
  - Removed Windows DLL path block completely.
  - Auto-detect latest checkpoint via glob (no hardcoded epoch003).
  - Saves optimizer+scheduler state so resume continues the LR schedule.
  - BF16 AMP wrapped in a no-op context on CPU (avoids the CUDA-only
    torch.autocast crash on CPU-only machines).
  - `model.token_embed` access is done on the *unwrapped* model
    (torch.compile wrappers hide sub-modules).
  - OMP/MKL thread-count tuned to physical cores on Linux.
  - torch.backends.cuda.matmul.allow_tf32 = True for Ampere (RTX 30xx).
  - CUDNN deterministic=False, benchmark=True (already present, kept).
"""

import torch
import torch.optim as optim
import json
import os
import time
import copy
import random
import glob
from transformers import GPT2Tokenizer
from mamba_llm_diffusion import DiM_LLM, MaskedDiffusionEngine, Config

# ── Linux-specific performance flags ──────────────────────────────────────────
# Tune OpenMP / MKL to physical cores (avoids hyperthreading overhead)
_phys_cores = max(1, os.cpu_count() // 2)
os.environ.setdefault("OMP_NUM_THREADS", str(_phys_cores))
os.environ.setdefault("MKL_NUM_THREADS", str(_phys_cores))
torch.set_num_threads(_phys_cores)

# Ampere (RTX 30xx / A-series) FP32 matmul TF32 shortcut
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

import argparse

# ── Hyper-parameters ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data",       type=str,   default="silver_v4_data.json")
parser.add_argument("--lr",         type=float, default=2.1e-5)
parser.add_argument("--epochs",     type=int,   default=40)
parser.add_argument("--batch_size", type=int,   default=2)
parser.add_argument("--seq_len",    type=int,   default=1024)
args = parser.parse_args()

DSR_FILE       = args.data
DSR_RATIO      = 0.25
BATCH_SIZE     = args.batch_size
EPOCHS         = args.epochs
SEQ_LEN = 1024 
LR = 1e-4      # Fresh start base LR
EARLY_STOP_PAT = 3
ACCUM_STEPS    = 4


# ── Helpers ───────────────────────────────────────────────────────────────────

def pick_latest_checkpoint() -> str | None:
    """Return the best available checkpoint path, preferring EMA variants."""
    for candidate in [
        "latest_checkpoint.pt",       # Specifically check for the most recent generic name
        "dim_llm_checkpoint.pt",      # Full bundle (weights + optim + epoch)
        "dim_llm_ema_checkpoint.pt",  # Latest EMA (rolling weights)
        "dim_llm_ema_best.pt",        # Best-val EMA
        *sorted(glob.glob("dim_llm_checkpoint_epoch*.pt"), reverse=True),
        *sorted(glob.glob("dim_llm_epoch*.pt"), reverse=True),
        *sorted(glob.glob("dim_llm_ema_epoch*.pt"), reverse=True),
    ]:
        if os.path.exists(candidate):
            return candidate
    return None


def build_dsr_chunks(tokenizer: GPT2Tokenizer, seq_len: int) -> list:
    """Load Diversified Synthetic Reasoning data into fixed-length chunks."""
    if not os.path.exists(DSR_FILE):
        print("  [DSR] Not found - disabled.")
        return []
    with open(DSR_FILE, "r") as f:
        dsr_list = json.load(f)
    
    all_ids = []
    all_masks = []
    for item in dsr_list:
        if isinstance(item, dict):
            text = item.get("text", "")
        else:
            text = item
            
        parts = text.split(" | Assistant: ")
        if len(parts) == 2:
            u_ids = tokenizer.encode(parts[0] + " | Assistant: ", add_special_tokens=False)
            a_ids = tokenizer.encode(parts[1] + " " + tokenizer.eos_token + " ", add_special_tokens=False)
            all_ids.extend(u_ids)
            all_masks.extend([0] * len(u_ids))
            all_ids.extend(a_ids)
            all_masks.extend([1] * len(a_ids))
        else:
            ids = tokenizer.encode(text + " " + tokenizer.eos_token + " ", add_special_tokens=False)
            all_ids.extend(ids)
            all_masks.extend([1] * len(ids))

    ids = torch.tensor(all_ids, dtype=torch.long)
    masks = torch.tensor(all_masks, dtype=torch.uint8)
    
    # Pad if total length is less than seq_len
    if len(ids) < seq_len:
        pad_len = seq_len - len(ids)
        # Use eos_token_id if pad_token_id is somehow still None
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
        masks = torch.cat([masks, torch.zeros(pad_len, dtype=torch.uint8)])
        chunks = [(ids, masks)]
    else:
        chunks = [(ids[i : i + seq_len], masks[i : i + seq_len]) for i in range(0, len(ids) - seq_len, seq_len)]
    
    random.shuffle(chunks)
    print(f"  [DSR] {len(chunks)} chunks from {len(dsr_list)} examples.")
    return chunks


def get_unwrapped_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap a torch.compile / DataParallel wrapper to access sub-modules."""
    if hasattr(model, "_orig_mod"):   # torch.compile
        return model._orig_mod
    if hasattr(model, "module"):      # DataParallel
        return model.module
    return model


class AdaptiveGradientClipper:
    """
    AdaGC (Adaptive Gradient Clipping)
    Tracks per-tensor gradient norm EMA and scales back "screaming" weights.
    """
    def __init__(self, parameters, lambda_factor=0.05, ema_decay=0.99, max_abs_grad=5.0):
        self.parameters = [p for p in parameters if p.requires_grad]
        self.lambda_factor = lambda_factor
        self.ema_decay = ema_decay
        self.max_abs_grad = max_abs_grad
        self.device = self.parameters[0].device
        # Historical EMA of gradient norms for each parameter
        self.norm_emas = [torch.zeros(1, device=self.device) for _ in self.parameters]

    @torch.no_grad()
    def step(self) -> int:
        spike_count = 0
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            
            # calculate current norm for this specific tensor
            curr_norm = torch.norm(p.grad)
            
            # initialize EMA if zero
            if self.norm_emas[i] == 0:
                self.norm_emas[i].copy_(curr_norm)
            
            # Tracking value for EMA update
            ema_update_norm = curr_norm
            
            # Clipping logic (AdaGC)
            threshold = self.norm_emas[i] * (1.0 + self.lambda_factor)
            if curr_norm > threshold:
                scale = threshold / (curr_norm + 1e-8)
                p.grad.detach().mul_(scale)
                spike_count += 1
                # Constraint: update EMA with threshold value to avoid history pollution
                ema_update_norm = threshold
            
            # Absolute hard cap (prevent explosion even if EMA is high)
            abs_norm = torch.norm(p.grad)
            if abs_norm > self.max_abs_grad:
                p.grad.detach().mul_(self.max_abs_grad / (abs_norm + 1e-8))
                ema_update_norm = torch.min(ema_update_norm, torch.tensor(self.max_abs_grad, device=self.device))

            # Update EMA using stable (clipped) norm
            self.norm_emas[i].mul_(self.ema_decay).add_(ema_update_norm, alpha=1.0 - self.ema_decay)
            
        return spike_count


# ── BF16 AMP context (no-op on CPU) ──────────────────────────────────────────
import contextlib

def autocast_ctx(device: str):
    """Return a BF16 autocast context for CUDA or a no-op on CPU."""
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ── Socratic Benchmark Hook ──────────────────────────────────────────────────
def run_socratic_benchmark(engine, tokenizer, step, device="cuda"):
    """
    V4 Diagnostic Hook: Tests for Decoupled Logic and JSON Leakage.
    """
    print(f"\n[🔍] Running Socratic Benchmark at Step {step}...")
    
    # We intentionally use the Plain Text format from the Silver Dataset.
    # If the model outputs brackets here, the syntax and semantics are still entangled.
    test_prompt = (
        "Consider the following: Alice is taller than Bob. Also, Bob is taller than Charlie. "
        "Who is the tallest among Alice, Bob, and Charlie? Well, following the logic:"
    )
    
    ids = torch.tensor([tokenizer.encode(test_prompt)], device=device)
    
    # Toggle to eval mode to disable dropout and lock batch norm stats
    engine.model.eval()
    if engine.ema_model:
        engine.ema_model.eval()

    with torch.no_grad():
        out = engine.sample(
            n_samples=1, 
            steps=250,  # 250 steps is sufficient to trigger the V4 Commitment Cliff
            prompt_ids=ids, 
            base_temp=0.5
        )
    # Return to training mode
    engine.model.train()
    
    response = tokenizer.decode(out[0], skip_special_tokens=True)
    generated_text = response[len(test_prompt):].strip()
    
    print(f"Output: {generated_text[:100]}...\n")
    
    # Forensic Criteria
    target_found = "Alice" in generated_text
    json_leak = any(char in generated_text for char in ['{', '}', '"'])
    
    status = "RED"
    if target_found and not json_leak:
        print("🟢 [SEMANTIC SNAP DETECTED] Logic calculated. Zero JSON leakage.")
        status = "GREEN"
    elif target_found and json_leak:
        print("🟡 [SYNTACTIC LEAK] Target found, but JSON brackets hallucinated.")
        status = "YELLOW"
    else:
        print("🔴 [PRE-SNAP] Logic not yet grounded.")
        status = "RED"

    return {
        "step": step,
        "prompt": test_prompt,
        "response": generated_text,
        "target_found": target_found,
        "json_leak": json_leak,
        "status": status
    }

# ── Training ──────────────────────────────────────────────────────────────────

def train() -> None:
    """Main training loop for DiM-LLM v3.2 on Linux."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDiM-LLM v4.0 | Clean Slate Protocol | device={device}")
    print(f"  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")

    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    tokenizer.pad_token = tokenizer.eos_token

    # Data
    def process_dataset(data_list, tokenizer):
        all_ids = []
        all_masks = []
        for text in data_list:
            parts = text.split(" | Assistant: ")
            if len(parts) == 2:
                u_ids = tokenizer.encode(parts[0] + " | Assistant: ", add_special_tokens=False)
                a_ids = tokenizer.encode(parts[1] + " " + tokenizer.eos_token + " ", add_special_tokens=False)
                all_ids.extend(u_ids)
                all_masks.extend([0] * len(u_ids))
                all_ids.extend(a_ids)
                all_masks.extend([1] * len(a_ids))
            else:
                ids = tokenizer.encode(text + " " + tokenizer.eos_token + " ", add_special_tokens=False)
                all_ids.extend(ids)
                all_masks.extend([1] * len(ids))
        return torch.tensor(all_ids, dtype=torch.long), torch.tensor(all_masks, dtype=torch.uint8)

    # 1. Load Main Dataset (always, even if refinement)
    with open("train_data.json", "r") as f:
        train_list = json.load(f)
    with open("val_data.json", "r") as f:
        val_list = json.load(f)
    train_ids, train_masks = process_dataset(train_list, tokenizer)
    val_ids, val_masks = process_dataset(val_list, tokenizer)

    # 2. Load DSR / Alignment Dataset
    if DSR_FILE != "synthetic_dsr_data.json":
        print(f"  [Alignment Mode] Injecting custom 'Gold' data: {DSR_FILE}")
        # We don't force DSR_RATIO to 1.0 here; we let the model mix it with train_data
    
    dsr_chunks = build_dsr_chunks(tokenizer, SEQ_LEN)

    # Model + EMA
    config = Config(vocab_size=len(tokenizer), d_model=1024, n_layers=11, seq_len=1024)
    model = DiM_LLM(config).to(device)
    ema_model = copy.deepcopy(model).to(device)
    for p in ema_model.parameters():
        p.requires_grad = False

    engine            = MaskedDiffusionEngine(model, config, device=device, ema_decay=0.999)
    engine.ema_model  = ema_model

    # ── Checkpoint resume (MUST happen BEFORE torch.compile) ─────────────────
    start_epoch = 0
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    ckpt_path = pick_latest_checkpoint()
    if ckpt_path:
        print(f"🛠️ Recovering Project Phoenix from {ckpt_path}...")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)

        # The checkpoint may be a full dict (weights+optim) or just state_dict
        if isinstance(state, dict) and "model_state" in state:
            ckpt_dict = state["model_state"]
            model_dict = model.state_dict()
            
            # ── Robust Embedding Load (Handles Vocab Resize) ──────────────────
            if "token_embed.weight" in ckpt_dict:
                ckpt_w = ckpt_dict["token_embed.weight"]
                model_w = model_dict["token_embed.weight"]
                if ckpt_w.shape != model_w.shape:
                    print(f"     [!] Vocab mismatch: CKPT {ckpt_w.shape[0]} vs Model {model_w.shape[0]}")
                    rows = min(ckpt_w.shape[0], model_w.shape[0])
                    model_w.data[:rows].copy_(ckpt_w[:rows])
                    # Remove from ckpt_dict so strict=False doesn't overwrite our manual copy with nothing
                    del ckpt_dict["token_embed.weight"]

            # Same for EMA if present
            if "ema_state" in state and ema_model:
                ema_ckpt_dict = state["ema_state"]
                ema_model_dict = ema_model.state_dict()
                if "token_embed.weight" in ema_ckpt_dict:
                    e_ckpt_w = ema_ckpt_dict["token_embed.weight"]
                    e_model_w = ema_model_dict["token_embed.weight"]
                    if e_ckpt_w.shape != e_model_w.shape:
                        rows = min(e_ckpt_w.shape[0], e_model_w.shape[0])
                        e_model_w.data[:rows].copy_(e_ckpt_w[:rows])
                        del ema_ckpt_dict["token_embed.weight"]
                ema_model.load_state_dict(ema_ckpt_dict, strict=False)

            model.load_state_dict(ckpt_dict, strict=False)
            
            if "optimizer_state" in state:
                try:
                    optimizer.load_state_dict(state["optimizer_state"])
                    print("     Optimizer state restored.")
                except Exception as e:
                    print(f"     [!] Could not restore optimizer state ({e}). Starting optimizer fresh.")
            
            if "scheduler_state" in state:
                try:
                    scheduler.load_state_dict(state["scheduler_state"])
                    print("     Scheduler state restored.")
                except Exception as e:
                    print(f"     [!] Could not restore scheduler state ({e}).")
            
            start_epoch = state.get("epoch", 0)
            print(f"     Full checkpoint — resuming from epoch {start_epoch + 1}")
        else:
            # Legacy: plain state_dict (weights only)
            model.load_state_dict(state)
            ema_model.load_state_dict(state)
            print("     Weight-only checkpoint loaded.")
    else:
        print("   -> No recent checkpoint found, starting fresh sequence.")

    # 🛡️ Force stability: Zero-init AdaLN modulations (Identity Map start)
    raw_model = get_unwrapped_model(model)
    raw_model.zero_init_ada()

    # Compile on CUDA AFTER loading weights (avoids _orig_mod.* key mismatch)
    # DISABLED: The pure-PyTorch fallback for the Mamba scan uses a 256-step
    # for-loop. torch.compile(inductor) attempts to unroll this completely,
    # causing an indefinite hang (100% CPU, 60% GPU) during tracing.
    # if device == "cuda":
    #     try:
    #         print("  -> Compiling model with torch.compile (inductor)...")
    #         model = torch.compile(model, backend="inductor")
    #         print("  -> Compile OK")
    #     except Exception as exc:
    #         print(f"  -> torch.compile failed ({exc}), continuing without.")

    num_chunks = (len(train_ids) - 1) // SEQ_LEN
    if num_chunks <= 0 and dsr_chunks:
        num_chunks = len(dsr_chunks)
    print(
        f"  -> {num_chunks} train chunks | batch={BATCH_SIZE} "
        f"| accum={ACCUM_STEPS} (eff={BATCH_SIZE*ACCUM_STEPS}) "
        f"| lr={LR} | patience={EARLY_STOP_PAT}"
    )

    resume_step = start_epoch * num_chunks
    stats = {
        "step": 0, "running_loss": 0.0, "tps": 0.0, 
        "salads": [], "spike_score": 0, "inference_metrics": [], "logic_drift": False,
        "train_loss": [], "val_loss": [], "socratic_results": []
    }

    clipper = AdaptiveGradientClipper(model.parameters(), lambda_factor=0.01)
    
    best_val        = float("inf")
    patience_counter = 0

    # Cache the unwrapped model reference once (for token_embed access)
    raw_model = get_unwrapped_model(model)

    loss_window = []  # History for spike detection
    last_stable_loss = 9.0
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss  = 0.0
        indices     = list(range(num_chunks))
        random.shuffle(indices)
        dsr_cursor  = 0

        optimizer.zero_grad()

        for i in range(0, num_chunks, BATCH_SIZE):
            t0 = time.time()

            # ── Batch assembly ───────────────────────────────────────────────
            if dsr_chunks and random.random() < DSR_RATIO:
                batch_list = [
                    dsr_chunks[(dsr_cursor + k) % len(dsr_chunks)][0]
                    for k in range(BATCH_SIZE)
                ]
                mask_list = [
                    dsr_chunks[(dsr_cursor + k) % len(dsr_chunks)][1]
                    for k in range(BATCH_SIZE)
                ]
                context_list = [
                    torch.full((SEQ_LEN,), engine.mask_id, dtype=torch.long)
                    for _ in range(BATCH_SIZE)
                ]
                dsr_cursor += BATCH_SIZE
            else:
                batch_indices = indices[i : i + BATCH_SIZE]
                batch_list    = [
                    train_ids[idx * SEQ_LEN : (idx + 1) * SEQ_LEN]
                    for idx in batch_indices
                ]
                mask_list     = [
                    train_masks[idx * SEQ_LEN : (idx + 1) * SEQ_LEN]
                    for idx in batch_indices
                ]
                context_list  = []
                for idx in batch_indices:
                    if idx > 0:
                        history = train_ids[(idx - 1) * SEQ_LEN : idx * SEQ_LEN]
                    else:
                        history = torch.full((SEQ_LEN,), engine.mask_id, dtype=torch.long)
                    context_list.append(history)

            if not batch_list:
                continue

            batch_tokens   = torch.stack(batch_list).to(device, non_blocking=True)
            batch_masks    = torch.stack(mask_list).to(device, non_blocking=True)
            context_tokens = torch.stack(context_list).to(device, non_blocking=True)

            # --- GHOST SIEVE: Skip empty sequences ---
            if batch_tokens.numel() == 0 or batch_tokens.shape[1] == 0:
                print("⚠️ Skipping Ghost Batch.")
                continue

            # ── Context bank (no-grad) ───────────────────────────────────────
            with torch.no_grad():
                # Use raw_model to avoid torch.compile sub-module access issues
                context_embeddings = raw_model.token_embed(context_tokens)
                context_bank       = context_embeddings.mean(dim=1, keepdim=True)
                if random.random() < 0.15:   # CFG dropout
                    context_bank = None

            # ── Forward + loss ───────────────────────────────────────────────
            with autocast_ctx(device):
                loss = engine.forward_process(batch_tokens, context_bank=context_bank, loss_mask=batch_masks)

            # Watchdog: Detection of catastrophic divergence
            # Widened Threshold: 10-step average + higher tolerance
            if not torch.isnan(loss):
                loss_window.append(loss.item())
                if len(loss_window) > 10:
                    loss_window.pop(0)
                last_stable_loss = sum(loss_window) / len(loss_window)

            # ── Ghost Catch (Batch Sieve) ────────────────────────────────────
            # Increased threshold for fresh start (initial loss is high)
            if loss.item() > 150.0:
                print(f"⚠️ [GHOST CATCH] Batch Loss {loss.item():.2f} > 150.0. Purging batch gradients.")
                optimizer.zero_grad()
                continue

            if loss.item() > last_stable_loss + 6.5 and epoch >= 10:
                print(f"[!] CATASTROPHIC SPIKE: Loss {loss.item():.4f} exceeded 10-step avg {last_stable_loss:.4f} + 6.5")
                print("[!] Pausing training for emergency manual realignment.")
                torch.save(model.state_dict(), "diverged_state.pt")
                stats["diverged"] = True
                stats["divergence_loss"] = loss.item()
                with open("training_stats.json", "w") as fj:
                    json.dump(stats, fj)
                return  # Exit training loop

            loss = loss / ACCUM_STEPS
            loss.backward()

            # Running-loss EMA for monitor
            raw_loss = loss.item() * ACCUM_STEPS
            if stats["running_loss"] == 0:
                stats["running_loss"] = raw_loss
            else:
                stats["running_loss"] = 0.95 * stats["running_loss"] + 0.05 * raw_loss

            stats["step"] += 1

            # Weight update every ACCUM_STEPS micro-batches
            if stats["step"] % ACCUM_STEPS == 0:
                # Standard scheduler step
                curr_lr = scheduler.get_last_lr()[0]
                for g in optimizer.param_groups:
                    g['lr'] = curr_lr

                # 🚀 AdaGC Deployment
                spikes = clipper.step()
                stats["spike_score"] = spikes
                
                # 🛡️ Global Stability Guard
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                engine.update_ema()

                # ── Socratic Benchmark Trigger ───────────────────────────────
                if stats["step"] % 500 == 0:
                    soc_result = run_socratic_benchmark(engine, tokenizer, stats["step"], device=device)
                    stats["socratic_results"].append(soc_result)
                    if soc_result["status"] == "GREEN":
                        print("🧠 Jarvis has officially passed kindergarten logic.")

            elapsed    = max(time.time() - t0, 1e-6)
            batch_tps  = (batch_tokens.shape[0] * SEQ_LEN) / elapsed
            stats["tps"] = 0.9 * stats.get("tps", batch_tps) + 0.1 * batch_tps
            epoch_loss  += raw_loss

            if stats["step"] % 50 == 0:
                print(
                    f"  step={stats['step']:6d} "
                    f"loss={stats['running_loss']:.4f} "
                    f"tps={stats['tps']:.0f} "
                    f"spikes={stats['spike_score']}"
                )

            if stats["step"] % 100 == 0:
                # 🛰️ SYNC MONITOR
                with open("training_stats.json", "w") as f:
                    json.dump(stats, f)

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_accum      = 0.0
        num_val_chunks = (len(val_ids) - 1) // SEQ_LEN
        max_val        = min(num_val_chunks, 100)
        with torch.no_grad():
            for v in range(0, max_val, BATCH_SIZE):
                v_indices  = list(range(v, min(v + BATCH_SIZE, num_val_chunks)))
                vb = [val_ids[j * SEQ_LEN : (j + 1) * SEQ_LEN] for j in v_indices]
                vm = [val_masks[j * SEQ_LEN : (j + 1) * SEQ_LEN] for j in v_indices]
                vc = []
                for j in v_indices:
                    if j > 0:
                        vc.append(val_ids[(j - 1) * SEQ_LEN : j * SEQ_LEN])
                    else:
                        vc.append(torch.full((SEQ_LEN,), engine.mask_id, dtype=torch.long))
                if vb:
                    val_tokens          = torch.stack(vb).to(device, non_blocking=True)
                    val_batch_masks     = torch.stack(vm).to(device, non_blocking=True)
                    val_context_tokens  = torch.stack(vc).to(device, non_blocking=True)
                    val_embeddings      = raw_model.token_embed(val_context_tokens)
                    val_context_bank    = val_embeddings.mean(dim=1, keepdim=True)
                    with autocast_ctx(device):
                        val_accum += engine.forward_process(
                            val_tokens, context_bank=val_context_bank, loss_mask=val_batch_masks
                        ).item()

        avg_train = epoch_loss / max(1, num_chunks // BATCH_SIZE)
        avg_val   = val_accum  / max(1, max_val // BATCH_SIZE)
        stats["train_loss"].append(avg_train)
        stats["val_loss"].append(avg_val)
        cur_lr = scheduler.get_last_lr()[0]
        print(
            f"\nEpoch {epoch+1}/{EPOCHS} | "
            f"Train={avg_train:.4f} Val={avg_val:.4f} "
            f"LR={cur_lr:.2e} TPS={stats['tps']:.1f}"
        )

        # ── Save checkpoints ──────────────────────────────────────────────────
        epoch_tag = epoch + 1
        ckpt_body  = f"dim_llm_epoch{epoch_tag:03d}.pt"
        ema_body   = f"dim_llm_ema_epoch{epoch_tag:03d}.pt"

        # Full checkpoint bundle (weights + optimizer + scheduler + epoch)
        full_bundle = {
            "model_state":     model.state_dict(),
            "ema_state":       ema_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch":           epoch_tag,
        }
        torch.save(full_bundle,              ckpt_body)
        torch.save(ema_model.state_dict(),   ema_body)
        torch.save(full_bundle,              "dim_llm_checkpoint.pt")
        torch.save(ema_model.state_dict(),   "dim_llm_ema_checkpoint.pt")

        # Rotate: keep only the 3 most recent epoch files
        for pat, keep in [
            ("dim_llm_epoch*.pt", ckpt_body),
            ("dim_llm_ema_epoch*.pt", ema_body),
        ]:
            for old in sorted(glob.glob(pat))[:-3]:
                if old != keep:
                    os.remove(old)

        # ── Early stopping ────────────────────────────────────────────────────
        if avg_val < best_val:
            best_val         = avg_val
            patience_counter = 0
            torch.save(ema_model.state_dict(), "dim_llm_ema_best.pt")
            print(f"  ** Best val={best_val:.4f} -> dim_llm_ema_best.pt")
        else:
            patience_counter += 1
            print(f"  -- No val improvement {patience_counter}/{EARLY_STOP_PAT}")
            if patience_counter >= EARLY_STOP_PAT:
                print("Early stopping. Best model: dim_llm_ema_best.pt")
                break

        # ── Word-salad sample (Midnight Watch) ────────────────────────────────
        print("--- Word Salad (Midnight Watch) ---")
        
        prompts_to_test = [
            ("Greeting", "User: Hello there. | Assistant:"),
            ("Tool Call", "User: Find me an RTX 3060 on eBay for under $250. | Assistant:")
        ]
        
        current_salads = []
        for p_name, prompt_text in prompts_to_test:
            prompt_ids = torch.tensor([tokenizer.encode(prompt_text, add_special_tokens=False)], device=device)
            t_inf = time.time()
            # Use engine.adaptive_sample directly with triple return
            samples, steps, final_entropy = engine.adaptive_sample(
                n_samples=1, 
                prompt_ids=prompt_ids,
                max_steps=1000,
                min_steps=32,
                entropy_threshold=0.02
            )
            inf_elapsed = max(time.time() - t_inf, 1e-6)
            inf_tps = SEQ_LEN / inf_elapsed
            
            # 🧠 Automatic Balancing (Logic Drift) Check
            if p_name == "Greeting" and steps > 100:
                stats["logic_drift"] = True
            if p_name == "Tool Call" and steps < 100:
                stats["logic_drift"] = True

            dec = tokenizer.decode(samples[0].tolist(), skip_special_tokens=False)
            response = dec[len(prompt_text):].strip()
            
            thinking_mode = "FACT" if steps <= 32 else "REASON"
            print(f"  [Step: {steps:03d}/1000] | [TPS: {inf_tps:5.1f}] | [Entropy: {final_entropy:.2f}] | [Mode: {thinking_mode:6s}] | {response[:120]}")
            
            # Preservation for history (Grouped by epoch for Monitor UI)
            current_salads.append({
                "epoch": epoch_tag,
                "type": p_name,
                "steps": steps,
                "mode": thinking_mode,
                "inf_tps": inf_tps,
                "entropy": final_entropy,
                "prompt": f"[{p_name}] {prompt_text}", 
                "response": f"[Step: {steps:03d} | TPS: {inf_tps:.1f} | Ent: {final_entropy:.2f} | Mode: {thinking_mode}] {response}"
            })
            
        stats["salads"].append(current_salads)
        print("-----------------------------------")
        print("------------------\n")

        with open("training_stats.json", "w") as fj:
            json.dump(stats, fj)

    print("Training complete.")


if __name__ == "__main__":
    train()
