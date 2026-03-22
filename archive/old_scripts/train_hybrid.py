import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import argparse
from transformers import GPT2Tokenizer
from mamba_rbm import RecursiveMambaLM, Config
from phase_shift_scheduler import PhaseShiftLRScheduler

# 🚀 VRAM GUARD: Protocol v6.2.1
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import psutil
import gc

# 🧠 NARRATIVE ANALYSIS (Protocol v6.4.2)
NARRATIVE_MARKERS = [
    "one day", "importance of", "value of the", "there lived", 
    "many years ago", "lessons learned", "empathy", "curiosity",
    "named", "friends named", "adventure", "journey", "once upon",
    "nodded", "realized", "discovered", "timmy", "stuffed animal",
    "magical", "forest", "village"
]

def calculate_bias(text):
    text_lower = text.lower()
    score = 0
    for marker in NARRATIVE_MARKERS:
        score += text_lower.count(marker)
    return score

class AutoNScaler:
    def __init__(self, target_bias=1.5, cooldown=250):
        self.target_bias = target_bias
        self.cooldown_limit = cooldown
        self.last_scale_step = 0

    def check_and_scale(self, current_n, current_avg_bias, current_step):
        steps_since = current_step - self.last_scale_step
        if steps_since < self.cooldown_limit:
            return current_n

        if current_avg_bias < self.target_bias:
            new_n = min(current_n + 1, 3) # Cap at 3 for logic stability
            print(f"🚀 [AUTO-N] Bias threshold met ({current_avg_bias:.2f}) after {steps_since} steps. Scaling to N={new_n}")
            self.last_scale_step = current_step
            return new_n
            
        return current_n

class ThermalWatchdog:
    """
    Protocol v6.4: Hardware Survival Guard.
    Monitors CPU/GPU temps to prevent thermal/power trips.
    """
    def __init__(self, threshold=85):
        self.threshold = threshold
        self.paused = False

    def check(self):
        # 1. Check CPU Temps
        temps = psutil.sensors_temperatures()
        if 'coretemp' in temps:
            max_cpu = max(t.current for t in temps['coretemp'])
            if max_cpu > self.threshold:
                print(f"🌡️ THERMAL ALERT: CPU at {max_cpu}°C. Threshold: {self.threshold}°C.")
                return True
        
        # 2. Check GPU Temps
        try:
            import subprocess
            res = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"])
            max_gpu = int(res.decode().strip())
            if max_gpu > self.threshold:
                print(f"🌡️ THERMAL ALERT: GPU at {max_gpu}°C. Threshold: {self.threshold}°C.")
                return True
        except:
            pass
            
        return False

    def wait_if_needed(self):
        if self.check():
            print("🛑 THROWING ANCHOR: Pausing training for 45s to cool down hardware...")
            time.sleep(45)
            return True
        return False

# ── Hyper-parameters ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--logic_data",   type=str,   default="logic_v3.json")
parser.add_argument("--qa_data",      type=str,   default="qa_anchors.json")
parser.add_argument("--generic_bin",  type=str,   default="generic_150m_corpus.bin")
parser.add_argument("--lr",           type=float, default=4e-5) 
parser.add_argument("--epochs",       type=int,   default=20)
parser.add_argument("--batch_size",   type=int,   default=4) # 🛡️ Protocol v6.5.7: Increased from 2 to 4 to utilize remaining 5GB VRAM
parser.add_argument("--seq_len",      type=int,   default=1024)
parser.add_argument("--lr_restart",   action="store_true", help="SGDR warm restart: reset LR to start_lr and re-arm cosine decay.")
parser.add_argument("--force_path_a", action="store_true", help="Force N=1 (Path A only) immediately on resume for 500 consolidation steps then auto-restore N=3.")
parser.add_argument("--grafted",      action="store_true", help="Load mamba-130m grafted checkpoint (d_model=768, 24+2 layers). Freezes Path A, trains Path B + memory layers only.")
args = parser.parse_args()

LOGIC_FILE     = args.logic_data
QA_FILE        = args.qa_data
RELATIONAL_FILE = "relational_anchors.jsonl"
GENERIC_FILE   = args.generic_bin
BATCH_SIZE     = args.batch_size
EPOCHS         = args.epochs
SEQ_LEN        = args.seq_len
LR             = args.lr
ACCUM_STEPS    = 16 # 🛡️ Protocol v6.5.7: Halved to 16 to maintain effective batch size of 64 with actual batch 4
HYBRID_RATIO   = 0.20 # 🛡️ Protocol v6.5.6: Reverted to 20% to restore relational logic anchors after identity fix
CHECKPOINT_FREQ = 100 # 🛡️ Protocol v6.4.3: Increased frequency for anchoring stability

class HybridDataLoader:
    """
    Stochastic Windowing Loader for Protocol v6.0.
    10% Logic Anchor / 90% Generic Corpus.
    防止线性记忆 (Prevents Linear Memorization).
    """
    def __init__(self, tokenizer, seq_len, batch_size, ratio=0.1):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.ratio = ratio
        
        # 1. Load Logic Anchor (JSON -> Padded Chunks)
        self.logic_samples = []
        if os.path.exists(LOGIC_FILE):
            print(f"📂 Loading Logic v6.0 Anchor: {LOGIC_FILE}")
            with open(LOGIC_FILE, "r") as f:
                raw_logic = json.load(f)
            for item in raw_logic:
                text = item.get("text", "")
                ids = tokenizer.encode(text + " " + tokenizer.eos_token, add_special_tokens=False)
                if len(ids) > seq_len:
                    ids = ids[:seq_len]
                else:
                    ids = ids + [tokenizer.eos_token_id] * (seq_len - len(ids))
                self.logic_samples.append(torch.tensor(ids))
            
            del raw_logic
            gc.collect()
        
        # 2. Load Relational Anchors (JSONL -> Padded Chunks)
        self.relational_samples = []
        if os.path.exists(RELATIONAL_FILE):
            print(f"📂 Loading Relational Anchors: {RELATIONAL_FILE}")
            with open(RELATIONAL_FILE, "r") as f:
                for line in f:
                    item = json.loads(line)
                    text = item.get("text", "")
                    ids = tokenizer.encode(text + " " + tokenizer.eos_token, add_special_tokens=False)
                    if len(ids) > seq_len:
                        ids = ids[:seq_len]
                    else:
                        ids = ids + [tokenizer.eos_token_id] * (seq_len - len(ids))
                    self.relational_samples.append(torch.tensor(ids))

        # 2b. Load QA Anchors (JSON -> Padded Chunks)
        self.qa_samples = []
        if os.path.exists(QA_FILE):
            print(f"📂 Loading QA Anchors: {QA_FILE}")
            with open(QA_FILE, "r") as f:
                raw_qa = json.load(f)
            for item in raw_qa:
                text = item.get("text", "")
                ids = tokenizer.encode(text + " " + tokenizer.eos_token, add_special_tokens=False)
                if len(ids) > seq_len:
                    ids = ids[:seq_len]
                else:
                    ids = ids + [tokenizer.eos_token_id] * (seq_len - len(ids))
                self.qa_samples.append(torch.tensor(ids))
            del raw_qa
            gc.collect()
            
        # 3. Map Generic Library
        self.generic_data = None
        self.data_size = 0
        if os.path.exists(GENERIC_FILE):
            print(f"📂 Mapping Proto-v6 Library: {GENERIC_FILE}")
            self.generic_data = np.memmap(GENERIC_FILE, dtype=np.uint16, mode='r')
            self.data_size = len(self.generic_data)
        else:
            print(f"⚠️ Warning: {GENERIC_FILE} not found.")

    def __iter__(self):
        logic_indices = list(range(len(self.logic_samples)))
        rel_indices = list(range(len(self.relational_samples)))
        qa_indices = list(range(len(self.qa_samples)))
        random.shuffle(logic_indices)
        random.shuffle(rel_indices)
        if self.qa_samples: random.shuffle(qa_indices)
        l_ptr = 0
        r_ptr = 0
        q_ptr = 0
        
        # Determine batches based on total token count / seq_len
        if self.generic_data is not None:
            num_batches = (self.data_size // (self.seq_len * self.batch_size))
        else:
            num_batches = 1000
        
        for _ in range(num_batches):
            batch = []
            num_logic_total = max(1, int(self.batch_size * self.ratio))
            
            # Divide logic total between Math and Relational
            num_math = num_logic_total // 2
            num_rel  = num_logic_total - num_math
            num_qa   = 0
            
            # Devote 40% of logic items to QA (translates to ~10% of total batch items since hybrid ratio is 20-25%)
            if self.qa_samples and random.random() < 0.4 and num_logic_total > 0:
                num_qa = 1
                if num_math > 0:
                    num_math -= 1
                elif num_rel > 0:
                    num_rel -= 1
            
            num_generic = self.batch_size - num_logic_total
            
            # 1. Stochastic Math Logic
            for _ in range(num_math):
                if not self.logic_samples: break
                if l_ptr >= len(self.logic_samples):
                    random.shuffle(logic_indices)
                    l_ptr = 0
                batch.append(self.logic_samples[logic_indices[l_ptr]])
                l_ptr += 1
                
            # 2. Stochastic Relational Logic
            for _ in range(num_rel):
                if not self.relational_samples: break
                if r_ptr >= len(self.relational_samples):
                    random.shuffle(rel_indices)
                    r_ptr = 0
                batch.append(self.relational_samples[rel_indices[r_ptr]])
                r_ptr += 1
                
            # 2.5 Stochastic QA Extraction
            for _ in range(num_qa):
                if not self.qa_samples: break
                if q_ptr >= len(self.qa_samples):
                    random.shuffle(qa_indices)
                    q_ptr = 0
                batch.append(self.qa_samples[qa_indices[q_ptr]])
                q_ptr += 1
            
            # 3. Stochastic Generic (RANDOM WINDOW SHIFT)
            if self.generic_data is not None:
                while len(batch) < self.batch_size:
                    # Pick a random starting point in the 56M token ocean
                    max_pos = self.data_size - self.seq_len
                    start = random.randint(0, max_pos)
                    end = start + self.seq_len
                    chunk = torch.from_numpy(self.generic_data[start:end].astype(np.int64))
                    batch.append(chunk)
            else:
                while len(batch) < self.batch_size:
                    # Fallback to logic if no generic data
                    if self.logic_samples and random.random() < 0.5:
                        if l_ptr >= len(self.logic_samples):
                            random.shuffle(logic_indices)
                            l_ptr = 0
                        batch.append(self.logic_samples[logic_indices[l_ptr]])
                        l_ptr += 1
                    elif self.relational_samples:
                        if r_ptr >= len(self.relational_samples):
                            random.shuffle(rel_indices)
                            r_ptr = 0
                        batch.append(self.relational_samples[rel_indices[r_ptr]])
                        r_ptr += 1

            yield torch.stack(batch).pin_memory() # 🚀 PIN MEMORY FOR FAST PCIe LOAD

def train_hybrid():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 🧬 Protocol v6.8: Use the correct tokenizer for each model variant
    # mamba-130m was pretrained on gpt-neox-20b tokenizer — must match for embeddings to work
    if args.grafted:
        from transformers import AutoTokenizer as _AutoTok
        tokenizer = _AutoTok.from_pretrained("EleutherAI/gpt-neox-20b")
        print("🧬 [GRAFTED] Using gpt-neox-20b tokenizer (matches mamba-130m pretraining)")
    else:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    tokenizer.pad_token = tokenizer.eos_token

    # 🚀 PROTOCOL v6.0: Full Reasoning Deliberation
    if args.grafted:
        # 🧬 Protocol v6.8: Grafted mamba-130m checkpoint — use its exact dimensions
        config = Config(
            vocab_size=len(tokenizer),
            d_model=768,
            n_layers=24,
            seq_len=SEQ_LEN,
            n_reasoning=3,
            n_memory_layers=2,
            memory_d_state=32,
        )
        print("🧬 [GRAFTED MODE] d_model=768 | n_layers=24 | memory_layers=2 | d_state=32")
    else:
        config = Config(
            vocab_size=len(tokenizer),
            d_model=1024,
            n_layers=8,
            seq_len=SEQ_LEN,
            n_reasoning=1
        )

    model = RecursiveMambaLM(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    phase_scheduler = PhaseShiftLRScheduler(optimizer, min_lr=1e-6, decay_steps=1000, burn_in_steps=150)

    stats = {
        "train_loss": [],
        "val_loss": [],
        "salads": [],
        "tps": 0.0,
        "step": 0,
        "epoch": 0,
        "loss": 0.0,
        "diverged": False,
        "avg_bias": 10.0,
        "lr": LR,
        "socratic_results": []
    }
    
    bias_history = []
    scaler = AutoNScaler(target_bias=1.5, cooldown=250)
    
    # 💎 RESUME LOGIC (Protocol v6.0 Safe-Save)
    ckpt_path = "rbm_grafted_checkpoint.pt" if args.grafted else "latest_checkpoint.pt"
    if os.path.exists(ckpt_path):
        print(f"♻️ Resuming Protocol v6.0 from {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model_state"], strict=False)
        if not args.grafted:
            optimizer.load_state_dict(state["optimizer_state"])
        stats["step"] = state.get("step", 0)
        
        # 🧠 Load Reasoning Depth & Scaler State
        if "n_reasoning" in state:
            model.config.n_reasoning = state["n_reasoning"]
            print(f"🧠 Reasoning Depth Resumed: N={model.config.n_reasoning}")
        else:
            model.config.n_reasoning = 2
            print(f"🧠 Reasoning Depth Forced: N={model.config.n_reasoning} (Resume Fallback)")

        if "last_scale_step" in state:
            scaler.last_scale_step = state["last_scale_step"]
        if "bias_history" in state:
            bias_history = state["bias_history"]
            
        # 🔗 Fallback Sync Scaler: Only if not found in ckpt
        if "last_scale_step" not in state and stats["step"] % CHECKPOINT_FREQ == 0:
            scaler.last_scale_step = stats["step"] - 200 # Give it some head start
            
        # 🛡️ PROTOCOL v6.5.1: Memory Sanitation
        del state
        gc.collect()
        torch.cuda.empty_cache()
        print("🧹 Redundant weights cleared from System RAM/VRAM.")
        
        # 🔥 SGDR Warm Restart (triggered by --lr_restart flag)
        if args.lr_restart:
            phase_scheduler.warm_restart(stats["step"])
            print(f"🔥 LR Warm Restart applied. New LR: {args.lr:.2e}")

        # 🧠 Protocol v6.7: Manual Path A Fallback (triggered by --force_path_a flag)
        if args.force_path_a:
            model.config.n_reasoning = 1
            stats["path_a_fallback_active"] = True
            stats["path_a_fallback_start"] = stats["step"]
            print(f"\n⚠️  [MANUAL PATH A FALLBACK] Forced at Step {stats['step']}. Running N=1 for 500 consolidation steps.")
            print(f"⚠️  N=3 will auto-restore at Step {stats['step'] + 500}.")

        # 🧬 Protocol v6.8: Freeze Path A weights in grafted mode
        # Path A = pretrained 300B-token knowledge anchor (frozen)
        # Path B + memory layers = logic/QA fine-tuning target (trainable)
        if args.grafted:
            frozen_count = 0
            for layer in model.layers:
                m = layer["mamba"]
                for param in [m.in_proj.parameters(), m.out_proj.parameters(),
                               m.a_conv1d.parameters(), m.a_x_proj.parameters(),
                               m.a_dt_proj.parameters()]:
                    for p in param:
                        p.requires_grad = False
                        frozen_count += p.numel()
                m.a_A_log.requires_grad = False
                m.a_D.requires_grad    = False
                frozen_count += m.a_A_log.numel() + m.a_D.numel()
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n🧬 [GRAFTED] Path A frozen: {frozen_count:,} params | Trainable: {trainable:,} params")
            # Rebuild optimizer with only trainable params
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR, weight_decay=0.05
            )
            stats["step"] = 0  # Fresh fine-tune step counter

    else:
        print("💎 Protocol v6.0 Initialized: STOCHASTIC MANIFOLD MODE")

    loader = HybridDataLoader(tokenizer, SEQ_LEN, BATCH_SIZE, ratio=HYBRID_RATIO)
    watchdog = ThermalWatchdog(threshold=85)
    start_time = time.time()
    initial_step = stats["step"]

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(loader):
            # 🛡️ Protocol v6.4: Thermal Watchdog check
            if watchdog.wait_if_needed():
                start_time = time.time() # Reset clock to avoid TPS skew after pause
                
            batch = batch.to(device)
            inputs = batch[:, :-1].contiguous()
            targets = batch[:, 1:].contiguous()
            
            # 🚀 PROTOCOL v6.4.2: AUTO-N-SCALER
            # Depth is now scaled dynamically based on Narrative Bias scores
            pass # depth updated in diagnostic block below

            try:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(inputs)
                    # 🛡️ PROTOCOL v6.3: Increased Label Smoothing (0.1) to break Mode Collapse
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1), label_smoothing=0.1)
                
                (loss / ACCUM_STEPS).backward()
            except torch.cuda.OutOfMemoryError:
                print("🚨 OOM CAUGHT: Performing Emergency Cache Clear and Force-Save...")
                torch.cuda.empty_cache()
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "step": stats["step"],
                    "epoch": epoch
                }, "panic_checkpoint.pt")
                # Resume attempt or exit
                continue
            
            if (i + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # 🛡️ Protocol v6.6: Gradient Noise Injection (every 50 steps)
                # Adds σ=0.01 Gaussian noise to all gradients to prevent over-fitting
                # to sharp surface token patterns (fixes gradient sensitivity: 0.37 → target <0.1)
                if stats["step"] % 50 == 0:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.add_(torch.randn_like(p.grad) * 0.01)
                
                optimizer.step()
                optimizer.zero_grad()
                stats["step"] += 1
                
                # 🧠 Protocol v6.7: Automatic Path A Fallback Trigger
                # At Step 7100, check the rolling 50-step loss average.
                # If loss hasn't broken below 4.0, Path B representations aren't
                # reliable enough to guide themselves. Drop to N=1 (Path A only)
                # for 500 dedicated consolidation steps, then restore N=3.
                PATH_A_LOCK_STEP   = 7100   # Evaluation checkpoint
                PATH_A_DURATION    = 500    # Steps to run Path A only
                PATH_A_LOSS_TARGET = 4.0    # Loss threshold to stay at N=3
                PATH_B_RESTORE_STEP = PATH_A_LOCK_STEP + PATH_A_DURATION

                if stats["step"] == PATH_A_LOCK_STEP:
                    recent_losses = stats["train_loss"][-50:] if len(stats["train_loss"]) >= 50 else stats["train_loss"]
                    rolling_avg = sum(recent_losses) / max(len(recent_losses), 1)
                    if rolling_avg > PATH_A_LOSS_TARGET:
                        print(f"\n⚠️  [PATH A FALLBACK] Step {PATH_A_LOCK_STEP}: Rolling loss avg {rolling_avg:.4f} > {PATH_A_LOSS_TARGET}")
                        print(f"⚠️  Path B representations are not reliable. Dropping to N=1 for {PATH_A_DURATION} consolidation steps.")
                        model.config.n_reasoning = 1
                        stats["path_a_fallback_active"] = True
                    else:
                        print(f"\n✅  [PATH A CHECK] Step {PATH_A_LOCK_STEP}: Rolling loss avg {rolling_avg:.4f} ≤ {PATH_A_LOSS_TARGET}. N=3 maintained.")
                        stats["path_a_fallback_active"] = False

                # Re-introduce Path B with a graduated ramp to prevent loss spike:
                # N=1 → N=2 (200 step buffer) → N=3 (permanent)
                # Works for both auto-trigger (step 7100) and --force_path_a (any step)
                if stats.get("path_a_fallback_active"):
                    fallback_start = stats.get("path_a_fallback_start", PATH_A_LOCK_STEP)
                    n2_restore_at  = fallback_start + 500   # Step to introduce N=2
                    n3_restore_at  = n2_restore_at + 200    # Step to unlock N=3

                    if stats["step"] == n2_restore_at:
                        print(f"\n🔁  [RAMP N=2] Step {n2_restore_at}: Path A complete. Introducing N=2 for 200-step buffer.")
                        model.config.n_reasoning = 2
                        stats["path_a_ramp_n2"] = True

                    if stats.get("path_a_ramp_n2") and stats["step"] == n3_restore_at:
                        print(f"\n✅  [RAMP N=3] Step {n3_restore_at}: N=2 buffer complete. Unlocking full N=3.")
                        model.config.n_reasoning = 3
                        stats["path_a_fallback_active"] = False
                        stats["path_a_ramp_n2"] = False

                # 🚀 PHASE SHIFT: Protocol v6.5
                stats["lr"] = phase_scheduler.step(stats["step"], model.config.n_reasoning)

                
                # Update Telem every 10 complete steps
                if stats["step"] % 10 == 0:
                    steps_done = stats["step"] - initial_step
                    elapsed = time.time() - start_time
                    tps = (steps_done * BATCH_SIZE * ACCUM_STEPS * SEQ_LEN) / (elapsed + 1e-6)
                    stats["tps"] = tps
                    stats["loss"] = loss.item()
                    stats["train_loss"].append(loss.item())
                    
                    vram_used = torch.cuda.memory_reserved() / 1e9
                    
                    # 🛡️ Protocol v6.4.1: Live CPU Telemetry
                    cpu_temp = "N/A"
                    try:
                        temps = psutil.sensors_temperatures()
                        if 'coretemp' in temps:
                            max_c = max(t.current for t in temps['coretemp'])
                            cpu_temp = f"{max_c:.1f}°C"
                    except:
                        pass
                    
                    stats["cpu_temp"] = cpu_temp
                    stats["epoch"] = epoch + 1
                    print(f"Phase VI.2 | Epoch {epoch+1} | Step {stats['step']} (N={model.config.n_reasoning}) | Loss: {loss.item():.4f} | LR: {stats['lr']:.2e} | TPS: {tps:.0f} | VRAM: {vram_used:.2f}GB | CPU: {cpu_temp}")
                    
                    if vram_used > 10.5: # 🛡️ Threshold for dangerous fragmentation
                        torch.cuda.empty_cache()
                    
                    with open("training_stats.json", "w") as f:
                        json.dump(stats, f)

                # 🛡️ PERIODIC SAVE (Protocol v6.4)
                if stats["step"] % CHECKPOINT_FREQ == 0:
                    print(f"💾 Protocol v6.0 Safe-Save at Step {stats['step']}")
                    torch.save({
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "step": stats["step"],
                        "epoch": epoch,
                        "n_reasoning": model.config.n_reasoning,
                        "last_scale_step": scaler.last_scale_step,
                        "bias_history": bias_history
                    }, "latest_checkpoint.pt")

            # --- Periodic Diagnostic Sampling ---
            if stats["step"] > 0 and stats["step"] % 100 == 0 and (i + 1) % ACCUM_STEPS == 0:
                # 🛡️ Protocol v6.6: Rotating Reasoning Probes (Math, Logic, General)
                probe_list = [
                    "Question: If a car travels 60 miles per hour, how far will it travel in 2 hours? Answer:", # Math
                    "Context: Alice is taller than Bob. Bob is taller than Charlie. Question: Who is the shortest? Answer:", # Logic
                    "What is the capital of France?" # General Knowledge
                ]
                probe_index = (stats["step"] // 100) % len(probe_list)
                test_prompt = probe_list[probe_index]
                
                prompt_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    output_ids = model.generate(prompt_ids, max_new_tokens=64)
                    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                stats["salads"].append([{
                    "type": "HYBRID_PROBE",
                    "prompt": test_prompt,
                    "response": response,
                    "steps": config.n_reasoning,
                    "mode": "RBM_HYBRID"
                }])

                # Calculate Bias and Scale N
                b_score = calculate_bias(response)
                bias_history.append(b_score)
                if len(bias_history) > 5:
                    bias_history.pop(0)
                
                avg_bias = sum(bias_history) / len(bias_history)
                stats["avg_bias"] = avg_bias
                
                # Funnel to UI Feed
                stats["socratic_results"].append({
                    "step": stats["step"],
                    "prompt": test_prompt,
                    "response": response,
                    "bias": b_score,
                    "steps": config.n_reasoning
                })
                if len(stats["socratic_results"]) > 20:
                    stats["socratic_results"].pop(0)
                
                old_n = model.config.n_reasoning
                new_n = scaler.check_and_scale(old_n, avg_bias, stats["step"])
                model.config.n_reasoning = new_n
                
                if new_n != old_n:
                    print(f"💎 Manifold Elevation: N={old_n} -> N={new_n} (Avg Bias: {avg_bias:.2f})")
                    
                    if new_n == 3:
                        from tri_state_benchmark import run_tri_state_benchmark
                        results = run_tri_state_benchmark(model, tokenizer, stats["step"], n_depth=new_n)
                        stats["benchmarks"] = results # Dedicated UI container
                        stats["salads"].append(results) # Keep in historical feed

                model.train()

        # Save Checkpoint
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch
        }, f"rbm_hybrid_epoch_{epoch+1}.pt")
        # scheduler.step() removed in favor of PhaseShiftLRScheduler

if __name__ == "__main__":
    train_hybrid()
