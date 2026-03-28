"""
train_130m.py — Fully Automated 3-Phase RLF Training Pipeline for Mamba-130M
=============================================================================
Designed specifically for the 130M parameter model. Key differences from the
2.7B pipeline:

  Phase 1 — Warmup (clean chains, 2-5 hops, small LR, short run)
    - 1500 steps, batch=8, LR=1e-3
    - Data: 3000 unique prompts, hops 2-5 only
    - Stop early at rolling(100) acc >= 0.90

  Phase 2 — Joint Generalization (clean + longer chains, LR decay)
    - 2000 steps, batch=8, LR=5e-4
    - Data: 6000 prompts, hops 2-8 (wider distribution)
    - Stop early at rolling(100) acc >= 0.95

  Phase 3 — Adversarial Hardening (chaos + prose distractors, fine-tune LR)
    - 1500 steps, batch=4, LR=1e-4
    - Data: 4000 prompts with distractors, hops 2-8
    - Stop early at rolling(100) acc >= 0.92

All phases auto-chain. Final checkpoint saved as mamba130m_v2_best.pt.

Usage:
    python train_130m.py
    python train_130m.py --phase 2   # start from phase 2
    python train_130m.py --phase 3   # start from phase 3
"""

import torch
import random
import os
import string
import time
import argparse
import sys
from torch.utils.data import Dataset, DataLoader
from mamba_ssm import MambaLMHeadModel
from mamba1_engine import (
    RecursiveMamba1_PrefixScratchpad, MODEL_ID, tokenizer, HALT_ID as ENGINE_HALT_ID,
    freeze_for_phase1, get_phase1_optimizer
)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
HALT_ID   = ENGINE_HALT_ID          # 50278 — must match mamba1_engine's HALT_ID
PAD_ID    = tokenizer.eos_token_id  # 0 — padding token
SAVE_DIR  = "saved_weights"

# ── 130M-specific scaling constants ──────────────────────────────────────────
# The 2.7B pipeline used 10k-12k steps. 130M needs far less data to converge.
# Rule: ~10× fewer steps/data vs 2.7B, wider hop range for generalization.

# Phase 1: scratchpad warmup — simple numeric chains, no distractors
PHASE1 = dict(
    steps        = 1500,
    batch        = 8,
    lr           = 1e-3,
    data_size    = 3000,
    hop_min      = 2, hop_max = 5,
    adversarial  = False,
    sparse_reward = False,  # dense — bridge initialises on clean chains
    n_dark_loops  = 0,
    stop_acc     = 0.90,
    stop_after   = 200,
    log_every    = 50,
    ckpt_every   = 300,
    ckpt_name    = "mamba130m_v2_phase1",
)

# Phase 2: dense reward + numeric entropy + 20% word vocab mixing
# No sparse pressure yet — fully learn the routing task first.
# Mixed vocab forces the scratchpad routing to be vocabulary-agnostic,
# which closes the 0% novel-vocab OOD gap.
PHASE2 = dict(
    steps        = 6000,    # was 3000 — let it fully converge
    batch        = 8,
    lr           = 5e-4,
    data_size    = 8000,    # larger pool = more diversity per epoch
    hop_min      = 2, hop_max = 8,
    adversarial  = False,
    mixed_vocab  = True,    # 80% numeric, 20% word tokens
    sparse_reward = False,
    n_dark_loops  = 0,
    stop_acc     = 0.95,    # higher bar — must hit 95% ans_acc
    stop_after   = 2000,    # must sustain for 2000 steps before stopping
    log_every    = 50,
    ckpt_every   = 500,
    ckpt_name    = "mamba130m_v3_phase2",
)

# Phase 2b — Dense Adversarial Warmup: Lifeline ON, no dark loops
# Teach the model what chameleon distractors look like BEFORE dark loop pressure.
# The model had never seen sys=819 / buf=33 style noise during Phase 2.
# This step closes the missing rung: learn distractor filtering with full Lifeline support.
PHASE2B = dict(
    steps        = 2000,
    batch        = 8,
    lr           = 3e-4,
    data_size    = 6000,
    hop_min      = 2, hop_max = 8,
    adversarial  = True,    # chameleon distractors ON
    mixed_vocab  = True,    # 80% numeric, 20% word
    sparse_reward = False,  # dense — Lifeline ON, no dark loops
    n_dark_loops  = 0,
    stop_acc     = 0.90,
    stop_after   = 400,
    log_every    = 50,
    ckpt_every   = 300,
    ckpt_name    = "mamba130m_v5_phase2b",
)

# Phase 3 — The Primer: n_dark=2, adversarial distractors
# Two dark loops are enough to decouple the model from the Lifeline and force
# it to realise the prefix scratchpad is its only source of truth.
# The optimizer builds the "connective tissue" to hold latent state without
# the 3-loop void that caused Phase 3 to drown in v3.
# loss_weights: [dark_0=0.1, dark_1=0.2, reward_ans=1.0, reward_halt=1.0]
PHASE3 = dict(
    steps        = 4000,
    batch        = 4,
    lr           = 1e-4,
    data_size    = 6000,
    hop_min      = 2, hop_max = 8,
    adversarial  = True,
    mixed_vocab  = True,
    sparse_reward = True,
    n_dark_loops  = 2,
    loss_weights  = [0.1, 0.2, 1.0, 1.0],  # progressive: 2 dark + 2 reward
    stop_acc     = 0.75,
    stop_after   = 800,
    log_every    = 50,
    ckpt_every   = 300,
    ckpt_name    = "mamba130m_v5_phase3",
)

# Phase 4 — The Crucible: n_dark=3, full stack pressure
# Once the 2-loop hold is locked in, pull the floor out one more sweep.
# loss_weights: [dark_0=0.05, dark_1=0.1, dark_2=0.2, reward_ans=1.0, halt=1.0]
PHASE4 = dict(
    steps        = 4000,
    batch        = 4,
    lr           = 5e-5,
    data_size    = 6000,
    hop_min      = 2, hop_max = 10,
    adversarial  = True,
    mixed_vocab  = True,
    sparse_reward = True,
    n_dark_loops  = 3,
    loss_weights  = [0.1, 0.2, 0.4, 1.0, 1.0],   # progressive: 3 dark + 2 reward
    stop_acc     = 0.70,
    stop_after   = 800,
    log_every    = 50,
    ckpt_every   = 300,
    ckpt_name    = "mamba130m_v5_phase4",
)

# Phase 5 — Post-Training Annealing (Dense Cleanup)
# The forge (Phase 4) taught novel-vocab routing but dulled numeric precision.
# 1000 steps of dense reward with Lifeline ON acts as a cooling/annealing pass:
#   - Restores clean numeric routing (recovering the 11%→3% drop)
#   - Doesn't erase novel-vocab LoRA weights — just sharpens them
#   - Low LR (1e-5) to avoid catastrophic forgetting of Phase 4 gains
PHASE5 = dict(
    steps        = 1000,
    batch        = 8,
    lr           = 1e-5,        # recovery LR — fine annealing, not overwrite
    data_size    = 4000,
    hop_min      = 2, hop_max = 10,
    adversarial  = True,        # keep adversarial ON — sharpen distractor filter
    mixed_vocab  = True,        # keep novel vocab ON — don't regress
    sparse_reward = False,      # dense: Lifeline ON, no dark loops
    n_dark_loops  = 0,
    loss_weights  = None,
    stop_acc     = 0.90,
    stop_after   = 400,
    log_every    = 50,
    ckpt_every   = 200,
    ckpt_name    = "mamba130m_v5_phase5",
)

# Phase 6 — Syntactic Expansion (The bAbI Fix)
# Train the Phase 5 weights on a dataset where the routing logic is identical,
# but the grammatical syntax is highly randomized.
PHASE6 = dict(
    steps        = 2000,
    batch        = 8,
    lr           = 1e-4,
    data_size    = 8000,
    hop_min      = 2, hop_max = 6,
    adversarial  = False,
    mixed_vocab  = True,
    sparse_reward = False,
    n_dark_loops  = 0,
    syntax_var    = True,
    loss_weights  = None,
    stop_acc     = 0.90,
    stop_after   = 400,
    log_every    = 50,
    ckpt_every   = 200,
    ckpt_name    = "mamba130m_v6_phase6",
)

# Chameleon distractor variable names — look like real chain variables
# but are lowercase/short to visually blend with numeric payloads
_DISTRACTOR_KEYS = [
    "sys", "env", "tmp", "buf", "idx", "ptr", "cnt", "sum",
    "val", "key", "ref", "aux", "err", "bit", "reg", "mem",
]

# Numeric payload range — BPE tokenizer slices these into 1-3 tokens,
# breaking the loop-counter-as-index cheat code.
NUM_MIN = 1
NUM_MAX = 999_999

# Word vocab for mixed-vocab training (20% of Phase 2/3 samples).
# These are single-token words — forcing the model to learn that the routing
# mechanism is vocab-agnostic, not number-specific.
WORD_VALS = [
    "Blue", "Red", "Cat", "Dog", "Sun", "Moon", "Fire", "Star",
    "Gold", "Ice", "Sky", "Sea", "Oak", "Elm", "Ash", "Fox",
    "Owl", "Bat", "Bee", "Ant", "Alpha", "Beta", "Gamma", "Delta",
    "Zinc", "Flux", "Rune", "Dusk", "Gust", "Wren", "Crest", "Hazel",
    "Quartz", "Monk", "Vex", "Pyre", "Plume", "Knot", "Onyx", "Cyan",
    "True", "Zero", "Max", "Min", "Hex", "Key", "Arc", "Lux", "Nova", "Shard",
]


def _rand_num(rng: random.Random) -> str:
    """Generate a random integer string payload."""
    return str(rng.randint(NUM_MIN, NUM_MAX))


# ── Dataset ───────────────────────────────────────────────────────────────────
class Chain130MDataset(Dataset):
    """Variable-hop chain dataset — Numeric Entropy edition.

    Payloads are random integers 1–999,999 (multi-token BPE).
    Distractors are chameleon numeric assignments that look identical
    to real chain entries, forcing the model to read the = operator.

    Clean:       V1=48291. V2=V1. What is V2? Answer:
    Adversarial: sys=819. V1=48291. buf=33. V2=V1. What is V2? Answer:
    """

    def __init__(
        self,
        size: int,
        hop_min: int,
        hop_max: int,
        adversarial: bool = False,
        mixed_vocab: bool = False,
        syntax_var: bool = False,
        seed: int = 42,
    ) -> None:
        """Initialize dataset with given configuration.

        Args:
            mixed_vocab: If True, 20% of samples use word payloads from WORD_VALS
                instead of numeric payloads. Forces vocab-agnostic scratchpad routing.
            syntax_var: If True, use semantic grammar templates instead of V1=val.
        """
        self.size        = size
        self.hop_min     = hop_min
        self.hop_max     = hop_max
        self.adv         = adversarial
        self.mixed_vocab = mixed_vocab
        self.syntax_var  = syntax_var
        self.seed        = seed

    def __len__(self) -> int:
        """Return dataset size."""
        return self.size

    def __getitem__(self, idx: int) -> dict:
        """Generate and return a training sample with numeric or word payload."""
        rng  = random.Random(self.seed + idx * 31337)
        hops = rng.randint(self.hop_min, self.hop_max)

        # 20% word vocab when mixed_vocab=True — forces vocab-agnostic routing
        if self.mixed_vocab and rng.random() < 0.20:
            val = rng.choice(WORD_VALS)
        else:
            val = _rand_num(rng)  # e.g. "48291"

        # Core chain: V1=<num>. V2=V1. ... Vn=Vn-1. What is Vn? Answer:
        if self.syntax_var:
            init_tmpls = [
                "V1={}.",
                "Let V1 be {}.",
                "V1 is assigned {}.",
                "The value of V1 is {}.",
                "V1 equals {}."
            ]
            hop_tmpls = [
                "V{}=V{}.",
                "V{} holds the value of V{}.",
                "Let V{} equal V{}.",
                "V{} equals V{}.",
                "The value of V{} is given to V{}."
            ]
            chain_parts = [rng.choice(init_tmpls).format(val)]
            for i in range(2, hops + 1):
                chain_parts.append(rng.choice(hop_tmpls).format(i, i-1))
            chain_parts.append(f"What is V{hops}? Answer:")
        else:
            chain_parts = [f"V1={val}."]
            for i in range(2, hops + 1):
                chain_parts.append(f"V{i}=V{i-1}.")
            chain_parts.append(f"What is V{hops}? Answer:")

        if self.adv:
            # Chameleon distractors: fake numeric assignments.
            # They look EXACTLY like real chain entries (key=number.)
            # but use lowercase distractor keys that are not V1..Vn.
            distractors = []
            for _ in range(rng.randint(2, 5)):
                dk  = rng.choice(_DISTRACTOR_KEYS)
                dv  = _rand_num(rng)
                distractors.append(f"{dk}={dv}.")
            rng.shuffle(distractors)
            # Interleave distractors before and within the chain
            n_before = rng.randint(1, max(1, len(distractors) - 1))
            prompt = " ".join(
                distractors[:n_before] + chain_parts + distractors[n_before:]
            )
        else:
            prompt = " ".join(chain_parts)

        # Build target_ids for sparse reward:
        #   loop 0 → answer token(s) — first BPE token of the numeric answer
        #   loop 1 → HALT_ID
        # NOTE: numeric values tokenize to multiple sub-tokens; we supervise
        # only the FIRST sub-token so the target stays length-2 and consistent
        # with the engine's chain_targets[loop_i] indexing.
        input_ids  = tokenizer.encode(prompt, add_special_tokens=False)
        ans_start  = len(input_ids) - 1  # position of ':'

        # Encode " <num>" — the leading space matters for GPT-NeoX BPE
        val_toks   = tokenizer.encode(" " + val, add_special_tokens=False)
        answer_tok = val_toks[0]   # first (and often only) sub-token
        target_ids = [answer_tok, HALT_ID]

        return {
            "input_ids":  torch.tensor(input_ids,  dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "ans_start":  ans_start,
            # Store full val_toks so eval can check complete number later
            "val_str":    val,
        }



def collate_fn(batch: list[dict]) -> tuple:
    """Pad and stack a batch of samples."""
    input_ids  = [b["input_ids"]  for b in batch]
    target_ids = [b["target_ids"] for b in batch]
    ans_starts = torch.tensor([b["ans_start"] for b in batch])

    max_in  = max(len(x) for x in input_ids)
    max_tgt = max(len(x) for x in target_ids)

    inp_pad = torch.stack([
        torch.nn.functional.pad(x, (0, max_in  - len(x)), value=PAD_ID)
        for x in input_ids
    ])
    tgt_pad = torch.stack([
        torch.nn.functional.pad(x, (0, max_tgt - len(x)), value=PAD_ID)
        for x in target_ids
    ])
    return inp_pad, tgt_pad, ans_starts


# ── Phase runner ──────────────────────────────────────────────────────────────
def run_phase(
    model: RecursiveMamba1_PrefixScratchpad,
    cfg: dict,
    phase_num: int,
    log_handle,
) -> RecursiveMamba1_PrefixScratchpad:
    """Run a single training phase and return the model.

    Args:
        model: The model to train (in place).
        cfg: Phase configuration dict.
        phase_num: 1, 2, or 3.
        log_handle: Open file handle for the training log.

    Returns:
        Model with best weights loaded.
    """
    name = cfg["ckpt_name"]
    print(f"\n{'='*70}")
    print(f"  PHASE {phase_num}: {name}")
    print(f"  Steps: {cfg['steps']} | Batch: {cfg['batch']} | LR: {cfg['lr']} | Hops: {cfg['hop_min']}-{cfg['hop_max']}")
    print(f"{'='*70}\n")

    dataset = Chain130MDataset(
        size        = cfg["data_size"],
        hop_min     = cfg["hop_min"],
        hop_max     = cfg["hop_max"],
        adversarial = cfg.get("adversarial", False),
        mixed_vocab = cfg.get("mixed_vocab", False),
        syntax_var  = cfg.get("syntax_var", False),
        seed        = phase_num * 999,
    )
    loader = DataLoader(
        dataset,
        batch_size  = cfg["batch"],
        shuffle     = True,
        collate_fn  = collate_fn,
        drop_last   = True,
        num_workers = 0,
    )

    # Optimizer — different freeze policy per phase
    if phase_num == 1:
        freeze_for_phase1(model)
        optimizer = get_phase1_optimizer(model)
    else:
        # Phases 2 and 3: unfreeze LoRA + engine layers, keep base frozen
        for name_p, param in model.named_parameters():
            need = any(k in name_p.lower() for k in
                       ("lora", "loop_", "bridge", "prefix", "lifeline", "engram", "norm"))
            param.requires_grad = need
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params: {trainable:,}")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["lr"], weight_decay=0.01,
        )

    best_acc  = 0.0
    best_path = f"{SAVE_DIR}/{cfg['ckpt_name']}_best.pt"
    rolling   = []
    step      = 0
    stopped   = False

    model.train()
    t_start = time.time()

    while step < cfg["steps"] and not stopped:
        for inp, tgt, ans in loader:
            if step >= cfg["steps"]:
                break

            inp = inp.to(DEVICE)
            tgt = tgt.to(DEVICE)
            ans = ans.to(DEVICE)

            optimizer.zero_grad()
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                loss, acc, ans_acc, halt_acc = model(
                    inp, chain_targets=tgt, ans_starts=ans,
                    sparse_reward=cfg.get("sparse_reward", False),
                    n_dark_loops=cfg.get("n_dark_loops", 0),
                    loss_weights=cfg.get("loss_weights", None),
                )
                if torch.isnan(loss):
                    step += 1
                    continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Fix 4: checkpoint and roll on ANS_ACC, not avg_acc.
            # avg_acc mixed halt_acc (always ~1.0) with ans_acc, masking total failure.
            rolling.append(float(ans_acc))
            if len(rolling) > 100:
                rolling.pop(0)
            roll_avg = sum(rolling) / len(rolling)

            # Save best checkpoint on ans_acc only
            if float(ans_acc) > best_acc:
                best_acc = float(ans_acc)
                torch.save(model.state_dict(), best_path)

            # Log every N steps
            if step % cfg["log_every"] == 0:
                elapsed = time.time() - t_start
                line = (
                    f"P{phase_num} Step {step:5d} | Loss {loss.item():.4f} | "
                    f"RLF {acc:.2f} | Ans {ans_acc:.2f} | Halt {halt_acc:.2f} | "
                    f"Roll {roll_avg:.2f} | Best {best_acc:.2f} | {elapsed:.0f}s"
                )
                print(line)
                log_handle.write(line + "\n")
                log_handle.flush()

            # Periodic checkpoint
            if step > 0 and step % cfg["ckpt_every"] == 0:
                ckpt = f"{SAVE_DIR}/{cfg['ckpt_name']}_step{step}.pt"
                torch.save(model.state_dict(), ckpt)

            # Early stop — set flag to break outer while loop too
            if step >= cfg["stop_after"] and roll_avg >= cfg["stop_acc"]:
                msg = f"  ✅ Early stop at step {step} — Roll(100) acc {roll_avg:.3f} >= {cfg['stop_acc']}"
                print(msg)
                log_handle.write(msg + "\n")
                stopped = True
                break

            step += 1

    # Load best weights before returning
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    msg = f"\nPhase {phase_num} done. Best acc: {best_acc:.3f} → {best_path}\n"
    print(msg)
    log_handle.write(msg + "\n")
    return model


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main() -> None:
    """Run the full training pipeline from Phase 1 through Phase 3."""
    parser = argparse.ArgumentParser(description="Mamba-130M Automated Training Pipeline")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7],
                        help="Start from this phase (default: 1)")
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    log_path = "training_130m.log"

    print(f"\n{'='*70}")
    print("  MAMBA-130M AUTOMATED TRAINING PIPELINE")
    print(f"  Device: {DEVICE.upper()} | Start phase: {args.phase}")
    print(f"  Log: {log_path}")
    print(f"{'='*70}\n")

    with open(log_path, "a") as log:
        log.write(f"\n=== PIPELINE START {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

        # ── Load backbone ──────────────────────────────────────────────────────
        print("[INIT] Loading backbone and building model…")
        backbone = MambaLMHeadModel.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device=DEVICE
        )
        model = RecursiveMamba1_PrefixScratchpad(backbone, lora_rank=4).to(DEVICE)

        # ── Phase 1 ────────────────────────────────────────────────────────────
        if args.phase <= 1:
            model = run_phase(model, PHASE1, 1, log)
        else:
            # Load Phase 1 output to start Phase 2 or 3
            p1_ckpt = f"{SAVE_DIR}/mamba130m_v2_phase1_best.pt"
            if not os.path.exists(p1_ckpt):
                print(f"[ERROR] Phase 1 checkpoint not found: {p1_ckpt}")
                sys.exit(1)
            model.load_state_dict(torch.load(p1_ckpt, map_location=DEVICE))
            print(f"[SKIP] Loaded Phase 1 checkpoint: {p1_ckpt}")

        # ── Phase 2 ────────────────────────────────────────────────────────────
        if args.phase <= 2:
            model = run_phase(model, PHASE2, 2, log)
        else:
            p2_ckpt = f"{SAVE_DIR}/mamba130m_v3_phase2_best.pt"
            if not os.path.exists(p2_ckpt):
                print(f"[ERROR] Phase 2 checkpoint not found: {p2_ckpt}")
                sys.exit(1)
            model.load_state_dict(torch.load(p2_ckpt, map_location=DEVICE))
            print(f"[SKIP] Loaded Phase 2 checkpoint: {p2_ckpt}")

        # ── Phase 2b (Dense adversarial warmup) ───────────────────────────────
        if args.phase <= 3:
            model = run_phase(model, PHASE2B, 2, log)  # phase_num=2: same optimizer
        else:
            p2b_ckpt = f"{SAVE_DIR}/mamba130m_v5_phase2b_best.pt"
            if not os.path.exists(p2b_ckpt):
                # Fall back to Phase 3 ckpt if 2b wasn't produced
                p2b_ckpt = f"{SAVE_DIR}/mamba130m_v4_phase3_best.pt"
            model.load_state_dict(torch.load(p2b_ckpt, map_location=DEVICE))
            print(f"[SKIP] Loaded Phase 2b checkpoint: {p2b_ckpt}")

        # ── Phase 3 (Primer: n_dark=2) ─────────────────────────────────────────
        if args.phase <= 3:
            model = run_phase(model, PHASE3, 3, log)
        else:
            p3_ckpt = f"{SAVE_DIR}/mamba130m_v4_phase3_best.pt"
            if not os.path.exists(p3_ckpt):
                print(f"[ERROR] Phase 3 checkpoint not found: {p3_ckpt}")
                sys.exit(1)
            model.load_state_dict(torch.load(p3_ckpt, map_location=DEVICE))
            print(f"[SKIP] Loaded Phase 3 checkpoint: {p3_ckpt}")

        # ── Phase 4 (Crucible: n_dark=3) ───────────────────────────────────────
        if args.phase <= 4:
            model = run_phase(model, PHASE4, 4, log)
        else:
            p4_ckpt = f"{SAVE_DIR}/mamba130m_v5_phase4_best.pt"
            if not os.path.exists(p4_ckpt):
                print(f"[ERROR] Phase 4 checkpoint not found: {p4_ckpt}")
                sys.exit(1)
            model.load_state_dict(torch.load(p4_ckpt, map_location=DEVICE))
            print(f"[SKIP] Loaded Phase 4 checkpoint: {p4_ckpt}")

        # Phase 5 — Dense Cleanup Annealing
        if args.phase <= 5:
            model = run_phase(model, PHASE5, 2, log)  # phase_num=2: dense avg optimizer
        else:
            p5_ckpt = f"{SAVE_DIR}/mamba130m_v5_phase5_best.pt"
            if not os.path.exists(p5_ckpt):
                print(f"[ERROR] Phase 5 checkpoint not found: {p5_ckpt}")
                sys.exit(1)
            model.load_state_dict(torch.load(p5_ckpt, map_location=DEVICE))
            print(f"[SKIP] Loaded Phase 5 checkpoint: {p5_ckpt}")

        # Phase 6 — Syntactic Expansion (The bAbI Fix)
        if args.phase <= 6:
            model = run_phase(model, PHASE6, 2, log)  # phase_num=2: dense avg optimizer
        else:
            p6_ckpt = f"{SAVE_DIR}/mamba130m_v6_phase6_best.pt"
            if not os.path.exists(p6_ckpt):
                print(f"[ERROR] Phase 6 checkpoint not found: {p6_ckpt}")
                sys.exit(1)
            model.load_state_dict(torch.load(p6_ckpt, map_location=DEVICE))
            print(f"[SKIP] Loaded Phase 6 checkpoint: {p6_ckpt}")

        # ── Final save ────────────────────────────────────────────────────────
        final_path = f"{SAVE_DIR}/mamba130m_v6_best.pt"
        torch.save(model.state_dict(), final_path)
        msg = f"\n🏁 PIPELINE COMPLETE → {final_path}\n"
        print(msg)
        log.write(msg + "\n")
        log.write(f"=== PIPELINE END {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")


if __name__ == "__main__":
    main()
