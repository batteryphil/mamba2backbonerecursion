"""
rlf_trainer_1_4b.py — 3-Phase RLF Training for Mamba-1.4B
============================================================
Phase 3a: Scratchpad warmup    — 2000 steps, memory + bridge only
Phase 3b: RLF joint training   — 8000 steps, full RLF
Phase 3c: SFT recovery         — 1000 steps, LM head on code/math data

Usage:
    ./run_env.sh rlf_trainer_1_4b.py [--phase {3a,3b,3c,all}] [--resume]
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rlf_engine_1_4b import (
    RecursiveMamba1_PrefixScratchpad,
    freeze_for_phase3a,
    freeze_for_phase3b,
    freeze_for_phase3c,
    load_from_sft_checkpoint,
    tokenizer,
    HALT_ID,
    DEVICE,
)
from rlf_dataset import RLFDataset, collate_rlf

# ── Paths ─────────────────────────────────────────────────────────────────────
SFT_CKPT_DIR = Path("/hdd_data/latent-spacer-checkpoints/best")   # R3 weights
RLF_CKPT_DIR = Path("/hdd_data/rlf-1.4b-checkpoints")
SFT_DATA     = Path("/home/phil/.gemini/antigravity/scratch/tiny-refinement/"
                     "combined_training_data.jsonl")
LOG_PATH     = Path("/home/phil/.gemini/antigravity/scratch/tiny-refinement/"
                     "rlf_trainer.log")

# ── Phase config ──────────────────────────────────────────────────────────────
PHASE_CONFIG = {
    "3a": {"steps": 2000,  "lr_mem": 1e-3,  "lr_bridge": 5e-4, "lr_other": 0.0},
    "3b": {"steps": 8000,  "lr_mem": 5e-4,  "lr_bridge": 2e-4, "lr_other": 1e-4},
    "3c": {"steps": 1000,  "lr_mem": 0.0,   "lr_bridge": 0.0,  "lr_other": 1e-5},
}

LOG_EVERY    = 25
CKPT_EVERY   = 500
BATCH_SIZE   = 1
GRAD_ACCUM   = 8


def setup_logging(log_path: Path) -> logging.Logger:
    """Configure root logger to write to file and stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


# ── SFT Dataset (for Phase 3c recovery) ──────────────────────────────────────

class SFTDataset(Dataset):
    """Simple JSONL dataset for Phase 3c SFT recovery.

    Re-uses the combined_training_data.jsonl from the SFT rounds.
    """

    def __init__(self, path: Path, seq_len: int = 512, limit: int = 20000) -> None:
        """Load JSONL samples.

        Args:
            path:    path to combined_training_data.jsonl
            seq_len: maximum token length
            limit:   max samples to load (Phase 3c is short)
        """
        self.seq_len = seq_len
        self.pad_id  = tokenizer.eos_token_id
        self.samples: list[str] = []
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                line = line.strip().strip('\x00')   # strip null bytes (power-cut artifact)
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue                          # skip any other corrupted records
                text = f"[USER]\n{obj['prompt']}\n========\n[ANSWER]\n{obj['answer']}"
                self.samples.append(text)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Tokenize and pad one sample."""
        ids = tokenizer.encode(self.samples[idx])
        if len(ids) > self.seq_len:
            ids = ids[-self.seq_len :]
        pad = self.seq_len - len(ids)
        ids = ids + [self.pad_id] * pad
        return torch.tensor(ids, dtype=torch.long)


def sft_loss(
    model: RecursiveMamba1_PrefixScratchpad,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Causal LM loss for Phase 3c.

    Run 1 loop of RLF (no chain targets) using the actual engine API:
      _encode()          → (x, res) tuples through all 48 layers
      _lifeline_inject() → 4-arg signature (x_ext, res_ext, x_prompt, res_prompt)
      _apply_norm()      → norm_f(x + res)
    Compute next-token CE to recover code generation ability.
    """
    model.train()
    B = input_ids.shape[0]

    # Full 48-layer backbone encode — returns (hidden, residual)
    x, res = model._encode(input_ids)                      # [B, T, D] each

    # Detach anchors (same as main forward)
    x_prompt   = x.detach().clone()
    res_prompt = res.detach().clone() if res is not None else None

    # Prepend latent scratchpad
    mem     = model.latent_memory.expand(B, -1, -1)        # [B, M, D]
    x_ext   = torch.cat([mem, x], dim=1)                   # [B, M+T, D]
    if res is not None:
        res_pad = torch.zeros(B, model.M, model.d_model,
                              device=res.device, dtype=res.dtype)
        res_ext = torch.cat([res_pad, res], dim=1)
    else:
        res_ext = None

    # Single RLF loop (loop_i=0) — teaches lm_head to decode loop state
    x_ext, res_ext = model._lifeline_inject(x_ext, res_ext, x_prompt, res_prompt)
    x_ext = model.loop_rope(x_ext, 0)
    if res_ext is not None:
        res_ext = model.loop_rope(res_ext, 0)
    x_ext, res_ext = model._run_top_layers(x_ext, res_ext)
    x_ext = x_ext + model.mamba1_loop(x_ext)
    x_ext = model.loop_norm(x_ext)
    x_bridged = x_ext + model.bridge_up(model.bridge_down(x_ext))

    # Slice off prefix, normalise, project
    x_out    = x_bridged[:, model.M :, :]                  # [B, T, D]
    r_out    = res_ext[:, model.M :, :] if res_ext is not None else None
    x_normed = model._apply_norm(x_out, r_out)
    logits   = model.lm_head(x_normed)                     # [B, T, V]

    # Standard causal next-token CE
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=tokenizer.pad_token_id,
    )


def save_ckpt(
    model: RecursiveMamba1_PrefixScratchpad,
    step: int,
    phase: str,
    val_loss: float,
) -> None:
    """Save trainable RLF components to checkpoint directory.

    Only saves the trainable parts — base backbone weights are unchanged
    and can be reloaded from the original SFT checkpoint.
    """
    ckpt = RLF_CKPT_DIR / f"phase{phase}_step{step:06d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(model.latent_memory.data,   ckpt / "latent_memory.pt")
    torch.save(model.bridge_down.state_dict(), ckpt / "bridge_down.pt")
    torch.save(model.bridge_up.state_dict(),   ckpt / "bridge_up.pt")
    torch.save(model.mamba1_loop.state_dict(), ckpt / "mamba1_loop.pt")
    torch.save(model.loop_norm.state_dict(),   ckpt / "loop_norm.pt")
    torch.save(model.lifeline_gate.data,   ckpt / "lifeline_gate.pt")
    torch.save(model.lm_head.state_dict(), ckpt / "lm_head.pt")
    # LoRA weights from top layers
    lora_state = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state[name] = param.data
    torch.save(lora_state, ckpt / "lora.pt")
    logging.getLogger(__name__).info(f"Checkpoint: {ckpt} | val={val_loss:.4f}")


def load_rlf_components(
    model: RecursiveMamba1_PrefixScratchpad,
    ckpt_dir: Path,
) -> None:
    """Load saved RLF components into model (for resume)."""
    log = logging.getLogger(__name__)
    for fname, target in [
        ("latent_memory.pt", None),
        ("bridge_down.pt",   model.bridge_down),
        ("bridge_up.pt",     model.bridge_up),
        ("mamba1_loop.pt",   model.mamba1_loop),
        ("loop_norm.pt",     model.loop_norm),
        ("lifeline_gate.pt", None),
        ("lm_head.pt",       model.lm_head),
    ]:
        fpath = ckpt_dir / fname
        if not fpath.exists():
            continue
        if fname == "latent_memory.pt":
            model.latent_memory.data.copy_(torch.load(fpath, weights_only=True))
        elif fname == "lifeline_gate.pt":
            model.lifeline_gate.data.copy_(torch.load(fpath, weights_only=True))
        else:
            target.load_state_dict(
                torch.load(fpath, map_location=DEVICE, weights_only=True)
            )
        log.info(f"  Loaded: {fname}")

    lora_path = ckpt_dir / "lora.pt"
    if lora_path.exists():
        lora_state = torch.load(lora_path, map_location=DEVICE, weights_only=True)
        for name, param in model.named_parameters():
            if name in lora_state:
                param.data.copy_(lora_state[name])
        log.info(f"  Loaded: lora.pt ({len(lora_state)} tensors)")


# ── Phase runners ─────────────────────────────────────────────────────────────

def run_phase(
    model: RecursiveMamba1_PrefixScratchpad,
    phase: str,
    log: logging.Logger,
    resume_step: int = 0,
) -> None:
    """Run one training phase.

    Args:
        model:       RLF engine (already moved to DEVICE)
        phase:       "3a", "3b", or "3c"
        log:         logger
        resume_step: step to resume from (0 = start fresh)
    """
    cfg = PHASE_CONFIG[phase]

    # Apply freeze protocol
    if phase == "3a":
        freeze_for_phase3a(model)
    elif phase == "3b":
        freeze_for_phase3b(model)
    else:
        freeze_for_phase3c(model)

    # Build optimizer with per-group LRs
    param_groups = []
    if cfg["lr_mem"] > 0:
        param_groups.append({"params": [model.latent_memory], "lr": cfg["lr_mem"],
                              "weight_decay": 0.0})
    bridge_params = (list(model.bridge_down.parameters()) +
                     list(model.bridge_up.parameters()))
    if cfg["lr_bridge"] > 0:
        param_groups.append({"params": bridge_params, "lr": cfg["lr_bridge"],
                              "weight_decay": 0.01})
    other_params = [p for p in model.parameters()
                    if p.requires_grad
                    and p is not model.latent_memory
                    and not any(p is bp for bp in bridge_params)]
    if cfg["lr_other"] > 0 and other_params:
        param_groups.append({"params": other_params, "lr": cfg["lr_other"],
                              "weight_decay": 0.01})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0
    )

    # Build dataloader
    if phase == "3c":
        dataset    = SFTDataset(SFT_DATA)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    else:
        dataset    = RLFDataset(
            size=max(cfg["steps"] * BATCH_SIZE * 2, 15000),
            seq_len=256,
            adversarial_prob=0.4 if phase == "3b" else 0.0,
        )
        dataloader = DataLoader(
            dataset, batch_size=BATCH_SIZE, collate_fn=collate_rlf, shuffle=True
        )

    RLF_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    data_iter  = iter(dataloader)
    total_loss = 0.0
    step       = resume_step
    optimizer.zero_grad()

    log.info(f"Phase {phase}: {cfg['steps']} steps, batch={BATCH_SIZE}, "
             f"accum={GRAD_ACCUM}")

    while step < cfg["steps"]:
        model.train()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        if phase == "3c":
            input_ids = batch.to(DEVICE)
            loss = sft_loss(model, input_ids) / GRAD_ACCUM
            acc  = halt_acc = ans_acc = 0.0
        else:
            input_ids, chain_targets, ans_starts = batch
            input_ids = input_ids.to(DEVICE)
            loss, acc, ans_acc, halt_acc = model(
                input_ids, chain_targets, ans_starts
            )
            loss = loss / GRAD_ACCUM

        loss.backward()
        total_loss += loss.item() * GRAD_ACCUM

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            optimizer.zero_grad()

        step += 1

        if step % LOG_EVERY == 0:
            avg = total_loss / LOG_EVERY
            total_loss = 0.0
            vram = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0
            if phase == "3c":
                log.info(f"[Phase{phase}][S{step:05d}] Loss={avg:.4f} | "
                         f"VRAM={vram:.2f}GB")
            else:
                log.info(f"[Phase{phase}][S{step:05d}] Loss={avg:.4f} | "
                         f"Acc={float(acc):.3f} | AnsAcc={float(ans_acc):.3f} | "
                         f"HaltAcc={halt_acc:.3f} | VRAM={vram:.2f}GB")

        if step % CKPT_EVERY == 0 or step == cfg["steps"]:
            save_ckpt(model, step, phase, avg if step % LOG_EVERY != 0
                      else total_loss / max(step % LOG_EVERY, 1))

    log.info(f"Phase {phase} complete at step {step}.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse args, load model, run phases."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",  default="all", choices=["3a", "3b", "3c", "all"])
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in RLF_CKPT_DIR")
    parser.add_argument("--resume-step", type=int, default=0,
                        help="Step offset to resume from (skip already-done steps).")
    parser.add_argument("--resume-ckpt", type=str, default=None,
                        help="Explicit checkpoint directory to load from (overrides auto-detect).")
    args = parser.parse_args()

    log = setup_logging(LOG_PATH)
    log.info("="*60)
    log.info("  Mamba-1.4B RLF Trainer")
    log.info(f"  Phase: {args.phase}")
    log.info("="*60)

    torch.cuda.set_per_process_memory_fraction(0.92)

    log.info(f"Loading from SFT checkpoint: {SFT_CKPT_DIR}")
    model = load_from_sft_checkpoint(str(SFT_CKPT_DIR), DEVICE)

    if args.resume or args.resume_ckpt:
        # Find checkpoint to resume from
        if args.resume_ckpt:
            ckpt_path = Path(args.resume_ckpt)
        else:
            ckpts = sorted(RLF_CKPT_DIR.glob("phase*_step*"))
            ckpt_path = ckpts[-1] if ckpts else None
        if ckpt_path and ckpt_path.exists():
            log.info(f"Resuming from: {ckpt_path}")
            load_rlf_components(model, ckpt_path)
        else:
            log.warning(f"Checkpoint not found: {ckpt_path} — starting from scratch")

    phases = ["3a", "3b", "3c"] if args.phase == "all" else [args.phase]
    for i, phase in enumerate(phases):
        log.info(f"\n{'='*60}")
        log.info(f"  Starting Phase {phase}")
        log.info(f"{'='*60}")
        # Only apply resume_step to the first phase being run
        r_step = args.resume_step if i == 0 else 0
        if r_step:
            log.info(f"  Resuming from step {r_step} (skipping already-done steps)")
        run_phase(model, phase, log, resume_step=r_step)

    log.info("All phases complete.")
    final = RLF_CKPT_DIR / "final"
    final.mkdir(exist_ok=True)
    save_ckpt(model, 99999, "final", 0.0)
    log.info(f"Final checkpoint saved to: {final}")


if __name__ == "__main__":
    main()
