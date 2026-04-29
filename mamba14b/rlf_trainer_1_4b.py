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

# ── Training config ──────────────────────────────────────────────────────────
BATCH_SIZE   = 1
GRAD_ACCUM   = 8
LOG_EVERY    = 25
CKPT_EVERY   = 500

# V3 Phase config
# Phase 3a: 6000 steps (3× longer than V2) — gives ConceptPerceptron time
#           to converge on 1-hop bottleneck before Phase 3b unlocks.
# Phase 3b: LR halved to 5e-5 — slows convergence to attractors.
PHASE_CONFIG: dict[str, dict] = {
    "3a": {"steps": 6000, "lr_percep": 1e-3, "lr_bridge": 1e-3, "lr_other": 0.0},
    "3b": {"steps": 8000, "lr_percep": 5e-5, "lr_bridge": 5e-5, "lr_other": 5e-5},
    "3c": {"steps": 1000, "lr_percep": 0.0,  "lr_bridge": 0.0,  "lr_other": 5e-5},
}

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
                # Strip whitespace AND null bytes (line 7906 is all \x00)
                line = line.replace("\x00", "").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "prompt" not in obj or "answer" not in obj:
                        continue
                    text = f"[USER]\n{obj['prompt']}\n========\n[ANSWER]\n{obj['answer']}"
                    self.samples.append(text)
                except Exception:
                    continue  # skip any other malformed lines silently

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


HALT_SUPPRESS_LOOPS = 2   # mask §HALT from logits for loop positions < this


def dynamic_halt_weight(step: int) -> float:
    """Priority 5: Scale the HALT loss contribution by training progress.

    Early training (step < 2000): HW=0.1 — HALT is nearly irrelevant;
    the model must learn to output content first.
    Mid training (step < 4000): HW=0.3 — HALT starts to matter.
    Late training (step >= 4000): HW=1.0 — full HALT supervision.
    """
    if step < 2000:
        return 0.1
    if step < 4000:
        return 0.3
    return 1.0


def compute_rlf_loss(
    model: RecursiveMamba1_PrefixScratchpad,
    input_ids: torch.Tensor,
    chain_targets: list,
    ans_starts: list,
    global_step: int,
) -> tuple[torch.Tensor, float, float, float]:
    """V2 loss computation with HALT suppression and dynamic HALT weight.

    Wraps the engine's forward pass and post-processes logits to:
      1. Mask §HALT to -inf for the first HALT_SUPPRESS_LOOPS loop positions
         (Priority 1: physically prevent the model from taking the zero-cost
         early-exit shortcut before it has produced any content).
      2. Apply a dynamic HALT weight that starts at 0.1 and ramps to 1.0
         (Priority 5: gives AnsAcc time to establish before HALT competes).

    The engine's forward() already computes per-loop CE internally; to apply
    HALT suppression we intercept at the logit level by running loops manually
    when suppression is active (first 4000 steps), then delegating to the
    standard engine forward once suppression is off.

    Args:
        model:         RLF engine in train mode
        input_ids:     [B, T]
        chain_targets: list of B lists of target token ids
        ans_starts:    list of B answer-start positions
        global_step:   current optimizer step (for HW schedule)

    Returns:
        (loss, acc, ans_acc, halt_acc)
    """
    hw = dynamic_halt_weight(global_step)

    # After the HW schedule stabilises (step >= 4000) and suppression windows
    # have passed (loops 0,1 already muzzled during warmup), delegate to the
    # standard engine forward for speed.
    if global_step >= 4000:
        loss, acc, ans_acc, halt_acc = model(
            input_ids, chain_targets, ans_starts
        )
        return loss, float(acc), float(ans_acc), halt_acc

    # ── Manual loop unroll with HALT suppression ──────────────────────────
    # Mirror of engine forward() but intercepts logits before CE.
    model.train()
    B = input_ids.shape[0]
    x, res = model._encode(input_ids)
    x_prompt   = x.detach().clone()
    res_prompt = res.detach().clone() if res is not None else None

    mem   = model.concept_perceptron(x_prompt)
    x_ext = torch.cat([mem, x], dim=1)
    if res is not None:
        res_pad = torch.zeros(B, model.M, model.d_model,
                              device=res.device, dtype=res.dtype)
        res_ext = torch.cat([res_pad, res], dim=1)
    else:
        res_ext = None

    from rlf_engine_1_4b import LIFELINE_DECAY
    from torch.utils.checkpoint import checkpoint as grad_ckpt

    n_loops = max(len(t) for t in chain_targets)
    step_losses: list[torch.Tensor] = []
    step_accs:   list[torch.Tensor] = []
    halt_accs:   list[float]        = []

    def _top_ckpt(x_in, r_in):
        """Gradient-checkpointed top LoRA layers."""
        return model._run_top_layers(x_in, r_in)

    for loop_i in range(n_loops):
        decay = LIFELINE_DECAY ** loop_i
        x_prompt_d   = x_prompt * decay
        res_prompt_d = res_prompt * decay if res_prompt is not None else None
        x_ext, res_ext = model._lifeline_inject(
            x_ext, res_ext, x_prompt_d, res_prompt_d
        )
        x_ext    = model.loop_rope(x_ext, loop_i)
        if res_ext is not None:
            res_ext = model.loop_rope(res_ext, loop_i)

        x_ext, res_ext = grad_ckpt(_top_ckpt, x_ext, res_ext, use_reentrant=False)
        x_ext = x_ext + model.mamba1_loop(x_ext)
        x_ext = model.loop_norm(x_ext)
        x_bridged = x_ext + model.bridge_up(model.bridge_down(x_ext))

        x_out    = x_bridged[:, model.M:, :]
        r_out    = res_ext[:, model.M:, :] if res_ext is not None else None
        x_normed = model._apply_norm(x_out, r_out)
        logits   = model.lm_head(x_normed)      # [B, T, V]
        V        = logits.shape[-1]

        # Priority 1: HALT Suppression — mask §HALT on a per-sample basis.
        # Applied only when loop_i < HALT_SUPPRESS_LOOPS AND the target for
        # that sample is a content token (not HALT itself). Masking HALT when
        # the target IS HALT yields log(0) = inf loss — the original bug.

        loop_loss = torch.tensor(0.0, device=x_ext.device, requires_grad=True)
        loop_acc  = torch.tensor(0.0, device=x_ext.device)
        valid     = 0

        for b in range(B):
            as_ = (ans_starts[b] if ans_starts else x_out.shape[1] - 1)
            if as_ < 1 or as_ >= x_out.shape[1]:
                continue
            tgt_id = int(chain_targets[b][min(loop_i, len(chain_targets[b]) - 1)])
            if tgt_id >= V:
                continue
            lg_b    = logits[b, as_ - 1, :].clone()
            is_halt = (tgt_id == HALT_ID)

            # Mask §HALT only when we are in the suppression window AND the
            # target is a real content token — never mask what we're predicting.
            if loop_i < HALT_SUPPRESS_LOOPS and not is_halt:
                lg_b[HALT_ID] = float("-inf")

            pred  = lg_b.argmax().item()
            tgt_t = torch.tensor(tgt_id, device=x_ext.device)
            # Priority 5: scale HALT loss contribution by hw schedule.
            ce = F.cross_entropy(lg_b.unsqueeze(0), tgt_t.unsqueeze(0))
            loop_loss = loop_loss + (ce * hw if is_halt else ce)
            loop_acc  = loop_acc + float(pred == tgt_id)
            valid    += 1
            if is_halt:
                halt_accs.append(float(pred == tgt_id))

        if valid > 0:
            step_losses.append(loop_loss / valid)
            step_accs.append(loop_acc / valid)

    avg_loss = (torch.stack(step_losses).mean() if step_losses else
                torch.tensor(0.0, requires_grad=True))

    # V3 Priority 8.4: Norm penalty anchors mem_norm to 1.0.
    # Prevents ConceptPerceptron from silently collapsing to all-zeros map.
    # Penalty = 0.1 * (1 - mem_norm)^2, minimum at mem_norm = 1.0.
    mem_norm_val = getattr(model, "mem_norm", None)
    if mem_norm_val is not None and mem_norm_val.requires_grad:
        norm_penalty = 0.1 * ((1.0 - mem_norm_val) ** 2)
        avg_loss = avg_loss + norm_penalty

    avg_acc  = (torch.stack([a.detach() for a in step_accs]).mean()
                if step_accs else torch.tensor(0.0))
    ans_accs = step_accs[:-1] if len(step_accs) > 1 else step_accs
    ans_acc  = (torch.stack([a.detach() for a in ans_accs]).mean()
                if ans_accs else avg_acc)
    halt_acc = (sum(halt_accs) / len(halt_accs)) if halt_accs else 0.0
    mem_norm = float(mem_norm_val.detach()) if mem_norm_val is not None else 0.0
    return avg_loss, float(avg_acc), float(ans_acc), halt_acc, mem_norm


def sft_loss(
    model: RecursiveMamba1_PrefixScratchpad,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Causal LM loss for Phase 3c.

    Run 1 loop of RLF (no chain targets), take the LM head output,
    compute next-token CE. This recovers code generation ability.
    """
    model.train()
    B = input_ids.shape[0]

    # V2 API: _encode takes input_ids directly, returns (hidden, residual)
    x, res = model._encode(input_ids)
    x_prompt   = x.detach().clone()
    res_prompt = res.detach().clone() if res is not None else None

    # Build extended sequence with ConceptPerceptron scratchpad prefix
    mem   = model.concept_perceptron(x_prompt)       # [B, M, D]
    x_ext = torch.cat([mem, x], dim=1)               # [B, M+T, D]
    if res is not None:
        res_pad = torch.zeros(B, model.M, model.d_model,
                              device=res.device, dtype=res.dtype)
        res_ext = torch.cat([res_pad, res], dim=1)
    else:
        res_ext = None

    # Single RLF loop (loop_i=0) — no decay on first loop
    x_ext, res_ext = model._lifeline_inject(
        x_ext, res_ext, x_prompt, res_prompt
    )
    x_ext = model.loop_rope(x_ext, 0)
    if res_ext is not None:
        res_ext = model.loop_rope(res_ext, 0)

    from rlf_engine_1_4b import BASE_SPLIT
    x_ext, res_ext = model._run_top_layers(x_ext, res_ext)
    x_ext  = x_ext + model.mamba1_loop(x_ext)
    x_ext  = model.loop_norm(x_ext)
    x_bridged = x_ext + model.bridge_up(model.bridge_down(x_ext))

    x_out    = x_bridged[:, model.M:, :]             # [B, T, D]
    r_out    = res_ext[:, model.M:, :] if res_ext is not None else None
    x_normed = model._apply_norm(x_out, r_out)
    logits   = model.lm_head(x_normed)               # [B, T, V]

    # Causal next-token prediction
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

    V2: saves concept_perceptron instead of latent_memory.
    Only saves the trainable parts — base backbone weights are unchanged
    and can be reloaded from the original SFT checkpoint.
    """
    ckpt = RLF_CKPT_DIR / f"phase{phase}_step{step:06d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(model.concept_perceptron.state_dict(), ckpt / "concept_perceptron.pt")
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
    """Load saved RLF components into model (for resume).

    V2: loads concept_perceptron instead of latent_memory.
    """
    log = logging.getLogger(__name__)
    for fname, target in [
        ("concept_perceptron.pt", model.concept_perceptron),
        ("bridge_down.pt",        model.bridge_down),
        ("bridge_up.pt",          model.bridge_up),
        ("mamba1_loop.pt",        model.mamba1_loop),
        ("loop_norm.pt",          model.loop_norm),
        ("lifeline_gate.pt",      None),
        ("lm_head.pt",            model.lm_head),
    ]:
        fpath = ckpt_dir / fname
        if not fpath.exists():
            continue
        if fname == "lifeline_gate.pt":
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
    percep_params = list(model.concept_perceptron.parameters())
    if cfg["lr_percep"] > 0:
        param_groups.append({"params": percep_params, "lr": cfg["lr_percep"],
                              "weight_decay": 0.0})
    bridge_params = (list(model.bridge_down.parameters()) +
                     list(model.bridge_up.parameters()))
    if cfg["lr_bridge"] > 0:
        param_groups.append({"params": bridge_params, "lr": cfg["lr_bridge"],
                              "weight_decay": 0.01})
    other_params = [p for p in model.parameters()
                    if p.requires_grad
                    and not any(p is pp for pp in percep_params)
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
        # V3 Priority 8.1: Phase 3a uses 1-hop only bottleneck curriculum.
        # Phase 3b uses full V2 multi-hop mix.
        dataset    = RLFDataset(
            size=max(cfg["steps"] * BATCH_SIZE * 2, 15000),
            seq_len=256,
            adversarial_prob=0.4 if phase == "3b" else 0.0,
            phase3a_warmup=(phase == "3a"),
        )
        dataloader = DataLoader(
            dataset, batch_size=BATCH_SIZE, collate_fn=collate_rlf, shuffle=True
        )

    # V3 Priority 8.2: Freeze lm_head at Phase 3b start.
    # Prevents lm_head acting as a "garbage translator" while the
    # ConceptPerceptron is still learning. Unfrozen at step 2000.
    lm_head_frozen = False
    if phase == "3b":
        log.info("\u2744️ Phase 3b: lm_head FROZEN (V3 anti-attractor trap)")
        for param in model.lm_head.parameters():
            param.requires_grad = False
        lm_head_frozen = True

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
            acc  = halt_acc = ans_acc = mem_norm = 0.0
        else:
            input_ids, chain_targets, ans_starts = batch
            input_ids = input_ids.to(DEVICE)

            # V3 Priority 8.2: Unfreeze lm_head at Phase 3b step 2000.
            if phase == "3b" and lm_head_frozen and step >= 2000:
                log.info("\U0001f525 [V3 TRIGGER] Step 2000 — unfreezing lm_head")
                for param in model.lm_head.parameters():
                    param.requires_grad = True
                # CRITICAL: add newly unfrozen params to active optimizer group
                optimizer.add_param_group({
                    "params": list(model.lm_head.parameters()),
                    "lr": cfg["lr_other"],
                    "weight_decay": 0.01,
                })
                lm_head_frozen = False

            # V3: compute_rlf_loss now returns mem_norm as 5th element
            loss, acc, ans_acc, halt_acc, mem_norm = compute_rlf_loss(
                model, input_ids, chain_targets, ans_starts, step
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
            hw   = dynamic_halt_weight(step)
            if phase == "3c":
                log.info(f"[Phase{phase}][S{step:05d}] Loss={avg:.4f} | "
                         f"VRAM={vram:.2f}GB")
            else:
                log.info(f"[Phase{phase}][S{step:05d}] Loss={avg:.4f} | "
                         f"Acc={float(acc):.3f} | AnsAcc={float(ans_acc):.3f} | "
                         f"HaltAcc={halt_acc:.3f} | HW={hw:.2f} | "
                         f"mem_norm={mem_norm:.3f} | VRAM={vram:.2f}GB")

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
    args = parser.parse_args()

    log = setup_logging(LOG_PATH)
    log.info("="*60)
    log.info("  Mamba-1.4B RLF Trainer")
    log.info(f"  Phase: {args.phase}")
    log.info("="*60)

    torch.cuda.set_per_process_memory_fraction(0.92)

    log.info(f"Loading from SFT checkpoint: {SFT_CKPT_DIR}")
    model = load_from_sft_checkpoint(str(SFT_CKPT_DIR), DEVICE)

    if args.resume:
        # Find latest checkpoint
        ckpts = sorted(RLF_CKPT_DIR.glob("phase*_step*"))
        if ckpts:
            latest = ckpts[-1]
            log.info(f"Resuming from: {latest}")
            load_rlf_components(model, latest)

    phases = ["3a", "3b", "3c"] if args.phase == "all" else [args.phase]
    for phase in phases:
        log.info(f"\n{'='*60}")
        log.info(f"  Starting Phase {phase}")
        log.info(f"{'='*60}")
        run_phase(model, phase, log)

    log.info("All phases complete.")
    final = RLF_CKPT_DIR / "final"
    final.mkdir(exist_ok=True)
    save_ckpt(model, 99999, "final", 0.0)
    log.info(f"Final checkpoint saved to: {final}")


if __name__ == "__main__":
    main()
