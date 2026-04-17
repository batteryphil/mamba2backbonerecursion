#!/usr/bin/env python3
"""
phase2_sculptor_trainer.py -- Agent 2: The Sculptor
Project TinyRefinementModel

Trains Mamba-1.4B student to mimic teacher latent reasoning.
Uses mamba_ssm native loader + custom LoRA injection (PEFT cannot
handle Mamba's non-standard architecture directly).

Student: state-spaces/mamba-1.4b (d_model=2048, 48 layers, ~2.78 GB BF16)
LoRA: r=64, alpha=128 on in_proj / x_proj / dt_proj / out_proj / conv1d
HaltingHead: MLP on final hidden state
Loss: CrossEntropy (LM) + 0.1 * BCE (halting)

Usage:
    ./run_env.sh phase2_sculptor_trainer.py
    ./run_env.sh phase2_sculptor_trainer.py --dry-run
    ./run_env.sh phase2_sculptor_trainer.py --resume checkpoints/lora-step200
"""

import argparse
import gc
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKSPACE = Path(__file__).parent
TRAINING_DATA = WORKSPACE / "training_data.jsonl"
CHECKPOINT_DIR = WORKSPACE / "checkpoints"
LOG_FILE = WORKSPACE / "phase2_sculptor.log"

STUDENT_REPO = "state-spaces/mamba-1.4b"
GOLD_MASTER_DIR = WORKSPACE / "checkpoints" / "mamba-tiny-refinement-BF16-MASTER"
D_MODEL = 2048      # From config probe
N_LAYER = 48

# Instruct format boundaries — teach base model structural roles
USER_TAG = "[USER]"
REASONING_TAG = "[REASONING]"
ANSWER_TAG = "[ANSWER]"

LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
# conv1d excluded: depthwise groups make low-rank unsuitable; target linear projections only
LORA_TARGETS = ["in_proj", "x_proj", "dt_proj", "out_proj"]

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
HALT_LOSS_WEIGHT = 0.1

BATCH_SIZE = 1
GRADIENT_ACCUM_STEPS = 4
MAX_SEQ_LEN = 384      # Conservative for VRAM safety with long spacer seqs
LOG_EVERY_STEPS = 1    # Log every step (dataset is small)
CHECKPOINT_EVERY_STEPS = 200
VRAM_FLOOR_GB = 3.5    # Hard abort threshold during training
DEFAULT_EPOCHS = 50   # Multi-epoch over small dataset

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VRAM monitor
# ---------------------------------------------------------------------------

def vram_free_gb() -> float:
    """Return free VRAM in GB.

    Returns:
        Free VRAM in gigabytes.
    """
    free, _ = torch.cuda.mem_get_info()
    return free / 1024 ** 3


def assert_vram(min_gb: float, label: str = "") -> None:
    """Assert free VRAM is above floor, abort if not.

    Args:
        min_gb: Minimum required free VRAM.
        label: Context string for log message.

    Raises:
        SystemExit: If VRAM is below the floor.
    """
    free = vram_free_gb()
    tag = f" [{label}]" if label else ""
    log.info("VRAM%s: %.2f GB free", tag, free)
    if free < min_gb:
        log.error("VRAM floor breached%s: %.2f < %.2f GB. Aborting.", tag, free, min_gb)
        sys.exit(1)


# ---------------------------------------------------------------------------
# LoRA via weight-delta hooks
# ---------------------------------------------------------------------------
# Mamba's fast path (mamba_inner_fn) accesses .weight tensors directly,
# bypassing forward(). We solve this by:
#   1. Disabling use_fast_path on all Mamba mixer layers so standard
#      nn.Linear.forward() is called instead of the CUDA kernel.
#   2. Using register_forward_pre_hook / register_forward_hook on each
#      target nn.Linear to add/remove the LoRA delta from .weight in-place.
#      The delta is registered as a parameter on the parent module.
# ---------------------------------------------------------------------------

class LoRAHook:
    """Stateful LoRA hook pair attached to a single nn.Linear.

    Adds the LoRA delta (B @ A * scale) into .weight.data before the
    forward pass, and subtracts it afterward, leaving the frozen weights
    unchanged between steps.

    Args:
        layer: The frozen nn.Linear to adapt.
        owner: The nn.Module that will hold lora_A / lora_B as parameters.
        param_prefix: Prefix for parameter names on the owner module.
        r: LoRA rank.
        alpha: LoRA scaling factor.
    """

    def __init__(
        self,
        layer: nn.Linear,
        owner: nn.Module,
        param_prefix: str,
        r: int = LORA_RANK,
        alpha: int = LORA_ALPHA,
    ) -> None:
        """Initialize LoRAHook and register A/B parameters on owner."""
        self.layer = layer
        self.scale = alpha / r
        dtype = layer.weight.dtype
        out_f, in_f = layer.weight.shape

        # Register as parameters on owner so optimizer can find them
        lora_A = nn.Parameter(
            torch.randn(r, in_f, dtype=dtype, device=layer.weight.device)
            * (1.0 / math.sqrt(r))
        )
        lora_B = nn.Parameter(
            torch.zeros(out_f, r, dtype=dtype, device=layer.weight.device)
        )
        setattr(owner, f"{param_prefix}_lora_A", lora_A)
        setattr(owner, f"{param_prefix}_lora_B", lora_B)
        self.lora_A = lora_A
        self.lora_B = lora_B

        # Register hooks on the layer itself
        layer.register_forward_pre_hook(self._pre_hook)
        layer.register_forward_hook(self._post_hook)

    def _delta(self) -> torch.Tensor:
        """Compute the LoRA weight delta: B @ A * scale.

        Returns:
            Delta tensor of shape (out_features, in_features).
        """
        return (self.lora_B @ self.lora_A) * self.scale

    def _pre_hook(self, module: nn.Module, args: tuple) -> None:
        """Add LoRA delta to weight before forward.

        Args:
            module: The nn.Linear being called.
            args: Forward arguments (unused).
        """
        module.weight.data.add_(self._delta())

    def _post_hook(
        self, module: nn.Module, args: tuple, output: torch.Tensor
    ) -> torch.Tensor:
        """Subtract LoRA delta from weight after forward to restore frozen state.

        Args:
            module: The nn.Linear being called.
            args: Forward arguments.
            output: The layer output (returned unchanged).

        Returns:
            Unchanged output tensor.
        """
        module.weight.data.sub_(self._delta())
        return output


def inject_lora(
    model: nn.Module,
    targets: list[str] = LORA_TARGETS,
) -> int:
    """Inject LoRA hook pairs into all target nn.Linear layers in Mamba.

    Also disables use_fast_path on every Mamba mixer so the fused CUDA
    kernel is bypassed and standard nn.Linear.forward() runs, which our
    hooks can intercept.

    Args:
        model: MambaLMHeadModel to inject into.
        targets: List of attribute names to target (e.g. 'in_proj').

    Returns:
        Number of LoRA hook pairs injected.
    """
    # Step 1: disable fast path on all Mamba mixer layers
    for module in model.modules():
        if hasattr(module, "use_fast_path"):
            module.use_fast_path = False

    # Step 2: inject LoRA hooks on target Linear layers
    injected = 0
    for name, module in model.named_modules():
        short_name = name.split(".")[-1]
        if short_name not in targets:
            continue
        if not isinstance(module, nn.Linear):
            continue
        parent_name = ".".join(name.split(".")[:-1])
        parent = model.get_submodule(parent_name) if parent_name else model
        param_prefix = name.replace(".", "_")
        LoRAHook(
            layer=module,
            owner=parent,
            param_prefix=short_name,
            r=LORA_RANK,
            alpha=LORA_ALPHA,
        )
        injected += 1

    return injected


def get_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Collect all trainable LoRA adapter parameters (lora_A and lora_B).

    Args:
        model: Model with injected LoRA hooks.

    Returns:
        List of parameters with requires_grad=True.
    """
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Tokenizer (GPT-NeoX compatible)
# ---------------------------------------------------------------------------

def load_tokenizer():
    """Load the EleutherAI GPT-NeoX tokenizer used by Mamba-1.4B.

    Returns:
        Loaded tokenizer with pad_token set.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neox-20b",
        use_fast=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MambaLatentDataset(Dataset):
    """Dataset of (prompt, full_target) pairs from training_data.jsonl.

    Sequences are truncated to MAX_SEQ_LEN and labels mask the prompt
    so loss is computed only on the spacer+answer region.
    """

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer,
        max_seq_len: int = MAX_SEQ_LEN,
    ) -> None:
        """Load and tokenize all samples from the JSONL file.

        Args:
            jsonl_path: Path to training_data.jsonl.
            tokenizer: GPT-NeoX tokenizer.
            max_seq_len: Maximum token sequence length.
        """
        self.samples: list[dict] = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        log.info("Loading dataset: %s", jsonl_path)
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    self.samples.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
        log.info("Dataset: %d samples loaded.", len(self.samples))
        if len(self.samples) == 0:
            log.error("Empty dataset! Run phase1_distiller.py first.")
            sys.exit(1)

    def __len__(self) -> int:
        """Return sample count."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Tokenize and return one sample in instruct format.

        Wraps prompt and target in structural control tokens:
            [USER]\n{prompt}\n[REASONING]\n{spacers}\n[ANSWER]\n{answer}

        This teaches the base model that [REASONING] triggers latent loops
        and [ANSWER] signals output. Without this, the base model treats
        everything as autocomplete continuation.

        Args:
            idx: Sample index.

        Returns:
            Dict: input_ids, labels, halting_targets, attention_mask.
        """
        s = self.samples[idx]
        prompt = s["prompt"]
        spacer_seq = s.get("spacer_sequence", "")
        answer = s.get("answer", s.get("full_target", ""))
        n_spacers = s["cot_token_count"]

        # Instruct-formatted full sequence
        # Label masking: only compute loss on [REASONING] onward
        prompt_block = f"{USER_TAG}\n{prompt}\n{REASONING_TAG}\n"
        target_block = f"{spacer_seq}\n{ANSWER_TAG}\n{answer}"
        full_text = prompt_block + target_block

        enc = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        seq_len = input_ids.shape[0]

        # Compute prompt boundary (mask loss on prompt block)
        p_enc = self.tokenizer(
            prompt_block, truncation=True,
            max_length=self.max_seq_len, return_tensors="pt"
        )
        prompt_len = min(p_enc["input_ids"].shape[1], seq_len)

        # Labels: -100 on prompt block, predict spacer + answer
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        # Halting target: 1 at the spacer→answer transition
        halting_targets = torch.zeros(seq_len, dtype=torch.float32)
        transition = min(prompt_len + n_spacers, seq_len - 1)
        halting_targets[transition] = 1.0

        attention_mask = torch.ones(seq_len, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "halting_targets": halting_targets,
            "attention_mask": attention_mask,
        }


def collate_pad(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Pad a batch of variable-length samples to the longest sequence.

    Args:
        batch: List of sample dicts from __getitem__.

    Returns:
        Padded and stacked tensors.
    """
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids, labels, halts, masks = [], [], [], []
    for b in batch:
        pad = max_len - b["input_ids"].shape[0]
        input_ids.append(F.pad(b["input_ids"], (0, pad), value=0))
        labels.append(F.pad(b["labels"], (0, pad), value=-100))
        halts.append(F.pad(b["halting_targets"], (0, pad), value=0.0))
        masks.append(F.pad(b["attention_mask"], (0, pad), value=0))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "halting_targets": torch.stack(halts),
        "attention_mask": torch.stack(masks),
    }


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_mamba_student(load_from_master: bool = False):
    """Load Mamba-1.4B in BF16, freeze backbone, inject LoRA + HaltingHead.

    When load_from_master=True, loads weights from the local BF16 Gold Master
    checkpoint (which already contains Phase 2 LoRA merged in) instead of the
    fresh HuggingFace base. This lets Phase 2.5 build on top of Phase 2 work.

    After freezing, the lm_head is explicitly unfrozen so output logits for
    ==== tokens can be updated by the optimizer — critical for teaching the
    model to generate the spacer pattern.

    Args:
        load_from_master: If True, load from Gold Master instead of HF repo.

    Returns:
        Tuple of (model, halting_head).
    """
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
    from mamba_ssm.models.config_mamba import MambaConfig

    log.info("Loading Mamba-1.4B config from HF...")
    config_data = load_config_hf(STUDENT_REPO)
    model_cfg = MambaConfig(**config_data)
    log.info("d_model=%d  n_layer=%d  vocab=%d",
             model_cfg.d_model, model_cfg.n_layer, model_cfg.vocab_size)

    log.info("Instantiating model skeleton on CUDA in bfloat16...")
    model = MambaLMHeadModel(model_cfg, device="cuda", dtype=torch.bfloat16)

    if load_from_master and GOLD_MASTER_DIR.exists():
        master_weights = GOLD_MASTER_DIR / "pytorch_model.bin"
        log.info("Loading from BF16 Gold Master: %s", master_weights)
        state_dict = torch.load(str(master_weights), map_location="cuda")
        model.load_state_dict(state_dict)
        del state_dict
    else:
        if load_from_master:
            log.warning("Gold Master not found at %s — falling back to HF base.",
                        GOLD_MASTER_DIR)
        log.info("Downloading pretrained weights from %s...", STUDENT_REPO)
        state_dict = load_state_dict_hf(STUDENT_REPO, device="cuda", dtype=torch.bfloat16)
        model.load_state_dict(state_dict)
        del state_dict

    gc.collect()
    torch.cuda.empty_cache()

    free = vram_free_gb()
    log.info("Backbone loaded. VRAM free: %.2f GB", free)

    # Freeze all backbone parameters
    for p in model.parameters():
        p.requires_grad = False
    log.info("Backbone frozen.")

    # CRITICAL: Unfreeze backbone.embedding.weight (Mamba uses tied embeddings —
    # this tensor serves as BOTH the input embedding AND the output projection).
    # If frozen, the model can never shift its output distribution toward ==== tokens.
    lm_head_unfrozen = 0
    for name, p in model.named_parameters():
        if "backbone.embedding" in name:
            p.requires_grad = True
            lm_head_unfrozen += 1
            log.info("Unfrozen output projection: %s  %s", name, tuple(p.shape))
    log.info("Output projection unfrozen: %d tensors.", lm_head_unfrozen)

    # Inject LoRA adapters into SSM projection layers
    log.info("Injecting LoRA: rank=%d alpha=%d targets=%s",
             LORA_RANK, LORA_ALPHA, LORA_TARGETS)
    n_replaced = inject_lora(model, LORA_TARGETS)
    log.info("LoRA injected into %d layers.", n_replaced)

    lora_params = get_lora_params(model)
    n_trainable = sum(p.numel() for p in lora_params)
    n_total = sum(p.numel() for p in model.parameters())
    log.info("Trainable params: %d / %d (%.2f%%)",
             n_trainable, n_total, 100.0 * n_trainable / n_total)

    # Attach HaltingHead
    from phase2_halting_head import HaltingHead
    halting_head = HaltingHead(hidden_dim=model_cfg.d_model).to(torch.bfloat16).cuda()
    log.info("HaltingHead initialized. hidden_dim=%d  final_bias=-2.0", model_cfg.d_model)
    log.info("Post-load VRAM free: %.2f GB", vram_free_gb())

    return model, halting_head


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    halting_head: nn.Module,
    optimizer,
    scheduler,
    step: int,
    loss: float,
    tag: str = "",
) -> None:
    """Save LoRA weights + HaltingHead + optimizer state.

    Args:
        model: Model with injected LoRA layers.
        halting_head: HaltingHead module.
        optimizer: Optimizer (for resumable training).
        scheduler: LR scheduler.
        step: Global training step.
        loss: Current loss value.
        tag: Suffix for directory name (e.g. 'final').
    """
    suffix = f"-{tag}" if tag else f"-step{step}"
    ckpt_dir = CHECKPOINT_DIR / f"lora{suffix}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save only LoRA parameters (not frozen backbone)
    lora_state = {
        name: param.data
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    torch.save(lora_state, str(ckpt_dir / "lora_weights.pt"))
    torch.save(halting_head.state_dict(), str(ckpt_dir / "halting_head.pt"))
    torch.save({
        "step": step,
        "loss": loss,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, str(ckpt_dir / "training_state.pt"))

    log.info("Checkpoint saved: %s  step=%d  loss=%.4f", ckpt_dir, step, loss)


def load_checkpoint(
    model: nn.Module,
    halting_head: nn.Module,
    optimizer,
    scheduler,
    ckpt_dir: Path,
) -> int:
    """Restore LoRA weights + optimizer + scheduler from checkpoint.

    Args:
        model: Model with injected LoRA layers.
        halting_head: HaltingHead module.
        optimizer: To restore state into.
        scheduler: To restore state into.
        ckpt_dir: Checkpoint directory path.

    Returns:
        Step number from the checkpoint.
    """
    lora_path = ckpt_dir / "lora_weights.pt"
    halt_path = ckpt_dir / "halting_head.pt"
    state_path = ckpt_dir / "training_state.pt"

    if lora_path.exists():
        lora_state = torch.load(str(lora_path), map_location="cuda")
        for name, param in model.named_parameters():
            if param.requires_grad and name in lora_state:
                param.data.copy_(lora_state[name])
        log.info("LoRA weights restored.")

    if halt_path.exists():
        halting_head.load_state_dict(torch.load(str(halt_path), map_location="cuda"))
        log.info("HaltingHead restored.")

    if state_path.exists():
        state = torch.load(str(state_path), map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        log.info("Optimizer/scheduler restored at step %d.", state["step"])
        return state["step"]

    return 0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training(
    model: nn.Module,
    halting_head: nn.Module,
    dataset: MambaLatentDataset,
    optimizer,
    scheduler,
    resume_step: int = 0,
    dry_run: bool = False,
    epochs: int = DEFAULT_EPOCHS,
) -> None:
    """Multi-epoch gradient accumulation training loop.

    Args:
        model: Mamba model with LoRA injected.
        halting_head: HaltingHead MLP.
        dataset: MambaLatentDataset instance.
        optimizer: AdamW8bit over LoRA + HaltingHead params.
        scheduler: Cosine LR scheduler.
        resume_step: Start step for logging/checkpointing continuity.
        dry_run: Run only 3 gradient steps if True.
        epochs: Number of passes over the full dataset.
    """

    model.train()
    halting_head.train()

    steps_per_epoch = max(len(dataset) // (BATCH_SIZE * GRADIENT_ACCUM_STEPS), 1)
    n_grad_steps = steps_per_epoch * epochs if not dry_run else 3
    limit_steps = 3 if dry_run else n_grad_steps

    log.info("Training: %d samples | %d epochs | %d steps/epoch | %d total grad steps",
             len(dataset), epochs, steps_per_epoch, n_grad_steps)

    global_step = resume_step
    accum_lm = accum_halt = 0.0
    accum_n = 0
    optimizer.zero_grad()
    t0 = time.time()

    for epoch in range(epochs):
        if global_step >= limit_steps:
            break

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_pad,
            num_workers=0,
            pin_memory=False,
        )
        log.info("--- Epoch %d/%d ---", epoch + 1, epochs)

        for batch_idx, batch in enumerate(loader):
            if global_step >= limit_steps:
                break

            # VRAM safety check every 50 batches
            if batch_idx % 50 == 0 and batch_idx > 0:
                free = vram_free_gb()
                if free < VRAM_FLOOR_GB:
                    log.error("VRAM floor breached at step %d: %.2f GB. Saving emergency ckpt.",
                              global_step, free)
                    save_checkpoint(model, halting_head, optimizer, scheduler,
                                    global_step, float("nan"), tag="emergency")
                    sys.exit(1)

            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            halting_targets = batch["halting_targets"].cuda()

            try:
                out = model(input_ids, num_last_tokens=0)
                logits = out.logits         # (B, L, vocab)

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                lm_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

                with torch.no_grad():
                    hidden = model.backbone(input_ids)  # (B, L, d_model)
                hidden = hidden.detach()
                halting_logits = halting_head(hidden).squeeze(-1)  # (B, L)
                halt_loss = F.binary_cross_entropy_with_logits(
                    halting_logits, halting_targets
                )

                total_loss = lm_loss + HALT_LOSS_WEIGHT * halt_loss

            except torch.cuda.OutOfMemoryError:
                log.warning("OOM at batch %d — clearing and skipping.", batch_idx)
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
            except Exception as exc:
                log.warning("Forward error at batch %d: %s — skipping.", batch_idx, exc)
                optimizer.zero_grad()
                continue

            (total_loss / GRADIENT_ACCUM_STEPS).backward()
            accum_lm += lm_loss.item()
            accum_halt += halt_loss.item()
            accum_n += 1

            if (batch_idx + 1) % GRADIENT_ACCUM_STEPS == 0:
                all_trainable = get_lora_params(model) + list(halting_head.parameters())
                nn.utils.clip_grad_norm_(all_trainable, max_norm=GRAD_CLIP_NORM)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_lm = accum_lm / max(accum_n, 1)
                avg_halt = accum_halt / max(accum_n, 1)
                accum_lm = accum_halt = accum_n = 0

                if global_step % LOG_EVERY_STEPS == 0:
                    elapsed = time.time() - t0
                    sps = global_step / max(elapsed, 1e-6)
                    eta_min = (n_grad_steps - global_step) / max(sps, 1e-6) / 60
                    lr_now = scheduler.get_last_lr()[0]
                    log.info(
                        "[E%d] Step %d/%d | LM=%.4f | Halt=%.4f | LR=%.2e | "
                        "%.2f sps | ETA %.1fmin | VRAM %.2fGB",
                        epoch + 1, global_step, n_grad_steps,
                        avg_lm, avg_halt, lr_now,
                        sps, eta_min, vram_free_gb(),
                    )

                if global_step % CHECKPOINT_EVERY_STEPS == 0:
                    save_checkpoint(model, halting_head, optimizer, scheduler,
                                    global_step, avg_lm)

    log.info("Training complete at step %d / %d.", global_step, n_grad_steps)

    # Save final LoRA (consumed by Phase 3)
    final_dir = CHECKPOINT_DIR / "lora_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    lora_state = {
        n: p.data for n, p in model.named_parameters() if p.requires_grad
    }
    torch.save(lora_state, str(final_dir / "lora_weights.pt"))
    torch.save(halting_head.state_dict(), str(final_dir / "halting_head.pt"))
    log.info("Final LoRA saved: %s", final_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Phase 2 entry point: load student, inject LoRA, train, save."""
    parser = argparse.ArgumentParser(description="Agent 2: The Sculptor")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 3 gradient steps only (smoke test).")
    parser.add_argument("--resume", type=str, default=None, metavar="CKPT_DIR",
                        help="Checkpoint directory to resume from.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of full passes over the dataset (default: {DEFAULT_EPOCHS}).")
    parser.add_argument("--master", action="store_true",
                        help="Load from BF16 Gold Master instead of fresh HF base.")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("AGENT 2: THE SCULPTOR -- Project TinyRefinementModel")
    log.info("=" * 60)
    log.info("Student: %s | r=%d | alpha=%d | dry-run=%s",
             STUDENT_REPO, LORA_RANK, LORA_ALPHA, args.dry_run)

    # Pre-flight VRAM assertion
    assert_vram(min_gb=8.0, label="Phase 2 pre-flight")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Load student + LoRA + HaltingHead
    model, halting_head = load_mamba_student(load_from_master=args.master)
    assert_vram(min_gb=VRAM_FLOOR_GB, label="Post model load")

    # Build optimizer — TWO parameter groups to prevent catastrophic forgetting.
    #
    # backbone.embedding.weight (50,277 × 2048) encodes ALL Python syntax.
    # Training it at the full LoRA LR (1e-4) overwrites pre-trained vocab
    # geometry in ~150 steps → code repetition gets worse, not better.
    # Fix: 10x lower LR for the vocab matrix so it gently rotates toward
    # the evolved latent state without erasing Python token distributions.
    import bitsandbytes as bnb

    embedding_params: list[nn.Parameter] = []
    lora_head_params: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone.embedding" in name:
            embedding_params.append(p)
        else:
            lora_head_params.append(p)
    lora_head_params += list(halting_head.parameters())

    VOCAB_LR = LEARNING_RATE / 10  # 1e-5 catastrophic forgetting guard
    param_groups = [
        {"params": lora_head_params, "lr": LEARNING_RATE, "name": "lora_head"},
        {"params": embedding_params,  "lr": VOCAB_LR,     "name": "embedding"},
    ]
    n_lora  = sum(p.numel() for p in lora_head_params)
    n_embed = sum(p.numel() for p in embedding_params)
    log.info(
        "AdamW8bit — LoRA+Head: %dM @ LR=%.1e | Embedding: %dM @ LR=%.1e",
        n_lora // 1_000_000, LEARNING_RATE,
        n_embed // 1_000_000, VOCAB_LR,
    )
    optimizer = bnb.optim.AdamW8bit(param_groups, weight_decay=WEIGHT_DECAY)


    # Dataset
    tokenizer = load_tokenizer()
    dataset = MambaLatentDataset(TRAINING_DATA, tokenizer, MAX_SEQ_LEN)

    # LR scheduler: cosine with warmup over full epoch budget
    steps_per_epoch = max(len(dataset) // (BATCH_SIZE * GRADIENT_ACCUM_STEPS), 1)
    n_grad_steps = steps_per_epoch * args.epochs
    warmup = min(20, n_grad_steps // 10)

    def lr_lambda(step: int) -> float:
        """Cosine annealing with linear warmup."""
        if step < warmup:
            return float(step) / float(max(1, warmup))
        prog = (step - warmup) / max(1, n_grad_steps - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    resume_step = 0
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            resume_step = load_checkpoint(
                model, halting_head, optimizer, scheduler, ckpt_path
            )
        else:
            log.warning("Checkpoint not found: %s. Starting fresh.", ckpt_path)

    # Train
    run_training(
        model=model,
        halting_head=halting_head,
        dataset=dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        resume_step=resume_step,
        dry_run=args.dry_run,
        epochs=args.epochs,
    )

    log.info("=" * 60)
    log.info("AGENT 2 COMPLETE. Hand off to Agent 3 (The Blacksmith).")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
