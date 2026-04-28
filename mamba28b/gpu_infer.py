"""
gpu_infer.py — GPU inference for Mamba2-2.7B + RLF
====================================================
Loads the trained checkpoint and runs interactive inference on CUDA.
Bypasses the baremetal C engine entirely.

Usage:
  python gpu_infer.py                         # interactive REPL
  python gpu_infer.py --prompt "2+2="         # single prompt
  python gpu_infer.py --prompt "hello" -n 64  # generate 64 tokens
"""

import torch
import torch.nn.functional as F
import argparse
import sys
import os
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mamba_engine import (
    RecursiveMamba2_PrefixScratchpad,
    fuse_lora_weights,
    tokenizer,
    HALT_ID,
    MODEL_ID,
)


# ── Constants ─────────────────────────────────────────────────────────────────
CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mamba2_2.7b_phase2_joint_best.pt",
)
DEVICE = "cuda"
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.9


def load_model(checkpoint_path: str, device: str = "cuda") -> RecursiveMamba2_PrefixScratchpad:
    """Load the Mamba2-2.7B + RLF model from checkpoint onto GPU.

    Fuses LoRA weights to save ~0.5GB VRAM, then moves to device.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Target device ('cuda' or 'cpu')

    Returns:
        Model in eval mode on target device
    """
    print(f"Loading backbone: {MODEL_ID}")
    from mamba_ssm import MambaLMHeadModel

    backbone = MambaLMHeadModel.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device="cpu"
    )

    # Manually resize embeddings (MambaLMHeadModel lacks resize_token_embeddings)
    new_vocab = len(tokenizer)
    old_embed = backbone.backbone.embedding
    old_vocab = old_embed.weight.shape[0]
    if new_vocab > old_vocab:
        print(f"  Expanding vocab: {old_vocab} → {new_vocab} (+{new_vocab - old_vocab} tokens)")
        new_embed = torch.nn.Embedding(new_vocab, old_embed.embedding_dim, dtype=old_embed.weight.dtype)
        new_embed.weight.data[:old_vocab] = old_embed.weight.data
        torch.nn.init.normal_(new_embed.weight.data[old_vocab:], mean=0.0, std=0.02)
        backbone.backbone.embedding = new_embed

        # Also resize lm_head if it's a Linear
        old_head = backbone.lm_head
        if hasattr(old_head, 'weight') and old_head.weight.shape[0] == old_vocab:
            new_head = torch.nn.Linear(old_head.in_features, new_vocab, bias=old_head.bias is not None, dtype=old_head.weight.dtype)
            new_head.weight.data[:old_vocab] = old_head.weight.data
            torch.nn.init.zeros_(new_head.weight.data[old_vocab:])
            if old_head.bias is not None:
                new_head.bias.data[:old_vocab] = old_head.bias.data
                new_head.bias.data[old_vocab:] = 0.0
            backbone.lm_head = new_head

    print("Building RLF wrapper...")
    model = RecursiveMamba2_PrefixScratchpad(backbone, lora_rank=4)

    print(f"Loading checkpoint: {os.path.basename(checkpoint_path)}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)

    # Fuse LoRA weights to reclaim VRAM
    print("Fusing LoRA weights...")
    fuse_lora_weights(model)

    # Move to GPU
    print(f"Moving to {device}...")
    model = model.to(dtype=torch.bfloat16, device=device)
    model.eval()

    # Free CPU leftovers
    del backbone, ckpt, state
    gc.collect()
    torch.cuda.empty_cache()

    vram_used = torch.cuda.memory_allocated() / 1e9
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram_used:.1f}GB / {vram_total:.1f}GB")
    print()
    return model


def generate_rlf(
    model: RecursiveMamba2_PrefixScratchpad,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    top_p: float = TOP_P,
) -> str:
    """Generate text using the RLF reasoning loop.

    First runs the RLF loop to get the reasoning trace,
    then autoregressively generates tokens.

    Args:
        model: Loaded model in eval mode
        prompt: Input text prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering value
        top_p: Nucleus sampling threshold

    Returns:
        Generated text string
    """
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        # ── RLF reasoning loop ────────────────────────────────────────────
        n_loops, trace, last_answer = model(input_ids)
        print(f"  RLF: {n_loops} loops")
        for step, tok, conf in trace:
            print(f"    {step}: '{tok}' (p={conf})")

        # ── Autoregressive generation ─────────────────────────────────────
        generated_ids = input_ids.clone()
        generated_tokens = []

        for _ in range(max_new_tokens):
            # Run full forward pass for next-token prediction
            # Use the backbone directly for autoregressive continuation
            x = model.backbone.embedding(generated_ids)
            residual = None
            for layer in model.all_layers:
                x, residual = layer(x, residual)

            logits = model.lm_head(
                model.norm(x, residual, prenorm=False)
            )
            next_logits = logits[0, -1, :].float()

            # Temperature scaling
            if temperature > 0:
                next_logits = next_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    topk_vals, _ = torch.topk(next_logits, top_k)
                    threshold = topk_vals[-1]
                    next_logits[next_logits < threshold] = float('-inf')

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    remove_mask = cumulative_probs > top_p
                    remove_mask[1:] = remove_mask[:-1].clone()
                    remove_mask[0] = False
                    sorted_logits[remove_mask] = float('-inf')
                    next_logits = torch.zeros_like(next_logits).scatter(
                        0, sorted_indices, sorted_logits
                    )

                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = next_logits.argmax(dim=-1, keepdim=True)

            token_id = next_id.item()

            # Stop on EOS or HALT
            if token_id == tokenizer.eos_token_id or token_id == HALT_ID:
                break

            generated_tokens.append(token_id)
            generated_ids = torch.cat(
                [generated_ids, next_id.unsqueeze(0)], dim=1
            )

    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return output_text


def interactive_repl(model: RecursiveMamba2_PrefixScratchpad) -> None:
    """Run an interactive REPL for chatting with the model.

    Args:
        model: Loaded model in eval mode
    """
    print("═" * 60)
    print("  Mamba2-2.7B + RLF  ·  GPU Inference")
    print("  Type 'quit' or Ctrl+C to exit")
    print("═" * 60)
    print()

    while True:
        try:
            prompt = input(">>> ").strip()
            if not prompt or prompt.lower() in ("quit", "exit", "q"):
                break

            output = generate_rlf(model, prompt)
            print(f"\n{output}\n")

        except KeyboardInterrupt:
            print("\n\nExiting.")
            break
        except Exception as e:
            print(f"\n  Error: {e}\n")


def main() -> int:
    """Entry point: parse args, load model, run inference.

    Returns:
        Exit code (0 = success)
    """
    parser = argparse.ArgumentParser(
        description="Mamba2-2.7B + RLF GPU Inference"
    )
    parser.add_argument(
        "--prompt", "-p", type=str, default=None,
        help="Single prompt to run (skips REPL)"
    )
    parser.add_argument(
        "--max-tokens", "-n", type=int, default=MAX_NEW_TOKENS,
        help=f"Max new tokens to generate (default: {MAX_NEW_TOKENS})"
    )
    parser.add_argument(
        "--temperature", "-t", type=float, default=TEMPERATURE,
        help=f"Sampling temperature (default: {TEMPERATURE})"
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"Top-k filtering (default: {TOP_K})"
    )
    parser.add_argument(
        "--top-p", type=float, default=TOP_P,
        help=f"Top-p nucleus sampling (default: {TOP_P})"
    )
    parser.add_argument(
        "--checkpoint", "-c", type=str, default=CHECKPOINT,
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--device", "-d", type=str, default=DEVICE,
        help="Device (cuda/cpu)"
    )
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)

    if args.prompt:
        output = generate_rlf(
            model, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(output)
    else:
        interactive_repl(model)

    return 0


if __name__ == "__main__":
    sys.exit(main())
