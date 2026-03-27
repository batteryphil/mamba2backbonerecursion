"""
export_mamba_baremetal.py — Export RLF Mamba2 PyTorch checkpoint to .mamba.bin
================================================================================
Converts a trained RecursiveMamba2_v34 checkpoint into the flat float32
binary format readable by the bare-metal SSM inference engine (ssm_weights.c).

Usage:
    python export_mamba_baremetal.py [checkpoint.pt] [output.mamba.bin]

Default:
    checkpoint: mamba2_130m_v34_rope_best.pt
    output:     model.mamba.bin
"""

import struct
import sys
import os
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

MAMBA_BIN_MAGIC   = 0x4D414D42   # "MAMB" little-endian
MAMBA_BIN_VERSION = 1

# Header: 24 ints (4 bytes each) + 1 uint64 (8 bytes) = 104 bytes total
# reserved[0] is now quant_type: 0=fp32, 1=int8
HEADER_FORMAT = '<' + 'I' * 2 + 'i' * 13 + 'i' * 9 + 'Q'
HEADER_SIZE   = struct.calcsize(HEADER_FORMAT)

# Quantization types
QUANT_FP32 = 0
QUANT_INT8 = 1


def write_header(
    f,
    d_model: int,
    d_state: int,
    d_conv: int,
    expand: int,
    n_layers: int,
    vocab_size: int,
    max_seq_len: int,
    base_split: int,
    max_rlf_loops: int,
    halt_token_id: int,
    rope_base: int,
    dt_rank: int,
    has_rlf: int,
    total_bytes: int,
    quant_type: int = QUANT_FP32,
) -> None:
    """Write the .mamba.bin header to file handle."""
    reserved = [0] * 9
    reserved[0] = quant_type     # quant_type: 0=fp32, 1=int8
    header = struct.pack(
        HEADER_FORMAT,
        MAMBA_BIN_MAGIC,
        MAMBA_BIN_VERSION,
        d_model,
        d_state,
        d_conv,
        expand,
        n_layers,
        vocab_size,
        max_seq_len,
        base_split,
        max_rlf_loops,
        halt_token_id,
        rope_base,
        dt_rank,
        has_rlf,
        *reserved,
        total_bytes,
    )
    f.write(header)


def write_tensor(f, tensor, name: str = "", quant: int = QUANT_FP32) -> int:
    """Write a tensor to file. Supports fp32 and int8 quantization."""
    arr = tensor.detach().cpu().float().numpy().flatten()
    if quant == QUANT_INT8:
        # Per-tensor absmax quantization: scale = max(|W|) / 127
        absmax = np.max(np.abs(arr))
        scale = absmax / 127.0 if absmax > 0 else 1.0
        quantized = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
        # Write: [float32 scale][int8[] data]
        scale_bytes = np.float32(scale).tobytes()
        quant_bytes = quantized.tobytes()
        f.write(scale_bytes)
        f.write(quant_bytes)
        total = len(scale_bytes) + len(quant_bytes)
        if name:
            ratio = len(arr) * 4 / total
            print(f"    {name:40s} {str(list(tensor.shape)):20s} → {total:>12,} bytes (int8, {ratio:.1f}× smaller)")
        return total
    else:
        data = arr.tobytes()
        f.write(data)
        if name:
            print(f"    {name:40s} {str(list(tensor.shape)):20s} → {len(data):>12,} bytes")
        return len(data)


def calc_total_bytes(
    d_model: int,
    d_state: int,
    d_conv: int,
    expand: int,
    n_layers: int,
    vocab_size: int,
    dt_rank: int,
    has_rlf: bool,
) -> int:
    """Calculate total file size in bytes."""
    d_inner = d_model * expand
    xbc_dim = dt_rank + 2 * d_state

    total = HEADER_SIZE

    # Global tensors
    total += vocab_size * d_model * 4    # token_embedding
    total += vocab_size * d_model * 4    # lm_head
    total += d_model * 4                 # final_norm

    # Per-layer tensors
    per_layer = (
        d_model * 4 +                    # norm_weight
        2 * d_inner * d_model * 4 +      # in_proj
        d_inner * d_conv * 4 +           # conv1d_weight
        d_inner * 4 +                    # conv1d_bias
        xbc_dim * d_inner * 4 +          # x_proj
        d_inner * dt_rank * 4 +          # dt_proj_weight
        d_inner * 4 +                    # dt_proj_bias
        d_inner * d_state * 4 +          # A_log
        d_inner * 4 +                    # D
        d_model * d_inner * 4            # out_proj
    )
    total += per_layer * n_layers

    # RLF tensors
    if has_rlf:
        total += d_model * 4                 # lifeline_gate
        total += d_model * 4                 # loop_norm_weight
        total += 2 * d_inner * d_model * 4   # loop_in_proj
        total += d_inner * d_conv * 4        # loop_conv1d_weight
        total += d_inner * 4                 # loop_conv1d_bias
        total += xbc_dim * d_inner * 4       # loop_x_proj
        total += d_inner * dt_rank * 4       # loop_dt_proj_weight
        total += d_inner * 4                 # loop_dt_proj_bias
        total += d_inner * d_state * 4       # loop_A_log
        total += d_inner * 4                 # loop_D
        total += d_model * d_inner * 4       # loop_out_proj

    return total


def export_checkpoint(ckpt_path: str, output_path: str, quant_type: int = QUANT_FP32) -> None:
    """Export a trained RecursiveMamba2_v34 checkpoint to .mamba.bin."""
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch required for export. Install with: pip install torch")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  Mamba2 RLF → Bare-Metal Export")
    print(f"  Input:  {ckpt_path}")
    print(f"  Output: {output_path}")
    print(f"{'='*70}\n")

    # Load checkpoint
    print("  Loading checkpoint...", flush=True)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)

    # Detect model config from weight shapes
    # Find d_model from embedding
    emb_key = None
    for k in sd:
        if 'embedding' in k.lower() and 'weight' in k.lower():
            emb_key = k
            break
    if not emb_key:
        print("ERROR: Cannot find embedding weight in checkpoint")
        sys.exit(1)

    emb_w = sd[emb_key]
    vocab_size, d_model = emb_w.shape
    print(f"  Detected: vocab={vocab_size:,}, d_model={d_model}")

    # Count layers
    layer_indices = set()
    for k in sd:
        if 'all_layers.' in k:
            parts = k.split('.')
            for i, p in enumerate(parts):
                if p == 'all_layers' and i + 1 < len(parts):
                    try:
                        layer_indices.add(int(parts[i + 1]))
                    except ValueError:
                        pass
    n_layers = len(layer_indices)
    print(f"  Detected: n_layers={n_layers}")

    # Detect RLF
    has_rlf = 'lifeline_gate' in sd
    if has_rlf:
        print(f"  Detected: RLF lifeline gate present")

    # Mamba2 defaults
    d_state      = 64
    d_conv       = 4
    expand        = 2
    d_inner       = d_model * expand
    dt_rank       = d_model
    max_seq_len   = 256
    base_split    = 6
    max_rlf_loops = 16
    rope_base     = 10000

    # Extract halt_token_id from checkpoint
    halt_token_id = ckpt.get('halt_id', 50281)
    print(f"  halt_token_id: {halt_token_id}")

    # Calculate total size
    total_bytes = calc_total_bytes(
        d_model, d_state, d_conv, expand,
        n_layers, vocab_size, dt_rank, has_rlf
    )
    print(f"  Total export size: {total_bytes:,} bytes ({total_bytes/1e6:.1f} MB)\n")

    # Helper to find tensors by partial key
    def find(pattern: str, must: bool = True):
        """Find tensor by partial key match."""
        candidates = [k for k in sd if pattern in k]
        if len(candidates) == 1:
            return sd[candidates[0]]
        if len(candidates) > 1:
            # Prefer exact suffix match
            for c in candidates:
                if c.endswith(pattern):
                    return sd[c]
            return sd[candidates[0]]
        if must:
            print(f"  WARNING: tensor not found: {pattern}")
            return torch.zeros(1)
        return None

    def find_layer(layer_idx: int, suffix: str, must: bool = True):
        """Find per-layer tensor."""
        pattern = f"all_layers.{layer_idx}.mixer.{suffix}"
        return find(pattern, must=must)

    # ── Write binary ──────────────────────────────────────────────────────
    with open(output_path, 'wb') as f:
        # Header
        write_header(
            f,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_layers=n_layers,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            base_split=base_split,
            max_rlf_loops=max_rlf_loops,
            halt_token_id=halt_token_id,
            rope_base=rope_base,
            dt_rank=dt_rank,
            has_rlf=1 if has_rlf else 0,
            total_bytes=total_bytes,
            quant_type=quant_type,
        )

        q = quant_type  # shorthand for write_tensor calls
        print("  Writing global tensors:")
        write_tensor(f, emb_w, "token_embedding", quant=q)
        write_tensor(f, find('lm_head.weight'), "lm_head", quant=q)
        final_norm = find('norm_f.weight', must=False)
        if final_norm is None:
            final_norm = find('norm.weight', must=False)
        if final_norm is None:
            final_norm = find('backbone.norm_f.weight', must=False)
        if final_norm is None:
            import torch
            final_norm = torch.ones(d_model)
        write_tensor(f, final_norm, "final_norm", quant=q)
        print("\n  Writing per-layer tensors:")
        for l in sorted(layer_indices):
            print(f"    ── Layer {l} ──")

            # Norm weight
            norm_w = find(f'all_layers.{l}.norm.weight', must=False)
            if norm_w is None:
                norm_w = torch.ones(d_model)
            write_tensor(f, norm_w, f"  norm_weight[{l}]", quant=q)
            # in_proj (may have LoRA)
            in_proj_base = find_layer(l, 'in_proj.base_weight', must=False)
            in_proj_loraA = find_layer(l, 'in_proj.lora_A', must=False)
            in_proj_loraB = find_layer(l, 'in_proj.lora_B', must=False)
            in_proj_plain = find_layer(l, 'in_proj.weight', must=False)

            if in_proj_base is not None and in_proj_loraA is not None:
                # Merge LoRA: W = base + scale * (B @ A)
                rank = in_proj_loraA.shape[0]
                scale = 2.0  # alpha/rank, typically alpha = rank*2
                merged = in_proj_base + scale * (in_proj_loraB @ in_proj_loraA)
                write_tensor(f, merged, f"  in_proj[{l}] (LoRA merged)", quant=q)
            elif in_proj_plain is not None:
                write_tensor(f, in_proj_plain, f"  in_proj[{l}]", quant=q)
            else:
                write_tensor(f, torch.zeros(2 * d_inner, d_model),
                           f"  in_proj[{l}] (zeros)")

            # conv1d
            conv_w = find_layer(l, 'conv1d.weight', must=False)
            conv_b = find_layer(l, 'conv1d.bias', must=False)
            if conv_w is not None:
                # Conv weight shape: [d_inner, 1, d_conv] → [d_inner, d_conv]
                if conv_w.dim() == 3:
                    conv_w = conv_w.squeeze(1)
                write_tensor(f, conv_w, f"  conv1d_w[{l}]", quant=q)
            else:
                write_tensor(f, torch.zeros(d_inner, d_conv), f"  conv1d_w[{l}] (zeros)", quant=q)
            write_tensor(f, conv_b if conv_b is not None else torch.zeros(d_inner),
                       f"  conv1d_b[{l}]")

            # x_proj
            x_proj_w = find_layer(l, 'x_proj.weight', must=False)
            xbc_dim = dt_rank + 2 * d_state
            if x_proj_w is not None:
                write_tensor(f, x_proj_w, f"  x_proj[{l}]", quant=q)
            else:
                write_tensor(f, torch.zeros(xbc_dim, d_inner), f"  x_proj[{l}] (zeros)", quant=q)
            # dt_proj
            dt_w = find_layer(l, 'dt_proj.weight', must=False)
            dt_b = find_layer(l, 'dt_proj.bias', must=False)
            if dt_w is not None:
                write_tensor(f, dt_w, f"  dt_proj_w[{l}]", quant=q)
            else:
                write_tensor(f, torch.zeros(d_inner, dt_rank), f"  dt_proj_w[{l}] (zeros)", quant=q)
            write_tensor(f, dt_b if dt_b is not None else torch.zeros(d_inner),
                       f"  dt_proj_b[{l}]")

            # A_log
            a_log = find_layer(l, 'A_log', must=False)
            if a_log is not None:
                write_tensor(f, a_log, f"  A_log[{l}]", quant=q)
            else:
                write_tensor(f, torch.zeros(d_inner, d_state), f"  A_log[{l}] (zeros)", quant=q)
            # D
            d_param = find_layer(l, 'D', must=False)
            if d_param is not None:
                write_tensor(f, d_param, f"  D[{l}]", quant=q)
            else:
                write_tensor(f, torch.ones(d_inner), f"  D[{l}] (ones)", quant=q)
            # out_proj (may have LoRA)
            out_base = find_layer(l, 'out_proj.base_weight', must=False)
            out_loraA = find_layer(l, 'out_proj.lora_A', must=False)
            out_loraB = find_layer(l, 'out_proj.lora_B', must=False)
            out_plain = find_layer(l, 'out_proj.weight', must=False)

            if out_base is not None and out_loraA is not None:
                rank = out_loraA.shape[0]
                scale = 2.0
                merged = out_base + scale * (out_loraB @ out_loraA)
                write_tensor(f, merged, f"  out_proj[{l}] (LoRA merged)", quant=q)
            elif out_plain is not None:
                write_tensor(f, out_plain, f"  out_proj[{l}]", quant=q)
            else:
                write_tensor(f, torch.zeros(d_model, d_inner),
                           f"  out_proj[{l}] (zeros)")

        # RLF weights
        if has_rlf:
            print("\n  Writing RLF weights:")
            write_tensor(f, sd['lifeline_gate'], "lifeline_gate", quant=q)
            loop_norm = find('loop_norm.weight', must=False)
            write_tensor(f, loop_norm if loop_norm is not None else torch.ones(d_model),
                       "loop_norm_weight")

            # Loop core Mamba2 block
            loop_in = find('mamba2_core.in_proj.weight', must=False)
            write_tensor(f, loop_in if loop_in is not None
                       else torch.zeros(2 * d_inner, d_model),
                       "loop_in_proj")

            loop_conv_w = find('mamba2_core.conv1d.weight', must=False)
            if loop_conv_w is not None and loop_conv_w.dim() == 3:
                loop_conv_w = loop_conv_w.squeeze(1)
            write_tensor(f, loop_conv_w if loop_conv_w is not None
                       else torch.zeros(d_inner, d_conv),
                       "loop_conv1d_weight")

            loop_conv_b = find('mamba2_core.conv1d.bias', must=False)
            write_tensor(f, loop_conv_b if loop_conv_b is not None
                       else torch.zeros(d_inner),
                       "loop_conv1d_bias")

            loop_xp = find('mamba2_core.x_proj.weight', must=False)
            write_tensor(f, loop_xp if loop_xp is not None
                       else torch.zeros(dt_rank + 2 * d_state, d_inner),
                       "loop_x_proj")

            loop_dt_w = find('mamba2_core.dt_proj.weight', must=False)
            write_tensor(f, loop_dt_w if loop_dt_w is not None
                       else torch.zeros(d_inner, dt_rank),
                       "loop_dt_proj_weight")

            loop_dt_b = find('mamba2_core.dt_proj.bias', must=False)
            write_tensor(f, loop_dt_b if loop_dt_b is not None
                       else torch.zeros(d_inner),
                       "loop_dt_proj_bias")

            loop_a = find('mamba2_core.A_log', must=False)
            write_tensor(f, loop_a if loop_a is not None
                       else torch.zeros(d_inner, d_state),
                       "loop_A_log")

            loop_d = find('mamba2_core.D', must=False)
            write_tensor(f, loop_d if loop_d is not None
                       else torch.ones(d_inner),
                       "loop_D")

            loop_out = find('mamba2_core.out_proj.weight', must=False)
            write_tensor(f, loop_out if loop_out is not None
                       else torch.zeros(d_model, d_inner),
                       "loop_out_proj")

    actual_size = os.path.getsize(output_path)
    print(f"\n  ✅ Export complete: {output_path}")
    print(f"     Expected: {total_bytes:,} bytes")
    print(f"     Actual:   {actual_size:,} bytes")
    if actual_size != total_bytes:
        print(f"     ⚠️  Size mismatch! (expected {total_bytes}, got {actual_size})")
    else:
        print(f"     ✅ Size matches perfectly")
    print()


def create_test_model(output_path: str) -> None:
    """Create a tiny random .mamba.bin for testing (no PyTorch needed)."""
    d_model      = 32
    d_state      = 8
    d_conv       = 4
    expand        = 2
    d_inner       = d_model * expand
    n_layers      = 2
    vocab_size    = 100
    dt_rank       = d_model
    xbc_dim       = dt_rank + 2 * d_state

    total_bytes = calc_total_bytes(
        d_model, d_state, d_conv, expand,
        n_layers, vocab_size, dt_rank, has_rlf=True
    )

    rng = np.random.default_rng(42)

    with open(output_path, 'wb') as f:
        write_header(
            f,
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
            n_layers=n_layers, vocab_size=vocab_size, max_seq_len=64,
            base_split=1, max_rlf_loops=4, halt_token_id=99,
            rope_base=10000, dt_rank=dt_rank, has_rlf=1,
            total_bytes=total_bytes,
        )

        def write_rand(shape: tuple) -> None:
            """Write random float32 tensor."""
            count = 1
            for s in shape:
                count *= s
            data = rng.standard_normal(count).astype(np.float32)
            f.write(data.tobytes())

        # Global
        write_rand((vocab_size, d_model))   # token_embedding
        write_rand((vocab_size, d_model))   # lm_head
        write_rand((d_model,))              # final_norm

        # Per-layer
        for _ in range(n_layers):
            write_rand((d_model,))                   # norm_weight
            write_rand((2 * d_inner, d_model))       # in_proj
            write_rand((d_inner, d_conv))             # conv1d_weight
            write_rand((d_inner,))                    # conv1d_bias
            write_rand((xbc_dim, d_inner))            # x_proj
            write_rand((d_inner, dt_rank))            # dt_proj_weight
            write_rand((d_inner,))                    # dt_proj_bias
            write_rand((d_inner, d_state))            # A_log
            write_rand((d_inner,))                    # D
            write_rand((d_model, d_inner))            # out_proj

        # RLF
        write_rand((d_model,))                       # lifeline_gate
        write_rand((d_model,))                       # loop_norm_weight
        write_rand((2 * d_inner, d_model))           # loop_in_proj
        write_rand((d_inner, d_conv))                 # loop_conv1d_weight
        write_rand((d_inner,))                        # loop_conv1d_bias
        write_rand((xbc_dim, d_inner))                # loop_x_proj
        write_rand((d_inner, dt_rank))                # loop_dt_proj_weight
        write_rand((d_inner,))                        # loop_dt_proj_bias
        write_rand((d_inner, d_state))                # loop_A_log
        write_rand((d_inner,))                        # loop_D
        write_rand((d_model, d_inner))                # loop_out_proj

    actual = os.path.getsize(output_path)
    print(f"  Test model: {output_path} ({actual:,} bytes)")
    assert actual == total_bytes, f"Size mismatch: {actual} != {total_bytes}"


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export Mamba2 RLF to .mamba.bin')
    parser.add_argument('checkpoint', nargs='?', default='mamba2_130m_v34_rope_best.pt',
                        help='Path to checkpoint or --test')
    parser.add_argument('output', nargs='?', default='model.mamba.bin',
                        help='Output .mamba.bin path')
    parser.add_argument('--quantize', choices=['fp32', 'int8'], default='fp32',
                        help='Quantization type (default: fp32)')
    args = parser.parse_args()

    qt = QUANT_INT8 if args.quantize == 'int8' else QUANT_FP32

    if args.checkpoint == '--test':
        create_test_model(args.output)
    elif os.path.exists(args.checkpoint):
        export_checkpoint(args.checkpoint, args.output, quant_type=qt)

        # Generate SHA-256 hash and OOHANDOFF.TXT for oo-host integration
        import hashlib
        with open(args.output, 'rb') as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()
        print(f"\n  Model SHA-256: {model_hash}")

        handoff_path = os.path.join(os.path.dirname(args.output) or '.', 'OOHANDOFF.TXT')
        with open(handoff_path, 'w') as f:
            f.write(f"organism_id=sovereign\n")
            f.write(f"genesis_id=genesis\n")
            f.write(f"mode=normal\n")
            f.write(f"continuity_epoch=0\n")
            f.write(f"boot_count=1\n")
            f.write(f"model_hash={model_hash}\n")
            f.write(f"model_name={os.path.basename(args.output)}\n")
            f.write(f"quant_type={'int8' if qt == QUANT_INT8 else 'fp32'}\n")
            f.write(f"lifeline_enabled=1\n")
            f.write(f"rlf_max_loops=16\n")
        print(f"  OOHANDOFF.TXT → {handoff_path}")
        print(f"  (for oo-host sync: oo-bot sync-check --workspace .)")
    else:
        print(f"Checkpoint not found: {args.checkpoint}")
        parser.print_help()
        sys.exit(1)
