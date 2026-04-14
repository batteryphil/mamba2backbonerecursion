"""
export_mamba_baremetal.py — Export Mamba2-2.7B RLF checkpoint to .mamba.bin v2
================================================================================
Exports ALL tensors faithfully as flat float32/int8, using the actual Mamba2
multi-head SSM architecture (NOT Mamba1).

Mamba2 per-layer tensors:
  in_proj.weight  [2*d_inner + 2*ngroups*d_state + nheads, d_model]
  conv1d.weight   [d_inner + 2*ngroups*d_state, 1, d_conv]  → squeezed to 2D
  conv1d.bias     [d_inner + 2*ngroups*d_state]
  norm.weight     [d_inner]
  out_proj.weight [d_model, d_inner]
  A_log           [nheads]
  D               [nheads]
  dt_bias         [nheads]

Usage:
    python export_mamba_baremetal.py [checkpoint.pt] [output.mamba.bin] [--quantize int8]
"""

import struct
import sys
import os
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

MAMBA_BIN_MAGIC   = 0x4D414D42   # "MAMB" little-endian
MAMBA_BIN_VERSION = 2

# Header v2 layout:
#   2 × uint32  (magic, version)
#  25 × int32   (config fields + reserved)
#   1 × uint64  (total_bytes)
HEADER_FIELDS = [
    'magic', 'version',                              # 2 × uint32
    'd_model', 'd_state', 'd_conv', 'expand',        # model shape
    'n_layers', 'vocab_size', 'max_seq_len',
    'base_split', 'max_rlf_loops', 'halt_token_id',  # RLF
    'rope_base',
    'nheads', 'headdim', 'ngroups',                   # Mamba2-specific
    'has_rlf', 'quant_type',
    'prefix_m', 'bridge_rank',
    'loop_nheads', 'loop_headdim', 'loop_d_state',    # Loop engine
    'reserved0', 'reserved1',                          # reserved
    'total_bytes',                                     # uint64
]
HEADER_FORMAT = '<' + 'I' * 2 + 'i' * 23 + 'Q'
HEADER_SIZE   = struct.calcsize(HEADER_FORMAT)

QUANT_FP32 = 0
QUANT_INT8 = 1


def write_header(f, **kw) -> None:
    """Write the .mamba.bin v2 header to file handle."""
    header = struct.pack(
        HEADER_FORMAT,
        kw.get('magic', MAMBA_BIN_MAGIC),
        kw.get('version', MAMBA_BIN_VERSION),
        kw['d_model'], kw['d_state'], kw['d_conv'], kw['expand'],
        kw['n_layers'], kw['vocab_size'], kw['max_seq_len'],
        kw['base_split'], kw['max_rlf_loops'], kw['halt_token_id'],
        kw['rope_base'],
        kw['nheads'], kw['headdim'], kw['ngroups'],
        kw['has_rlf'], kw['quant_type'],
        kw['prefix_m'], kw['bridge_rank'],
        kw['loop_nheads'], kw['loop_headdim'], kw['loop_d_state'],
        kw.get('has_proprio_gate', 0), kw.get('has_post_lora', 0),
        kw['total_bytes'],
    )
    f.write(header)


def write_tensor(f, tensor, name: str = "", quant: int = QUANT_FP32) -> int:
    """Write a tensor to file. Supports fp32 and int8 quantization."""
    import torch
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().float().numpy().flatten()
    else:
        arr = np.asarray(tensor, dtype=np.float32).flatten()

    if quant == QUANT_INT8:
        absmax = np.max(np.abs(arr))
        scale = absmax / 127.0 if absmax > 0 else 1.0
        quantized = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
        scale_bytes = np.float32(scale).tobytes()
        quant_bytes = quantized.tobytes()
        f.write(scale_bytes)
        f.write(quant_bytes)
        total = len(scale_bytes) + len(quant_bytes)
        if name:
            ratio = len(arr) * 4 / total
            print(f"    {name:45s} {str(list(tensor.shape)):28s} → {total:>12,} bytes (int8, {ratio:.1f}×)")
        return total
    else:
        data = arr.tobytes()
        f.write(data)
        if name:
            print(f"    {name:45s} {str(list(tensor.shape)):28s} → {len(data):>12,} bytes")
        return len(data)


def export_checkpoint(ckpt_path: str, output_path: str, quant_type: int = QUANT_FP32) -> None:
    """Export a trained RecursiveMamba2_PrefixScratchpad checkpoint to .mamba.bin v2."""
    import torch

    print(f"\n{'='*72}")
    print(f"  Mamba2-2.7B RLF → Bare-Metal Export (v2, Mamba2 multi-head)")
    print(f"  Input:  {ckpt_path}")
    print(f"  Output: {output_path}")
    print(f"  Quant:  {'int8' if quant_type == QUANT_INT8 else 'fp32'}")
    print(f"{'='*72}\n")

    print("  Loading checkpoint...", flush=True)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)

    # ── Extract metadata from checkpoint ──────────────────────────────────
    d_model      = ckpt.get('d_model', 2560)
    halt_token_id = ckpt.get('halt_id', 50278)
    prefix_m     = ckpt.get('prefix_m', 8)
    has_bridge   = ckpt.get('has_bridge', True)

    # Detect vocab from embedding
    emb_key = next(k for k in sd if 'embedding' in k and 'weight' in k)
    emb_w = sd[emb_key]
    vocab_size = emb_w.shape[0]
    d_model = emb_w.shape[1]

    # Count layers
    layer_ids = set()
    for k in sd:
        if 'all_layers.' in k:
            parts = k.split('.')
            idx = parts.index('all_layers')
            if idx + 1 < len(parts):
                try:
                    layer_ids.add(int(parts[idx + 1]))
                except ValueError:
                    pass
    n_layers = len(layer_ids)

    # ── Detect Mamba2 architecture from layer 0 ──────────────────────────
    # in_proj = [2*d_inner + 2*ngroups*d_state + nheads, d_model]
    ip_key = next(k for k in sd if 'all_layers.0.mixer.in_proj' in k and 'weight' in k)
    in_proj_rows = sd[ip_key].shape[0]

    a_key = next(k for k in sd if 'all_layers.0.mixer.A_log' in k)
    nheads = sd[a_key].shape[0]
    headdim = d_model * 2 // nheads  # d_inner / nheads = (d_model * expand) / nheads
    expand = 2
    d_inner = d_model * expand
    d_state = (in_proj_rows - 2 * d_inner - nheads) // 2  # Solve for d_state (ngroups=1)
    ngroups = 1
    d_conv = 4

    # Conv dim = d_inner + 2*ngroups*d_state
    conv_key = next(k for k in sd if 'all_layers.0.mixer.conv1d.weight' in k)
    conv_dim = sd[conv_key].shape[0]
    expected_conv_dim = d_inner + 2 * ngroups * d_state
    assert conv_dim == expected_conv_dim, f"conv_dim={conv_dim} != expected={expected_conv_dim}"

    print(f"  Backbone:  d_model={d_model}, expand={expand}, d_inner={d_inner}")
    print(f"             nheads={nheads}, headdim={headdim}, d_state={d_state}, ngroups={ngroups}")
    print(f"             n_layers={n_layers}, vocab={vocab_size:,}")

    # ── Detect loop core architecture ─────────────────────────────────────
    loop_a_key = next((k for k in sd if 'mamba2_core.A_log' in k), None)
    if loop_a_key:
        loop_nheads = sd[loop_a_key].shape[0]
        loop_ip_key = next(k for k in sd if 'mamba2_core.in_proj.weight' in k)
        loop_in_rows = sd[loop_ip_key].shape[0]
        loop_out_key = next(k for k in sd if 'mamba2_core.out_proj.weight' in k)
        loop_d_inner = sd[loop_out_key].shape[1]
        loop_headdim = loop_d_inner // loop_nheads
        loop_d_state = (loop_in_rows - 2 * loop_d_inner - loop_nheads) // 2
        print(f"  Loop core: nheads={loop_nheads}, headdim={loop_headdim}, d_state={loop_d_state}, d_inner={loop_d_inner}")
    else:
        loop_nheads = 0
        loop_headdim = 0
        loop_d_state = 0

    has_rlf = 'lifeline_gate' in sd
    has_scratchpad = 'latent_memory' in sd
    bridge_rank = 64 if has_bridge else 0

    print(f"  RLF: {'✓' if has_rlf else '✗'}  Scratchpad: {'✓' if has_scratchpad else '✗'}  Bridge: {'✓' if has_bridge else '✗'}")
    print(f"  halt_id={halt_token_id}, prefix_m={prefix_m}, bridge_rank={bridge_rank}")

    # ── Helper functions ──────────────────────────────────────────────────
    def find(pattern: str, must: bool = True):
        """Find tensor by partial key match."""
        candidates = [k for k in sd if pattern in k]
        if len(candidates) == 1:
            return sd[candidates[0]]
        if len(candidates) > 1:
            for c in candidates:
                if c.endswith(pattern):
                    return sd[c]
            return sd[candidates[0]]
        if must:
            print(f"  WARNING: tensor not found: {pattern}")
            return torch.zeros(1)
        return None

    def merge_lora(base_key: str, prefix: str) -> torch.Tensor:
        """Merge LoRA weights if present, else return plain weight."""
        base = find(f'{prefix}.base_weight', must=False)
        lora_a = find(f'{prefix}.lora_A', must=False)
        lora_b = find(f'{prefix}.lora_B', must=False)
        plain = find(f'{prefix}.weight', must=False)

        if base is not None and lora_a is not None and lora_b is not None:
            rank = lora_a.shape[0]
            alpha = rank * 2.0
            scale = alpha / rank
            merged = base + scale * (lora_b @ lora_a)
            return merged
        elif plain is not None:
            return plain
        else:
            print(f"  WARNING: no weight found for {prefix}")
            return torch.zeros(1)

    # ── Write binary ──────────────────────────────────────────────────────
    with open(output_path, 'wb') as f:
        # Placeholder header
        f.write(b'\x00' * HEADER_SIZE)
        q = quant_type
        written = HEADER_SIZE

        # ── Global tensors ────────────────────────────────────────────────
        print("\n  Writing global tensors:")
        written += write_tensor(f, emb_w, "token_embedding", quant=q)
        written += write_tensor(f, find('lm_head.weight'), "lm_head", quant=q)

        fn = find('backbone.norm_f.weight', must=False)
        if fn is None:
            fn = find('norm_f.weight', must=False)
        if fn is None:
            fn = torch.ones(d_model)
        written += write_tensor(f, fn, "final_norm", quant=q)

        # ── Per-layer tensors (Mamba2 layout) ─────────────────────────────
        print("\n  Writing per-layer tensors:")
        for l in sorted(layer_ids):
            prefix = f"all_layers.{l}.mixer"

            # Layer norm (before mixer)
            lnorm = find(f'all_layers.{l}.norm.weight', must=False)
            if lnorm is None:
                lnorm = torch.ones(d_model)
            written += write_tensor(f, lnorm, f"  layer_norm[{l}]")

            # in_proj (may have LoRA on upper layers)
            in_w = merge_lora(f'{prefix}.in_proj', f'{prefix}.in_proj')
            written += write_tensor(f, in_w, f"  in_proj[{l}]", quant=q)

            # conv1d
            conv_w = find(f'{prefix}.conv1d.weight', must=False)
            if conv_w is not None and conv_w.dim() == 3:
                conv_w = conv_w.squeeze(1)  # [C, 1, K] → [C, K]
            if conv_w is None:
                conv_w = torch.zeros(conv_dim, d_conv)
            written += write_tensor(f, conv_w, f"  conv1d_w[{l}]", quant=q)

            conv_b = find(f'{prefix}.conv1d.bias', must=False)
            if conv_b is None:
                conv_b = torch.zeros(conv_dim)
            written += write_tensor(f, conv_b, f"  conv1d_b[{l}]")

            # Inner norm (Mamba2 has norm inside mixer)
            inorm = find(f'{prefix}.norm.weight', must=False)
            if inorm is None:
                inorm = torch.ones(d_inner)
            written += write_tensor(f, inorm, f"  inner_norm[{l}]")

            # out_proj (may have LoRA)
            out_w = merge_lora(f'{prefix}.out_proj', f'{prefix}.out_proj')
            written += write_tensor(f, out_w, f"  out_proj[{l}]", quant=q)

            # A_log, D, dt_bias — small, always fp32
            a_log = find(f'{prefix}.A_log', must=False)
            if a_log is None:
                a_log = torch.zeros(nheads)
            written += write_tensor(f, a_log, f"  A_log[{l}]")

            d_param = find(f'{prefix}.D', must=False)
            if d_param is None:
                d_param = torch.ones(nheads)
            written += write_tensor(f, d_param, f"  D[{l}]")

            dt_b = find(f'{prefix}.dt_bias', must=False)
            if dt_b is None:
                dt_b = torch.zeros(nheads)
            written += write_tensor(f, dt_b, f"  dt_bias[{l}]")

        # ── RLF weights ───────────────────────────────────────────────────
        if has_rlf:
            print("\n  Writing RLF weights:")
            written += write_tensor(f, sd['lifeline_gate'], "lifeline_gate")
            ln = find('loop_norm.weight', must=False)
            if ln is None:
                ln = torch.ones(d_model)
            written += write_tensor(f, ln, "loop_norm_weight")

            # Loop core Mamba2 block — same structure as backbone
            loop_in = find('mamba2_core.in_proj.weight', must=False)
            if loop_in is not None:
                written += write_tensor(f, loop_in, "loop_in_proj", quant=q)
            else:
                written += write_tensor(f, torch.zeros(1), "loop_in_proj (missing)")

            loop_cw = find('mamba2_core.conv1d.weight', must=False)
            if loop_cw is not None and loop_cw.dim() == 3:
                loop_cw = loop_cw.squeeze(1)
            if loop_cw is not None:
                written += write_tensor(f, loop_cw, "loop_conv1d_w", quant=q)
            else:
                written += write_tensor(f, torch.zeros(1), "loop_conv1d_w (missing)")

            loop_cb = find('mamba2_core.conv1d.bias', must=False)
            if loop_cb is not None:
                written += write_tensor(f, loop_cb, "loop_conv1d_b")
            else:
                written += write_tensor(f, torch.zeros(1), "loop_conv1d_b (missing)")

            loop_norm_inner = find('mamba2_core.norm.weight', must=False)
            if loop_norm_inner is not None:
                written += write_tensor(f, loop_norm_inner, "loop_inner_norm")
            else:
                written += write_tensor(f, torch.zeros(1), "loop_inner_norm (missing)")

            loop_out = find('mamba2_core.out_proj.weight', must=False)
            if loop_out is not None:
                written += write_tensor(f, loop_out, "loop_out_proj", quant=q)
            else:
                written += write_tensor(f, torch.zeros(1), "loop_out_proj (missing)")

            # A_log, D, dt_bias for loop core
            for tname in ['A_log', 'D', 'dt_bias']:
                t = find(f'mamba2_core.{tname}', must=False)
                if t is not None:
                    written += write_tensor(f, t, f"loop_{tname}")
                else:
                    written += write_tensor(f, torch.zeros(1), f"loop_{tname} (missing)")

        # ── Prefix Latent Scratchpad ──────────────────────────────────────
        if has_scratchpad:
            print("\n  Writing Prefix Scratchpad:")
            mem = sd['latent_memory']
            if mem.dim() == 3:
                mem = mem.squeeze(0)  # [1, M, d_model] → [M, d_model]
            written += write_tensor(f, mem, f"latent_memory [{prefix_m}×{d_model}]")

        # ── Latent Bridge ─────────────────────────────────────────────────
        if has_bridge:
            print("\n  Writing Latent Bridge:")
            bd = find('bridge_down.weight', must=False)
            bu = find('bridge_up.weight', must=False)
            if bd is not None:
                written += write_tensor(f, bd, f"bridge_down [{bridge_rank}×{d_model}]", quant=q)
            if bu is not None:
                written += write_tensor(f, bu, f"bridge_up [{d_model}×{bridge_rank}]", quant=q)

        # ── Phase 10: Proprioception Gate, LoRA, and Halting Head ─────────
        has_proprio = 'W_g' in sd or os.path.exists('proprio_gate_2.8b.pt')
        has_lora = any('lora_A' in k for k in sd) or os.path.exists('lora_oo_r16_final.pt')
        
        if has_proprio:
            print("\n  Writing Proprioception Gate:")
            wg = sd.get('W_g') 
            if wg is None:
                try: wg = torch.load('proprio_gate_2.8b.pt', map_location='cpu', weights_only=True)
                except: wg = torch.zeros(3, d_model)
            written += write_tensor(f, wg, "proprio_gate [3×2560]")

        if has_lora:
            print("\n  Writing Post-Backbone LoRA (OO Domain):")
            # Fallback to reading the discrete adapter if not baked into the main state_dict
            lora_sd = sd
            if not any('lora_A' in k for k in lora_sd):
                try: lora_sd = torch.load('lora_oo_r16_final.pt', map_location='cpu', weights_only=True)
                except: pass
            
            for l in range(6): # Default n_layers=6
                la = lora_sd.get(f'layers.{l}.lora_A.weight', torch.zeros(16, d_model))
                lb = lora_sd.get(f'layers.{l}.lora_B.weight', torch.zeros(d_model, 16))
                written += write_tensor(f, la, f"post_lora_A[{l}]")
                written += write_tensor(f, lb, f"post_lora_B[{l}]")

        print("\n  Writing HaltingHead v2 (OO-Semantic Classifier):")
        try:
            hh = torch.load('halting_head_v2.pt', map_location='cpu', weights_only=True)
            for i in range(1, 5):
                written += write_tensor(f, hh.get(f'mlp.{i*2-2}.weight', torch.zeros(1)), f"halt_head_w{i}")
                written += write_tensor(f, hh.get(f'mlp.{i*2-2}.bias', torch.zeros(1)), f"halt_head_b{i}")
        except:
            print("  [SKIP] halting_head_v2.pt not found")

        # ── Write real header ─────────────────────────────────────────────
        total_bytes = written
        f.seek(0)
        write_header(
            f,
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
            n_layers=n_layers, vocab_size=vocab_size, max_seq_len=512,
            base_split=48, max_rlf_loops=6, halt_token_id=halt_token_id,
            rope_base=10000,
            nheads=nheads, headdim=headdim, ngroups=ngroups,
            has_rlf=1 if has_rlf else 0, quant_type=quant_type,
            prefix_m=prefix_m if has_scratchpad else 0,
            bridge_rank=bridge_rank,
            loop_nheads=loop_nheads, loop_headdim=loop_headdim,
            loop_d_state=loop_d_state,
            has_proprio_gate=1 if has_proprio else 0,
            has_post_lora=1 if has_lora else 0,
            total_bytes=total_bytes,
        )

    actual_size = os.path.getsize(output_path)
    print(f"\n  ✅ Export complete: {output_path}")
    print(f"     Total:   {actual_size:,} bytes ({actual_size/1e9:.2f} GB)")
    if actual_size == total_bytes:
        print(f"     ✅ Size matches header")
    else:
        print(f"     ⚠️  Size mismatch: header says {total_bytes}, actual {actual_size}")
    print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export Mamba2 2.7B RLF to .mamba.bin v2')
    parser.add_argument('checkpoint', nargs='?',
                        default='mamba2_2.7b_phase2_joint_best.pt',
                        help='Path to checkpoint')
    parser.add_argument('output', nargs='?', default='model_2.7b.mamba.bin',
                        help='Output .mamba.bin path')
    parser.add_argument('--quantize', choices=['fp32', 'int8'], default='int8',
                        help='Quantization type (default: int8 for 2.7B)')
    args = parser.parse_args()

    qt = QUANT_INT8 if args.quantize == 'int8' else QUANT_FP32

    if os.path.exists(args.checkpoint):
        export_checkpoint(args.checkpoint, args.output, quant_type=qt)

        import hashlib
        with open(args.output, 'rb') as fh:
            model_hash = hashlib.sha256(fh.read()).hexdigest()
        print(f"  Model SHA-256: {model_hash}")
    else:
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
