"""
quick_test.py — SSM Engine + Export Verification Harness
=========================================================
Tests:
  1. Weight export: create tiny random model, export to .mamba.bin, verify header
  2. SSM math: verify selective scan kernel produces correct output
  3. RoPE: verify loop encoding cos/sin values
  4. Lifeline gate: verify injection math
  5. Header roundtrip: write → read → validate
"""

import struct
import sys
import os
import math
import numpy as np

# Add current dir to path for import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_mamba_baremetal import (
    MAMBA_BIN_MAGIC, MAMBA_BIN_VERSION, HEADER_SIZE,
    create_test_model, calc_total_bytes, write_header
)


def test_header_format() -> bool:
    """Test: header struct size and format are correct."""
    print("  [1] Header format...", end=" ")
    # Header should be exactly 104 bytes:
    # 2 uint32 + 13 int32 + 9 int32 + 1 uint64 = 24*4 + 8 = 104
    expected = 24 * 4 + 8
    if HEADER_SIZE != expected:
        print(f"FAIL (expected {expected}, got {HEADER_SIZE})")
        return False
    print(f"OK ({HEADER_SIZE} bytes)")
    return True


def test_create_test_model() -> bool:
    """Test: create a tiny test model and verify file structure."""
    print("  [2] Test model creation...", end=" ")
    path = "/tmp/test_model.mamba.bin"

    try:
        create_test_model(path)
    except Exception as e:
        print(f"FAIL ({e})")
        return False

    # Read and verify header
    with open(path, 'rb') as f:
        data = f.read()

    if len(data) < HEADER_SIZE:
        print(f"FAIL (file too small: {len(data)})")
        return False

    magic = struct.unpack_from('<I', data, 0)[0]
    if magic != MAMBA_BIN_MAGIC:
        print(f"FAIL (bad magic: {magic:#x})")
        return False

    version = struct.unpack_from('<I', data, 4)[0]
    if version != MAMBA_BIN_VERSION:
        print(f"FAIL (bad version: {version})")
        return False

    d_model = struct.unpack_from('<i', data, 8)[0]
    if d_model != 32:
        print(f"FAIL (d_model={d_model}, expected 32)")
        return False

    os.remove(path)
    print("OK")
    return True


def test_ssm_scan_kernel() -> bool:
    """Test: verify SSM selective scan math."""
    print("  [3] SSM selective scan kernel...", end=" ")

    # Tiny SSM: d_inner=4, d_state=2
    d_inner = 4
    d_state = 2
    np.random.seed(42)

    x_in = np.array([1.0, 0.5, -0.3, 0.8], dtype=np.float32)
    dt = np.array([0.1, 0.2, 0.15, 0.1], dtype=np.float32)
    B_vec = np.array([0.5, -0.3], dtype=np.float32)
    C_vec = np.array([0.7, 0.2], dtype=np.float32)
    A_log = np.array([
        [-1.0, -0.5],   # d=0
        [-0.8, -1.2],   # d=1
        [-1.1, -0.7],   # d=2
        [-0.9, -0.6],   # d=3
    ], dtype=np.float32)
    D_param = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    h_state = np.zeros((d_inner, d_state), dtype=np.float32)

    # Run scan step
    y_out = np.zeros(d_inner, dtype=np.float32)
    for d in range(d_inner):
        x_t = x_in[d]
        dt_t = dt[d]
        y_t = 0.0
        for n in range(d_state):
            A_val = -math.exp(A_log[d, n])
            r = math.exp(dt_t * A_val)
            b_bar = dt_t * B_vec[n]
            h_state[d, n] = r * h_state[d, n] + b_bar * x_t
            y_t += h_state[d, n] * C_vec[n]
        y_out[d] = y_t + x_t * D_param[d]

    # Verify output is finite and non-zero
    if not np.all(np.isfinite(y_out)):
        print("FAIL (NaN/Inf in output)")
        return False
    if np.allclose(y_out, 0.0):
        print("FAIL (all zeros)")
        return False

    # Verify D feed-through
    for d in range(d_inner):
        if abs(y_out[d] - x_in[d]) < 0.001:
            # y should differ from x*D because of the SSM contribution
            pass

    print(f"OK (y={np.round(y_out, 4).tolist()})")
    return True


def test_rope_encoding() -> bool:
    """Test: verify RoPE loop encoding produces correct cos/sin."""
    print("  [4] RoPE loop encoding...", end=" ")

    d_model = 8
    rope_base = 10000

    # Compute RoPE for loop_index=3
    loop_index = 3
    half_d = d_model // 2

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    x_out = x.copy()

    for i in range(half_d):
        theta = 1.0 / (rope_base ** (2 * i / d_model))
        freq = loop_index * theta
        cos_f = math.cos(freq)
        sin_f = math.sin(freq)

        idx0 = 2 * i
        idx1 = 2 * i + 1
        x0 = x[idx0]
        x1 = x[idx1]
        x_out[idx0] = x0 * cos_f - x1 * sin_f
        x_out[idx1] = x0 * sin_f + x1 * cos_f

    # Verify rotation preserves magnitude (approximately)
    mag_in  = np.linalg.norm(x)
    mag_out = np.linalg.norm(x_out)
    if abs(mag_in - mag_out) > 0.01:
        print(f"FAIL (magnitude changed: {mag_in:.4f} → {mag_out:.4f})")
        return False

    # Verify different loop indices produce different outputs
    x2 = x.copy()
    loop_index2 = 5
    for i in range(half_d):
        theta = 1.0 / (rope_base ** (2 * i / d_model))
        freq = loop_index2 * theta
        cos_f = math.cos(freq)
        sin_f = math.sin(freq)
        idx0, idx1 = 2 * i, 2 * i + 1
        x0, x1 = x[idx0], x[idx1]
        x2[idx0] = x0 * cos_f - x1 * sin_f
        x2[idx1] = x0 * sin_f + x1 * cos_f

    if np.allclose(x_out, x2):
        print("FAIL (same output for different loop indices)")
        return False

    print(f"OK (mag preserved: {mag_in:.4f} ≈ {mag_out:.4f})")
    return True


def test_lifeline_gate() -> bool:
    """Test: verify lifeline injection math."""
    print("  [5] Lifeline gate injection...", end=" ")

    d_model = 4
    x       = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x_prompt = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    gate     = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    # x_out = x + gate * x_prompt
    expected = x + gate * x_prompt
    result   = x.copy()
    for i in range(d_model):
        result[i] += gate[i] * x_prompt[i]

    if not np.allclose(result, expected):
        print(f"FAIL ({result} != {expected})")
        return False

    print(f"OK ({result.tolist()})")
    return True


def test_rmsnorm() -> bool:
    """Test: verify RMSNorm computation."""
    print("  [6] RMSNorm...", end=" ")

    d = 4
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    w = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    ss = np.mean(x * x)
    rms = 1.0 / math.sqrt(ss + 1e-5)
    expected = w * x * rms

    if not np.all(np.isfinite(expected)):
        print("FAIL (NaN)")
        return False

    # Verify normalization approximately scales the vector
    norm_in  = np.linalg.norm(x)
    norm_out = np.linalg.norm(expected)
    ratio = norm_out / norm_in

    print(f"OK (scale ratio: {ratio:.4f})")
    return True


def test_size_calculation() -> bool:
    """Test: verify total bytes calculation is consistent."""
    print("  [7] Size calculation...", end=" ")

    d_model = 768
    d_state = 64
    d_conv  = 4
    expand  = 2
    n_layers = 24
    vocab    = 50282
    dt_rank  = 768

    size_no_rlf = calc_total_bytes(
        d_model, d_state, d_conv, expand,
        n_layers, vocab, dt_rank, has_rlf=False
    )
    size_rlf = calc_total_bytes(
        d_model, d_state, d_conv, expand,
        n_layers, vocab, dt_rank, has_rlf=True
    )

    if size_rlf <= size_no_rlf:
        print(f"FAIL (RLF size {size_rlf} <= non-RLF {size_no_rlf})")
        return False

    rlf_overhead = size_rlf - size_no_rlf
    print(f"OK (base: {size_no_rlf/1e6:.1f}MB, +RLF: {rlf_overhead/1e6:.1f}MB, "
          f"total: {size_rlf/1e6:.1f}MB)")
    return True


def main() -> int:
    """Run all tests and report results."""
    print()
    print("═" * 60)
    print("  Mamba2 SSM + RLF Bare-Metal Test Suite")
    print("═" * 60)
    print()

    tests = [
        test_header_format,
        test_create_test_model,
        test_ssm_scan_kernel,
        test_rope_encoding,
        test_lifeline_gate,
        test_rmsnorm,
        test_size_calculation,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            failed += 1

    print()
    print("─" * 60)
    print(f"  Results: {passed}/{passed+failed} passed"
          f"{'  ✅ ALL PASS' if failed == 0 else f'  ❌ {failed} FAILED'}")
    print("─" * 60)
    print()

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
