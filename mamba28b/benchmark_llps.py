"""
benchmark_llps.py — Latent Loops Per Second (LLPS) Benchmark
=============================================================
Phase 3: Measure loop latency: stateful cache vs original re-tokenize.

Usage:
    python benchmark_llps.py [engine_dir] [--runs N] [--loops N]
"""

import torch
import time
import sys
import os
import statistics
from transformers import AutoTokenizer, AutoModelForCausalLM


def benchmark_original(model, tok, prompt, max_loops, device, runs=20):
    """Benchmark original re-tokenize approach."""
    all_loop_times = []

    for run in range(runs):
        with torch.no_grad():
            for lp in range(max_loops):
                text = prompt + "=" * lp
                toks = tok(text, return_tensors="pt",
                           truncation=True, max_length=256)
                input_ids = toks.input_ids.to(device)

                t0 = time.perf_counter()
                out = model(input_ids=input_ids, output_hidden_states=True)
                _ = out.hidden_states[-1][0, -1, :]
                elapsed = (time.perf_counter() - t0) * 1000
                all_loop_times.append(elapsed)

    return all_loop_times


def benchmark_stateful(model, tok, prompt, spacer_id, max_loops, device, runs=20):
    """Benchmark stateful cache approach."""
    all_loop_times = []

    for run in range(runs):
        with torch.no_grad():
            # Prefill (not counted in loop latency)
            toks = tok(prompt, return_tensors="pt",
                       truncation=True, max_length=256)
            input_ids = toks.input_ids.to(device)
            seq_len = input_ids.shape[1]

            out = model(input_ids=input_ids, use_cache=True,
                        output_hidden_states=True)
            cache = out.cache_params

            # Measure loop iterations only
            spacer = torch.tensor([[spacer_id]], device=device)
            for lp in range(max_loops):
                cache_pos = torch.tensor([seq_len + lp], device=device)

                t0 = time.perf_counter()
                step_out = model(
                    input_ids=spacer,
                    cache_params=cache,
                    cache_position=cache_pos,
                    use_cache=True,
                    output_hidden_states=True
                )
                _ = step_out.hidden_states[-1][0, -1, :]
                elapsed = (time.perf_counter() - t0) * 1000
                all_loop_times.append(elapsed)

    return all_loop_times


def compute_stats(times):
    """Compute benchmark statistics."""
    times_sorted = sorted(times)
    n = len(times_sorted)
    return {
        "avg": statistics.mean(times),
        "median": statistics.median(times),
        "p95": times_sorted[int(n * 0.95)] if n > 20 else times_sorted[-1],
        "p99": times_sorted[int(n * 0.99)] if n > 100 else times_sorted[-1],
        "min": min(times),
        "max": max(times),
        "stdev": statistics.stdev(times) if n > 1 else 0,
        "n": n,
    }


def main():
    engine_dir = "checkpoints/mamba-2.8b-latent"
    runs = 20
    max_loops = 7

    args = sys.argv[1:]
    skip_next = False
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--runs" and i + 1 < len(args):
            runs = int(args[i + 1])
            skip_next = True
        elif arg == "--loops" and i + 1 < len(args):
            max_loops = int(args[i + 1])
            skip_next = True
        elif not arg.startswith("--"):
            engine_dir = arg

    if not os.path.isdir(engine_dir):
        engine_dir = "state-spaces/mamba-130m-hf"
        print(f"[INFO] Checkpoint not found, using base model: {engine_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INIT] Loading {engine_dir} on {device}...")

    tok = AutoTokenizer.from_pretrained(engine_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        engine_dir,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    model.eval()

    spacer_id = tok.convert_tokens_to_ids("=")
    prompt = "[LOGIC] X=5. Y=X*2. Z=Y+3. W=Z-X. Output W. ===="

    print(f"[CONFIG] runs={runs}, max_loops={max_loops}, device={device}")
    print(f"[CONFIG] prompt tokens: {len(tok(prompt).input_ids)}")

    # Warmup
    print("\n[WARMUP] Running warmup passes...")
    with torch.no_grad():
        toks = tok(prompt, return_tensors="pt").to(device)
        for _ in range(3):
            model(input_ids=toks.input_ids, output_hidden_states=True)

    # Benchmark original
    print(f"\n[BENCH] Original (re-tokenize): {runs} runs x {max_loops} loops...")
    orig_times = benchmark_original(model, tok, prompt, max_loops, device, runs)
    orig_stats = compute_stats(orig_times)

    # Benchmark stateful
    print(f"[BENCH] Stateful (cache step):  {runs} runs x {max_loops} loops...")
    stat_times = benchmark_stateful(model, tok, prompt, spacer_id,
                                     max_loops, device, runs)
    stat_stats = compute_stats(stat_times)

    # Results
    banner = "=" * 70
    print(f"\n{banner}")
    print(f"  LLPS BENCHMARK RESULTS")
    print(f"  Model: {engine_dir}")
    print(f"  Device: {device}")
    print(f"  Runs: {runs}, Loops per run: {max_loops}")
    print(f"{banner}\n")

    print(f"  {'Metric':<20} | {'Original (ms)':<16} | {'Stateful (ms)':<16} | Speedup")
    print(f"  {'-'*20}-+-{'-'*16}-+-{'-'*16}-+--------")
    for metric in ["avg", "median", "p95", "min", "max", "stdev"]:
        o = orig_stats[metric]
        s = stat_stats[metric]
        speedup = o / s if s > 0 else float('inf')
        print(f"  {metric:<20} | {o:>14.2f}  | {s:>14.2f}  | {speedup:>5.2f}x")

    orig_llps = 1000 / orig_stats["avg"] if orig_stats["avg"] > 0 else 0
    stat_llps = 1000 / stat_stats["avg"] if stat_stats["avg"] > 0 else 0
    print(f"\n  Original LLPS:  {orig_llps:>8.1f} loops/sec")
    print(f"  Stateful LLPS:  {stat_llps:>8.1f} loops/sec")
    print(f"  Throughput gain: {stat_llps/orig_llps:.2f}x" if orig_llps > 0 else "")

    print(f"\n  Samples: original={orig_stats['n']}, stateful={stat_stats['n']}")
    print(f"{banner}\n")

    # Write results to file
    results_path = "docs/llps_benchmark.md"
    with open(results_path, "w") as f:
        f.write("# Phase 3: LLPS Benchmark Results\n\n")
        f.write(f"## Environment\n\n")
        f.write(f"- **Model**: {engine_dir}\n")
        f.write(f"- **Device**: {device}\n")
        f.write(f"- **Runs**: {runs}\n")
        f.write(f"- **Loops per run**: {max_loops}\n")
        f.write(f"- **Prompt tokens**: {len(tok(prompt).input_ids)}\n\n")
        f.write(f"## Results\n\n")
        f.write(f"| Approach | Avg Loop ms | Median ms | p95 ms | LLPS | Notes |\n")
        f.write(f"|----------|------------|-----------|--------|------|-------|\n")
        f.write(f"| Original (re-tokenize) | {orig_stats['avg']:.2f} | {orig_stats['median']:.2f} | {orig_stats['p95']:.2f} | {orig_llps:.1f} | Sequence grows each loop |\n")
        f.write(f"| Stateful cache | {stat_stats['avg']:.2f} | {stat_stats['median']:.2f} | {stat_stats['p95']:.2f} | {stat_llps:.1f} | Single-token recurrent step |\n\n")
        f.write(f"**Speedup: {orig_stats['avg']/stat_stats['avg']:.2f}x** (avg latency)\n\n")
        f.write(f"**Throughput gain: {stat_llps/orig_llps:.2f}x** (LLPS)\n\n")
        f.write(f"## Detailed Statistics\n\n")
        f.write(f"| Metric | Original (ms) | Stateful (ms) | Speedup |\n")
        f.write(f"|--------|--------------|---------------|--------|\n")
        for metric in ["avg", "median", "p95", "min", "max", "stdev"]:
            o = orig_stats[metric]
            s = stat_stats[metric]
            sp = o / s if s > 0 else float('inf')
            f.write(f"| {metric} | {o:.2f} | {s:.2f} | {sp:.2f}x |\n")
        f.write(f"| n (samples) | {orig_stats['n']} | {stat_stats['n']} | — |\n")
        f.write(f"\n## Analysis\n\n")
        f.write(f"The stateful approach processes a single token per iteration "
                f"(O(1) per step), while the original re-tokenizes the entire "
                f"prompt + spacers each loop (O(n) per step where n grows).\n\n")
        f.write(f"On GPU with the 2.8B model, the speedup should be significantly "
                f"larger because:\n")
        f.write(f"1. The original approach's cost scales with prompt length "
                f"(more tokens = more compute)\n")
        f.write(f"2. The stateful approach is constant regardless of prompt length\n")
        f.write(f"3. GPU kernel launch overhead is amortized better with "
                f"single-token steps\n")

    print(f"[DONE] Results written to {results_path}")


if __name__ == "__main__":
    main()
