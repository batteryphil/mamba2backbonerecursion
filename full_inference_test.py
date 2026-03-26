"""
full_inference_test.py — Comprehensive Mamba2+RLF Inference Test
================================================================
Standalone test: reconstructs model architecture from checkpoint
without importing the training script (avoids re-training).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT   = "mamba2_130m_v34_rope_best.pt"


def load_model() -> tuple:
    """Load model from checkpoint with standalone architecture reconstruction.

    Returns:
        (model, tokenizer, d_model) tuple
    """
    from transformers import AutoTokenizer
    from mamba_ssm import MambaLMHeadModel, Mamba2

    print(f"\n{'='*60}")
    print(f"  Mamba2-130M + RLF — Full Inference Test")
    print(f"{'='*60}")

    # Tokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": ["<THINK>", "<HALT>"]})
    halt_id = tok.convert_tokens_to_ids("<HALT>")
    print(f"  Tokenizer: {len(tok):,} vocab, <HALT>={halt_id}")

    # Load checkpoint
    print(f"  Loading checkpoint: {CKPT}")
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    sd = ckpt["model_state_dict"]
    step = ckpt.get("step", "?")
    d_model = ckpt.get("d_model", 768)
    print(f"  Step: {step}, d_model: {d_model}")

    # Reconstruct base model
    print(f"  Loading base state-spaces/mamba2-130m...")
    base = MambaLMHeadModel.from_pretrained(
        "state-spaces/mamba2-130m", dtype=torch.bfloat16, device=DEVICE
    )

    # Expand vocab for <THINK>, <HALT>
    new_vocab = len(tok)
    old_vocab = base.backbone.embedding.weight.shape[0]
    if new_vocab > old_vocab:
        ne = nn.Embedding(new_vocab, d_model, dtype=torch.bfloat16)
        nn.init.normal_(ne.weight, std=0.02)
        ne.weight.data[:old_vocab] = base.backbone.embedding.weight.data
        base.backbone.embedding = ne
        nh = nn.Linear(d_model, new_vocab, bias=False, dtype=torch.bfloat16)
        nn.init.normal_(nh.weight, std=0.02)
        nh.weight.data[:old_vocab] = base.lm_head.weight.data
        base.lm_head = nh

    # Build RLF model (lightweight wrapper — just what inference needs)
    class InferenceModel(nn.Module):
        """Minimal RLF inference wrapper."""

        def __init__(self, base_model: MambaLMHeadModel, d: int) -> None:
            """Initialize from base model + RLF components."""
            super().__init__()
            self.backbone = base_model.backbone
            self.lm_head = base_model.lm_head
            self.all_layers = nn.ModuleList(base_model.backbone.layers)
            self.norm = base_model.backbone.norm_f
            self.d_model = d

            # RLF components (will be loaded from checkpoint)
            self.loop_norm = nn.RMSNorm(d).to(torch.bfloat16)
            self.mamba2_core = Mamba2(
                d_model=d, d_state=64, d_conv=4, expand=2,
                headdim=64, chunk_size=64
            ).to(torch.bfloat16)
            self.lifeline_gate = nn.Parameter(torch.ones(d, dtype=torch.float32))

            # RoPE (analytical, no params)
            inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
            self.register_buffer("inv_freq", inv_freq)

        def apply_rope(self, x: torch.Tensor, loop_i: int) -> torch.Tensor:
            """Apply 1D RoPE rotation for loop index."""
            n = torch.tensor(float(loop_i), device=x.device)
            freqs = n * self.inv_freq.to(device=x.device, dtype=torch.float32)
            cos_f = freqs.cos()
            sin_f = freqs.sin()
            cos_v = torch.stack([cos_f, cos_f], dim=-1).flatten()[:self.d_model].to(x.dtype)
            sin_v = torch.stack([sin_f, sin_f], dim=-1).flatten()[:self.d_model].to(x.dtype)
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            rot = torch.stack([-x2, x1], dim=-1).flatten(-2)
            return x * cos_v + rot * sin_v

    model = InferenceModel(base, d_model).to(DEVICE)

    # Load trained weights
    own_sd = model.state_dict()
    loaded = 0
    skipped = []
    for k, v in sd.items():
        if k in own_sd and own_sd[k].shape == v.shape:
            own_sd[k] = v
            loaded += 1
        else:
            skipped.append(k)
    model.load_state_dict(own_sd, strict=False)
    print(f"  Loaded {loaded} tensors, skipped {len(skipped)}")
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {n_params/1e6:.1f}M")

    return model, tok, d_model


@torch.no_grad()
def infer(model: nn.Module, tok: object, prompt: str, max_loops: int = 16,
          verbose: bool = True) -> dict:
    """Run RLF inference with per-loop trace.

    Args:
        model: loaded model
        tok: tokenizer
        prompt: input text
        max_loops: maximum loop iterations
        verbose: print per-loop output

    Returns:
        dict with answer, loops, timing, trace
    """
    ids = tok.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    halt_id = tok.convert_tokens_to_ids("<HALT>")

    t0 = time.perf_counter()

    # Base encoding
    x = model.backbone.embedding(ids)
    residual = None
    for layer in model.all_layers:
        x, residual = layer(x, residual)

    x_prompt = x.clone().detach()
    answer = ""
    trace = []

    for loop_i in range(max_loops):
        # Lifeline
        gate = model.lifeline_gate.to(x.dtype)
        x = x + gate.unsqueeze(0).unsqueeze(0) * x_prompt
        # RoPE
        x = model.apply_rope(x, loop_i)
        # Loop block
        for layer in model.all_layers[6:]:
            x, residual = layer(x, residual)
        x = x + model.mamba2_core(x)
        x = model.loop_norm(x)
        # Predict
        lg = model.lm_head(model.norm(x, residual, prenorm=False))
        p = torch.softmax(lg[0, -1, :].float(), dim=-1)
        tid = p.argmax().item()
        token = tok.decode([tid]).strip()
        prob = p[tid].item()

        trace.append({"loop": loop_i + 1, "token": token, "prob": prob})

        if verbose:
            halt_mark = " <HALT>" if tid == halt_id else ""
            print(f"    L{loop_i+1:2d}  {token!r:16s} p={prob:.4f}{halt_mark}")

        if tid == halt_id:
            break
        answer = token

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "answer": answer,
        "loops": len(trace),
        "elapsed_ms": elapsed_ms,
        "trace": trace,
        "halted": trace[-1]["token"] == "<HALT>" if trace else False,
    }


def run_tests() -> None:
    """Run the complete test suite."""
    model, tok, d_model = load_model()

    # ── Test 1: Reasoning chains ──────────────────────────────────
    print("\n\n── Test 1: Reasoning Chains (1-5 hops) ──")
    chains = [
        ("1-hop", "A = blue. What is A?\nAnswer:", "blue"),
        ("2-hop", "A = red. B = A. What is B?\nAnswer:", "red"),
        ("3-hop", "X = green. Y = X. Z = Y. What is Z?\nAnswer:", "green"),
        ("4-hop", "A = purple. B = A. C = B. D = C. What is D?\nAnswer:", "purple"),
        ("5-hop", "M = orange. N = M. O = N. P = O. Q = P. What is Q?\nAnswer:", "orange"),
    ]

    pass_count = 0
    for label, prompt, expected in chains:
        print(f"\n  [{label}] Expected: '{expected}'")
        r = infer(model, tok, prompt, verbose=True)
        ok = expected.lower() in r["answer"].lower()
        if ok:
            pass_count += 1
        status = "✅" if ok else "❌"
        print(f"  → {status} Answer='{r['answer']}' "
              f"({r['elapsed_ms']:.0f}ms, {r['loops']} loops, halted={r['halted']})")

    print(f"\n  In-distribution score: {pass_count}/{len(chains)}")

    # ── Test 2: OOD length generalization ─────────────────────────
    print("\n\n── Test 2: OOD Length Generalization ──")
    ood_tests = [
        ("6-hop", "A = jazz. B = A. C = B. D = C. E = D. F = E. What is F?\nAnswer:", "jazz"),
        ("8-hop", "P = rocket. Q = P. R = Q. S = R. T = S. U = T. V = U. W = V. What is W?\nAnswer:", "rocket"),
    ]
    for label, prompt, expected in ood_tests:
        print(f"\n  [{label}] Expected: '{expected}'")
        r = infer(model, tok, prompt, verbose=True)
        ok = expected.lower() in r["answer"].lower()
        status = "✅ GENERALIZES" if ok else f"❌ Expected bottleneck"
        print(f"  → {status}: '{r['answer']}'")

    # ── Test 3: Lifeline ablation ─────────────────────────────────
    print("\n\n── Test 3: Lifeline Ablation ──")
    test_prompt = "A = blue. B = A. What is B?\nAnswer:"

    # With lifeline
    r_on = infer(model, tok, test_prompt, verbose=False)

    # Without lifeline
    orig_gate = model.lifeline_gate.data.clone()
    model.lifeline_gate.data.zero_()
    r_off = infer(model, tok, test_prompt, verbose=False)
    model.lifeline_gate.data.copy_(orig_gate)

    print(f"  Lifeline ON:  '{r_on['answer']}' ({r_on['loops']} loops, {r_on['elapsed_ms']:.0f}ms)")
    print(f"  Lifeline OFF: '{r_off['answer']}' ({r_off['loops']} loops, {r_off['elapsed_ms']:.0f}ms)")
    print(f"  Phase transition: {'YES' if r_on['answer'] != r_off['answer'] else 'NO'}")

    # ── Test 4: Gate statistics ───────────────────────────────────
    print("\n\n── Test 4: Lifeline Gate Analysis ──")
    g = model.lifeline_gate.data
    ram = (g > 1.0).sum().item()
    alu = (g <= 1.0).sum().item()
    print(f"  μ={g.mean():.6f}  σ={g.std():.6f}")
    print(f"  min={g.min():.6f}  max={g.max():.6f}")
    print(f"  RAM dims: {ram}/{g.numel()} ({100*ram/g.numel():.1f}%)")
    print(f"  ALU dims: {alu}/{g.numel()} ({100*alu/g.numel():.1f}%)")

    # ── Test 5: Throughput ────────────────────────────────────────
    print("\n\n── Test 5: Throughput ──")
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        infer(model, tok, "A = blue. B = A. What is B?\nAnswer:", max_loops=8, verbose=False)
    elapsed = time.perf_counter() - t0
    print(f"  {n} inferences in {elapsed:.2f}s")
    print(f"  Avg: {elapsed/n*1000:.1f}ms per query")
    print(f"  Throughput: {n/elapsed:.1f} queries/sec")

    # ── Test 6: Memory ────────────────────────────────────────────
    print("\n\n── Test 6: Memory ──")
    vram = torch.cuda.max_memory_allocated() / 1e6 if DEVICE == "cuda" else 0
    print(f"  Peak VRAM: {vram:.0f} MB")

    # ── Test 7: Export artifacts ──────────────────────────────────
    print("\n\n── Test 7: Artifacts ──")
    for f, desc in [
        ("model.mamba.bin", "FP32 export"),
        ("model_int8.mamba.bin", "INT8 export"),
        ("tokenizer.bpe.bin", "BPE tokenizer"),
        ("OOHANDOFF.TXT", "Handoff receipt"),
        ("system2_logic_v2.json", "Train data v2"),
        ("system2_logic_v2_ood.json", "OOD data"),
    ]:
        if os.path.exists(f):
            sz = os.path.getsize(f)
            unit = f"{sz/1e6:.1f}MB" if sz > 1e6 else f"{sz/1e3:.0f}KB"
            print(f"  ✅ {f:30s} {unit:>8s}  ({desc})")
        else:
            print(f"  ❌ {f:30s}          ({desc})")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Reasoning (in-dist): {pass_count}/{len(chains)}")
    print(f"  Lifeline ablation:   {'Confirmed' if r_on['answer'] != r_off['answer'] else 'Same'}")
    print(f"  Gate health:         μ={g.mean():.4f} σ={g.std():.4f}")
    print(f"  Throughput:          {n/elapsed:.1f} q/s ({elapsed/n*1000:.0f}ms avg)")
    print(f"  VRAM:                {vram:.0f} MB")
    if os.path.exists("model.mamba.bin") and os.path.exists("model_int8.mamba.bin"):
        fp32 = os.path.getsize("model.mamba.bin")
        int8 = os.path.getsize("model_int8.mamba.bin")
        print(f"  Compression:         {fp32/1e6:.0f}MB → {int8/1e6:.0f}MB "
              f"({int8/fp32*100:.0f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_tests()
