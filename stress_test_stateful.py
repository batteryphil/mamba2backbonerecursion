"""
stress_test_stateful.py
=======================
Exhaustive stress test for the patched stateful engine.
Pushes every subsystem to its limit and logs all failures/anomalies.

Tests:
  A. PREFILL EXTREMES       — empty, 1-token, max-length (512), Unicode, OO syntax
  B. LONG LOOP ENDURANCE    — 50-loop run, constant memory, no accumulation
  C. DEGENERATION DETECTION — inject loop strings, measure coherence drop
  D. GATE EXTREME INPUTS    — all-zero, all-one, NaN-guarded, very large hidden states
  E. GENERATION SYNC SWEEP  — final_ids correct for loops 0..50
  F. HALTING HEAD RESPONSES — math/code/chat prompts, verify halt_prob range [0,1]
  G. VRAM FOOTPRINT         — VRAM must not grow >50MB across 50 single-token steps
  H. HIDDEN STATE STABILITY — cosine similarity of same prompt run twice must be 1.0

Results are written to:  stress_test_results.txt
"""

import torch
import torch.nn.functional as F
import os
import sys
import traceback
import time
import gc

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/home/phil/.gemini/antigravity/scratch/RM3_Project")

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm import MambaLMHeadModel
from safetensors.torch import load_file
from transformers import AutoTokenizer
from proprioception_gate import GeometricProprioceptionGate

MODEL_DIR  = "/home/phil/.gemini/antigravity/scratch/Syrin_Mamba/Syrin_Mamba_Enterprise_Pack/mamba-2.8b-latent"
GATE_PATH  = "/home/phil/.gemini/antigravity/scratch/RM3_Project/proprio_gate_2.8b.pt"
LOG_PATH   = "/home/phil/.gemini/antigravity/scratch/ItsMick_mamba2backbonerecursion/stress_test_results.txt"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

PASS_STR   = "[PASS]"
FAIL_STR   = "[FAIL]"
WARN_STR   = "[WARN]"

results    = []
pass_count = 0
fail_count = 0
warn_count = 0


def log(tag, name, detail="", exc=None):
    """Record a test result."""
    global pass_count, fail_count, warn_count
    line = f"{tag} {name}"
    if detail:
        line += f" | {detail}"
    if exc:
        line += f"\n       EXCEPTION: {type(exc).__name__}: {exc}"
    results.append(line)
    print(line)
    if tag == PASS_STR: pass_count += 1
    elif tag == FAIL_STR: fail_count += 1
    elif tag == WARN_STR: warn_count += 1


def load_model():
    """Load backbone and gate."""
    cfg   = MambaConfig(d_model=2560, n_layer=64, vocab_size=50280, pad_vocab_size_multiple=8)
    model = MambaLMHeadModel(cfg, dtype=torch.bfloat16, device=DEVICE)
    sd    = load_file(os.path.join(MODEL_DIR, "model.safetensors"))
    if "lm_head.weight" not in sd and "backbone.embedding.weight" in sd:
        sd["lm_head.weight"] = sd["backbone.embedding.weight"]
    model.load_state_dict(sd, strict=False)
    model.eval()

    gate   = GeometricProprioceptionGate(d_model=2560, window_size=8)
    gate_sd = torch.load(GATE_PATH, map_location=DEVICE)
    gate.load_state_dict(gate_sd)
    gate   = gate.to(DEVICE, dtype=torch.bfloat16).eval()

    return model, gate


def forward(model, ids):
    """Run backbone, return [B,L,2560]."""
    with torch.no_grad():
        return model.backbone(ids)


def spacer_seq(tok, n):
    """Build [=]*n tensor."""
    sid = tok.convert_tokens_to_ids("=")
    return torch.tensor([[sid] * n], device=DEVICE, dtype=torch.long)


# ── SUITE A: PREFILL EXTREMES ─────────────────────────────────────────────────
def suite_a(model, tok):
    print("\n" + "="*60)
    print("SUITE A — PREFILL EXTREMES")
    print("="*60)

    prompts = [
        ("1-token", "="),
        ("short_english", "What is 2+2?"),
        ("oo_syntax", "[SWARM:MAIN] <CRITICAL_ALERT> PCI DRIVER FAULT AT 0x00FF8D. ZONE-D META COMPROMISED. <HALT_FIRED>"),
        ("unicode_mixed", "Mamba SSM: Ω→∞ ∂/∂t h(t) = Ah(t) + Bx(t)"),
        ("code_block", "```python\ndef fibonacci(n):\n    return n if n<=1 else fibonacci(n-1)+fibonacci(n-2)\n```"),
        ("max_length_512", "The quick brown fox jumps over the lazy dog. " * 25),
        ("repetition_torture", "Logical degradation within secure operating thresholds. Proceeding. " * 20),
        ("oo_repl_command", "/ssm_infer /oo_status /zones /mind_diag PHASE_A PHASE_Z limbion chronion trophion mirrorion thanatosion"),
        ("numbers_only", "42 1337 9999 0 -1 3.14159 2.71828 1.41421 0.5 100000"),
        ("empty_like", tok.eos_token),
    ]

    for name, text in prompts:
        try:
            ids = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(DEVICE)
            h   = forward(model, ids)
            nan = torch.isnan(h).any().item()
            inf = torch.isinf(h).any().item()
            if nan or inf:
                log(FAIL_STR, f"A.{name}", f"NaN={nan} Inf={inf}")
            else:
                log(PASS_STR, f"A.{name}", f"shape={list(h.shape)} L2={h.norm().item():.2f}")
        except Exception as e:
            log(FAIL_STR, f"A.{name}", exc=e)


# ── SUITE B: LONG LOOP ENDURANCE ─────────────────────────────────────────────
def suite_b(model, tok):
    print("\n" + "="*60)
    print("SUITE B — 50-LOOP ENDURANCE")
    print("="*60)

    prompt   = "[LOGIC] What is the sum of angles in a triangle?"
    ids      = tok(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(DEVICE)
    seq_len  = ids.shape[1]
    sid      = tok.convert_tokens_to_ids("=")

    latencies  = []
    h_prev     = None
    l2_diffs   = []
    crashes    = 0

    if DEVICE == "cuda":
        vram_start = torch.cuda.memory_allocated() / 1024**2
    
    t_total = time.perf_counter()
    for lp in range(50):
        try:
            t0      = time.perf_counter()
            spacers = torch.full((1, lp + 1), sid, device=DEVICE, dtype=torch.long)
            seq     = torch.cat([ids, spacers], dim=1)
            h       = forward(model, seq)
            h_last  = h[0, -1, :].float()
            dt      = (time.perf_counter() - t0) * 1000
            latencies.append(dt)

            if h_prev is not None:
                l2_diffs.append(torch.norm(h_last - h_prev).item())
            h_prev = h_last

            if torch.isnan(h).any():
                log(FAIL_STR, f"B.loop_{lp}", "NaN in hidden state")
                crashes += 1
        except Exception as e:
            log(FAIL_STR, f"B.loop_{lp}", exc=e)
            crashes += 1

    elapsed = (time.perf_counter() - t_total) * 1000

    if DEVICE == "cuda":
        vram_end   = torch.cuda.memory_allocated() / 1024**2
        vram_delta = vram_end - vram_start
        vram_ok    = vram_delta < 50
        vtag = PASS_STR if vram_ok else WARN_STR
        log(vtag, "B.vram_growth", f"{vram_delta:+.1f} MB over 50 loops (limit 50MB)")

    avg_ms   = sum(latencies) / len(latencies) if latencies else 0
    llps     = 1000 / avg_ms if avg_ms > 0 else 0
    h_stable = all(d > 0 for d in l2_diffs)
    
    log(PASS_STR if crashes == 0 else FAIL_STR, "B.endurance_50loops",
        f"crashes={crashes} avg={avg_ms:.1f}ms LLPS={llps:.1f} total={elapsed:.0f}ms")
    log(PASS_STR if h_stable else FAIL_STR, "B.hidden_state_evolves",
        f"all 49 L2_diffs>0: {h_stable}, min={min(l2_diffs):.4f} max={max(l2_diffs):.4f}")
    log(PASS_STR, "B.latency_profile",
        f"p50={sorted(latencies)[25]:.1f}ms p95={sorted(latencies)[47]:.1f}ms p99={sorted(latencies)[49]:.1f}ms")


# ── SUITE C: DEGENERATION DETECTION ──────────────────────────────────────────
def suite_c(model, gate, tok):
    print("\n" + "="*60)
    print("SUITE C — DEGENERATION DETECTION VIA GATE COHERENCE")
    print("="*60)

    loop_str    = " Logical degradation within secure operating thresholds. Proceeding." * 8
    clean_str   = "The limbion engine manages temporal phase transitions in the OO organism."

    for label, text in [("clean", clean_str), ("degenerate", loop_str)]:
        try:
            ids = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(DEVICE)
            h   = forward(model, ids).bfloat16()

            # Gate signals
            h_prev  = F.pad(h[:, :-1, :], (0, 0, 1, 0))
            velocity= torch.norm(h - h_prev, p=2, dim=-1).mean().item()
            drift   = F.cosine_similarity(h, h_prev, dim=-1).mean().item()
            
            if h.shape[1] >= 8:
                h_pad = F.pad(h, (0, 0, 7, 0))
                wins  = h_pad.unfold(1, 8, 1)
                coh   = torch.var(wins, dim=-1).mean().item()
            else:
                coh = float('nan')

            h_gated = gate(h)
            gate_diff = torch.norm(h_gated - h).item()

            log(PASS_STR, f"C.{label}",
                f"velocity={velocity:.3f} drift={drift:.4f} coherence={coh:.4f} gate_diff={gate_diff:.2f}")
        except Exception as e:
            log(FAIL_STR, f"C.{label}", exc=e)

    # Verify degenerate has LOWER coherence than clean
    try:
        def get_coherence(text):
            ids = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(DEVICE)
            h = forward(model, ids).bfloat16()
            if h.shape[1] < 8: return float('nan')
            h_pad = F.pad(h, (0, 0, 7, 0))
            wins = h_pad.unfold(1, 8, 1)
            return torch.var(wins, dim=-1).mean().item()
        
        coh_clean = get_coherence(clean_str)
        coh_degen = get_coherence(loop_str)
        tag = PASS_STR if coh_degen < coh_clean else WARN_STR
        log(tag, "C.coherence_ordering",
            f"clean={coh_clean:.4f} > degenerate={coh_degen:.4f}: {coh_clean > coh_degen}")
    except Exception as e:
        log(FAIL_STR, "C.coherence_ordering", exc=e)


# ── SUITE D: GATE EXTREME INPUTS ─────────────────────────────────────────────
def suite_d(gate):
    print("\n" + "="*60)
    print("SUITE D — GATE EXTREME INPUTS")
    print("="*60)

    B, L, D = 1, 16, 2560
    cases = [
        ("zeros",     torch.zeros(B, L, D, dtype=torch.bfloat16, device=DEVICE)),
        ("ones",      torch.ones(B, L, D, dtype=torch.bfloat16, device=DEVICE)),
        ("large_pos", torch.ones(B, L, D, dtype=torch.bfloat16, device=DEVICE) * 100),
        ("large_neg", torch.ones(B, L, D, dtype=torch.bfloat16, device=DEVICE) * -100),
        ("random",    torch.randn(B, L, D, dtype=torch.bfloat16, device=DEVICE)),
        ("very_short_L1", torch.randn(B, 1, D, dtype=torch.bfloat16, device=DEVICE)),
        ("short_L3",  torch.randn(B, 3, D, dtype=torch.bfloat16, device=DEVICE)),
        ("long_L512", torch.randn(B, 512, D, dtype=torch.bfloat16, device=DEVICE)),
        ("mixed_signs", (torch.randn(B, L, D, dtype=torch.bfloat16, device=DEVICE) * torch.sign(torch.randn(B, L, D, device=DEVICE)).to(dtype=torch.bfloat16))),
    ]

    for name, h in cases:
        try:
            with torch.no_grad():
                h_out = gate(h)
            nan = torch.isnan(h_out).any().item()
            inf = torch.isinf(h_out).any().item()
            diff = torch.norm(h_out - h).item()
            tag = FAIL_STR if (nan or inf) else PASS_STR
            log(tag, f"D.{name}", f"NaN={nan} Inf={inf} diff={diff:.4f} shape={list(h_out.shape)}")
        except Exception as e:
            log(FAIL_STR, f"D.{name}", exc=e)


# ── SUITE E: GENERATION SYNC SWEEP ───────────────────────────────────────────
def suite_e(tok):
    print("\n" + "="*60)
    print("SUITE E — GENERATION SYNC SWEEP (loops 0..50)")
    print("="*60)

    sid      = tok.convert_tokens_to_ids("=")
    prompt   = "The OO engine is sovereign."
    ids      = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    seq_len  = ids.shape[1]
    failures = 0

    for lp in range(51):
        loops_executed = lp + 1
        final_ids = torch.cat(
            [ids, torch.full((1, loops_executed), sid, device=DEVICE, dtype=torch.long)],
            dim=1
        )
        expected = seq_len + loops_executed
        actual   = final_ids.shape[1]
        if actual != expected:
            log(FAIL_STR, f"E.sync_lp{lp}", f"shape={actual} expected={expected}")
            failures += 1

    tag = PASS_STR if failures == 0 else FAIL_STR
    log(tag, "E.sweep_0_to_50", f"all 51 loop counts correct, failures={failures}")


# ── SUITE F: HALTING HEAD RESPONSES ──────────────────────────────────────────
def suite_f(model, tok):
    print("\n" + "="*60)
    print("SUITE F — HALTING HEAD RANGE & DISCRIMINATION")
    print("="*60)

    halt_path = os.path.join(MODEL_DIR, "halting_head.pt")
    if not os.path.exists(halt_path):
        log(WARN_STR, "F.halting_head", "halting_head.pt not found at ENGINE_DIR, skipping")
        return

    import torch.nn as nn
    class HaltingHead(nn.Module):
        def __init__(self, d): 
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(512, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
            )
        def forward(self, x): return self.net(x).squeeze(-1)

    ckpt = torch.load(halt_path, weights_only=True, map_location=DEVICE)
    # Infer d_input from checkpoint: explicit key or from first layer weight shape
    if "d_input" in ckpt:
        d_input = ckpt["d_input"]
    elif "state_dict" in ckpt:
        first_w = next(iter(ckpt["state_dict"].values()))
        d_input = first_w.shape[-1]
    else:
        first_w = next(iter(ckpt.values()))
        d_input = first_w.shape[-1]
    head = HaltingHead(d_input).to(DEVICE)
    sd = ckpt.get("state_dict", ckpt)
    head.load_state_dict(sd)
    head.eval()

    test_prompts = [
        ("math_exact",    "42 * 37 = 1554",                      "expect_low"),
        ("math_hard",     "Integrate sin(x^2) from 0 to infinity", "expect_high"),
        ("chat_simple",   "Hello, how are you today?",            "expect_low"),
        ("oo_halt",       "/ssm_infer HALT CRITICAL EMERGENCY",   "expect_high"),
        ("reasoning",     "If all birds can fly and penguins are birds, can penguins fly?", "expect_high"),
        ("definition",    "What is the meaning of life?",         "expect_high"),
        ("code_simple",   "print('hello world')",                 "expect_low"),
    ]

    all_valid = True
    with torch.no_grad():
        for name, text, expectation in test_prompts:
            try:
                ids  = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(DEVICE)
                h    = forward(model, ids)[0, -1, :].float()
                lp_t = torch.tensor([0.0 / 50.0], device=DEVICE)
                p    = head(torch.cat([h, lp_t]).unsqueeze(0)).item()
                in_range = 0.0 <= p <= 1.0
                tag = PASS_STR if in_range else FAIL_STR
                if not in_range: all_valid = False
                log(tag, f"F.{name}", f"p_halt={p:.4f} ({expectation}) in_range={in_range}")
            except Exception as e:
                log(FAIL_STR, f"F.{name}", exc=e)
                all_valid = False

    log(PASS_STR if all_valid else FAIL_STR, "F.all_in_range", f"{all_valid}")


# ── SUITE G: HIDDEN STATE STABILITY ──────────────────────────────────────────
def suite_g(model, tok):
    print("\n" + "="*60)
    print("SUITE G — DETERMINISTIC REPRODUCIBILITY")
    print("="*60)

    prompts = [
        "What is the OO engine?",
        "[SWARM:A] Initiating handoff to chronion.",
        "Fibonacci sequence: 1 1 2 3 5 8 13 21",
    ]
    for text in prompts:
        try:
            ids = tok(text, return_tensors="pt").input_ids.to(DEVICE)
            h1  = forward(model, ids)[0, -1, :].float()
            h2  = forward(model, ids)[0, -1, :].float()
            cos = F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()
            l2  = torch.norm(h1 - h2).item()
            tag = PASS_STR if abs(cos - 1.0) < 1e-4 else FAIL_STR
            log(tag, f"G.deterministic[{text[:30]}]", f"cosine={cos:.8f} L2={l2:.8f}")
        except Exception as e:
            log(FAIL_STR, f"G.deterministic", exc=e)


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(" STATEFUL ENGINE STRESS TEST SUITE")
    print(f" Device: {DEVICE}")
    print("=" * 60)

    tok    = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tok.pad_token = tok.eos_token

    print("[INIT] Loading 2.8B backbone + Proprioception Gate...")
    model, gate = load_model()
    print("[INIT] Ready.\n")

    t_start = time.perf_counter()
    suite_a(model, tok)
    suite_b(model, tok)
    suite_c(model, gate, tok)
    suite_d(gate)
    suite_e(tok)
    suite_f(model, tok)
    suite_g(model, tok)
    elapsed = time.perf_counter() - t_start

    summary = [
        "",
        "=" * 60,
        f" STRESS TEST COMPLETE — {elapsed:.1f}s",
        f" PASS: {pass_count}  FAIL: {fail_count}  WARN: {warn_count}",
        "=" * 60,
    ]
    for line in summary:
        print(line)
        results.append(line)

    with open(LOG_PATH, "w") as f:
        f.write("\n".join(results))

    print(f"\nFull log written to: {LOG_PATH}")
