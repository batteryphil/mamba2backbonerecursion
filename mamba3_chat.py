"""
mamba3_chat.py — Phase 14 Inner-Loop Bypass Inference Engine
=============================================================
Proves the core O(1) memory thesis:
  - Input: natural language math or logic query
  - Compute: HaltingHead autonomously decides how many SSM ticks to run
  - Output: answer + loop count + peak VRAM (invariant regardless of loop count)

Usage:
    # Phase 14 bypass model (recommended):
    python mamba3_chat.py --checkpoint checkpoints/mamba3_p14_bypass_mastered.pt \\
                          --halting_head checkpoints/mamba3_p14_halting_head_mastered.pt

    # Phase 13 fallback (autoregressive dark loops):
    python mamba3_chat.py --checkpoint checkpoints/mamba3_p13_universal_mastered.pt \\
                          --loops 10
"""
import argparse
import time
import torch
import torch.nn as nn
from mamba_ssm import MambaLMHeadModel
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_MODEL = 768
HALT_THRESHOLD = 0.70
MAX_LOOPS = 30
MIN_LOOPS = 1
ROMI_PERIOD = 5


# ─── HaltingHead (must match Phase 14 training definition) ───────────────────
class HaltingHead(nn.Module):
    """Lightweight binary classifier: P(halt | SSM hidden state)."""

    def __init__(self, d_model: int) -> None:
        """Initialize with 2-layer MLP."""
        super().__init__()
        self.probe = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Return P(halt) as a scalar per batch item."""
        return self.probe(hidden_state.mean(dim=1)).squeeze(-1)


# ─── Model Loading ────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str) -> MambaLMHeadModel:
    """Load Mamba-3 from checkpoint."""
    print(f"[INIT] Loading base architecture: state-spaces/mamba-130m ...")
    model = MambaLMHeadModel.from_pretrained(
        "state-spaces/mamba-130m", device=DEVICE, dtype=torch.bfloat16
    )
    print(f"[INIT] Injecting checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model


def load_halting_head(halting_head_path: str) -> HaltingHead:
    """Load the trained HaltingHead binary classifier."""
    head = HaltingHead(d_model=D_MODEL).to(DEVICE).to(torch.bfloat16)
    head.load_state_dict(torch.load(halting_head_path, map_location=DEVICE))
    head.eval()
    return head


# ─── Phase 14 Inner-Loop Inference ───────────────────────────────────────────
def inner_loop_generate(
    model: MambaLMHeadModel,
    halting_head: HaltingHead,
    tokenizer,
    problem: str,
    max_new_tokens: int = 60,
    temperature: float = 0.3,
) -> tuple[str, int, float, float]:
    """
    Phase 14 inference: HaltingHead-steered inner-loop bypass.

    The LM Head fires ONCE at the end — not per tick. This makes
    inference latency proportional to ticks × SSM layer compute,
    while VRAM stays flat regardless of loop count (O(1) memory).

    Returns:
        answer_text: Decoded model output
        n_loops: Number of SSM ticks HaltingHead decided to run
        peak_vram_gb: Peak GPU memory during inference
        latency_ms: Wall-clock time for the full inference pass
    """
    prefix = "[LOGIC] " if any(
        kw in problem.lower() for kw in ["what", "how", "many", "total", "sum", "if"]
    ) else "[CHAT] "
    prompt_text = f"{prefix}{problem}\nSolution: "

    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(DEVICE)

    torch.cuda.reset_peak_memory_stats(DEVICE) if DEVICE == "cuda" else None
    t_start = time.perf_counter()

    with torch.no_grad():
        # Compute frozen ROM embedding for re-injection lifeline
        hidden_states = model.backbone.embedding(input_ids)
        rom_embedding = hidden_states.clone()
        residual = None

        # Initial full forward pass
        for layer in model.backbone.layers:
            hidden_states, residual = layer(hidden_states, residual=residual)

        n_loops = 0
        halt_log = []

        # Inner loop — HaltingHead steers, LM Head is dormant
        while n_loops < MAX_LOOPS:
            n_loops += 1

            # ROM Re-injection to prevent bfloat16 washout
            if n_loops % ROMI_PERIOD == 0:
                rom_pooled = rom_embedding.mean(dim=1, keepdim=True)
                hidden_states = hidden_states + rom_pooled.to(hidden_states.dtype)

            for layer in model.backbone.layers:
                hidden_states, residual = layer(hidden_states, residual=residual)

            p_halt = halting_head(hidden_states).mean().item()
            halt_log.append(round(p_halt, 3))

            if p_halt > HALT_THRESHOLD and n_loops >= MIN_LOOPS:
                break

        # LM Head fires exactly once
        if residual is not None:
            final_hidden = model.backbone.norm_f(hidden_states + residual)
        else:
            final_hidden = model.backbone.norm_f(hidden_states)

        logits = model.lm_head(final_hidden.to(torch.bfloat16))

        # Greedy decode with temperature
        next_ids = []
        for _ in range(max_new_tokens):
            scaled = logits[0, -1, :] / max(temperature, 1e-6)
            probs = torch.softmax(scaled, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            if next_tok.item() == tokenizer.eos_token_id:
                break
            next_ids.append(next_tok.item())
            # Feed token back in for next position
            tok_embed = model.backbone.embedding(next_tok.unsqueeze(0))
            for layer in model.backbone.layers:
                tok_embed, _ = layer(tok_embed, residual=None)
            if residual is not None:
                logits = model.lm_head(model.backbone.norm_f(tok_embed).to(torch.bfloat16))
            else:
                logits = model.lm_head(model.backbone.norm_f(tok_embed).to(torch.bfloat16))

    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000

    peak_vram = (
        torch.cuda.max_memory_allocated(DEVICE) / 1e9
        if DEVICE == "cuda" else 0.0
    )

    answer = tokenizer.decode(next_ids, skip_special_tokens=True)
    return answer, n_loops, peak_vram, latency_ms


# ─── Phase 13 Fallback (autoregressive dark loops) ───────────────────────────
def autoregressive_generate(
    model: MambaLMHeadModel,
    tokenizer,
    problem: str,
    n_loops: int = 10,
    max_new_tokens: int = 60,
    temperature: float = 0.3,
) -> tuple[str, int, float, float]:
    """Legacy Phase 13 inference with fixed autoregressive dark loops."""
    prefix = "[LOGIC] " if any(
        kw in problem.lower() for kw in ["what", "how", "many", "total", "sum", "if"]
    ) else "[CHAT] "
    prompt = f"{prefix}{problem}\nSolution: " + "=" * n_loops

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE) if DEVICE == "cuda" else None
    t_start = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=input_ids.shape[1] + max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
        )

    t_end = time.perf_counter()
    peak_vram = torch.cuda.max_memory_allocated(DEVICE) / 1e9 if DEVICE == "cuda" else 0.0
    answer = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    return answer, n_loops, peak_vram, (t_end - t_start) * 1000


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    """Interactive inference loop with telemetry display."""
    parser = argparse.ArgumentParser(description="Mamba-3 Latent Reasoning Inference Engine")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/mamba3_p13_universal_mastered.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--halting_head",
        type=str,
        default=None,
        help="Path to HaltingHead weights (Phase 14 only). Omit to use Phase 13 autoregressive mode."
    )
    parser.add_argument("--loops", type=int, default=10, help="Dark loops for Phase 13 fallback mode")
    parser.add_argument("--tokens", type=int, default=60, help="Max new tokens to generate")
    parser.add_argument("--temp", type=float, default=0.3, help="Sampling temperature")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.checkpoint)

    use_phase14 = args.halting_head is not None
    halting_head = None
    if use_phase14:
        print(f"[INIT] Loading HaltingHead: {args.halting_head}")
        halting_head = load_halting_head(args.halting_head)
        mode_str = "Phase 14 — Inner-Loop Bypass (HaltingHead Active)"
    else:
        mode_str = f"Phase 13 — Autoregressive Dark Loops (N={args.loops})"

    baseline_vram = torch.cuda.memory_allocated(DEVICE) / 1e9 if DEVICE == "cuda" else 0.0

    print(f"\n{'='*62}")
    print(f"  MAMBA-3 LATENT REASONING ENGINE")
    print(f"  Mode: {mode_str}")
    print(f"  Device: {DEVICE.upper()} | Baseline VRAM: {baseline_vram:.2f} GB")
    print(f"  Use [LOGIC] prefix for math — [CHAT] for conversation")
    print(f"{'='*62}\n")

    while True:
        try:
            problem = input("❯ ").strip()
            if not problem or problem.lower() in ("exit", "quit", "q"):
                break

            print("  ⟳ Computing in latent state space...", end="\r", flush=True)

            if use_phase14:
                answer, n_loops, peak_vram, latency_ms = inner_loop_generate(
                    model, halting_head, tokenizer, problem,
                    max_new_tokens=args.tokens, temperature=args.temp
                )
            else:
                answer, n_loops, peak_vram, latency_ms = autoregressive_generate(
                    model, tokenizer, problem,
                    n_loops=args.loops,
                    max_new_tokens=args.tokens,
                    temperature=args.temp
                )

            print(f"  {'─'*58}")
            print(f"  OUTPUT  : {answer}")
            print(f"  {'─'*58}")
            print(f"  Loops   : {n_loops:>3d} ticks  │  "
                  f"VRAM: {peak_vram:.2f} GB  │  "
                  f"Latency: {latency_ms:.0f} ms")
            print()

        except KeyboardInterrupt:
            break

    print("\n[SYSTEM] Engine offline.")


if __name__ == "__main__":
    main()
