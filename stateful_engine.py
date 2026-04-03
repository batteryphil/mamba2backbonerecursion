"""
stateful_engine.py — True O(1) Stateful Loop Engine
====================================================
Replaces the re-tokenize-per-loop approach with MambaCache recurrent steps.

The original engine (mamba_engine.py / session_memory.py / the_crucible.py)
does this each iteration:
    toks = tok(prompt + "=" * lp, ...)
    h = model(**toks, output_hidden_states=True).hidden_states[-1][0,-1,:]

This rebuilds the full SSM state from scratch every loop — O(n) per iteration
where n is prompt_length + loop_count.

This engine instead:
  1. Runs one prefill pass to build the SSM state from the prompt
  2. Iterates by feeding a single spacer token while passing cache forward
  3. Reads h_t from the cached hidden state after each step

Each loop iteration is a single-token recurrent step — O(1) per iteration,
constant memory, no sequence growth.

API Note (transformers 5.3.0):
  - Mamba uses `cache_params` (NOT `past_key_values`)
  - `cache_position` is REQUIRED when passing cache manually
  - Prefill: cache_position.shape[0] == conv_kernel_size
  - Decode:  cache_position.shape[0] == 1 (single token)
  See docs/cache_api_findings.md for full details.
"""

import torch
import torch.nn as nn
import time
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, MambaCache


class HaltingHead(nn.Module):
    """Position-conditioned P(halt) probe. Copied from mamba_engine.py."""
    def __init__(self, d_input: int = 2561):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class StatefulLoopEngine:
    """
    True O(1) latent iteration using MambaCache recurrent steps.

    Unlike the original engine which re-tokenizes `prompt + "=" * lp` each loop,
    this engine:
      1. Runs one full forward pass to build SSM state from the prompt
      2. Iterates by feeding a single spacer token while passing cache forward
      3. Reads h_t from the cached state after each step

    Each loop iteration is a single-token recurrent step — sequence length never grows.
    """

    DOMAIN_MAX = {"chat": 5, "math": 25, "code": 45, "tool": 10}

    def __init__(self, engine_dir: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tok = AutoTokenizer.from_pretrained(engine_dir, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            engine_dir, dtype=torch.bfloat16,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        self.model.eval()

        # Load HaltingHead
        head_path = os.path.join(engine_dir, "halting_head.pt")
        if os.path.exists(head_path):
            ckpt = torch.load(head_path, weights_only=True, map_location=self.device)
            self.head = HaltingHead(ckpt["d_input"]).to(self.device)
            self.head.load_state_dict(ckpt["state_dict"])
            self.head.eval()
            self._has_head = True
        else:
            self._has_head = False
            self.head = None

        # Get spacer token ID
        self.spacer_id = self.tok.convert_tokens_to_ids("=")
        assert self.spacer_id != self.tok.unk_token_id, \
            "Spacer token '=' not in vocabulary — check tokenizer"

    def _new_cache(self) -> MambaCache:
        """Allocate a fresh MambaCache."""
        return MambaCache(
            self.model.config,
            max_batch_size=1,
            dtype=torch.bfloat16,
            device=self.device
        )

    def generate(self, prompt: str, domain: str = "chat",
                 halt_threshold: float = 0.70, max_new: int = 100,
                 verbose: bool = False):
        """
        Run latent loops then generate.

        Returns: (answer_text, loop_count, p_halt, loop_latencies_ms)
        """
        max_loops = self.DOMAIN_MAX.get(domain, 10)
        spacer = torch.tensor([[self.spacer_id]], device=self.device)
        loop_latencies = []

        with torch.no_grad():
            # --- Build initial SSM state from prompt (prefill) ---
            toks = self.tok(prompt, return_tensors="pt",
                            truncation=True, max_length=512)
            input_ids = toks.input_ids.to(self.device)
            seq_len = input_ids.shape[1]

            out = self.model(
                input_ids=input_ids,
                use_cache=True,
                output_hidden_states=True
            )
            cache = out.cache_params
            h = out.hidden_states[-1][0, -1, :].float()

            # --- Iterate: single-token recurrent steps ---
            p_halt = 0.0
            lp = 0
            for lp in range(max_loops):
                t0 = time.perf_counter()

                # Halting check
                if self._has_head:
                    ln = torch.tensor([lp / max_loops],
                                      dtype=torch.float32, device=self.device)
                    p_halt = self.head(torch.cat([h, ln]).unsqueeze(0)).item()

                    if verbose:
                        print(f"  Loop {lp}: P(halt)={p_halt:.3f}")

                    if p_halt >= halt_threshold:
                        loop_latencies.append((time.perf_counter() - t0) * 1000)
                        break
                elif verbose:
                    print(f"  Loop {lp}: (no halting head)")

                # Single-token recurrent step — O(1), no sequence growth
                cache_pos = torch.tensor([seq_len + lp], device=self.device)
                step_out = self.model(
                    input_ids=spacer,
                    cache_params=cache,
                    cache_position=cache_pos,
                    use_cache=True,
                    output_hidden_states=True
                )
                # cache is mutated in-place; step_out.cache_params is the same object
                h = step_out.hidden_states[-1][0, -1, :].float()

                loop_latencies.append((time.perf_counter() - t0) * 1000)

            # --- Generate answer from final state ---
            # Pass the accumulated cache to generate. The cache already holds
            # the full context (prompt + all spacer iterations).
            try:
                gen_cache_pos = torch.tensor(
                    [seq_len + lp + 1], device=self.device
                )
                out_ids = self.model.generate(
                    input_ids=spacer,
                    cache_params=cache,
                    cache_position=gen_cache_pos,
                    max_new_tokens=max_new,
                    do_sample=False,
                    repetition_penalty=1.1,
                    use_cache=True
                )
                # Decode only the generated tokens (skip the spacer input)
                answer = self.tok.decode(
                    out_ids[0][1:],
                    skip_special_tokens=True
                )
            except Exception as e:
                # KILL SWITCH: generate() may not accept pre-built cache.
                # Fall back to stateless generate from the original prompt.
                if verbose:
                    print(f"  [FALLBACK] generate with cache failed: {e}")
                    print(f"  [FALLBACK] Falling back to stateless generate")
                final_prompt = prompt + "=" * (lp + 1)
                final_toks = self.tok(final_prompt, return_tensors="pt",
                                      truncation=True, max_length=512)
                final_ids = final_toks.input_ids.to(self.device)
                out_ids = self.model.generate(
                    input_ids=final_ids,
                    max_new_tokens=max_new,
                    do_sample=False,
                    repetition_penalty=1.1
                )
                answer = self.tok.decode(
                    out_ids[0][final_ids.shape[1]:],
                    skip_special_tokens=True
                )

        return answer, lp, p_halt, loop_latencies

    def get_cache(self) -> MambaCache:
        """Get a fresh cache for manual use (e.g. session memory)."""
        return self._new_cache()


def detect_domain(text: str) -> str:
    """Heuristically detect the reasoning domain from prompt text."""
    t = text.lower()
    if any(w in t for w in ["def ", "class ", "```python", "function", "import"]):
        return "code"
    if any(w in t for w in ["calculate", "solve", "miles", "speed", "equation",
                             "formula", "logic", "x=", "y="]):
        return "math"
    if any(w in t for w in ["bash", "terminal", "command", "disk", "file", "process"]):
        return "tool"
    return "chat"


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    engine_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/mamba-2.8b-latent"
    if not os.path.isdir(engine_dir):
        # Fall back to base model for testing
        engine_dir = "state-spaces/mamba-2.8b-hf"
        print(f"[INFO] Checkpoint not found, using base model: {engine_dir}")

    print("[INIT] Loading StatefulLoopEngine...")
    eng = StatefulLoopEngine(engine_dir)
    print("[INIT] Ready.\n")

    # Quick self-test
    prompts = [
        ("[LOGIC] X=5. Y=X*2. Z=Y+3. W=Z-X. Output W. ====", "math"),
        ("[CHAT] The sky is ====", "chat"),
        ("[LOGIC] All birds have feathers. Penguins are birds. Can penguins fly? ====", "math"),
    ]

    for prompt, domain in prompts:
        print(f"Prompt: {prompt[:60]}...")
        answer, loops, p_halt, latencies = eng.generate(
            prompt, domain=domain, verbose=True
        )
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        print(f"  Answer: {answer[:80]}")
        print(f"  Loops: {loops}, P(halt): {p_halt:.3f}")
        print(f"  Avg loop latency: {avg_lat:.2f}ms")
        print()
