#!/usr/bin/env python3
"""
mamba_syrin_gate.py -- Local Mamba Reasoning Gate Daemon
Project TinyRefinementModel

OpenAI-compatible /v1/chat/completions endpoint wrapping the Two-Phase
Thinker → Coder pipeline. 100% local. No cloud API calls.

Routing logic (fully local):
  ==== spacers < 5  → simple query, return Phase 1 text directly
  ==== spacers >= 5 → complex query, run Phase 2 code synthesis

Usage:
    ./run_env.sh mamba_syrin_gate.py [--port 8742] [--model q4|q2]
    curl http://localhost:8742/v1/chat/completions \
         -H "Content-Type: application/json" \
         -d '{"model":"mamba-local","messages":[{"role":"user","content":"Implement binary search in Python."}]}'
"""

import argparse
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

WORKSPACE = Path(__file__).parent
GGUF_DIR  = WORKSPACE / "checkpoints" / "gguf"
Q4_GGUF   = GGUF_DIR / "mamba-tiny-refinement-q4_k_m.gguf"
Q2_GGUF   = GGUF_DIR / "mamba-tiny-refinement-q2_k.gguf"

# --------------------------------------------------------------------------
# Two-Phase Inference Parameters
# (Strictly local — no external API calls)
# --------------------------------------------------------------------------

PHASE1_BUDGET       = 1024   # O(1) VRAM cost — free to think for 1024 tokens
PHASE1_TEMPERATURE  = 0.4    # 0.4: breaks echo loops without hallucinated tags
PHASE1_TOP_K        = 10
PHASE1_REP_PENALTY  = 1.0    # MUST stay 1.0 so ==== can repeat freely

PHASE2_BUDGET       = 500    # Code synthesis budget
PHASE2_TEMPERATURE  = 0.3    # 0.3 breaks SSM limit-cycle / fractal elif loops
PHASE2_TOP_K        = 5
PHASE2_REP_PENALTY  = 1.15   # Re-enable to prevent Phase 2 code repetition

ROUTING_THRESHOLD   = 5      # spacers < threshold → simple; >= threshold → Phase 2

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("syrin-gate")


# --------------------------------------------------------------------------
# Model loader (singleton)
# --------------------------------------------------------------------------

_model_cache: dict[str, Any] = {}


def get_model(gguf_path: Path, n_ctx: int = 2048):
    """Load and cache the GGUF model singleton.

    The model is loaded once and reused across requests to avoid
    VRAM thrashing on every inference call.

    Args:
        gguf_path: Path to the quantized GGUF file.
        n_ctx: Context window size for llama.cpp bookkeeping.
               (Mamba state is O(1) regardless of this value.)

    Returns:
        Loaded Llama model instance.
    """
    cache_key = str(gguf_path)
    if cache_key not in _model_cache:
        from llama_cpp import Llama
        log.info("Loading model: %s (n_ctx=%d)", gguf_path.name, n_ctx)
        _model_cache[cache_key] = Llama(
            model_path=str(gguf_path),
            n_gpu_layers=-1,
            n_ctx=n_ctx,
            verbose=False,
        )
        log.info("Model loaded and cached.")
    return _model_cache[cache_key]


# --------------------------------------------------------------------------
# Two-Phase Inference Engine (100% local)
# --------------------------------------------------------------------------

def run_two_phase(model, prompt: str) -> dict[str, Any]:
    """Execute the full Thinker → Coder two-phase pipeline for one prompt.

    Phase 1 — The Thinker:
        Runs the ==== reasoning loop with repeat_penalty=1.0 and
        temperature=0.6 until [ANSWER] is predicted or PHASE1_BUDGET
        tokens are exhausted. The SSM hidden state saturates.

    Router:
        Count ==== spacer tokens in Phase 1 output.
        < ROUTING_THRESHOLD spacers → simple query; return Phase 1 text.
        >= ROUTING_THRESHOLD spacers → complex query; run Phase 2.

    Phase 2 — The Coder:
        Inject [ANSWER] into the context, then generate with
        repeat_penalty=1.15 and temperature=0.3 to break limit cycles
        and synthesize strict Python code from the evolved hidden state.

    No cloud API calls are made at any point. All inference runs locally
    on the GGUF model loaded from disk.

    Args:
        model: Loaded Llama model instance.
        prompt: Raw user task string.

    Returns:
        Dict with keys: thought_text, code_text, spacers,
                        routing_decision, elapsed_s, thought_toks,
                        code_toks, full_response.
    """
    t0           = time.time()
    phase1_prompt = f"[USER]\n{prompt}\n[REASONING]\n===="

    # ------------------------------------------------------------------
    # PHASE 1: THE THINKER
    # ------------------------------------------------------------------
    log.info("Phase 1 — Thinker  budget=%d  T=%.1f", PHASE1_BUDGET, PHASE1_TEMPERATURE)
    r1 = model(
        phase1_prompt,
        max_tokens=PHASE1_BUDGET,
        stop=["[ANSWER]"],
        repeat_penalty=PHASE1_REP_PENALTY,
        temperature=PHASE1_TEMPERATURE,
        top_k=PHASE1_TOP_K,
        echo=False,
    )
    thought_text = r1["choices"][0]["text"]
    thought_toks = r1["usage"]["completion_tokens"]
    stop_reason  = r1["choices"][0].get("finish_reason", "unknown")
    spacers      = thought_text.count("====")

    log.info("Phase 1 done — stop=%s  toks=%d  spacers=%d",
             stop_reason, thought_toks, spacers)

    # ------------------------------------------------------------------
    # ROUTER: simple vs complex
    # ------------------------------------------------------------------
    if spacers < ROUTING_THRESHOLD:
        routing = "simple"
        log.info("Router → SIMPLE (spacers=%d < %d)", spacers, ROUTING_THRESHOLD)
        # Clean up Phase 1 output: strip structural tags, return plain text
        clean = _strip_structural_tags(thought_text)
        elapsed = time.time() - t0
        return {
            "thought_text":      thought_text,
            "code_text":         "",
            "spacers":           spacers,
            "routing_decision":  routing,
            "elapsed_s":         round(elapsed, 2),
            "thought_toks":      thought_toks,
            "code_toks":         0,
            "full_response":     clean.strip(),
        }

    # ------------------------------------------------------------------
    # CONTEXT SWITCH → PHASE 2: THE CODER
    # ------------------------------------------------------------------
    routing = "complex"
    log.info("Router → COMPLEX (spacers=%d >= %d) — Phase 2 synthesis",
             spacers, ROUTING_THRESHOLD)

    if stop_reason == "stop":
        synthesis_prompt = phase1_prompt + thought_text + "[ANSWER]\n"
    else:
        synthesis_prompt = phase1_prompt + thought_text + "\n[ANSWER]\n"

    log.info("Phase 2 — Coder  budget=%d  T=%.1f  rep=%.2f",
             PHASE2_BUDGET, PHASE2_TEMPERATURE, PHASE2_REP_PENALTY)
    r2 = model(
        synthesis_prompt,
        max_tokens=PHASE2_BUDGET,
        stop=["[USER]"],
        repeat_penalty=PHASE2_REP_PENALTY,
        temperature=PHASE2_TEMPERATURE,
        top_k=PHASE2_TOP_K,
        echo=False,
    )
    code_text  = r2["choices"][0]["text"]
    code_toks  = r2["usage"]["completion_tokens"]
    elapsed    = time.time() - t0

    log.info("Phase 2 done — toks=%d  has_def=%s  total=%.1fs",
             code_toks, ("def " in code_text), elapsed)

    # Combine: thought summary + synthesized code
    full_response = _build_full_response(thought_text, code_text, spacers)

    return {
        "thought_text":      thought_text,
        "code_text":         code_text,
        "spacers":           spacers,
        "routing_decision":  routing,
        "elapsed_s":         round(elapsed, 2),
        "thought_toks":      thought_toks,
        "code_toks":         code_toks,
        "full_response":     full_response,
    }


def _strip_structural_tags(text: str) -> str:
    """Remove internal structural control tokens from output text.

    Args:
        text: Raw model output containing [TAG] markers.

    Returns:
        Cleaned text with structural tags removed.
    """
    import re
    text = re.sub(r"\[USER\]|\[REASONING\]|\[ANSWER\]|\[IMPORTANT\]|"
                  r"\[IMPLEMENT\]|\[REVIEW\]|\[COMPUTING\]|\[EXAMPLES\]|"
                  r"\[IMPLEMENTATION\]|\[OUTPUT\]|\[QUESTION\]|\[TIP\]",
                  "", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _build_full_response(thought: str, code: str, spacers: int) -> str:
    """Format the combined response for API output.

    Args:
        thought: Phase 1 reasoning text.
        code: Phase 2 synthesized code.
        spacers: Number of ==== tokens generated.

    Returns:
        Formatted string suitable for the API response content field.
    """
    lines = []
    lines.append(f"<!-- reasoning: {spacers} latent steps -->")
    if code.strip():
        # Strip leading spacers/tags from the code block
        code_clean = _strip_structural_tags(code)
        # Wrap in a markdown code block if it's not already
        if "```" not in code_clean and ("def " in code_clean or "import " in code_clean):
            code_clean = f"```python\n{code_clean.strip()}\n```"
        lines.append(code_clean.strip())
    else:
        lines.append(_strip_structural_tags(thought).strip())
    return "\n\n".join(lines)


# --------------------------------------------------------------------------
# FastAPI Server
# --------------------------------------------------------------------------

def build_app(model_path: Path):
    """Build the FastAPI application with the two-phase endpoint.

    Implements the OpenAI /v1/chat/completions interface so this daemon
    can be used as a drop-in local backend for VSCode/Cursor AI assistants.

    Args:
        model_path: Path to the GGUF model to load.

    Returns:
        FastAPI application instance.
    """
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse
    except ImportError:
        log.error("FastAPI not found. Install with: pip install fastapi uvicorn")
        sys.exit(1)

    app = FastAPI(
        title="Mamba Syrin Gate",
        description=(
            "Local Mamba reasoning gate — OpenAI-compatible endpoint. "
            "Two-Phase Thinker→Coder pipeline. 100% local. No cloud API."
        ),
        version="1.0.0",
    )

    # Pre-load model on startup
    @app.on_event("startup")
    async def startup_event():
        """Load the GGUF model into VRAM on server start."""
        log.info("Daemon starting — loading model into VRAM...")
        get_model(model_path)
        log.info("Server ready. Listening for requests.")

    @app.get("/health")
    async def health():
        """Health check endpoint.

        Returns:
            Dict confirming server is alive and model is loaded.
        """
        return {
            "status": "ok",
            "model": model_path.name,
            "routing_threshold": ROUTING_THRESHOLD,
            "phase1_budget": PHASE1_BUDGET,
            "architecture": "Mamba-1.4B SSM O(1) memory",
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """OpenAI-compatible chat completions endpoint.

        Accepts the standard OpenAI request format. Extracts the last
        user message, runs it through the two-phase pipeline, and returns
        an OpenAI-shaped response object.

        The routing decision (simple vs complex) is made locally based on
        the ==== spacer count from Phase 1. No cloud API is called.

        Args:
            request: FastAPI Request object containing JSON body.

        Returns:
            JSONResponse matching OpenAI /v1/chat/completions schema.

        Raises:
            HTTPException: 400 if request body is malformed.
            HTTPException: 500 if inference fails.
        """
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        messages = body.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="messages array required")

        # Extract the last user message
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    user_msg = content
                elif isinstance(content, list):
                    # Handle multi-part content (text only)
                    user_msg = " ".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                break

        if not user_msg.strip():
            raise HTTPException(status_code=400, detail="No user message found")

        model = get_model(model_path)

        try:
            result = run_two_phase(model, user_msg)
        except Exception as exc:
            log.exception("Inference error: %s", exc)
            raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

        # Build OpenAI-compatible response
        completion_id = f"chatcmpl-mamba-{uuid.uuid4().hex[:12]}"
        finish_reason = "stop"
        response_body = {
            "id":      completion_id,
            "object":  "chat.completion",
            "created": int(time.time()),
            "model":   "mamba-tiny-refinement",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role":    "assistant",
                        "content": result["full_response"],
                    },
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens":     len(user_msg.split()),   # approx
                "completion_tokens": result["thought_toks"] + result["code_toks"],
                "total_tokens":      (len(user_msg.split())
                                      + result["thought_toks"]
                                      + result["code_toks"]),
            },
            # Non-standard extension: expose routing metadata
            "x_mamba_metadata": {
                "routing_decision":  result["routing_decision"],
                "spacers":           result["spacers"],
                "thought_toks":      result["thought_toks"],
                "code_toks":         result["code_toks"],
                "elapsed_s":         result["elapsed_s"],
                "phase1_budget":     PHASE1_BUDGET,
                "routing_threshold": ROUTING_THRESHOLD,
                "architecture":      "Mamba-1.4B SSM O(1) memory — no KV cache",
                "cloud_api_used":    False,  # Always False. This is 100% local.
            },
        }

        log.info(
            "Request served — routing=%s  spacers=%d  toks=%d+%d  %.1fs",
            result["routing_decision"],
            result["spacers"],
            result["thought_toks"],
            result["code_toks"],
            result["elapsed_s"],
        )

        return JSONResponse(content=response_body)

    @app.get("/v1/models")
    async def list_models():
        """List available models (OpenAI-compatible).

        Returns:
            Dict listing the local Mamba model.
        """
        return {
            "object": "list",
            "data": [
                {
                    "id":       "mamba-tiny-refinement",
                    "object":   "model",
                    "created":  1713000000,
                    "owned_by": "local",
                }
            ],
        }

    return app


# --------------------------------------------------------------------------
# CLI Entry Point
# --------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and launch the Syrin Gate daemon."""
    parser = argparse.ArgumentParser(
        description="Mamba Syrin Gate — local OpenAI-compatible reasoning daemon"
    )
    parser.add_argument("--port",  type=int, default=8742,
                        help="Port to listen on (default: 8742)")
    parser.add_argument("--host",  type=str, default="127.0.0.1",
                        help="Host to bind (default: 127.0.0.1 — localhost only)")
    parser.add_argument("--model", choices=["q4", "q2"], default="q4",
                        help="Which quantization to load: q4=Q4_K_M (default), q2=Q2_K")
    parser.add_argument("--threshold", type=int, default=ROUTING_THRESHOLD,
                        help=f"Spacer routing threshold (default: {ROUTING_THRESHOLD})")
    args = parser.parse_args()

    # Override module-level threshold if specified
    global ROUTING_THRESHOLD
    ROUTING_THRESHOLD = args.threshold

    model_path = Q4_GGUF if args.model == "q4" else Q2_GGUF
    if not model_path.exists():
        log.error("GGUF not found: %s — run phase3_blacksmith.py first.", model_path)
        sys.exit(1)

    log.info("=" * 60)
    log.info("MAMBA SYRIN GATE — Project TinyRefinementModel")
    log.info("=" * 60)
    log.info("Model:             %s", model_path.name)
    log.info("Architecture:      Mamba-1.4B SSM  O(1) memory  No KV cache")
    log.info("Cloud API:         DISABLED (100%% local)")
    log.info("Routing threshold: %d spacers", ROUTING_THRESHOLD)
    log.info("Phase 1 budget:    %d tokens (O(1) VRAM cost)", PHASE1_BUDGET)
    log.info("Endpoint:          http://%s:%d/v1/chat/completions", args.host, args.port)
    log.info("Health check:      http://%s:%d/health", args.host, args.port)
    log.info("=" * 60)

    app = build_app(model_path)

    try:
        import uvicorn
    except ImportError:
        log.error("uvicorn not found. Install with: pip install uvicorn")
        sys.exit(1)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
