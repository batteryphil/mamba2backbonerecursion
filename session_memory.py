"""
session_memory.py
=================
Phase 6: Persistent Multi-Turn Session Memory

Serializes and restores the Mamba SSM recurrent state (h_t) between
conversations. Each session is a ~5 MB tensor file on disk.

Capabilities:
  - Zero VRAM growth between turns (no KV cache accumulation)
  - Zero prefill latency on resume (state loaded directly into SRAM)
  - Infinite effective context window
  - Multi-session identity (phil.pt, project_x.pt, etc.)

Usage:
  python session_memory.py                  # start a new session
  python session_memory.py --resume phil    # resume named session
  python session_memory.py --list           # list saved sessions
"""

import torch
import torch.nn as nn
import os
import sys
import time
import glob
import readline
from transformers import AutoTokenizer, AutoModelForCausalLM, MambaCache

ENGINE_DIR    = "checkpoints/mamba-2.8b-latent"
SESSION_DIR   = "sessions"
HALT_THRESH   = 0.7
MAX_LOOPS     = 50
DOMAIN_MAX    = {"math": 25, "code": 45, "chat": 5, "tool": 10}
os.makedirs(SESSION_DIR, exist_ok=True)


class HaltingHead(nn.Module):
    """Position-conditioned P(halt) probe."""
    def __init__(self, d_input: int = 2561):
        """Init."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self.net(x).squeeze(-1)


def detect_domain(text: str) -> str:
    """Heuristically detect the reasoning domain from prompt text."""
    t = text.lower()
    if any(w in t for w in ["def ", "class ", "```python", "function", "import"]):
        return "code"
    if any(w in t for w in ["calculate", "solve", "miles", "speed", "equation", "formula"]):
        return "math"
    if any(w in t for w in ["bash", "terminal", "command", "disk", "file", "process"]):
        return "tool"
    return "chat"


def load_engine():
    """Load model, tokenizer, and HaltingHead."""
    print("=" * 58)
    print("  MAMBA-2.8B  —  SESSION MEMORY ENGINE")
    print("=" * 58)
    tok = AutoTokenizer.from_pretrained(ENGINE_DIR, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        ENGINE_DIR, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    mdl.eval()
    ck  = torch.load(f"{ENGINE_DIR}/halting_head.pt", weights_only=True)
    hd  = HaltingHead(ck["d_input"]).cuda()
    hd.load_state_dict(ck["state_dict"])
    hd.eval()
    return tok, mdl, hd


def new_cache(mdl) -> MambaCache:
    """Allocate a fresh Mamba recurrent state cache."""
    return MambaCache(
        mdl.config,
        max_batch_size=1,
        dtype=torch.bfloat16,
        device="cuda"
    )


def save_session(cache: MambaCache, name: str, history: list) -> str:
    """Serialize the SSM h_t state and conversation history to disk."""
    path = os.path.join(SESSION_DIR, f"{name}.pt")
    # Extract tensors from MambaCache
    state = {
        "conv_states": [s.cpu() for s in cache.conv_states],
        "ssm_states":  [s.cpu() for s in cache.ssm_states],
        "history":     history,
        "saved_at":    time.time()
    }
    torch.save(state, path)
    size_kb = os.path.getsize(path) / 1024
    return f"{path} ({size_kb:.0f} KB)"


def load_session(mdl, name: str):
    """Restore a MambaCache and conversation history from disk."""
    path = os.path.join(SESSION_DIR, f"{name}.pt")
    if not os.path.exists(path):
        print(f"[SESSION] No session found: {path}")
        return new_cache(mdl), []
    state = torch.load(path, weights_only=True)
    cache = new_cache(mdl)
    for i, s in enumerate(state["conv_states"]):
        cache.conv_states[i][:] = s.cuda()
    for i, s in enumerate(state["ssm_states"]):
        cache.ssm_states[i][:] = s.cuda()
    age  = (time.time() - state["saved_at"]) / 3600
    hist = state.get("history", [])
    print(f"[SESSION] Resumed '{name}' — {len(hist)} turns, {age:.1f}h ago")
    return cache, hist


def list_sessions():
    """Print all saved sessions."""
    files = sorted(glob.glob(os.path.join(SESSION_DIR, "*.pt")))
    if not files:
        print("[SESSION] No saved sessions found.")
        return
    print(f"[SESSION] Saved sessions ({len(files)}):")
    for f in files:
        state   = torch.load(f, weights_only=True)
        turns   = len(state.get("history", []))
        age_h   = (time.time() - state["saved_at"]) / 3600
        kb      = os.path.getsize(f) / 1024
        name    = os.path.basename(f).replace(".pt", "")
        print(f"  {name:<20} {turns:>3} turns  {age_h:>5.1f}h ago  {kb:.0f} KB")


def latent_turn(prompt: str, cache: MambaCache, tok, mdl, head) -> tuple:
    """Run one conversation turn through the latent engine with live cache.

    Uses O(1) stateful iteration: prefill once, then single-token recurrent
    steps via MambaCache. Each loop iteration feeds one spacer token while
    passing the cache state forward — no re-tokenization, no sequence growth.
    """
    domain   = detect_domain(prompt)
    m        = DOMAIN_MAX.get(domain, 5)
    p        = 0.0
    lp       = 0
    spacer_id = tok.convert_tokens_to_ids("=")

    with torch.no_grad():
        # Prefill: process prompt through model, building SSM state
        toks     = tok(prompt, return_tensors="pt",
                       truncation=True, max_length=512).to("cuda")
        seq_len  = toks["input_ids"].shape[1]
        out      = mdl(
            **toks,
            cache_params=cache,
            use_cache=True,
            output_hidden_states=True
        )
        h = out.hidden_states[-1][0, -1, :].float()

        # O(1) loop: single-token recurrent steps
        spacer = torch.tensor([[spacer_id]], device="cuda")
        for lp in range(MAX_LOOPS):
            ln = torch.tensor([lp / m], dtype=torch.float32, device="cuda")
            p  = head(torch.cat([h, ln]).unsqueeze(0)).item()
            if p >= HALT_THRESH:
                break

            cache_pos = torch.tensor([seq_len + lp], device="cuda")
            step_out  = mdl(
                input_ids=spacer,
                cache_params=cache,
                cache_position=cache_pos,
                use_cache=True,
                output_hidden_states=True
            )
            h = step_out.hidden_states[-1][0, -1, :].float()

        # Autoregressive surface generation from accumulated cache state
        gen_cache_pos = torch.tensor([seq_len + lp + 1], device="cuda")
        gen_out = mdl.generate(
            spacer,
            cache_params=cache,
            cache_position=gen_cache_pos,
            max_new_tokens=120,
            do_sample=False,
            repetition_penalty=1.1,
            use_cache=True
        )

    surface = tok.decode(
        gen_out[0][1:],
        skip_special_tokens=True
    ).strip()
    return surface, lp + 1, round(p, 3)


def chat_loop(session_name: str, tok, mdl, head) -> None:
    """Run an interactive multi-turn session with persistent memory."""
    vram_start  = torch.cuda.memory_allocated() / 1024 ** 2

    print(f"\n[SESSION] Starting session: '{session_name}'")
    print("  Commands: /save  /load  /new  /history  /quit")
    print("  The SSM h_t state persists across ALL turns.\n")

    cache, history = load_session(mdl, session_name)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        # Commands
        if user_input == "/quit":
            break
        if user_input == "/save":
            path = save_session(cache, session_name, history)
            print(f"[SESSION] Saved → {path}")
            continue
        if user_input == "/history":
            if not history:
                print("[SESSION] No history yet.")
            for t, a in history:
                print(f"  You:    {t}")
                print(f"  Agent:  {a}\n")
            continue
        if user_input == "/new":
            cache   = new_cache(mdl)
            history = []
            print("[SESSION] New session started (state cleared).")
            continue

        # Run latent turn
        answer, loops, p_halt = latent_turn(user_input, cache, tok, mdl, head)
        history.append((user_input, answer))

        vram_now   = torch.cuda.memory_allocated() / 1024 ** 2
        vram_delta = vram_now - vram_start

        print(f"Agent [{loops}L P={p_halt} VRAM+{vram_delta:.1f}MB]: {answer}\n")

    # Auto-save on exit
    path = save_session(cache, session_name, history)
    print(f"\n[SESSION] Auto-saved → {path}")
    print(f"[SESSION] Resume with: python session_memory.py --resume {session_name}")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--list" in args:
        list_sessions()
        sys.exit(0)

    name = "default"
    if "--resume" in args:
        idx  = args.index("--resume")
        name = args[idx + 1] if idx + 1 < len(args) else "default"
    elif args:
        name = args[0]

    tok, mdl, head = load_engine()
    chat_loop(name, tok, mdl, head)
