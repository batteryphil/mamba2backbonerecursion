"""
agent_loop.py — Phase 7: Autonomous Live Bash Executor

Closes the agentic loop: latent engine emits <TOOL: BASH>,
Python executes it via subprocess, stdout is injected as <RESULT>,
and the loop continues until a final natural language answer emerges.

Usage:
  python agent_loop.py "How much disk space is available?"
  python agent_loop.py "Find the largest file in the checkpoints dir."
  python agent_loop.py "What AI packages do I have installed?"
"""

import torch, torch.nn as nn, subprocess, re, sys, time
from transformers import AutoTokenizer, AutoModelForCausalLM

ENGINE_DIR  = "checkpoints/mamba-2.8b-latent"
HALT_THRESH = 0.7
MAX_LOOPS   = 50
MAX_TURNS   = 6
DOMAIN_MAX  = {"tool": 10, "math": 25, "code": 45, "chat": 5}
CMD_TIMEOUT = 10
CMD_MAX     = 1500

TOOL_PAT   = re.compile(r"<TOOL: BASH>\s*(.*?)\s*</TOOL>", re.DOTALL)
UNSAFE_PAT = re.compile(
    r"\b(rm\s+-rf|mkfs|dd\s+if=|shutdown|reboot|wget|curl\s+-O)\b",
    re.IGNORECASE
)


class HaltingHead(nn.Module):
    """Position-conditioned P(halt) probe: [h_2560 | loop_pos_norm] -> P(halt)."""
    def __init__(self, d_input: int = 2561):
        """Initialize layers."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x).squeeze(-1)


def load_engine(engine_dir: str):
    """Load tokenizer, model, and HaltingHead from the engine directory."""
    print("=" * 58)
    print("  MAMBA-2.8B LATENT ENGINE  —  AGENT MODE")
    print("=" * 58)
    _tok = AutoTokenizer.from_pretrained(engine_dir, trust_remote_code=True)
    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token
    _mdl = AutoModelForCausalLM.from_pretrained(
        engine_dir, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    _mdl.eval()
    _ck  = torch.load(f"{engine_dir}/halting_head.pt", weights_only=True)
    _hd  = HaltingHead(_ck["d_input"]).cuda()
    _hd.load_state_dict(_ck["state_dict"])
    _hd.eval()
    print("[READY] Engine and HaltingHead online.\n")
    return _tok, _mdl, _hd


def latent_generate(context: str, tok, mdl, head,
                    domain: str = "tool", max_new: int = 120) -> dict:
    """Run one latent generation pass with HaltingHead loop control."""
    m = DOMAIN_MAX.get(domain, 10)
    p = 0.0
    lp = 0
    with torch.no_grad():
        for lp in range(MAX_LOOPS):
            toks = tok(
                context + "=" * lp, return_tensors="pt",
                truncation=True, max_length=512
            ).to("cuda")
            h  = mdl(**toks, output_hidden_states=True).hidden_states[-1][0,-1,:].float()
            ln = torch.tensor([lp / m], dtype=torch.float32, device="cuda")
            p  = head(torch.cat([h, ln]).unsqueeze(0)).item()
            if p >= HALT_THRESH:
                break
        out = mdl.generate(**toks, max_new_tokens=max_new,
                           do_sample=False, repetition_penalty=1.1)
    surface = tok.decode(out[0][toks["input_ids"].shape[1]:],
                         skip_special_tokens=False).strip()
    return {"text": surface, "loops": lp + 1, "p_halt": round(p, 3)}


def execute_bash(cmd: str) -> str:
    """Execute a shell command safely and return stdout."""
    cmd = cmd.strip()
    if UNSAFE_PAT.search(cmd):
        return "[BLOCKED] Unsafe command pattern — execution skipped."
    print(f"  $ {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True,
            text=True, timeout=CMD_TIMEOUT
        )
        out = (result.stdout.strip() or result.stderr.strip() or "(no output)")[:CMD_MAX]
    except subprocess.TimeoutExpired:
        out = f"[TIMEOUT] Exceeded {CMD_TIMEOUT}s"
    except Exception as exc:
        out = f"[ERROR] {exc}"
    print(f"  >> {out[:160]}{'...' if len(out) > 160 else ''}")
    return out


def run_agent(task: str, tok, mdl, head) -> None:
    """Execute the full autonomous agentic loop for a task."""
    s = "-" * 58
    print(f"\n{s}\n  TASK: {task[:50]}\n{s}")
    context = f"[AGENT] {task}\n"
    t0 = time.time()

    for turn in range(MAX_TURNS):
        print(f"\n  [Turn {turn + 1}/{MAX_TURNS}]", flush=True)
        result = latent_generate(context, tok, mdl, head)
        text   = result["text"]
        print(f"  Loops: {result['loops']}  P(halt): {result['p_halt']}")

        match = TOOL_PAT.search(text)
        if match:
            cmd    = match.group(1).strip()
            stdout = execute_bash(cmd)
            context += (
                text[:match.start()].strip() + "\n"
                + f"<TOOL: BASH>\n{cmd}\n</TOOL>\n"
                + f"<RESULT>\n{stdout}\n</RESULT>\n"
            )
        else:
            # Strip loop tokens and print final answer
            answer = re.sub(r"=+", "", text)
            answer = re.sub(r"<[^>]+>", "", answer).strip()
            elapsed = round(time.time() - t0, 1)
            print(f"\n  ANSWER ({elapsed}s):")
            print(f"  {answer[:600]}")
            print(f"\n{s}")
            return

    print("\n  [MAX TURNS] Returning last known context.")
    print(f"\n{s}")


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "How much disk space is available?"
    tok, mdl, head = load_engine(ENGINE_DIR)
    run_agent(task, tok, mdl, head)
