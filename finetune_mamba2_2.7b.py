"""
finetune_mamba2_2.7b.py — Recursive Latent Forcing on Mamba2-2.7B
================================================================================
Scale-up of v34 (130M) to 2.7B: 20x more parameters, same RLF architecture.

Architecture: Mamba2-2.7B (d_model=2560, n_layers=64, tie_embeddings=True)
Training: LoRA rank-4 on upper 32 layers + Mamba2 loop engine + RoPE
VRAM: ~11GB (fits RTX 3060 12GB with BS=1, grad checkpointing, bf16)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel, Mamba2
import json, random, time, os, re, sys
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint as grad_ckpt

# ── Live log tee ──────────────────────────────────────────────────────────────
class Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, log_path: str):
        """Init with log file path."""
        self.terminal = sys.stdout
        self.log = open(log_path, 'w')
    def write(self, message: str) -> None:
        """Write to both outputs."""
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self) -> None:
        """Flush both outputs."""
        self.terminal.flush()
        self.log.flush()

sys.stdout = Tee('training_2.7b.log')
print(f"  Live log: training_2.7b.log  (watch with: tail -f training_2.7b.log)")

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID    = "state-spaces/mamba2-2.7b"
BASE_SPLIT  = 48            # Freeze layers 0-47, LoRA on last 16 layers
LORA_RANK   = 4             # Rank-4: wider layers need less rank
MAX_LOOPS   = 6             # Cap BPTT depth for VRAM (RoPE extrapolates)
SEQ_LEN     = 128           # Our prompts are short; saves ~40% activation mem
BATCH_SIZE  = 1             # Reduced for 12GB VRAM
ACCUM       = 16            # Effective batch = 16
STEPS       = 5_000         # Larger model converges faster
LOG_EVERY   = 50
VAL_EVERY   = 500
EARLY_STOP_ACC   = 95.0
EARLY_STOP_COUNT = 3
RESUME_FROM = None          # Fresh start from pretrained 2.7B base
SAVE_PATH   = "mamba2_2.7b_rlf_rope.pt"

LOOP_HEADDIM = 128          # Scaled for d_model=2560
LOOP_D_STATE = 32           # Reduced from 64 to save VRAM
LOOP_EXPAND  = 1            # Reduced from 2 to save VRAM (halves loop engine)

print(f"\n{'='*70}", flush=True)
print(f"  Mamba2-2.7B — Recursive Latent Forcing + RoPE Loop Encoding", flush=True)
print(f"  Scale-up from 130M: 20x params, same RLF architecture", flush=True)
print(f"  Device: {DEVICE} | Steps={STEPS} | MAX_LOOPS={MAX_LOOPS}", flush=True)
print(f"{'='*70}\n", flush=True)


# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>", "<HALT>"]})
HALT_ID = tokenizer.convert_tokens_to_ids("<HALT>")
print(f"  Vocab: {len(tokenizer):,} | <HALT>: {HALT_ID}")


# ── 1D RoPE for Loop Index ─────────────────────────────────────────────────────
class LoopRoPE(nn.Module):
    """1D Rotary Position Embedding for loop index encoding.

    Applies analytical rotation to hidden state at each loop step.
    Composable for any loop index — no table boundary.
    """

    def __init__(self, d_model: int, base: int = 10000):
        """Init: precompute frequency bands."""
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def _get_sincos(self, loop_index: int, device: torch.device, dtype: torch.dtype):
        """Compute cos/sin for a given loop index."""
        n = torch.tensor(float(loop_index), device=device)
        freqs = n * self.inv_freq.to(device=device, dtype=torch.float32)
        cos_f = freqs.cos()
        sin_f = freqs.sin()
        cos_v = torch.stack([cos_f, cos_f], dim=-1).flatten()[:self.d_model]
        sin_v = torch.stack([sin_f, sin_f], dim=-1).flatten()[:self.d_model]
        return cos_v.to(dtype=dtype), sin_v.to(dtype=dtype)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate pairs: [x1, x2, ...] → [-x2, x1, ...]."""
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1)
        return rotated.flatten(-2)

    def forward(self, x: torch.Tensor, loop_index: int) -> torch.Tensor:
        """Apply RoPE rotation for loop_index to x. x: [B, T, d_model]."""
        cos_v, sin_v = self._get_sincos(loop_index, x.device, x.dtype)
        return x * cos_v + self._rotate_half(x) * sin_v


# ── Chain parsers (same as 130M) ──────────────────────────────────────────────
def _parse_chain(text: str) -> list[str] | None:
    """Extract per-loop targets + HALT from chain text."""
    assignments = re.findall(r'([A-Za-z_]\w*)\s*=\s*(\S+?)[.\n]', text)
    if len(assignments) < 2:
        return None
    val: dict[str, str] = {}
    for var, expr in assignments:
        val[var] = expr
    chain_vars = [assignments[0][0]]
    for var, _ in assignments[1:]:
        chain_vars.append(var)
    targets: list[str] = [chain_vars[i] for i in range(len(chain_vars) - 1)]
    final_var = chain_vars[-1]
    resolved  = val.get(final_var, final_var)
    visited: set[str] = set()
    while resolved in val and resolved not in visited:
        visited.add(resolved)
        resolved = val[resolved]
    targets.append(resolved)
    targets.append("<HALT>")
    return targets if len(targets) >= 3 else None


def _parse_v2_chain(answer: str) -> list[str] | None:
    """Extract per-loop targets from v2_clean answer format.

    Input: '[Reasoning] Z ← pink; W ← Z = pink [Answer] pink <HALT>'
    Output: ['Z', 'pink', 'pink', '<HALT>']

    Args:
        answer: v2_clean answer string

    Returns:
        list of target tokens for chain training, or None
    """
    m = re.match(
        r'\[Reasoning\]\s*(.+?)\s*\[Answer\]\s*(.+?)\s*<HALT>',
        answer.strip()
    )
    if not m:
        return None

    reasoning = m.group(1).strip()
    final_ans = m.group(2).strip()
    targets: list[str] = []
    steps = [s.strip() for s in reasoning.split(';') if s.strip()]

    for step in steps:
        arith_m = re.search(r'=\s*(\S+)\s*$', step)
        if arith_m:
            targets.append(arith_m.group(1)); continue
        assign_m = re.search(r'←\s*(\S+?)(?:\s*=\s*(\S+))?\s*$', step)
        if assign_m:
            targets.append(assign_m.group(2) or assign_m.group(1)); continue
        cont_m = re.search(r'→\s*(\S+)\s*$', step)
        if cont_m:
            targets.append(cont_m.group(1)); continue
        if step.lower().startswith('given'):
            continue
        step_m = re.search(r'Step \d+:\s*(.+)', step)
        if step_m:
            words = step_m.group(1).split()
            if words:
                targets.append(words[-1])
            continue

    if not targets or targets[-1] != final_ans:
        targets.append(final_ans)
    targets.append("<HALT>")
    return targets if len(targets) >= 2 else None


def _parse_override(sample: dict) -> list[str]:
    """Override: single direct answer then HALT."""
    return [sample["answer"], "<HALT>"]


def find_answer_start(ids: list[int]) -> int:
    """Find first position after 'Answer:' boundary."""
    for boundary in (
        tokenizer.encode("Answer:",  add_special_tokens=False),
        tokenizer.encode(" Answer:", add_special_tokens=False),
        tokenizer.encode("\nAnswer:", add_special_tokens=False),
    ):
        n = len(boundary)
        for i in range(len(ids) - n + 1):
            if ids[i:i + n] == boundary:
                return min(i + n, len(ids) - 1)
    return -1


# ── LoRA ──────────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """Low-rank adapter. lora_B init to zero → identity at warmup."""

    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = 8.0):
        """Init from base linear, preserving dtype."""
        super().__init__()
        self.bias  = linear.bias
        d_out, d_in = linear.weight.shape
        dtype = linear.weight.dtype
        self.register_buffer("base_weight", linear.weight.data.clone())
        self.lora_A = nn.Parameter(torch.empty(rank, d_in,  dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, dtype=dtype))
        self.scale  = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A)

    @property
    def weight(self) -> torch.Tensor:
        """Fused weight: base + scaled LoRA."""
        return self.base_weight + self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with fused LoRA weight."""
        return F.linear(x, self.weight, self.bias)


# ── Recursive Mamba2-2.7B: RoPE Loop Encoding ────────────────────────────────
class RecursiveMamba2_RLF(nn.Module):
    """RLF wrapper for Mamba2-2.7B with RoPE loop encoding.

    Same architecture as v34 (130M) — LoRA on upper layers, Mamba2 loop
    engine, lifeline gate, RoPE loop index. Scaled for d_model=2560.
    """

    MAX_LOOPS: int = MAX_LOOPS

    def __init__(self, backbone: MambaLMHeadModel, lora_rank: int = 4):
        """Init: freeze base, LoRA top, Mamba2 loop, float32 vector gate, RoPE."""
        super().__init__()
        self.backbone   = backbone.backbone
        self.lm_head    = backbone.lm_head
        self.all_layers = nn.ModuleList(backbone.backbone.layers)
        self.norm       = backbone.backbone.norm_f
        d_model         = backbone.backbone.embedding.embedding_dim

        # Freeze lower layers (0 to BASE_SPLIT-1)
        for layer in self.all_layers[:BASE_SPLIT]:
            for p in layer.parameters():
                p.requires_grad = False

        # LoRA on upper layers (BASE_SPLIT to end)
        for layer in self.all_layers[BASE_SPLIT:]:
            mx = layer.mixer
            for attr in ("in_proj", "out_proj"):
                if hasattr(mx, attr):
                    setattr(mx, attr, LoRALinear(getattr(mx, attr),
                                                 rank=lora_rank,
                                                 alpha=lora_rank * 2.0))

        # RoPE loop encoding (analytical, no learnable params)
        self.loop_rope   = LoopRoPE(d_model)

        # Loop engine
        self.loop_norm   = nn.RMSNorm(d_model).to(torch.bfloat16)
        self.mamba2_core = Mamba2(
            d_model    = d_model,
            d_state    = LOOP_D_STATE,
            d_conv     = 4,
            expand     = LOOP_EXPAND,
            headdim    = LOOP_HEADDIM,
            chunk_size = 64,
        ).to(torch.bfloat16)
        nn.init.zeros_(self.mamba2_core.out_proj.weight)

        # Float32 vector lifeline gate
        self.lifeline_gate = nn.Parameter(
            torch.ones(d_model, dtype=torch.float32)
        )
        self.d_model = d_model

        n_lora = sum(p.numel() for n, p in self.named_parameters()
                     if p.requires_grad and "lora" in n.lower())
        total  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  LoRA params:     {n_lora:,}")
        print(f"  Loop engine:     {sum(p.numel() for p in self.mamba2_core.parameters()):,}")
        print(f"  RoPE:            0 params (analytical)")
        print(f"  Lifeline gate:   {d_model:,} floats (float32 vector)")
        print(f"  Total trainable: {total:,}")
        print(f"  Base frozen:     {sum(p.numel() for p in self.parameters() if not p.requires_grad):,}")
        print(f"  Loop encoding:   RoPE (loop_i) — valid for any loop index\n")

    def _lifeline_inject(self, x: torch.Tensor, x_prompt: torch.Tensor) -> torch.Tensor:
        """Per-dimension lifeline injection in fp32, cast back to bf16."""
        gate = self.lifeline_gate.to(x.dtype)
        return x + gate.unsqueeze(0).unsqueeze(0) * x_prompt

    def forward(
        self,
        input_ids:     torch.Tensor,
        chain_targets: list | None = None,
        ans_starts:    list | None = None,
    ) -> tuple:
        """Forward: base encode → lifeline → RoPE loop → predict."""
        x        = self.backbone.embedding(input_ids)
        residual = None
        for layer in self.all_layers:
            x, residual = layer(x, residual)

        x_prompt = x.clone().detach()   # Prompt Lifeline anchor

        # ── Training ──────────────────────────────────────────────────────────
        if self.training and chain_targets is not None:
            B, max_len = input_ids.shape
            n_loops    = max(len(t) for t in chain_targets)

            def run_lora(x_in, res_in):
                """Run LoRA layers with gradient checkpointing."""
                for layer in self.all_layers[BASE_SPLIT:]:
                    x_in, res_in = layer(x_in, res_in)
                return x_in, res_in

            step_losses: list[torch.Tensor] = []
            step_accs:   list[torch.Tensor] = []
            halt_accs:   list[float]        = []

            for loop_i in range(n_loops):
                x = self._lifeline_inject(x, x_prompt)
                x = self.loop_rope(x, loop_i)
                x, residual = grad_ckpt(run_lora, x, residual, use_reentrant=False)
                x = x + self.mamba2_core(x)
                x = self.loop_norm(x)

                logits_step = self.lm_head(self.norm(x, residual, prenorm=False))
                vocab_size  = logits_step.shape[-1]

                loop_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
                loop_acc  = torch.tensor(0.0, device=x.device)
                valid = 0

                for b in range(B):
                    as_ = ans_starts[b]
                    if as_ < 1 or as_ >= max_len:
                        continue
                    btgt   = chain_targets[b]
                    tgt_id = int(btgt[min(loop_i, len(btgt) - 1)])
                    if tgt_id >= vocab_size:
                        continue
                    logits_b = logits_step[b, as_ - 1, :]
                    pred_tok = logits_b.argmax().item()
                    tgt_t    = torch.tensor(tgt_id, device=x.device)
                    loop_loss = loop_loss + F.cross_entropy(
                        logits_b.unsqueeze(0), tgt_t.unsqueeze(0)
                    )
                    loop_acc = loop_acc + float(pred_tok == tgt_id)
                    valid   += 1
                    if tgt_id == HALT_ID:
                        halt_accs.append(float(pred_tok == tgt_id))

                if valid > 0:
                    step_losses.append(loop_loss / valid)
                    step_accs.append(loop_acc   / valid)

            avg_loss   = (torch.stack(step_losses).mean()
                          if step_losses else
                          torch.tensor(0.0, device=x.device, requires_grad=True))
            avg_acc    = (torch.stack([a.clone().detach() for a in step_accs]).mean()
                          if step_accs else torch.tensor(0.0))
            ans_accs   = step_accs[:-1] if len(step_accs) > 1 else step_accs
            answer_acc = (torch.stack([a.clone().detach() for a in ans_accs]).mean()
                          if ans_accs else avg_acc)
            halt_acc   = (sum(halt_accs) / len(halt_accs)) if halt_accs else 0.0
            return avg_loss, avg_acc, answer_acc, halt_acc

        # ── Inference ─────────────────────────────────────────────────────────
        else:
            trace: list[tuple] = []; last_answer = ""
            for loop_i in range(self.MAX_LOOPS):
                x = self._lifeline_inject(x, x_prompt)
                x = self.loop_rope(x, loop_i)
                for layer in self.all_layers[BASE_SPLIT:]:
                    x, residual = layer(x, residual)
                x = x + self.mamba2_core(x)
                x = self.loop_norm(x)
                lg  = self.lm_head(self.norm(x, residual, prenorm=False))
                p   = torch.softmax(lg[0, -1, :].float(), dim=-1)
                tid = p.argmax().item()
                tok = tokenizer.decode([tid]).strip()
                trace.append((f"L{loop_i+1}", tok, round(p[tid].item(), 4)))
                if tid == HALT_ID:
                    trace[-1] = (f"L{loop_i+1}", "<HALT>", round(p[tid].item(), 4))
                    return loop_i + 1, trace, last_answer
                last_answer = tok
            return self.MAX_LOOPS, trace, last_answer


# ── Data Pipeline ─────────────────────────────────────────────────────────────
def load_training_data() -> list[dict]:
    """Load training data: v3_curriculum > v2_clean > v33 > v32."""
    samples: list[dict] = []

    def _prep(text: str, tgt_strs: list[str]) -> dict | None:
        """Prepare a sample for training."""
        enc_ids   = tokenizer.encode(text, add_special_tokens=False)
        ans_start = find_answer_start(enc_ids)
        if ans_start < 1:
            return None
        ans_start = min(ans_start, len(enc_ids) - 1)
        tgt_ids: list[int] = []
        for ts in tgt_strs:
            if ts == "<HALT>":
                tgt_ids.append(HALT_ID); continue
            for pfx in (" ", ""):
                cands = tokenizer.encode(pfx + ts, add_special_tokens=False)
                if cands: tgt_ids.append(cands[0]); break
            else:
                tgt_ids.append(tokenizer.eos_token_id)
        return {
            "text": text, "hops": len(tgt_strs) - 1,
            "chain_targets": tgt_strs, "chain_tgt_ids": tgt_ids,
            "ans_start": ans_start,
        }

    for fname, label in [("system2_logic_v3_curriculum.json", "v3_curriculum"),
                          ("system2_logic_v2_clean.json", "v2_clean"),
                          ("system2_logic_v33.json", "v33"),
                          ("system2_logic_v32.json", "v32")]:
        if not os.path.exists(fname):
            continue
        with open(fname) as f:
            data = json.load(f)
        ok = 0
        for item in data:
            if "[Reasoning]" in item.get("answer", ""):
                ct = _parse_v2_chain(item["answer"])
                full_text = item["text"] + item["answer"]
                s = _prep(full_text, ct) if ct and len(ct) >= 2 else None
            elif item.get("type") == "override":
                ct = _parse_override(item)
                s = _prep(item["text"], ct) if ct and len(ct) >= 2 else None
            else:
                ct = _parse_chain(item["text"])
                s = _prep(item["text"], ct) if ct and len(ct) >= 2 else None
            if s:
                samples.append(s)
                ok += 1
        print(f"  {label}: {ok:,} samples loaded")
        if ok > 0:
            break

    hop_dist = {}
    for s in samples: hop_dist[s["hops"]] = hop_dist.get(s["hops"], 0) + 1
    print(f"  Total: {len(samples):,} | Hop dist: {dict(sorted(hop_dist.items()))}\n")
    return samples


def make_batch(pool: list[dict], seed: int) -> tuple:
    """Sample a padded batch."""
    batch = random.Random(seed).sample(pool, min(BATCH_SIZE, len(pool)))
    enc   = tokenizer(
        [s["text"] for s in batch],
        max_length=SEQ_LEN, truncation=True, padding="max_length", return_tensors="pt"
    )
    return (
        enc["input_ids"].to(DEVICE),
        [s["chain_tgt_ids"] for s in batch],
        [s["ans_start"]     for s in batch],
    )


# ── Load Model ────────────────────────────────────────────────────────────────
print(f"  Loading {MODEL_ID} (bfloat16)...", flush=True)
base_model = MambaLMHeadModel.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device=DEVICE)

new_vocab = len(tokenizer); old_vocab = base_model.backbone.embedding.weight.shape[0]
d_model   = base_model.backbone.embedding.embedding_dim
print(f"  d_model: {d_model}, n_layers: {len(base_model.backbone.layers)}")

if new_vocab > old_vocab:
    ne = nn.Embedding(new_vocab, d_model, dtype=torch.bfloat16)
    nn.init.normal_(ne.weight, std=0.02)
    ne.weight.data[:old_vocab] = base_model.backbone.embedding.weight.data
    base_model.backbone.embedding = ne
    # Handle tie_embeddings: lm_head shares weight with embedding
    # Create separate lm_head to expand vocab
    nh = nn.Linear(d_model, new_vocab, bias=False, dtype=torch.bfloat16)
    nn.init.normal_(nh.weight, std=0.02)
    nh.weight.data[:old_vocab] = base_model.lm_head.weight.data
    base_model.lm_head = nh

for p in base_model.parameters(): p.requires_grad = False
base_model.backbone.embedding.weight.requires_grad = True
base_model.lm_head.weight.requires_grad             = True

model = RecursiveMamba2_RLF(base_model, lora_rank=LORA_RANK).to(DEVICE)

if RESUME_FROM and os.path.exists(RESUME_FROM):
    ckpt = torch.load(RESUME_FROM, map_location=DEVICE)
    sd   = ckpt.get("model_state_dict", ckpt)
    own_sd   = model.state_dict()
    filtered = {k: v for k, v in sd.items()
                if k in own_sd and own_sd[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    skipped = len(sd) - len(filtered)
    print(f"  Warm-start: {len(filtered)} tensors from {RESUME_FROM} "
          f"({skipped} skipped)")
else:
    print("  Training from pretrained mamba2-2.7b base")

# ── Optimizer ─────────────────────────────────────────────────────────────────
samples = load_training_data()
if not samples:
    raise RuntimeError("No training data found.")

random.seed(42)
hop_buckets: dict[int, list] = {}
for s in samples:
    hop_buckets.setdefault(s["hops"], []).append(s)
train_samples: list[dict] = []; val_samples: list[dict] = []
for hop, bucket in sorted(hop_buckets.items()):
    random.shuffle(bucket)
    n_val = max(1, int(len(bucket) * 0.10))
    val_samples.extend(bucket[:n_val])
    train_samples.extend(bucket[n_val:])

print(f"  Train: {len(train_samples):,} | Val: {len(val_samples):,}")
hv = {}
for s in val_samples: hv[s["hops"]] = hv.get(s["hops"], 0) + 1
print(f"  Val hop dist: {dict(sorted(hv.items()))}\n")

own_ids   = set(id(p) for p in [model.lifeline_gate])
g_gate    = [model.lifeline_gate]
g_new     = (list(model.loop_norm.parameters())
             + list(model.mamba2_core.parameters())
             + [base_model.backbone.embedding.weight, base_model.lm_head.weight])
g_new_ids = {id(p) for p in g_gate + g_new}
g_lora    = [p for p in model.parameters()
             if p.requires_grad and id(p) not in g_new_ids]

optimizer = optim.AdamW([
    {"params": g_gate, "lr": 5e-4, "weight_decay": 0.0},
    {"params": g_new,  "lr": 5e-4, "weight_decay": 0.0},
    {"params": g_lora, "lr": 1e-4, "weight_decay": 0.01},
])
scheduler = CosineAnnealingLR(optimizer, T_max=STEPS // ACCUM, eta_min=1e-6)

vram_init = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0
print(f"{'─'*70}")
print(f"  Mamba2-2.7B RLF | RoPE | LoRA r={LORA_RANK} | {STEPS} steps")
print(f"  BS={BATCH_SIZE} × ACCUM={ACCUM} = effective {BATCH_SIZE*ACCUM}")
print(f"  VRAM after init: {vram_init:.2f}GB")
print(f"{'─'*70}\n")


def run_validation(val_pool: list, n_batches: int = 40) -> tuple:
    """Full-vocab validation."""
    vl = vaa = vfa = vha = 0.0; valid = 0
    with torch.no_grad():
        for i in range(n_batches):
            try:
                ids, ct, ans = make_batch(val_pool, seed=99999 + i)
                loss, aa, fa, ha = model(ids, chain_targets=ct, ans_starts=ans)
                if torch.isfinite(loss):
                    vl += loss.item(); vaa += aa.item()*100
                    vfa += fa.item()*100; vha += ha*100; valid += 1
            except Exception:
                pass
    if valid == 0: return float("inf"), 0.0, 0.0, 0.0
    return vl/valid, vaa/valid, vfa/valid, vha/valid


# ── Training Loop ─────────────────────────────────────────────────────────────
model.train(); optimizer.zero_grad()
t0 = time.time()
total_loss = total_aa = total_fa = total_ha = 0.0
early_stop_hits = 0; best_val_acc = 0.0; last_train_aa = 0.0

for step in range(1, STEPS + 1):
    for accum_i in range(ACCUM):
        ids, ct, ans = make_batch(train_samples, seed=step * ACCUM + accum_i)
        loss, aa, fa, ha = model(ids, chain_targets=ct, ans_starts=ans)
        (loss / ACCUM).backward()
        total_loss += loss.item(); total_aa += aa.item()*100
        total_fa   += fa.item()*100; total_ha += ha*100

    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
    optimizer.step(); scheduler.step(); optimizer.zero_grad()

    if step % LOG_EVERY == 0:
        n = LOG_EVERY * ACCUM
        al = total_loss/n; aa = total_aa/n; fa = total_fa/n; ha = total_ha/n
        last_train_aa = aa; total_loss = total_aa = total_fa = total_ha = 0.0
        vram = torch.cuda.memory_allocated()/1e9 if DEVICE == "cuda" else 0
        tps  = (BATCH_SIZE * ACCUM * SEQ_LEN * LOG_EVERY) / (time.time() - t0); t0 = time.time()
        g    = model.lifeline_gate.data
        print(
            f"  Step {step:5d} | Loss: {al:.4f} | AllLoop: {aa:5.1f}%"
            f" | Answer: {fa:5.1f}% | Halt: {ha:5.1f}%"
            f" | Gate μ={g.mean():.3f} σ={g.std():.4f}"
            f" | LR: {optimizer.param_groups[0]['lr']:.1e}"
            f" | TPS: {int(tps)} | VRAM: {vram:.2f}GB",
            flush=True)

    if step % VAL_EVERY == 0:
        vl, vaa, vfa, vha = run_validation(val_samples)
        gap  = last_train_aa - vaa
        flag = " ⚠️ OVERFIT" if gap > 10.0 else ""
        print(f"\n  ── VAL @ step {step} {'─'*44}", flush=True)
        print(f"  Val AllLoop: {vaa:5.1f}% | Answer: {vfa:5.1f}% | Halt: {vha:5.1f}%  [RoPE]", flush=True)
        print(f"  Train-Val gap: {gap:+.1f}pp{flag}", flush=True)

        if vaa > best_val_acc:
            best_val_acc = vaa; best_path = SAVE_PATH.replace(".pt", "_best.pt")
            torch.save({"model_state_dict": model.state_dict(),
                        "step": step, "val_allloop_acc": vaa,
                        "halt_id": HALT_ID, "no_mask": True,
                        "lifeline_gate": model.lifeline_gate.data.clone(),
                        "rope": True, "d_model": d_model,
                        "model_id": MODEL_ID}, best_path)
            print(f"  🏆 Best val {vaa:.1f}% → {best_path}", flush=True)

        if vaa >= EARLY_STOP_ACC:
            early_stop_hits += 1
            print(f"  ⏱  Early-stop: {early_stop_hits}/{EARLY_STOP_COUNT}  (val={vaa:.1f}%)", flush=True)
            if early_stop_hits >= EARLY_STOP_COUNT:
                print(f"\n  ✅ EARLY STOP @ step {step}. Val={vaa:.1f}%\n", flush=True)
                torch.save({"model_state_dict": model.state_dict(),
                            "step": step, "halt_id": HALT_ID, "no_mask": True,
                            "lifeline_gate": model.lifeline_gate.data.clone(),
                            "rope": True, "d_model": d_model,
                            "model_id": MODEL_ID}, SAVE_PATH)
                break
        else:
            early_stop_hits = 0
        print(f"  {'─'*60}\n", flush=True)

    if step % 1000 == 0:
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step, "halt_id": HALT_ID, "rope": True,
                    "lifeline_gate": model.lifeline_gate.data.clone(),
                    "d_model": d_model, "model_id": MODEL_ID},
                   SAVE_PATH.replace(".pt", f"_step{step}.pt"))
        print(f"  💾 step {step} checkpoint", flush=True)
else:
    torch.save({"model_state_dict": model.state_dict(), "step": STEPS,
                "halt_id": HALT_ID, "no_mask": True, "rope": True,
                "lifeline_gate": model.lifeline_gate.data.clone(),
                "d_model": d_model, "model_id": MODEL_ID}, SAVE_PATH)
    print(f"\n✅ 2.7B RLF complete — {SAVE_PATH}")


# ── OOD Length Test ───────────────────────────────────────────────────────────
model.eval()
print("="*70)
print("  Mamba2-2.7B OOD LENGTH TEST — trained 1-10 hops, testing 12/15/20")
print("="*70)

ood_tests = [
    ("P = blue. Q = P. R = Q. S = R. What is S?\nAnswer:",
     "blue", "4-hop (in-dist)"),
    ("A = democracy. B = A. C = B. D = C. E = D. F = E. What is F?\nAnswer:",
     "democracy", "6-hop (curriculum)"),
    ("P = algorithm. Q = P. R = Q. S = R. T = S. U = T. V = U. W = V. What is W?\nAnswer:",
     "algorithm", "8-hop (curriculum)"),
    ("X1=parliament. X2=X1. X3=X2. X4=X3. X5=X4. X6=X5. X7=X6. X8=X7. X9=X8. X10=X9. X11=X10. X12=X11. What is X12?\nAnswer:",
     "parliament", "12-hop (OOD)"),
    ("X1=quantum. X2=X1. X3=X2. X4=X3. X5=X4. X6=X5. X7=X6. X8=X7. X9=X8. X10=X9. X11=X10. X12=X11. X13=X12. X14=X13. X15=X14. What is X15?\nAnswer:",
     "quantum", "15-hop (OOD)"),
]

for prompt, expected, label in ood_tests:
    ids_ = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        n, trace, answer = model(ids_)
    hit = "✅ GENERALIZES" if expected.lower() in answer.lower() else f"❌ FAILS → {answer!r}"
    print(f"\n  {label}: {hit}")
    for lbl, tok, prob in trace:
        mark = " ← ✅" if tok.lower() == expected.lower() else (" <HALT>" if tok == "<HALT>" else "")
        print(f"    {lbl:5s}  {tok!r:14s}  p={prob:.4f}{mark}")

print("\n" + "="*70)
print("  If 12-hop works: 2.7B + RLF achieves true OOD generalization")
print("  Next: comprehensive_eval with knowledge Q&A tests")
print("="*70 + "\n")
