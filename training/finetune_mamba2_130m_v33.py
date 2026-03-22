"""
finetune_mamba2_130m_v33.py — The Final Run: Mamba2 + Latent Forcing + Vector Lifeline Gate
============================================================================================
Three precision upgrades from v32 analysis:

UPGRADE 1: Single-Token Vocabulary (Data Fix)
  v32 failure: "carburetor" → ['carb', 'uretor'] → model outputs 'carb' (correct first
  token, but graded as wrong). Data builder now enforces: every chain value tokenizes
  to EXACTLY ONE token. No ambiguity, no partial credit. 100% here = 100% real.

UPGRADE 2: Reality Override Counterfactuals (Gradient Fix)
  v32 failure: "Fire is icy cold" → 'Bob' (prior bias wins).
  Training had zero counterfactual gradient. v33 adds 3,000 procedurally generated
  examples: fire=icy, gravity=up, dogs→meow, etc. Now the model has an explicit
  gradient incentive to override pretrained priors with in-context logic.

UPGRADE 3: Float32 d_model-Vector Lifeline Gate (Architecture Fix + Interpretability)
  v32: scalar bfloat16 gate frozen at 1.0 (sub-precision floor, couldn't move).
  v33: d_model-dimensional float32 vector gate — one scalar per embedding dimension.
  Scientific value: after training, plot gate.weight — this is the mechanistic
  interpretability chart showing which embedding dimensions the model uses as RAM
  (high gate = copy from prompt) vs. ALU (low gate = protect Mamba state for routing).
  This is the single most interpretable artifact in the whole paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel, Mamba2
import json, random, time, os, re
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint as grad_ckpt

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID    = "state-spaces/mamba2-130m"
BASE_SPLIT  = 6
LORA_RANK   = 8
MAX_LOOPS   = 8
SEQ_LEN     = 256
BATCH_SIZE  = 8
ACCUM       = 4
STEPS       = 100_000
LOG_EVERY   = 50
VAL_EVERY   = 500
EARLY_STOP_ACC   = 95.0
EARLY_STOP_COUNT = 3
RESUME_FROM = "mamba2_130m_v32_lifeline_best.pt"   # warm-start from v32
SAVE_PATH   = "mamba2_130m_v33_final.pt"

LOOP_HEADDIM = 64
LOOP_D_STATE = 64

print(f"\n{'='*70}", flush=True)
print(f"  Mamba2-130m v33 — The Final Run", flush=True)
print(f"  FIX 1: single-token vocab — no BPE splits, no ambiguity", flush=True)
print(f"  FIX 2: reality override training — counterfactual gradient", flush=True)
print(f"  FIX 3: float32 d_model vector gate — RAM vs ALU interpretability", flush=True)
print(f"  Device: {DEVICE} | Steps={STEPS}", flush=True)
print(f"{'='*70}\n", flush=True)


# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>", "<HALT>"]})
HALT_ID = tokenizer.convert_tokens_to_ids("<HALT>")
print(f"  Vocab:    {len(tokenizer):,} | <HALT>: {HALT_ID}")


def _parse_chain(text: str) -> list[str] | None:
    """Extract per-loop targets + HALT from chain text."""
    assignments = re.findall(r'([A-Za-z_]\w*)\s*=\s*(\S+?)[\.\n]', text)
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


def _parse_override(sample: dict) -> list[str]:
    """Override samples: single direct answer then HALT."""
    return [sample["answer"], "<HALT>"]


def find_answer_start(ids: list[int]) -> int:
    """Find position after 'Answer:' — handles tail position (v33 format)."""
    for boundary in (
        tokenizer.encode("Answer:",  add_special_tokens=False),
        tokenizer.encode(" Answer:", add_special_tokens=False),
        tokenizer.encode("\nAnswer:", add_special_tokens=False),
    ):
        n = len(boundary)
        for i in range(len(ids) - n + 1):
            if ids[i:i + n] == boundary:
                pos = i + n
                return min(pos, len(ids) - 1)
    return -1


# ── LoRA Linear ───────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """LoRA adapter. lora_B init to zero → identity at warmup."""

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
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
        return self.base_weight + self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# ── Recursive Mamba2-130m v33 ─────────────────────────────────────────────────
class RecursiveMamba2_v33(nn.Module):
    """
    v33: Mamba2 + Latent Forcing + <HALT> + prompt lifeline.

    KEY CHANGE from v32: lifeline_gate is now a d_model-dimensional float32
    VECTOR instead of a scalar bfloat16. This enables:
      - Actual gradient updates (float32 precision >> bfloat16 floor)
      - Per-dimension RAM/ALU routing — the model learns WHICH embedding
        dimensions should copy from the prompt vs. be protected for pointer logic
      - Mechanistic interpretability: plotting gate.weight after training shows
        exactly how the model partitions its internal representation

    gate[d] ≈ 1.0 → dimension d carries data payload (RAM)
    gate[d] ≈ 0.0 → dimension d carries control flow (ALU/pointer state)
    gate[d]  > 1.0 → dimension d amplifies the prompt signal
    """

    MAX_LOOPS: int = MAX_LOOPS

    def __init__(self, backbone: MambaLMHeadModel, lora_rank: int = 8):
        """Init: freeze base, LoRA top, Mamba2 loop, float32 vector gate."""
        super().__init__()
        self.backbone   = backbone.backbone
        self.lm_head    = backbone.lm_head
        self.all_layers = nn.ModuleList(backbone.backbone.layers)
        self.norm       = backbone.backbone.norm_f
        d_model         = backbone.backbone.embedding.embedding_dim

        for layer in self.all_layers[:BASE_SPLIT]:
            for p in layer.parameters():
                p.requires_grad = False

        for layer in self.all_layers[BASE_SPLIT:]:
            mx = layer.mixer
            for attr in ("in_proj", "out_proj"):
                if hasattr(mx, attr):
                    setattr(mx, attr, LoRALinear(getattr(mx, attr),
                                                 rank=lora_rank,
                                                 alpha=lora_rank * 2.0))

        self.step_emb    = nn.Embedding(self.MAX_LOOPS, d_model).to(torch.bfloat16)
        self.loop_norm   = nn.RMSNorm(d_model).to(torch.bfloat16)
        self.mamba2_core = Mamba2(
            d_model    = d_model,
            d_state    = LOOP_D_STATE,
            d_conv     = 4,
            expand     = 2,
            headdim    = LOOP_HEADDIM,
            chunk_size = 64,
        ).to(torch.bfloat16)
        nn.init.zeros_(self.mamba2_core.out_proj.weight)

        # ── THE VECTOR GATE (float32, d_model-dimensional) ────────────────────
        # Previously: scalar bfloat16 → frozen at 1.0 (below precision floor)
        # Now: float32 vector → can move freely, one gate per embedding dimension
        # Init to 1.0: start with full lifeline injection, learn to modulate
        self.lifeline_gate = nn.Parameter(
            torch.ones(d_model, dtype=torch.float32)
        )
        self.d_model = d_model

        n_lora = sum(p.numel() for n, p in self.named_parameters()
                     if p.requires_grad and "lora" in n.lower())
        n_gate = d_model   # the main upgrade
        total  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  LoRA params:     {n_lora:,}")
        print(f"  Loop engine:     {sum(p.numel() for p in self.mamba2_core.parameters()):,}")
        print(f"  Lifeline gate:   {n_gate:,} floats (d_model vector, float32)")
        print(f"  Total trainable: {total:,}")
        print(f"  Pointer mask:    NONE (full {len(tokenizer):,}-token softmax)")
        print(f"  Lifeline:        x = x + gate_vec * x_prompt  (per-dimension)\n")

    def _lifeline_inject(self, x: torch.Tensor, x_prompt: torch.Tensor) -> torch.Tensor:
        """Apply per-dimension gate in float32, cast back to bf16."""
        gate = self.lifeline_gate.to(x.dtype)   # bf16 for the multiply
        return x + gate.unsqueeze(0).unsqueeze(0) * x_prompt

    def forward(
        self,
        input_ids:     torch.Tensor,
        chain_targets: list | None = None,
        ans_starts:    list | None = None,
    ) -> tuple:
        """Forward: base encode → save x_prompt → loop with vector lifeline."""
        x        = self.backbone.embedding(input_ids)
        residual = None
        for layer in self.all_layers:
            x, residual = layer(x, residual)

        # ── PROMPT LIFELINE: save uncorrupted base state before loops ─────────
        x_prompt = x.clone().detach()

        # ── Training ──────────────────────────────────────────────────────────
        if self.training and chain_targets is not None:
            B, max_len = input_ids.shape
            n_loops    = max(len(t) for t in chain_targets)

            def run_lora(x_in, res_in):
                """LoRA Mamba2 upper layers."""
                for layer in self.all_layers[BASE_SPLIT:]:
                    x_in, res_in = layer(x_in, res_in)
                return x_in, res_in

            step_losses: list[torch.Tensor] = []
            step_accs:   list[torch.Tensor] = []
            halt_accs:   list[float]        = []

            for loop_i in range(n_loops):
                x = self._lifeline_inject(x, x_prompt)
                sv = self.step_emb(
                    torch.tensor(min(loop_i, self.MAX_LOOPS - 1), device=x.device)
                )
                x = x + sv
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
                          if step_accs else torch.tensor(0.0, device=x.device))
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
                sv = self.step_emb(
                    torch.tensor(min(loop_i, self.MAX_LOOPS - 1), device=x.device)
                )
                x = x + sv
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
    """Load v33 single-token chains + overrides."""
    samples: list[dict] = []

    def _prep(text: str, tgt_strs: list[str]) -> dict | None:
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

    if os.path.exists("system2_logic_v33.json"):
        with open("system2_logic_v33.json") as f:
            v33_data = json.load(f)
        chain_ok = ovr_ok = 0
        for item in v33_data:
            text = item["text"]
            if item.get("type") == "override":
                ct = _parse_override(item)
            else:
                ct = _parse_chain(text)
                if not ct or len(ct) < 3:
                    continue
            s = _prep(text, ct)
            if s:
                samples.append(s)
                if item.get("type") == "override": ovr_ok += 1
                else: chain_ok += 1
        hop_dist = {}
        for s in samples:
            hop_dist[s["hops"]] = hop_dist.get(s["hops"], 0) + 1
        print(f"  v33 chains:    {chain_ok:,} | overrides: {ovr_ok:,}")
        print(f"  Hop dist:      {dict(sorted(hop_dist.items()))}")
    else:
        print(f"  ⚠️  system2_logic_v33.json not found — run v33_data_builder.py first")

    if os.path.exists("system2_logic_v32.json"):
        with open("system2_logic_v32.json") as f:
            v32 = json.load(f)
        v32_ok = 0
        for item in v32:
            if item.get("type") == "override": continue
            ct = _parse_chain(item["text"])
            if ct and len(ct) >= 3:
                s = _prep(item["text"], ct)
                if s: samples.append(s); v32_ok += 1
        print(f"  v32 chains:    {v32_ok:,} (supplement)")

    print(f"  Total:         {len(samples):,}\n")
    return samples


def make_batch(pool: list[dict], seed: int) -> tuple:
    """Sample padded batch."""
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

if new_vocab > old_vocab:
    ne = nn.Embedding(new_vocab, d_model, dtype=torch.bfloat16)
    nn.init.normal_(ne.weight, std=0.02)
    ne.weight.data[:old_vocab] = base_model.backbone.embedding.weight.data
    base_model.backbone.embedding = ne
    nh = nn.Linear(d_model, new_vocab, bias=False, dtype=torch.bfloat16)
    nn.init.normal_(nh.weight, std=0.02)
    nh.weight.data[:old_vocab] = base_model.lm_head.weight.data
    base_model.lm_head = nh

for p in base_model.parameters(): p.requires_grad = False
base_model.backbone.embedding.weight.requires_grad = True
base_model.lm_head.weight.requires_grad             = True

model = RecursiveMamba2_v33(base_model, lora_rank=LORA_RANK).to(DEVICE)

# Warm-start from v32 — compatible architecture (lifeline_gate shape differs: OK)
if RESUME_FROM and os.path.exists(RESUME_FROM):
    ckpt   = torch.load(RESUME_FROM, map_location=DEVICE)
    sd     = ckpt.get("model_state_dict", ckpt)
    own_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items()
                if k in own_sd and own_sd[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    skipped = len(sd) - len(filtered)
    print(f"  Warm-start: {len(filtered)} tensors from {RESUME_FROM} "
          f"({skipped} skipped — lifeline_gate shape changed: scalar→vector)")
else:
    print(f"  Training from pretrained mamba2-130m base")


# ── Optimizer ─────────────────────────────────────────────────────────────────
samples = load_training_data()
if not samples:
    raise RuntimeError("No training data. Run v33_data_builder.py first.")

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
hop_val = {}
for s in val_samples: hop_val[s["hops"]] = hop_val.get(s["hops"], 0) + 1
print(f"  Val hop dist: {dict(sorted(hop_val.items()))}\n")

# The vector gate is float32 — use higher precision group
g_gate   = [model.lifeline_gate]
g_new    = (list(model.step_emb.parameters())
            + list(model.loop_norm.parameters())
            + list(model.mamba2_core.parameters())
            + [base_model.backbone.embedding.weight, base_model.lm_head.weight])
g_new_ids = {id(p) for p in g_gate + g_new}
g_lora   = [p for p in model.parameters()
            if p.requires_grad and id(p) not in g_new_ids]

optimizer = optim.AdamW([
    {"params": g_gate, "lr": 5e-4, "weight_decay": 0.0},   # vector gate: moderate lr
    {"params": g_new,  "lr": 5e-4, "weight_decay": 0.0},   # new modules
    {"params": g_lora, "lr": 2e-4, "weight_decay": 0.01},  # LoRA: lower lr (warm-start)
])
scheduler = CosineAnnealingLR(optimizer, T_max=STEPS // ACCUM, eta_min=1e-6)

print(f"{'─'*70}")
print(f"  v33 | NO MASK | Vector Lifeline (d={d_model}) | {STEPS} steps")
print(f"  Data: single-token random vocab + reality overrides")
print(f"  Early stop: val ≥ {EARLY_STOP_ACC}% × {EARLY_STOP_COUNT}")
print(f"{'─'*70}\n")


def run_validation(val_pool: list[dict], n_batches: int = 40) -> tuple:
    """Full-vocab validation, no mask."""
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
        # Summarize gate: mean, std, min, max — the interpretability window
        g = model.lifeline_gate.data
        gate_stats = f"gate μ={g.mean():.3f} σ={g.std():.3f} [{g.min():.3f},{g.max():.3f}]"
        print(
            f"  Step {step:5d} | Loss: {al:.4f} | AllLoop: {aa:5.1f}%"
            f" | Answer: {fa:5.1f}% | Halt: {ha:5.1f}%"
            f" | {gate_stats}"
            f" | LR: {optimizer.param_groups[0]['lr']:.1e}"
            f" | TPS: {int(tps)} | VRAM: {vram:.2f}GB",
            flush=True)

    if step % VAL_EVERY == 0:
        vl, vaa, vfa, vha = run_validation(val_samples)
        gap  = last_train_aa - vaa
        flag = " ⚠️  OVERFIT" if gap > 10.0 else ""
        print(f"\n  ── VAL @ step {step} ─────────────────────────────────────────", flush=True)
        print(f"  Val AllLoop: {vaa:5.1f}% | Answer: {vfa:5.1f}% | Halt: {vha:5.1f}%  [full vocab]", flush=True)
        print(f"  Train-Val gap: {gap:+.1f}pp{flag}", flush=True)
        g = model.lifeline_gate.data
        print(f"  Gate vector:  μ={g.mean():.4f} σ={g.std():.4f} min={g.min():.4f} max={g.max():.4f}", flush=True)

        if vaa > best_val_acc:
            best_val_acc = vaa; best_path = SAVE_PATH.replace(".pt", "_best.pt")
            torch.save({"model_state_dict": model.state_dict(),
                        "step": step, "val_allloop_acc": vaa,
                        "halt_id": HALT_ID, "no_mask": True,
                        "lifeline_gate": model.lifeline_gate.data.clone(),
                        "d_model": d_model, "backbone": MODEL_ID}, best_path)
            print(f"  🏆 Best val {vaa:.1f}% → {best_path}", flush=True)

        if vaa >= EARLY_STOP_ACC:
            early_stop_hits += 1
            print(f"  ⏱  Early-stop: {early_stop_hits}/{EARLY_STOP_COUNT}  (val={vaa:.1f}%)", flush=True)
            if early_stop_hits >= EARLY_STOP_COUNT:
                print(f"\n  ✅ EARLY STOP @ step {step}. Val={vaa:.1f}%", flush=True)
                torch.save({"model_state_dict": model.state_dict(),
                            "step": step, "halt_id": HALT_ID, "no_mask": True,
                            "lifeline_gate": model.lifeline_gate.data.clone()}, SAVE_PATH)
                break
        else:
            early_stop_hits = 0
        print(f"  {'─'*60}\n", flush=True)

    if step % 2000 == 0:
        ckpt = SAVE_PATH.replace(".pt", f"_step{step}.pt")
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step, "halt_id": HALT_ID,
                    "lifeline_gate": model.lifeline_gate.data.clone()}, ckpt)
        print(f"  💾 {ckpt}", flush=True)
else:
    torch.save({"model_state_dict": model.state_dict(), "step": STEPS,
                "halt_id": HALT_ID, "no_mask": True,
                "lifeline_gate": model.lifeline_gate.data.clone()}, SAVE_PATH)
    print(f"\n✅ v33 complete — {SAVE_PATH}\n")


# ── Quick Inference Test ───────────────────────────────────────────────────────
model.eval()
tests = [
    ("A = democracy. B = A. What is B?\nAnswer:",                         "democracy", "1-hop"),
    ("X = algorithm. Y = X. Z = Y. What is Z?\nAnswer:",                 "algorithm", "3-hop"),
    ("A = phosphorus. B = A. C = B. D = C. What is D?\nAnswer:",         "phosphorus","4-hop"),
    ("Fire is icy cold. Bob touched fire. What did Bob feel?\nAnswer:",   "cold",      "override"),
    ("In this world dogs meow. Alice has a dog. What sound?\nAnswer:",    "meow",      "override"),
]
print("=" * 70)
print("  v33 Inference — vector lifeline, single-token vocab, full 50k")
print("=" * 70)
with torch.no_grad():
    for prompt, expected, label in tests:
        ids_ = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
        loops, trace, answer = model(ids_)
        chain_str = " → ".join(t[1] for t in trace)
        match = "✅" if expected.lower() in answer.lower() else "❌"
        print(f"  {match} [{label}] {prompt.splitlines()[0][:55]!r}")
        print(f"     {chain_str} | Answer: {answer!r}")

print("\n  Gate vector analysis (top 5 RAM dims, bottom 5 ALU dims):")
gate = model.lifeline_gate.data.cpu()
top5   = gate.topk(5)
bot5   = gate.topk(5, largest=False)
print(f"  Highest (RAM):  {list(zip(top5.indices.tolist(), [f'{v:.4f}' for v in top5.values.tolist()]))}")
print(f"  Lowest  (ALU):  {list(zip(bot5.indices.tolist(), [f'{v:.4f}' for v in bot5.values.tolist()]))}")
print("=" * 70)
