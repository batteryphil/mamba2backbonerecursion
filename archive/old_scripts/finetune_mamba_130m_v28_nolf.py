"""
finetune_mamba_130m_v28_nolf.py — Ablation: No Latent Forcing (NoLF)
=====================================================================
IDENTICAL architecture to finetune_mamba_130m_v28.py.

THE ONE DIFFERENCE:
  v28 (Latent Forcing):   loss computed at EVERY loop against chain_targets[loop_i]
  v28_nolf (this file):   loss computed ONLY at the FINAL loop against chain_targets[-1]

This is the proper scientific control. If v28 shows pointer movement in Phase 2
but v28_nolf shows the same token at every loop (like the frozen BASE), then
Latent Forcing is definitively proven to cause the observed behavior.

If BOTH models show pointer movement, the claim collapses (architecture does it).
If NEITHER shows pointer movement, both training methods fail.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel, Mamba
import json
import random
import time
import os
import re
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint as grad_ckpt

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID    = "state-spaces/mamba-130m"
BASE_SPLIT  = 6
LORA_RANK   = 8
MAX_LOOPS   = 6
SEQ_LEN     = 256
BATCH_SIZE  = 8
ACCUM       = 4
STEPS       = 50_000
LOG_EVERY   = 50
VAL_EVERY   = 500
EARLY_STOP_ACC   = 99.5
EARLY_STOP_COUNT = 3
SAVE_PATH   = "mamba_130m_v28_nolf.pt"  # NoLF checkpoint

print(f"\n{'='*60}")
print(f"  Mamba-130m v28-NoLF — ABLATION (final-answer supervision only)")
print(f"  NO intermediate loop supervision — only final loop loss")
print(f"  Device: {DEVICE} | Steps={STEPS}")
print(f"{'='*60}\n")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>"]})


def _parse_chain(text: str) -> list[str] | None:
    """Extract per-loop targets from variable assignment chain."""
    assignments = re.findall(r'([A-Za-z_]\w*)\s*=\s*(\S+?)[\.\n]', text)
    if len(assignments) < 2:
        return None
    val: dict[str, str] = {}
    for var, expr in assignments:
        val[var] = expr
    chain_vars = [assignments[0][0]]
    for var, _ in assignments[1:]:
        chain_vars.append(var)
    targets: list[str] = []
    for i in range(len(chain_vars) - 1):
        targets.append(chain_vars[i])
    final_var = chain_vars[-1]
    resolved  = val.get(final_var, final_var)
    visited: set[str] = set()
    while resolved in val and resolved not in visited:
        visited.add(resolved)
        resolved = val[resolved]
    targets.append(resolved)
    return targets if len(targets) >= 2 else None


def find_answer_start(ids: list[int]) -> int:
    """First token position after 'Answer:' boundary."""
    for boundary in (
        tokenizer.encode("Answer:",  add_special_tokens=False),
        tokenizer.encode(" Answer:", add_special_tokens=False),
    ):
        n = len(boundary)
        for i in range(len(ids) - n):
            if ids[i:i + n] == boundary:
                return i + n
    return -1


class LoRALinear(nn.Module):
    """LoRA adapter — same as v28."""

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        """Initialize LoRA with zero lora_B for near-identity warmup."""
        super().__init__()
        self.bias  = linear.bias
        d_out, d_in = linear.weight.shape
        dtype = linear.weight.dtype
        self.register_buffer("base_weight", linear.weight.data.clone())
        self.lora_A = nn.Parameter(torch.empty(rank, d_in, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, dtype=dtype))
        self.scale  = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A)

    @property
    def weight(self) -> torch.Tensor:
        """Fused LoRA weight."""
        return self.base_weight + self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear forward."""
        return F.linear(x, self.weight, self.bias)


class RecursiveMamba130m_NoLF(nn.Module):
    """
    IDENTICAL architecture to RecursiveMamba130m_v28.
    THE ONLY DIFFERENCE: loss is computed at the final loop only
    (chain_targets[-1]), not at every loop.

    This is the ablation control for the Latent Forcing experiment.
    """

    MAX_LOOPS: int = MAX_LOOPS

    def __init__(self, backbone: MambaLMHeadModel, lora_rank: int = 8):
        """Same init as v28 — identical architecture."""
        super().__init__()
        self.backbone    = backbone.backbone
        self.lm_head     = backbone.lm_head
        self.all_layers  = nn.ModuleList(backbone.backbone.layers)
        self.norm        = backbone.backbone.norm_f
        d_model          = backbone.backbone.embedding.embedding_dim

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
        self.mamba3_core = Mamba(
            d_model=d_model, d_state=16, d_conv=4, expand=2
        ).to(torch.bfloat16)
        nn.init.zeros_(self.mamba3_core.out_proj.weight)

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  NoLF Total trainable: {total:,}")

    def forward(
        self,
        input_ids:     torch.Tensor,
        chain_targets: list | None = None,
        ans_starts:    list | None = None,
    ) -> tuple:
        """
        Training: compute loss at FINAL LOOP ONLY (no intermediate supervision).
        Inference: same as v28 — run MAX_LOOPS, halt on token repeat.
        """
        x        = self.backbone.embedding(input_ids)
        residual = None

        for layer in self.all_layers:
            x, residual = layer(x, residual)

        # ── Training path ──────────────────────────────────────────────────────
        if self.training and chain_targets is not None:
            B, max_len = input_ids.shape
            n_loops    = max(len(t) for t in chain_targets)

            def run_lora_layers(x_in, res_in):
                """Re-run LoRA top layers per loop."""
                for layer in self.all_layers[BASE_SPLIT:]:
                    x_in, res_in = layer(x_in, res_in)
                return x_in, res_in

            # Run ALL loops (same as v28) — but only compute loss on the last one
            for loop_i in range(n_loops):
                step_vec = self.step_emb(
                    torch.tensor(loop_i, device=x.device)
                )
                x = x + step_vec
                x, residual = grad_ckpt(run_lora_layers, x, residual,
                                        use_reentrant=False)
                x = x + self.mamba3_core(x)
                x = self.loop_norm(x)

            # ── FINAL LOOP ONLY: compute loss ──────────────────────────────────
            # (This is the key difference from v28 — no intermediate loop losses)
            x_normed    = self.norm(x, residual, prenorm=False)
            logits_final = self.lm_head(x_normed)
            vocab_size   = logits_final.shape[-1]

            total_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
            total_acc  = torch.tensor(0.0, device=x.device)
            valid      = 0

            for b in range(B):
                as_ = ans_starts[b]
                if as_ < 1 or as_ >= max_len:
                    continue
                # Use FINAL target only (chain_targets[b][-1])
                btgt   = chain_targets[b]
                tgt_id = int(btgt[-1])

                uniq = torch.unique(input_ids[b])
                mask = torch.full((vocab_size,), float("-inf"), device=x.device)
                mask[uniq] = 0.0

                if tgt_id >= vocab_size or mask[tgt_id].item() == float("-inf"):
                    continue

                logits_b = logits_final[b, as_ - 1, :] + mask
                pred_tok = logits_b.argmax().item()
                tgt_t    = torch.tensor(tgt_id, device=x.device)
                total_loss = total_loss + F.cross_entropy(
                    logits_b.unsqueeze(0), tgt_t.unsqueeze(0)
                )
                total_acc = total_acc + float(pred_tok == tgt_id)
                valid    += 1

            if valid > 0:
                avg_loss = total_loss / valid
                avg_acc  = total_acc  / valid
            else:
                avg_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
                avg_acc  = torch.tensor(0.0, device=x.device)

            # Return same shape as v28 for compatible training loop
            return avg_loss, avg_acc, avg_acc

        # ── Inference path (identical to v28) ──────────────────────────────────
        else:
            vocab_size = self.lm_head.weight.shape[0]
            mask = torch.full((vocab_size,), float("-inf"), device=x.device)
            mask[torch.unique(input_ids[0])] = 0.0

            trace    = []
            prev_tok = None

            for loop_i in range(self.MAX_LOOPS):
                sv = self.step_emb(torch.tensor(loop_i, device=x.device))
                x  = x + sv
                for layer in self.all_layers[BASE_SPLIT:]:
                    x, residual = layer(x, residual)
                x = x + self.mamba3_core(x)
                x = self.loop_norm(x)

                lg  = self.lm_head(self.norm(x, residual, prenorm=False))
                lg[0, -1, :] += mask
                p   = torch.softmax(lg[0, -1, :], dim=-1)
                tid = p.argmax().item()
                tok = tokenizer.decode([tid]).strip()
                trace.append((f"L{loop_i+1}", tok, round(p.max().item(), 3)))

                if prev_tok is not None and tid == prev_tok:
                    return loop_i + 1, trace
                prev_tok = tid

            return self.MAX_LOOPS, trace


# ── Data Pipeline (identical to v28) ─────────────────────────────────────────
def load_training_data() -> list[dict]:
    """Load same data as v28 — identical pre-tokenized pipeline."""
    samples: list[dict] = []

    def _extract_answer(text: str) -> str | None:
        m = re.search(r'[Aa]nswer:\s*(\S+)', text)
        return m.group(1).rstrip('.,!?') if m else None

    def _prep(text: str, chain_targets: list[str]) -> dict | None:
        enc      = tokenizer.encode(text, add_special_tokens=False)
        enc_set  = set(enc)
        ans_start = find_answer_start(enc)
        if ans_start < 1:
            return None
        tgt_ids: list[int] = []
        for tgt_str in chain_targets:
            matched = None
            for pfx in (" ", ""):
                cands = tokenizer.encode(pfx + tgt_str, add_special_tokens=False)
                if cands and cands[0] in enc_set:
                    matched = cands[0]
                    break
            if matched is None:
                cands   = tokenizer.encode(" " + tgt_str, add_special_tokens=False)
                matched = cands[0] if cands else tokenizer.eos_token_id
            tgt_ids.append(matched)
        return {
            "text":          text,
            "hops":          len(chain_targets),
            "chain_targets": chain_targets,
            "chain_tgt_ids": tgt_ids,
            "ans_start":     ans_start,
        }

    if os.path.exists("system2_logic_v1.json"):
        with open("system2_logic_v1.json") as f:
            data = json.load(f)
        for item in data:
            text = item.get("text", item.get("prompt", ""))
            if not text:
                continue
            chain_tgts = _parse_chain(text)
            if chain_tgts and len(chain_tgts) >= 2:
                s = _prep(text, chain_tgts)
                if s:
                    samples.append(s)
            else:
                ans = _extract_answer(text)
                if ans:
                    s = _prep(text, [ans])
                    if s:
                        samples.append(s)

    if os.path.exists("mmlu_format_v17.json"):
        with open("mmlu_format_v17.json") as f:
            mmlu = json.load(f)
        for item in mmlu[:5_000]:
            text = item.get("text", "")
            ans  = _extract_answer(text)
            if ans:
                s = _prep(text, [ans])
                if s:
                    samples.append(s)

    for cf in [
        ("The sun is freezing cold. John's coffee is the temperature of the sun. What temperature is John's coffee?\nAnswer: cold",  ["cold"]),
        ("Fire is icy cold. Bob touched fire. What did Bob feel?\nAnswer: cold",                                                     ["cold"]),
        ("In this world dogs meow and cats bark. Sarah has a cat. What sound does she hear?\nAnswer: bark",                          ["bark"]),
        ("'heavy' means weighs nothing. A rock is heavy. How much does it weigh?\nAnswer: nothing",                                 ["nothing"]),
    ]:
        s = _prep(cf[0], cf[1])
        if s:
            samples.append(s)

    hop_dist: dict = {}
    for s in samples:
        hop_dist[s["hops"]] = hop_dist.get(s["hops"], 0) + 1
    print(f"  Hop dist: {dict(sorted(hop_dist.items()))}")
    print(f"  Total:    {len(samples):,}\n")
    return samples


def make_batch(pool: list[dict], seed: int) -> tuple:
    """Same batch creation as v28."""
    batch = random.Random(seed).sample(pool, min(BATCH_SIZE, len(pool)))
    enc   = tokenizer(
        [s["text"] for s in batch],
        max_length=SEQ_LEN, truncation=True,
        padding="max_length", return_tensors="pt"
    )
    return (
        enc["input_ids"].to(DEVICE),
        [s["chain_tgt_ids"] for s in batch],
        [s["ans_start"]     for s in batch],
    )


# ── Load Model ────────────────────────────────────────────────────────────────
print(f"  Loading {MODEL_ID} (bfloat16)...", flush=True)
base_model = MambaLMHeadModel.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device=DEVICE
)

new_vocab = len(tokenizer)
old_vocab = base_model.backbone.embedding.weight.shape[0]
d_model   = base_model.backbone.embedding.embedding_dim

if new_vocab > old_vocab:
    new_emb = nn.Embedding(new_vocab, d_model, dtype=torch.bfloat16)
    nn.init.normal_(new_emb.weight, std=0.02)
    new_emb.weight.data[:old_vocab] = base_model.backbone.embedding.weight.data
    base_model.backbone.embedding   = new_emb
    new_head = nn.Linear(d_model, new_vocab, bias=False, dtype=torch.bfloat16)
    nn.init.normal_(new_head.weight, std=0.02)
    new_head.weight.data[:old_vocab] = base_model.lm_head.weight.data
    base_model.lm_head = new_head

for p in base_model.parameters():
    p.requires_grad = False
base_model.backbone.embedding.weight.requires_grad = True
base_model.lm_head.weight.requires_grad             = True

model = RecursiveMamba130m_NoLF(base_model, lora_rank=LORA_RANK).to(DEVICE)

# ── Optimizer + Data ─────────────────────────────────────────────────────────
samples = load_training_data()

random.seed(42)
train_samples: list[dict] = []
val_samples:   list[dict] = []
hop_buckets: dict[int, list] = {}
for s in samples:
    hop_buckets.setdefault(s["hops"], []).append(s)
for hop, bucket in sorted(hop_buckets.items()):
    random.shuffle(bucket)
    n_val = max(1, int(len(bucket) * 0.10))
    val_samples.extend(bucket[:n_val])
    train_samples.extend(bucket[n_val:])

print(f"  Train: {len(train_samples):,} | Val: {len(val_samples):,}")

g1_params = (
    [model.step_emb.weight]
    + list(model.loop_norm.parameters())
    + list(model.mamba3_core.parameters())
    + [base_model.backbone.embedding.weight, base_model.lm_head.weight]
)
g1_ids    = {id(p) for p in g1_params}
g2_params = [p for p in model.parameters()
             if p.requires_grad and id(p) not in g1_ids]

optimizer = optim.AdamW([
    {"params": g1_params, "lr": 1e-3,  "weight_decay": 0.0},
    {"params": g2_params, "lr": 5e-4,  "weight_decay": 0.01},
])
scheduler = CosineAnnealingLR(optimizer, T_max=STEPS // ACCUM, eta_min=1e-6)

print(f"\n{'─'*60}")
print(f"  130m v28-NoLF | {STEPS} steps | eff batch={BATCH_SIZE*ACCUM}")
print(f"  ABLATION: loss ONLY at final loop (no intermediate supervision)")
print(f"{'─'*60}\n")


def run_validation(val_pool: list[dict], n_batches: int = 30) -> tuple:
    """Val pass in train mode so forward() uses loss branch."""
    v_loss = v_aa = v_fa = 0.0
    valid  = 0
    with torch.no_grad():
        for i in range(n_batches):
            try:
                ids, ctgts, ans = make_batch(val_pool, seed=99999 + i)
                loss, aa, fa    = model(ids, chain_targets=ctgts, ans_starts=ans)
                if torch.isfinite(loss):
                    v_loss += loss.item()
                    v_aa   += aa.item() * 100
                    v_fa   += fa.item() * 100
                    valid  += 1
            except Exception:
                pass
    if valid == 0:
        return float("inf"), 0.0, 0.0
    return v_loss / valid, v_aa / valid, v_fa / valid


# ── Training Loop ─────────────────────────────────────────────────────────────
model.train()
optimizer.zero_grad()
t0 = time.time()
total_loss = total_avg_acc = total_final_acc = 0.0
early_stop_hits = 0
best_val_acc    = 0.0
last_train_aa   = 0.0

for step in range(1, STEPS + 1):
    for accum_i in range(ACCUM):
        ids, ctgts, ans = make_batch(train_samples, seed=step * ACCUM + accum_i)
        loss, aa, fa    = model(ids, chain_targets=ctgts, ans_starts=ans)
        (loss / ACCUM).backward()
        total_loss      += loss.item()
        total_avg_acc   += aa.item() * 100
        total_final_acc += fa.item() * 100

    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0
    )
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    if step % LOG_EVERY == 0:
        n  = LOG_EVERY * ACCUM
        al = total_loss / n
        aa = total_avg_acc / n
        fa = total_final_acc / n
        last_train_aa = aa
        total_loss = total_avg_acc = total_final_acc = 0.0
        vram = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0
        tps  = (BATCH_SIZE * ACCUM * SEQ_LEN * LOG_EVERY) / (time.time() - t0)
        t0   = time.time()
        print(
            f"  Step {step:5d} | Loss: {al:.4f} | FinalAcc: {aa:5.1f}%"
            f" | LR: {optimizer.param_groups[0]['lr']:.1e}"
            f" | TPS: {int(tps)} | VRAM: {vram:.2f}GB",
            flush=True,
        )

    if step % VAL_EVERY == 0:
        vl, vaa, vfa = run_validation(val_samples)
        gap          = last_train_aa - vaa
        flag         = " ⚠️  OVERFIT" if gap > 5.0 else ""
        print(f"\n  ── VAL @ step {step} ───────────────────────────────────", flush=True)
        print(f"  Val Loss: {vl:.4f} | Val FinalAcc: {vaa:5.1f}%  Train-Val gap: {gap:+.1f}pp{flag}", flush=True)

        if vaa > best_val_acc:
            best_val_acc = vaa
            best_path    = SAVE_PATH.replace(".pt", "_best.pt")
            torch.save({"model_state_dict": model.state_dict(),
                        "step": step, "val_final_acc": vaa}, best_path)
            print(f"  🏆 Best val {vaa:.1f}% → {best_path}", flush=True)

        if vaa >= EARLY_STOP_ACC:
            early_stop_hits += 1
            print(f"  ⏱  Early-stop: {early_stop_hits}/{EARLY_STOP_COUNT}  (val={vaa:.1f}%)", flush=True)
            if early_stop_hits >= EARLY_STOP_COUNT:
                print(f"\n  ✅ EARLY STOP @ step {step}. Val FinalAcc={vaa:.1f}%", flush=True)
                torch.save({"model_state_dict": model.state_dict(), "step": step}, SAVE_PATH)
                break
        else:
            early_stop_hits = 0
        print(f"  {'─'*52}\n", flush=True)

    if step % 500 == 0:
        ckpt_path = SAVE_PATH.replace(".pt", f"_step{step}.pt")
        torch.save({
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
        }, ckpt_path)
        print(f"  💾 {ckpt_path}", flush=True)

else:
    torch.save({"model_state_dict": model.state_dict(), "step": STEPS}, SAVE_PATH)
    print(f"\n✅ NoLF ablation complete — {SAVE_PATH}\n")
