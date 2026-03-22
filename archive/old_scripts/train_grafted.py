"""
train_grafted.py v3 — Back to the working RecursiveMamba130M wrapper.

The full DualCausalMambaBlock graft was broken: random Path B weights
poisoned frozen Path A activations at every layer, producing garbage even
before training.

This version returns to the RecursiveMamba130M top-layer wrapper which
worked (loss 0.18 in 2000 steps), but fixes the mode collapse from v1:
  1. LR: 1e-4 → 5e-5
  2. 70% WikiText-2 generic text + 30% logic/QA per batch
  3. Entropy regularisation (coefficient 0.02)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel
from datasets import load_dataset
import json, random, time, os

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_OUT    = "mamba130m_finetuned_v3.pt"
LOGIC_FILE  = "logic_v5.json"
QA_FILE     = "qa_anchors.json"
STEPS       = 5000
LR          = 5e-5
SEQ_LEN     = 256
BATCH_SIZE  = 4
ACCUM       = 8
LOG_EVERY   = 10
SAVE_EVERY  = 500
ENTROPY_REG = 0.02
GENERIC_MIX = 0.70
N_REASON    = 3

print(f"\n{'='*60}")
print(f"  RecursiveMamba130M Anti-Collapse Fine-Tune v3")
print(f"  LR={LR} | EntropyReg={ENTROPY_REG} | GenericMix={GENERIC_MIX}")
print(f"{'='*60}\n")


# ── Tokenizer ─────────────────────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tok.pad_token = tok.eos_token


# ── Recursive wrapper (same as working v1) ────────────────────────────────────
class RecursiveMamba130M(nn.Module):
    """Top-2-layer recursive wrapper over frozen mamba-130m backbone."""

    def __init__(self, base_model, n_reasoning=3):
        """Wrap pretrained mamba-130m with N recursive passes over top 2 layers."""
        super().__init__()
        self.backbone    = base_model.backbone
        self.lm_head     = base_model.lm_head
        self.n_reasoning = n_reasoning
        self.top_layers  = nn.ModuleList(
            [self.backbone.layers[i] for i in range(22, len(self.backbone.layers))]
        )
        self.norm    = self.backbone.norm_f
        self.dropout = nn.Dropout(0.05)

    def forward(self, input_ids):
        """Embed → frozen lower layers → N recursive top-layer passes → head."""
        x = self.backbone.embedding(input_ids)
        residual = None
        for layer in self.backbone.layers[:22]:
            x, residual = layer(x, residual)
        for _ in range(self.n_reasoning):
            loop_residual = residual
            for layer in self.top_layers:
                x, residual = layer(x, residual)
            if self.training:
                x = self.dropout(x)
            residual = residual + loop_residual if loop_residual is not None else residual
        x = self.norm(x, residual, prenorm=False)
        return self.lm_head(x)


# ── Load model ────────────────────────────────────────────────────────────────
print("Loading state-spaces/mamba-130m...")
base = MambaLMHeadModel.from_pretrained(
    "state-spaces/mamba-130m", dtype=torch.float32, device=DEVICE
)

# Freeze bottom 22 layers — only top 2 are tunable
for i, layer in enumerate(base.backbone.layers):
    if i < 22:
        for p in layer.parameters():
            p.requires_grad = False

model = RecursiveMamba130M(base, n_reasoning=N_REASON).to(DEVICE)
tunable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Tunable params: {tunable:,} (top 2 layers + head)")
model.train()


# ── Data ──────────────────────────────────────────────────────────────────────
print("\nLoading WikiText-2...")
wiki          = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
generic_texts = [t for t in wiki["text"] if len(t.strip()) > 80]
print(f"  WikiText-2:  {len(generic_texts):,} samples")

logic_samples = []
if os.path.exists(LOGIC_FILE):
    for item in json.load(open(LOGIC_FILE)):
        txt = item.get("text") or item.get("prompt", "") if isinstance(item, dict) else str(item)
        if txt: logic_samples.append(txt)
if os.path.exists(QA_FILE):
    for item in json.load(open(QA_FILE)):
        if isinstance(item, dict):
            c,q,a = item.get("context",""), item.get("question",""), item.get("answer","")
            if c and q and a:
                logic_samples.append(f"Context: [{c}] Question: {q} Answer: {a}")
print(f"  Logic/QA:    {len(logic_samples):,} samples\n")


def get_batch():
    """Mixed batch: GENERIC_MIX generic, rest logic/QA."""
    texts = [random.choice(generic_texts) if random.random() < GENERIC_MIX
             else random.choice(logic_samples) for _ in range(BATCH_SIZE)]
    enc = tok(texts, max_length=SEQ_LEN, truncation=True,
               padding="max_length", return_tensors="pt")
    return enc["input_ids"].to(DEVICE)


# ── Optimizer ─────────────────────────────────────────────────────────────────
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=0.01
)

print(f"{'─'*60}")
print(f"  Training {STEPS} steps | batch={BATCH_SIZE*ACCUM} | LR={LR}")
print(f"{'─'*60}")

start_time   = time.time()
running_loss = 0.0
optimizer.zero_grad()

for step in range(1, STEPS + 1):
    ids           = get_batch()
    logits        = model(ids)
    shift_logits  = logits[:, :-1, :].contiguous()
    shift_labels  = ids[:, 1:].contiguous()

    ce_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=tok.pad_token_id or -100
    )
    probs   = F.softmax(shift_logits, dim=-1)
    entropy = -(probs * (probs + 1e-9).log()).sum(-1).mean()
    loss    = ce_loss - ENTROPY_REG * entropy

    (loss / ACCUM).backward()
    running_loss += ce_loss.item()

    if step % ACCUM == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    if step % LOG_EVERY == 0:
        elapsed = time.time() - start_time
        tps     = (step * BATCH_SIZE * SEQ_LEN) / elapsed
        vram    = torch.cuda.memory_reserved() / 1e9 if DEVICE == "cuda" else 0.0
        avg     = running_loss / LOG_EVERY
        print(f"  Step {step:>5} | Loss: {avg:.4f} | TPS: {tps:.0f} | VRAM: {vram:.2f}GB")
        running_loss = 0.0
        import json as _j
        with open("training_stats.json", "w") as f:
            _j.dump({"step": step, "loss": avg, "lr": LR, "tps": tps,
                     "train_loss": [avg], "val_loss": [], "epoch": 1,
                     "diverged": avg > 15.0, "avg_bias": 0.0, "cpu_temp": "N/A",
                     "socratic_results": [], "experiment": "mamba130m-v3"}, f)

    if step % SAVE_EVERY == 0:
        torch.save({"model_state": model.state_dict(), "step": step,
                    "n_reasoning": N_REASON}, CKPT_OUT)
        print(f"  💾 Checkpoint at Step {step}")

torch.save({"model_state": model.state_dict(), "step": STEPS,
            "n_reasoning": N_REASON}, CKPT_OUT)
print(f"\n✅ Complete! Saved: {CKPT_OUT}")
