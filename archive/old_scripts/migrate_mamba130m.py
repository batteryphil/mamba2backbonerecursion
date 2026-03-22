"""
migrate_mamba130m.py — Weight Graft: mamba-130m → RBM Dual-Path + 2 Memory Layers

Architecture of the resulting model:
  - d_model    = 768  (matches mamba-130m exactly)
  - n_layers   = 24   (all 24 pretrained mamba-130m blocks)
  - n_memory_layers = 2  (new high-d_state blocks appended, random init)
  - n_reasoning = 3   (N=3 recursive passes)

Weight mapping per layer:
  mamba-130m mixer.*         →  Our DualCausalMambaBlock
  ─────────────────────────────────────────────────────
  mixer.in_proj.weight       →  in_proj.weight       (shared by both paths)
  mixer.out_proj.weight      →  out_proj.weight      (shared)
  mixer.conv1d.weight/bias   →  a_conv1d.*  AND  b_conv1d.*  (copied to both)
  mixer.x_proj.weight        →  a_x_proj.*  AND  b_x_proj.*
  mixer.dt_proj.weight/bias  →  a_dt_proj.* AND  b_dt_proj.*
  mixer.A_log                →  a_A_log     AND  b_A_log
  mixer.D                    →  a_D         AND  b_D
  layer.norm.weight/bias     →  layer norm in our ModuleDict

Result: Path A = pretrained knowledge. Path B = copy of Path A (same start,
diverges during logic fine-tuning). Memory layers = random init, high d_state.
"""
import torch
import torch.nn as nn
from mamba_ssm import MambaLMHeadModel
from transformers import GPT2Tokenizer
from mamba_rbm import RecursiveMambaLM, Config
import copy

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
OUT_PATH = "rbm_grafted_checkpoint.pt"

SEP = "=" * 60
print(f"\n{SEP}\n  RBM Weight Graft: mamba-130m → Dual-Path + Memory\n{SEP}\n")


# ── Step 1: Load source model ──────────────────────────────────────────────
print("📦 Loading state-spaces/mamba-130m...")
src = MambaLMHeadModel.from_pretrained(
    "state-spaces/mamba-130m", dtype=torch.float32
).cpu()
src.eval()

src_layers   = src.backbone.layers
n_src_layers = len(src_layers)
src_d_model  = src.backbone.embedding.weight.shape[1]
src_vocab    = src.backbone.embedding.weight.shape[0]

print(f"  Source: {n_src_layers} layers | d_model={src_d_model} | vocab={src_vocab}")


# ── Step 2: Build target RBM architecture ─────────────────────────────────
# Use the SAME tokenizer as mamba-130m pretraining to ensure embedding alignment
from transformers import AutoTokenizer as _AutoTok
tokenizer = _AutoTok.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token

config = Config(
    vocab_size      = len(tokenizer),   # neox vocab = 50280 (matches mamba-130m exactly)
    d_model         = src_d_model,      # 768
    n_layers        = n_src_layers,     # 24
    seq_len         = 1024,
    n_reasoning     = 3,
    n_memory_layers = 2,
    memory_d_state  = 32,
)

print(f"\n  Target RBM config:")
print(f"    d_model={config.d_model} | n_layers={config.n_layers} | memory_layers={config.n_memory_layers}")
print(f"    memory_d_state={config.memory_d_state} | n_reasoning={config.n_reasoning}")

tgt = RecursiveMambaLM(config).cpu()


# ── Step 3: Graft embedding weight ────────────────────────────────────────
print(f"\n{'─'*50}")
print("  Grafting embedding layer...")
# mamba-130m uses gpt-neox-20b tokenizer (50277 tokens) — our model uses GPT2 (50257)
# We take the first 50257 rows, skipping the last 20 mamba-130m tokens
src_emb  = src.backbone.embedding.weight.data   # (50277, 768)
tgt_rows = config.vocab_size                     # 50257

tgt.token_embed.weight.data[:tgt_rows] = src_emb[:tgt_rows].clone()
print(f"    Copied embedding: {src_emb.shape} → {tgt.token_embed.weight.data.shape} (trimmed {src_emb.shape[0]-tgt_rows} tokens)")


# ── Step 4: Graft each of the 24 pretrained layers ────────────────────────
print(f"\n  Grafting {n_src_layers} pretrained layers into dual-path blocks...")
grafted, skipped = 0, 0

for i, src_layer in enumerate(src_layers):
    tgt_layer  = tgt.layers[i]
    tgt_mamba  = tgt_layer["mamba"]
    src_mixer  = getattr(src_layer, "mixer", None)
    src_norm   = getattr(src_layer, "norm", None)

    if src_mixer is None:
        print(f"    Layer {i:02d} SKIPPED — no mixer found")
        skipped += 1
        continue

    try:
        # ── Shared projections (in_proj, out_proj) ────────────────────
        tgt_mamba.in_proj.weight.data.copy_(src_mixer.in_proj.weight.data)
        tgt_mamba.out_proj.weight.data.copy_(src_mixer.out_proj.weight.data)

        # ── Path A ← pretrained mixer ─────────────────────────────────
        tgt_mamba.a_conv1d.weight.data.copy_(src_mixer.conv1d.weight.data)
        tgt_mamba.a_conv1d.bias.data.copy_(src_mixer.conv1d.bias.data)
        tgt_mamba.a_x_proj.weight.data.copy_(src_mixer.x_proj.weight.data)
        tgt_mamba.a_dt_proj.weight.data.copy_(src_mixer.dt_proj.weight.data)
        tgt_mamba.a_dt_proj.bias.data.copy_(src_mixer.dt_proj.bias.data)
        tgt_mamba.a_A_log.data.copy_(src_mixer.A_log.data)
        tgt_mamba.a_D.data.copy_(src_mixer.D.data)

        # ── Path B ← copy of Path A (same knowledge, will diverge during fine-tune) ──
        tgt_mamba.b_conv1d.weight.data.copy_(src_mixer.conv1d.weight.data)
        tgt_mamba.b_conv1d.bias.data.copy_(src_mixer.conv1d.bias.data)
        tgt_mamba.b_x_proj.weight.data.copy_(src_mixer.x_proj.weight.data)
        tgt_mamba.b_dt_proj.weight.data.copy_(src_mixer.dt_proj.weight.data)
        tgt_mamba.b_dt_proj.bias.data.copy_(src_mixer.dt_proj.bias.data)
        tgt_mamba.b_A_log.data.copy_(src_mixer.A_log.data)
        tgt_mamba.b_D.data.copy_(src_mixer.D.data)

        # ── Layer norm weight/bias (RMSNorm has no bias — skip gracefully) ──
        tgt_layer["norm"].weight.data.copy_(src_layer.norm.weight.data)
        if src_layer.norm.bias is not None and tgt_layer["norm"].bias is not None:
            tgt_layer["norm"].bias.data.copy_(src_layer.norm.bias.data)

        grafted += 1
        if i % 6 == 0:
            print(f"    Layer {i:02d}/{n_src_layers} grafted ✓")

    except Exception as e:
        import traceback
        print(f"    Layer {i:02d} FAILED: {e}")
        traceback.print_exc()
        skipped += 1

print(f"\n  Grafted: {grafted}/{n_src_layers} layers  |  Skipped: {skipped}")


# ── Step 5: Copy final norm from mamba-130m ────────────────────────────────
try:
    tgt.final_norm.weight.data.copy_(src.backbone.norm_f.weight.data)
    if getattr(src.backbone.norm_f, "bias", None) is not None:
        tgt.final_norm.bias.data.copy_(src.backbone.norm_f.bias.data)
    print("  Final LayerNorm grafted ✓")
except Exception as e:
    print(f"  Final LayerNorm FAILED: {e}")


# ── Step 6: Memory layers are already randomly initialized in __init__ ──────
print(f"\n  Memory consolidation layers (x{config.n_memory_layers}): randomly initialized")
print(f"    d_state={config.memory_d_state} (2x standard)  — will learn long-range retention")


# ── Step 7: Save the grafted checkpoint ────────────────────────────────────
stats = {
    "model_state":     tgt.state_dict(),
    "step":            0,
    "n_reasoning":     config.n_reasoning,
    "graft_source":    "state-spaces/mamba-130m",
    "n_layers":        config.n_layers,
    "n_memory_layers": config.n_memory_layers,
    "d_model":         config.d_model,
    "memory_d_state":  config.memory_d_state,
}
torch.save(stats, OUT_PATH)
print(f"\n{'─'*50}")
print(f"  Checkpoint saved: {OUT_PATH}")

total_params = sum(p.numel() for p in tgt.parameters())
trainable    = sum(p.numel() for p in tgt.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Memory layers add: {sum(p.numel() for ml in tgt.memory_layers for p in ml.parameters()):,} new params")
print(f"\n{'='*60}")
print("  GRAFT COMPLETE — ready to fine-tune with train_hybrid.py")
print(f"  Next step: python3 train_hybrid.py --logic_data logic_v4.json --qa_data qa_anchors.json")
print(f"{'='*60}\n")
