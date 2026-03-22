"""
v34_trajectory_plot.py — Internal State Trajectory Visualization
Extracts hidden-state vectors at the answer position across reasoning loops,
projects to 2D via PCA, and overlays Lifeline=1.0 vs Lifeline=0.0 trajectories.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.decomposition import PCA
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel, Mamba2

# ── Inline model classes (self-contained) ─────────────────────────

class LoopRoPE(nn.Module):
    """1D Rotary Position Embeddings for loop index."""
    def __init__(self, d_model: int, base: int = 10000):
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, loop_index: int) -> torch.Tensor:
        """Apply RoPE rotation to hidden state."""
        n = torch.tensor(float(loop_index), device=x.device)
        freqs = n * self.inv_freq.to(device=x.device, dtype=torch.float32)
        emb = torch.stack([freqs, freqs], dim=-1).flatten()[:self.d_model]
        cos_v, sin_v = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rot = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return x * cos_v + rot * sin_v

class LoRALinear(nn.Module):
    """Low-rank adapter for linear layers."""
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.bias = linear.bias
        self.register_buffer("base_weight", linear.weight.data.clone())
        self.lora_A = nn.Parameter(torch.empty(rank, linear.weight.shape[1], dtype=linear.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(linear.weight.shape[0], rank, dtype=linear.weight.dtype))
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A)

    @property
    def weight(self):
        """Compute effective weight."""
        return self.base_weight + self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA."""
        return F.linear(x, self.weight, self.bias)

class RecursiveMamba2_v34(nn.Module):
    """v34 Recursive Mamba2 with Prompt Lifeline + RoPE."""
    def __init__(self, backbone: MambaLMHeadModel, lora_rank: int = 8):
        super().__init__()
        self.backbone = backbone.backbone
        self.lm_head = backbone.lm_head
        self.all_layers = nn.ModuleList(self.backbone.layers)
        self.norm = self.backbone.norm_f
        d_model = self.backbone.embedding.embedding_dim
        for layer in self.all_layers[6:]:
            for attr in ("in_proj", "out_proj"):
                if hasattr(layer.mixer, attr):
                    setattr(layer.mixer, attr, LoRALinear(getattr(layer.mixer, attr), rank=lora_rank, alpha=lora_rank * 2.0))
        self.loop_rope = LoopRoPE(d_model)
        self.loop_norm = nn.RMSNorm(d_model).to(torch.bfloat16)
        self.mamba2_core = Mamba2(d_model=d_model, d_state=64, d_conv=4, expand=2, headdim=64, chunk_size=64).to(torch.bfloat16)
        self.lifeline_gate = nn.Parameter(torch.ones(d_model, dtype=torch.float32))

    def _lifeline_inject(self, x: torch.Tensor, x_prompt: torch.Tensor) -> torch.Tensor:
        """Inject prompt via learned gate."""
        return x + self.lifeline_gate.to(x.dtype).unsqueeze(0).unsqueeze(0) * x_prompt

# ── Setup ─────────────────────────────────────────────────────────

DEVICE = "cuda"
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>", "<HALT>"]})
HALT_ID = tokenizer.convert_tokens_to_ids("<HALT>")

def find_answer_start(ids: list[int]) -> int:
    """Find the token position right after 'Answer:'."""
    for boundary in (tokenizer.encode("Answer:", add_special_tokens=False),
                     tokenizer.encode(" Answer:", add_special_tokens=False),
                     tokenizer.encode("\nAnswer:", add_special_tokens=False)):
        n = len(boundary)
        for i in range(len(ids) - n + 1):
            if ids[i:i + n] == boundary:
                return min(i + n, len(ids) - 1)
    return -1

print("Loading model...")
base_model = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-130m", dtype=torch.bfloat16, device=DEVICE)
new_vocab = len(tokenizer)
old_vocab = base_model.backbone.embedding.weight.shape[0]
d_model = base_model.backbone.embedding.embedding_dim

if new_vocab > old_vocab:
    ne = nn.Embedding(new_vocab, d_model, dtype=torch.bfloat16)
    ne.weight.data[:old_vocab] = base_model.backbone.embedding.weight.data
    base_model.backbone.embedding = ne
    nh = nn.Linear(d_model, new_vocab, bias=False, dtype=torch.bfloat16)
    nh.weight.data[:old_vocab] = base_model.lm_head.weight.data
    base_model.lm_head = nh

model = RecursiveMamba2_v34(base_model, lora_rank=8).to(DEVICE)
ckpt = torch.load("mamba2_130m_v34_rope_best.pt", map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ── Trajectory extraction ─────────────────────────────────────────

def extract_trajectory(gate_scale: float = 1.0, n_loops: int = 10) -> dict:
    """Run inference and capture hidden state + predictions at each loop."""
    prompt = "P = algorithm. Q = P. R = Q. S = R. T = S. U = T. V = U. W = V. What is W?\nAnswer:"
    ids_ = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    ans_start = find_answer_start(ids_[0].tolist())

    states = []
    tokens = []
    logit_margins = []

    with torch.no_grad():
        x = model.backbone.embedding(ids_)
        residual = None
        for layer in model.all_layers[:6]:
            x, residual = layer(x, residual)
        x_prompt = x.clone().detach()

        for loop_i in range(n_loops):
            # Inject lifeline with scaled gate
            x = x + (model.lifeline_gate * gate_scale).to(x.dtype).unsqueeze(0).unsqueeze(0) * x_prompt
            x = model.loop_rope(x, loop_i)

            for layer in model.all_layers[6:]:
                x, residual = layer(x, residual)
            x = x + model.mamba2_core(x)
            x = model.loop_norm(x)

            # Extract state vector at answer position
            state_vec = x[0, ans_start - 1, :].float().cpu().numpy()
            states.append(state_vec)

            # Get prediction
            logits = model.lm_head(model.norm(x, residual, prenorm=False))[0, ans_start - 1, :]
            probs = torch.softmax(logits.float(), dim=-1)
            top2 = probs.topk(2)
            top_id = top2.indices[0].item()
            margin = (top2.values[0] - top2.values[1]).item()

            tok = tokenizer.decode([top_id]).strip()
            if top_id == HALT_ID:
                tok = "⏹"
            tokens.append(tok)
            logit_margins.append(margin)

    return {"states": np.array(states), "tokens": tokens, "margins": logit_margins}

# ── Run both conditions ───────────────────────────────────────────

print("Extracting trajectory: Lifeline = 1.0 ...")
traj_on  = extract_trajectory(gate_scale=1.0, n_loops=10)
print("Extracting trajectory: Lifeline = 0.0 ...")
traj_off = extract_trajectory(gate_scale=0.0, n_loops=10)

# ── PCA projection ────────────────────────────────────────────────

all_states = np.vstack([traj_on["states"], traj_off["states"]])
pca = PCA(n_components=2)
projected = pca.fit_transform(all_states)
n = len(traj_on["states"])
proj_on  = projected[:n]
proj_off = projected[n:]

# ── Plotting ──────────────────────────────────────────────────────

plt.style.use("dark_background")
fig, ax = plt.subplots(1, 1, figsize=(14, 9))

# Background gradient
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

# Color maps
cmap_on  = plt.cm.plasma
cmap_off = plt.cm.cool

# Plot trajectories with arrows
for i in range(n - 1):
    t = i / (n - 1)
    # Lifeline ON
    ax.annotate("", xy=(proj_on[i+1, 0], proj_on[i+1, 1]),
                xytext=(proj_on[i, 0], proj_on[i, 1]),
                arrowprops=dict(arrowstyle="-|>", color=cmap_on(t),
                                lw=2.5, mutation_scale=15))
    # Lifeline OFF
    ax.annotate("", xy=(proj_off[i+1, 0], proj_off[i+1, 1]),
                xytext=(proj_off[i, 0], proj_off[i, 1]),
                arrowprops=dict(arrowstyle="-|>", color=cmap_off(t),
                                lw=2.0, mutation_scale=12, linestyle="--"))

# Plot points with token labels
for i in range(n):
    t = i / (n - 1)
    # Lifeline ON point
    ax.scatter(proj_on[i, 0], proj_on[i, 1], s=180, c=[cmap_on(t)],
               edgecolors="white", linewidths=1.5, zorder=5)
    ax.annotate(f"L{i+1}: {traj_on['tokens'][i]}",
                (proj_on[i, 0], proj_on[i, 1]),
                textcoords="offset points", xytext=(12, 8),
                fontsize=9, fontweight="bold", color=cmap_on(t),
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    # Lifeline OFF point
    ax.scatter(proj_off[i, 0], proj_off[i, 1], s=120, c=[cmap_off(t)],
               edgecolors="white", linewidths=1.0, zorder=5, marker="D")
    ax.annotate(f"{traj_off['tokens'][i]}",
                (proj_off[i, 0], proj_off[i, 1]),
                textcoords="offset points", xytext=(-8, -14),
                fontsize=8, color=cmap_off(t), fontstyle="italic",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=cmap_on(0.5), lw=3, label="Lifeline = 1.0 (full gate)"),
    Line2D([0], [0], color=cmap_off(0.5), lw=2, linestyle="--", marker="D",
           markersize=6, label="Lifeline = 0.0 (gate severed)"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=12,
          facecolor="#161b22", edgecolor="#30363d", labelcolor="white")

# Labels
ax.set_xlabel("PCA Component 1", fontsize=13, color="#8b949e")
ax.set_ylabel("PCA Component 2", fontsize=13, color="#8b949e")
ax.set_title("Internal State Trajectory: Training-Inference Phase Transition\n"
             "The Mamba2 core executes the same FSM whether or not the Prompt Lifeline is connected",
             fontsize=14, fontweight="bold", color="white", pad=20)

# Explained variance annotation
var1, var2 = pca.explained_variance_ratio_
ax.text(0.98, 0.02, f"PCA var: {var1:.1%} + {var2:.1%} = {var1+var2:.1%}",
        transform=ax.transAxes, fontsize=9, color="#8b949e",
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22", edgecolor="#30363d"))

# Subtitle with prompt
ax.text(0.5, -0.08,
        "Prompt: P = algorithm. Q = P. R = Q. S = R. T = S. U = T. V = U. W = V. What is W?",
        transform=ax.transAxes, fontsize=10, color="#8b949e", ha="center",
        fontstyle="italic")

# Grid
ax.grid(True, alpha=0.15, color="#30363d")
ax.tick_params(colors="#8b949e")
for spine in ax.spines.values():
    spine.set_color("#30363d")

plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "..", "phase_transition_trajectory.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved: {out_path}")

# Also print the comparison table
print("\n" + "="*70)
print("TRAJECTORY COMPARISON: Lifeline=1.0 vs Lifeline=0.0")
print("="*70)
print(f"{'Loop':<6} {'Gate=1.0':<12} {'Gate=0.0':<12} {'Match':<8} {'Margin(1.0)':<12} {'Margin(0.0)':<12}")
print("-"*70)
for i in range(n):
    match = "✅" if traj_on["tokens"][i] == traj_off["tokens"][i] else "❌"
    print(f"L{i+1:<5} {traj_on['tokens'][i]:<12} {traj_off['tokens'][i]:<12} {match:<8} {traj_on['margins'][i]:<12.3f} {traj_off['margins'][i]:<12.3f}")
print("="*70)
