"""probe_v15.py — Single-token entropy probe for v15 checkpoint."""
import sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/phil/Desktop/mambadiff/mambadiff llm tts")
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel
import warnings; warnings.filterwarnings("ignore")

DEVICE = "cuda"
CKPT   = "/home/phil/Desktop/mambadiff/mambadiff llm tts/mamba130m_finetuned_v14_step500.pt"
MAX_LOOPS = 10
THR = 2.0

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tok.pad_token = tok.eos_token


class LoRALinear(nn.Module):
    """LoRA wrapper matching training script exactly."""
    def __init__(self, linear, rank=8, alpha=16.0):
        super().__init__()
        self.bias = linear.bias
        d_out, d_in = linear.weight.shape
        self.register_buffer("base_weight", linear.weight.data.clone())
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.scale  = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A)

    @property
    def weight(self):
        return self.base_weight + self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class RecursiveProbe(nn.Module):
    """Probe model matching v15 inference path exactly."""
    MAX_LOOPS = 10

    def __init__(self, bm):
        super().__init__()
        self.backbone   = bm.backbone
        self.lm_head    = bm.lm_head
        self.top_layers = nn.ModuleList(
            [bm.backbone.layers[i] for i in range(6, len(bm.backbone.layers))]
        )
        self.norm = bm.backbone.norm_f
        d = bm.backbone.embedding.embedding_dim
        # Inject LoRA
        for layer in self.top_layers:
            mx = layer.mixer
            for attr in ("in_proj", "x_proj", "dt_proj"):
                if hasattr(mx, attr):
                    setattr(mx, attr, LoRALinear(getattr(mx, attr), rank=8, alpha=16.0))
        self.step_emb = nn.Embedding(MAX_LOOPS, d)
        nn.init.normal_(self.step_emb.weight, std=0.01)
        self.loop_norm = nn.RMSNorm(d)

    def forward(self, ids):
        x = self.backbone.embedding(ids)
        r = None
        for l in self.backbone.layers[:6]:
            x, r = l(x, r)
        bf = x.clone()
        trace = []
        for si in range(MAX_LOOPS):
            x = x + self.step_emb(torch.tensor(si, device=x.device))
            for l in self.top_layers:
                x, r = l(x, r)
            x = x + bf
            x = self.loop_norm(x)
            pl = self.lm_head(self.norm(x, r, prenorm=False))
            p2 = torch.softmax(pl[0, -1, :], dim=-1)
            e2 = -(p2 * (p2 + 1e-12).log()).sum().item()
            top1 = tok.decode([p2.argmax().item()]).strip()
            trace.append((f"L{si+1}", round(e2, 2), top1))
            if e2 < THR:
                break
        x = self.norm(x, r, prenorm=False)
        return self.lm_head(x), si + 1, trace


# Load
bm = MambaLMHeadModel.from_pretrained(
    "state-spaces/mamba-130m", dtype=torch.float32, device=DEVICE
)
for p in bm.parameters():
    p.requires_grad = False

model = RecursiveProbe(bm).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE), strict=False)
model.eval()

CASES = [
    ("2-var easy",  "Alice is taller than Bob. Who is shorter? Answer:",         "Bob"),
    ("passive",     "Bob is outweighed by Alice. Who weighs more? Answer:",      "Alice"),
    ("negation",    "Yuki is not taller than Sven. Who is taller? Answer:",      "Sven"),
    ("3-var",       "X > Y > Z height. Who is shortest? Answer:",                "Z"),
    ("3-var mid",   "X > Y > Z height. Who is second? Answer:",                  "Y"),
    ("4-var",       "Red>Blue>Green>Yellow speed. Who is slowest? Answer:",      "Yellow"),
    ("OOD",         "Zorblax outweighs Quibble. Who is lighter? Answer:",        "Quibble"),
    ("inversion",   "A is shorter than B. Who is tallest? Answer:",              "B"),
    ("QA extract",  "Context: [Code is Omega-9.] Question: code? Answer:",       "Omega"),
    ("multi-hop",   "Context: [Alice code=Delta-3. Bob uses Alice's code.] Q: Bob's code? Answer:", "Delta"),
    ("bool-yes",    "Is gold heavier than feathers? Answer:",                    "Yes"),
    ("bool-no",     "Is a mouse larger than an elephant? Answer:",               "No"),
    ("geography",   "What is the capital of France? Answer:",                    "Paris"),
    ("science",     "What is the powerhouse of the cell? Answer:",               "Mitochondria"),
]

print(f"\n{'='*72}")
print(f"  v15 Deep Dive | Step 500 | Dual LR | LoRA rank-8 | RMSNorm | Stoch.Depth")
print(f"{'='*72}")
print(f"  {'Label':<14} L    Entropy curve")
print(f"  {'─'*70}")

correct = 0
for label, prompt, expected in CASES:
    ids = tok.encode(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits, loops, trace = model(ids[:, -256:])
    top1 = tok.decode([logits[0, -1, :].argmax().item()]).strip()
    hit  = expected.lower() in top1.lower()
    if hit:
        correct += 1
    parts = [f"{e}({t!r})" for _, e, t in trace[:5]]
    curve = " → ".join(parts) + ("..." if len(trace) > 5 else "")
    mono  = "📉" if (len(trace) > 1 and
                     all(trace[i][1] >= trace[i+1][1]
                         for i in range(min(len(trace)-1, 4)))) else "〰️"
    print(f"  {'✅' if hit else '❌'} {label:<13} L={loops:<3} {mono} {curve}")

print(f"\n  Top-1 accuracy: {correct}/{len(CASES)} ({correct/len(CASES)*100:.0f}%)")
print(f"\n  step_emb norms:")
for i in range(MAX_LOOPS):
    print(f"    [{i}] norm={model.step_emb.weight[i].norm():.3f}")
print(f"  loop_norm weight mean={model.loop_norm.weight.mean():.4f}")
print(f"{'='*72}\n")
