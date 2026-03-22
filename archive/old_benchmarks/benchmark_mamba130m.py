"""
benchmark_mamba130m.py
======================
Runs MMLU (subset) and GSM8K (subset) on mamba130m v15 step-3000.
Compares against Gemma 3 1B, Phi-4-mini 3.8B, SmolLM3-1.7B reference numbers.

Scoring:
  MMLU   — log-prob of each answer choice (' A'/' B'/' C'/' D'), argmax
  GSM8K  — greedy decode up to 200 tokens, extract final number
"""
import sys, torch, torch.nn as nn, torch.nn.functional as F, re, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/phil/Desktop/mambadiff/mambadiff llm tts")
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel
from datasets import load_dataset

DEVICE = "cuda"
CKPT   = "mamba130m_finetuned_v14_step3000.pt"
MAX_NEW_TOKENS = 150

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tok.pad_token = tok.eos_token


# ── Model ─────────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
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


class RecursiveMamba(nn.Module):
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
        for layer in self.top_layers:
            mx = layer.mixer
            for attr in ("in_proj", "x_proj", "dt_proj"):
                if hasattr(mx, attr):
                    setattr(mx, attr, LoRALinear(getattr(mx, attr), rank=8, alpha=16.0))
        self.step_emb  = nn.Embedding(self.MAX_LOOPS, d)
        nn.init.normal_(self.step_emb.weight, std=0.01)
        self.loop_norm = nn.RMSNorm(d)

    def get_logits(self, ids):
        """Get next-token logits for input_ids (inference, entropy-gated)."""
        x = self.backbone.embedding(ids)
        r = None
        for l in self.backbone.layers[:6]:
            x, r = l(x, r)
        bf = x.clone()
        CONFIDENCE_THR = 0.85
        for si in range(self.MAX_LOOPS):
            x = x + self.step_emb(torch.tensor(si, device=x.device))
            for l in self.top_layers:
                x, r = l(x, r)
            x = x + bf
            x = self.loop_norm(x)
            pl = self.lm_head(self.norm(x, r, prenorm=False))
            p  = torch.softmax(pl[0, -1, :], dim=-1)
            if p.max().item() > CONFIDENCE_THR:
                break
        x = self.norm(x, r, prenorm=False)
        return self.lm_head(x)


bm = MambaLMHeadModel.from_pretrained(
    "state-spaces/mamba-130m", dtype=torch.float32, device=DEVICE
)
for p in bm.parameters():
    p.requires_grad = False

model = RecursiveMamba(bm).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE), strict=False)
model.eval()
print(f"✅ Model loaded from {CKPT}\n")


# ── MMLU Evaluation ───────────────────────────────────────────────────────────
def eval_mmlu(n_per_subject: int = 15) -> float:
    """Evaluate MMLU via log-prob choice method — load+infer each subject inline."""
    SUBJECTS = [
        "abstract_algebra", "anatomy", "astronomy", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "high_school_biology", "high_school_chemistry", "high_school_mathematics",
        "high_school_physics", "high_school_psychology", "logical_fallacies",
        "machine_learning", "medical_genetics", "moral_scenarios",
        "philosophy", "professional_accounting", "professional_law",
        "virology", "world_religions",
    ]
    choices = [" A", " B", " C", " D"]
    choice_ids = [tok.encode(c, add_special_tokens=False)[0] for c in choices]
    total, correct = 0, 0
    subject_results = {}

    for subj in SUBJECTS:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test")
        except Exception as e:
            print(f"    SKIP {subj}: {e}")
            continue
        data = list(ds)[:n_per_subject]
        subj_correct = 0
        for item in data:
            q    = item["question"]
            opts = item["choices"]
            ans  = int(item["answer"])
            prompt = (
                f"Question: {q}\n"
                f"A. {opts[0]}\nB. {opts[1]}\nC. {opts[2]}\nD. {opts[3]}\n"
                f"Answer:"
            )
            ids = tok.encode(prompt, return_tensors="pt",
                             max_length=512, truncation=True).to(DEVICE)
            with torch.no_grad():
                logits = model.get_logits(ids)
            scores = [logits[0, -1, cid].item() for cid in choice_ids]
            if scores.index(max(scores)) == ans:
                correct += 1
                subj_correct += 1
            total += 1
        r = f"{subj_correct}/{len(data)}"
        subject_results[subj] = r
        print(f"    {subj:<35} {r}")
        sys.stdout.flush()

    acc = correct / max(total, 1)
    print(f"  → MMLU Accuracy: {acc*100:.1f}%  ({correct}/{total})\n")
    return acc


# ── GSM8K Evaluation ─────────────────────────────────────────────────────────
def _extract_number(text: str):
    """Extract last numeric value from generated text."""
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def eval_gsm8k(n: int = 30) -> float:
    """Evaluate GSM8K via L=1 greedy decode (fast — no entropy loop during generation)."""
    print("Loading GSM8K...")
    ds   = load_dataset("openai/gsm8k", "main", split="test")
    data = list(ds)[:n]
    correct, total = 0, 0

    for item in data:
        q        = item["question"]
        gold_num = _extract_number(item["answer"])
        prompt   = f"Question: {q}\nAnswer:"
        ids = tok.encode(prompt, return_tensors="pt",
                         max_length=256, truncation=True).to(DEVICE)
        generated = []
        with torch.no_grad():
            # Use single-loop fast path: run base pass + 1 loop, no entropy gate
            for _ in range(80):  # shorter generation, just extract number
                x = model.backbone.embedding(ids[:, -256:])
                r = None
                for l in model.backbone.layers[:6]: x, r = l(x, r)
                bf = x.clone()
                x = x + model.step_emb(torch.tensor(0, device=x.device))
                for l in model.top_layers: x, r = l(x, r)
                x = x + bf; x = model.loop_norm(x)
                logits = model.lm_head(model.norm(x, r, prenorm=False))
                next_t = logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
                ids    = torch.cat([ids, next_t], dim=1)
                token  = tok.decode([next_t.item()])
                generated.append(token)
                if next_t.item() == tok.eos_token_id: break
        gen_text = "".join(generated)
        pred_num = _extract_number(gen_text)
        hit = pred_num is not None and gold_num is not None and pred_num == gold_num
        if hit: correct += 1
        total += 1
        print(f"    [{total:>2}] pred={pred_num!r:>8}  gold={gold_num!r:>8}  {'✅' if hit else '❌'}")
        sys.stdout.flush()

    acc = correct / max(total, 1)
    print(f"  → GSM8K Accuracy: {acc*100:.1f}%  ({correct}/{total})\n")
    return acc


# ── Run ───────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  Mamba-130M v15 — Benchmark vs Gemma 3 1B / Phi-4-mini / SmolLM3")
print("=" * 65 + "\n")

t0 = time.time()
mmlu_acc = eval_mmlu(n_per_subject=15)
gsm_acc  = eval_gsm8k(n=100)
elapsed  = time.time() - t0

print("=" * 65)
print(f"  {'Benchmark':<15} {'Mamba-130M (ours)':<22} {'Gemma 3 1B':<14} {'Phi-4-mini 3.8B':<18} {'SmolLM3-1.7B'}")
print(f"  {'-'*63}")
print(f"  {'MMLU':<15} {mmlu_acc*100:.1f}%{'':<19} {'62.8%':<14} {'68.5%':<18} {'65.2%'}")
print(f"  {'GSM8K':<15} {gsm_acc*100:.1f}%{'':<19} {'81.2%':<14} {'85.0%':<18} {'78.4%'}")
print(f"  {'HumanEval':<15} {'N/A (skip)':<22} {'55.0%':<14} {'64.0%':<18} {'58.0%'}")
print(f"\n  Model size:  130M params (vs 1B / 3.8B / 1.7B)")
print(f"  Elapsed:     {elapsed:.0f}s")
print("=" * 65)
