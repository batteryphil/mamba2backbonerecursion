"""test_v27.py — standalone inference test for mamba2_13b_finetuned_v27.pt"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import MambaLMHeadModel
from transformers import AutoTokenizer

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CKPT        = "mamba2_13b_finetuned_v27.pt"
MODEL_ID    = "state-spaces/mamba2-1.3b"
BASE_SPLIT  = 24
LORA_RANK   = 8

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>"]})
THINK_TOKEN_ID  = tokenizer.convert_tokens_to_ids("<THINK>")
ALLOWED_CORE    = [tokenizer.eos_token_id, THINK_TOKEN_ID]


class LoRALinear(nn.Module):
    def __init__(self, linear, rank=8, alpha=16.0):
        super().__init__()
        self.bias = linear.bias
        d_out, d_in = linear.weight.shape
        dtype = linear.weight.dtype
        self.register_buffer("base_weight", linear.weight.data.clone())
        self.lora_A = nn.Parameter(torch.empty(rank, d_in, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, dtype=dtype))
        self.scale  = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A)

    @property
    def weight(self):
        return self.base_weight + self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


import torch.jit
@torch.jit.script
def fused_mimo(x_in, r_s, i_s, cos_t, sin_t, Br, Bi, Cr, Ci):
    Br=Br.unsqueeze(0).unsqueeze(0); Bi=Bi.unsqueeze(0).unsqueeze(0)
    Cr=Cr.unsqueeze(0).unsqueeze(0); Ci=Ci.unsqueeze(0).unsqueeze(0)
    bxr=Br*x_in; bxi=Bi*x_in
    nr=(cos_t*r_s-sin_t*i_s)+bxr; ni=(sin_t*r_s+cos_t*i_s)+bxi
    y=(Cr*nr-Ci*ni).sum(dim=-1)
    return y, nr, ni


class Mamba3Core(nn.Module):
    def __init__(self, d, nc=2, ds=16):
        super().__init__()
        self.d=d; self.nc=nc; self.ds=ds
        self.in_proj=nn.Linear(d,nc*d,bias=False)
        self.A_theta=nn.Parameter(torch.randn(nc,d,ds)*0.1)
        self.B_real=nn.Parameter(torch.randn(nc,d,ds)*0.1)
        self.B_imag=nn.Parameter(torch.randn(nc,d,ds)*0.1)
        self.C_real=nn.Parameter(torch.randn(nc,d,ds)*0.1)
        self.C_imag=nn.Parameter(torch.randn(nc,d,ds)*0.1)
        self.out_proj=nn.Linear(nc*d,d,bias=False)
        self.norm=nn.RMSNorm(d)
    def forward(self, x, rs=None, is_=None, ct=None, st=None):
        B,L,_=x.shape
        xi=self.in_proj(x).view(B,L,self.nc,self.d).unsqueeze(-1)
        if rs is None:
            rs=torch.zeros(B,L,self.nc,self.d,self.ds,device=x.device,dtype=x.dtype)
            is_=torch.zeros_like(rs)
        if ct is None:
            ct=torch.cos(self.A_theta).unsqueeze(0).unsqueeze(0)
            st=torch.sin(self.A_theta).unsqueeze(0).unsqueeze(0)
        y,nr,ni=fused_mimo(xi,rs,is_,ct,st,self.B_real,self.B_imag,self.C_real,self.C_imag)
        out=self.out_proj(y.view(B,L,self.nc*self.d))
        return x+self.norm(out), nr, ni


class RecursiveMamba2_13B(nn.Module):
    MAX_LOOPS = 6
    def __init__(self, base, rank=8):
        super().__init__()
        self.backbone=base.backbone; self.lm_head=base.lm_head
        self.top_layers=nn.ModuleList([base.backbone.layers[i] for i in range(BASE_SPLIT,48)])
        self.norm=base.backbone.norm_f
        d=base.backbone.embedding.embedding_dim
        for layer in self.top_layers:
            mx=layer.mixer
            for a in ("in_proj","out_proj"):
                if hasattr(mx,a): setattr(mx,a,LoRALinear(getattr(mx,a),rank=rank,alpha=rank*2.0))
        self.step_emb=nn.Embedding(self.MAX_LOOPS,d).to(torch.bfloat16)
        self.loop_norm=nn.RMSNorm(d).to(torch.bfloat16)
        self.mamba3_core=Mamba3Core(d).to(torch.bfloat16)

    def forward(self, input_ids):
        x=self.backbone.embedding(input_ids); res=None
        for layer in self.backbone.layers[:BASE_SPLIT]:
            x,res=layer(x,res)
        vocab=self.lm_head.weight.shape[0]
        mask=torch.full((vocab,),float("-inf"),device=x.device)
        uniq=torch.unique(input_ids[0])
        allowed=torch.cat([uniq,torch.tensor(ALLOWED_CORE,device=x.device)]).unique()
        mask[allowed]=0.0
        rs=is_=None
        ct=torch.cos(self.mamba3_core.A_theta).unsqueeze(0).unsqueeze(0)
        st=torch.sin(self.mamba3_core.A_theta).unsqueeze(0).unsqueeze(0)
        trace=[]; loops=self.MAX_LOOPS
        for i in range(self.MAX_LOOPS):
            sv=self.step_emb(torch.tensor(i,device=x.device)); x=x+sv
            for layer in self.top_layers: x,res=layer(x,res)
            x,rs,is_=self.mamba3_core(x,rs,is_,ct,st)
            x=self.loop_norm(x)
            lg=self.lm_head(self.norm(x,res,prenorm=False))
            lg[0,-1,:]+=mask
            p=torch.softmax(lg[0,-1,:],dim=-1)
            tid=p.argmax().item(); tok=tokenizer.decode([tid]).strip()
            trace.append((f"L{i+1}",round(p.max().item(),2),tok))
            if tid!=THINK_TOKEN_ID: loops=i+1; break
        return loops, trace


print("Loading model...", flush=True)
base = MambaLMHeadModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device=DEVICE)
nv=len(tokenizer); ov=base.backbone.embedding.weight.shape[0]; d=base.backbone.embedding.embedding_dim
if nv>ov:
    e=nn.Embedding(nv,d); nn.init.normal_(e.weight,std=0.02); e.weight.data[:ov]=base.backbone.embedding.weight.data; base.backbone.embedding=e
    h=nn.Linear(d,nv,bias=False); nn.init.normal_(h.weight,std=0.02); h.weight.data[:ov]=base.backbone.lm_head.weight.data if hasattr(base.backbone,'lm_head') else base.lm_head.weight.data; base.lm_head=h
for p in base.parameters(): p.requires_grad=False
base.backbone.embedding.weight.requires_grad=True; base.lm_head.weight.requires_grad=True

model=RecursiveMamba2_13B(base,rank=LORA_RANK).to(DEVICE)
ckpt=torch.load(CKPT,map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"],strict=False)
model.eval()
print("Ready.\n")

tests = [
    ("2-hop", "A = blue. B = A. What is B?\nAnswer:",         "blue"),
    ("2-hop", "X = 7. Y = X. What is Y?\nAnswer:",            "7"),
    ("2-hop", "color = green. shade = color. What is shade?\nAnswer:", "green"),
    ("3-hop", "A = red. B = A. C = B. What is C?\nAnswer:",   "red"),
    ("3-hop", "P = cat. Q = P. R = Q. What is R?\nAnswer:",   "cat"),
    ("3-hop", "X = sky. Y = X. Z = Y. What is Z?\nAnswer:",   "sky"),
    ("MMLU",  "What is the capital of France?\nA. Berlin\nB. Paris\nC. Rome\nD. Madrid\nAnswer:", "B"),
    ("MMLU",  "What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:", "B"),
    ("MMLU",  "Which planet is closest to the Sun?\nA. Earth\nB. Mars\nC. Mercury\nD. Venus\nAnswer:", "C"),
    ("logic", "A is taller than B. B is taller than C. Who is shortest?\nAnswer:", "C"),
    ("logic", "Tom is faster than Sue. Sue is faster than Ren. Who is slowest?\nAnswer:", "Ren"),
    ("logic", "Alice is heavier than Bob. Bob is heavier than Carol. Who is lightest?\nAnswer:", "Carol"),
]

correct=0
with torch.no_grad():
    for tag, prompt, expected in tests:
        ids=tokenizer.encode(prompt,add_special_tokens=False,return_tensors="pt").to(DEVICE)
        loops, trace = model(ids)
        ans=trace[-1][2]
        ok="✅" if expected.lower() in ans.lower() else "❌"
        correct += expected.lower() in ans.lower()
        chain=" → ".join(f"{t[0]}:{t[2]}" for t in trace)
        print(f"  [{tag}] {ok}  {prompt.splitlines()[0][:50]!r}")
        print(f"         {chain}")
        print(f"         want={expected!r}  got={ans!r}  ({loops} loops)\n")

print(f"{'='*50}")
print(f"  SCORE: {correct}/{len(tests)}  ({100*correct//len(tests)}%)")
print(f"{'='*50}")
