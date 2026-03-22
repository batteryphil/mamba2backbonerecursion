import torch
import torch.nn as nn
from transformers import AutoTokenizer
from finetune_mamba2_130m_v34 import RecursiveMamba2_v34, find_answer_start, HALT_ID
from mamba_ssm import MambaLMHeadModel

DEVICE = "cuda"
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>", "<HALT>"]})

print("Loading model for final checks...")
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

def run_test(gate_mult=1.0, noise=False, shuffle=False):
    prompt = "P = algorithm. Q = P. R = Q. S = R. T = S. U = T. V = U. W = V. What is W?\nAnswer:"
    ids_ = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    ans_start = find_answer_start(ids_[0].tolist())
    
    trace = []
    with torch.no_grad():
        x = model.backbone.embedding(ids_)
        residual = None
        for layer in model.all_layers[:6]: x, residual = layer(x, residual)
        x_prompt = x.clone().detach() * gate_mult
        if noise: x_prompt = torch.randn_like(x_prompt)
        if shuffle: x_prompt = x_prompt[:, torch.randperm(x_prompt.shape[1]), :]
            
        for loop_i in range(8):
            x = model._lifeline_inject(x, x_prompt)
            x = model.loop_rope(x, loop_i)
            for layer in model.all_layers[6:]:
                x, residual = layer(x, residual)
            x = x + model.mamba2_core(x)
            x = model.loop_norm(x)
            
            logits = model.lm_head(model.norm(x, residual, prenorm=False))[0, ans_start - 1, :]
            top_id = logits.argmax().item()
            tok = tokenizer.decode([top_id]).strip()
            if top_id == HALT_ID: tok = "<HALT>"
            trace.append(tok)
    return trace

print("Base:   ", run_test())
print("Zero:   ", run_test(gate_mult=0.0))
print("Noise:  ", run_test(noise=True))
print("Shuffle:", run_test(shuffle=True))
