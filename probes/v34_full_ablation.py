import torch
from transformers import AutoTokenizer
from v34_causal_gate_ablation import DEVICE, model, find_answer_start, HALT_ID

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>", "<HALT>"]})

def run_trace(model_obj):
    prompt = "P = telescope. Q = P. R = Q. S = R. What is S?\nAnswer:"
    ids_ = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    ans_start = find_answer_start(ids_[0].tolist())
    trace = []
    with torch.no_grad():
        x = model_obj.backbone.embedding(ids_)
        residual = None
        for layer in model_obj.all_layers[:6]: x, residual = layer(x, residual)
        x_prompt = x.clone().detach()
        for loop_i in range(5):
            x = model_obj._lifeline_inject(x, x_prompt)
            x = model_obj.loop_rope(x, loop_i)
            for layer in model_obj.all_layers[6:]: x, residual = layer(x, residual)
            x = x + model_obj.mamba2_core(x)
            x = model_obj.loop_norm(x)
            logits = model_obj.lm_head(model_obj.norm(x, residual, prenorm=False))[0, ans_start - 1, :]
            top_id = logits.argmax().item()
            tok = tokenizer.decode([top_id]).strip()
            if top_id == HALT_ID: tok = "<HALT>"
            trace.append(tok)
    return trace

print("\n--- BASE TRACE ---")
print(run_trace(model))

print("\n--- ZERO ABLATION (Lifeline Removed) ---")
model.lifeline_gate.data.fill_(0.0)
print(run_trace(model))

