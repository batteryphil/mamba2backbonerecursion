import json
import numpy as np
import os
from transformers import GPT2Tokenizer
from tqdm import tqdm

def inject_identity_and_personas():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 📚 SOURCE MANIFEST
    sources = [
        "/home/phil/local-agent-wrapper/identity_corpus.jsonl",
        "/home/phil/Desktop/jarvis new/Project-Bit-Jarvis/data/processed/personality_flood.jsonl"
    ]
    
    target_bin = "generic_150m_corpus.bin"
    seen_texts = set()
    total_tokens = 0
    unique_tokens = set()
    
    print(f"🕵️  Scraping Identity Manifolds for Phase II Injection...")
    
    all_chunks = []
    
    for src in sources:
        if not os.path.exists(src):
            print(f"⚠️ Source not found: {src}")
            continue
            
        print(f"📖 Reading {os.path.basename(src)}...")
        with open(src, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    instr = data.get("instruction", "").strip()
                    out = data.get("output", "").strip()
                    
                    if not instr or not out:
                        continue
                    
                    # 🛡️ Deduplication & Low-Info Filtering
                    combined = f"{instr} | {out}"
                    if combined in seen_texts or len(combined) < 10:
                        continue
                    seen_texts.add(combined)
                    
                    # 🎭 Context Formatting
                    # Converting to a cohesive "Instruction-Response" prompt
                    formatted_text = f"User: {instr} | Assistant: {out} {tokenizer.eos_token}"
                    
                    # Tokenization
                    ids = tokenizer.encode(formatted_text, add_special_tokens=False)
                    for idx in ids:
                        unique_tokens.add(idx)
                    
                    all_chunks.append(np.array(ids, dtype=np.uint16))
                    total_tokens += len(ids)
                except Exception:
                    continue

    if not all_chunks:
        print("❌ No new unique identity samples found.")
        return

    # 🚀 BINARY APPEND
    print(f"💉 Injecting {total_tokens:,} tokens into {target_bin}...")
    with open(target_bin, "ab") as f:
        for chunk in all_chunks:
            f.write(chunk.tobytes())
            
    # 📊 ENTROPY CHECK
    # Diversity Score = Unique Tokens / Total tokens in the new set
    diversity_score = len(unique_tokens) / (total_tokens + 1e-6)
    
    print("\n" + "="*50)
    print("💎 IDENTITY INJECTION COMPLETE")
    print("="*50)
    print(f"New tokens added:    {total_tokens:,}")
    print(f"Unique tokens:       {len(unique_tokens):,}")
    print(f"Diversity Score:     {diversity_score:.4f} (Entropy Target: >0.0100)")
    print(f"Total stream size:   {os.path.getsize(target_bin) // 2:,} tokens")
    
    if diversity_score > 0.01:
        print("🟢 ENTROPY PASSED: New data is sufficiently diverse.")
    else:
        print("🟡 ENTROPY WARNING: High lexical overlap detected.")
    print("="*50)

if __name__ == "__main__":
    inject_identity_and_personas()
