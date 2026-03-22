import json
import numpy as np
import os
from transformers import GPT2Tokenizer
from tqdm import tqdm

def build_protocol_v6_corpus():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 📚 SOURCE MANIFEST
    major_corpus = "/home/phil/Desktop/jarvis new/Project-Bit-Jarvis/data/raw/cosmo_tiny.jsonl"
    identity_src = "/home/phil/Desktop/jarvis new/Project-Bit-Jarvis/data/processed/personality_flood.jsonl"
    legacy_json = "train_data.json"
    
    target_bin = "generic_150m_corpus.bin"
    total_tokens = 0
    unique_tokens = set()
    
    print(f"🚀 PROTOCOL v6.0: Building 200M+ Token Manifold...")
    
    # Simple deduplication set for small identity samples
    seen_identity = set()

    # Create/Clear the file
    with open(target_bin, "wb") as f_out:
        
        # 1. Process 412MB Cosmopedia (The Library)
        if os.path.exists(major_corpus):
            print(f"📖 Streaming {os.path.basename(major_corpus)}...")
            with open(major_corpus, "r") as f:
                for line in tqdm(f, desc="Tokenizing Cosmopedia"):
                    try:
                        data = json.loads(line)
                        text = data.get("text", "").strip()
                        if not text: continue
                        
                        ids = tokenizer.encode(text + " " + tokenizer.eos_token, add_special_tokens=False)
                        f_out.write(np.array(ids, dtype=np.uint16).tobytes())
                        total_tokens += len(ids)
                    except Exception:
                        continue
        
        # 2. Process Identity & Personas
        if os.path.exists(identity_src):
            print(f"🎭 Injecting Identity Manifold...")
            with open(identity_src, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        instr = data.get("instruction", "").strip()
                        out = data.get("output", "").strip()
                        combined = f"User: {instr} | Assistant: {out} {tokenizer.eos_token}"
                        
                        if combined in seen_identity: continue
                        seen_identity.add(combined)
                        
                        ids = tokenizer.encode(combined, add_special_tokens=False)
                        f_out.write(np.array(ids, dtype=np.uint16).tobytes())
                        total_tokens += len(ids)
                    except Exception:
                        continue
                        
        # 3. Process Legacy Training Data
        if os.path.exists(legacy_json):
            print(f"📚 Appending Legacy Logic Data...")
            with open(legacy_json, "r") as f:
                data = json.load(f)
                for item in data:
                    text = item.strip() if isinstance(item, str) else str(item)
                    ids = tokenizer.encode(text + " " + tokenizer.eos_token, add_special_tokens=False)
                    f_out.write(np.array(ids, dtype=np.uint16).tobytes())
                    total_tokens += len(ids)

    print("\n" + "="*50)
    print("💎 PROTOCOL v6.0 MANIFOLD COMPLETE")
    print("="*50)
    print(f"Total Tokens:    {total_tokens:,}")
    print(f"Final File Size: {os.path.getsize(target_bin) // (1024*1024)} MB")
    print(f"Estimated Epoch Time (5k TPS): {total_tokens // (5000 * 60):,} minutes")
    print("="*50)

if __name__ == "__main__":
    build_protocol_v6_corpus()
