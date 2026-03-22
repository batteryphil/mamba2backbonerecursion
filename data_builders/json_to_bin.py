import json
import numpy as np
import os
from transformers import GPT2Tokenizer
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

def convert_json_to_bin(input_file="train_data.json", output_file="generic_150m_corpus.bin"):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"🧠 Initializing Tokenization Manifold: {input_file}")
    
    # We use a streaming approach to handle massive JSON files
    # If the file is a simple list of strings, we can process it efficiently.
    token_accumulator = []
    total_tokens = 0
    
    print("🚀 Streaming JSON data...")
    # For a list of strings, we can use a basic parser if it's too large, 
    # but for 150M tokens, we'll use a robust approach.
    with open(input_file, 'r', encoding='utf-8') as f:
        # If the file is relatively small (like the 9.7MB one found), load all.
        # If it were actually 150M tokens, we'd use ijson.
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Error decoding JSON: {e}")
            return

    with open(output_file, 'wb') as f_out:
        for text in tqdm(data, desc="Tokenizing"):
            # Ensure text is string
            if not isinstance(text, str):
                continue
                
            # Encode text + EOS
            ids = tokenizer.encode(text + " " + tokenizer.eos_token, add_special_tokens=False)
            
            # Convert to uint16 (max 65535, GPT2 vocab is 50257)
            arr = np.array(ids, dtype=np.uint16)
            f_out.write(arr.tobytes())
            
            total_tokens += len(ids)

    print("\n" + "="*50)
    print("✅ BINARY CONVERSION COMPLETE")
    print("="*50)
    print(f"Output File: {output_file}")
    print(f"Total Tokens: {total_tokens:,}")
    
    expected_size = total_tokens * 2
    actual_size = os.path.getsize(output_file)
    
    print(f"Expected Size: {expected_size:,} bytes")
    print(f"Actual Size:   {actual_size:,} bytes")
    
    if expected_size == actual_size:
        print("💎 VALIDATION PASSED: Binary integrity confirmed.")
    else:
        print("⚠️ VALIDATION WARNING: Size mismatch detected.")
    print("="*50)

if __name__ == "__main__":
    convert_json_to_bin()
