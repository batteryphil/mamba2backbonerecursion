"""
export_bpe_table.py — Export GPT-NeoX BPE tokenizer for bare-metal C inference.
=================================================================================
Extracts vocabulary + merge table from HuggingFace tokenizer and serializes
to a compact .bpe.bin binary format.

Format:
    [4B magic "BPE\0"]
    [4B vocab_size]
    [4B merge_count]
    [4B max_token_len]
    [vocab_size entries: [2B len][len bytes utf8]]
    [merge_count entries: [2B id_a][2B id_b][2B result_id]]

Usage:
    python export_bpe_table.py [output.bpe.bin]
"""
import struct
import sys
from transformers import AutoTokenizer


def export_bpe(output_path: str = "tokenizer.bpe.bin") -> None:
    """Export GPT-NeoX tokenizer to binary format for bare-metal."""
    print(f"Loading GPT-NeoX tokenizer...")
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tok.add_special_tokens({"additional_special_tokens": ["<THINK>", "<HALT>"]})

    vocab = tok.get_vocab()
    vocab_size = len(vocab)

    # Build id→token mapping
    id_to_token = [""] * vocab_size
    for token_str, token_id in vocab.items():
        if token_id < vocab_size:
            id_to_token[token_id] = token_str

    # Extract merges from tokenizer
    merges = []
    if hasattr(tok, 'bpe_ranks') and tok.bpe_ranks:
        # Direct access to BPE ranks
        for (a, b), rank in sorted(tok.bpe_ranks.items(), key=lambda x: x[1]):
            merged = a + b
            a_id = vocab.get(a, -1)
            b_id = vocab.get(b, -1)
            m_id = vocab.get(merged, -1)
            if a_id >= 0 and b_id >= 0 and m_id >= 0:
                merges.append((a_id, b_id, m_id))
    elif hasattr(tok.backend_tokenizer, 'model') and hasattr(tok.backend_tokenizer.model, 'merges'):
        # Tokenizers library path
        for merge_str in tok.backend_tokenizer.model.merges:
            parts = merge_str.split(' ')
            if len(parts) == 2:
                a, b = parts
                merged = a + b
                a_id = vocab.get(a, -1)
                b_id = vocab.get(b, -1)
                m_id = vocab.get(merged, -1)
                if a_id >= 0 and b_id >= 0 and m_id >= 0:
                    merges.append((a_id, b_id, m_id))

    # Find max token length
    max_token_len = max(len(t.encode('utf-8', errors='replace')) for t in id_to_token)

    print(f"  Vocab size:    {vocab_size:,}")
    print(f"  Merge count:   {len(merges):,}")
    print(f"  Max token len: {max_token_len}")
    print(f"  <HALT> ID:     {vocab.get('<HALT>', -1)}")

    # Write binary
    with open(output_path, 'wb') as f:
        # Header: magic + sizes
        f.write(b'BPE\x00')
        f.write(struct.pack('<III', vocab_size, len(merges), max_token_len))

        # Vocabulary: [2B len][len bytes utf8] for each token
        for i, token in enumerate(id_to_token):
            encoded = token.encode('utf-8', errors='replace')
            f.write(struct.pack('<H', len(encoded)))
            f.write(encoded)

        # Merges: [2B id_a][2B id_b][2B result_id]
        for a_id, b_id, m_id in merges:
            f.write(struct.pack('<HHH', a_id, b_id, m_id))

    import os
    file_size = os.path.getsize(output_path)
    print(f"\n  Saved: {output_path} ({file_size:,} bytes, {file_size/1024:.0f} KB)")

    # Verify round-trip
    test_strs = [
        "Hello, world!",
        "A = blue. B = A. What is B?\\nAnswer:",
        "The quick brown fox"
    ]
    print(f"\n  Round-trip verification:")
    for s in test_strs:
        ids = tok.encode(s, add_special_tokens=False)
        decoded = tok.decode(ids)
        match = "✓" if decoded.strip() == s.strip() else "✗"
        print(f"    {match} \"{s[:40]}\" → {len(ids)} tokens → \"{decoded[:40]}\"")


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'tokenizer.bpe.bin'
    export_bpe(out)
"""
Export GPT-NeoX BPE tokenizer for bare-metal C inference.
"""
