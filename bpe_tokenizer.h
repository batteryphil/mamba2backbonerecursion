/*
 * bpe_tokenizer.h — BPE Tokenizer for Bare-Metal (UEFI/x86)
 *
 * Loads a pre-built BPE vocabulary table from .bpe.bin and performs
 * encoding (text → token IDs) and decoding (token IDs → text).
 *
 * Uses GPT-NeoX BPE format exported by export_bpe_table.py.
 */

#ifndef BPE_TOKENIZER_H
#define BPE_TOKENIZER_H

#include <stdint.h>

/* ── BPE Tokenizer State ─────────────────────────────────────────────────── */

typedef struct {
    int     vocab_size;       /* Total vocabulary size             */
    int     merge_count;      /* Number of BPE merge rules         */
    int     max_token_len;    /* Maximum token length in bytes      */

    /* Vocabulary: array of strings (UTF-8) */
    char  **vocab_strs;       /* [vocab_size] token strings         */
    int    *vocab_lens;       /* [vocab_size] string lengths        */

    /* Merge table: sorted by priority */
    int    *merge_a;          /* [merge_count] first token ID       */
    int    *merge_b;          /* [merge_count] second token ID      */
    int    *merge_result;     /* [merge_count] merged token ID      */

    /* Byte-to-token map for initial character splitting */
    int     byte_to_id[256];  /* ASCII byte → nearest token ID      */
} BpeTokenizer;

/* ── API ─────────────────────────────────────────────────────────────────── */

/**
 * Load a BPE tokenizer from a .bpe.bin buffer.
 * Returns 0 on success, -1 on error.
 */
int bpe_load(BpeTokenizer *bpe, const void *data, uint64_t data_len);

/**
 * Encode a UTF-8 string into token IDs.
 * out_ids: pre-allocated buffer for output token IDs.
 * max_tokens: size of out_ids buffer.
 * Returns number of tokens produced.
 */
int bpe_encode(
    const BpeTokenizer *bpe,
    const char         *text,
    int                *out_ids,
    int                 max_tokens
);

/**
 * Decode token IDs back to UTF-8 text.
 * out_text: pre-allocated buffer for output string.
 * max_len: size of out_text buffer.
 * Returns number of characters written.
 */
int bpe_decode(
    const BpeTokenizer *bpe,
    const int          *token_ids,
    int                 n_tokens,
    char               *out_text,
    int                 max_len
);

/**
 * Free tokenizer resources.
 */
void bpe_free(BpeTokenizer *bpe);

#endif /* BPE_TOKENIZER_H */
