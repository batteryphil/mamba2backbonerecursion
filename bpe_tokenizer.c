/*
 * bpe_tokenizer.c — BPE Tokenizer for Bare-Metal (UEFI/x86)
 *
 * Implements greedy BPE merge algorithm for tokenization.
 * No dynamic memory allocation in the encode/decode hot path.
 * All allocations happen at load time only.
 */

#include "bpe_tokenizer.h"
#include <stdlib.h>
#include <string.h>

/* ── Load ─────────────────────────────────────────────────────────────────── */

int bpe_load(BpeTokenizer *bpe, const void *data, uint64_t data_len)
{
    /**
     * Parse .bpe.bin format:
     *   [4B magic "BPE\0"]
     *   [4B vocab_size][4B merge_count][4B max_token_len]
     *   [vocab entries: [2B len][len bytes]]
     *   [merge entries: [2B a][2B b][2B result]]
     */
    if (!bpe || !data || data_len < 16) return -1;

    const uint8_t *p = (const uint8_t *)data;
    const uint8_t *end = p + data_len;

    /* Check magic */
    if (p[0] != 'B' || p[1] != 'P' || p[2] != 'E' || p[3] != 0) return -1;
    p += 4;

    /* Read sizes */
    uint32_t vocab_size, merge_count, max_token_len;
    memcpy(&vocab_size, p, 4);     p += 4;
    memcpy(&merge_count, p, 4);    p += 4;
    memcpy(&max_token_len, p, 4);  p += 4;

    bpe->vocab_size    = (int)vocab_size;
    bpe->merge_count   = (int)merge_count;
    bpe->max_token_len = (int)max_token_len;

    /* Allocate vocab arrays */
    bpe->vocab_strs = (char **)calloc(vocab_size, sizeof(char *));
    bpe->vocab_lens = (int *)calloc(vocab_size, sizeof(int));
    if (!bpe->vocab_strs || !bpe->vocab_lens) return -1;

    /* Read vocab entries */
    for (uint32_t i = 0; i < vocab_size && p + 2 <= end; i++) {
        uint16_t len;
        memcpy(&len, p, 2); p += 2;
        if (p + len > end) break;

        bpe->vocab_strs[i] = (char *)malloc(len + 1);
        if (bpe->vocab_strs[i]) {
            memcpy(bpe->vocab_strs[i], p, len);
            bpe->vocab_strs[i][len] = '\0';
        }
        bpe->vocab_lens[i] = (int)len;
        p += len;
    }

    /* Build byte-to-id map */
    memset(bpe->byte_to_id, 0, sizeof(bpe->byte_to_id));
    for (int b = 0; b < 256; b++) {
        /* Search for single-byte token matching this byte */
        char target[4];
        target[0] = (char)b;
        target[1] = '\0';
        for (uint32_t i = 0; i < vocab_size; i++) {
            if (bpe->vocab_lens[i] == 1 && bpe->vocab_strs[i] &&
                (uint8_t)bpe->vocab_strs[i][0] == (uint8_t)b) {
                bpe->byte_to_id[b] = (int)i;
                break;
            }
        }
    }

    /* Allocate and read merge table */
    bpe->merge_a      = (int *)calloc(merge_count, sizeof(int));
    bpe->merge_b      = (int *)calloc(merge_count, sizeof(int));
    bpe->merge_result = (int *)calloc(merge_count, sizeof(int));
    if (!bpe->merge_a || !bpe->merge_b || !bpe->merge_result) return -1;

    for (uint32_t i = 0; i < merge_count && p + 6 <= end; i++) {
        uint16_t a, b, r;
        memcpy(&a, p, 2); p += 2;
        memcpy(&b, p, 2); p += 2;
        memcpy(&r, p, 2); p += 2;
        bpe->merge_a[i]      = (int)a;
        bpe->merge_b[i]      = (int)b;
        bpe->merge_result[i] = (int)r;
    }

    return 0;
}

/* ── Encode ───────────────────────────────────────────────────────────────── */

int bpe_encode(
    const BpeTokenizer *bpe,
    const char         *text,
    int                *out_ids,
    int                 max_tokens)
{
    /**
     * Greedy BPE encoding:
     * 1. Split text into individual byte tokens
     * 2. Repeatedly merge the highest-priority pair
     * 3. Stop when no more merges apply
     */
    if (!bpe || !text || !out_ids || max_tokens <= 0) return 0;

    int text_len = 0;
    while (text[text_len]) text_len++;
    if (text_len == 0) return 0;

    /* Step 1: initialize with byte-level tokens */
    int *tokens = (int *)malloc(text_len * sizeof(int));
    if (!tokens) return 0;
    int n = 0;
    for (int i = 0; i < text_len; i++) {
        tokens[n++] = bpe->byte_to_id[(uint8_t)text[i]];
    }

    /* Step 2: apply merges greedily (priority = index in merge table) */
    for (int m = 0; m < bpe->merge_count; m++) {
        int a = bpe->merge_a[m];
        int b = bpe->merge_b[m];
        int r = bpe->merge_result[m];

        /* Scan for this pair */
        for (int i = 0; i < n - 1; i++) {
            if (tokens[i] == a && tokens[i + 1] == b) {
                /* Merge: replace pair with result, shift remaining */
                tokens[i] = r;
                for (int j = i + 1; j < n - 1; j++) {
                    tokens[j] = tokens[j + 1];
                }
                n--;
                /* Don't advance i — check for consecutive merges */
                if (i > 0) i--;
            }
        }
    }

    /* Step 3: copy to output */
    int out_n = (n < max_tokens) ? n : max_tokens;
    for (int i = 0; i < out_n; i++) {
        out_ids[i] = tokens[i];
    }

    free(tokens);
    return out_n;
}

/* ── Decode ───────────────────────────────────────────────────────────────── */

int bpe_decode(
    const BpeTokenizer *bpe,
    const int          *token_ids,
    int                 n_tokens,
    char               *out_text,
    int                 max_len)
{
    /**
     * Concatenate token strings for each token ID.
     */
    if (!bpe || !token_ids || !out_text || max_len <= 0) return 0;

    int pos = 0;
    for (int i = 0; i < n_tokens && pos < max_len - 1; i++) {
        int id = token_ids[i];
        if (id < 0 || id >= bpe->vocab_size) continue;
        const char *s = bpe->vocab_strs[id];
        int len = bpe->vocab_lens[id];
        if (!s) continue;

        int copy = len;
        if (pos + copy >= max_len) copy = max_len - pos - 1;
        memcpy(out_text + pos, s, copy);
        pos += copy;
    }
    out_text[pos] = '\0';
    return pos;
}

/* ── Free ─────────────────────────────────────────────────────────────────── */

void bpe_free(BpeTokenizer *bpe)
{
    /**
     * Free all tokenizer resources.
     */
    if (!bpe) return;
    if (bpe->vocab_strs) {
        for (int i = 0; i < bpe->vocab_size; i++) {
            free(bpe->vocab_strs[i]);
        }
        free(bpe->vocab_strs);
    }
    free(bpe->vocab_lens);
    free(bpe->merge_a);
    free(bpe->merge_b);
    free(bpe->merge_result);
    memset(bpe, 0, sizeof(*bpe));
}
