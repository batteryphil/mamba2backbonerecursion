/*
 * ssm_weights.c — Weight Loader for Mamba2 SSM Bare Metal
 *
 * Loads .mamba.bin files into MambaWeights struct.
 * Zero-copy: weight pointers point directly into the memory-mapped blob.
 */

#include "ssm_weights.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Header Parsing ───────────────────────────────────────────────────────── */

int mamba_parse_header(
    MambaBinHeader *hdr,
    const void     *data,
    uint64_t        data_len
)
{
    /**
     * Parse .mamba.bin header from raw bytes.
     * Validates magic and version.
     */
    if (!hdr || !data || data_len < sizeof(MambaBinHeader)) return -1;

    memcpy(hdr, data, sizeof(MambaBinHeader));

    if (hdr->magic != MAMBA_BIN_MAGIC) {
        return -1;
    }
    if (hdr->version != MAMBA_BIN_VERSION) {
        return -1;
    }
    if (hdr->d_model <= 0 || hdr->n_layers <= 0 || hdr->vocab_size <= 0) {
        return -1;
    }
    return 0;
}

/* ── Weight Loading ───────────────────────────────────────────────────────── */

/**
 * Helper: advance cursor pointer and return float* pointing at current pos.
 */
static float *claim_floats(const uint8_t **cursor, uint64_t *remaining, int64_t count)
{
    /**
     * Claim a contiguous block of float32 values from the data stream.
     */
    if (count <= 0) return NULL;
    uint64_t bytes = (uint64_t)count * sizeof(float);
    if (bytes > *remaining) return NULL;

    float *ptr = (float *)(*cursor);
    *cursor    += bytes;
    *remaining -= bytes;
    return ptr;
}

int mamba_load_weights(
    MambaWeights       *wt,
    MambaConfig        *cfg,
    const void         *data,
    uint64_t            data_len
)
{
    /**
     * Load all weights from .mamba.bin buffer.
     *
     * Weight layout (after header):
     *   token_embedding  [vocab_size * d_model]
     *   lm_head          [vocab_size * d_model]
     *   final_norm       [d_model]
     *   For each layer 0..n_layers-1:
     *     norm_weight     [d_model]
     *     in_proj         [2*d_inner * d_model]
     *     conv1d_weight   [d_inner * d_conv]
     *     conv1d_bias     [d_inner]
     *     x_proj          [(dt_rank + 2*d_state) * d_inner]
     *     dt_proj_weight  [d_inner * dt_rank]
     *     dt_proj_bias    [d_inner]
     *     A_log           [d_inner * d_state]
     *     D               [d_inner]
     *     out_proj        [d_model * d_inner]
     *   If has_rlf:
     *     lifeline_gate      [d_model]
     *     loop_norm_weight   [d_model]
     *     loop_in_proj       [2*d_inner * d_model]
     *     loop_conv1d_weight [d_inner * d_conv]
     *     loop_conv1d_bias   [d_inner]
     *     loop_x_proj        [(dt_rank + 2*d_state) * d_inner]
     *     loop_dt_proj_w     [d_inner * dt_rank]
     *     loop_dt_proj_b     [d_inner]
     *     loop_A_log         [d_inner * d_state]
     *     loop_D             [d_inner]
     *     loop_out_proj      [d_model * d_inner]
     */
    if (!wt || !cfg || !data) return -1;
    memset(wt, 0, sizeof(*wt));

    MambaBinHeader hdr;
    if (mamba_parse_header(&hdr, data, data_len) != 0) return -1;

    /* Fill config from header */
    cfg->d_model       = hdr.d_model;
    cfg->d_state       = hdr.d_state;
    cfg->d_conv        = hdr.d_conv;
    cfg->expand        = hdr.expand;
    cfg->d_inner       = hdr.d_model * hdr.expand;
    cfg->n_layers      = hdr.n_layers;
    cfg->vocab_size    = hdr.vocab_size;
    cfg->max_seq_len   = hdr.max_seq_len;
    cfg->base_split    = hdr.base_split;
    cfg->max_rlf_loops = hdr.max_rlf_loops;
    cfg->halt_token_id = hdr.halt_token_id;
    cfg->rope_base     = hdr.rope_base;

    int d_model  = hdr.d_model;
    int d_inner  = cfg->d_inner;
    int d_state  = hdr.d_state;
    int d_conv   = hdr.d_conv;
    int n_layers = hdr.n_layers;
    int vocab    = hdr.vocab_size;
    int dt_rank  = hdr.dt_rank > 0 ? hdr.dt_rank : d_model;
    int xbc_dim  = dt_rank + 2 * d_state;

    /* Allocate per-layer pointer arrays */
    wt->in_proj_weight  = (float **)calloc(n_layers, sizeof(float *));
    wt->conv1d_weight   = (float **)calloc(n_layers, sizeof(float *));
    wt->conv1d_bias     = (float **)calloc(n_layers, sizeof(float *));
    wt->x_proj_weight   = (float **)calloc(n_layers, sizeof(float *));
    wt->dt_proj_weight  = (float **)calloc(n_layers, sizeof(float *));
    wt->dt_proj_bias    = (float **)calloc(n_layers, sizeof(float *));
    wt->A_log           = (float **)calloc(n_layers, sizeof(float *));
    wt->D               = (float **)calloc(n_layers, sizeof(float *));
    wt->out_proj_weight = (float **)calloc(n_layers, sizeof(float *));
    wt->norm_weight     = (float **)calloc(n_layers, sizeof(float *));

    if (!wt->in_proj_weight || !wt->norm_weight) {
        mamba_free_weight_arrays(wt);
        return -1;
    }

    /* Walk through binary data after header */
    const uint8_t *cursor   = (const uint8_t *)data + sizeof(MambaBinHeader);
    uint64_t       remaining = data_len - sizeof(MambaBinHeader);

    /* Global tensors */
    wt->token_embedding  = claim_floats(&cursor, &remaining, (int64_t)vocab * d_model);
    wt->lm_head          = claim_floats(&cursor, &remaining, (int64_t)vocab * d_model);
    wt->final_norm_weight = claim_floats(&cursor, &remaining, d_model);

    if (!wt->token_embedding || !wt->lm_head || !wt->final_norm_weight) {
        mamba_free_weight_arrays(wt);
        return -1;
    }

    /* Per-layer tensors */
    for (int l = 0; l < n_layers; l++) {
        wt->norm_weight[l]     = claim_floats(&cursor, &remaining, d_model);
        wt->in_proj_weight[l]  = claim_floats(&cursor, &remaining, (int64_t)2 * d_inner * d_model);
        wt->conv1d_weight[l]   = claim_floats(&cursor, &remaining, (int64_t)d_inner * d_conv);
        wt->conv1d_bias[l]     = claim_floats(&cursor, &remaining, d_inner);
        wt->x_proj_weight[l]   = claim_floats(&cursor, &remaining, (int64_t)xbc_dim * d_inner);
        wt->dt_proj_weight[l]  = claim_floats(&cursor, &remaining, (int64_t)d_inner * dt_rank);
        wt->dt_proj_bias[l]    = claim_floats(&cursor, &remaining, d_inner);
        wt->A_log[l]           = claim_floats(&cursor, &remaining, (int64_t)d_inner * d_state);
        wt->D[l]               = claim_floats(&cursor, &remaining, d_inner);
        wt->out_proj_weight[l] = claim_floats(&cursor, &remaining, (int64_t)d_model * d_inner);

        if (!wt->norm_weight[l] || !wt->in_proj_weight[l]) {
            mamba_free_weight_arrays(wt);
            return -1;
        }
    }

    /* RLF-specific weights (optional) */
    if (hdr.has_rlf) {
        wt->lifeline_gate       = claim_floats(&cursor, &remaining, d_model);
        wt->loop_norm_weight    = claim_floats(&cursor, &remaining, d_model);
        wt->loop_in_proj        = claim_floats(&cursor, &remaining, (int64_t)2 * d_inner * d_model);
        wt->loop_conv1d_weight  = claim_floats(&cursor, &remaining, (int64_t)d_inner * d_conv);
        wt->loop_conv1d_bias    = claim_floats(&cursor, &remaining, d_inner);
        wt->loop_x_proj         = claim_floats(&cursor, &remaining, (int64_t)xbc_dim * d_inner);
        wt->loop_dt_proj_weight = claim_floats(&cursor, &remaining, (int64_t)d_inner * dt_rank);
        wt->loop_dt_proj_bias   = claim_floats(&cursor, &remaining, d_inner);
        wt->loop_A_log          = claim_floats(&cursor, &remaining, (int64_t)d_inner * d_state);
        wt->loop_D              = claim_floats(&cursor, &remaining, d_inner);
        wt->loop_out_proj       = claim_floats(&cursor, &remaining, (int64_t)d_model * d_inner);
    }

    /* Store blob reference */
    wt->_blob       = (float *)data;
    wt->_blob_bytes = data_len;

    return 0;
}

void mamba_free_weight_arrays(MambaWeights *wt)
{
    /**
     * Free per-layer pointer arrays (but NOT the backing data blob).
     */
    if (!wt) return;
    free(wt->in_proj_weight);   wt->in_proj_weight  = NULL;
    free(wt->conv1d_weight);    wt->conv1d_weight   = NULL;
    free(wt->conv1d_bias);      wt->conv1d_bias     = NULL;
    free(wt->x_proj_weight);    wt->x_proj_weight   = NULL;
    free(wt->dt_proj_weight);   wt->dt_proj_weight  = NULL;
    free(wt->dt_proj_bias);     wt->dt_proj_bias    = NULL;
    free(wt->A_log);            wt->A_log           = NULL;
    free(wt->D);                wt->D               = NULL;
    free(wt->out_proj_weight);  wt->out_proj_weight = NULL;
    free(wt->norm_weight);      wt->norm_weight     = NULL;
}

void mamba_print_summary(const MambaConfig *cfg, const MambaWeights *wt)
{
    /**
     * Print model loading summary.
     */
    if (!cfg) return;
    printf("\n══════════════════════════════════════════════════════════════\n");
    printf("  Mamba2 SSM + RLF Bare-Metal Model\n");
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  d_model:       %d\n", cfg->d_model);
    printf("  d_state:       %d\n", cfg->d_state);
    printf("  d_conv:        %d\n", cfg->d_conv);
    printf("  expand:        %d\n", cfg->expand);
    printf("  d_inner:       %d\n", cfg->d_inner);
    printf("  n_layers:      %d\n", cfg->n_layers);
    printf("  vocab_size:    %d\n", cfg->vocab_size);
    printf("  max_seq_len:   %d\n", cfg->max_seq_len);
    printf("  base_split:    %d (RLF)\n", cfg->base_split);
    printf("  max_rlf_loops: %d\n", cfg->max_rlf_loops);
    printf("  halt_token_id: %d\n", cfg->halt_token_id);
    printf("  rope_base:     %d\n", cfg->rope_base);
    if (wt) {
        printf("  lifeline_gate: %s\n", wt->lifeline_gate ? "loaded" : "none");
        printf("  loop_core:     %s\n", wt->loop_in_proj ? "loaded" : "none");
        printf("  blob_bytes:    %llu\n", (unsigned long long)wt->_blob_bytes);
    }
    printf("══════════════════════════════════════════════════════════════\n\n");
}
