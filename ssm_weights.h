/*
 * ssm_weights.h — Weight Format & Loader for Mamba2 SSM Bare Metal
 *
 * Custom .mamba.bin binary format for UEFI bare-metal weight loading.
 * All weights stored as flat float32 tensors.
 */

#ifndef SSM_WEIGHTS_H
#define SSM_WEIGHTS_H

#include "ssm_infer.h"
#include <stdint.h>

/* ── File Format ──────────────────────────────────────────────────────────── */

#define MAMBA_BIN_MAGIC  0x4D414D42  /* "MAMB" in little-endian */
#define MAMBA_BIN_VERSION 1

typedef struct __attribute__((packed)) {
    uint32_t magic;           /* MAMBA_BIN_MAGIC                    */
    uint32_t version;         /* Format version                     */
    int32_t  d_model;         /* Hidden dimension                   */
    int32_t  d_state;         /* SSM state dimension N              */
    int32_t  d_conv;          /* Conv1d kernel width                */
    int32_t  expand;          /* Inner dimension expansion factor   */
    int32_t  n_layers;        /* Number of SSM blocks               */
    int32_t  vocab_size;      /* Vocabulary size                    */
    int32_t  max_seq_len;     /* Maximum sequence length            */
    int32_t  base_split;      /* RLF layer split point              */
    int32_t  max_rlf_loops;   /* Max recursive inference loops      */
    int32_t  halt_token_id;   /* <HALT> token ID                    */
    int32_t  rope_base;       /* RoPE frequency base                */
    int32_t  dt_rank;         /* Rank of dt projection              */
    int32_t  has_rlf;         /* 1 if RLF weights included          */
    int32_t  quant_type;      /* 0=fp32, 1=int8 (was reserved[0])   */
    int32_t  _reserved[8];    /* Reserved for future use            */
    uint64_t total_bytes;     /* Total file size in bytes           */
} MambaBinHeader;

/* Quantization types */
#define MAMBA_QUANT_FP32  0
#define MAMBA_QUANT_INT8  1

/* ── Loader API ───────────────────────────────────────────────────────────── */

/**
 * Parse the .mamba.bin header from a file buffer.
 * Returns 0 on success, -1 on invalid format.
 */
int mamba_parse_header(
    MambaBinHeader *hdr,
    const void     *data,
    uint64_t        data_len
);

/**
 * Load weights from a .mamba.bin buffer into MambaWeights.
 * The data buffer must remain valid for the lifetime of weights
 * (we point directly into it — zero-copy).
 *
 * Returns 0 on success, -1 on error.
 */
int mamba_load_weights(
    MambaWeights       *wt,
    MambaConfig        *cfg,
    const void         *data,
    uint64_t            data_len
);

/**
 * Free weight pointer arrays (but NOT the backing blob).
 */
void mamba_free_weight_arrays(MambaWeights *wt);

/**
 * Print weight loading summary to stdout.
 */
void mamba_print_summary(const MambaConfig *cfg, const MambaWeights *wt);

#endif /* SSM_WEIGHTS_H */
