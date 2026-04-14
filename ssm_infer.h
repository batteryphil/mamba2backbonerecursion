/*
 * ssm_infer.h — Mamba2 SSM Inference Engine for Bare Metal
 * Selective-scan forward pass + Recursive Latent Forcing (RLF) loop.
 *
 * Ported from mamba2backbonerecursion (batteryphil) for llm-baremetal (Djiby Diop).
 *
 * Architecture: Mamba2-2.7B multi-head SSM + RLF
 *   - Fused in_proj → [z, xBC, dt]
 *   - Per-head A_log, D, dt_bias (1D vectors, nheads elements)
 *   - Inner RMSNorm on x before SSM scan
 *   - Prefix Latent Scratchpad (M learnable tokens)
 *   - Latent Bridge (low-rank d_model → rank → d_model + residual)
 */

#ifndef SSM_INFER_H
#define SSM_INFER_H

#include <stdint.h>

/* ── Model Configuration ──────────────────────────────────────────────────── */

typedef struct {
    int d_model;        /* Hidden dimension (2560)                */
    int d_state;        /* SSM state dimension (128)              */
    int d_conv;         /* Causal conv1d kernel width (4)         */
    int expand;         /* Expansion factor for inner dim (2)     */
    int d_inner;        /* d_model * expand (5120)                */
    int n_layers;       /* Number of SSM blocks (64)              */
    int vocab_size;     /* Vocabulary size                        */
    int max_seq_len;    /* Maximum sequence length                */

    /* Mamba2 multi-head SSM */
    int nheads;         /* Number of SSM heads (80)               */
    int headdim;        /* Dimension per head (64)                */
    int ngroups;        /* Number of B/C groups (1)               */

    /* Derived Mamba2 dimensions (computed at load) */
    int conv_dim;       /* d_inner + 2*ngroups*d_state (5376)     */
    int in_proj_dim;    /* 2*d_inner + 2*ngroups*d_state + nheads */

    /* RLF parameters */
    int base_split;     /* Layer split for RLF (frozen vs LoRA)   */
    int max_rlf_loops;  /* Maximum recursive inference loops      */
    int halt_token_id;  /* <HALT> token ID for stopping recursion */

    /* RoPE parameters */
    int rope_base;      /* RoPE frequency base (default 10000)    */

    /* Runtime flags */
    int lifeline_enabled; /* 1=inject lifeline, 0=ablation mode   */

    /* Prefix Latent Scratchpad */
    int prefix_m;       /* Number of prefix memory tokens (8)     */

    /* Latent Bridge */
    int bridge_rank;    /* Low-rank bottleneck dim (64)           */

    /* Loop engine dimensions */
    int loop_nheads;    /* Loop engine SSM heads (20)             */
    int loop_headdim;   /* Loop engine head dimension (128)       */
    int loop_d_state;   /* Loop engine SSM state dim              */
    int loop_d_inner;   /* Loop d_model * 1 (computed)            */
    int loop_conv_dim;  /* Loop conv dimension (computed)         */
} MambaConfig;

/* ── Per-token SSM Running State ──────────────────────────────────────────── */

typedef struct {
    float *conv_buf;    /* Circular buffer for conv1d                            */
    float *ssm_state;   /* Hidden state h[n] [n_layers * d_inner * d_state]      */
    int    conv_pos;    /* Current position in circular conv buffer              */
    int    seq_pos;     /* Current position in the sequence                      */
} MambaState;

/* ── Model Weights (flat pointers into loaded weight blob) ────────────────── */

typedef struct {
    /* Embedding + output head */
    float *token_embedding;     /* [vocab_size, d_model]              */
    float *lm_head;             /* [vocab_size, d_model]              */
    float *final_norm_weight;   /* [d_model]                          */

    /* Per-layer Mamba2 weights (arrays of n_layers pointers) */
    float **layer_norm;         /* [n_layers] each [d_model]           */
    float **in_proj;            /* [n_layers] each [in_proj_dim, d_model] */
    float **conv1d_weight;      /* [n_layers] each [conv_dim, d_conv]  */
    float **conv1d_bias;        /* [n_layers] each [conv_dim]          */
    float **inner_norm;         /* [n_layers] each [d_inner]           */
    float **out_proj;           /* [n_layers] each [d_model, d_inner]  */
    float **A_log;              /* [n_layers] each [nheads]            */
    float **D;                  /* [n_layers] each [nheads]            */
    float **dt_bias;            /* [n_layers] each [nheads]            */

    /* RLF-specific weights */
    float *lifeline_gate;       /* [d_model]                          */
    float *loop_norm_weight;    /* [d_model]                          */

    /* Mamba2 loop core */
    float *loop_in_proj;        /* [loop_in_proj_dim, d_model]        */
    float *loop_conv1d_weight;  /* [loop_conv_dim, d_conv]            */
    float *loop_conv1d_bias;    /* [loop_conv_dim]                    */
    float *loop_inner_norm;     /* [loop_d_inner]                     */
    float *loop_out_proj;       /* [d_model, loop_d_inner]            */
    float *loop_A_log;          /* [loop_nheads]                      */
    float *loop_D;              /* [loop_nheads]                      */
    float *loop_dt_bias;        /* [loop_nheads]                      */

    /* Prefix Latent Scratchpad */
    float *latent_memory;       /* [prefix_m, d_model]                */

    /* Latent Communication Bridge */
    float *bridge_down;         /* [bridge_rank, d_model]             */
    float *bridge_up;           /* [d_model, bridge_rank]             */

    /* Phase 10: OO-SomaMind Custom Hardware Plugs */
    float *proprio_gate;        /* Proprioception Gate W_g [3, d_model] */
    float **post_lora_A;        /* Post-backbone LoRA matrices [L_layers] */
    float **post_lora_B;        /* Post-backbone LoRA matrices [L_layers] */
    float *halt_head_w[4];      /* HaltingHead v2 MLPs */
    float *halt_head_b[4];      

    /* Weight blob */
    float *_blob;
    uint64_t _blob_bytes;
} MambaWeights;

/* ── RLF Inference Trace ──────────────────────────────────────────────────── */

typedef struct {
    int    loop_index;
    int    predicted_token;
    float  confidence;
    int    is_halt;
} RlfTraceEntry;

typedef struct {
    int           n_loops;
    int           final_token;
    float         final_conf;
    RlfTraceEntry trace[64];
} RlfInferenceResult;

/* ── SSM Telemetry ────────────────────────────────────────────────────────── */

#define SSM_TELEM_MAX_LAYERS 72

typedef struct {
    int    telemetry_enabled;
    float  layer_gate_mean[SSM_TELEM_MAX_LAYERS];
    float  layer_gate_std[SSM_TELEM_MAX_LAYERS];
    float  layer_dt_mean[SSM_TELEM_MAX_LAYERS];
    float  lifeline_mean;
    float  lifeline_std;
    int    rlf_loops_used;
    float  rlf_conf_per_loop[64];
    float  output_entropy;
    float  output_top1_prob;
    float  inference_ms;
    float  tokens_per_sec;
    float  hidden_state_norm;
    int    nan_count;
    int    inf_count;
} SsmTelemetry;

/* ── Core API ─────────────────────────────────────────────────────────────── */

/**
 * Initialize default config for Mamba2-2.7B + RLF.
 */
void mamba_config_default(MambaConfig *cfg);

/**
 * Allocate running state for inference.
 */
int mamba_state_alloc(MambaState *state, const MambaConfig *cfg);

/**
 * Free running state.
 */
void mamba_state_free(MambaState *state);

/**
 * Reset state to zeros (call before new sequence).
 */
void mamba_state_reset(MambaState *state, const MambaConfig *cfg);

/**
 * Single-token Mamba2 block forward pass.
 */
void mamba2_block_forward(
    float       *x_out,
    const float *x_in,
    int          layer_idx,
    const MambaConfig  *cfg,
    const MambaWeights *wt,
    MambaState         *state
);

/**
 * Full forward pass through all layers for one token.
 */
void mamba_forward(
    float       *x,
    int          token_id,
    const MambaConfig  *cfg,
    const MambaWeights *wt,
    MambaState         *state
);

/**
 * RLF recursive inference loop with bridge + scratchpad.
 */
void mamba_rlf_infer(
    RlfInferenceResult *result,
    const int          *prompt_ids,
    int                 prompt_len,
    const MambaConfig  *cfg,
    const MambaWeights *wt,
    MambaState         *state
);

/**
 * Greedy autoregressive generation.
 */
int mamba_generate(
    int                *out_tokens,
    int                 max_gen,
    const int          *prompt_ids,
    int                 prompt_len,
    float               temperature,
    const MambaConfig  *cfg,
    const MambaWeights *wt,
    MambaState         *state
);

/* ── Math Utilities ───────────────────────────────────────────────────────── */

void ssm_rmsnorm(float *x_out, const float *x_in, const float *weight, int d);
float ssm_silu(float x);
void ssm_softmax(float *v, int n);
void ssm_rope_apply(float *x, int d_model, int loop_index, int rope_base);
void ssm_matvec(float *out, const float *W, const float *x, int m, int n);
void ssm_matvec_acc(float *out, const float *W, const float *x, int m, int n);

/* ── AVX2 SIMD ────────────────────────────────────────────────────────────── */

void ssm_detect_simd(void);
int ssm_has_avx2(void);
void ssm_matvec_avx2(float *out, const float *W, const float *x, int m, int n);
void ssm_matvec_acc_avx2(float *out, const float *W, const float *x, int m, int n);
void ssm_rmsnorm_avx2(float *x_out, const float *x_in, const float *weight, int d);
void ssm_softmax_avx2(float *v, int n);

#endif /* SSM_INFER_H */
