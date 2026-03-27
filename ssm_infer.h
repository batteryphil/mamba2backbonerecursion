/*
 * ssm_infer.h — Mamba2 SSM Inference Engine for Bare Metal
 * Selective-scan forward pass + Recursive Latent Forcing (RLF) loop.
 *
 * Ported from mamba2backbonerecursion (batteryphil) for llm-baremetal (Djiby Diop).
 * This replaces the Transformer attention mechanism with an SSM selective scan.
 *
 * Architecture: Mamba2 + RLF v34 (RoPE loop encoding, Prompt Lifeline)
 */

#ifndef SSM_INFER_H
#define SSM_INFER_H

#include <stdint.h>

/* ── Model Configuration ──────────────────────────────────────────────────── */

typedef struct {
    int d_model;        /* Hidden dimension (e.g. 768 for mamba2-130m) */
    int d_state;        /* SSM state dimension N (e.g. 64)            */
    int d_conv;         /* Causal conv1d kernel width (e.g. 4)        */
    int expand;         /* Expansion factor for inner dim (e.g. 2)    */
    int d_inner;        /* d_model * expand (computed)                */
    int n_layers;       /* Number of SSM blocks (e.g. 24)             */
    int vocab_size;     /* Vocabulary size                            */
    int max_seq_len;    /* Maximum sequence length                    */

    /* RLF parameters */
    int base_split;     /* Layer split for RLF (frozen vs LoRA)       */
    int max_rlf_loops;  /* Maximum recursive inference loops          */
    int halt_token_id;  /* <HALT> token ID for stopping recursion     */

    /* RoPE parameters */
    int rope_base;      /* RoPE frequency base (default 10000)        */

    /* Runtime flags */
    int lifeline_enabled; /* 1=inject lifeline, 0=ablation mode       */
} MambaConfig;

/* ── Per-token SSM Running State ──────────────────────────────────────────── */

typedef struct {
    float *conv_buf;    /* Circular buffer for conv1d [n_layers * d_inner * d_conv]   */
    float *ssm_state;   /* Hidden state h[n]        [n_layers * d_inner * d_state]    */
    int    conv_pos;    /* Current position in circular conv buffer                    */
    int    seq_pos;     /* Current position in the sequence                            */
} MambaState;

/* ── Model Weights (flat float32 pointers into loaded weight blob) ────────── */

typedef struct {
    /* Embedding + output head */
    float *token_embedding;     /* [vocab_size, d_model]              */
    float *lm_head;             /* [vocab_size, d_model]              */

    /* Per-layer SSM weights (arrays of n_layers pointers) */
    /* Each SSM block: in_proj → conv1d → SSM scan → out_proj */
    float **in_proj_weight;     /* [n_layers] each [2*d_inner, d_model]  (gate+x) */
    float **conv1d_weight;      /* [n_layers] each [d_inner, d_conv]               */
    float **conv1d_bias;        /* [n_layers] each [d_inner]                       */

    /* SSM parameters */
    float **x_proj_weight;      /* [n_layers] each [dt_rank+2*d_state, d_inner]    */
    float **dt_proj_weight;     /* [n_layers] each [d_inner, dt_rank]              */
    float **dt_proj_bias;       /* [n_layers] each [d_inner]                       */
    float **A_log;              /* [n_layers] each [d_inner, d_state]              */
    float **D;                  /* [n_layers] each [d_inner]                       */

    float **out_proj_weight;    /* [n_layers] each [d_model, d_inner]              */

    /* Norm weights */
    float **norm_weight;        /* [n_layers] each [d_model] — RMSNorm             */
    float *final_norm_weight;   /* [d_model] — final RMSNorm                       */

    /* RLF-specific weights */
    float *lifeline_gate;       /* [d_model] — float32 vector gate                 */
    float *loop_norm_weight;    /* [d_model] — RMSNorm for loop output             */

    /* Mamba2 loop core (recursive SSM block) */
    float *loop_in_proj;        /* [2*loop_d_inner, d_model]                       */
    float *loop_conv1d_weight;  /* [loop_d_inner, d_conv]                          */
    float *loop_conv1d_bias;    /* [loop_d_inner]                                  */
    float *loop_x_proj;         /* [dt_rank+2*d_state, loop_d_inner]               */
    float *loop_dt_proj_weight; /* [loop_d_inner, dt_rank]                         */
    float *loop_dt_proj_bias;   /* [loop_d_inner]                                  */
    float *loop_A_log;          /* [loop_d_inner, d_state]                         */
    float *loop_D;              /* [loop_d_inner]                                  */
    float *loop_out_proj;       /* [d_model, loop_d_inner]                         */

    /* Weight blob (for freeing) */
    float *_blob;               /* Raw allocated blob — caller manages lifetime    */
    uint64_t _blob_bytes;       /* Size of blob in bytes                           */
} MambaWeights;

/* ── RLF Inference Trace (per-loop prediction record) ─────────────────────── */

typedef struct {
    int    loop_index;      /* 0-based loop number                  */
    int    predicted_token;  /* Top-1 token at answer position       */
    float  confidence;       /* Softmax probability of top-1         */
    int    is_halt;          /* 1 if predicted_token == halt_id      */
} RlfTraceEntry;

typedef struct {
    int           n_loops;       /* Total loops executed          */
    int           final_token;   /* Final answer token            */
    float         final_conf;    /* Final answer confidence       */
    RlfTraceEntry trace[64];     /* Per-loop trace (max 64)       */
} RlfInferenceResult;

/* ── SSM Telemetry (for OO Engine consumption) ────────────────────────────── */

#define SSM_TELEM_MAX_LAYERS 32

typedef struct {
    /* Per-inference metrics (updated each forward pass) */
    int    telemetry_enabled;              /* 1=collect metrics, 0=skip    */

    /* Per-layer scan gate statistics */
    float  layer_gate_mean[SSM_TELEM_MAX_LAYERS];   /* Mean gate activation  */
    float  layer_gate_std[SSM_TELEM_MAX_LAYERS];    /* Std dev of gates      */
    float  layer_dt_mean[SSM_TELEM_MAX_LAYERS];     /* Mean dt (timestep)    */

    /* Lifeline gate analysis (RAM vs ALU partition) */
    float  lifeline_mean;           /* Mean gate value               */
    float  lifeline_std;            /* Std dev of gate               */
    float  lifeline_ram_frac;       /* Fraction of dims > mean+1σ    */
    float  lifeline_alu_frac;       /* Fraction of dims < mean-1σ    */
    int    lifeline_ram_dims;       /* Count of RAM dimensions       */
    int    lifeline_alu_dims;       /* Count of ALU dimensions       */

    /* RLF loop convergence */
    int    rlf_loops_used;          /* Loops until HALT or max       */
    float  rlf_conf_per_loop[64];   /* Confidence at each loop       */
    float  rlf_convergence_rate;    /* Avg confidence increase/loop  */

    /* Output distribution */
    float  output_entropy;          /* Softmax entropy (bits)        */
    float  output_top1_prob;        /* Top-1 probability             */
    float  output_top5_prob;        /* Sum of top-5 probabilities    */

    /* Performance */
    float  inference_ms;            /* Total inference time (ms)     */
    float  tokens_per_sec;          /* Throughput                    */

    /* Memory state health */
    float  hidden_state_norm;       /* L2 norm of final hidden state */
    int    nan_count;               /* NaN detections                */
    int    inf_count;               /* Inf detections                */
} SsmTelemetry;

/**
 * Analyze lifeline gate to compute RAM/ALU partition metrics.
 * Populates lifeline_* fields in telemetry.
 */
void ssm_analyze_lifeline(SsmTelemetry *telem, const float *gate, int d_model);

/**
 * Compute softmax entropy in bits.
 */
float ssm_entropy(const float *probs, int n);

/* ── Core API ─────────────────────────────────────────────────────────────── */

/**
 * Initialize default config for Mamba2-130m + RLF v34.
 */
void mamba_config_default(MambaConfig *cfg);

/**
 * Allocate running state for inference.
 * Returns 0 on success, -1 on allocation failure.
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
 * Single-token SSM forward pass through one Mamba block.
 * x_in: [d_model] → x_out: [d_model]
 */
void mamba_block_forward(
    float       *x_out,
    const float *x_in,
    int          layer_idx,
    const MambaConfig  *cfg,
    const MambaWeights *weights,
    MambaState         *state
);

/**
 * Full forward pass through all layers for one token.
 * x: [d_model] (in/out, modified in place)
 */
void mamba_forward(
    float       *x,
    int          token_id,
    const MambaConfig  *cfg,
    const MambaWeights *weights,
    MambaState         *state
);

/**
 * RLF recursive inference loop.
 * Given a tokenized prompt, runs the recursive reasoning loop
 * until <HALT> or max_loops, returning the per-loop trace.
 */
void mamba_rlf_infer(
    RlfInferenceResult *result,
    const int          *prompt_ids,
    int                 prompt_len,
    const MambaConfig  *cfg,
    const MambaWeights *weights,
    MambaState         *state
);

/**
 * Greedy autoregressive generation (non-RLF, standard LM mode).
 * out_tokens: pre-allocated buffer for generated tokens.
 * Returns number of tokens generated.
 */
int mamba_generate(
    int                *out_tokens,
    int                 max_gen,
    const int          *prompt_ids,
    int                 prompt_len,
    float               temperature,
    const MambaConfig  *cfg,
    const MambaWeights *weights,
    MambaState         *state
);

/* ── Math Utilities ───────────────────────────────────────────────────────── */

/**
 * RMSNorm: x_out[i] = weight[i] * x_in[i] / rms(x_in)
 */
void ssm_rmsnorm(float *x_out, const float *x_in, const float *weight, int d);

/**
 * SiLU (Swish): x → x * sigmoid(x)
 */
float ssm_silu(float x);

/**
 * Softmax in-place over v[n].
 */
void ssm_softmax(float *v, int n);

/**
 * Apply 1D RoPE rotation for a given loop index.
 * x: [d_model] (in/out, modified in place)
 */
void ssm_rope_apply(float *x, int d_model, int loop_index, int rope_base);

/**
 * Matmul: out[m] = W[m,n] @ x[n]
 */
void ssm_matvec(float *out, const float *W, const float *x, int m, int n);

/**
 * Matmul with accumulate: out[m] += W[m,n] @ x[n]
 */
void ssm_matvec_acc(float *out, const float *W, const float *x, int m, int n);

/* ── AVX2 SIMD-Optimized Variants ─────────────────────────────────────────── */

/**
 * Detect AVX2+FMA support. Call once at boot.
 */
void ssm_detect_simd(void);

/**
 * Check if AVX2 is available on this CPU.
 */
int ssm_has_avx2(void);

/**
 * AVX2 matvec: out[m] = W[m,n] @ x[n] (8 floats/cycle with FMA).
 */
void ssm_matvec_avx2(float *out, const float *W, const float *x, int m, int n);

/**
 * AVX2 matvec accumulate: out[m] += W[m,n] @ x[n].
 */
void ssm_matvec_acc_avx2(float *out, const float *W, const float *x, int m, int n);

/**
 * AVX2 RMSNorm.
 */
void ssm_rmsnorm_avx2(float *x_out, const float *x_in, const float *weight, int d);

/**
 * AVX2 softmax in-place over v[n].
 */
void ssm_softmax_avx2(float *v, int n);

#endif /* SSM_INFER_H */
