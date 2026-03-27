/*
 * ssm_infer.c — Mamba2 SSM Inference Engine for Bare Metal
 *
 * Pure-C implementation of:
 *   1. Mamba2 selective scan (SSM forward pass)
 *   2. Recursive Latent Forcing (RLF) inference loop
 *   3. RoPE loop encoding + Prompt Lifeline injection
 *
 * Ported from mamba2backbonerecursion/mamba_scan_kernel.cu
 * and training/finetune_mamba2_130m_v34.py.
 *
 * Created for llm-baremetal (Djiby Diop) by batteryphil.
 */

#include "ssm_infer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Portable min/max ──────────────────────────────────────────────────────── */

static inline int  ssm_min_i(int a, int b)   { return a < b ? a : b; }
static inline int  ssm_max_i(int a, int b)   { return a > b ? a : b; }
static inline float ssm_absf(float x)        { return x < 0.0f ? -x : x; }

/* ── Config Default ───────────────────────────────────────────────────────── */

void mamba_config_default(MambaConfig *cfg)
{
    /**
     * Default config for Mamba2-130m + RLF v34.
     */
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));

    cfg->d_model      = 768;
    cfg->d_state      = 64;     /* LOOP_D_STATE from v34 */
    cfg->d_conv       = 4;
    cfg->expand        = 2;
    cfg->d_inner       = cfg->d_model * cfg->expand;   /* 1536 */
    cfg->n_layers      = 24;
    cfg->vocab_size    = 50282;  /* gpt-neox-20b + <THINK> + <HALT> */
    cfg->max_seq_len   = 256;

    cfg->base_split    = 6;
    cfg->max_rlf_loops = 16;
    cfg->halt_token_id = 50281;  /* Typical <HALT> ID after add_special_tokens */

    cfg->rope_base     = 10000;
    cfg->lifeline_enabled = 1;  /* Lifeline ON by default */
}

/* ── Math Utilities ───────────────────────────────────────────────────────── */

float ssm_silu(float x)
{
    /**
     * SiLU/Swish activation: x * sigmoid(x).
     */
    return x / (1.0f + expf(-x));
}

void ssm_rmsnorm(float *x_out, const float *x_in, const float *weight, int d)
{
    /**
     * RMSNorm: out[i] = weight[i] * x_in[i] / sqrt(mean(x_in^2) + eps).
     */
    if (!x_out || !x_in || !weight || d <= 0) return;

    float ss = 0.0f;
    for (int i = 0; i < d; i++) {
        ss += x_in[i] * x_in[i];
    }
    ss = 1.0f / sqrtf(ss / (float)d + 1e-5f);
    for (int i = 0; i < d; i++) {
        x_out[i] = weight[i] * x_in[i] * ss;
    }
}

void ssm_softmax(float *v, int n)
{
    /**
     * In-place softmax with numerical stability (max subtraction).
     */
    if (!v || n <= 0) return;

    float max_val = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > max_val) max_val = v[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        v[i] = expf(v[i] - max_val);
        sum += v[i];
    }
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int i = 0; i < n; i++) {
            v[i] *= inv;
        }
    }
}

void ssm_matvec(float *out, const float *W, const float *x, int m, int n)
{
    /**
     * Matrix-vector multiply: out[m] = W[m,n] @ x[n].
     * W is row-major: W[i*n + j].
     */
    if (!out || !W || !x || m <= 0 || n <= 0) return;

    for (int i = 0; i < m; i++) {
        float acc = 0.0f;
        const float *row = W + i * n;
        for (int j = 0; j < n; j++) {
            acc += row[j] * x[j];
        }
        out[i] = acc;
    }
}

void ssm_matvec_acc(float *out, const float *W, const float *x, int m, int n)
{
    /**
     * Matrix-vector multiply with accumulate: out[m] += W[m,n] @ x[n].
     */
    if (!out || !W || !x || m <= 0 || n <= 0) return;

    for (int i = 0; i < m; i++) {
        float acc = 0.0f;
        const float *row = W + i * n;
        for (int j = 0; j < n; j++) {
            acc += row[j] * x[j];
        }
        out[i] += acc;
    }
}

void ssm_rope_apply(float *x, int d_model, int loop_index, int rope_base)
{
    /**
     * Apply 1D RoPE rotation for a given loop index.
     *
     * Ported from LoopRoPE in finetune_mamba2_130m_v34.py:
     *   inv_freq = 1.0 / (base ** (arange(0, d, 2) / d))
     *   freqs = loop_index * inv_freq
     *   x_out = x * cos(freqs) + rotate_half(x) * sin(freqs)
     *
     * The rotation makes the model's representation of "I am at loop N"
     * a continuous function of N rather than a table lookup.
     */
    if (!x || d_model <= 0) return;

    int half_d = d_model / 2;
    float n = (float)loop_index;

    for (int i = 0; i < half_d; i++) {
        /* Frequency band for dimension pair i */
        float theta = 1.0f / powf((float)rope_base, (float)(2 * i) / (float)d_model);
        float freq = n * theta;
        float cos_f = cosf(freq);
        float sin_f = sinf(freq);

        /* Apply rotation to pair (x[2i], x[2i+1]) */
        int idx0 = 2 * i;
        int idx1 = 2 * i + 1;
        if (idx1 >= d_model) break;

        float x0 = x[idx0];
        float x1 = x[idx1];
        x[idx0] = x0 * cos_f - x1 * sin_f;
        x[idx1] = x0 * sin_f + x1 * cos_f;
    }
}

/* ── SSM Selective Scan (Core Kernel) ─────────────────────────────────────── */

/**
 * Single-step selective scan for one SSM block.
 *
 * This is the pure-C equivalent of mamba_scan_kernel.cu:
 *   h[n] = exp(dt * A[n]) * h[n] + dt * B[n] * x
 *   y += h[n] * C[n]
 *   y_final = y + x * D_param
 *
 * Arguments:
 *   y_out:    output [d_inner]
 *   x_in:     input after conv+gate [d_inner]
 *   dt:       time step [d_inner]
 *   B_vec:    input-dependent SSM matrix row [d_state]
 *   C_vec:    input-dependent SSM matrix row [d_state]
 *   A_log:    log of diagonal A [d_inner, d_state]
 *   D_param:  feed-through [d_inner]
 *   h_state:  running SSM hidden state [d_inner * d_state] (modified in place!)
 *   d_inner:  inner dimension
 *   d_state:  SSM state dimension N
 */
static void ssm_selective_scan_step(
    float       *y_out,
    const float *x_in,
    const float *dt,
    const float *B_vec,
    const float *C_vec,
    const float *A_log,
    const float *D_param,
    float       *h_state,
    int          d_inner,
    int          d_state
)
{
    /**
     * Core SSM scan kernel ported from CUDA shared-memory implementation.
     */
    for (int d = 0; d < d_inner; d++) {
        float x_t  = x_in[d];
        float dt_t = dt[d];
        float y_t  = 0.0f;

        float *h = h_state + d * d_state;
        const float *A_row = A_log + d * d_state;

        for (int n = 0; n < d_state; n++) {
            /* A is stored as log — exponentiate: A_real = -exp(A_log) */
            float A_val = -expf(A_row[n]);
            float r = expf(dt_t * A_val);
            float b_bar = dt_t * B_vec[n];

            h[n] = r * h[n] + b_bar * x_t;
            y_t += h[n] * C_vec[n];
        }

        y_out[d] = y_t + x_t * D_param[d];
    }
}

/* ── Conv1d Step ──────────────────────────────────────────────────────────── */

/**
 * Single-step causal conv1d using circular buffer.
 *
 * conv_buf: [d_inner * d_conv] circular buffer
 * conv_w:  [d_inner, d_conv] conv weights
 * conv_b:  [d_inner] conv bias
 * x_in:    [d_inner] current input (stored into circular buffer)
 * x_out:   [d_inner] convolution output
 * conv_pos: pointer to circular buffer position (modified!)
 */
static void ssm_conv1d_step(
    float       *x_out,
    const float *x_in,
    float       *conv_buf,
    int         *conv_pos,
    const float *conv_w,
    const float *conv_b,
    int          d_inner,
    int          d_conv
)
{
    /**
     * Causal 1D convolution with circular buffer for streaming inference.
     */
    int pos = *conv_pos;

    /* Store current input into circular buffer */
    for (int d = 0; d < d_inner; d++) {
        conv_buf[d * d_conv + pos] = x_in[d];
    }

    /* Compute convolution */
    for (int d = 0; d < d_inner; d++) {
        float acc = conv_b ? conv_b[d] : 0.0f;
        for (int k = 0; k < d_conv; k++) {
            int buf_idx = (pos - d_conv + 1 + k + d_conv * 256) % d_conv;
            acc += conv_buf[d * d_conv + buf_idx] * conv_w[d * d_conv + k];
        }
        x_out[d] = acc;
    }

    *conv_pos = (pos + 1) % d_conv;
}

/* ── State Management ─────────────────────────────────────────────────────── */

int mamba_state_alloc(MambaState *state, const MambaConfig *cfg)
{
    /**
     * Allocate running state buffers for Mamba inference.
     */
    if (!state || !cfg) return -1;
    memset(state, 0, sizeof(*state));

    int d_inner = cfg->d_inner;

    /* Conv buffer: [n_layers * d_inner * d_conv] */
    size_t conv_bytes = (size_t)cfg->n_layers * d_inner * cfg->d_conv * sizeof(float);
    state->conv_buf = (float *)calloc(1, conv_bytes);
    if (!state->conv_buf) return -1;

    /* SSM state: [n_layers * d_inner * d_state] */
    size_t ssm_bytes = (size_t)cfg->n_layers * d_inner * cfg->d_state * sizeof(float);
    state->ssm_state = (float *)calloc(1, ssm_bytes);
    if (!state->ssm_state) {
        free(state->conv_buf);
        state->conv_buf = NULL;
        return -1;
    }

    state->conv_pos = 0;
    state->seq_pos = 0;
    return 0;
}

void mamba_state_free(MambaState *state)
{
    /**
     * Free all running state buffers.
     */
    if (!state) return;
    if (state->conv_buf) { free(state->conv_buf); state->conv_buf = NULL; }
    if (state->ssm_state) { free(state->ssm_state); state->ssm_state = NULL; }
    state->conv_pos = 0;
    state->seq_pos = 0;
}

void mamba_state_reset(MambaState *state, const MambaConfig *cfg)
{
    /**
     * Reset state to zeros for a new sequence.
     */
    if (!state || !cfg) return;
    int d_inner = cfg->d_inner;
    if (state->conv_buf) {
        memset(state->conv_buf, 0,
               (size_t)cfg->n_layers * d_inner * cfg->d_conv * sizeof(float));
    }
    if (state->ssm_state) {
        memset(state->ssm_state, 0,
               (size_t)cfg->n_layers * d_inner * cfg->d_state * sizeof(float));
    }
    state->conv_pos = 0;
    state->seq_pos = 0;
}

/* ── Single Mamba Block Forward ───────────────────────────────────────────── */

void mamba_block_forward(
    float       *x_out,
    const float *x_in,
    int          layer_idx,
    const MambaConfig  *cfg,
    const MambaWeights *wt,
    MambaState         *state
)
{
    /**
     * Forward pass through one Mamba2 SSM block (single token).
     *
     * Architecture per block:
     *   1. RMSNorm
     *   2. in_proj → (gate, x) split
     *   3. Conv1d (causal, d_conv=4)
     *   4. SiLU activation on x
     *   5. SSM scan: x_proj → (dt, B, C); selective scan with h state
     *   6. Gate: y = silu(gate) * scan_output
     *   7. out_proj
     *   8. Residual add
     */
    int d_model = cfg->d_model;
    int d_inner = cfg->d_inner;
    int d_state = cfg->d_state;
    int d_conv  = cfg->d_conv;
    int dt_rank = d_model;  /* Mamba2 uses dt_rank == d_model by default */

    /* Scratch buffers — stack allocated for typical sizes */
    float norm_buf[2048];     /* d_model */
    float proj_buf[4096];     /* 2 * d_inner */
    float conv_out[2048];     /* d_inner */
    float xbc_buf[4096];      /* dt_rank + 2*d_state */
    float dt_buf[2048];       /* d_inner */
    float scan_out[2048];     /* d_inner */
    float gated[2048];        /* d_inner */

    /* Safety: clip to stack buffer sizes */
    if (d_model > 2048 || d_inner > 2048) {
        /* Fall through: copy input to output as identity */
        memcpy(x_out, x_in, d_model * sizeof(float));
        return;
    }

    /* 1. RMSNorm */
    ssm_rmsnorm(norm_buf, x_in, wt->norm_weight[layer_idx], d_model);

    /* 2. in_proj: [2*d_inner, d_model] @ x → (gate, x_ssm) */
    ssm_matvec(proj_buf, wt->in_proj_weight[layer_idx], norm_buf, 2 * d_inner, d_model);
    float *gate_vec = proj_buf;            /* first d_inner */
    float *x_ssm    = proj_buf + d_inner;  /* second d_inner */

    /* 3. Conv1d (causal) */
    float *layer_conv_buf = state->conv_buf +
        (size_t)layer_idx * d_inner * d_conv;
    int conv_pos = state->conv_pos;
    ssm_conv1d_step(conv_out, x_ssm, layer_conv_buf, &conv_pos,
                    wt->conv1d_weight[layer_idx],
                    wt->conv1d_bias[layer_idx],
                    d_inner, d_conv);

    /* 4. SiLU on conv output */
    for (int i = 0; i < d_inner; i++) {
        conv_out[i] = ssm_silu(conv_out[i]);
    }

    /* 5. x_proj → (dt, B, C) */
    int xbc_dim = dt_rank + 2 * d_state;
    if (xbc_dim > 4096) xbc_dim = 4096;
    ssm_matvec(xbc_buf, wt->x_proj_weight[layer_idx], conv_out, xbc_dim, d_inner);

    float *dt_raw   = xbc_buf;
    float *B_vec    = xbc_buf + dt_rank;
    float *C_vec    = xbc_buf + dt_rank + d_state;

    /* dt_proj: [d_inner, dt_rank] @ dt_raw + bias → dt */
    ssm_matvec(dt_buf, wt->dt_proj_weight[layer_idx], dt_raw, d_inner, dt_rank);
    if (wt->dt_proj_bias[layer_idx]) {
        for (int i = 0; i < d_inner; i++) {
            dt_buf[i] += wt->dt_proj_bias[layer_idx][i];
        }
    }
    /* Softplus on dt */
    for (int i = 0; i < d_inner; i++) {
        dt_buf[i] = logf(1.0f + expf(dt_buf[i]));
    }

    /* 6. Selective scan */
    float *layer_h = state->ssm_state +
        (size_t)layer_idx * d_inner * d_state;
    ssm_selective_scan_step(scan_out, conv_out, dt_buf,
                            B_vec, C_vec,
                            wt->A_log[layer_idx],
                            wt->D[layer_idx],
                            layer_h,
                            d_inner, d_state);

    /* 7. Gate: y = silu(gate) * scan_output */
    for (int i = 0; i < d_inner; i++) {
        gated[i] = ssm_silu(gate_vec[i]) * scan_out[i];
    }

    /* 8. out_proj + residual */
    ssm_matvec(x_out, wt->out_proj_weight[layer_idx], gated, d_model, d_inner);
    for (int i = 0; i < d_model; i++) {
        x_out[i] += x_in[i];  /* Residual connection */
    }
}

/* ── Full Model Forward (Single Token) ────────────────────────────────────── */

void mamba_forward(
    float       *x,
    int          token_id,
    const MambaConfig  *cfg,
    const MambaWeights *wt,
    MambaState         *state
)
{
    /**
     * Full Mamba forward pass for a single token.
     *
     * 1. Embed token → x[d_model]
     * 2. For each layer: mamba_block_forward (norm → proj → conv → SSM → gate → proj + residual)
     * 3. Final norm
     * (Caller applies lm_head to get logits)
     */
    int d_model = cfg->d_model;

    /* 1. Token embedding */
    if (token_id >= 0 && token_id < cfg->vocab_size && wt->token_embedding) {
        const float *emb = wt->token_embedding + (size_t)token_id * d_model;
        memcpy(x, emb, d_model * sizeof(float));
    }

    /* 2. Layer-by-layer forward */
    float x_buf[2048];
    if (d_model > 2048) return;

    for (int l = 0; l < cfg->n_layers; l++) {
        mamba_block_forward(x_buf, x, l, cfg, wt, state);
        memcpy(x, x_buf, d_model * sizeof(float));
    }

    /* 3. Final RMSNorm */
    if (wt->final_norm_weight) {
        memcpy(x_buf, x, d_model * sizeof(float));
        ssm_rmsnorm(x, x_buf, wt->final_norm_weight, d_model);
    }

    state->seq_pos++;
}

/* ── Logits + Sampling ────────────────────────────────────────────────────── */

static int ssm_argmax(const float *v, int n)
{
    /**
     * Return index of maximum value in v[n].
     */
    if (!v || n <= 0) return 0;
    int best = 0;
    float best_val = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > best_val) {
            best_val = v[i];
            best = i;
        }
    }
    return best;
}

static int ssm_sample_topk(const float *logits, int vocab, float temp, int top_k)
{
    /**
     * Temperature-scaled top-k sampling.
     */
    if (!logits || vocab <= 0) return 0;
    if (temp <= 0.0f || top_k <= 1) return ssm_argmax(logits, vocab);

    /* Apply temperature */
    float *probs = (float *)malloc(vocab * sizeof(float));
    if (!probs) return ssm_argmax(logits, vocab);

    for (int i = 0; i < vocab; i++) {
        probs[i] = logits[i] / temp;
    }
    ssm_softmax(probs, vocab);

    /* Simple greedy for now (top-k sorting is expensive in bare-metal) */
    int tok = ssm_argmax(probs, vocab);
    free(probs);
    return tok;
}

/* ── RLF Recursive Loop SSM Block ─────────────────────────────────────────── */

/**
 * Forward pass through the RLF loop's dedicated Mamba2 core block.
 * This is the "reasoning engine" that runs inside the recursive loop.
 *
 * x: [d_model] hidden state (modified in place, residual added)
 */
static void ssm_rlf_loop_block(
    float       *x,
    const MambaConfig  *cfg,
    const MambaWeights *wt
)
{
    /**
     * Mamba2 loop core block (the recursive reasoning SSM).
     *
     * Mirrors mamba2_core in RecursiveMamba2_v34:
     *   x = x + self.mamba2_core(x)
     *   x = self.loop_norm(x)
     */
    int d_model = cfg->d_model;
    int d_inner = d_model * cfg->expand;
    int d_state = cfg->d_state;
    int dt_rank = d_model;

    /* Stack scratch buffers */
    float proj_buf[4096];
    float conv_out[2048];
    float xbc_buf[4096];
    float dt_buf[2048];
    float scan_out[2048];
    float gated[2048];
    float proj_out[2048];

    if (d_model > 2048 || d_inner > 2048) return;

    /* in_proj: (gate, x) */
    if (wt->loop_in_proj) {
        ssm_matvec(proj_buf, wt->loop_in_proj, x, 2 * d_inner, d_model);
    } else {
        memset(proj_buf, 0, 2 * d_inner * sizeof(float));
    }
    float *gate_vec = proj_buf;
    float *x_ssm    = proj_buf + d_inner;

    /* Conv1d (no state tracking for loop core — uses input directly) */
    for (int i = 0; i < d_inner; i++) {
        conv_out[i] = ssm_silu(x_ssm[i]);
    }

    /* x_proj → (dt, B, C) */
    int xbc_dim = dt_rank + 2 * d_state;
    if (wt->loop_x_proj) {
        ssm_matvec(xbc_buf, wt->loop_x_proj, conv_out,
                   ssm_min_i(xbc_dim, 4096), d_inner);
    } else {
        memset(xbc_buf, 0, xbc_dim * sizeof(float));
    }

    float *dt_raw = xbc_buf;
    float *B_vec  = xbc_buf + dt_rank;
    float *C_vec  = xbc_buf + dt_rank + d_state;

    /* dt_proj + softplus */
    if (wt->loop_dt_proj_weight) {
        ssm_matvec(dt_buf, wt->loop_dt_proj_weight, dt_raw, d_inner, dt_rank);
    } else {
        memset(dt_buf, 0, d_inner * sizeof(float));
    }
    if (wt->loop_dt_proj_bias) {
        for (int i = 0; i < d_inner; i++) dt_buf[i] += wt->loop_dt_proj_bias[i];
    }
    for (int i = 0; i < d_inner; i++) {
        dt_buf[i] = logf(1.0f + expf(dt_buf[i]));
    }

    /* Selective scan (use A_log, D from loop core) */
    /* Temporary zero h-state per loop iteration (stateless reasoning) */
    float *h_temp = (float *)calloc(d_inner * d_state, sizeof(float));
    if (h_temp) {
        ssm_selective_scan_step(scan_out, conv_out, dt_buf,
                               B_vec, C_vec,
                               wt->loop_A_log  ? wt->loop_A_log  : wt->A_log[0],
                               wt->loop_D      ? wt->loop_D      : wt->D[0],
                               h_temp, d_inner, d_state);
        free(h_temp);
    }

    /* Gate */
    for (int i = 0; i < d_inner; i++) {
        gated[i] = ssm_silu(gate_vec[i]) * scan_out[i];
    }

    /* out_proj */
    if (wt->loop_out_proj) {
        ssm_matvec(proj_out, wt->loop_out_proj, gated, d_model, d_inner);
    } else {
        memset(proj_out, 0, d_model * sizeof(float));
    }

    /* Residual add: x = x + mamba2_core(x) */
    for (int i = 0; i < d_model; i++) {
        x[i] += proj_out[i];
    }

    /* Loop norm */
    if (wt->loop_norm_weight) {
        float tmp[2048];
        memcpy(tmp, x, d_model * sizeof(float));
        ssm_rmsnorm(x, tmp, wt->loop_norm_weight, d_model);
    }
}

/* ── Lifeline Injection ───────────────────────────────────────────────────── */

static void ssm_lifeline_inject(
    float       *x,
    const float *x_prompt,
    const float *gate,
    int          d_model
)
{
    /**
     * Prompt Lifeline injection: x = x + gate * x_prompt.
     * This provides the O(1) gradient shortcut during training.
     * At inference, the gate values have been learned to selectively
     * route prompt information through the recursive loop.
     */
    if (!x || !x_prompt || !gate) return;
    for (int i = 0; i < d_model; i++) {
        x[i] += gate[i] * x_prompt[i];
    }
}

/* ── RLF Recursive Inference ──────────────────────────────────────────────── */

void mamba_rlf_infer(
    RlfInferenceResult *result,
    const int          *prompt_ids,
    int                 prompt_len,
    const MambaConfig  *cfg,
    const MambaWeights *wt,
    MambaState         *state
)
{
    /**
     * RLF recursive inference loop.
     *
     * Ported from RecursiveMamba2_v34.forward() inference path:
     *   1. Encode full prompt through all layers → x
     *   2. Save x_prompt = x.clone().detach()  (Prompt Lifeline anchor)
     *   3. For each loop_i in range(MAX_LOOPS):
     *      a. x = lifeline_inject(x, x_prompt)    — gradient shortcut
     *      b. x = loop_rope(x, loop_i)            — RoPE loop encoding
     *      c. x = LoRA_layers(x)                  — top layers forward
     *      d. x = x + mamba2_core(x)              — SSM reasoning block
     *      e. x = loop_norm(x)
     *      f. logits = lm_head(final_norm(x))
     *      g. if argmax == <HALT>: break
     */
    int d_model = cfg->d_model;
    int max_loops = cfg->max_rlf_loops;

    if (!result || !prompt_ids || !cfg || !wt || !state) return;
    memset(result, 0, sizeof(*result));

    /* 1. Encode prompt through full model */
    float *x = (float *)calloc(d_model, sizeof(float));
    float *x_prompt = (float *)calloc(d_model, sizeof(float));
    float *logits = (float *)calloc(cfg->vocab_size, sizeof(float));
    if (!x || !x_prompt || !logits) {
        free(x); free(x_prompt); free(logits);
        return;
    }

    mamba_state_reset(state, cfg);
    for (int t = 0; t < prompt_len; t++) {
        mamba_forward(x, prompt_ids[t], cfg, wt, state);
    }

    /* 2. Save Prompt Lifeline anchor */
    memcpy(x_prompt, x, d_model * sizeof(float));

    /* 3. Recursive inference loop */
    for (int loop_i = 0; loop_i < max_loops && loop_i < 64; loop_i++) {
        /* a. Lifeline injection (gated for ablation testing) */
        if (cfg->lifeline_enabled && wt->lifeline_gate) {
            ssm_lifeline_inject(x, x_prompt, wt->lifeline_gate, d_model);
        }

        /* b. RoPE loop encoding */
        ssm_rope_apply(x, d_model, loop_i, cfg->rope_base);

        /* c. Top layers (base_split to n_layers) */
        float x_buf[2048];
        if (d_model <= 2048) {
            for (int l = cfg->base_split; l < cfg->n_layers; l++) {
                mamba_block_forward(x_buf, x, l, cfg, wt, state);
                memcpy(x, x_buf, d_model * sizeof(float));
            }
        }

        /* d. Mamba2 loop core (reasoning SSM block) */
        ssm_rlf_loop_block(x, cfg, wt);

        /* e. Final norm → logits */
        float x_norm[2048];
        if (wt->final_norm_weight && d_model <= 2048) {
            ssm_rmsnorm(x_norm, x, wt->final_norm_weight, d_model);
        } else {
            memcpy(x_norm, x, d_model * sizeof(float));
        }

        /* f. lm_head: logits = W @ x_norm */
        ssm_matvec(logits, wt->lm_head, x_norm, cfg->vocab_size, d_model);
        ssm_softmax(logits, cfg->vocab_size);

        int pred_tok = ssm_argmax(logits, cfg->vocab_size);
        float conf = logits[pred_tok];
        int is_halt = (pred_tok == cfg->halt_token_id) ? 1 : 0;

        /* Record trace */
        result->trace[loop_i].loop_index     = loop_i;
        result->trace[loop_i].predicted_token = pred_tok;
        result->trace[loop_i].confidence      = conf;
        result->trace[loop_i].is_halt         = is_halt;
        result->n_loops = loop_i + 1;

        if (is_halt) {
            /* Use previous loop's answer as final */
            if (loop_i > 0) {
                result->final_token = result->trace[loop_i - 1].predicted_token;
                result->final_conf  = result->trace[loop_i - 1].confidence;
            }
            break;
        }

        result->final_token = pred_tok;
        result->final_conf  = conf;
    }

    free(x);
    free(x_prompt);
    free(logits);
}

/* ── Standard Greedy Generation ───────────────────────────────────────────── */

int mamba_generate(
    int                *out_tokens,
    int                 max_gen,
    const int          *prompt_ids,
    int                 prompt_len,
    float               temperature,
    const MambaConfig  *cfg,
    const MambaWeights *wt,
    MambaState         *state
)
{
    /**
     * Standard autoregressive generation (no RLF loop).
     * For regular language model use (non-reasoning queries).
     */
    int d_model = cfg->d_model;
    float *x = (float *)calloc(d_model, sizeof(float));
    float *logits = (float *)calloc(cfg->vocab_size, sizeof(float));
    if (!x || !logits) { free(x); free(logits); return 0; }

    mamba_state_reset(state, cfg);

    /* Encode prompt */
    for (int t = 0; t < prompt_len; t++) {
        mamba_forward(x, prompt_ids[t], cfg, wt, state);
    }

    /* Generate */
    int n_gen = 0;
    for (int g = 0; g < max_gen; g++) {
        /* logits = lm_head @ x */
        ssm_matvec(logits, wt->lm_head, x, cfg->vocab_size, d_model);

        int tok = ssm_sample_topk(logits, cfg->vocab_size, temperature, 5);
        out_tokens[n_gen++] = tok;

        /* Check EOS */
        if (tok == 0) break;

        /* Feed back */
        mamba_forward(x, tok, cfg, wt, state);
    }

    free(x);
    free(logits);
    return n_gen;
}

/* ── SSM Telemetry ────────────────────────────────────────────────────────── */

/* Global telemetry instance — OO engines read from this */
static SsmTelemetry g_ssm_telemetry;

SsmTelemetry *ssm_get_telemetry(void)
{
    /**
     * Get pointer to global telemetry struct.
     * OO engines call this to read SSM metrics.
     */
    return &g_ssm_telemetry;
}

void ssm_analyze_lifeline(SsmTelemetry *telem, const float *gate, int d_model)
{
    /**
     * Analyze lifeline gate to compute RAM/ALU partition:
     * - RAM dims: gate > mean+1σ  (amplify prompt — memory retrieval)
     * - ALU dims: gate < mean-1σ  (suppress prompt — computation)
     *
     * From PAPER.md: "16.1% RAM, 2.0% ALU" — the model evolved
     * a von Neumann architecture in its gate vector.
     */
    if (!telem || !gate || d_model <= 0) return;

    /* Compute mean */
    float sum = 0.0f;
    for (int i = 0; i < d_model; i++) sum += gate[i];
    float mean = sum / (float)d_model;

    /* Compute std dev */
    float var = 0.0f;
    for (int i = 0; i < d_model; i++) {
        float d = gate[i] - mean;
        var += d * d;
    }
    float std = sqrtf(var / (float)d_model);

    /* Count RAM and ALU dims */
    float ram_thresh = mean + std;
    float alu_thresh = mean - std;
    int ram = 0, alu = 0;
    for (int i = 0; i < d_model; i++) {
        if (gate[i] > ram_thresh) ram++;
        if (gate[i] < alu_thresh) alu++;
    }

    telem->lifeline_mean = mean;
    telem->lifeline_std  = std;
    telem->lifeline_ram_dims = ram;
    telem->lifeline_alu_dims = alu;
    telem->lifeline_ram_frac = (float)ram / (float)d_model;
    telem->lifeline_alu_frac = (float)alu / (float)d_model;
}

float ssm_entropy(const float *probs, int n)
{
    /**
     * Compute Shannon entropy of probability distribution in bits.
     * H = -Σ p * log2(p)
     * Lower entropy = more certain prediction.
     */
    float h = 0.0f;
    for (int i = 0; i < n; i++) {
        if (probs[i] > 1e-10f) {
            h -= probs[i] * log2f(probs[i]);
        }
    }
    return h;
}

void ssm_compute_output_stats(SsmTelemetry *telem, const float *logits, int vocab_size)
{
    /**
     * Compute output distribution statistics from logits.
     * Modifies a copy — does not alter the input logits.
     */
    if (!telem || !logits || vocab_size <= 0) return;

    /* Find top-1 and top-5 from logits */
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    /* Compute softmax on-the-fly and entropy */
    float sum = 0.0f;
    float top5[5] = {0};
    for (int i = 0; i < vocab_size; i++) {
        float p = expf(logits[i] - max_val);
        sum += p;
        /* Track top 5 */
        for (int j = 0; j < 5; j++) {
            if (p > top5[j]) {
                for (int k = 4; k > j; k--) top5[k] = top5[k-1];
                top5[j] = p;
                break;
            }
        }
    }

    float inv = 1.0f / sum;
    telem->output_top1_prob = top5[0] * inv;
    telem->output_top5_prob = 0.0f;
    for (int j = 0; j < 5; j++) telem->output_top5_prob += top5[j] * inv;

    /* Entropy */
    float h = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        float p = expf(logits[i] - max_val) * inv;
        if (p > 1e-10f) h -= p * log2f(p);
    }
    telem->output_entropy = h;
}
