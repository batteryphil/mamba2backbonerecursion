/*
 * ssm_infer.c — Mamba2 Multi-Head SSM Inference Engine for Bare Metal
 *
 * Pure-C implementation of:
 *   1. Mamba2 multi-head selective scan (fused in_proj, per-head SSM)
 *   2. Recursive Latent Forcing (RLF) inference loop
 *   3. RoPE loop encoding + Prompt Lifeline injection
 *   4. Latent Bridge (low-rank translation)
 *
 * Architecture per block:
 *   in_proj(x) → [z, xBC, dt]
 *   conv1d(xBC) → SiLU → split [x, B, C]
 *   inner_norm(x)
 *   SSM scan per-head: h = A*h + B*x, y = C*h + D*x
 *   gate: y = z * SiLU(y)
 *   out_proj(y)
 *
 * Created for llm-baremetal (Djiby Diop) by batteryphil.
 */

#include "ssm_infer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Portable helpers ─────────────────────────────────────────────────────── */

static inline int  ssm_min_i(int a, int b)   { return a < b ? a : b; }
static inline float ssm_absf(float x)        { return x < 0.0f ? -x : x; }

/* ── Config Default ───────────────────────────────────────────────────────── */

void mamba_config_default(MambaConfig *cfg)
{
    /**
     * Default config for Mamba2-2.7B + RLF w/ Prefix Scratchpad.
     */
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));

    cfg->d_model      = 2560;
    cfg->d_state      = 128;
    cfg->d_conv       = 4;
    cfg->expand        = 2;
    cfg->d_inner       = 5120;
    cfg->n_layers      = 64;
    cfg->vocab_size    = 50288;
    cfg->max_seq_len   = 512;

    /* Mamba2 multi-head */
    cfg->nheads        = 80;
    cfg->headdim       = 64;
    cfg->ngroups       = 1;
    cfg->conv_dim      = cfg->d_inner + 2 * cfg->ngroups * cfg->d_state;  /* 5376 */
    cfg->in_proj_dim   = 2 * cfg->d_inner + 2 * cfg->ngroups * cfg->d_state + cfg->nheads; /* 10576 */

    /* RLF */
    cfg->base_split    = 48;
    cfg->max_rlf_loops = 6;
    cfg->halt_token_id = 50278;
    cfg->rope_base     = 10000;
    cfg->lifeline_enabled = 1;

    /* Scratchpad + Bridge */
    cfg->prefix_m     = 8;
    cfg->bridge_rank  = 64;

    /* Loop engine */
    cfg->loop_nheads   = 20;
    cfg->loop_headdim  = 128;
    cfg->loop_d_state  = 32;
    cfg->loop_d_inner  = 2560;  /* d_model * 1 (expand=1) */
    cfg->loop_conv_dim = cfg->loop_d_inner + 2 * 1 * cfg->loop_d_state; /* 2624 */
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
    for (int i = 0; i < d; i++) ss += x_in[i] * x_in[i];
    ss = 1.0f / sqrtf(ss / (float)d + 1e-5f);
    for (int i = 0; i < d; i++) x_out[i] = weight[i] * x_in[i] * ss;
}

void ssm_softmax(float *v, int n)
{
    /**
     * In-place softmax with numerical stability.
     */
    if (!v || n <= 0) return;
    float max_val = v[0];
    for (int i = 1; i < n; i++) if (v[i] > max_val) max_val = v[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { v[i] = expf(v[i] - max_val); sum += v[i]; }
    if (sum > 0.0f) { float inv = 1.0f / sum; for (int i = 0; i < n; i++) v[i] *= inv; }
}

void ssm_matvec(float *out, const float *W, const float *x, int m, int n)
{
    /**
     * Matrix-vector multiply: out[m] = W[m,n] @ x[n].
     */
    if (!out || !W || !x || m <= 0 || n <= 0) return;
    for (int i = 0; i < m; i++) {
        float acc = 0.0f;
        const float *row = W + (int64_t)i * n;
        for (int j = 0; j < n; j++) acc += row[j] * x[j];
        out[i] = acc;
    }
}

void ssm_matvec_acc(float *out, const float *W, const float *x, int m, int n)
{
    /**
     * Matrix-vector multiply with accumulate.
     */
    if (!out || !W || !x || m <= 0 || n <= 0) return;
    for (int i = 0; i < m; i++) {
        float acc = 0.0f;
        const float *row = W + (int64_t)i * n;
        for (int j = 0; j < n; j++) acc += row[j] * x[j];
        out[i] += acc;
    }
}

void ssm_rope_apply(float *x, int d_model, int loop_index, int rope_base)
{
    /**
     * Apply 1D RoPE rotation for a given loop index.
     */
    if (!x || d_model <= 0) return;
    int half_d = d_model / 2;
    float n = (float)loop_index;
    for (int i = 0; i < half_d; i++) {
        float theta = 1.0f / powf((float)rope_base, (float)(2 * i) / (float)d_model);
        float freq = n * theta;
        float cos_f = cosf(freq);
        float sin_f = sinf(freq);
        int idx0 = 2 * i, idx1 = 2 * i + 1;
        if (idx1 >= d_model) break;
        float x0 = x[idx0], x1 = x[idx1];
        x[idx0] = x0 * cos_f - x1 * sin_f;
        x[idx1] = x0 * sin_f + x1 * cos_f;
    }
}

/* ── State Management ─────────────────────────────────────────────────────── */

int mamba_state_alloc(MambaState *state, const MambaConfig *cfg)
{
    /**
     * Allocate running state buffers for Mamba2 inference.
     */
    if (!state || !cfg) return -1;
    memset(state, 0, sizeof(*state));

    /* Conv buffer: [n_layers * conv_dim * d_conv] */
    size_t conv_bytes = (size_t)cfg->n_layers * cfg->conv_dim * cfg->d_conv * sizeof(float);
    state->conv_buf = (float *)calloc(1, conv_bytes);
    if (!state->conv_buf) return -1;

    /* SSM state: [n_layers * nheads * headdim * d_state] */
    size_t ssm_bytes = (size_t)cfg->n_layers * cfg->nheads * cfg->headdim * cfg->d_state * sizeof(float);
    state->ssm_state = (float *)calloc(1, ssm_bytes);
    if (!state->ssm_state) { free(state->conv_buf); state->conv_buf = NULL; return -1; }

    return 0;
}

void mamba_state_free(MambaState *state)
{
    /**
     * Free all running state buffers.
     */
    if (!state) return;
    free(state->conv_buf); state->conv_buf = NULL;
    free(state->ssm_state); state->ssm_state = NULL;
}

void mamba_state_reset(MambaState *state, const MambaConfig *cfg)
{
    /**
     * Reset state to zeros for a new sequence.
     */
    if (!state || !cfg) return;
    if (state->conv_buf)
        memset(state->conv_buf, 0, (size_t)cfg->n_layers * cfg->conv_dim * cfg->d_conv * sizeof(float));
    if (state->ssm_state)
        memset(state->ssm_state, 0, (size_t)cfg->n_layers * cfg->nheads * cfg->headdim * cfg->d_state * sizeof(float));
    state->conv_pos = 0;
    state->seq_pos = 0;
}

/* ── Mamba2 Block Forward (Single Token) ──────────────────────────────────── */

void mamba2_block_forward(
    float       *x_out,
    const float *x_in,
    int          layer_idx,
    const MambaConfig  *cfg,
    const MambaWeights *wt,
    MambaState         *state
)
{
    /**
     * Forward pass through one Mamba2 multi-head SSM block.
     *
     * 1. RMSNorm(x)
     * 2. in_proj: [in_proj_dim, d_model] @ x → [z, xBC, dt]
     * 3. Conv1d on xBC (causal, d_conv=4)
     * 4. SiLU on xBC
     * 5. Split xBC → [x_ssm, B, C]
     * 6. Inner RMSNorm on x_ssm
     * 7. Per-head SSM scan: h = A*h + B*x, y = h*C + D*x
     * 8. Gate: out = z * SiLU(y_flat)  [wait, Mamba2 gates differently]
     *    Actually: out = y_flat * SiLU(z)
     * 9. out_proj + residual
     */
    int d_model   = cfg->d_model;
    int d_inner   = cfg->d_inner;
    int d_state   = cfg->d_state;
    int d_conv    = cfg->d_conv;
    int nheads    = cfg->nheads;
    int headdim   = cfg->headdim;
    int ngroups   = cfg->ngroups;
    int conv_dim  = cfg->conv_dim;
    int in_proj_d = cfg->in_proj_dim;

    /* Heap buffers */
    float *norm_buf  = (float *)malloc(d_model * sizeof(float));
    float *proj_buf  = (float *)malloc(in_proj_d * sizeof(float));
    float *conv_out  = (float *)malloc(conv_dim * sizeof(float));
    float *y_out     = (float *)malloc(d_inner * sizeof(float));
    float *gated     = (float *)malloc(d_inner * sizeof(float));

    if (!norm_buf || !proj_buf || !conv_out || !y_out || !gated) {
        memcpy(x_out, x_in, d_model * sizeof(float));
        free(norm_buf); free(proj_buf); free(conv_out); free(y_out); free(gated);
        return;
    }

    /* 1. Layer RMSNorm */
    ssm_rmsnorm(norm_buf, x_in, wt->layer_norm[layer_idx], d_model);

    /* 2. in_proj: fused [z, xBC, dt] projection */
    ssm_matvec(proj_buf, wt->in_proj[layer_idx], norm_buf, in_proj_d, d_model);

    /* Split: z[d_inner], xBC[conv_dim], dt[nheads] */
    float *z_vec    = proj_buf;                          /* [d_inner] */
    float *xBC_vec  = proj_buf + d_inner;                /* [conv_dim] = d_inner + 2*ngroups*d_state */
    float *dt_vec   = proj_buf + d_inner + conv_dim;     /* [nheads] */

    /* dt = softplus(dt + dt_bias) */
    if (wt->dt_bias[layer_idx]) {
        for (int i = 0; i < nheads; i++) {
            dt_vec[i] += wt->dt_bias[layer_idx][i];
            dt_vec[i] = logf(1.0f + expf(dt_vec[i]));
        }
    }

    /* 3. Conv1d on xBC (causal with circular buffer) */
    float *layer_conv = state->conv_buf + (size_t)layer_idx * conv_dim * d_conv;
    int pos = state->conv_pos;

    /* Store current xBC into circular buffer */
    for (int d = 0; d < conv_dim; d++) {
        layer_conv[d * d_conv + pos] = xBC_vec[d];
    }

    /* Compute convolution */
    for (int d = 0; d < conv_dim; d++) {
        float acc = wt->conv1d_bias[layer_idx] ? wt->conv1d_bias[layer_idx][d] : 0.0f;
        for (int k = 0; k < d_conv; k++) {
            int buf_idx = (pos - d_conv + 1 + k + d_conv * 256) % d_conv;
            acc += layer_conv[d * d_conv + buf_idx] * wt->conv1d_weight[layer_idx][d * d_conv + k];
        }
        conv_out[d] = acc;
    }

    /* 4. SiLU on the conv output */
    for (int d = 0; d < conv_dim; d++) {
        conv_out[d] = ssm_silu(conv_out[d]);
    }

    /* 5. Split conv_out → [x_ssm, B, C] */
    float *x_ssm = conv_out;                                   /* [d_inner] */
    float *B_flat = conv_out + d_inner;                         /* [ngroups * d_state] */
    float *C_flat = conv_out + d_inner + ngroups * d_state;     /* [ngroups * d_state] */

    /* 6. Inner RMSNorm on x_ssm */
    if (wt->inner_norm[layer_idx]) {
        float *x_tmp = (float *)malloc(d_inner * sizeof(float));
        if (x_tmp) {
            memcpy(x_tmp, x_ssm, d_inner * sizeof(float));
            ssm_rmsnorm(x_ssm, x_tmp, wt->inner_norm[layer_idx], d_inner);
            free(x_tmp);
        }
    }

    /* 7. Per-head SSM scan — single token (T=1), O(1) per step
     *
     * ONNX #7689 State Interface Contract (gated SSM / "gated" update_rule):
     *
     *   State:   S ∈ R^[B × H × d_k × d_v]  — FIXED shape, never grows
     *   Per-token update (recurrent decode, T=1):
     *
     *     S_t  = exp(g_t)  ·  S_{t-1}  +  B_bar_t ⊗ x_t
     *     o_t  = scale  ·  q_t^T  S_t
     *
     *   where:
     *     g_t     = dt_h * A_val   (log-space decay, pre-exponentiated below as r)
     *     B_bar_t = dt_h * B_vec   (discretised input projection)
     *     x_t     = x_h[d]         (head input scalar)
     *     o_t     = y_h (accumulated across d_state)
     *
     *   This maps directly to the code below:
     *     r       = expf(dt_h * A_val)      ← exp(g_t)
     *     h[n]    = r * h[n] + b_bar * x_t  ← S_t update
     *     y_t    += h[n] * C_vec[n]          ← output read
     *
     *   Per MLX #980: state shape [n_layers × nheads × headdim × d_state]
     *   is IMMUTABLE — never trim, never slice at an arbitrary token boundary.
     *   Only mamba_state_reset() (full wipe) is a safe rollback operation.
     *
     * State layout: ssm_state[layer, head, d, n]
     *   flattened as: layer_idx * nheads * headdim * d_state
     *                 + head * headdim * d_state
     *                 + d * d_state + n
     */
    float *layer_h = state->ssm_state +
        (size_t)layer_idx * nheads * headdim * d_state;

    int heads_per_group = nheads / ngroups;

    for (int head = 0; head < nheads; head++) {
        int grp = head / heads_per_group;
        float *B_vec = B_flat + grp * d_state;
        float *C_vec = C_flat + grp * d_state;
        float *h     = layer_h + (size_t)head * headdim * d_state;
        float *x_h   = x_ssm + head * headdim;
        float *y_h   = y_out + head * headdim;

        float A_val = -expf(wt->A_log[layer_idx][head]);
        float dt_h  = dt_vec[head];
        float D_h   = wt->D[layer_idx] ? wt->D[layer_idx][head] : 1.0f;

        for (int d = 0; d < headdim; d++) {
            float x_t = x_h[d];
            float y_t = 0.0f;

            for (int n = 0; n < d_state; n++) {
                float r = expf(dt_h * A_val);
                float b_bar = dt_h * B_vec[n];
                h[d * d_state + n] = r * h[d * d_state + n] + b_bar * x_t;
                y_t += h[d * d_state + n] * C_vec[n];
            }

            y_h[d] = y_t + D_h * x_t;
        }
    }

    /* 8. Gate: out = y * SiLU(z) */
    for (int i = 0; i < d_inner; i++) {
        gated[i] = y_out[i] * ssm_silu(z_vec[i]);
    }

    /* 9. out_proj + residual */
    ssm_matvec(x_out, wt->out_proj[layer_idx], gated, d_model, d_inner);
    for (int i = 0; i < d_model; i++) {
        x_out[i] += x_in[i];
    }

    free(norm_buf); free(proj_buf); free(conv_out); free(y_out); free(gated);
}

/* ── Full Model Forward ───────────────────────────────────────────────────── */

void mamba_forward(
    float       *x,
    int          token_id,
    const MambaConfig  *cfg,
    const MambaWeights *wt,
    MambaState         *state
)
{
    /**
     * Full Mamba2 forward pass for a single token.
     */
    int d_model = cfg->d_model;

    /* 1. Token embedding */
    if (token_id >= 0 && token_id < cfg->vocab_size && wt->token_embedding) {
        const float *emb = wt->token_embedding + (size_t)token_id * d_model;
        memcpy(x, emb, d_model * sizeof(float));
    }

    /* 2. Layer-by-layer forward */
    float *x_buf = (float *)malloc(d_model * sizeof(float));
    if (!x_buf) return;

    for (int l = 0; l < cfg->n_layers; l++) {
        mamba2_block_forward(x_buf, x, l, cfg, wt, state);
        memcpy(x, x_buf, d_model * sizeof(float));
    }

    /* 3. Final RMSNorm */
    if (wt->final_norm_weight) {
        memcpy(x_buf, x, d_model * sizeof(float));
        ssm_rmsnorm(x, x_buf, wt->final_norm_weight, d_model);
    }
    free(x_buf);

    state->conv_pos = (state->conv_pos + 1) % cfg->d_conv;
    state->seq_pos++;
}

/* ── Argmax / Sampling ────────────────────────────────────────────────────── */

static int ssm_argmax(const float *v, int n)
{
    /**
     * Return index of maximum value.
     */
    if (!v || n <= 0) return 0;
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

/* ── Lifeline Injection ───────────────────────────────────────────────────── */

static void ssm_lifeline_inject(float *x, const float *x_prompt, const float *gate, int d)
{
    /**
     * Prompt Lifeline: x += gate * x_prompt.
     */
    if (!x || !x_prompt || !gate) return;
    for (int i = 0; i < d; i++) x[i] += gate[i] * x_prompt[i];
}

/* ── RLF Loop Core Block ─────────────────────────────────────────────────── */

static void ssm_rlf_loop_block(
    float *x,
    const MambaConfig *cfg,
    const MambaWeights *wt
)
{
    /**
     * Mamba2 loop core block (recursive reasoning engine).
     * x = x + loop_mamba2(x), then loop_norm(x).
     */
    int d_model      = cfg->d_model;
    int loop_nheads   = cfg->loop_nheads;
    int loop_headdim  = cfg->loop_headdim;
    int loop_d_state  = cfg->loop_d_state;
    int loop_d_inner  = cfg->loop_d_inner;
    int loop_conv_dim = cfg->loop_conv_dim;
    int d_conv        = cfg->d_conv;

    /* Compute in_proj_dim for loop */
    int loop_in_proj_dim = 2 * loop_d_inner + 2 * 1 * loop_d_state + loop_nheads;

    float *proj_buf  = (float *)malloc(loop_in_proj_dim * sizeof(float));
    float *conv_out  = (float *)malloc(loop_conv_dim * sizeof(float));
    float *y_out     = (float *)malloc(loop_d_inner * sizeof(float));
    float *proj_out  = (float *)malloc(d_model * sizeof(float));

    if (!proj_buf || !conv_out || !y_out || !proj_out) {
        free(proj_buf); free(conv_out); free(y_out); free(proj_out);
        return;
    }

    /* in_proj */
    if (wt->loop_in_proj)
        ssm_matvec(proj_buf, wt->loop_in_proj, x, loop_in_proj_dim, d_model);
    else
        memset(proj_buf, 0, loop_in_proj_dim * sizeof(float));

    float *z_vec   = proj_buf;
    float *xBC_vec = proj_buf + loop_d_inner;
    float *dt_vec  = proj_buf + loop_d_inner + loop_conv_dim;

    /* dt = softplus(dt + dt_bias) */
    if (wt->loop_dt_bias) {
        for (int i = 0; i < loop_nheads; i++) {
            dt_vec[i] += wt->loop_dt_bias[i];
            dt_vec[i] = logf(1.0f + expf(dt_vec[i]));
        }
    }

    /* Skip conv for loop (just apply SiLU directly) */
    for (int i = 0; i < loop_conv_dim; i++) {
        conv_out[i] = ssm_silu(xBC_vec[i]);
    }

    /* Split → x_ssm, B, C */
    float *x_ssm = conv_out;
    float *B_flat = conv_out + loop_d_inner;
    float *C_flat = conv_out + loop_d_inner + loop_d_state;

    /* Inner norm */
    if (wt->loop_inner_norm) {
        float *tmp = (float *)malloc(loop_d_inner * sizeof(float));
        if (tmp) {
            memcpy(tmp, x_ssm, loop_d_inner * sizeof(float));
            ssm_rmsnorm(x_ssm, tmp, wt->loop_inner_norm, loop_d_inner);
            free(tmp);
        }
    }

    /* Per-head SSM scan (stateless — fresh h each time) */
    float *h_temp = (float *)calloc((size_t)loop_nheads * loop_headdim * loop_d_state, sizeof(float));
    if (h_temp) {
        for (int head = 0; head < loop_nheads; head++) {
            float *B_vec = B_flat;  /* ngroups=1 so all heads share */
            float *C_vec = C_flat;
            float *h = h_temp + (size_t)head * loop_headdim * loop_d_state;
            float *x_h = x_ssm + head * loop_headdim;
            float *y_h = y_out + head * loop_headdim;

            float A_val = wt->loop_A_log ? -expf(wt->loop_A_log[head]) : -1.0f;
            float dt_h  = dt_vec[head];
            float D_h   = wt->loop_D ? wt->loop_D[head] : 1.0f;

            for (int d = 0; d < loop_headdim; d++) {
                float x_t = x_h[d];
                float y_t = 0.0f;
                for (int n = 0; n < loop_d_state; n++) {
                    float r = expf(dt_h * A_val);
                    h[d * loop_d_state + n] = r * h[d * loop_d_state + n] + dt_h * B_vec[n] * x_t;
                    y_t += h[d * loop_d_state + n] * C_vec[n];
                }
                y_h[d] = y_t + D_h * x_t;
            }
        }
        free(h_temp);
    }

    /* Gate: y * SiLU(z) */
    float *gated = y_out;  /* reuse in-place */
    for (int i = 0; i < loop_d_inner; i++) {
        gated[i] = y_out[i] * ssm_silu(z_vec[i]);
    }

    /* out_proj + residual */
    if (wt->loop_out_proj)
        ssm_matvec(proj_out, wt->loop_out_proj, gated, d_model, loop_d_inner);
    else
        memset(proj_out, 0, d_model * sizeof(float));

    for (int i = 0; i < d_model; i++) x[i] += proj_out[i];

    /* Loop norm */
    if (wt->loop_norm_weight) {
        float *tmp = (float *)malloc(d_model * sizeof(float));
        if (tmp) {
            memcpy(tmp, x, d_model * sizeof(float));
            ssm_rmsnorm(x, tmp, wt->loop_norm_weight, d_model);
            free(tmp);
        }
    }

    free(proj_buf); free(conv_out); free(y_out); free(proj_out);
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
     * RLF recursive inference loop with Latent Bridge.
     */
    int d_model    = cfg->d_model;
    int max_loops  = cfg->max_rlf_loops;
    int bridge_rank = cfg->bridge_rank;

    if (!result || !prompt_ids || !cfg || !wt || !state) return;
    memset(result, 0, sizeof(*result));

    float *x         = (float *)calloc(d_model, sizeof(float));
    float *x_prompt  = (float *)calloc(d_model, sizeof(float));
    float *logits    = (float *)calloc(cfg->vocab_size, sizeof(float));
    float *x_buf     = (float *)malloc(d_model * sizeof(float));
    float *x_norm    = (float *)malloc(d_model * sizeof(float));
    float *bridge_t  = bridge_rank > 0 ? (float *)malloc(bridge_rank * sizeof(float)) : NULL;

    if (!x || !x_prompt || !logits || !x_buf || !x_norm) {
        free(x); free(x_prompt); free(logits); free(x_buf); free(x_norm); free(bridge_t);
        return;
    }

    /* 1. Encode prompt through full model */
    mamba_state_reset(state, cfg);
    for (int t = 0; t < prompt_len; t++) {
        mamba_forward(x, prompt_ids[t], cfg, wt, state);
    }

    /* 2. Save Prompt Lifeline anchor */
    memcpy(x_prompt, x, d_model * sizeof(float));

    /* 3. Recursive inference loop */
    for (int loop_i = 0; loop_i < max_loops && loop_i < 64; loop_i++) {
        /* a. Lifeline injection */
        if (cfg->lifeline_enabled && wt->lifeline_gate)
            ssm_lifeline_inject(x, x_prompt, wt->lifeline_gate, d_model);

        /* b. RoPE loop encoding */
        ssm_rope_apply(x, d_model, loop_i, cfg->rope_base);

        /* c. Top layers (base_split to n_layers) */
        for (int l = cfg->base_split; l < cfg->n_layers; l++) {
            mamba2_block_forward(x_buf, x, l, cfg, wt, state);
            memcpy(x, x_buf, d_model * sizeof(float));
        }

        /* d. Loop engine */
        ssm_rlf_loop_block(x, cfg, wt);

        /* e. Latent Bridge: x += bridge_up(bridge_down(x)) */
        if (bridge_rank > 0 && bridge_t && wt->bridge_down && wt->bridge_up) {
            ssm_matvec(bridge_t, wt->bridge_down, x, bridge_rank, d_model);
            ssm_matvec(x_norm, wt->bridge_up, bridge_t, d_model, bridge_rank);
            for (int i = 0; i < d_model; i++) x[i] += x_norm[i];
        }

        /* f. Final norm → logits */
        if (wt->final_norm_weight)
            ssm_rmsnorm(x_norm, x, wt->final_norm_weight, d_model);
        else
            memcpy(x_norm, x, d_model * sizeof(float));

        ssm_matvec(logits, wt->lm_head, x_norm, cfg->vocab_size, d_model);
        ssm_softmax(logits, cfg->vocab_size);

        int pred = ssm_argmax(logits, cfg->vocab_size);
        float conf = logits[pred];

        result->trace[loop_i].loop_index     = loop_i;
        result->trace[loop_i].predicted_token = pred;
        result->trace[loop_i].confidence      = conf;
        result->trace[loop_i].is_halt         = (pred == cfg->halt_token_id) ? 1 : 0;
        result->n_loops = loop_i + 1;

        if (pred == cfg->halt_token_id) {
            if (loop_i > 0) {
                result->final_token = result->trace[loop_i - 1].predicted_token;
                result->final_conf  = result->trace[loop_i - 1].confidence;
            }
            break;
        }
        result->final_token = pred;
        result->final_conf  = conf;
    }

    free(x); free(x_prompt); free(logits); free(x_buf); free(x_norm); free(bridge_t);
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
     * Greedy autoregressive generation.
     */
    int d_model = cfg->d_model;
    float *x      = (float *)calloc(d_model, sizeof(float));
    float *x_norm = (float *)malloc(d_model * sizeof(float));
    float *logits = (float *)calloc(cfg->vocab_size, sizeof(float));
    if (!x || !x_norm || !logits) { free(x); free(x_norm); free(logits); return 0; }

    mamba_state_reset(state, cfg);

    /* Process prompt */
    for (int t = 0; t < prompt_len; t++)
        mamba_forward(x, prompt_ids[t], cfg, wt, state);

    /* Generate */
    int gen = 0;
    for (int i = 0; i < max_gen; i++) {
        if (wt->final_norm_weight)
            ssm_rmsnorm(x_norm, x, wt->final_norm_weight, d_model);
        else
            memcpy(x_norm, x, d_model * sizeof(float));

        ssm_matvec(logits, wt->lm_head, x_norm, cfg->vocab_size, d_model);

        if (temperature > 0.0f) {
            for (int v = 0; v < cfg->vocab_size; v++) logits[v] /= temperature;
        }
        ssm_softmax(logits, cfg->vocab_size);

        int tok = ssm_argmax(logits, cfg->vocab_size);
        out_tokens[gen++] = tok;

        mamba_forward(x, tok, cfg, wt, state);
    }

    free(x); free(x_norm); free(logits);
    return gen;
}

/* ── Telemetry helpers ────────────────────────────────────────────────────── */

void ssm_analyze_lifeline(SsmTelemetry *telem, const float *gate, int d_model)
{
    /**
     * Analyze lifeline gate for RAM/ALU partitioning.
     */
    if (!telem || !gate || d_model <= 0) return;
    float sum = 0.0f, sum2 = 0.0f;
    for (int i = 0; i < d_model; i++) { sum += gate[i]; sum2 += gate[i]*gate[i]; }
    telem->lifeline_mean = sum / d_model;
    telem->lifeline_std = sqrtf(sum2/d_model - telem->lifeline_mean*telem->lifeline_mean);
}

float ssm_entropy(const float *probs, int n)
{
    /**
     * Compute entropy in bits.
     */
    if (!probs || n <= 0) return 0.0f;
    float e = 0.0f;
    for (int i = 0; i < n; i++) {
        if (probs[i] > 1e-10f) e -= probs[i] * log2f(probs[i]);
    }
    return e;
}
