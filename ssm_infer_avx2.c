/*
 * ssm_infer_avx2.c — AVX2/SSE4 SIMD-Optimized SSM Inference Kernels
 *
 * Drop-in replacements for scalar functions in ssm_infer.c.
 * Process 8 floats per cycle using 256-bit AVX2 registers.
 * Auto-dispatch: call ssm_detect_simd() at boot to select fastest path.
 *
 * Requires: -mavx2 -mfma compiler flags.
 */

#include "ssm_infer.h"
#include <string.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define HAS_AVX2 1
#else
#define HAS_AVX2 0
#endif

/* ── CPUID-based Feature Detection ───────────────────────────────────────── */

static int g_has_avx2 = 0;

void ssm_detect_simd(void)
{
    /**
     * Detect AVX2+FMA support at runtime via CPUID.
     * Call once at boot before any inference.
     */
#if HAS_AVX2
  #if defined(__GNUC__) || defined(__clang__)
    unsigned int eax, ebx, ecx, edx;
    __asm__ __volatile__("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                         : "a"(7), "c"(0));
    g_has_avx2 = (ebx >> 5) & 1;  /* AVX2 = bit 5 of EBX */
  #else
    g_has_avx2 = 1;  /* Assume supported if compiled with -mavx2 */
  #endif
#else
    g_has_avx2 = 0;
#endif
}

int ssm_has_avx2(void)
{
    /**
     * Check if AVX2 is available on this CPU.
     */
    return g_has_avx2;
}

/* ── AVX2 Implementations ─────────────────────────────────────────────────── */

#if HAS_AVX2

void ssm_matvec_avx2(float *out, const float *W, const float *x, int m, int n)
{
    /**
     * SIMD matrix-vector multiply: out[m] = W[m,n] @ x[n].
     * Process 8 elements at a time with FMA (fused multiply-add).
     */
    int n8 = n & ~7;  /* Round down to multiple of 8 */

    for (int i = 0; i < m; i++) {
        __m256 acc = _mm256_setzero_ps();
        const float *row = W + i * n;

        /* Vectorized inner loop */
        for (int j = 0; j < n8; j += 8) {
            __m256 w = _mm256_loadu_ps(row + j);
            __m256 v = _mm256_loadu_ps(x + j);
            acc = _mm256_fmadd_ps(w, v, acc);
        }

        /* Horizontal sum of 8 floats in acc */
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 sum4 = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(sum4);
        __m128 sum2 = _mm_add_ps(sum4, shuf);
        shuf = _mm_movehl_ps(shuf, sum2);
        __m128 sum1 = _mm_add_ss(sum2, shuf);
        float result = _mm_cvtss_f32(sum1);

        /* Scalar tail */
        for (int j = n8; j < n; j++) {
            result += row[j] * x[j];
        }

        out[i] = result;
    }
}

void ssm_matvec_acc_avx2(float *out, const float *W, const float *x, int m, int n)
{
    /**
     * SIMD matrix-vector multiply-accumulate: out[m] += W[m,n] @ x[n].
     */
    int n8 = n & ~7;

    for (int i = 0; i < m; i++) {
        __m256 acc = _mm256_setzero_ps();
        const float *row = W + i * n;

        for (int j = 0; j < n8; j += 8) {
            __m256 w = _mm256_loadu_ps(row + j);
            __m256 v = _mm256_loadu_ps(x + j);
            acc = _mm256_fmadd_ps(w, v, acc);
        }

        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 sum4 = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(sum4);
        __m128 sum2 = _mm_add_ps(sum4, shuf);
        shuf = _mm_movehl_ps(shuf, sum2);
        __m128 sum1 = _mm_add_ss(sum2, shuf);
        float result = _mm_cvtss_f32(sum1);

        for (int j = n8; j < n; j++) {
            result += row[j] * x[j];
        }

        out[i] += result;
    }
}

void ssm_rmsnorm_avx2(float *x_out, const float *x_in, const float *weight, int d)
{
    /**
     * SIMD RMSNorm: x_out[i] = weight[i] * x_in[i] / rms(x_in).
     */
    int d8 = d & ~7;

    /* Compute sum of squares */
    __m256 ss = _mm256_setzero_ps();
    for (int i = 0; i < d8; i += 8) {
        __m256 v = _mm256_loadu_ps(x_in + i);
        ss = _mm256_fmadd_ps(v, v, ss);
    }
    __m128 hi = _mm256_extractf128_ps(ss, 1);
    __m128 lo = _mm256_castps256_ps128(ss);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum4);
    __m128 sum2 = _mm_add_ps(sum4, shuf);
    shuf = _mm_movehl_ps(shuf, sum2);
    __m128 sum1 = _mm_add_ss(sum2, shuf);
    float sumsq = _mm_cvtss_f32(sum1);

    /* Scalar tail for sum of squares */
    for (int i = d8; i < d; i++) {
        sumsq += x_in[i] * x_in[i];
    }

    float rms_inv = 1.0f / sqrtf(sumsq / (float)d + 1e-5f);
    __m256 scale = _mm256_set1_ps(rms_inv);

    /* Normalize and scale */
    for (int i = 0; i < d8; i += 8) {
        __m256 v = _mm256_loadu_ps(x_in + i);
        __m256 w = _mm256_loadu_ps(weight + i);
        __m256 normed = _mm256_mul_ps(v, scale);
        __m256 result = _mm256_mul_ps(normed, w);
        _mm256_storeu_ps(x_out + i, result);
    }

    /* Scalar tail */
    for (int i = d8; i < d; i++) {
        x_out[i] = weight[i] * x_in[i] * rms_inv;
    }
}

void ssm_softmax_avx2(float *v, int n)
{
    /**
     * SIMD in-place softmax over v[n].
     * Uses scalar exp() since there's no AVX2 intrinsic for it.
     */
    int n8 = n & ~7;

    /* Find max */
    __m256 vmax = _mm256_set1_ps(-1e30f);
    for (int i = 0; i < n8; i += 8) {
        __m256 val = _mm256_loadu_ps(v + i);
        vmax = _mm256_max_ps(vmax, val);
    }
    /* Horizontal max */
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 mx4 = _mm_max_ps(lo, hi);
    mx4 = _mm_max_ps(mx4, _mm_shuffle_ps(mx4, mx4, _MM_SHUFFLE(2,3,0,1)));
    mx4 = _mm_max_ps(mx4, _mm_shuffle_ps(mx4, mx4, _MM_SHUFFLE(1,0,3,2)));
    float max_val = _mm_cvtss_f32(mx4);
    for (int i = n8; i < n; i++) {
        if (v[i] > max_val) max_val = v[i];
    }

    /* exp(v - max) and sum — scalar (no AVX2 exp) */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        v[i] = expf(v[i] - max_val);
        sum += v[i];
    }

    /* Normalize */
    float inv_sum = 1.0f / sum;
    __m256 inv = _mm256_set1_ps(inv_sum);
    for (int i = 0; i < n8; i += 8) {
        __m256 val = _mm256_loadu_ps(v + i);
        _mm256_storeu_ps(v + i, _mm256_mul_ps(val, inv));
    }
    for (int i = n8; i < n; i++) {
        v[i] *= inv_sum;
    }
}

#else /* No AVX2 — provide stubs that fall through to scalar */

void ssm_matvec_avx2(float *out, const float *W, const float *x, int m, int n)
{
    /**
     * Fallback: call scalar implementation.
     */
    ssm_matvec(out, W, x, m, n);
}

void ssm_matvec_acc_avx2(float *out, const float *W, const float *x, int m, int n)
{
    /**
     * Fallback: call scalar implementation.
     */
    ssm_matvec_acc(out, W, x, m, n);
}

void ssm_rmsnorm_avx2(float *x_out, const float *x_in, const float *weight, int d)
{
    /**
     * Fallback: call scalar implementation.
     */
    ssm_rmsnorm(x_out, x_in, weight, d);
}

void ssm_softmax_avx2(float *v, int n)
{
    /**
     * Fallback: call scalar implementation.
     */
    ssm_softmax(v, n);
}

#endif /* HAS_AVX2 */
