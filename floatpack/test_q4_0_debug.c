#include "floatpack.h"
#include "fp4_format.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define QK4_0 32
typedef struct {
    uint16_t d;
    uint8_t qs[QK4_0 / 2];
} block_q4_0;

// Proper fp16 -> fp32
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        // Subnormal: normalize
        if (mant == 0) {
            f = sign;
        } else {
            int e = -1;
            uint32_t m = mant;
            while (!(m & 0x400)) { m <<= 1; e--; }
            f = sign | ((uint32_t)(e + 127 + 14) << 23) | ((m & 0x3FF) << 13);
        }
    } else if (exp == 31) {
        f = sign | (0xFF << 23) | (mant << 13);
    } else {
        // Normal: exp = e - 15, so fp32 exp = (e - 15) + 127 = e + 112
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

static inline uint16_t fp32_to_fp16(float x) {
    uint32_t f;
    memcpy(&f, &x, sizeof(f));
    uint32_t sign = (f >> 31) & 1;
    int32_t exp = ((f >> 23) & 0xFF) - 127;
    uint32_t mant = f & 0x7FFFFF;
    if (exp > 15) { exp = 15; mant = 0; }
    if (exp < -24) { return sign << 15; }
    if (exp < -14) {
        // Subnormal
        uint16_t h = (uint16_t)(sign << 15);
        mant = (mant | 0x800000) >> (14 + 127 - (exp + 127));
        h = (uint16_t)(h | (mant >> 13));
        return h;
    }
    return (uint16_t)((sign << 15) | ((exp + 15) << 10) | (mant >> 13));
}

// Reference F32 matmul (B is KxN row-major)
static void matmul_f32_ref(int M, int N, int K,
                           const float *A, const float *B, float *C) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0;
            for (int k = 0; k < K; k++)
                acc += A[i*K + k] * B[k*N + j];
            C[i*N + j] = acc;
        }
}

static float max_abs_err(const float *a, const float *b, int n) {
    float max_err = 0;
    for (int i = 0; i < n; i++) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

int main(void) {
    printf("=== Floatpack Q4_0 POC Debug ===\n\n");

    const int M = 2, K = 32, N = 4;
    int n_blocks = K / QK4_0;

    float *A_f32 = (float *)malloc((size_t)M * K * sizeof(float));
    float *B_f32 = (float *)malloc((size_t)K * N * sizeof(float));
    float *C_ref = (float *)malloc((size_t)M * N * sizeof(float));
    float *C_fp  = (float *)malloc((size_t)M * N * sizeof(float));
    block_q4_0 *A_q4 = (block_q4_0 *)malloc((size_t)M * n_blocks * sizeof(block_q4_0));

    // Simple values
    for (int i = 0; i < M * K; i++) A_f32[i] = (float)(i % 8) - 3.5f;
    // B identity-like
    memset(B_f32, 0, (size_t)K * N * sizeof(float));
    for (int k = 0; k < K && k < N; k++) B_f32[k * N + k] = 1.0f;

    // Compute reference
    matmul_f32_ref(M, N, K, A_f32, B_f32, C_ref);

    // Create Q4_0 blocks
    for (int row = 0; row < M; row++) {
        for (int b = 0; b < n_blocks; b++) {
            block_q4_0 *blk = &A_q4[row * n_blocks + b];
            float max_val = 0;
            for (int i = 0; i < QK4_0; i++) {
                float v = fabsf(A_f32[row * K + b * QK4_0 + i]);
                if (v > max_val) max_val = v;
            }
            float d = (max_val < 1e-10f) ? 1.0f : (max_val / 7.0f);
            blk->d = fp32_to_fp16(d);
            float d_decoded = fp16_to_fp32(blk->d);
            printf("Row %d Block %d: d=%f (stored=0x%04x decoded=%f)\n",
                   row, b, d, blk->d, d_decoded);

            // Verify fp16 roundtrip
            uint16_t reenc = fp32_to_fp16(d_decoded);
            float redec = fp16_to_fp32(reenc);
            printf("  fp16 roundtrip: %f -> 0x%04x -> %f\n", d_decoded, reenc, redec);

            for (int i = 0; i < QK4_0; i++) {
                float val = A_f32[row * K + b * QK4_0 + i];
                int q = (int)(val / d + 8.0f + 0.5f);
                if (q < 0) q = 0;
                if (q > 15) q = 15;
                if (i & 1)
                    blk->qs[i/2] = (blk->qs[i/2] & 0x0F) | (q << 4);
                else
                    blk->qs[i/2] = (blk->qs[i/2] & 0xF0) | q;
            }
        }
    }

    // Verify Q4 encoding: decode back to float and compare
    printf("\nQ4_0 roundtrip check (first 8 values):\n");
    for (int i = 0; i < 8; i++) {
        float d = fp16_to_fp32(A_q4[0].d);
        int nib = (A_q4[0].qs[i>>1] >> ((i&1)?4:0)) & 0xF;
        float decoded = d * (float)(nib - 8);
        printf("  A[0][%d] = %.2f -> q=%d -> decoded=%.2f\n",
               i, A_f32[i], nib, decoded);
    }

    // Compute with floatpack
    if (floatpack_init() != 0) {
        printf("floatpack_init FAILED\n");
        return 1;
    }
    printf("\nfloatpack OK\n");

    int ret = floatpack_matmul_q4_0(M, N, K, A_q4, B_f32, C_fp);
    printf("floatpack_matmul_q4_0 returned %d\n", ret);
    if (ret == 0) {
        printf("\nResults:\n");
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                printf("  C[%d][%d] ref=%.4f fp4=%.4f (diff=%.4f)\n",
                       i, j, C_ref[i*N+j], C_fp[i*N+j],
                       fabsf(C_ref[i*N+j] - C_fp[i*N+j]));
            }
        float err = max_abs_err(C_ref, C_fp, M*N);
        printf("\nMax abs error: %.4f\n", err);
        printf("%s\n", err < 3.0f ? "PASSED" : "FAILED (high error)");
    }

    floatpack_cleanup();
    free(A_f32); free(B_f32); free(C_ref); free(C_fp); free(A_q4);
    return ret ? 1 : 0;
}
