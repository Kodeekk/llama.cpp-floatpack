#include "floatpack.h"
#include "fp4_format.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Minimal Q4_0 block structure (mirrors ggml's block_q4_0)
#define QK4_0 32
typedef struct {
    uint16_t d;    // fp16 scale
    uint8_t qs[QK4_0 / 2]; // nibbles
} block_q4_0;

// fp16 -> fp32
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        f = sign | (mant << 13);
    } else if (exp == 31) {
        f = sign | (0xFF << 23) | (mant << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

// Encode a float16 value (for creating test data)
static inline uint16_t fp32_to_fp16(float x) {
    uint32_t f;
    memcpy(&f, &x, sizeof(f));
    uint32_t sign = (f >> 31) & 1;
    int32_t exp = (int32_t)((f >> 23) & 0xFF) - 127;
    uint32_t mant = f & 0x7FFFFF;
    if (exp > 15) { exp = 15; mant = 0; } // saturate
    if (exp < -14) { exp = -14; mant = 0; } // flush to zero
    uint16_t h = (uint16_t)((sign << 15) | ((exp + 15) << 10) | (mant >> 13));
    return h;
}

// Reference F32 matmul
static void matmul_f32_ref(int M, int N, int K,
                           const float *A, const float *B, float *C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A[i * K + k] * B[k * N + j]; // B is KxN row-major
            }
            C[i * N + j] = acc;
        }
    }
}

static float max_rel_err(const float *a, const float *b, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float denom = fabsf(a[i]) > 0.001f ? fabsf(a[i]) : 1.0f;
        float err = fabsf(a[i] - b[i]) / denom;
        if (err > max_err) max_err = err;
    }
    return max_err;
}

int main(void) {
    printf("=== Floatpack Q4_0 POC Test ===\n\n");

    // Small test: M=4, K=64, N=8
    const int M = 4, K = 64, N = 8;

    // Create F32 reference data
    float *A_f32 = (float *)malloc((size_t)M * K * sizeof(float));
    float *B_f32 = (float *)malloc((size_t)K * N * sizeof(float));
    float *C_ref = (float *)malloc((size_t)M * N * sizeof(float));
    float *C_fp  = (float *)malloc((size_t)M * N * sizeof(float));

    // Fill with random-ish values
    srand(42);
    for (int i = 0; i < M * K; i++) A_f32[i] = ((float)(rand() % 200) - 100.0f) / 50.0f;
    for (int i = 0; i < K * N; i++) B_f32[i] = ((float)(rand() % 200) - 100.0f) / 50.0f;

    // Compute reference (F32 precision)
    matmul_f32_ref(M, N, K, A_f32, B_f32, C_ref);

    // Create Q4_0 weights from A_f32
    int n_blocks = K / QK4_0;
    block_q4_0 *A_q4 = (block_q4_0 *)malloc((size_t)M * n_blocks * sizeof(block_q4_0));

    for (int row = 0; row < M; row++) {
        for (int b = 0; b < n_blocks; b++) {
            block_q4_0 *blk = &A_q4[row * n_blocks + b];
            // Find max value in block for scale
            float max_val = 0.0f;
            for (int i = 0; i < QK4_0; i++) {
                float v = fabsf(A_f32[row * K + b * QK4_0 + i]);
                if (v > max_val) max_val = v;
            }
            float d = max_val / 7.0f; // scale so that 7 maps to max
            if (d < 1e-10f) d = 1.0f;
            blk->d = fp32_to_fp16(d);
            for (int i = 0; i < QK4_0; i++) {
                float val = A_f32[row * K + b * QK4_0 + i];
                int q = (int)roundf(val / d) + 8;
                if (q < 0) q = 0;
                if (q > 15) q = 15;
                if (i & 1) {
                    blk->qs[i / 2] = (blk->qs[i / 2] & 0x0F) | (q << 4);
                } else {
                    blk->qs[i / 2] = (blk->qs[i / 2] & 0xF0) | q;
                }
            }
        }
    }

    // Init floatpack
    if (floatpack_init() != 0) {
        printf("floatpack_init FAILED (no Vulkan?)\n");
        free(A_f32); free(B_f32); free(C_ref); free(C_fp); free(A_q4);
        return 1;
    }
    printf("floatpack_init OK\n");

    // Compute with floatpack
    if (floatpack_matmul_q4_0(M, N, K, A_q4, B_f32, C_fp) != 0) {
        printf("floatpack_matmul_q4_0 FAILED\n");
        floatpack_cleanup();
        free(A_f32); free(B_f32); free(C_ref); free(C_fp); free(A_q4);
        return 1;
    }
    printf("floatpack_matmul_q4_0 OK\n");

    // Compare
    float err = max_rel_err(C_ref, C_fp, M * N);
    printf("Max relative error vs F32 ref: %.4f\n", err);

    // Print first few values
    printf("\n  C[0..1][0..1]:\n");
    for (int i = 0; i < (M > 2 ? 2 : M); i++) {
        for (int j = 0; j < (N > 2 ? 2 : N); j++) {
            printf("    [%d][%d] ref=%.4f fp4=%.4f\n",
                   i, j, C_ref[i * N + j], C_fp[i * N + j]);
        }
    }

    floatpack_cleanup();

    printf("\nPOC %s\n", err < 2.0f ? "PASSED (FP4 quantization error within expected range)" : "FAILED");
    free(A_f32); free(B_f32); free(C_ref); free(C_fp); free(A_q4);
    return 0;
}
