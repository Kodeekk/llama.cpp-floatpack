// Test fused Q4_0 matmul shader
// Compile: gcc -o test_q4_0_fused test_q4_0_fused.c floatpack.c -lm -lvulkan -I.
// First compile shader: glslc floatpack_q4_0.comp -o floatpack_q4_0.spv

#include "floatpack.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define QK4_0 32
typedef struct {
    uint16_t d;           // fp16 scale
    uint8_t  qs[QK4_0/2];// nibbles
} block_q4_0;

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t f = ((h & 0x8000) << 16) | (((h & 0x7C00) + 0x1C000) << 13) | ((h & 0x03FF) << 13);
    float r; memcpy(&r, &f, sizeof(r)); return r;
}

static inline uint16_t fp32_to_fp16(float x) {
    uint32_t f; memcpy(&f, &x, sizeof(f));
    return (uint16_t)(((f >> 16) & 0x8000) | ((((f & 0x7F800000) - 0x38000000) >> 13) & 0x7C00) | ((f >> 13) & 0x03FF));
}

// Compute C = A * B where A is Q4_0 blocks, done entirely on CPU
static void matmul_q4_0_cpu(int M, int N, int K,
                            const block_q4_0 *A_q4, const float *B, float *C) {
    int n_blocks = K / QK4_0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int kb = 0; kb < n_blocks; kb++) {
                const block_q4_0 *blk = &A_q4[i * n_blocks + kb];
                float d = fp16_to_fp32(blk->d);
                for (int w = 0; w < QK4_0; w++) {
                    int nib = (blk->qs[w >> 1] >> ((w & 1) ? 4 : 0)) & 0xF;
                    float a_val = d * ((float)(nib - 8));
                    acc += a_val * B[(kb * QK4_0 + w) * N + j];
                }
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
    printf("=== Q4_0 Fused Matmul Test ===\n\n");

    const int M = 8, K = 64, N = 8;

    float *A_f32 = (float *)malloc((size_t)M * K * sizeof(float));
    float *B_f32 = (float *)malloc((size_t)K * N * sizeof(float));
    float *C_cpu = (float *)malloc((size_t)M * N * sizeof(float));
    float *C_fused = (float *)malloc((size_t)M * N * sizeof(float));

    srand(12345);
    for (int i = 0; i < M * K; i++) A_f32[i] = ((float)(rand() % 200) - 100.0f) / 50.0f;
    for (int i = 0; i < K * N; i++) B_f32[i] = ((float)(rand() % 200) - 100.0f) / 50.0f;

    // Quantize A to Q4_0
    int n_blocks = K / QK4_0;
    block_q4_0 *A_q4 = (block_q4_0 *)malloc((size_t)M * n_blocks * sizeof(block_q4_0));

    for (int row = 0; row < M; row++) {
        for (int b = 0; b < n_blocks; b++) {
            block_q4_0 *blk = &A_q4[row * n_blocks + b];
            float max_val = 0.0f;
            for (int i = 0; i < QK4_0; i++) {
                float v = fabsf(A_f32[row * K + b * QK4_0 + i]);
                if (v > max_val) max_val = v;
            }
            float d = max_val / 7.0f;
            if (d < 1e-10f) d = 1.0f;
            blk->d = fp32_to_fp16(d);
            memset(blk->qs, 0, QK4_0/2);
            for (int i = 0; i < QK4_0; i++) {
                float val = A_f32[row * K + b * QK4_0 + i];
                int q = (int)roundf(val / d) + 8;
                if (q < 0) q = 0;
                if (q > 15) q = 15;
                if (i & 1)
                    blk->qs[i / 2] |= (q << 4);
                else
                    blk->qs[i / 2] |= q;
            }
        }
    }
    printf("Q4_0 quantization done.\n");

    // CPU reference (identical Q4_0 data, dequantized and matmul on CPU)
    matmul_q4_0_cpu(M, N, K, A_q4, B_f32, C_cpu);
    printf("CPU Q4_0 matmul done.\n");

    // Init floatpack and run fused shader
    if (floatpack_init() != 0) {
        printf("floatpack_init FAILED (no Vulkan?)\n");
        free(A_f32); free(B_f32); free(C_cpu); free(C_fused); free(A_q4);
        return 1;
    }
    printf("floatpack_init OK\n");

    if (floatpack_matmul_q4_0_fused(M, N, K, A_q4, B_f32, C_fused) != 0) {
        printf("floatpack_matmul_q4_0_fused FAILED\n");
        floatpack_cleanup();
        free(A_f32); free(B_f32); free(C_cpu); free(C_fused); free(A_q4);
        return 1;
    }
    printf("floatpack_matmul_q4_0_fused OK\n");

    // Compare fused vs CPU (same Q4_0 data -> should match closely)
    float err = max_rel_err(C_cpu, C_fused, M * N);
    printf("Max relative error (fused vs CPU Q4_0): %.6f\n", err);

    printf("\n  First 2x2 elements:\n");
    for (int i = 0; i < (M > 2 ? 2 : M); i++) {
        for (int j = 0; j < (N > 2 ? 2 : N); j++) {
            printf("    [%d][%d] cpu=%.6f fused=%.6f\n",
                   i, j, C_cpu[i*N + j], C_fused[i*N + j]);
        }
    }

    // Also run old Q4_0->FP4 path for comparison
    float *C_old = (float *)malloc((size_t)M * N * sizeof(float));
    if (floatpack_matmul_q4_0(M, N, K, A_q4, B_f32, C_old) == 0) {
        float err_old = max_rel_err(C_cpu, C_old, M * N);
        printf("\n  Old Q4_0->FP4 vs CPU Q4_0:    %.6f\n", err_old);
        float err_vs_old = max_rel_err(C_fused, C_old, M * N);
        printf("  Fused vs old path:             %.6f\n", err_vs_old);
    }
    free(C_old);

    floatpack_cleanup();

    int pass = err < 1e-4f; // shader should match CPU F32 within float precision
    printf("\n%s\n", pass ? "PASSED (shader matches CPU Q4_0)" : "FAILED (shader deviates from CPU)");
    free(A_f32); free(B_f32); free(C_cpu); free(C_fused); free(A_q4);
    return pass ? 0 : 1;
}
