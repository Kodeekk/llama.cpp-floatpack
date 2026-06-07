// Standalone validation test: load real Q4_K model data and validate floatpack
// Build: gcc -o test_val test_validate.c floatpack.c -I.. -lm -lvulkan -lpthread -g -O0
#include "fp4_format.h"
#include "floatpack.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Copy of the converters since they're static in floatpack.c
static void my_convert_f32_to_fp4_colmaj(const float *B_f32, int K, int N,
                                        uint16_t *B_fp4) {
    int K_div4 = K / 4;
    for (int col = 0; col < N; col++) {
        for (int k = 0; k < K; k += 4) {
            uint8_t nibbles[4];
            for (int j = 0; j < 4; j++) {
                nibbles[j] = fp4_encode(B_f32[(k + j) * N + col]);
            }
            B_fp4[col * K_div4 + (k / 4)] = fp4_pack_4(nibbles);
        }
    }
}

static void my_convert_f32_to_fp4_rowmaj(const float *A_f32, int M, int K,
                                          uint16_t *A_fp4) {
    int K_div4 = K / 4;
    for (int row = 0; row < M; row++) {
        for (int k = 0; k < K; k += 4) {
            uint8_t nibbles[4];
            for (int j = 0; j < 4; j++) {
                nibbles[j] = fp4_encode(A_f32[row * K + k + j]);
            }
            A_fp4[row * K_div4 + (k / 4)] = fp4_pack_4(nibbles);
        }
    }
}

// F32 matmul reference
static void matmul_f32(int M, int N, int K,
                       const float *A, const float *B, float *C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

static float rand_float(void) {
    return ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

int main(int argc, char **argv) {
    // Test 1: Small synthetic test
    printf("=== Test 1: Small synthetic matmul ===\n");
    {
        int M=4, K=16, N=3;
        float A[4*16], B[16*3], C_ref[4*3], C_fp4_cpu[4*3], C_fp4_gpu[4*3];
        for (int i=0; i<M*K; i++) A[i] = rand_float();
        for (int i=0; i<K*N; i++) B[i] = rand_float();

        matmul_f32(M, N, K, A, B, C_ref);

        // CPU FP4 path
        int K_div4 = K/4;
        uint16_t *A_p = (uint16_t*)malloc(M*K_div4*2);
        uint16_t *B_p = (uint16_t*)malloc(N*K_div4*2);
        my_convert_f32_to_fp4_rowmaj(A, M, K, A_p);
        my_convert_f32_to_fp4_colmaj(B, K, N, B_p);
        floatpack_matmul_fp4_cpu(M, N, K_div4, A_p, B_p, C_fp4_cpu);

        float err = 0;
        for (int i=0; i<M*N; i++) {
            float d = C_fp4_cpu[i] - C_ref[i];
            err += d*d;
        }
        err = sqrtf(err / (M*N));
        printf("  CPU FP4 vs F32 ref: RMSE=%f\n", err);

        // GPU FP4 path
        if (floatpack_ready()) {
            floatpack_matmul_fp4(M, N, K_div4, A_p, B_p, C_fp4_gpu);
            float gpu_err = 0;
            for (int i=0; i<M*N; i++) {
                float d = C_fp4_gpu[i] - C_fp4_cpu[i];
                gpu_err += d*d;
            }
            gpu_err = sqrtf(gpu_err / (M*N));
            printf("  GPU FP4 vs CPU FP4: RMSE=%f\n", gpu_err);
        } else {
            printf("  floatpack not ready, skipping GPU test\n");
        }
        free(A_p); free(B_p);
    }

    printf("\n=== Test 2: Model-sized synthetic matmul (M=896, K=4864, N=30) ===\n");
    {
        int M=896, K=4864, N=30;
        size_t szA = (size_t)M * K * sizeof(float);
        size_t szB = (size_t)K * N * sizeof(float);
        size_t szC = (size_t)M * N * sizeof(float);
        int K_div4 = K/4;

        float *A = (float*)malloc(szA);
        float *B = (float*)malloc(szB);
        float *C_ref = (float*)malloc(szC);
        float *C_fp4_cpu = (float*)malloc(szC);
        float *C_fp4_gpu = (float*)malloc(szC);

        // Fill with realistic-looking values (small, centered around 0)
        for (int i=0; i<M*K; i++) A[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
        for (int i=0; i<K*N; i++) B[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;

        // Verify FP4 range
        float max_a = 0, max_b = 0;
        for (int i=0; i<M*K; i++) { float v = fabsf(A[i]); if (v>max_a) max_a=v; }
        for (int i=0; i<K*N; i++) { float v = fabsf(B[i]); if (v>max_b) max_b=v; }
        printf("  A range: [-%f, %f]\n", max_a, max_a);
        printf("  B range: [-%f, %f]\n", max_b, max_b);

        matmul_f32(M, N, K, A, B, C_ref);
        printf("  F32 ref done\n");

        uint16_t *A_p = (uint16_t*)malloc(M*K_div4*2);
        uint16_t *B_p = (uint16_t*)malloc(N*K_div4*2);
        my_convert_f32_to_fp4_rowmaj(A, M, K, A_p);
        my_convert_f32_to_fp4_colmaj(B, K, N, B_p);
        printf("  FP4 conversion done\n");

        floatpack_matmul_fp4_cpu(M, N, K_div4, A_p, B_p, C_fp4_cpu);
        printf("  CPU FP4 done\n");

        float err_cpu = 0;
        for (int i=0; i<M*N; i++) {
            float d = C_fp4_cpu[i] - C_ref[i];
            err_cpu += d*d;
        }
        err_cpu = sqrtf(err_cpu / (M*N));
        printf("  CPU FP4 vs F32 ref: RMSE=%f\n", err_cpu);

        // Compare a few values
        printf("  Top 5 values: F32_ref vs FP4_cpu:\n");
        for (int i=0; i<5 && i<M*N; i++) {
            printf("    [%d]: %f vs %f\n", i, C_ref[i], C_fp4_cpu[i]);
        }

        if (floatpack_ready()) {
            floatpack_matmul_fp4(M, N, K_div4, A_p, B_p, C_fp4_gpu);
            float gpu_err = 0;
            for (int i=0; i<M*N; i++) {
                float d = C_fp4_gpu[i] - C_fp4_cpu[i];
                gpu_err += d*d;
            }
            gpu_err = sqrtf(gpu_err / (M*N));
            printf("  GPU FP4 vs CPU FP4: RMSE=%f\n", gpu_err);
            printf("  Top 5: F32_ref vs FP4_cpu vs FP4_gpu:\n");
            for (int i=0; i<5 && i<M*N; i++) {
                printf("    [%d]: %f vs %f vs %f\n", i, C_ref[i], C_fp4_cpu[i], C_fp4_gpu[i]);
            }
        }

        free(A_p); free(B_p);
        free(A); free(B); free(C_ref); free(C_fp4_cpu); free(C_fp4_gpu);
    }

    // Test 3: Extreme value test (what if activations exceed ±6?)
    printf("\n=== Test 3: Extreme values (clipping test) ===\n");
    {
        int M=2, K=8, N=1;
        float A[] = {0.5, 1.0, 2.0, 10.0, -0.5, -1.0, -2.0, -10.0,
                     3.0, 4.0, 6.0, 7.0, -3.0, -4.0, -6.0, -7.0};
        float B[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        float C_ref[2], C_fp4[2];

        matmul_f32(M, N, K, A, B, C_ref);
        int K_div4 = K/4;
        uint16_t A_p[2*2], B_p[1*2];
        my_convert_f32_to_fp4_rowmaj(A, M, K, A_p);
        my_convert_f32_to_fp4_colmaj(B, K, N, B_p);
        floatpack_matmul_fp4_cpu(M, N, K_div4, A_p, B_p, C_fp4);
        printf("  F32 ref: %f %f\n", C_ref[0], C_ref[1]);
        printf("  FP4:     %f %f\n", C_fp4[0], C_fp4[1]);
        // Values >6 or <-6 get clipped to ±6
    }

    printf("\n=== All tests done ===\n");
    return 0;
}
