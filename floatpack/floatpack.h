#ifndef FLOATPACK_H
#define FLOATPACK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the floatpack Vulkan backend.
// Returns 0 on success, nonzero on failure.
int floatpack_init(void);

// Cleanup the floatpack Vulkan backend.
void floatpack_cleanup(void);

// Compute C = A * B where A and B are packed FP4 matrices.
//
// A is (M x K) stored as row-major uint16_t with K/4 elements per row
//   (each uint16_t packs four FP4 values).
// B is (N x K) stored as column-major uint16_t with K/4 elements per column.
// C is (M x N) float output.
//
// K must be a multiple of 4.
// K_div4 = K / 4 is the number of packed words per row/column.
int floatpack_matmul_fp4(int M, int N, int K_div4,
                         const uint16_t *A, const uint16_t *B, float *C);

// Returns 1 if the Vulkan backend was initialized successfully, 0 otherwise.
int floatpack_ready(void);

// CPU fallback: compute FP4 matmul on CPU for verification.
void floatpack_matmul_fp4_cpu(int M, int N, int K_div4,
                              const uint16_t *A, const uint16_t *B, float *C);

// Compute C = A * B where A is F32 (pre-dequantized from any quantized type).
// A: (M x K) as F32 row-major
// B: (K x N) as F32 row-major
// C: (M x N) F32 output
// Internally converts F32 -> FP4-packed, runs GPU matmul.
int floatpack_matmul_f32(int M, int N, int K,
                         const float *A_f32, const float *B_f32, float *C_f32);

// Compute C = A * B where A is Q4_0-quantized weights.
// A: (M x K) as block_q4_0 row-major (K must be multiple of 32)
// B: (K x N) as F32 row-major
// Convenience wrapper converting Q4_0 -> FP4 internally.
int floatpack_matmul_q4_0(int M, int N, int K,
                          const void *A_q4, const float *B_f32, float *C_f32);

// Fused Q4_0 matmul: reads Q4_0 blocks directly on GPU, dequantizes in shader.
// No intermediate FP4 conversion. B must be F32 row-major (K x N).
// K must be multiple of 32.
int floatpack_matmul_q4_0_fused(int M, int N, int K,
                                const void *A_q4, const float *B_f32, float *C_f32);

#ifdef __cplusplus
}
#endif

#endif // FLOATPACK_H
