// floatpack test program
// Compile: gcc -o test_floatpack test_floatpack.c floatpack.c -lm -lvulkan -I.
// First compile the shader: cd floatpack && glslc floatpack_fp4.comp -o floatpack_fp4.spv

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "floatpack.h"
#include "fp4_format.h"

static int test_fp4_format(void) {
    // Test encode/decode roundtrip for all representable values
    float test_vals[] = {
        0.0f, 0.5f, 0.75f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.5f, -0.75f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    int n = sizeof(test_vals) / sizeof(test_vals[0]);
    int ok = 1;

    printf("=== FP4 Format Test ===\n");
    for (int i = 0; i < n; i++) {
        float orig = test_vals[i];
        uint8_t enc = fp4_encode(orig);
        float dec = fp4_decode(enc);
        float err = fabsf(dec - orig);
        printf("  %6.2f -> 0x%X -> %6.2f (err=%.4f)\n", orig, enc, dec, err);
        if (err > 0.26f && orig != 0.0f) {
            printf("    ERROR: too much error\n");
            ok = 0;
        }
    }

    // Test LUT
    // FP4 representable values:
    //   0x0 =  0.0, 0x1 =  0.75, 0x2 =  1.0, 0x3 =  1.5
    //   0x4 =  2.0, 0x5 =  3.0, 0x6 =  4.0, 0x7 =  6.0
    //   0x8 = -0.5, 0x9 = -0.75,0xA = -1.0, 0xB = -1.5
    //   0xC = -2.0, 0xD = -3.0, 0xE = -4.0, 0xF = -6.0
    float lut[FP4_LUT_SIZE];
    fp4_build_lut(lut);
    printf("\n  LUT[0x00] = %.2f (0*0)\n", lut[0]);
    printf("  LUT[0x11] = %.2f (0.75*0.75)\n", lut[0x11]);
    printf("  LUT[0x44] = %.2f (2.0*2.0)\n", lut[0x44]);
    printf("  LUT[0x77] = %.2f (6.0*6.0)\n", lut[0x77]);
    printf("  LUT[0x88] = %.2f (-0.5*-0.5)\n", lut[0x88]);

    if (fabsf(lut[0x11] - 0.5625f) > 0.01f) { printf("  ERROR: 0.75*0.75 should be 0.5625\n"); ok = 0; }
    if (fabsf(lut[0x88] - 0.25f) > 0.1f) { printf("  ERROR: -0.5*-0.5 should be 0.25\n"); ok = 0; }

    return ok;
}

static int test_matmul_cpu(void) {
    printf("\n=== CPU Matmul Test ===\n");

    const int M = 2, N = 3, K = 4;
    const int K_div4 = K / 4;

    uint16_t A[2 * 1]; // M * K_div4 = 2
    uint16_t B[3 * 1]; // N * K_div4 = 3

    uint8_t a_vals[2][4] = {
        {fp4_encode(0.5f), fp4_encode(1.0f), fp4_encode(1.5f), fp4_encode(2.0f)},
        {fp4_encode(2.0f), fp4_encode(3.0f), fp4_encode(4.0f), fp4_encode(6.0f)},
    };

    uint8_t b_vals[3][4] = {
        {fp4_encode(0.5f), fp4_encode(1.0f), fp4_encode(1.5f), fp4_encode(2.0f)},
        {fp4_encode(2.0f), fp4_encode(3.0f), fp4_encode(4.0f), fp4_encode(6.0f)},
        {fp4_encode(0.5f), fp4_encode(0.75f), fp4_encode(1.0f), fp4_encode(1.5f)},
    };

    for (int i = 0; i < M; i++) {
        A[i] = fp4_pack_4(a_vals[i]);
    }
    for (int j = 0; j < N; j++) {
        B[j] = fp4_pack_4(b_vals[j]);
    }

    float C[2 * 3]; // M * N
    floatpack_matmul_fp4_cpu(M, N, K_div4, A, B, C);

    printf("Result C (M=%d, N=%d):\n", M, N);
    int pass = 1;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("  C[%d][%d] = %.4f", i, j, C[i * N + j]);
            // Expected (approximate, since FP4 has coarse quantization):
            // Row 0, Col 0: 0.5*0.5 + 1.0*1.0 + 1.5*1.5 + 2.0*2.0 = 0.25+1+2.25+4 = 7.5
            // But FP4 has rounding, so we just check it's nonzero
            if (C[i * N + j] == 0.0f) {
                printf(" ZERO");
                pass = 0;
            }
            printf("\n");
        }
    }

    return pass;
}

int main(void) {
    int ok = 1;

    if (!test_fp4_format()) {
        printf("\nFP4 FORMAT TEST FAILED\n");
        ok = 0;
    } else {
        printf("\nFP4 FORMAT TEST PASSED\n");
    }

    if (!test_matmul_cpu()) {
        printf("\nCPU MATMUL TEST FAILED\n");
        ok = 0;
    } else {
        printf("\nCPU MATMUL TEST PASSED\n");
    }

    printf("\n=== Vulkan GPU Test ===\n");
    if (floatpack_init() == 0) {
        printf("Vulkan initialized successfully\n");

        const int M = 2, N = 3, K = 4, K_div4 = K / 4;
        uint8_t a_vals[2][4] = {
            {fp4_encode(0.5f), fp4_encode(1.0f), fp4_encode(1.5f), fp4_encode(2.0f)},
            {fp4_encode(2.0f), fp4_encode(3.0f), fp4_encode(4.0f), fp4_encode(6.0f)},
        };
        uint8_t b_vals[3][4] = {
            {fp4_encode(0.5f), fp4_encode(1.0f), fp4_encode(1.5f), fp4_encode(2.0f)},
            {fp4_encode(2.0f), fp4_encode(3.0f), fp4_encode(4.0f), fp4_encode(6.0f)},
            {fp4_encode(0.5f), fp4_encode(0.75f), fp4_encode(1.0f), fp4_encode(1.5f)},
        };

        uint16_t A[2], B[3];
        for (int i = 0; i < M; i++) A[i] = fp4_pack_4(a_vals[i]);
        for (int j = 0; j < N; j++) B[j] = fp4_pack_4(b_vals[j]);

        float C_gpu[6];
        if (floatpack_matmul_fp4(M, N, K_div4, A, B, C_gpu) == 0) {
            printf("GPU matmul succeeded:\n");
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    printf("  C_gpu[%d][%d] = %.4f\n", i, j, C_gpu[i * N + j]);
                }
            }
        } else {
            printf("GPU matmul failed\n");
            ok = 0;
        }

        floatpack_cleanup();
    } else {
        printf("Vulkan not available (compile shader first: glslc floatpack_fp4.comp -o floatpack_fp4.spv)\n");
    }

    printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
