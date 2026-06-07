// Debug FP4 encoding/decoding and matmul
#include "fp4_format.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void fp4_debug_build_lut(float *lut) {
    for (int a = 0; a < 16; a++) {
        float fa = fp4_decode((uint8_t)a);
        for (int b = 0; b < 16; b++) {
            float fb = fp4_decode((uint8_t)b);
            lut[(a << 4) | b] = fa * fb;
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

int main() {
    printf("=== FP4 Encoding Table ===\n");
    printf("idx -> decode -> encode -> roundtrip\n");
    for (int i = 0; i < 16; i++) {
        float d = fp4_decode((uint8_t)i);
        uint8_t e = fp4_encode(d);
        float r = fp4_decode(e);
        printf("  0x%X -> %7.4f -> 0x%X -> %7.4f%s\n",
               i, d, e, r, (i==e)?"":" REROUTED");
        if (i != e) printf("         *** encode-decode mismatch! ***\n");
    }

    // Check: does 0.5 encode correctly?
    printf("\n=== Key encoding checks ===\n");
    float test_vals[] = {0.0f, 0.5f, 0.75f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, 10.0f, -0.5f, -6.0f, -10.0f};
    for (int i = 0; i < sizeof(test_vals)/sizeof(float); i++) {
        float v = test_vals[i];
        uint8_t e = fp4_encode(v);
        float d = fp4_decode(e);
        printf("  %7.3f -> 0x%X -> %7.3f (err=%7.4f)\n", v, e, d, d-v);
    }

    printf("\n=== LUT verification ===\n");
    float lut[256];
    fp4_debug_build_lut(lut);
    // Check specific entries
    for (int a = 0; a < 16; a++) {
        for (int b = 0; b < 16; b++) {
            float expected = fp4_decode(a) * fp4_decode(b);
            float actual = lut[(a << 4) | b];
            if (fabsf(expected - actual) > 1e-6f) {
                printf("  LUT[0x%02X] MISMATCH: expected %f, got %f\n",
                       (a<<4)|b, expected, actual);
            }
        }
    }
    printf("  LUT verification complete\n");

    printf("\n=== Test data: A[0..7] = [0.5, 1.0, 2.0, 10.0, -0.5, -1.0, -2.0, -10.0] ===\n");
    printf("  B = all 1.0 (K=8, N=1)\n");
    float A_data[] = {0.5f, 1.0f, 2.0f, 10.0f, -0.5f, -1.0f, -2.0f, -10.0f};
    float B_data[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    int M=1, K=8, N=1;
    
    // F32 ref
    float C_ref[1];
    matmul_f32(M, N, K, A_data, B_data, C_ref);
    printf("  F32 result: %f (should be 0.0)\n", C_ref[0]);

    // Manual FP4 matmul trace
    int K_div4 = K/4;
    uint16_t A_packed[2], B_packed[2];
    
    // Encode A: row 0, group 0: A[0..3] = [0.5, 1.0, 2.0, 10.0]
    uint8_t a_nib0 = fp4_encode(0.5f);
    uint8_t a_nib1 = fp4_encode(1.0f);
    uint8_t a_nib2 = fp4_encode(2.0f);
    uint8_t a_nib3 = fp4_encode(10.0f);
    A_packed[0] = fp4_pack_4((uint8_t[]){a_nib0, a_nib1, a_nib2, a_nib3});
    printf("  A group 0 nibbles: 0x%X 0x%X 0x%X 0x%X (packed=0x%04X)\n",
           a_nib0, a_nib1, a_nib2, a_nib3, A_packed[0]);
    printf("    decoded: %f %f %f %f\n",
           fp4_decode(a_nib0), fp4_decode(a_nib1), fp4_decode(a_nib2), fp4_decode(a_nib3));
    
    // Encode A: group 1: A[4..7] = [-0.5, -1.0, -2.0, -10.0]
    uint8_t a_nib4 = fp4_encode(-0.5f);
    uint8_t a_nib5 = fp4_encode(-1.0f);
    uint8_t a_nib6 = fp4_encode(-2.0f);
    uint8_t a_nib7 = fp4_encode(-10.0f);
    A_packed[1] = fp4_pack_4((uint8_t[]){a_nib4, a_nib5, a_nib6, a_nib7});
    printf("  A group 1 nibbles: 0x%X 0x%X 0x%X 0x%X (packed=0x%04X)\n",
           a_nib4, a_nib5, a_nib6, a_nib7, A_packed[1]);
    printf("    decoded: %f %f %f %f\n",
           fp4_decode(a_nib4), fp4_decode(a_nib5), fp4_decode(a_nib6), fp4_decode(a_nib7));
    
    // Encode B: all 1.0, N=1, K_div4=2
    // Column 0: B[0..3] and B[4..7]
    uint8_t b_nib0 = fp4_encode(1.0f);
    uint8_t b_nib1 = fp4_encode(1.0f);
    uint8_t b_nib2 = fp4_encode(1.0f);
    uint8_t b_nib3 = fp4_encode(1.0f);
    B_packed[0] = fp4_pack_4((uint8_t[]){b_nib0, b_nib1, b_nib2, b_nib3});
    B_packed[1] = fp4_pack_4((uint8_t[]){b_nib0, b_nib1, b_nib2, b_nib3});
    printf("  B packed: 0x%04X 0x%04X\n", B_packed[0], B_packed[1]);
    
    // Manual FP4 matmul
    float acc = 0.0f;
    for (int k = 0; k < K_div4; k++) {
        uint16_t a_word = A_packed[k];
        uint16_t b_word = B_packed[0 * K_div4 + k];  // col 0
        
        uint8_t a[4], b[4];
        fp4_unpack_4(a_word, a);
        fp4_unpack_4(b_word, b);
        
        printf("  k=%d: a_word=0x%04X b_word=0x%04X\n", k, a_word, b_word);
        printf("    a_nibbles=[0x%X,0x%X,0x%X,0x%X]\n", a[0],a[1],a[2],a[3]);
        printf("    b_nibbles=[0x%X,0x%X,0x%X,0x%X]\n", b[0],b[1],b[2],b[3]);
        
        for (int l = 0; l < 4; l++) {
            float fa = fp4_decode(a[l]);
            float fb = fp4_decode(b[l]);
            float lut_val = lut[(a[l] << 4) | b[l]];
            printf("      l=%d: a=0x%X(%+5.2f) b=0x%X(%+5.2f) LUT=%.6f fa*fb=%.6f\n",
                   l, a[l], fa, b[l], fb, lut_val, fa*fb);
            acc += lut_val;
        }
        printf("    partial acc after k=%d: %f\n", k, acc);
    }
    printf("  Final FP4 result: %f\n", acc);
    printf("  F32 reference: %f\n", C_ref[0]);
    printf("  Error: %f\n", acc - C_ref[0]);

    return 0;
}
