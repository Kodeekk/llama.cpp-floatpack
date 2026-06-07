#ifndef FP4_FORMAT_H
#define FP4_FORMAT_H

#include <stdint.h>
#include <math.h>
#include <string.h>

#define FP4_LUT_SIZE 256

// FP4 bit layout: s e1 e0 m  (4 bits)
//   s = sign (1 bit)
//   e = 2-bit exponent, bias = 1 -> real exponent = e - 1
//   m = 1-bit mantissa, significand = 1 + 0.5*m
//
// Special case: 0b0000 = exact zero.
//
// Positive values:
//   0000 ->  0.0
//   0001 ->  0.5
//   0010 ->  0.75
//   0011 ->  1.0
//   0100 ->  1.5
//   0101 ->  2.0
//   0110 ->  3.0
//   0111 ->  4.0
//   1000 ->  6.0
// (sign bit adds negative equivalents)

static inline float fp4_decode(uint8_t bits) {
    if (bits == 0) return 0.0f;
    int s = bits >> 3;
    int e = (bits >> 1) & 0x3;
    int m = bits & 1;
    float val = (1.0f + 0.5f * (float)m) * ldexpf(1.0f, e - 1);
    return s ? -val : val;
}

static inline uint8_t fp4_encode(float x) {
    if (x == 0.0f) return 0;
    int s = (x < 0);
    float mag = fabsf(x);
    if (mag >= 6.0f) mag = 6.0f;
    int e = 0;
    while (mag >= 2.0f) { mag *= 0.5f; e++; }
    while (mag < 1.0f)  { mag *= 2.0f; e--; }
    e += 1;
    if (e < 0) e = 0;
    if (e > 3) e = 3;
    int m = (mag - 1.0f) >= 0.25f ? 1 : 0;
    uint8_t bits = (uint8_t)((s << 3) | (e << 1) | m);
    // Fix: 0x0 is the zero pattern; if the value wasn't zero, round up to the smallest
    // representable non-zero (0x1 = 0.75) to avoid 0.5 rounding to 0.0
    if (bits == 0) bits = 1;
    return bits;
}

static inline void fp4_build_lut(float *lut) {
    for (int a = 0; a < 16; a++) {
        float fa = fp4_decode((uint8_t)a);
        for (int b = 0; b < 16; b++) {
            float fb = fp4_decode((uint8_t)b);
            lut[(a << 4) | b] = fa * fb;
        }
    }
}

// Packed structure: four 4-bit floats stored in a uint16_t
// word = [ f3(15:12) | f2(11:8) | f1(7:4) | f0(3:0) ]

static inline uint16_t fp4_pack_4(const uint8_t f[4]) {
    return (uint16_t)(f[0] | (f[1] << 4) | (f[2] << 8) | (f[3] << 12));
}

static inline void fp4_unpack_4(uint16_t word, uint8_t f[4]) {
    f[0] = (uint8_t)(word & 0xF);
    f[1] = (uint8_t)((word >> 4) & 0xF);
    f[2] = (uint8_t)((word >> 8) & 0xF);
    f[3] = (uint8_t)((word >> 12) & 0xF);
}

#endif // FP4_FORMAT_H
