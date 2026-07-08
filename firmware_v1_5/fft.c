/* vibrolab v1.5 · fft.c — float32 radix-2 Cooley-Tukey. */
#include "fft.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void bit_reverse(float *re, float *im, int n) {
    int i, j = 0;
    for (i = 1; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
}

void vbl_fft(float *re, float *im, int n) {
    bit_reverse(re, im, n);
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * (float)M_PI / len;
        float wr = cosf(ang), wi = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (int k = 0; k < len / 2; ++k) {
                float tr = cr * re[i+k+len/2] - ci * im[i+k+len/2];
                float ti = cr * im[i+k+len/2] + ci * re[i+k+len/2];
                re[i+k+len/2] = re[i+k] - tr;
                im[i+k+len/2] = im[i+k] - ti;
                re[i+k] += tr; im[i+k] += ti;
                float ncr = cr*wr - ci*wi;
                ci = cr*wi + ci*wr;
                cr = ncr;
            }
        }
    }
}

void vbl_ifft(float *re, float *im, int n) {
    for (int i = 0; i < n; ++i) im[i] = -im[i];
    vbl_fft(re, im, n);
    for (int i = 0; i < n; ++i) { re[i] /= n; im[i] = -im[i] / n; }
}
