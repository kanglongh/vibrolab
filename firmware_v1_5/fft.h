/* vibrolab v1.5 · fft.h — float32 radix-2 FFT (用 ESP32-S3 硬件单精 FPU, ~5ms/窗). */
#ifndef VBL2_FFT_H
#define VBL2_FFT_H
void vbl_fft(float *re, float *im, int n);
void vbl_ifft(float *re, float *im, int n);
#endif
