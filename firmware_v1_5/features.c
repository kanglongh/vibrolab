/* vibrolab v1.5 · features.c — 120 维 CFD, float32 (ESP32-S3 硬件 FPU).
 * 镜像 Python vibrolab/features.py. 5 模块共用一次正向 FFT, 倒谱额外一次 IFFT.
 * 大缓冲全放 static (.bss) 不占 8KB 主栈. */
#include "features.h"
#include "fft.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define N VBL2_WINDOW
#define NBINS (N/2 + 1)

static float s_re[N], s_im[N], s_mag[NBINS], s_freqs[NBINS];
static float s_cre[N], s_cim[N];           /* 倒谱 IFFT */

static int cmp_float(const void *a, const void *b) {
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static void extract_time(const float *w, float *out) {
    float mean = 0.0f;
    for (int i = 0; i < N; ++i) mean += w[i];
    mean /= N;
    float sumsq = 0.0f, sum_abs = 0.0f, sum_sqrt_abs = 0.0f, peak = 0.0f, mn = w[0];
    float m3 = 0.0f, m4 = 0.0f;
    for (int i = 0; i < N; ++i) {
        float v = w[i], av = fabsf(v);
        sumsq += v*v; sum_abs += av; sum_sqrt_abs += sqrtf(av);
        if (av > peak) peak = av;
        if (v < mn) mn = v;
        float d = v - mean; m3 += d*d*d; m4 += d*d*d*d;
    }
    float rms = sqrtf(sumsq/N) + 1e-12f;
    float std = sqrtf(sumsq/N - mean*mean) + 1e-12f;
    float mean_abs = sum_abs/N;
    float sra = sum_sqrt_abs/N; sra *= sra;
    out[0]=mean; out[1]=std; out[2]=rms; out[3]=peak; out[4]=peak-mn;
    out[5]=mean_abs; out[6]=sra; out[7]=(m3/N)/(std*std*std); out[8]=(m4/N)/(std*std*std*std);
    out[9]=peak/rms; out[10]=peak/(mean_abs+1e-12f); out[11]=rms/(mean_abs+1e-12f);
}

static void extract_freq(const float *mag, const float *freqs, float *out) {
    float total = 0.0f;
    for (int k = 0; k < NBINS; ++k) total += mag[k];
    total += 1e-12f;
    float centroid = 0.0f, sum_log = 0.0f, mean_mag = 0.0f, max_mag = mag[0], min_mag = mag[0];
    for (int k = 0; k < NBINS; ++k) {
        float p = mag[k]/total;
        centroid += p*freqs[k]; sum_log += logf(mag[k]+1e-12f); mean_mag += mag[k];
        if (mag[k] > max_mag) max_mag = mag[k];
        if (mag[k] < min_mag) min_mag = mag[k];
    }
    mean_mag /= NBINS;
    float bw = 0.0f, entropy = 0.0f;
    for (int k = 0; k < NBINS; ++k) {
        float p = mag[k]/total;
        bw += p*(freqs[k]-centroid)*(freqs[k]-centroid);
        entropy -= p*logf(p+1e-12f);
    }
    bw = sqrtf(bw < 0 ? 0 : bw);
    float cs = 0.0f, thr85 = 0.85f*total, thr95 = 0.95f*total, roll85 = freqs[NBINS-1], roll95 = freqs[NBINS-1];
    int got85 = 0;
    for (int k = 0; k < NBINS; ++k) {
        cs += mag[k];
        if (!got85 && cs >= thr85) { roll85 = freqs[k]; got85 = 1; }
        if (cs >= thr95) { roll95 = freqs[k]; break; }
    }
    float Sx=0, Sy=0, Sxy=0, Sxx=0;
    for (int k = 0; k < NBINS; ++k) {
        float x = logf(freqs[k]+1.0f), y = logf(mag[k]+1e-12f);
        Sx+=x; Sy+=y; Sxy+=x*y; Sxx+=x*x;
    }
    float slope = (NBINS*Sxy - Sx*Sy)/(NBINS*Sxx - Sx*Sx);
    float flatness = expf(sum_log/NBINS)/(mean_mag+1e-12f);
    float var_mag = 0.0f;
    for (int k = 0; k < NBINS; ++k) { float d = mag[k]-mean_mag; var_mag += d*d; }
    var_mag /= NBINS;
    /* p25/p50/p75/p90 (dim 23-26): 拷贝 mag[] 到静态缓冲后 qsort, 取分位值.
     * 静态缓冲 4KB 放 .bss, 不占 8KB 主栈. qsort O(N log N) ~10K 比较, <1ms. */
    static float mag_sorted[NBINS];
    for (int k = 0; k < NBINS; ++k) mag_sorted[k] = mag[k];
    qsort(mag_sorted, NBINS, sizeof(float), cmp_float);
    float p25 = mag_sorted[(int)(0.25f*(NBINS-1))];
    float p50 = mag_sorted[(int)(0.50f*(NBINS-1))];
    float p75 = mag_sorted[(int)(0.75f*(NBINS-1))];
    float p90 = mag_sorted[(int)(0.90f*(NBINS-1))];
    out[0]=centroid; out[1]=bw; out[2]=roll85; out[3]=roll95; out[4]=slope;
    out[5]=flatness; out[6]=entropy; out[7]=mean_mag; out[8]=sqrtf(var_mag);
    out[9]=max_mag; out[10]=min_mag; out[11]=p25; out[12]=p50; out[13]=p75; out[14]=p90;
}

static void extract_band(const float *mag, const float *freqs, float *out) {
    int nb = 32; float maxf = VBL2_FS/2.0f, total_e = 0.0f;
    for (int k = 0; k < NBINS; ++k) total_e += mag[k]*mag[k];
    total_e += 1e-12f;
    for (int i = 0; i < nb; ++i) {
        float lo = maxf*i/nb, hi = maxf*(i+1)/nb, be = 0.0f;
        for (int k = 0; k < NBINS; ++k) if (freqs[k] >= lo && freqs[k] < hi) be += mag[k]*mag[k];
        out[i] = be/total_e; out[nb+i] = logf(be+1e-12f);
    }
}

static void extract_peaks(const float *mag, const float *freqs, float *out) {
    int n_peaks = 6;
    static int idx[NBINS]; int n_pk = 0;
    for (int k = 1; k < NBINS-1; ++k)
        if (mag[k] > mag[k-1] && mag[k] > mag[k+1]) idx[n_pk++] = k;
    for (int t = 0; t < n_peaks && t < n_pk; ++t) {
        int best = t;
        for (int j = t+1; j < n_pk; ++j) if (mag[idx[j]] > mag[idx[best]]) best = j;
        int tmp = idx[t]; idx[t] = idx[best]; idx[best] = tmp;
    }
    int take = n_pk < n_peaks ? n_pk : n_peaks;
    for (int a = 0; a < take; ++a)
        for (int b = a+1; b < take; ++b) if (freqs[idx[b]] < freqs[idx[a]]) { int t=idx[a]; idx[a]=idx[b]; idx[b]=t; }
    for (int i = 0; i < 2*n_peaks; ++i) out[i] = 0.0f;
    for (int i = 0; i < take; ++i) { out[2*i]=freqs[idx[i]]; out[2*i+1]=mag[idx[i]]; }
}

static void extract_cepstral(float *out) {
    for (int k = 0; k < N; ++k) { s_cre[k] = 0.0f; s_cim[k] = 0.0f; }
    s_cre[0] = logf(s_mag[0]+1e-12f);
    for (int k = 1; k < NBINS-1; ++k) { float ls = logf(s_mag[k]+1e-12f); s_cre[k] = ls; s_cre[N-k] = ls; }
    s_cre[NBINS-1] = logf(s_mag[NBINS-1]+1e-12f);
    vbl_ifft(s_cre, s_cim, N);
    for (int i = 0; i < 17; ++i) out[i] = s_cre[i];
}

void vbl_extract_cfd(const float *window, int win_len, float feat[120]) {
    (void)win_len;
    extract_time(window, feat);
    for (int i = 0; i < N; ++i) { s_re[i] = window[i]; s_im[i] = 0.0f; }
    vbl_fft(s_re, s_im, N);
    for (int k = 0; k < NBINS; ++k) {
        s_mag[k] = sqrtf(s_re[k]*s_re[k] + s_im[k]*s_im[k]);
        s_freqs[k] = (float)k * VBL2_FS / N;
    }
    extract_freq(s_mag, s_freqs, feat+12);
    extract_band(s_mag, s_freqs, feat+27);
    extract_peaks(s_mag, s_freqs, feat+91);
    extract_cepstral(feat+103);
}
