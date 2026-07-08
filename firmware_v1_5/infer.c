/* vibrolab v1.5 · infer.c — Bot-40 选维 + 标准化 + LR (float32). */
#include "infer.h"
#include "model.h"
#include <math.h>

int vbl_infer(const float feat[120], float *out_conf) {
    float x_std[40];
    for (int j = 0; j < VIBROLAB_BOT40_K; ++j) {
        int idx = VIBROLAB_BOT40_INDICES[j];
        float s = VIBROLAB_SCALER_SCALE[j];
        x_std[j] = (feat[idx] - VIBROLAB_SCALER_MEAN[j]) / (s == 0.0f ? 1e-12f : s);
    }
    float z[16];
    float zmax = -1e30f;
    for (int k = 0; k < VIBROLAB_N_CLASSES; ++k) {
        float acc = VIBROLAB_LR_INTERCEPT[k];
        const float *row = VIBROLAB_LR_COEF[k];
        for (int j = 0; j < VIBROLAB_BOT40_K; ++j) acc += row[j] * x_std[j];
        z[k] = acc;
        if (acc > zmax) zmax = acc;
    }
    int pred = 0;
    for (int k = 1; k < VIBROLAB_N_CLASSES; ++k) if (z[k] > z[pred]) pred = k;
    if (out_conf) {
        float sumexp = 0.0f;
        for (int k = 0; k < VIBROLAB_N_CLASSES; ++k) sumexp += expf(z[k] - zmax);
        *out_conf = expf(z[pred] - zmax) / sumexp;
    }
    return pred;
}
