/* vibrolab v1.5 · features.h — 120 维 CFD 特征 (float32, 镜像 Python vibrolab/features.py). */
#ifndef VBL2_FEATURES_H
#define VBL2_FEATURES_H
#ifdef __cplusplus
extern "C" {
#endif
#define VBL2_WINDOW 2048
#define VBL2_FS 12000
void vbl_extract_cfd(const float *window, int win_len, float feat[120]);
#ifdef __cplusplus
}
#endif
#endif
