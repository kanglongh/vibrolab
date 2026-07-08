/* vibrolab v1.5 · infer.h — Bot-40 选维 + 标准化 + LR 推理 (float32). */
#ifndef VBL_INFER_H
#define VBL_INFER_H
#ifdef __cplusplus
extern "C" {
#endif
int vbl_infer(const float feat[120], float *out_conf);
#ifdef __cplusplus
}
#endif
#endif
