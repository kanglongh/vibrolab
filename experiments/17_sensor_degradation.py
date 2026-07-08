"""实验 17 · 传感器降级模拟: 便宜传感器能不能撑住 vibrolab 精度?

原理: CWRU 用工业 IEPE 加速度计 (低噪声, 高带宽) 采的干净数据.
真机部署若换便宜 MEMS, 信号会:
    1. 带宽变窄 (低通滤波掉高频)
    2. 噪声底变高 (加白噪声)
    3. ADC 位深降低 (量化噪声)

本脚本对 CWRU 信号做这三重降级, 用同一个 Bot-40+LR 模型测精度衰减.
用来预估真机表现, 不用买传感器.

档位 (datasheet 典型值 + 工程师会选的量程档):
    IEPE 基线      : CWRU 原信号 (PCB 353B33 类, 6 kHz, 4 μg/√Hz), 无降级
    ADXL1002 档   : 11 kHz 带宽, 25 μg/√Hz 噪声, 12-bit ADC (ESP32 内置), ±50g 量程 (原生)
    ADXL355 档    :  1.9 kHz 带宽, 25 μg/√Hz 噪声 @ ±8g 档, 20-bit ADC
    MPU6050 档    :  260 Hz 带宽, 400 μg/√Hz 噪声, 16-bit ADC, ±16g 量程 (AFS_SEL=3)

量程选择说明: CWRU 信号峰值 ~5g. 工程师会选量程 ≥5g 避免 clip. ADXL355 ±2g 档 22.5
μg/√Hz 的噪声不适用 (会 clip), 用 ±8g 档 25 μg. MPU6050 类似, 用 ±16g 档.

关键: 大量程 + 有限 bit → 有效量化 SNR 差. 这是模拟输出→ADC 的自然物理结果.
比如 ADXL1002 ±50g @ 12-bit, 步长 = 100/4096 ≈ 24 mg; CWRU 5g 信号只用满量程 10%,
量化 SNR 就比"12-bit 全量程"差 20 dB (~3.3 bit).

datasheet 出处:
    ADXL1002: ADI datasheet Rev 0, "Linear frequency response range from dc to 11 kHz"
    ADXL355:  ADI datasheet Rev C, "BANDWIDTH 3 dB: 1.9 kHz"
    MPU6050:  InvenSense PS-MPU-6000A, DLPF 默认配置下加速度计带宽 260 Hz

用法:
    python experiments/17_sensor_degradation.py
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))

from vibrolab import io as vio, features as vbf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.signal import butter, filtfilt

WINDOW = 2048
STEP = 2048
FS = 12000
TRAIN_LOADS = [0, 3]
K_BOT = 40
SEED = 42

# 传感器档位: (bw_Hz, noise_ug/√Hz, adc_bits, full_scale_g)
SENSOR_SPECS = [
    ('IEPE 基线 (CWRU 原信号)',      None,  4,   24, None),
    ('ADXL1002 (工业 MEMS)',       11000, 25,   12,   50),
    ('ADXL355  (低带宽高精度 MEMS)', 1900, 25,   20,    8),
    ('MPU6050  (消费级 IMU)',       260,  400,  16,   16),
]


def degrade(signal, bandwidth_hz, noise_ug_sqrtHz, adc_bits, full_scale_g, rng):
    """把干净信号降级到指定传感器规格. signal: (N,) float32, 单位 g.

    模拟三重损失:
      1. 传感器内置低通 (带宽限制) → butterworth 4 阶低通
      2. 传感器噪声底 → 每采样点独立高斯白噪声, σ = density × √(有效带宽)
      3. ADC 固定量程量化 → 越大量程 SNR 越差 (模拟输出→ADC 的自然物理结果)
    """
    x = signal.astype(np.float64).copy()

    # 1. 低通滤波 (传感器带宽限制)
    if bandwidth_hz is not None and bandwidth_hz < FS / 2:
        b, a = butter(4, bandwidth_hz / (FS/2), 'low')
        x = filtfilt(b, a, x)

    # 2. 加白噪声 (每采样点独立高斯). 有效噪声带宽 = min(信号带宽, 系统 Nyquist).
    #    (先滤波后加噪声, 所以噪声带宽 = 滤波后的带宽)
    enb = min(bandwidth_hz or FS/2, FS/2)
    noise_std_g = noise_ug_sqrtHz * 1e-6 * np.sqrt(enb)
    x = x + rng.randn(len(x)) * noise_std_g

    # 3. ADC 固定量程量化. 超过量程被 clip (真实 ADC 行为).
    #    量化 SNR = 满量程 / 量化步长 → 大量程 + 少 bit → SNR 差 (物理事实)
    if full_scale_g is not None and adc_bits < 24:
        x = np.clip(x, -full_scale_g, full_scale_g)
        step = 2.0 * full_scale_g / (2 ** adc_bits)
        x = np.round(x / step) * step

    return x.astype(np.float32)


def eval_degraded(X_raw_test, y_test, spec, bot40, sc, lr, rng):
    """对测试集施加降级后重新提特征 + 分类."""
    bandwidth, noise, bits, full_scale = spec
    X_deg = np.array([degrade(X_raw_test[i], bandwidth, noise, bits, full_scale, rng)
                      for i in range(len(X_raw_test))])
    feats = vbf.extract_cfd(X_deg, fs=FS).astype(np.float32)
    yp = lr.predict(sc.transform(feats[:, bot40]))
    return accuracy_score(y_test, yp)


def main():
    print('=' * 68)
    print(f'传感器降级模拟: 训练 {TRAIN_LOADS}HP → 测 held-out 1/2HP')
    print(f'训练用 CWRU 原信号 (IEEE 级), 测试信号按不同传感器规格降级')
    print('=' * 68)

    # ---- Step 1: 加载 + 训模型 (在干净 0+3HP 上, 只训一次) ----
    print('\n[1] 加载 CWRU + 提特征 (干净信号)...')
    samples = vio.load_cwru(loads=[0, 1, 2, 3])
    X_raw, y, ld = vio.build_dataset(samples, window=WINDOW, step=STEP)
    X_clean = vbf.extract_cfd(X_raw, fs=FS).astype(np.float32)

    print('[2] Cohen\'s d 选 Bot-40 + LR 训练 (仅训练工况, 干净数据)...')
    X0, X3 = X_clean[ld == 0], X_clean[ld == 3]
    d_full, ranked = vbf.prealign_select(X0, X3, method='cohen')
    bot40 = np.sort(vbf.select_low_drift(ranked, k=K_BOT))
    train_mask = np.isin(ld, TRAIN_LOADS)
    Xs, ys = X_clean[train_mask][:, bot40], y[train_mask]
    sc = StandardScaler().fit(Xs)
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED).fit(sc.transform(Xs), ys)

    # ---- Step 3: 对每个 held-out 工况 × 每个传感器档 测精度 ----
    held_out_loads = [L for L in [0, 1, 2, 3] if L not in TRAIN_LOADS]
    print(f'\n[3] Held-out 工况 {held_out_loads}HP × 4 档传感器:\n')
    print(f'{"传感器档":<26} {"1HP":>8} {"2HP":>8} {"均值":>8}')
    print('-' * 55)

    for name, bw, noise, bits, fs_g in SENSOR_SPECS:
        rng = np.random.RandomState(SEED)   # 每档独立种子, 可复现
        accs = []
        for L in held_out_loads:
            m = ld == L
            acc = eval_degraded(X_raw[m], y[m], (bw, noise, bits, fs_g),
                                bot40, sc, lr, rng)
            accs.append(acc)
        mean_acc = np.mean(accs)
        row = f'{name:<26}'
        for a in accs:
            row += f' {a*100:>7.2f}%'
        row += f' {mean_acc*100:>7.2f}%'
        print(row)

    print('\n结论:')
    print('  - IEEE 基线是天花板 (干净信号)')
    print('  - 精度掉幅 = 传感器降级带来的信息损失')
    print('  - 便宜传感器能否用, 取决于业务能否接受这个衰减')


if __name__ == '__main__':
    main()
