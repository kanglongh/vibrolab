"""
Experiment 10 · 目标域加噪声鲁棒性测试.

背景
----
CWRU 数据集在实验室采集, 噪声水平低 (信噪比 SNR ≈ 30+ dB). 真实工业现场振动
信号往往 SNR 只有 -5 到 15 dB. 本实验模拟这个 gap: 给目标域信号添加高斯噪声,
按 SNR = {∞, 20, 10, 5, 0, -5} dB 6 档扫描, 观察 Bot-40 修复效果是否仍稳定.

如果 Bot-40 相比全 120 维在低 SNR 下**优势仍在**, 说明 Cohen's d 挑出的低漂
维度对噪声也稳; 反之如果崩塌, 说明 Bot-40 只对特定 domain shift 稳, 对噪声
无免疫力 — 这是需要坦白说明的局限.

输出:
    outputs/exp10_noise_robustness.csv
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vibrolab import io as vio, features, paths

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
K_BOT = 40
N_SEEDS = 3

CLASSIFIERS = {
    'SVM-RBF':   lambda: SVC(kernel='rbf', class_weight='balanced', random_state=RANDOM_STATE),
    'LinearSVM': lambda: LinearSVC(class_weight='balanced', max_iter=5000, dual='auto', random_state=RANDOM_STATE),
    'LR':        lambda: LogisticRegression(class_weight='balanced', max_iter=5000, random_state=RANDOM_STATE),
    'KNN':       lambda: KNeighborsClassifier(5),
}

# SNR (dB) 档位, None = 无噪声 (作为参考)
SNR_LEVELS = [None, 20, 10, 5, 0, -5]


def _add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.RandomState) -> np.ndarray:
    """给一维/二维信号添加高斯白噪声, 达到指定 SNR (dB).

    SNR = 10 * log10(P_signal / P_noise)
    """
    if snr_db is None or np.isinf(snr_db):
        return signal
    sig_power = np.mean(signal ** 2, axis=-1, keepdims=True)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.randn(*signal.shape).astype(signal.dtype) * np.sqrt(noise_power)
    return signal + noise


def _eval(X_src, y_src, X_tgt, y_tgt, dims, clf_factory):
    sc = StandardScaler().fit(X_src[:, dims])
    clf = clf_factory()
    clf.fit(sc.transform(X_src[:, dims]), y_src)
    return accuracy_score(y_tgt, clf.predict(sc.transform(X_tgt[:, dims])))


def main(src_load: int = 0, tgt_load: int = 3, window: int = 2048, step: int = 2048):
    t0 = time.time()
    print('=' * 70)
    print(f'Experiment 10 · Noise Robustness ({src_load}HP -> {tgt_load}HP, 10-class)')
    print('=' * 70)

    print(f'\n[1/3] Loading and extracting source features (clean)...')
    src_samples = vio.load_cwru(loads=[src_load])
    X_src_raw, y_src, _ = vio.build_dataset(src_samples, window=window, step=step)
    X_src = features.extract_cfd(X_src_raw, fs=vio.FS_12K)

    tgt_samples = vio.load_cwru(loads=[tgt_load])
    X_tgt_raw_clean, y_tgt, _ = vio.build_dataset(tgt_samples, window=window, step=step)
    print(f'  Source: {X_src.shape}  Target-clean: {X_tgt_raw_clean.shape}')

    print(f'\n[2/3] Sweeping SNR = {SNR_LEVELS} dB (Bot-{K_BOT}, {N_SEEDS} seeds)...')
    rows = []
    for snr in SNR_LEVELS:
        for seed in range(N_SEEDS):
            rng = np.random.RandomState(seed)
            X_tgt_raw_noisy = _add_awgn(X_tgt_raw_clean, snr, rng)
            X_tgt = features.extract_cfd(X_tgt_raw_noisy, fs=vio.FS_12K)
            # 在噪声目标域上算漂移, 挑 Bot-40 低漂子集
            _, ranked = features.prealign_select(X_src, X_tgt, method='cohen')
            bot_dims = features.select_low_drift(ranked, k=K_BOT)
            for name, fac in CLASSIFIERS.items():
                acc_full = _eval(X_src, y_src, X_tgt, y_tgt, np.arange(120), fac)
                acc_bot  = _eval(X_src, y_src, X_tgt, y_tgt, bot_dims, fac)
                rows.append({
                    'snr_db': 'clean' if snr is None else snr,
                    'seed': seed,
                    'classifier': name,
                    'full_120': acc_full,
                    'bot40':    acc_bot,
                    'gain':     acc_bot - acc_full,
                })
        sub = pd.DataFrame([r for r in rows if r['snr_db'] == ('clean' if snr is None else snr)])
        label = 'clean' if snr is None else f'{snr:+3d}dB'
        cells = '  '.join(
            f'{c}: full={sub[sub["classifier"]==c]["full_120"].mean():.3f} bot={sub[sub["classifier"]==c]["bot40"].mean():.3f}'
            for c in CLASSIFIERS
        )
        print(f'  SNR={label:>7s}  {cells}')

    df = pd.DataFrame(rows)
    csv = paths.OUT / 'exp10_noise_robustness.csv'
    df.to_csv(csv, index=False, encoding='utf-8-sig')
    print(f'\n[Save] {csv.name}')

    print(f'\n[Summary] Bot-{K_BOT} vs full-120 across SNR:')
    print(f'  {"SNR":>6s}  ' + '  '.join(f'{c+"_full":>13s} {c+"_bot":>13s}' for c in CLASSIFIERS))
    for snr in SNR_LEVELS:
        tag = 'clean' if snr is None else f'{snr}'
        sub = df[df['snr_db'] == ('clean' if snr is None else snr)]
        cells = []
        for c in CLASSIFIERS:
            sub_c = sub[sub['classifier'] == c]
            cells.append(f'{sub_c["full_120"].mean():.3f}±{sub_c["full_120"].std():.3f}')
            cells.append(f'{sub_c["bot40"].mean():.3f}±{sub_c["bot40"].std():.3f}')
        print(f'  {tag:>6s}  ' + '  '.join(f'{c:>13s}' for c in cells))

    print(f'\n[Done] Total time: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
