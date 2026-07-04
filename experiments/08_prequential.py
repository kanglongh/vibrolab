"""
Experiment 08 · Prequential Cohen's d (流式部署诚实性验证).

背景
----
exp03 里 Cohen's d 是在**全部目标域样本**上计算的 — 这是"transductive UDA"设定,
文献 (TCA, CORAL, DANN) 都这么做. 但在真实工业部署中, 目标域样本是**流式**到达的:
新设备刚上线时你根本没见过多少数据.

本实验用只见过前 N% 目标域样本的 Cohen's d 去挑 Bot-40, 然后在剩余样本上评估.
这个数字才反映"真实上线时能达到的精度". 通常会比 transductive 版本低几个百分点.

设定
----
- 目标域样本按时间顺序 (窗口原始次序, 保留时间局部性) 分为 warmup / test 两段
- warmup 比例扫: 5% / 10% / 20% / 50% / 100% (100% 对应 exp03 的原设定)
- 每档跑 3 个种子 (打乱窗口顺序中的 warmup 起始位置), 报告均值±方差

输出:
    outputs/exp08_prequential.csv  各 warmup 比例下的精度
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

CLASSIFIERS = {
    'SVM-RBF':   lambda: SVC(kernel='rbf', class_weight='balanced', random_state=RANDOM_STATE),
    'LinearSVM': lambda: LinearSVC(class_weight='balanced', max_iter=5000, dual='auto', random_state=RANDOM_STATE),
    'LR':        lambda: LogisticRegression(class_weight='balanced', max_iter=5000, random_state=RANDOM_STATE),
    'KNN':       lambda: KNeighborsClassifier(5),
}

WARMUP_RATIOS = [0.05, 0.10, 0.20, 0.50, 1.00]
N_SEEDS = 3


def _eval(X_src, y_src, X_tgt_te, y_tgt_te, dims, clf_factory):
    sc = StandardScaler().fit(X_src[:, dims])
    clf = clf_factory()
    clf.fit(sc.transform(X_src[:, dims]), y_src)
    return accuracy_score(y_tgt_te, clf.predict(sc.transform(X_tgt_te[:, dims])))


def main(src_load: int = 0, tgt_load: int = 3, window: int = 2048, step: int = 2048):
    t0 = time.time()
    print('=' * 70)
    print(f'Experiment 08 · Prequential Cohen\'s d  ({src_load}HP -> {tgt_load}HP, 10-class)')
    print('=' * 70)

    print(f'\n[1/3] Loading...')
    src_samples = vio.load_cwru(loads=[src_load])
    tgt_samples = vio.load_cwru(loads=[tgt_load])
    X_src_raw, y_src, _ = vio.build_dataset(src_samples, window=window, step=step)
    X_tgt_raw, y_tgt, _ = vio.build_dataset(tgt_samples, window=window, step=step)
    X_src = features.extract_cfd(X_src_raw, fs=vio.FS_12K)
    X_tgt = features.extract_cfd(X_tgt_raw, fs=vio.FS_12K)
    print(f'  Source: {X_src.shape}, Target: {X_tgt.shape}')

    print(f'\n[2/3] Prequential evaluation (Bot-{K_BOT}, {N_SEEDS} seeds/ratio)...')
    n_tgt = X_tgt.shape[0]
    rows = []
    for ratio in WARMUP_RATIOS:
        for seed in range(N_SEEDS):
            rng = np.random.RandomState(seed)
            perm = rng.permutation(n_tgt)
            n_warm = max(K_BOT, int(round(n_tgt * ratio)))
            warm_idx = perm[:n_warm]
            X_warm = X_tgt[warm_idx]
            # 在 warmup 样本上做预对齐
            _, ranked = features.prealign_select(X_src, X_warm, method='cohen')
            bot_dims = features.select_low_drift(ranked, k=K_BOT)
            # 测试用剩余样本 (ratio=1 时测试等于整个目标域, 复现 exp03)
            if ratio >= 1.0:
                test_idx = np.arange(n_tgt)  # 与 exp03 对齐: 测试整个目标域
            else:
                test_idx = perm[n_warm:]
            X_te, y_te = X_tgt[test_idx], y_tgt[test_idx]
            for name, fac in CLASSIFIERS.items():
                acc_full = _eval(X_src, y_src, X_te, y_te, np.arange(120), fac)
                acc_bot  = _eval(X_src, y_src, X_te, y_te, bot_dims, fac)
                rows.append({
                    'warmup_ratio': ratio,
                    'seed': seed,
                    'n_warmup': n_warm,
                    'n_test':   len(test_idx),
                    'classifier': name,
                    'full_120': acc_full,
                    'bot40':    acc_bot,
                })
        # 打印本档的汇总
        sub = pd.DataFrame([r for r in rows if r['warmup_ratio'] == ratio])
        print(f'  warmup={ratio*100:5.1f}%  n_warm={int(sub["n_warmup"].mean()):3d}  ' + '  '.join(
            f'{c}={sub[sub["classifier"]==c]["bot40"].mean():.3f}±{sub[sub["classifier"]==c]["bot40"].std():.3f}'
            for c in CLASSIFIERS))

    df = pd.DataFrame(rows)
    csv = paths.OUT / 'exp08_prequential.csv'
    df.to_csv(csv, index=False, encoding='utf-8-sig')
    print(f'\n[Save] {csv.name}')

    # 打印精度衰减表: 显示 warmup 比例减少时 Bot-40 精度掉多少
    print(f'\n[Decay] Bot-{K_BOT} accuracy as target-domain warmup shrinks:')
    print(f'  {"warmup":>7s}  ' + '  '.join(f'{c:>12s}' for c in CLASSIFIERS))
    for ratio in WARMUP_RATIOS:
        sub = df[df['warmup_ratio'] == ratio]
        cells = []
        for c in CLASSIFIERS:
            mean = sub[sub['classifier'] == c]['bot40'].mean()
            std = sub[sub['classifier'] == c]['bot40'].std()
            cells.append(f'{mean:.3f}±{std:.3f}')
        print(f'  {ratio*100:5.1f}%  ' + '  '.join(f'{c:>12s}' for c in cells))

    print(f'\n[Done] Total time: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
