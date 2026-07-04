"""
Experiment 03 · 跨负载条件下的诊断精度断崖与修复.

场景:
    源域 = 0 HP 负载下的 4 类样本
    目标域 = 3 HP 负载下的 4 类样本
    在源域训练, 在目标域直接测试

三段实验:
    A. 全量 120 维: 展示"跨负载精度断崖"
    B. Cohen's d 归因: 找出高漂移维
    C. Bot-40 低漂子集: 修复跨负载精度

输出:
    outputs/exp03_cross_condition.csv    全量 vs Bot-40 精度对比表
    outputs/exp03_cohens_d.csv           120 维 Cohen's d 排序
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

CLASSIFIERS = {
    'SVM-RBF':   lambda: SVC(kernel='rbf', class_weight='balanced', random_state=RANDOM_STATE),
    'LinearSVM': lambda: LinearSVC(class_weight='balanced', max_iter=5000, dual='auto', random_state=RANDOM_STATE),
    'LR':        lambda: LogisticRegression(class_weight='balanced', max_iter=5000, random_state=RANDOM_STATE),
    'KNN':       lambda: KNeighborsClassifier(5),
}


def _eval(X_src, y_src, X_tgt, y_tgt, dims, clf_factory):
    """标准化 -> 训练 -> 测试, 返回目标域精度."""
    sc = StandardScaler().fit(X_src[:, dims])
    clf = clf_factory()
    clf.fit(sc.transform(X_src[:, dims]), y_src)
    return accuracy_score(y_tgt, clf.predict(sc.transform(X_tgt[:, dims])))


def _eval_all(X_src, y_src, X_tgt, y_tgt, dims):
    return {name: _eval(X_src, y_src, X_tgt, y_tgt, dims, fac)
            for name, fac in CLASSIFIERS.items()}


def main(src_load: int = 0, tgt_load: int = 3, window: int = 2048, step: int = 2048):
    t0 = time.time()
    print('=' * 70)
    print(f'Experiment 03 · Cross-Load Diagnosis  ({src_load}HP → {tgt_load}HP, 10-class)')
    print('=' * 70)

    # ------------------------------------------------------------
    # 1. 加载源域 / 目标域数据
    # ------------------------------------------------------------
    print(f'\n[1/4] Loading CWRU data (10-class scheme)...')
    src_samples = vio.load_cwru(loads=[src_load])
    tgt_samples = vio.load_cwru(loads=[tgt_load])
    X_src_raw, y_src, _ = vio.build_dataset(src_samples, window=window, step=step)
    X_tgt_raw, y_tgt, _ = vio.build_dataset(tgt_samples, window=window, step=step)
    print(f'  Source ({src_load}HP): {X_src_raw.shape[0]} windows, {len(np.unique(y_src))} classes')
    print(f'  Target ({tgt_load}HP): {X_tgt_raw.shape[0]} windows, {len(np.unique(y_tgt))} classes')

    # ------------------------------------------------------------
    # 2. 提取 CFD 120 维特征
    # ------------------------------------------------------------
    print(f'\n[2/4] Extracting CFD features (120 dims)...')
    X_src = features.extract_cfd(X_src_raw, fs=vio.FS_12K)
    X_tgt = features.extract_cfd(X_tgt_raw, fs=vio.FS_12K)
    print(f'  Source: {X_src.shape}, Target: {X_tgt.shape}')

    # ------------------------------------------------------------
    # 3A. 全量 120 维基线: 展示跨域断崖
    # ------------------------------------------------------------
    print(f'\n[3/4] Baseline: full 120-dim cross-load evaluation')
    ALL = np.arange(120)
    full_accs = _eval_all(X_src, y_src, X_tgt, y_tgt, ALL)
    for c, a in full_accs.items():
        marker = '  <-- CLIFF' if a < 0.70 else ''
        print(f'  {c:12s}  acc = {a:.4f}{marker}')

    # ------------------------------------------------------------
    # 3B. 特征预对齐
    # ------------------------------------------------------------
    print(f'\n[3/4] Feature pre-alignment: ranking by distribution shift')
    d_vals, ranked = features.prealign_select(X_src, X_tgt, method='cohen')
    print(f'  d range: [{d_vals.min():.3f}, {d_vals.max():.3f}]')
    print(f'  Top-5 highest-drift dims:')
    for r, i in enumerate(ranked[:5], 1):
        module = next(m for m, sl in features.MODULES.items() if sl.start <= i < sl.stop)
        print(f'    #{r}  dim={i:3d}  d={d_vals[i]:.3f}  module={module}')
    print(f'  Bottom-5 lowest-drift dims:')
    for r, i in enumerate(ranked[-5:], 1):
        module = next(m for m, sl in features.MODULES.items() if sl.start <= i < sl.stop)
        print(f'    #{r}  dim={i:3d}  d={d_vals[i]:.3f}  module={module}')

    # ------------------------------------------------------------
    # 3C. Bot-40 低漂特征子集
    # ------------------------------------------------------------
    print(f'\n[4/4] Evaluation: Bot-40 low-drift subset (k=40)')
    bot40 = features.select_low_drift(ranked, k=40)
    # 消融对照: 随机选 40 维作为最朴素的 baseline (跨 3 个种子取均值)
    rng = np.random.RandomState(RANDOM_STATE)
    rand_accs_list = []
    for _ in range(3):
        rand40 = rng.choice(120, size=40, replace=False)
        rand_accs_list.append(_eval_all(X_src, y_src, X_tgt, y_tgt, rand40))
    rand_accs = {c: float(np.mean([r[c] for r in rand_accs_list])) for c in CLASSIFIERS}
    bot_accs = _eval_all(X_src, y_src, X_tgt, y_tgt, bot40)
    for c in CLASSIFIERS:
        gain = bot_accs[c] - full_accs[c]
        print(f'  {c:12s}  full={full_accs[c]:.4f}  bot40={bot_accs[c]:.4f}  '
              f'rand40={rand_accs[c]:.4f}  gain={gain:+.4f}')

    # ------------------------------------------------------------
    # 4. 保存结果
    # ------------------------------------------------------------
    print(f'\n[Save] Writing CSVs to {paths.OUT}')
    rows = []
    for c in CLASSIFIERS:
        rows.append({
            'classifier': c,
            'full_120': full_accs[c],
            'bot40':    bot_accs[c],
            'rand40':   rand_accs[c],
            'gain_bot40_vs_full': bot_accs[c] - full_accs[c],
        })
    df = pd.DataFrame(rows)
    csv1 = paths.OUT / 'exp03_cross_condition.csv'
    df.to_csv(csv1, index=False, encoding='utf-8-sig')
    print(f'  {csv1.name}')

    d_df = pd.DataFrame({
        'dim': np.arange(120),
        'cohens_d': d_vals,
        'module': [next(m for m, sl in features.MODULES.items() if sl.start <= i < sl.stop)
                   for i in range(120)],
    }).sort_values('cohens_d', ascending=False).reset_index(drop=True)
    csv2 = paths.OUT / 'exp03_cohens_d.csv'
    d_df.to_csv(csv2, index=False, encoding='utf-8-sig')
    print(f'  {csv2.name}')

    print(f'\n[Done] Total time: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
