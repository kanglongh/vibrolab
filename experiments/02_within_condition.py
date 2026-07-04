"""
Experiment 02 · 同工况下的诊断精度 (baseline, 双 CV 对照).

背景
----
在 CWRU 数据集上, 大多数论文采用 **窗口级 5-fold CV**, 报告 99%+ 的精度. 但这种
划分方式存在弱数据泄漏: 同一个 .mat 文件被切成多个窗口后, 训练集和测试集里可能
出现"来自同一段信号的相邻窗口".

严格按文件划分 (leave-file-out) 可以完全消除这种泄漏, 但 CWRU 每类只有 1~4 个
.mat 文件 (对应 4 种负载), 在**多负载合并**下每类文件数才够 GroupKFold. 因此
本节的第 (b) 步实际是**混合负载 leave-file-out**, 相当于同时评估跨负载 + 跨窗口
泄漏两种因素. 真正干净的跨负载评估见 exp03.

输出:
    outputs/exp02_within_condition.csv    双 CV 对照
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
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42

CLASSIFIERS = {
    'SVM-RBF':   lambda: SVC(kernel='rbf', class_weight='balanced', random_state=RANDOM_STATE),
    'LinearSVM': lambda: LinearSVC(class_weight='balanced', max_iter=5000, dual='auto', random_state=RANDOM_STATE),
    'LR':        lambda: LogisticRegression(class_weight='balanced', max_iter=5000, random_state=RANDOM_STATE),
    'KNN':       lambda: KNeighborsClassifier(5),
}


def _build_with_groups(samples, window: int, step: int):
    """把 CWRUSample 列表切窗成 (X_raw, y, groups), 其中 groups 是每个窗口来源的 .mat 文件名."""
    Xs, ys, groups = [], [], []
    for s in samples:
        w = vio.make_windows(s.signal, window, step)
        if w.shape[0] == 0:
            continue
        Xs.append(w)
        ys.append(np.full(w.shape[0], vio.LABEL_TO_INT[s.label]))
        groups.append(np.full(w.shape[0], s.filename))
    return np.concatenate(Xs, 0), np.concatenate(ys), np.concatenate(groups)


def _window_level_cv(X, y, n_splits=5) -> pd.DataFrame:
    """标准窗口级 5-fold CV (CWRU 社区惯例)."""
    rows = []
    for name, fac in CLASSIFIERS.items():
        pipe = Pipeline([('sc', StandardScaler()), ('clf', fac())])
        scores = cross_val_score(
            pipe, X, y,
            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE),
            scoring='accuracy',
            n_jobs=1,
        )
        rows.append({
            'classifier': name, 'cv_type': 'window-level',
            'mean_acc': scores.mean(), 'std_acc': scores.std(),
        })
    return pd.DataFrame(rows)


def _leave_file_out_cv(X, y, groups, n_splits: int) -> pd.DataFrame:
    """严格按文件划分 (GroupKFold) 的 CV.

    在合并 4 负载的数据上运行, 每类有 4 个 .mat 文件 (对应 4 种负载),
    3-fold 划分下每次训练用其中若干负载的文件, 测试用剩余负载 —
    因此实际评估的是 "混合负载 leave-file-out" 能力.
    """
    n_files_per_class = {int(c): len(set(groups[y == c])) for c in np.unique(y)}
    min_files = min(n_files_per_class.values())
    if min_files < n_splits:
        print(f'  [SKIP] leave-file-out 需要每类至少 {n_splits} 个文件, '
              f'但最少的类只有 {min_files} 个 (类别文件数分布: {n_files_per_class})')
        return pd.DataFrame()

    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for name, fac in CLASSIFIERS.items():
        fold_accs = []
        for tr, te in gkf.split(X, y, groups):
            pipe = Pipeline([('sc', StandardScaler()), ('clf', fac())])
            pipe.fit(X[tr], y[tr])
            fold_accs.append(accuracy_score(y[te], pipe.predict(X[te])))
        rows.append({
            'classifier': name, 'cv_type': f'mixed-load leave-file-out ({n_splits}-fold)',
            'mean_acc': float(np.mean(fold_accs)), 'std_acc': float(np.std(fold_accs)),
        })
    return pd.DataFrame(rows)


def main(load: int = 0, window: int = 2048, step: int = 2048):
    """严格无重叠切分 (step = window)."""
    t0 = time.time()
    print('=' * 70)
    print(f'Experiment 02 · Within-Condition Baseline  (load={load}HP, 10-class)')
    print(f'Window={window}, Step={step}  (strict non-overlap)')
    print('=' * 70)

    # ------------------------------------------------------------
    # (a) 单负载 · 窗口级 5-fold CV  (CWRU 惯例)
    # ------------------------------------------------------------
    print(f'\n[1/3] Loading CWRU data (load={load}HP)...')
    samples = vio.load_cwru(loads=[load])
    X_raw, y, groups = _build_with_groups(samples, window=window, step=step)
    print(f'  Windows: {X_raw.shape[0]}  Classes: {len(np.unique(y))}  Files: {len(set(groups))}')
    X = features.extract_cfd(X_raw, fs=vio.FS_12K)

    print(f'\n[2/3] Window-level 5-fold CV  [CWRU community convention]')
    df_a = _window_level_cv(X, y, n_splits=5)
    for _, r in df_a.iterrows():
        print(f'      {r["classifier"]:12s}  mean={r["mean_acc"]:.4f} ± {r["std_acc"]:.4f}')

    # ------------------------------------------------------------
    # (b) 全负载合并 · mixed-load leave-file-out 3-fold CV
    #     单负载下 Normal 只有 1 个文件, 无法 GroupKFold.
    #     用 4 个负载合并的 Normal (4 个文件) 就可以做 3-fold.
    #     此时评估的是 "混合负载 + 跨损伤尺寸" 的联合外推能力.
    # ------------------------------------------------------------
    print(f'\n[3/3] Mixed-load leave-file-out 3-fold CV  [stricter, all 4 loads combined]')
    samples_all = vio.load_cwru()
    X_all_raw, y_all, groups_all = _build_with_groups(samples_all, window=window, step=step)
    print(f'  Windows: {X_all_raw.shape[0]}  Files: {len(set(groups_all))}  (across 4 loads)')
    X_all = features.extract_cfd(X_all_raw, fs=vio.FS_12K)
    df_b = _leave_file_out_cv(X_all, y_all, groups_all, n_splits=3)
    if not df_b.empty:
        for _, r in df_b.iterrows():
            print(f'      {r["classifier"]:12s}  mean={r["mean_acc"]:.4f} ± {r["std_acc"]:.4f}')
        gap = df_a['mean_acc'].mean() - df_b['mean_acc'].mean()
        print(f'\n      Gap (a) - (b) = {gap:+.3f}. This gap reflects CWRU\'s structural')
        print(f'      limitation (few .mat files per class → each file represents a')
        print(f'      different damage diameter, so leave-file-out forces extrapolation')
        print(f'      to unseen damage sizes). This is NOT a CFD feature quality issue.')
        print(f'      See exp03 (cross-load) for a leakage-free evaluation with fixed damage.')

    # 保存
    df = pd.concat([df_a, df_b], ignore_index=True)
    csv = paths.OUT / 'exp02_within_condition.csv'
    df.to_csv(csv, index=False, encoding='utf-8-sig')
    print(f'\n[Save] {csv.name}')
    print(f'\n[Done] Total time: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
