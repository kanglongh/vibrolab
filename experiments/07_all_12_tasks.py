"""
Experiment 07 · 全 12 任务跨负载评估.

目的
----
只在 0HP → 3HP 一个任务上报数字, 会让审阅者怀疑"cherry-picking".
本实验遍历所有 (src_load, tgt_load) 组合 (12 个任务), 报告每个任务和平均值,
让读者看到方差和最差情况. 这是 DCDAN/AMDA/CDHM 等论文的标配.

输出:
    outputs/exp07_all_tasks.csv         每任务 × 每分类器 的 full/bot40 精度
    outputs/exp07_task_summary.csv      按分类器聚合的 12 任务平均/最差/方差
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
    sc = StandardScaler().fit(X_src[:, dims])
    clf = clf_factory()
    clf.fit(sc.transform(X_src[:, dims]), y_src)
    return accuracy_score(y_tgt, clf.predict(sc.transform(X_tgt[:, dims])))


def _extract_by_load(all_X, all_y, all_loads, load):
    """从合并数据中筛出指定负载."""
    m = (all_loads == load)
    return all_X[m], all_y[m]


def main(window: int = 2048, step: int = 2048, k: int = 40):
    t0 = time.time()
    print('=' * 70)
    print('Experiment 07 · Full 12-Task Cross-Load Benchmark  (10-class)')
    print('=' * 70)

    # 一次性加载所有 4 负载, 提取特征, 之后按负载切子集
    print('\n[1/3] Loading CWRU (all 4 loads)...')
    samples = vio.load_cwru()
    X_raw, y, loads = vio.build_dataset(samples, window=window, step=step)
    print(f'  Total windows: {X_raw.shape[0]}, classes: {len(np.unique(y))}')
    for ld in vio.LOADS:
        print(f'    load={ld}HP: {(loads == ld).sum()} windows')

    print('\n[2/3] Extracting CFD features (once for all)...')
    X = features.extract_cfd(X_raw, fs=vio.FS_12K)
    print(f'  Feature matrix: {X.shape}')

    # 遍历所有 (src, tgt) 组合 (src != tgt), 共 4 × 3 = 12 个任务
    print(f'\n[3/3] Running 12 cross-load tasks (Bot-{k})...')
    tasks = [(s, t) for s in vio.LOADS for t in vio.LOADS if s != t]
    print(f'  Tasks: {tasks}\n')

    rows = []
    for src, tgt in tasks:
        X_src, y_src = _extract_by_load(X, y, loads, src)
        X_tgt, y_tgt = _extract_by_load(X, y, loads, tgt)
        # 特征预对齐: 选分布差异小的维度
        _, ranked = features.prealign_select(X_src, X_tgt, method='cohen')
        bot_dims = features.select_low_drift(ranked, k=k)
        ALL = np.arange(120)

        for name, fac in CLASSIFIERS.items():
            acc_full = _eval(X_src, y_src, X_tgt, y_tgt, ALL, fac)
            acc_bot  = _eval(X_src, y_src, X_tgt, y_tgt, bot_dims, fac)
            rows.append({
                'src_load': src, 'tgt_load': tgt, 'task': f'{src}HP->{tgt}HP',
                'classifier': name,
                'full_120': acc_full, 'bot40': acc_bot,
                'gain': acc_bot - acc_full,
            })
        print(f'  {src}HP->{tgt}HP  ' +
              '  '.join(f'{c}={_row(rows, src, tgt, c):.3f}' for c in CLASSIFIERS))

    df = pd.DataFrame(rows)
    csv1 = paths.OUT / 'exp07_all_tasks.csv'
    df.to_csv(csv1, index=False, encoding='utf-8-sig')
    print(f'\n[Save] {csv1.name}')

    # 汇总: 每分类器的 12 任务统计
    print(f'\n[Summary] Bot-{k} across 12 tasks:')
    print(f'  {"Classifier":12s}  {"mean":>7s}  {"std":>6s}  {"min":>7s}  {"max":>7s}  |  {"full mean":>9s}')
    summary_rows = []
    for c in CLASSIFIERS:
        sub = df[df['classifier'] == c]
        bot = sub['bot40'].values
        full = sub['full_120'].values
        summary_rows.append({
            'classifier': c,
            'bot40_mean': bot.mean(),
            'bot40_std':  bot.std(),
            'bot40_min':  bot.min(),
            'bot40_max':  bot.max(),
            'full_mean':  full.mean(),
            'full_min':   full.min(),
            'gain_mean':  (bot - full).mean(),
        })
        print(f'  {c:12s}  {bot.mean():.4f}  {bot.std():.4f}  {bot.min():.4f}  {bot.max():.4f}  |  {full.mean():.4f}')

    df_sum = pd.DataFrame(summary_rows)
    csv2 = paths.OUT / 'exp07_task_summary.csv'
    df_sum.to_csv(csv2, index=False, encoding='utf-8-sig')
    print(f'\n[Save] {csv2.name}')
    print(f'\n[Done] Total time: {time.time() - t0:.1f}s')


def _row(rows, s, t, c):
    for r in rows[::-1]:
        if r['src_load'] == s and r['tgt_load'] == t and r['classifier'] == c:
            return r['bot40']
    return float('nan')


if __name__ == '__main__':
    main()
