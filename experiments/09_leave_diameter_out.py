"""
Experiment 09 · Leave-One-Diameter-Out (未见损伤尺寸外推).

背景
----
exp02 的 "同工况" CV 用了合并 4 负载做 GroupKFold, 但训练/测试集之间**共享损伤
尺寸**, 只是负载不同 — 因此 99.7%+ 的数字其实混入了跨负载效应, 不代表"未见样本
外推能力". 真正衡量泛化的实验应该让模型面对**未见的损伤尺寸**.

CWRU 12k_DE 里 B/IR/OR 三大类各有 3 种损伤尺寸 (0.007", 0.014", 0.021"),
每个尺寸有 4 个 .mat 文件 (对应 4 种负载). 本实验:

    - 用 (故障类型, 尺寸) 作为 group, 共 9 个非 Normal group + 1 个 Normal group
    - Leave-one-diameter-out: 每次留 1 个 group (所有负载对应的文件) 作测试
    - 测试类的 3 个尺寸中另外 2 个尺寸在训练集里, 因此**类别不消失, 只是尺寸变了**

这个数字反映的才是"CFD 特征能否识别未见损伤尺寸下的故障". 通常会比 exp02 显著低,
因为不同损伤尺寸的振动特征差异不小. 我们用它来诚实地设置 CFD 特征的能力上限.

输出:
    outputs/exp09_leave_diameter_out.csv
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vibrolab import io as vio, features, paths

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42

CLASSIFIERS = {
    'SVM-RBF':   lambda: SVC(kernel='rbf', class_weight='balanced', random_state=RANDOM_STATE),
    'LinearSVM': lambda: LinearSVC(class_weight='balanced', max_iter=5000, dual='auto', random_state=RANDOM_STATE),
    'LR':        lambda: LogisticRegression(class_weight='balanced', max_iter=5000, random_state=RANDOM_STATE),
    'KNN':       lambda: KNeighborsClassifier(5),
}


# 把 10 类 (维持) 映射到 4 大类 (评估用): N/B/IR/OR
LABEL_TO_MAJOR = {
    'N':        'N',
    'B007':     'B',  'B014': 'B',  'B021': 'B',
    'IR007':    'IR', 'IR014': 'IR', 'IR021': 'IR',
    'OR007@6':  'OR', 'OR014@6': 'OR', 'OR021@6': 'OR',
}
MAJOR_LABEL_NAMES = ['N', 'B', 'IR', 'OR']
MAJOR_TO_INT = {n: i for i, n in enumerate(MAJOR_LABEL_NAMES)}


def _build_with_groups(samples, window, step):
    """把样本按 (故障类型, 尺寸) 作为 group, 4 大类作为 label 输出.

    Normal 只有一个 group ("N"), 其它 3 大类各 3 个尺寸 group.
    """
    Xs, ys, groups = [], [], []
    for s in samples:
        w = vio.make_windows(s.signal, window, step)
        if w.shape[0] == 0:
            continue
        # 用 10 类标签作 group ID (每个 group 代表一种 (类, 尺寸) 组合)
        group_id = s.label
        major_lbl = LABEL_TO_MAJOR[s.label]
        Xs.append(w)
        ys.append(np.full(w.shape[0], MAJOR_TO_INT[major_lbl]))
        groups.append(np.full(w.shape[0], group_id))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(groups)


def main(window: int = 2048, step: int = 2048):
    t0 = time.time()
    print('=' * 70)
    print('Experiment 09 · Leave-One-Diameter-Out  (all 4 loads, 4-major-class output)')
    print('=' * 70)

    print(f'\n[1/3] Loading (all 4 loads, 10-class fine labels)...')
    samples = vio.load_cwru()
    X_raw, y, groups = _build_with_groups(samples, window, step)
    print(f'  Windows: {X_raw.shape[0]}  Major classes: {len(np.unique(y))}')
    all_groups = sorted(set(groups))
    print(f'  Groups (diameters): {all_groups}')

    print(f'\n[2/3] Extracting CFD features...')
    X = features.extract_cfd(X_raw, fs=vio.FS_12K)

    # 只对 B/IR/OR 三大类做 leave-one-diameter-out (Normal 只有一个 group, 不能留出)
    holdout_groups = [g for g in all_groups if g != 'N']
    print(f'\n[3/3] Leave-one-diameter-out ({len(holdout_groups)} folds)')
    print(f'  Normal 组在训练集中始终保留, 只留出故障组\n')

    rows = []
    for hold in holdout_groups:
        te_mask = (groups == hold)
        tr_mask = ~te_mask
        X_tr, y_tr = X[tr_mask], y[tr_mask]
        X_te, y_te = X[te_mask], y[te_mask]
        # 训练集里应该还有该大类的其他尺寸 (2 个), 因此模型见过该大类
        major = LABEL_TO_MAJOR[hold]
        tr_groups_same_major = [g for g in np.unique(groups[tr_mask]) if LABEL_TO_MAJOR[g] == major]
        for name, fac in CLASSIFIERS.items():
            pipe = Pipeline([('sc', StandardScaler()), ('clf', fac())])
            pipe.fit(X_tr, y_tr)
            acc = accuracy_score(y_te, pipe.predict(X_te))
            rows.append({
                'holdout_diameter': hold,
                'major_class': major,
                'train_diameters_same_major': ';'.join(tr_groups_same_major),
                'classifier': name,
                'acc': acc,
            })
        # 打印每折
        cells = '  '.join(f'{c}={[r for r in rows if r["holdout_diameter"]==hold and r["classifier"]==c][0]["acc"]:.3f}'
                          for c in CLASSIFIERS)
        print(f'  hold={hold:8s} (major={major})  same-major trained on {tr_groups_same_major}  |  {cells}')

    df = pd.DataFrame(rows)
    csv = paths.OUT / 'exp09_leave_diameter_out.csv'
    df.to_csv(csv, index=False, encoding='utf-8-sig')
    print(f'\n[Save] {csv.name}')

    # 汇总
    print(f'\n[Summary] mean ± std over {len(holdout_groups)} held-out diameters:')
    for c in CLASSIFIERS:
        vals = df[df['classifier'] == c]['acc'].values
        print(f'  {c:12s}  mean={vals.mean():.4f}  std={vals.std():.4f}  min={vals.min():.4f}  max={vals.max():.4f}')

    print(f'\n[Done] Total time: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
