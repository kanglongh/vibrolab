"""
Experiment 04 · 出图脚本 (跑完 exp02 + exp03 后生成的三张图).

输出到 outputs/figures/:
    fig1_within_vs_cross.png    同工况 vs 跨工况精度对比 (4 分类器)
    fig2_cohens_d.png           Cohen's d 逐维排序 (Top20 高漂 + Bot20 低漂)
    fig3_ablation.png           全 120 / Bot-40 / Random-40 三层对照
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vibrolab import paths

# ============================================================
# 论文级出图样式
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Arial'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'axes.unicode_minus': False,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
})

C_FULL = '#90A4AE'    # 灰: 全量基线
C_BOT  = '#1976D2'    # 蓝: 本文 Bot-40
C_TOP  = '#D32F2F'    # 红: 随机基线 Random-40
C_HIGH = '#D32F2F'    # 高漂
C_LOW  = '#2E7D32'    # 低漂

FIG_DIR = paths.OUT / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    fp = FIG_DIR / name
    fig.savefig(fp, facecolor='white')
    plt.close(fig)
    print(f'  Saved: {fp.name}')


# ============================================================
# Fig 1 · 同工况 vs 跨工况精度对比
# ============================================================
def fig1_within_vs_cross():
    """Fig 1 · 同工况 vs 跨工况精度对比 (含 leave-file-out)"""
    print('Fig 1 · 同工况 vs 跨工况精度')
    exp02 = pd.read_csv(paths.OUT / 'exp02_within_condition.csv')
    exp03 = pd.read_csv(paths.OUT / 'exp03_cross_condition.csv')

    within_a = exp02[exp02.cv_type == 'window-level'].set_index('classifier')['mean_acc']
    within_b_df = exp02[exp02.cv_type.str.contains('leave-file-out', na=False)]
    within_b = within_b_df.set_index('classifier')['mean_acc'] if not within_b_df.empty else None
    cross_full = exp03.set_index('classifier')['full_120']
    cross_bot40 = exp03.set_index('classifier')['bot40']

    classifiers = ['SVM-RBF', 'LinearSVM', 'LR', 'KNN']
    x = np.arange(len(classifiers))
    n_bars = 4 if within_b is not None else 3
    w = 0.85 / n_bars

    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(n_bars) - (n_bars - 1) / 2

    bars_a = ax.bar(x + positions[0] * w, [within_a[c] * 100 for c in classifiers], w,
                    label='同工况 5-fold CV (窗口级)', color='#2E7D32', edgecolor='white')
    if within_b is not None:
        bars_b = ax.bar(x + positions[1] * w, [within_b[c] * 100 for c in classifiers], w,
                        label='同工况 Leave-file-out (严格)', color='#66BB6A', edgecolor='white', hatch='//')
        bars_full = ax.bar(x + positions[2] * w, [cross_full[c] * 100 for c in classifiers], w,
                           label='跨工况 · 全 120 维', color=C_FULL, edgecolor='white')
        bars_bot = ax.bar(x + positions[3] * w, [cross_bot40[c] * 100 for c in classifiers], w,
                          label='跨工况 · Bot-40 (本文)', color=C_BOT, edgecolor='white')
    else:
        bars_full = ax.bar(x + positions[1] * w, [cross_full[c] * 100 for c in classifiers], w,
                           label='跨工况 · 全 120 维', color=C_FULL, edgecolor='white')
        bars_bot = ax.bar(x + positions[2] * w, [cross_bot40[c] * 100 for c in classifiers], w,
                          label='跨工况 · Bot-40 (本文)', color=C_BOT, edgecolor='white')

    bar_groups = [bars_a, bars_full, bars_bot] if within_b is None else [bars_a, bars_b, bars_full, bars_bot]
    for bars in bar_groups:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 1.3, f'{h:.1f}%',
                    ha='center', fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(classifiers)
    ax.set_ylabel('准确率 (%)')
    ax.set_ylim(40, 108)
    ax.set_title('图 1 · CWRU 轴承数据集 (10 类): 同工况 vs 跨工况 (0HP → 3HP) 精度对比')
    ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='gray', fontsize=8)

    # 从 CSV 里动态取 SVM-RBF 的断崖数字, 避免和实际数据脱节
    svm_within = within_a['SVM-RBF'] * 100
    svm_cross = cross_full['SVM-RBF'] * 100
    ax.annotate(f'SVM-RBF 精度断崖\n{svm_within:.1f}% → {svm_cross:.1f}%',
                xy=(0, svm_cross + 2), xytext=(0.5, max(svm_cross - 15, 55)),
                fontsize=9, color=C_TOP, ha='center',
                arrowprops=dict(arrowstyle='->', color=C_TOP, lw=1.2))
    _save(fig, 'fig1_within_vs_cross.png')


# ============================================================
# Fig 2 · Cohen's d 逐维排序 (Top20 高漂 + Bot20 低漂)
# ============================================================
def fig2_cohens_d():
    print('Fig 2 · Cohen\'s d 排序')
    df = pd.read_csv(paths.OUT / 'exp03_cohens_d.csv')
    top20 = df.head(20).copy()
    bot20 = df.tail(20).copy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    ax1, ax2 = axes

    # 左: Top-20 高漂
    y1 = np.arange(len(top20))
    ax1.barh(y1, top20['cohens_d'], color=C_HIGH, edgecolor='white', height=0.7)
    labels1 = [f"dim{r['dim']}·{r['module']}" for _, r in top20.iterrows()]
    ax1.set_yticks(y1); ax1.set_yticklabels(labels1, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Cohen's d")
    ax1.set_title(f"Top-20 高漂移维 (跨域时不稳定)")
    ax1.axvline(0.2, color='gray', ls=':', alpha=0.5, lw=0.8)

    # 右: Bot-20 低漂
    y2 = np.arange(len(bot20))
    ax2.barh(y2, bot20['cohens_d'], color=C_LOW, edgecolor='white', height=0.7)
    labels2 = [f"dim{r['dim']}·{r['module']}" for _, r in bot20.iterrows()]
    ax2.set_yticks(y2); ax2.set_yticklabels(labels2, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Cohen's d")
    ax2.set_title(f"Bot-20 低漂移维 (跨域时稳定, 本文保留)")
    ax2.set_xlim(0, top20['cohens_d'].max() * 1.1)   # 与左图同 x 轴范围, 视觉对比
    ax2.axvline(0.2, color='gray', ls=':', alpha=0.5, lw=0.8)

    fig.suptitle('图 2 · CFD 120 维 Cohen\'s d 排序 (源域 0HP vs 目标域 3HP)',
                 fontsize=12, y=0.995)
    fig.tight_layout()
    _save(fig, 'fig2_cohens_d.png')


# ============================================================
# Fig 3 · 全 120 / Bot-40 / Random-40 三层对照
# ============================================================
def fig3_ablation():
    print('Fig 3 · 三层对照 (全120 / Bot-40 / Random-40)')
    df = pd.read_csv(paths.OUT / 'exp03_cross_condition.csv')

    classifiers = df['classifier'].tolist()
    x = np.arange(len(classifiers))
    w = 0.27

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    b1 = ax.bar(x - w, df['full_120'] * 100, w, label='全 120 维 (基线)', color=C_FULL, edgecolor='white')
    b2 = ax.bar(x, df['bot40'] * 100, w, label='Bot-40 (本文, 低漂保留)', color=C_BOT, edgecolor='white')
    b3 = ax.bar(x + w, df['rand40'] * 100, w, label='Random-40 (随机基线)', color=C_TOP, edgecolor='white')

    for bars in (b1, b2, b3):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 1.2, f'{h:.1f}%',
                    ha='center', fontsize=7.5)

    ax.set_xticks(x); ax.set_xticklabels(classifiers)
    ax.set_ylabel('跨工况精度 (%)')
    ax.set_ylim(50, 108)
    ax.set_title('图 3 · 三层对照: 全 120 vs Bot-40 (本文) vs Random-40 (随机基线)')
    ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='gray')

    # 标注 SVM 上的显著增益
    svm_full = df[df.classifier == 'SVM-RBF']['full_120'].values[0] * 100
    svm_bot = df[df.classifier == 'SVM-RBF']['bot40'].values[0] * 100
    ax.annotate(f'+{svm_bot - svm_full:.1f} pp',
                xy=(0, svm_bot + 0.5), xytext=(0, svm_full - 8),
                ha='center', fontsize=10, color=C_BOT, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_BOT, lw=1.3))
    _save(fig, 'fig3_ablation.png')


def main():
    print('=' * 60)
    print('Experiment 04 · Plotting')
    print('=' * 60)
    print(f'Output: {FIG_DIR}\n')
    fig1_within_vs_cross()
    fig2_cohens_d()
    fig3_ablation()
    print(f'\n[Done] 3 figures saved to {FIG_DIR}')


if __name__ == '__main__':
    main()
