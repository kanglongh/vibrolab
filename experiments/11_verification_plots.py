"""
Experiment 11 · 诚实性验证图集 (exp07/08/09/10 的可视化).

产出 4 张图:
    fig5_12_task_heatmap.png       exp07 · 12 任务跨负载热图
    fig6_prequential_curve.png     exp08 · Prequential warmup 曲线
    fig7_leave_diameter_out.png    exp09 · 未见损伤尺寸精度
    fig8_snr_robustness.png        exp10 · 加噪声 Bot-40 vs 全 120 对比
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
# 论文级出图样式 (与 04_plots.py 保持一致)
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

C_FULL = '#90A4AE'
C_BOT  = '#1976D2'
C_TOP  = '#D32F2F'
C_GOOD = '#2E7D32'
C_WARN = '#F57C00'

FIG_DIR = paths.OUT / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    fp = FIG_DIR / name
    fig.savefig(fp, facecolor='white')
    plt.close(fig)
    print(f'  Saved: {fp.name}')


# ============================================================
# Fig 5 · exp07 · 12 任务热图 (Bot-40 精度矩阵)
# ============================================================
def fig5_12_task_heatmap():
    print("Fig 5 · exp07 · 12 任务跨负载热图")
    df = pd.read_csv(paths.OUT / 'exp07_all_tasks.csv')

    loads = [0, 1, 2, 3]
    classifiers = ['SVM-RBF', 'LinearSVM', 'LR', 'KNN']

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
    for ax, clf in zip(axes, classifiers):
        sub = df[df['classifier'] == clf]
        mat = np.full((4, 4), np.nan)
        for _, r in sub.iterrows():
            mat[int(r['src_load']), int(r['tgt_load'])] = r['bot40']
        im = ax.imshow(mat, cmap='RdYlGn', vmin=0.85, vmax=1.0, aspect='equal')
        for i in range(4):
            for j in range(4):
                if not np.isnan(mat[i, j]):
                    val = mat[i, j] * 100
                    color = 'white' if val < 92 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                            color=color, fontsize=10, fontweight='bold')
                else:
                    # 对角线: 源域 == 目标域, 属于同工况 (非跨工况任务)
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                facecolor='#DDDDDD', edgecolor='#999999',
                                                hatch='//', linewidth=0.5, zorder=2))
                    ax.text(j, i, '同工况\n(不评估)', ha='center', va='center',
                            color='#555555', fontsize=8, style='italic', zorder=3)
        ax.set_xticks(range(4)); ax.set_xticklabels([f'{l}HP' for l in loads])
        ax.set_yticks(range(4)); ax.set_yticklabels([f'{l}HP' for l in loads])
        ax.set_xlabel('目标域')
        ax.set_ylabel('源域' if clf == 'SVM-RBF' else '')
        ax.set_title(f'{clf}\n12 任务平均 {sub["bot40"].mean()*100:.2f}%  最差 {sub["bot40"].min()*100:.2f}%',
                     fontsize=10)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.14, aspect=50)
    cbar.set_label('Bot-40 跨工况精度 (对角线为同工况, 不计入)')
    fig.suptitle('图 5 · exp07 · 12 任务跨负载 Bot-40 精度矩阵 (CWRU 10 类, 源域→目标域)',
                 fontsize=12, y=1.02)
    fig.text(0.5, -0.05,
             '读图说明: 这不是混淆矩阵. 每行代表源域负载 (训练数据在这个负载下采集), 每列代表目标域负载 (测试数据在这个负载下采集); '
             '格子里的数字 = 在源域训练 Bot-40 特征子集, 在目标域上评估的 10 分类精度. 数字越大 (颜色越绿), 跨工况迁移越成功. '
             '对角线源域=目标域即同工况, 不属于跨工况评估范围, 灰色斜线标出.',
             ha='center', fontsize=8, color='#555', style='italic', wrap=True)
    _save(fig, 'fig5_12_task_heatmap.png')


# ============================================================
# Fig 6 · exp08 · Prequential warmup 曲线
# ============================================================
def fig6_prequential():
    print("Fig 6 · exp08 · Prequential warmup 曲线")
    df = pd.read_csv(paths.OUT / 'exp08_prequential.csv')

    ratios = sorted(df['warmup_ratio'].unique())
    classifiers = ['SVM-RBF', 'LinearSVM', 'LR', 'KNN']
    colors = {'SVM-RBF': '#D32F2F', 'LinearSVM': '#F57C00',
              'LR': '#1976D2', 'KNN': '#2E7D32'}

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for clf in classifiers:
        means, stds = [], []
        for r in ratios:
            sub = df[(df['warmup_ratio'] == r) & (df['classifier'] == clf)]
            means.append(sub['bot40'].mean() * 100)
            stds.append(sub['bot40'].std() * 100)
        means = np.array(means); stds = np.array(stds)
        xs = np.array(ratios) * 100
        ax.plot(xs, means, marker='o', label=clf, color=colors[clf], linewidth=2)
        ax.fill_between(xs, means - stds, means + stds, alpha=0.15, color=colors[clf])

    ax.set_xscale('log')
    ax.set_xticks([5, 10, 20, 50, 100])
    ax.set_xticklabels(['5%', '10%', '20%', '50%', '100%'])
    ax.set_xlabel('用于估计 Cohen\'s d 的目标域样本比例')
    ax.set_ylabel('Bot-40 跨工况精度 (%)')
    ax.set_ylim(75, 102)
    ax.set_title('图 6 · exp08 · Prequential — 流式部署下 Bot-40 精度衰减 (0HP→3HP, 3 seeds)',
                 fontsize=11)
    ax.axhline(99, color='gray', ls=':', alpha=0.5, lw=0.8)
    ax.text(4.7, 99.3, '99% 参考线', fontsize=8, color='gray')
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='gray')

    # 加一条说明: 40 样本就够
    ax.annotate('仅 40 个无标签目标域样本\n(5%)即可稳定挑维度',
                xy=(5, 99), xytext=(11, 84),
                fontsize=9, color=C_BOT,
                arrowprops=dict(arrowstyle='->', color=C_BOT, lw=1.2))
    _save(fig, 'fig6_prequential_curve.png')


# ============================================================
# Fig 7 · exp09 · 未见损伤尺寸精度 (逐折条形图)
# ============================================================
def fig7_leave_diameter_out():
    print("Fig 7 · exp09 · 未见损伤尺寸 (Leave-One-Diameter-Out)")
    df = pd.read_csv(paths.OUT / 'exp09_leave_diameter_out.csv')

    holdouts = sorted(df['holdout_diameter'].unique())
    classifiers = ['SVM-RBF', 'LinearSVM', 'LR', 'KNN']
    colors = {'SVM-RBF': '#D32F2F', 'LinearSVM': '#F57C00',
              'LR': '#1976D2', 'KNN': '#2E7D32'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                    gridspec_kw={'width_ratios': [2.6, 1]})

    # 左: 每折精度条形图 (4 个分类器)
    x = np.arange(len(holdouts))
    w = 0.2
    for i, clf in enumerate(classifiers):
        sub = df[df['classifier'] == clf].set_index('holdout_diameter').loc[holdouts]
        ax1.bar(x + (i - 1.5) * w, sub['acc'].values * 100, w,
                label=clf, color=colors[clf], edgecolor='white')

    ax1.set_xticks(x)
    ax1.set_xticklabels(holdouts, rotation=30, ha='right')
    ax1.set_ylabel('4 大类分类精度 (%)')
    ax1.set_ylim(0, 108)
    ax1.set_title('每个"留出损伤尺寸"下的精度 (训练集包含同大类的另外 2 个尺寸)',
                  fontsize=11)
    ax1.axhline(25, color='gray', ls=':', alpha=0.5, lw=0.8)
    ax1.text(len(holdouts) - 0.5, 26, '4 类随机猜基线 (25%)', fontsize=8, color='gray', ha='right')
    ax1.legend(loc='lower right', frameon=True, facecolor='white',
               edgecolor='gray', ncol=4, fontsize=8)

    # 右: 汇总条形 (mean±std)
    means, stds = [], []
    for clf in classifiers:
        vals = df[df['classifier'] == clf]['acc'].values * 100
        means.append(vals.mean()); stds.append(vals.std())
    y = np.arange(len(classifiers))
    bars = ax2.barh(y, means, xerr=stds,
                    color=[colors[c] for c in classifiers],
                    edgecolor='white', height=0.6, capsize=5)
    for bar, m in zip(bars, means):
        ax2.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                 f'{m:.1f}%', va='center', fontsize=10, fontweight='bold')
    ax2.set_yticks(y); ax2.set_yticklabels(classifiers)
    ax2.set_xlabel('9 折平均精度 (%)')
    ax2.set_xlim(0, 100)
    ax2.set_title('9 折平均 (mean ± std)', fontsize=11)
    ax2.axvline(25, color='gray', ls=':', alpha=0.5, lw=0.8)
    ax2.invert_yaxis()

    fig.suptitle('图 7 · exp09 · Leave-One-Diameter-Out 泛化能力评估',
                 fontsize=12, y=1.00)
    fig.tight_layout()
    _save(fig, 'fig7_leave_diameter_out.png')


# ============================================================
# Fig 8 · exp10 · 加噪声 Bot-40 vs 全 120 精度曲线
# ============================================================
def fig8_snr_robustness():
    print("Fig 8 · exp10 · 加噪声鲁棒性")
    df = pd.read_csv(paths.OUT / 'exp10_noise_robustness.csv')

    # SNR 排序: clean → -5 dB
    snr_order = ['clean', '20', '10', '5', '0', '-5']
    df['snr_str'] = df['snr_db'].astype(str)
    classifiers = ['SVM-RBF', 'LinearSVM', 'LR', 'KNN']
    colors = {'SVM-RBF': '#D32F2F', 'LinearSVM': '#F57C00',
              'LR': '#1976D2', 'KNN': '#2E7D32'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    xs = np.arange(len(snr_order))

    # 左: 全 120 维
    ax = axes[0]
    for clf in classifiers:
        means, stds = [], []
        for s in snr_order:
            sub = df[(df['snr_str'] == s) & (df['classifier'] == clf)]
            means.append(sub['full_120'].mean() * 100)
            stds.append(sub['full_120'].std() * 100)
        means = np.array(means); stds = np.array(stds)
        ax.plot(xs, means, marker='o', label=clf, color=colors[clf], linewidth=2)
        ax.fill_between(xs, means - stds, means + stds, alpha=0.15, color=colors[clf])
    ax.set_xticks(xs); ax.set_xticklabels(snr_order)
    ax.set_xlabel('目标域 SNR (dB)')
    ax.set_ylabel('跨工况精度 (%)')
    ax.set_ylim(0, 105)
    ax.set_title('全 120 维基线 · 噪声下崩塌', fontsize=11)
    ax.axhline(10, color='gray', ls=':', alpha=0.5, lw=0.8)
    ax.text(0.2, 11, '10 类随机猜 (10%)', fontsize=8, color='gray')
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='gray', fontsize=8)

    # 右: Bot-40
    ax = axes[1]
    for clf in classifiers:
        means, stds = [], []
        for s in snr_order:
            sub = df[(df['snr_str'] == s) & (df['classifier'] == clf)]
            means.append(sub['bot40'].mean() * 100)
            stds.append(sub['bot40'].std() * 100)
        means = np.array(means); stds = np.array(stds)
        ax.plot(xs, means, marker='o', label=clf, color=colors[clf], linewidth=2)
        ax.fill_between(xs, means - stds, means + stds, alpha=0.15, color=colors[clf])
    ax.set_xticks(xs); ax.set_xticklabels(snr_order)
    ax.set_xlabel('目标域 SNR (dB)')
    ax.set_title('Bot-40 (本文) · 在 +10 dB 仍 >94%', fontsize=11)
    ax.axhline(10, color='gray', ls=':', alpha=0.5, lw=0.8)
    ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='gray', fontsize=8)

    fig.suptitle('图 8 · exp10 · 加噪声鲁棒性 — 全 120 维在 +10 dB 崩至随机猜, Bot-40 仍 94-99%',
                 fontsize=12, y=1.00)
    fig.tight_layout()
    _save(fig, 'fig8_snr_robustness.png')


def main():
    print('=' * 60)
    print('Experiment 11 · Verification Figures')
    print('=' * 60)
    print(f'Output: {FIG_DIR}\n')
    fig5_12_task_heatmap()
    fig6_prequential()
    fig7_leave_diameter_out()
    fig8_snr_robustness()
    print(f'\n[Done] 4 figures saved to {FIG_DIR}')


if __name__ == '__main__':
    main()
