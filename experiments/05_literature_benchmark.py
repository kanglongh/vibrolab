"""
Experiment 05 · 与文献 CWRU 跨负载基准对比

数据来源:
    [1] DCDAN — Zhang et al., 2020, Shock and Vibration (DOI: 10.1155/2020/8850976)
    [2] CDHM — Zhao et al., 2023, arXiv:2308.03027
    [3] AMDA — Ragab et al., 2021, IEEE Trans. Instrumentation and Measurement
    [4] MaxDD — 2023, EURASIP J. Adv. Signal Processing (DOI: 10.1186/s13634-023-01107-x)

比较任务: CWRU 10 类跨负载 0HP → 3HP

诚实性说明:
    - 所有文献数字来自作者调研报告 (见 CWRU_跨负载文献调研报告.md)
    - 我们的方法数字来自 exp03_cross_condition.py
    - 只对比同任务同类别数 (10 类, 0HP→3HP), 保证可比性
    - 严格来说不同论文的实验方案可能略有差异 (窗口长度、采样率、数据切分),
      但差异一般在 1-3 pp 之内, 用于宏观趋势对比是合理的

输出:
    outputs/exp05_benchmark_table.csv     完整对比表
    outputs/figures/fig4_literature_benchmark.png    对比图
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vibrolab import paths

# ============================================================
# 图表样式
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


# ============================================================
# 文献对比数据 (CWRU 10 类, 0HP → 3HP)
# ============================================================
LITERATURE_DATA = [
    # (方法名, 类别, 精度, 来源, 备注)
    ('CNN (无 DA)',        '深度基线 (无迁移)',     80.50, 'DCDAN [1]',   '10 类'),
    ('CORAL',              '浅层 DA (对齐)',        49.21, 'AMDA [3]',    '手工特征 + CORAL'),
    ('TCA',                '浅层 DA (对齐)',        86.56, 'DCDAN [1]',   '手工特征 + TCA'),
    ('JDA',                '浅层 DA (对齐)',        82.23, 'AMDA [3]',    '手工特征 + JDA'),
    ('DDC (MMD)',          '深度 DA',               92.33, 'DCDAN [1]',   'CNN + MMD 对齐'),
    ('Deep CORAL',         '深度 DA',               97.52, 'AMDA [3]',    'CNN + CORAL loss'),
    ('DANN',               '深度 DA (对抗)',        97.82, 'CDHM [2]',    '梯度反转层'),
    ('DCTLN',              '深度 DA (SOTA)',        98.92, 'CDHM [2]',    'Guo et al., 2019'),
    ('DCDAN',              '深度 DA (SOTA)',        98.87, 'DCDAN [1]',   'MMD + 域判别器'),
    ('AMDA',               '深度 DA (SOTA)',        99.52, 'AMDA [3]',    '多目标域对抗'),
    ('CDHM',               '深度 DA (SOTA)',        99.69, 'CDHM [2]',    '因果解耦'),
]


def load_our_results():
    """从 exp03_cross_condition.csv 读我们的结果."""
    csv = paths.OUT / 'exp03_cross_condition.csv'
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    return df


def build_benchmark_table():
    """构建完整对比表, 保存到 CSV."""
    df_lit = pd.DataFrame(LITERATURE_DATA, columns=['方法', '类别', '精度%', '来源', '备注'])

    df_our = load_our_results()
    if df_our is not None:
        for _, r in df_our.iterrows():
            df_lit = pd.concat([df_lit, pd.DataFrame([{
                '方法': f'Bot-40 + {r["classifier"]} (本文)',
                '类别': '特征选择 (零训练, 零标注)',
                '精度%': round(r['bot40'] * 100, 2),
                '来源': '本文 exp03',
                '备注': '10 类, 0HP→3HP',
            }])], ignore_index=True)
        # 同时加入全量 SVM baseline 作为参照
        for _, r in df_our.iterrows():
            if r['classifier'] == 'SVM-RBF':
                df_lit = pd.concat([df_lit, pd.DataFrame([{
                    '方法': f'全 120 维 + {r["classifier"]} (本文基线)',
                    '类别': '传统 SVM (无 DA)',
                    '精度%': round(r['full_120'] * 100, 2),
                    '来源': '本文 exp03',
                    '备注': '10 类, 0HP→3HP',
                }])], ignore_index=True)

    df_lit = df_lit.sort_values('精度%', ascending=False).reset_index(drop=True)
    csv = paths.OUT / 'exp05_benchmark_table.csv'
    df_lit.to_csv(csv, index=False, encoding='utf-8-sig')
    print(f'[Save] {csv.name}')
    return df_lit


def plot_benchmark(df):
    """生成对比图: 横向条形图, 按精度降序."""
    fig, ax = plt.subplots(figsize=(12, 7))

    is_ours = df['来源'] == '本文 exp03'
    is_baseline = df['方法'].str.contains('基线') | df['方法'].str.contains('CNN \\(无 DA\\)') | df['方法'].str.contains('全 120')

    colors = []
    for _, row in df.iterrows():
        if row['来源'] == '本文 exp03':
            if '基线' in row['方法']:
                colors.append('#D32F2F')
            else:
                colors.append('#1976D2')
        elif '基线' in row['类别'] or 'CNN' in row['方法'] or '无 DA' in row['类别']:
            colors.append('#90A4AE')
        elif '浅层' in row['类别']:
            colors.append('#FFA726')
        elif 'SOTA' in row['类别']:
            colors.append('#2E7D32')
        elif '深度' in row['类别']:
            colors.append('#66BB6A')
        else:
            colors.append('#607D8B')

    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df['精度%'], color=colors, edgecolor='white', height=0.7)

    for i, (bar, acc, method) in enumerate(zip(bars, df['精度%'], df['方法'])):
        if '本文' in method:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{acc:.2f}%', va='center', fontsize=9, fontweight='bold', color='#1976D2')
        else:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{acc:.2f}%', va='center', fontsize=8, color='#333')

    ax.set_yticks(y_pos)
    labels = []
    for _, row in df.iterrows():
        m = row['方法']
        src = row['来源']
        if src == '本文 exp03':
            labels.append(m)
        else:
            labels.append(f'{m}  [{src}]')
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()

    ax.set_xlabel('跨负载准确率 (%)', fontsize=11)
    ax.set_xlim(0, 105)
    ax.set_title('图 4 · CWRU 跨负载诊断 (0HP → 3HP, 10 类) 学界对比\n'
                 '本文 Bot-40 特征选择 vs 主流域适应方法',
                 fontsize=12, pad=15)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(color='#1976D2', label='本文 Bot-40 (零训练零标注)'),
        Patch(color='#D32F2F', label='本文基线 (无迁移, 无特征选择)'),
        Patch(color='#2E7D32', label='文献 SOTA (深度 DA, 需 GPU 训练)'),
        Patch(color='#66BB6A', label='文献 · 深度 DA (需 GPU 训练)'),
        Patch(color='#FFA726', label='文献 · 浅层 DA (无需 GPU)'),
        Patch(color='#90A4AE', label='文献 · 无迁移基线'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8, frameon=True)

    ax.axvline(90, color='gray', ls='--', alpha=0.3, lw=0.8)
    ax.axvline(95, color='gray', ls='--', alpha=0.3, lw=0.8)
    ax.axvline(99, color='gray', ls='--', alpha=0.3, lw=0.8)
    ax.text(90.2, len(df) - 0.5, '90%', color='gray', fontsize=7, alpha=0.6)
    ax.text(95.2, len(df) - 0.5, '95%', color='gray', fontsize=7, alpha=0.6)
    ax.text(99.2, len(df) - 0.5, '99%', color='gray', fontsize=7, alpha=0.6)

    fig.text(0.02, 0.02,
             '数据来源: [1] Zhang 2020, DCDAN; [2] Zhao 2023, CDHM; [3] Ragab 2021, AMDA; 本文 exp03\n'
             '注: 不同论文的实验设置可能存在细微差异 (窗口长度、切分等), 精度差异在 1-3 pp 之内正常.',
             fontsize=7, color='gray')

    fig_dir = paths.OUT / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_path = fig_dir / 'fig4_literature_benchmark.png'
    fig.savefig(save_path, facecolor='white')
    plt.close(fig)
    print(f'[Save] {save_path.name}')


def main():
    print('=' * 60)
    print('Experiment 05 · CWRU Literature Benchmark')
    print('=' * 60)
    df = build_benchmark_table()
    print(f'\n[Table] {len(df)} methods compared:')
    print(df.to_string(index=False))
    print()
    plot_benchmark(df)
    print('\n[Done]')


if __name__ == '__main__':
    main()
