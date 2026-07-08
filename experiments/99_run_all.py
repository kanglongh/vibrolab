"""
Experiment 99 · 一键运行全链路.

按顺序执行:
    02 · 同工况 baseline (双 CV 对照)
    03 · 跨工况诊断: 断崖 + Bot-40 修复
    04 · exp02/03 出图 (fig1-fig3)
    05 · 文献对标 (fig4)
    07 · 12 任务跨负载全评估
    08 · Prequential Cohen's d
    09 · 未见损伤尺寸外推
    10 · 加噪声鲁棒性
    11 · exp07-10 出图 (fig5-fig8)
    12 · 边缘部署延迟与产物规格
    15a · 1D-CNN 效率基线实测
    15b · TL 方法效率谱 (fig9)
    17 · 传感器降级模拟 (IEPE/ADXL1002/ADXL355/MPU6050)
    06 · LLM 诊断解释接口示例 (可选)

跑完后, outputs/ 下会有完整的 CSV / Markdown 报告 / PNG 图集.
LLM 部分放到最后, 因为需要加载模型 (相对耗时且非核心链路).
"""
from __future__ import annotations

import os
import sys
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))


STEPS = [
    ('02_within_condition.py',    'Within-condition baseline (5-fold CV + mixed-load LFO)'),
    ('03_cross_condition.py',     'Cross-load diagnosis (baseline cliff + Bot-40 repair)'),
    ('04_plots.py',               'Plots for exp02 & exp03 (fig1-fig3)'),
    ('05_literature_benchmark.py','Literature benchmark comparison (fig4)'),
    ('07_all_12_tasks.py',        'Full 12-task cross-load benchmark'),
    ('08_prequential.py',         'Prequential Cohen\'s d (streaming deployment)'),
    ('09_leave_diameter_out.py',  'Leave-one-diameter-out (unseen damage size)'),
    ('10_noise_robustness.py',    'Noise robustness (AWGN sweep)'),
    ('11_verification_plots.py',  'Verification plots (fig5-fig8)'),
    ('12_edge_latency.py',        'Edge deployment latency & artifact footprint'),
    ('15a_efficiency_baseline.py','1D-CNN efficiency baseline (PyTorch)'),
    ('15b_tl_efficiency.py',      'TL method efficiency spectrum (fig9)'),
    ('17_sensor_degradation.py',  'Sensor degradation simulation (MEMS tiers)'),
    ('06_llm_diagnosis.py',       'LLM diagnosis explanation interface (optional)'),
]


def run_step(script_name: str, description: str):
    print('\n' + '=' * 70)
    print(f'STEP · {script_name}')
    print(f'      {description}')
    print('=' * 70)
    script = os.path.join(HERE, script_name)
    if not os.path.exists(script):
        print(f'[SKIP] Not found: {script}')
        return False
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script],
        cwd=os.path.dirname(HERE),
    )
    dt = time.time() - t0
    ok = result.returncode == 0
    print(f'\n[{"OK" if ok else "FAIL"}] {script_name}  time={dt:.1f}s')
    return ok


def main():
    print('vibrolab · Full pipeline')
    print('=' * 70)
    print(f'This will run {len(STEPS)} experiments end-to-end.')
    print('Estimated time: ~3-5 minutes for exp02-12 (CPU), + ~1 min for exp15a (1D-CNN training), + ~1 min for exp06 (LLM).')
    print('Outputs: outputs/ (CSV + Markdown reports + PNG figures)')

    t_all = time.time()
    n_ok = 0
    for script, desc in STEPS:
        if run_step(script, desc):
            n_ok += 1
    total = time.time() - t_all
    print('\n' + '=' * 70)
    print(f'Summary: {n_ok}/{len(STEPS)} steps OK, total {total:.1f}s')
    print('Check outputs/ for CSVs and exp06_diagnosis_report.md')
    print('=' * 70)


if __name__ == '__main__':
    main()
