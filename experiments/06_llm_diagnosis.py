"""
Experiment 06 · LLM 诊断解释接口示例.

演示 pipeline 末端"结构化诊断结果 → 自然语言维修建议"的接口调用方式.
后端可插拔 (本地小模型 / API), 见 llm/backends.py.

生成质量取决于所选模型; 生产环境建议使用 7B+ 本地模型或云端 API.

输出: outputs/exp06_diagnosis_report.md  接口调用示例
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

from vibrolab import paths
from llm.backends import get_backend

# 10 类标签 -> 中文故障描述
FAULT_ZH = {
    'N':        '正常无故障',
    'B007':     '滚动体故障 (损伤直径 0.007")',
    'B014':     '滚动体故障 (损伤直径 0.014")',
    'B021':     '滚动体故障 (损伤直径 0.021")',
    'IR007':    '内圈故障 (损伤直径 0.007")',
    'IR014':    '内圈故障 (损伤直径 0.014")',
    'IR021':    '内圈故障 (损伤直径 0.021")',
    'OR007@6':  '外圈故障 (损伤直径 0.007", @6 点钟位置)',
    'OR014@6':  '外圈故障 (损伤直径 0.014", @6 点钟位置)',
    'OR021@6':  '外圈故障 (损伤直径 0.021", @6 点钟位置)',
}


def load_prompt_template():
    p = paths.ROOT / 'llm' / 'prompts' / 'diagnosis_zh.txt'
    return p.read_text(encoding='utf-8')


def format_key_features(csv_path, n_top: int = 3) -> str:
    """把稳定特征翻译成给 LLM/维修工程师看得懂的物理说法.

    直接把 dim62(band) 之类的内部索引透传给 0.5B 会让它硬塞进输出.
    这里做一层"内部索引 → 物理身份"的映射, 只给 LLM 物理直觉描述.
    """
    if not csv_path.exists():
        return "中频段能量占比、倒谱系数等调制型特征"
    df = pd.read_csv(csv_path)
    top_modules = df.sort_values('cohens_d', ascending=True).head(n_top)['module'].tolist()
    mod_desc = {
        'time':     '时域波形统计量',
        'freq':     '频域全局统计量',
        'band':     '分频带能量分布',
        'peaks':    '主导频率峰',
        'cepstral': '倒谱调制系数',
    }
    # 保序去重
    seen, uniq = set(), []
    for m in top_modules:
        if m not in seen:
            seen.add(m); uniq.append(m)
    return '、'.join(mod_desc.get(m, m) for m in uniq)


def build_prompt(fault_label: str, confidence: float, key_features: str) -> str:
    tmpl = load_prompt_template()
    return tmpl.format(
        fault_type=FAULT_ZH.get(fault_label, fault_label),
        confidence=confidence,
        key_features=key_features,
    )


def main():
    print('=' * 70)
    print('Experiment 06 · LLM Diagnosis Explainer')
    print('=' * 70)

    key_features = format_key_features(paths.OUT / 'exp03_cohens_d.csv', n_top=3)
    print(f'[Info] Key stable features from exp03: {key_features}')

    # 优雅跳过: 本地模型未配置时给出说明并退出 0, 避免 99_run_all.py 中断
    if not os.environ.get('QWEN_MODEL_PATH'):
        print()
        print('[SKIP] QWEN_MODEL_PATH not set — skipping local-model demo.')
        print('       To run this step, either:')
        print('         (a) Download a local Qwen (or compatible) model and set QWEN_MODEL_PATH; see llm/README.md')
        print('         (b) Use the API backend instead: `python experiments/06b_llm_diagnosis_api.py`')
        print('       Both backends share the same prompt template; API backend is optional but generally recommended for quality.')
        return

    llm = get_backend()

    # 演示: 覆盖 4 大类故障 (N + B + IR + OR), 每类取一个中等损伤尺寸做示例
    cases = [
        ('N',       0.99, '正常, 定期巡检'),
        ('B014',    0.94, '中等尺寸滚动体故障'),
        ('IR014',   0.96, '中等尺寸内圈故障, 需及时处理'),
        ('OR014@6', 0.97, '中等尺寸外圈故障, 严重程度高'),
    ]

    report_lines = [
        '# LLM 诊断解释接口 · 示例输出', '',
        '> ⚠️ **免责声明**：本文档为 vibrolab pipeline 末端 LLM 接口的**调用示例**，',
        '> 用于展示"结构化诊断结果 → 自然语言维修建议"的数据流。',
        '> 下文中的建议由所选 LLM 后端自动生成，**不构成任何实际维修指导**，',
        '> 生成质量与所选模型规模强相关；生产环境请接入 7B+ 模型或云端 API。',
        '',
        '由 vibrolab 基于 CWRU 轴承数据集 (10 类) 生成。',
        '', '---', '']

    for lbl, conf, remark in cases:
        prompt = build_prompt(lbl, conf, key_features)
        print(f'\n[{lbl}] Generating advice for "{FAULT_ZH[lbl]}"...')
        advice = llm.generate(prompt, max_new_tokens=200, temperature=0.2)
        # 防终端 GBK 编码报错 (Windows cmd 无法打印非法字符)
        safe = advice.encode('gbk', errors='replace').decode('gbk')
        print(f'--- LLM 输出 ---\n{safe}\n')
        report_lines.extend([
            f'## 故障类型: {FAULT_ZH[lbl]} (置信度 {conf:.1%})',
            f'',
            f'**关键相关特征**: {key_features}',
            f'',
            f'**LLM 维修建议**:',
            f'',
            advice,
            f'',
            f'---',
            f'',
        ])

    report_path = paths.OUT / 'exp06_diagnosis_report.md'
    report_path.write_text('\n'.join(report_lines), encoding='utf-8')
    print(f'\n[Done] Report saved to {report_path}')


if __name__ == '__main__':
    main()
