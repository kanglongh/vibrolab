"""
Experiment 06b · LLM 诊断解释接口 (云端 API 版).

与 exp06 (本地小模型) 同 prompt / 同接口, 后端换成 OpenAI-Compatible 云端服务
(阿里云百炼 / DeepSeek / 智谱 / Moonshot / OpenAI 官方 / vLLM 自部署 均可).

前置: 复制 .env.example 为 .env, 填入 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL /
      LLM_PRICE_IN / LLM_PRICE_OUT, 然后加载到当前 shell (见 .env.example 顶部).

输出:
    outputs/exp06b_diagnosis_report_api.md    云端生成报告
    outputs/exp06b_api_usage.csv              每次调用的 token 与成本明细
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Windows cmd 默认 GBK, 打印 ¥ / 中文标点等会 UnicodeEncodeError; 统一切 UTF-8.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import pandas as pd

from vibrolab import paths
from llm.backends import OpenAICompatibleBackend

# 复用 exp06 里的中文标签映射 / prompt 组装, 避免重复维护.
# 文件名以数字开头, 用 importlib.util 从路径 load.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'exp06_llm_diagnosis',
    os.path.join(os.path.dirname(__file__), '06_llm_diagnosis.py'),
)
_mod06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod06)
FAULT_ZH = _mod06.FAULT_ZH
build_prompt = _mod06.build_prompt
format_key_features = _mod06.format_key_features


def main():
    print('=' * 70)
    print('Experiment 06b · LLM Diagnosis (OpenAI-Compatible API)')
    print('=' * 70)

    # 环境变量三件套检查
    missing = [k for k in ('LLM_API_KEY', 'LLM_BASE_URL', 'LLM_MODEL')
               if not os.environ.get(k)]
    if missing:
        print(f'[ABORT] Missing env vars: {missing}. See .env.example.')
        sys.exit(1)

    model_name = os.environ['LLM_MODEL']
    provider = os.environ.get('LLM_PROVIDER', '云端 LLM API')   # 可选的服务商展示名

    key_features = format_key_features(paths.OUT / 'exp03_cohens_d.csv', n_top=3)
    print(f'[Info] Key stable features from exp03: {key_features}')
    print(f'[Info] Model = {model_name}, provider = {provider}')

    llm = OpenAICompatibleBackend()

    cases = [
        ('N',       0.99, '正常, 定期巡检'),
        ('B014',    0.94, '中等尺寸滚动体故障'),
        ('IR014',   0.96, '中等尺寸内圈故障, 需及时处理'),
        ('OR014@6', 0.97, '中等尺寸外圈故障, 严重程度高'),
    ]

    report_lines = [
        f'# LLM 诊断解释接口 · 云端 API 输出示例 ({model_name})', '',
        f'> 本文档展示 pipeline 末端 LLM 接口切换到**云端 API 后端**的生成结果,',
        f'> 与本地小模型输出 ([exp06 报告](exp06_diagnosis_report.md)) 使用**完全相同的 prompt 与接口调用**,',
        f'> 仅后端不同. 用于说明:',
        f'>',
        f'> 1. 接口层可插拔 — 相同上游诊断结果 + 相同 prompt, 后端可任意切换 (OpenAI-Compatible 协议, 支持阿里百炼/DeepSeek/智谱/Moonshot/OpenAI 官方/vLLM 自部署等)',
        f'> 2. 生成质量与模型规模强相关 — 旗舰级模型的建议在专业性和结构完整度上有显著提升',
        f'> 3. 云端 API 单条调用成本可精确核算 (见文末 Token 与成本明细)',
        f'',
        f'由 vibrolab 基于 CWRU 轴承数据集 (10 类) 生成, 后端: **{provider} · {model_name}**.',
        f'', '---', '',
    ]

    usage_rows = []
    t_all = time.time()
    for lbl, conf, remark in cases:
        prompt = build_prompt(lbl, conf, key_features)
        print(f'\n[{lbl}] Generating "{FAULT_ZH[lbl]}"...')
        t0 = time.time()
        advice = llm.generate(prompt, max_new_tokens=400, temperature=0.2)
        dt = time.time() - t0
        u = llm.last_usage or {}
        print(f'  in={u.get("input_tokens", 0)}tok  out={u.get("output_tokens", 0)}tok  '
              f'cost=¥{u.get("cost_rmb", 0):.5f}  latency={dt:.2f}s')
        try:
            print(advice.encode('gbk', errors='replace').decode('gbk')[:200] + '...')
        except Exception:
            pass
        usage_rows.append({
            'label': lbl,
            'fault_zh': FAULT_ZH[lbl],
            'input_tokens':  u.get('input_tokens', 0),
            'output_tokens': u.get('output_tokens', 0),
            'total_tokens':  u.get('total_tokens', 0),
            'cost_rmb':      round(u.get('cost_rmb', 0.0), 6),
            'latency_s':     round(dt, 2),
        })
        report_lines.extend([
            f'## 故障类型: {FAULT_ZH[lbl]} (置信度 {conf:.1%})',
            '',
            f'**关键相关特征**: {key_features}',
            '',
            f'**LLM 维修建议** (由 {model_name} 生成):',
            '',
            advice,
            '',
            f'*Token: in={u.get("input_tokens", 0)}, out={u.get("output_tokens", 0)}, '
            f'成本: ¥{u.get("cost_rmb", 0):.5f}, 延迟: {dt:.2f}s*',
            '',
            '---',
            '',
        ])

    # 成本汇总卡 (面向读者证明可核算能力)
    total_dt = time.time() - t_all
    df = pd.DataFrame(usage_rows)
    csv = paths.OUT / 'exp06b_api_usage.csv'
    df.to_csv(csv, index=False, encoding='utf-8-sig')
    print(f'\n[Save] {csv.name}')

    report_lines.extend([
        '## 附录 · Token 消耗与成本核算',
        '',
        f'| 用例 | 输入 tok | 输出 tok | 单次成本 (¥) | 延迟 (s) |',
        f'|---|---:|---:|---:|---:|',
    ])
    for r in usage_rows:
        report_lines.append(
            f'| {r["fault_zh"]} | {r["input_tokens"]} | {r["output_tokens"]} '
            f'| {r["cost_rmb"]:.5f} | {r["latency_s"]:.2f} |'
        )
    report_lines.extend([
        f'| **合计 ({len(usage_rows)} 条)** | **{df["input_tokens"].sum()}** '
        f'| **{df["output_tokens"].sum()}** | **{df["cost_rmb"].sum():.5f}** '
        f'| {total_dt:.2f} (总耗时) |',
        '',
        f'> **单位成本**: 输入 ¥{llm.price_in}/百万 tok · 输出 ¥{llm.price_out}/百万 tok  ',
        f'> **规模估算**: 按上表均值, 每日 1000 台设备 × 1 次诊断 ≈ '
        f'¥{df["cost_rmb"].mean() * 1000:.2f}/天, 全年约 ¥{df["cost_rmb"].mean() * 1000 * 365:.0f}',
        '',
    ])

    report_path = paths.OUT / 'exp06b_diagnosis_report_api.md'
    report_path.write_text('\n'.join(report_lines), encoding='utf-8')
    print(f'\n[Save] {report_path.name}')
    print(f'\n[Summary] Total: {llm.total_tokens} tokens, ¥{llm.total_cost_rmb:.5f}, {total_dt:.1f}s')


if __name__ == '__main__':
    main()
