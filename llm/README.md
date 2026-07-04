# LLM 模块说明

本项目的 LLM 诊断解释接口支持**双后端**——本地小模型（离线）与 OpenAI-Compatible 云端 API（联网）。两个后端共享同一份 prompt 模板 (`prompts/diagnosis_zh.txt`) 和同一个组装函数，切换后端只改环境变量、不改代码。

## 两种后端

| 后端 | 触发方式 | 用途 |
|---|---|---|
| **`local`** (默认) | 未设置 `LLM_BACKEND` 或设为 `local` | 离线可用、内网环境、无 API 费用；生成质量取决于所选本地模型 |
| **`api`** | `LLM_BACKEND=api` | 联网可用、生成质量高、按 token 计费；OpenAI-Compatible 协议 |

## 云端 API 后端配置

任何符合 **OpenAI Chat Completions 协议**的服务商都可对接（阿里百炼 / DeepSeek / 智谱 / Moonshot / OpenAI 官方 / vLLM 自部署等）。

复制项目根目录的 [`.env.example`](../.env.example) 为 `.env`, 填入四件套并加载:

```bash
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-max
LLM_PROVIDER=阿里云百炼          # 可选, 仅用于报告落款
LLM_PRICE_IN=12                  # 可选, 输入 ¥/百万 token, 用于成本核算
LLM_PRICE_OUT=36                 # 可选, 输出 ¥/百万 token
```

跑一次示例:

```bash
python experiments/06b_llm_diagnosis_api.py
```

产物 [`outputs/exp06b_diagnosis_report_api.md`](../outputs/exp06b_diagnosis_report_api.md) 包含 4 条示例维修建议 + token 消耗表 + 规模成本估算.

## 本地小模型后端配置

`llm/backends.py` 里的 `LocalQwenBackend` 通过 `transformers` 加载, 支持 Qwen 系列 (通过 chat template) 及任何遵循相同接口的开源模型.

### 本地模型下载

**方法 1 · ModelScope** (国内推荐):

```bash
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('qwen/Qwen2.5-0.5B-Instruct', local_dir='./llm/models/Qwen2.5-0.5B-Instruct')
"
```

**方法 2 · HuggingFace**:

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir llm/models/Qwen2.5-0.5B-Instruct
```

**方法 3 · 已有模型**——通过环境变量指定绝对路径:

```bash
export QWEN_MODEL_PATH=/absolute/path/to/Qwen2.5-0.5B-Instruct
python experiments/06_llm_diagnosis.py
```

模型文件夹需包含 `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json` (含 `chat_template`), `generation_config.json`.

## Prompt 定制

[`prompts/diagnosis_zh.txt`](prompts/diagnosis_zh.txt) 是当前使用的中文 prompt 模板. 三段式结构:

1. **紧急程度** (紧急 / 需在下次停机时处理 / 定期观察)
2. **推荐动作** (具体检查/维修操作, 严格限定在滚动轴承与旋转机械范畴)
3. **风险提示** (若不处理的后果)

可根据具体设备类型或组织内规范修改, exp06 和 exp06b 会同时使用修改后的模板.

## 关于本地小模型的能力边界

本仓库的示例后端演示用了 Qwen2.5-0.5B-Instruct (~988 MB, CPU 可跑). 这个模型:

- ✅ 中文流畅, 可按 prompt 生成结构化短建议 (3~5 句)
- ✅ 具备一定工业术语理解 (轴承、内圈、滚动体等)
- ⚠️ 5 亿参数, **不擅长复杂推理**——prompt 需保持模板化, 避免开放式提问
- ⚠️ 可能出现 `max_new_tokens` 边界处的句子截断

**生产部署建议**接入更大的本地模型 (如 Qwen2.5-7B-Instruct) 或云端 API. 本仓库 [`outputs/exp06_diagnosis_report.md`](../outputs/exp06_diagnosis_report.md) 与 [`outputs/exp06b_diagnosis_report_api.md`](../outputs/exp06b_diagnosis_report_api.md) 提供了两个后端的质量对比示例, 后者显著更贴近现场维修工程师的语言.

## LLM 层的定位

CFD 120 维本身具备物理可解释性 (时域统计 / 频域分布 / 分频带能量 / 主峰 / 倒谱), 生产环境下 **LLM 层为可选组件, 不影响诊断精度**. 如果目标场景是接入现有 SCADA / MES 系统, 分类器输出的结构化标签 (故障类型 + 置信度 + 关键特征模块) 可直接展示, 不需要经过 LLM.
