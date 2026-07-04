# vibrolab

> **工业振动信号跨工况故障诊断全链路工具库**
> CWRU 轴承 · 10 类 · 0HP → 3HP 跨负载 · 无监督特征预对齐

---

## 30 秒定位

- **跨工况精度**: 全 120 维基线 78.93% → Bot-40 稳定子集 **98.96%** (SVM-RBF, 0HP→3HP)
- **12 任务矩阵**: 4 路分类器平均 **98.6% ~ 99.4%**, 最差单点 89% (LinearSVM · 0HP→3HP)
- **流式部署**: 只需 **5% 目标域预热样本** (约 38 个窗口, 相当于新设备运行 6 到 7 秒)
- **边缘规格**: 单窗口推理 **<1 ms** (LR/LinearSVM), 部署产物 **4.2 KB**, 无 GPU 依赖
- **一键复现**: `python experiments/99_run_all.py`, CPU 约 2 分钟

---

## 一分钟看结果

**CWRU 10 类轴承数据集, 0 HP → 3 HP 跨负载测试** (严格无重叠切分, 训练/测试不共享 .mat 文件):

| | SVM-RBF | LinearSVM | LR | KNN |
|---|---:|---:|---:|---:|
| 同工况 5-fold CV | 100.00% | 100.00% | 100.00% | 100.00% |
| 同工况留一文件 3-fold (严格) | 99.77% | 99.93% | 99.70% | 99.70% |
| 全 120 维直接跨负载 (无对齐) | 78.93% | 84.66% | 84.66% | 85.96% |
| **Bot-40 低漂稳定子集** | **98.96%** | 89.34% | **98.83%** | **99.35%** |

*完整方法学讨论与 12 任务全负载矩阵见 [`outputs/VERIFICATION.md`](outputs/VERIFICATION.md).*

### 最有说服力的一张对比 · 加噪声鲁棒性

![加噪声鲁棒性对比](outputs/figures/fig8_snr_robustness.png)

*全 120 维特征在信噪比 +10 dB 时直接崩到 7.7% (10 类均匀猜测下限), Bot-40 稳定子集在同一噪声档位保持 94 到 99%. 详见 [`exp10_noise_robustness.csv`](outputs/exp10_noise_robustness.csv).*

---

## 关于本项目

我是燕山大学机械工程学院 2027 届硕士研究生, 研究方向是**故障诊断**, 具体做**液压柱塞泵**的故障诊断——这是我独立完成的毕业课题, 从选题、方法到实现都没有导师或师兄给现成的思路. 那边的工作走的是**物理特征提取 + 传统机器学习**这一路: 不追求 SOTA, 追求在小样本、跨工况、可解释的前提下把工程指标做扎实.

做完柱塞泵的主线之后, 我一直好奇一个问题: **这套方法到底是我针对柱塞泵调出来的偶然, 还是它反映了旋转/流体机械故障诊断的某种通用范式?** 光在同一份数据上跑证明不了这个问题——那是**内部一致性**, 不是**外部有效性**. 所以我拿了轴承故障诊断领域的标准基准 (CWRU), 把整套流水线原封不动地搬过去, 看能不能拿到同一量级的精度.

**这个仓库就是这次迁移验证的完整记录**——数据、代码、CSV、图表、部署规格、LLM 接口示例全部公开可复现. 简历上如果只写一行"跨工况诊断 X%", 读者会怀疑这个数字是不是造出来的; 但把整套流程、每一个数字、每一个阴性结果 (见 [exp09](outputs/VERIFICATION.md)) 都放到 GitHub 上, 至少证明**我这套方法不是柱塞泵专属的偶然**.

---

## 快速开始

```bash
pip install -r requirements.txt
```

下载 CWRU 数据集到 `../Data/` 目录下 (`Normal` 和 `12k_DE` 两个子文件夹), 然后一键跑完所有实验:

```bash
python experiments/99_run_all.py
```

产物落在 [`outputs/`](outputs/) 文件夹下:
- 10 个实验的 CSV 数据表
- 8 张结果图 (`outputs/figures/`)
- 完整验证报告 [`VERIFICATION.md`](outputs/VERIFICATION.md)
- 部署规格卡 [`exp12_deploy_spec.txt`](outputs/exp12_deploy_spec.txt)
- LLM 诊断接口示例产物 [`exp06_diagnosis_report.md`](outputs/exp06_diagnosis_report.md) 和 [`exp06b_diagnosis_report_api.md`](outputs/exp06b_diagnosis_report_api.md)

---

## 全链路架构

```
原始振动信号 (.mat, 12 kHz)
    ↓
120 维物理可解释特征 (CFD)
    ├── 时域 12 维    (RMS / 峭度 / 波形因子 / ...)
    ├── 频域 15 维    (质心 / 带宽 / 谱斜率 / ...)
    ├── 分频带 64 维  (32 带能量比 + 32 带对数能量)
    ├── 主峰 12 维    (Top-6 主导频率与幅值)
    └── 倒谱 17 维    (调制模式捕捉)
    ↓
无监督特征预对齐 (Cohen's d 逐维分布漂移排序, 取最稳定 40 维 = Bot-40)
    ↓
sklearn 分类器 (SVM-RBF / LinearSVM / LR / KNN, 四路平行对照)
    ↓
结构化诊断结果 (故障类型 + 置信度 + 关键特征模块)
    ↓
[可选] LLM 后端 (本地小模型 / OpenAI-Compatible 云端 API 双通道)
    ↓
自然语言维修建议
```

---

## 扩展验证实验

完整讨论见 [`outputs/VERIFICATION.md`](outputs/VERIFICATION.md), 每一个数字都可追溯到对应 CSV.

### 12 任务全负载矩阵 (exp07)

![12 任务热图](outputs/figures/fig5_12_task_heatmap.png)

4 路分类器在全部 12 个 (源, 目标) 组合上的 Bot-40 精度. 平均 98.6% 到 99.4%, 最差单点约 89%.

### 流式部署 Prequential 验证 (exp08)

![Prequential 衰减](outputs/figures/fig6_prequential_curve.png)

预热样本比例扫: 5% / 10% / 20% / 50% / 100%. 结论: 5% 已足够, SVM-RBF/LR/KNN 三路的精度衰减在 <0.5 pp 内.

### 未见损伤尺寸外推 · 阴性结果 (exp09)

![未见损伤尺寸结果](outputs/figures/fig7_leave_diameter_out.png)

留一损伤尺寸交叉验证. 9 组均值精度 37.8% 到 63.8%, 内圈中大尺寸 (IR014, IR021) 直接崩到 0%. **这是 CWRU 数据集本身的结构性天花板**——每类只有 3 种损伤直径, 中间尺寸的连续外推无法完成. 主动列出这个阴性结果, 是为了让读者对本仓库其他实验里的 "99%" 有正确的心理锚点.

| 实验 | 核心结论 |
|---|---|
| [exp02](experiments/02_within_condition.py) | 同工况严格协议 (留一文件) 99.7%, 说明 CFD 表达力天花板高 |
| [exp03](experiments/03_cross_condition.py) | 0HP→3HP 全 120 维 78.93% → Bot-40 98.96%, +20 pp 提升 |
| [exp05](experiments/05_literature_benchmark.py) | 与 DCDAN / AMDA / CDHM 等深度 DA 方法同量级, 零训练 |
| [exp07](experiments/07_all_12_tasks.py) | 12 任务矩阵均值 98.6% 到 99.4%, 非 cherry-picking |
| [exp08](experiments/08_prequential.py) | 5% 目标域预热已足够, 流式部署可行 |
| [exp09](experiments/09_leave_diameter_out.py) | 未见损伤尺寸外推崩塌, 数据集天花板 |
| [exp10](experiments/10_noise_robustness.py) | 信噪比 +10 dB 时全 120 维崩到 7.7%, Bot-40 保持 94%+ |
| [exp12](experiments/12_edge_latency.py) | 单窗口 <1 ms, 部署产物 <10 KB |

---

## 边缘部署规格 (exp12)

单窗口延迟与部署产物体积 (AMD Ryzen, 单核 CPU, 100 次采样中位数与 p95):

| 分类器 | 单窗口延迟 (中位数) | 单窗口延迟 (p95) | 部署产物大小 |
|---|---:|---:|---:|
| **LR** | **0.69 ms** | 0.90 ms | **4.2 KB** |
| LinearSVM | 0.71 ms | 1.10 ms | 5.7 KB |
| SVM-RBF | 0.91 ms | 1.24 ms | 70.6 KB |
| KNN | 2.14 ms | 2.50 ms | 109.2 KB |

*窗口 = 2048 采样点 @ 12 kHz (等效 170.7 ms 物理时长), 推理耗时 <1% 窗口时长. 完整规格见 [`exp12_deploy_spec.txt`](outputs/exp12_deploy_spec.txt).*

**综合选型**: LR 单窗口延迟最短、部署产物最小、12 任务平均精度最高 (99.41%)、方差最小——这是本流水线的默认推荐选型.

### 部署侧调用示意

训练侧完成后, 生产端调用只需 3 行:

```python
from vibrolab import features
import pickle

# 加载部署产物 (< 5 KB)
sc, clf, bot40 = pickle.loads(open('cwru_lr.pkl', 'rb').read())

# 单个 2048 采样点窗口 → 故障标签
feat = features.extract_cfd(signal_window[None, :], fs=12000)
label = clf.predict(sc.transform(feat[:, bot40]))[0]
```

**运行时依赖**: 仅 `numpy` 加 `scikit-learn`, 可 PyInstaller 打包为独立可执行文件 (约 30 MB), 或裁剪核心 numpy 路径下沉到 ARM Cortex-A / Cortex-M7 级别边缘设备.

---

## LLM 诊断解释接口 (可选)

流水线末端预留了"结构化诊断结果 → 自然语言维修建议"的接口, 后端可插拔:

- **本地小模型后端** ([`exp06_diagnosis_report.md`](outputs/exp06_diagnosis_report.md)) — 离线可用, 适合内网环境
- **云端 API 后端** ([`exp06b_diagnosis_report_api.md`](outputs/exp06b_diagnosis_report_api.md)) — 遵循 OpenAI-Compatible 协议, 阿里百炼 / DeepSeek / 智谱 / Moonshot / OpenAI 官方 / vLLM 自部署皆可对接; 附 token 消耗与成本核算 (示例合计 ¥0.16, 规模估算 ¥40/天/1000 设备)

**云端配置**: 复制 [`.env.example`](.env.example) 为 `.env`, 填入 `LLM_API_KEY / LLM_BASE_URL / LLM_MODEL / LLM_PRICE_IN / LLM_PRICE_OUT` 五个变量, 然后 `python experiments/06b_llm_diagnosis_api.py`.

> ℹ️ CFD 120 维本身具备物理可解释性 (时域统计 / 频域分布 / 分频带能量 / 主峰 / 倒谱), 生产环境下 LLM 层为可选组件, 不影响诊断精度.

---

## 目录结构

```
vibrolab/
├── vibrolab/                        # 核心 Python 包
│   ├── io.py                        # CWRU 数据加载 + 滑窗切分
│   ├── features.py                  # 120 维 CFD 特征 + Cohen's d 预对齐
│   └── paths.py                     # 路径管理
├── llm/                             # LLM 后端 (本地 / OpenAI-Compatible 云端)
│   ├── backends.py
│   └── prompts/diagnosis_zh.txt
├── experiments/                     # 11 个实验脚本 + 一键运行入口
│   ├── 02_within_condition.py       # 同工况基线 (5-fold + 留一文件)
│   ├── 03_cross_condition.py        # 跨工况: 精度断崖与 Bot-40 修复
│   ├── 04_plots.py                  # exp02/03 出图
│   ├── 05_literature_benchmark.py   # 与主流方法对比
│   ├── 06_llm_diagnosis.py          # LLM 接口 (本地后端)
│   ├── 06b_llm_diagnosis_api.py     # LLM 接口 (云端 API 后端)
│   ├── 07_all_12_tasks.py           # 12 任务全负载矩阵
│   ├── 08_prequential.py            # 流式部署 Prequential 验证
│   ├── 09_leave_diameter_out.py     # 未见损伤尺寸外推 (阴性)
│   ├── 10_noise_robustness.py       # 加噪声鲁棒性
│   ├── 11_verification_plots.py     # exp07-10 出图
│   ├── 12_edge_latency.py           # 边缘部署延迟与产物规格
│   └── 99_run_all.py                # 一键跑完所有
├── outputs/                         # 实验产物
│   ├── VERIFICATION.md              # 完整验证报告 (15 分钟深度阅读)
│   ├── *.csv                        # 各实验数据
│   ├── figures/                     # 8 张结果图
│   ├── exp12_deploy_spec.txt        # 边缘部署规格卡
│   └── exp06*_diagnosis_report*.md  # LLM 接口示例产物
├── data/                            # 数据存放说明 (CWRU .mat 不在 git 内)
├── .env.example                     # 云端 LLM 环境变量模板
├── requirements.txt
└── LICENSE
```

---

## 关于数据集

**CWRU Bearing Data Center**, 12 kHz 采样, DE (Drive End) 端加速度信号, 4 类工作负载 (0/1/2/3 HP) × 10 类工况标签. 数据下载: <https://engineering.case.edu/bearingdatacenter>.

---

## 已知局限

- **单数据集验证**. 仅在 CWRU 上评估, 未在 PU / MFPT / XJTU-SY 等其他公开数据集上验证. 完整的外部有效性论证需要多数据集补充, 但那超出了本次独立复现验证的范围.
- **transductive UDA 设定**. 跨工况实验属于标准 transductive UDA 设定——特征筛选阶段可以观察目标域样本分布 (不使用目标域标签). exp08 补充了流式部署下只使用少量目标域样本的对照, 但严格 inductive (完全零目标样本) 的评估不在本仓库范围内.
- **CWRU 数据集本身的结构性天花板**. exp09 阴性结果表明: 未见损伤尺寸外推精度显著下降 (均值 37.8% 到 63.8%), 这不是本方法的问题, 是数据集每类只有 3 种损伤直径的样本量约束. CWRU 上的高精度并不足以证明"完全未见工况可泛化".
- **传统机器学习路线**. 全程只用 sklearn 里的四路分类器, 没有与深度模型做端到端比较——本方法的价值主张是"零训练特征选择", 不是精度 SOTA. 场景需要处理复杂时序建模的话, 本方法未必最优.

---

## Author

**康龙辉** · 燕山大学机械工程学院 2027 届硕士研究生
📧 [k3132755765@163.com](mailto:k3132755765@163.com) · GitHub [@kanglongh](https://github.com/kanglongh)

如果本仓库对你有帮助, 欢迎 Star ⭐ 或提 Issue 讨论. 商业使用请遵循 MIT License.

---

## License

MIT. 见 [`LICENSE`](LICENSE).
