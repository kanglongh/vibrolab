# CWRU 数据下载指引

本项目使用 **Case Western Reserve University Bearing Data Center** 的公开轴承故障数据集.

## 下载

前往官网: <https://engineering.case.edu/bearingdatacenter>

或者使用整理好的 GitHub 镜像仓库: <https://github.com/mattdellorto/CWRU-1> (原始镜像, 本项目使用的数据来源).

## 目录组织

把 `Normal/` 和 `12k_DE/` 两个子文件夹直接放到本 `data/` 目录下, 结构如下:

```
vibrolab/                        # 本项目根
├── data/                        # ← 你在这里
│   ├── README.md                # (本文件)
│   ├── Normal/                  # 4 个正常样本 (0/1/2/3 HP 负载各一)
│   │   ├── Normal_0.mat
│   │   ├── Normal_1.mat
│   │   ├── Normal_2.mat
│   │   └── Normal_3.mat
│   └── 12k_DE/                  # 60 个故障样本, 12 kHz 驱动端
│       ├── B007_0.mat           # B/IR/OR × 007/014/021/028 尺寸 × 0/1/2/3 HP
│       ├── ...
│       └── OR021@6_3.mat
├── vibrolab/                    # Python 包
├── experiments/                 # 实验脚本
└── firmware_v1_5/               # 边缘部署
```

**只需要 `Normal/` 和 `12k_DE/` 两个文件夹** (合计 ~187 MB). CWRU 原始压缩包里还有 `12k_FE/` (风扇端) 和 `48k_DE/` (高采样率), 本项目不使用.

`.gitignore` 会自动挡住 `data/**/*.mat`, 数据文件不会被 git 追踪.

## 自定义数据路径

若你的数据不想放在 `data/`, 通过环境变量指定:

```bash
# Linux/macOS
export CWRU_DATA_ROOT=/path/to/your/CWRU/Data

# Windows PowerShell
$env:CWRU_DATA_ROOT = "C:\your\path\to\CWRU\Data"

# Windows CMD
set CWRU_DATA_ROOT=C:\your\path\to\CWRU\Data

# 然后运行
python experiments/99_run_all.py
```

## 命名规则

CWRU 文件名编码了 (故障类型, 损伤尺寸, 负载) 三元组:

- `Normal_{L}.mat`  → 正常, {L} HP 负载
- `B{D}_{L}.mat`    → 滚动体故障, {D} 千分英寸损伤尺寸, {L} HP
- `IR{D}_{L}.mat`   → 内圈故障
- `OR{D}@{P}_{L}.mat` → 外圈故障 (@3 / @6 / @12 分别是外圈的三个位置)

例如 `IR014_2.mat` = 内圈 0.014 英寸损伤在 2 HP 负载下.

`vibrolab.io._parse_filename` 会自动解析这些编码.
