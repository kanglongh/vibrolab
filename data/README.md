# CWRU 数据下载指引

本项目使用 **Case Western Reserve University Bearing Data Center** 的公开轴承故障数据集.

## 下载

前往官网: <https://engineering.case.edu/bearingdatacenter>

或者使用整理好的 GitHub 镜像仓库: <https://github.com/mattdellorto/CWRU-1> (原始镜像, 本项目使用的数据来源).

## 目录组织

将 `Data/` 文件夹放到本项目**上层目录**, 结构如下:

```
GitHub/                          # 或任何你的上层目录
├── Data/                        # ← 从 CWRU 下载, 只需下面这两个子文件夹
│   ├── Normal/                  # 4 个正常样本 (0/1/2/3 HP 负载各一)
│   │   ├── Normal_0.mat
│   │   ├── Normal_1.mat
│   │   ├── Normal_2.mat
│   │   └── Normal_3.mat
│   └── 12k_DE/                  # 60 个故障样本, 12 kHz 驱动端
│       ├── B007_0.mat           # B/IR/OR × 007/014/021/028 尺寸 × 0/1/2/3 HP
│       ├── ...
│       └── OR021@6_3.mat
│
└── vibrolab/                    # 本项目
```

**只需要 `Normal/` 和 `12k_DE/` 两个文件夹** (合计 ~187 MB). CWRU 原始压缩包里还有 `12k_FE/` (风扇端) 和 `48k_DE/` (高采样率), 本项目不使用.

## 自定义数据路径

若你的 `Data/` 不在项目上层目录, 通过环境变量指定:

```bash
# Linux/macOS
export CWRU_DATA_ROOT=/path/to/your/CWRU/Data

# Windows PowerShell
$env:CWRU_DATA_ROOT = "C:\your\path\to\CWRU\Data"

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
