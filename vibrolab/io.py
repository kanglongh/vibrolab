"""
CWRU 数据加载 + 滑窗切分.

数据集: Case Western Reserve University Bearing Data Center
    - 采样率: 12 kHz (12k_DE) 或 48 kHz (48k_DE)
    - 4 种负载: 0 HP / 1 HP / 2 HP / 3 HP (对应文件后缀 _0/_1/_2/_3)
    - 每个 .mat 文件包含 ~120k 或 ~240k 采样点

本模块只处理 12k_DE + Normal 两个文件夹, 因为:
    - Nyquist 6 kHz 足以覆盖轴承故障特征频段
    - DE (Drive End) 传感器是 CWRU 事实标准
    - Normal 是唯一的正常样本来源

10 类方案 (与 DCDAN [Zhang 2020], AMDA [Ragab 2021], CDHM [Zhao 2023] 对齐):
    N + B007/B014/B021 + IR007/IR014/IR021 + OR007@6/OR014@6/OR021@6
    注意: B028/IR028 尺寸未使用 (缺 0HP 数据), OR 位置只用 @6.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import scipy.io

from .paths import ROOT

# CWRU 数据默认路径 (相对 vibrolab/ 上层)
# 用户可通过环境变量 CWRU_DATA_ROOT 覆盖
_default_data_root = ROOT.parent / 'Data'
DATA_ROOT = Path(os.environ.get('CWRU_DATA_ROOT', str(_default_data_root)))

FS_12K = 12000    # 采样率 (Hz)

# 10 类方案 (学术标准, 与主流论文一致)
# 顺序: N + B007/B014/B021 + IR007/IR014/IR021 + OR007@6/OR014@6/OR021@6
LABEL_NAMES = [
    'N',
    'B007', 'B014', 'B021',
    'IR007', 'IR014', 'IR021',
    'OR007@6', 'OR014@6', 'OR021@6',
]
LABEL_TO_INT = {n: i for i, n in enumerate(LABEL_NAMES)}

# 4 种负载 (HP), 对应文件后缀
LOADS = [0, 1, 2, 3]


@dataclass
class CWRUSample:
    """单个 CWRU 信号段的容器."""
    signal: np.ndarray          # 一维振动信号
    label: str                  # 10 类方案下的标签, 见 LABEL_NAMES
    load: int                   # 0 / 1 / 2 / 3 (HP)
    filename: str               # 原始 .mat 文件名, 用于溯源


def _extract_de_signal(mat_dict: dict) -> np.ndarray:
    """从 .mat 字典里找 _DE_time 后缀的字段, 返回一维信号."""
    for key, val in mat_dict.items():
        if key.startswith('__'):
            continue
        if key.endswith('_DE_time'):
            return np.asarray(val).squeeze()
    raise KeyError(f"No *_DE_time field in mat: keys={list(mat_dict)}")


def _parse_filename(fname: str) -> tuple:
    """解析 CWRU 文件名, 提取 (label, load).

    支持的文件类型 (10 类方案):
    - 'Normal_N.mat'          -> ('N', N)
    - 'B007_N.mat' / 'B014_N.mat' / 'B021_N.mat'
    - 'IR007_N.mat' / 'IR014_N.mat' / 'IR021_N.mat'
    - 'OR007@6_N.mat' / 'OR014@6_N.mat' / 'OR021@6_N.mat'

    以下情况返回 (None, load) 表示跳过:
    - B028 / IR028 尺寸: 缺 0HP 数据, 不使用
    - OR 其他位置 (@3, @12): 不使用 (与主流论文对齐, 只用 @6)
    """
    stem = fname.rsplit('.', 1)[0]
    load = int(stem[-1])
    # 去掉尾部的 _{load}
    stem_no_load = stem[:-2] if stem[-2] == '_' else stem

    if stem_no_load.startswith('Normal'):
        return 'N', load
    if stem_no_load in LABEL_NAMES:
        return stem_no_load, load
    # 其他类型 (B028, IR028, OR@3, OR@12) 返回 None 表示跳过
    return None, load


def load_cwru(
    data_root: Path = None,
    loads: List[int] = None,
    labels: List[str] = None,
) -> List[CWRUSample]:
    """加载 CWRU 12k_DE + Normal 全部文件 (10 类方案).

    Parameters
    ----------
    data_root : Path
        CWRU Data/ 目录, 需包含 Normal/ 和 12k_DE/ 两个子文件夹.
        若为 None, 使用模块级 DATA_ROOT (可通过 CWRU_DATA_ROOT 环境变量指定).
    loads : list[int]
        要加载的负载子集, 例如 [0, 3] 只取 0HP 和 3HP. None 表示全 4 种.
    labels : list[str]
        要加载的故障类子集. None 表示加载 LABEL_NAMES 中全部 10 类.

    Returns
    -------
    samples : list of CWRUSample
    """
    if data_root is None:
        data_root = DATA_ROOT
    if not data_root.exists():
        raise FileNotFoundError(f"CWRU data not found at {data_root}. "
                                f"Set env var CWRU_DATA_ROOT or pass data_root explicitly.")

    loads_filter = set(loads) if loads else set(LOADS)
    labels_filter = set(labels) if labels else set(LABEL_NAMES)

    samples = []
    for sub in ['Normal', '12k_DE']:
        folder = data_root / sub
        if not folder.exists():
            print(f"[WARN] Missing folder: {folder}, skipping")
            continue
        for mat_path in sorted(folder.glob('*.mat')):
            lbl, ld = _parse_filename(mat_path.name)
            # lbl 为 None 表示该文件不属于 10 类方案 (如 B028 / OR@3)
            if lbl is None:
                continue
            if lbl not in labels_filter or ld not in loads_filter:
                continue
            try:
                sig = _extract_de_signal(scipy.io.loadmat(str(mat_path)))
            except Exception as e:
                print(f"[WARN] Failed to load {mat_path.name}: {e}")
                continue
            samples.append(CWRUSample(
                signal=sig.astype(np.float32),
                label=lbl,
                load=ld,
                filename=mat_path.name,
            ))
    return samples


def make_windows(signal: np.ndarray, window: int, step: int) -> np.ndarray:
    """滑窗切分一维信号 -> 二维 (n_windows, window).

    Parameters
    ----------
    signal : (N,) ndarray
    window : int
        窗长, e.g. 2048.
    step : int
        步长, e.g. 2048 (无重叠) 或 1024 (50% overlap).

    Returns
    -------
    windows : (n_windows, window) ndarray
    """
    n = len(signal)
    if n < window:
        return np.empty((0, window), dtype=signal.dtype)
    n_win = (n - window) // step + 1
    idx = np.arange(n_win)[:, None] * step + np.arange(window)[None, :]
    return signal[idx]


def build_dataset(
    samples: List[CWRUSample],
    window: int = 2048,
    step: int = 2048,
) -> tuple:
    """把 CWRUSample 列表切窗成 (X_windows, y, loads) 三个数组.

    Returns
    -------
    X : (N, window) ndarray, dtype=float32
    y : (N,) ndarray, int64 (10 类标签编号, 见 LABEL_NAMES)
    loads : (N,) ndarray, int
    """
    Xs, ys, lds = [], [], []
    for s in samples:
        w = make_windows(s.signal, window, step)
        if w.shape[0] == 0:
            continue
        Xs.append(w)
        ys.append(np.full(w.shape[0], LABEL_TO_INT[s.label], dtype=np.int64))
        lds.append(np.full(w.shape[0], s.load, dtype=np.int32))
    X = np.concatenate(Xs, axis=0) if Xs else np.empty((0, window), dtype=np.float32)
    y = np.concatenate(ys) if ys else np.empty(0, dtype=np.int64)
    loads = np.concatenate(lds) if lds else np.empty(0, dtype=np.int32)
    return X, y, loads


if __name__ == '__main__':
    from collections import Counter

    print(f"CWRU data root: {DATA_ROOT}")
    print(f"Exists: {DATA_ROOT.exists()}")
    if not DATA_ROOT.exists():
        raise SystemExit(1)

    print('\n=== 10 类方案自检 (loads=[0, 3]) ===')
    samples = load_cwru(loads=[0, 3])
    print(f"Loaded {len(samples)} files.")
    c = Counter((s.label, s.load) for s in samples)
    for k, v in sorted(c.items()):
        print(f"  {k}: {v} files")
    X, y, loads = build_dataset(samples, window=2048, step=2048)
    print(f"\nDataset: X={X.shape} y={y.shape} loads={loads.shape}")
    print(f"  Windows per class: {dict(Counter(y.tolist()))}")
    print(f"  Windows per load:  {dict(Counter(loads.tolist()))}")
    print(f"  n_classes: {len(np.unique(y))}")
