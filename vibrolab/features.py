"""
vibrolab.features — 120 维物理可解释振动特征.

五模块设计
----------
    time     [0:12]      时域统计 (RMS/峰度/波形因子/间隙因子/...)
    freq     [12:27]     频域统计 (质心/带宽/rolloff/谱斜率/...)
    band     [27:91]     分频带能量特征 (32 带 × 2 [ratio + logenergy])
    peaks    [91:103]    主峰 6 个 × 2 [freq + amp]
    cepstral [103:120]   倒谱系数 (捕捉调制模式)

实现说明: 时域 / 主峰两块是教科书级实现; 频域 / 分频带 / 倒谱三块基于常见的
物理特征公式实现. 全部走 numpy, 无外部依赖.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal
from scipy.fft import rfft, rfftfreq

FS_DEFAULT = 12000
N_DIMS = 120

# 五模块索引
MODULES = {
    'time':     slice(0, 12),
    'freq':     slice(12, 27),
    'band':     slice(27, 91),
    'peaks':    slice(91, 103),
    'cepstral': slice(103, 120),
}


# ============================================================
# 时域 12 维 (公开)
# ============================================================
def _extract_time(w: np.ndarray) -> np.ndarray:
    """时域 12 维, w: (N,) 单窗."""
    absw = np.abs(w)
    mean = w.mean()
    std = w.std() + 1e-12
    peak = absw.max()
    rms = np.sqrt(np.mean(w ** 2)) + 1e-12
    return np.array([
        mean,                                      # 0 · Mean
        std,                                       # 1 · Std
        rms,                                       # 2 · RMS
        peak,                                      # 3 · Peak (max abs)
        peak - w.min(),                            # 4 · PeakToPeak
        absw.mean(),                               # 5 · MeanAbs
        np.mean(np.sqrt(absw)) ** 2,               # 6 · SRA (Square Root Amplitude)
        np.mean((w - mean) ** 3) / std ** 3,       # 7 · Skewness
        np.mean((w - mean) ** 4) / std ** 4,       # 8 · Kurtosis
        peak / rms,                                # 9 · CrestFactor
        peak / (absw.mean() + 1e-12),              # 10 · ImpulseFactor
        rms / (absw.mean() + 1e-12),               # 11 · ShapeFactor
    ], dtype=np.float32)


# ============================================================
# 主峰 12 维 (公开)
# ============================================================
def _extract_peaks(mag: np.ndarray, freqs: np.ndarray, n_peaks: int = 6) -> np.ndarray:
    """主峰 6 个 × [freq, amp] = 12 维. 谱峰按幅值降序取前 n_peaks."""
    idx, _ = sp_signal.find_peaks(mag)
    if len(idx) == 0:
        return np.zeros(2 * n_peaks, dtype=np.float32)
    top = idx[np.argsort(mag[idx])[::-1][:n_peaks]]
    top_sorted = top[np.argsort(freqs[top])]        # 按频率排序, 便于跨样本对齐
    out = np.zeros(2 * n_peaks, dtype=np.float32)
    for i, k in enumerate(top_sorted):
        out[2 * i]     = freqs[k]
        out[2 * i + 1] = mag[k]
    return out


# ============================================================
# 频域 15 + 分频带 64 + 倒谱 17
# ============================================================
def _extract_freq(mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """15 维频域统计: 质心/带宽/rolloff/谱斜率/谱平坦度/谱熵 + 幅值谱统计."""
    total = mag.sum() + 1e-12
    p = mag / total                                    # normalize
    centroid = (p * freqs).sum()
    bw = np.sqrt((p * (freqs - centroid) ** 2).sum())
    cs = np.cumsum(mag)
    rolloff85 = freqs[np.searchsorted(cs, 0.85 * cs[-1])] if cs[-1] > 0 else 0
    rolloff95 = freqs[np.searchsorted(cs, 0.95 * cs[-1])] if cs[-1] > 0 else 0
    x_ax = np.log(freqs + 1)
    y_ax = np.log(mag + 1e-12)
    slope = np.polyfit(x_ax, y_ax, 1)[0] if len(x_ax) > 1 else 0
    flatness = (np.exp(np.mean(np.log(mag + 1e-12))) / (mag.mean() + 1e-12))
    entropy = -(p * np.log(p + 1e-12)).sum()
    out = np.array([
        centroid, bw, rolloff85, rolloff95, slope, flatness, entropy,
        mag.mean(), mag.std(), mag.max(), mag.min(),
        np.percentile(mag, 25), np.percentile(mag, 50), np.percentile(mag, 75),
        np.percentile(mag, 90),
    ], dtype=np.float32)
    return out


def _extract_band(mag: np.ndarray, freqs: np.ndarray, fs: int) -> np.ndarray:
    """32 带 × [能量占比, log 能量] = 64 维."""
    n_bands = 32
    max_f = fs / 2
    band_edges = np.linspace(0, max_f, n_bands + 1)
    total_e = (mag ** 2).sum() + 1e-12
    ratios = np.zeros(n_bands, dtype=np.float32)
    logenergies = np.zeros(n_bands, dtype=np.float32)
    for i in range(n_bands):
        m = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
        be = (mag[m] ** 2).sum()
        ratios[i] = be / total_e
        logenergies[i] = np.log(be + 1e-12)
    return np.concatenate([ratios, logenergies])


def _extract_cepstral(w: np.ndarray) -> np.ndarray:
    """17 维倒谱系数."""
    spec = np.abs(rfft(w)) + 1e-12
    log_spec = np.log(spec)
    ceps = np.fft.irfft(log_spec).real
    return ceps[:17].astype(np.float32)


# ============================================================
# 主入口
# ============================================================
def extract_cfd(windows: np.ndarray, fs: int = FS_DEFAULT) -> np.ndarray:
    """把 (n_windows, window) 二维信号提取为 (n_windows, 120) 特征矩阵.

    Parameters
    ----------
    windows : (N, W) ndarray
        每行一个窗口的一维信号.
    fs : int
        采样率 (Hz). 12000 (12k) 或 48000 (48k).

    Returns
    -------
    feats : (N, 120) ndarray, dtype=float32
    """
    if windows.ndim != 2:
        raise ValueError(f"Expected 2D windows, got shape {windows.shape}")
    n = windows.shape[0]
    out = np.zeros((n, N_DIMS), dtype=np.float32)
    freqs = rfftfreq(windows.shape[1], 1.0 / fs)
    for i in range(n):
        w = windows[i]
        mag = np.abs(rfft(w))
        out[i, MODULES['time']]     = _extract_time(w)
        out[i, MODULES['freq']]     = _extract_freq(mag, freqs)
        out[i, MODULES['band']]     = _extract_band(mag, freqs, fs)
        out[i, MODULES['peaks']]    = _extract_peaks(mag, freqs)
        out[i, MODULES['cepstral']] = _extract_cepstral(w)
    return out


# ============================================================
# 特征预对齐工具 (跨工况分布自适应)
# ============================================================
def _cohens_d_naive(X_src: np.ndarray, X_tgt: np.ndarray) -> np.ndarray:
    """逐维 Cohen's d 效应量绝对值.
    试过 KL / Wasserstein, 这个算得最快, 效果也够用.
    """
    n_dims = X_src.shape[1]
    d_vals = np.zeros(n_dims, dtype=np.float32)
    for i in range(n_dims):
        src_mu = X_src[:, i].mean()
        tgt_mu = X_tgt[:, i].mean()
        src_var = X_src[:, i].var()
        tgt_var = X_tgt[:, i].var()
        pooled_std = np.sqrt((src_var + tgt_var) / 2) + 1e-8  # 防除零
        d_vals[i] = abs(src_mu - tgt_mu) / pooled_std
    return d_vals


def _kl_divergence_1d(X_src: np.ndarray, X_tgt: np.ndarray, bins: int = 30) -> np.ndarray:
    """KL 散度 (备选, 效果不如 Cohen's d, 留着以后试)."""
    n_dims = X_src.shape[1]
    kl = np.zeros(n_dims, dtype=np.float32)
    for i in range(n_dims):
        hist_src, edges = np.histogram(X_src[:, i], bins=bins, density=True)
        hist_tgt, _ = np.histogram(X_tgt[:, i], bins=edges, density=True)
        hist_src += 1e-8
        hist_tgt += 1e-8
        kl[i] = np.sum(hist_src * np.log(hist_src / hist_tgt))
    return kl


def _wass_1d(X_src: np.ndarray, X_tgt: np.ndarray) -> np.ndarray:
    """1 阶 Wasserstein 距离 (备选, 计算太慢了, 实际不用)."""
    n_dims = X_src.shape[1]
    w = np.zeros(n_dims, dtype=np.float32)
    for i in range(n_dims):
        w[i] = np.abs(np.sort(X_src[:, i]) - np.sort(X_tgt[:, i])).mean()
    return w


def prealign_select(X_src: np.ndarray, X_tgt: np.ndarray, method: str = 'cohen') -> tuple[np.ndarray, np.ndarray]:
    """预对齐: 按源域/目标域分布差异排序, 返回漂移值和维度索引.
    这步只是简单的特征预处理, 不需要目标域标签. method='cohen' 计算最快,
    'kl' / 'wass' 备选, 实际效果差异不大.
    """
    if method == 'cohen':
        d = _cohens_d_naive(X_src, X_tgt)
    elif method == 'kl':
        d = _kl_divergence_1d(X_src, X_tgt)
    elif method == 'wass':
        d = _wass_1d(X_src, X_tgt)
    else:
        raise ValueError(f"unknown drift method: {method}")
    return d, np.argsort(d)[::-1]  # (d_vals, ranked_idx) 高漂 → 低漂


def select_low_drift(ranked_dims: np.ndarray, k: int = 40) -> np.ndarray:
    """取漂移量最小的 k 个维度做跨工况诊断.
    k=40 是在 CWRU 上调出来的经验值, 不同设备可根据数据调整.
    """
    return ranked_dims[-k:]


if __name__ == '__main__':
    # 快速自检
    np.random.seed(42)
    w = np.random.randn(3, 2048).astype(np.float32)
    feats = extract_cfd(w, fs=12000)
    print(f"Features shape: {feats.shape}, dtype: {feats.dtype}")
    print(f"Modules:")
    for name, sl in MODULES.items():
        seg = feats[:, sl]
        print(f"  {name:9s} [{sl.start:3d}:{sl.stop:3d}] {seg.shape[1]:2d}d  "
              f"mean={seg.mean():+.3e}  finite={np.isfinite(seg).all()}")
