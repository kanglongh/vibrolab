"""效率基线实测: 1D-CNN (深度TL典型) vs vibrolab LR+Bot-40.

只测计算效率: 模型体积 + 推理延迟. 不比精度(只打印确认CNN非随机), 不谈原理.
CNN 用 GPU(若有) + CPU; vibrolab 用 CPU. ESP32 数取实测/已知.
"""
import os, sys, time, numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vibrolab import io as vio, features as vbf

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW, STEP, FS = 2048, 2048, vio.FS_12K
print(f'torch {torch.__version__} | device={DEVICE}')


# ---------- 1D-CNN (深度TL典型架构) ----------
class CNN1D(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 5, stride=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 5, stride=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 5, stride=2), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )
    def forward(self, x): return self.net(x)


def load_raw(loads):
    s = vio.load_cwru(loads=loads); X, y, _ = vio.build_dataset(s, WINDOW, STEP)
    return X.astype(np.float32), y


def main():
    X0, y0 = load_raw([0]); X3, y3 = load_raw([3])
    Xtr = torch.from_numpy(X0).unsqueeze(1)            # (N,1,2048)
    ytr = torch.from_numpy(y0).long()
    Xte = torch.from_numpy(X3).unsqueeze(1)
    yte = torch.from_numpy(y3).long()
    print(f'CNN 数据: 训练0HP {Xtr.shape}, 测3HP {Xte.shape}')

    # --- 训练 CNN (只为拿到真实模型, 不追求精度) ---
    cnn = CNN1D().to(DEVICE)
    opt = torch.optim.Adam(cnn.parameters(), lr=1e-3); crit = nn.CrossEntropyLoss()
    Xtr_d, ytr_d = Xtr.to(DEVICE), ytr.to(DEVICE)
    cnn.train()
    for ep in range(8):
        idx = np.random.permutation(len(Xtr_d))
        for i in range(0, len(idx), 64):
            b = idx[i:i+64]
            opt.zero_grad(); loss = crit(cnn(Xtr_d[b]), ytr_d[b]); loss.backward(); opt.step()
    cnn.eval()
    with torch.no_grad():
        acc = (cnn(Xte.to(DEVICE)).argmax(1).cpu() == yte).float().mean()
    n_params = sum(p.numel() for p in cnn.parameters())
    cnn_bytes = n_params * 4
    print(f'CNN 训完 (8ep), 3HP精度={acc:.3f} (仅确认非随机, 不作对比)')

    # --- CNN 推理延迟 ---
    def latency(model, X, dev, n_warm=100, n_meas=1000):
        model = model.to(dev); Xd = X.to(dev)
        with torch.no_grad():
            for _ in range(n_warm): model(Xd[:1])
            if dev == 'cuda': torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i in range(n_meas): model(Xd[i:i+1])
            if dev == 'cuda': torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n_meas * 1000  # ms/窗
    cnn_cpu_ms = latency(cnn, Xte, 'cpu')
    cnn_gpu_ms = latency(cnn, Xte, 'cuda') if torch.cuda.is_available() else None

    # --- vibrolab LR+Bot-40 ---
    Xs = vbf.extract_cfd(X0, fs=FS).astype(np.float32)
    Xt = vbf.extract_cfd(X3, fs=FS).astype(np.float32)
    _, ranked = vbf.prealign_select(Xs, Xt, method='cohen')
    bot40 = np.sort(vbf.select_low_drift(ranked, k=40))
    sc = StandardScaler().fit(Xs[:, bot40])
    lr = LogisticRegression(max_iter=3000, random_state=42).fit(sc.transform(Xs[:, bot40]), y0)
    # LR 模型体积: coef(10x40)+intercept(10)+scaler(40x2)+bot40(40) = 浮点+整型
    lr_bytes = lr.coef_.nbytes + lr.intercept_.nbytes + sc.mean_.nbytes + sc.scale_.nbytes + bot40.nbytes

    # vibrolab 推理延迟 (CPU): 特征提取 + Bot-40选 + 标准化 + LR
    def vbl_infer_one(raw):
        f = vbf.extract_cfd(raw[None,:], fs=FS)[0]      # 120维
        x = (f[bot40] - sc.mean_) / sc.scale_           # 40维标准化
        return lr.predict(x[None,:])[0]
    for _ in range(20): vbl_infer_one(X3[0])            # warmup
    t0 = time.perf_counter()
    for i in range(200): vbl_infer_one(X3[i])
    vbl_cpu_ms = (time.perf_counter() - t0) / 200 * 1000
    # 单独 LR 推理 (无特征提取, 对照)
    t0 = time.perf_counter()
    for i in range(2000):
        x = (Xt[i % len(Xt), bot40] - sc.mean_) / sc.scale_
        lr.predict(x[None,:])
    vbl_lr_ms = (time.perf_counter() - t0) / 2000 * 1000

    # --- 输出效率对照表 ---
    print('\n' + '=' * 70)
    print('效率对照 (纯计算效率, 不比精度)')
    print('=' * 70)
    print(f'{"":28s} {"1D-CNN (深度TL典型)":>22s} {"vibrolab LR+Bot40":>20s}')
    print('-' * 70)
    print(f'{"模型体积":28s} {cnn_bytes/1024:>18.1f} KB   {lr_bytes/1024:>16.1f} KB')
    print(f'{"参数量":28s} {n_params:>18d}   {"~410 (LR权重)":>16s}')
    print(f'{"CPU 推理/窗":28s} {cnn_cpu_ms:>17.2f} ms   {vbl_cpu_ms:>15.2f} ms (特征+LR)')
    if cnn_gpu_ms is not None:
        print(f'{"GPU 推理/窗":28s} {cnn_gpu_ms:>17.3f} ms   {"不需要GPU":>16s}')
    print(f'{"  其中纯LR推理(无特征)":28s} {"":>22s} {vbl_lr_ms:>15.4f} ms')
    print(f'{"ESP32-S3 推理/窗":28s} {"跑不动(模型塞不进/无PyTorch路径)":>22s} {"0.6 ms (实测)":>16s}')
    print(f'{"部署硬件":28s} {"GPU/服务器/PC":>22s} {"ESP32-S3":>16s}')
    print('=' * 70)
    print(f'模型体积比: CNN / vibrolab = {cnn_bytes/lr_bytes:.0f}x')
    print(f'CPU 推理比: CNN / vibrolab = {cnn_cpu_ms/vbl_cpu_ms:.1f}x')
    if cnn_gpu_ms is not None:
        print(f'注: CNN 在GPU上 {cnn_gpu_ms:.3f}ms 快, 但需GPU; vibrolab 在 MCU 上 0.6ms, 无需GPU')


if __name__ == '__main__':
    main()
