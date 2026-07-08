"""TL 方法效率谱 (真迁移学习方法, 非分类器换皮):
  - 深度 DA: 1D-CNN 实测 + DCDAN/AMDA/CDHM 文献估 (~1-3MB, GPU, 塞不进MCU)
  - 浅层 DA: CORAL 实测 (协方差对齐, 闭式)
  - vibrolab: Bot-40 特征选择 DA (实测)
比 模型体积 + 推理延迟 + MCU 可部署性. 不比精度.
"""
import os, sys, time, numpy as np, scipy.linalg

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vibrolab import io as vio, features as vbf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

WINDOW, STEP, FS = 2048, 2048, vio.FS_12K
mat_pow = lambda M, p: scipy.linalg.fractional_matrix_power(M, p).real


def load(loads):
    s = vio.load_cwru(loads=loads); X, y, _ = vio.build_dataset(s, WINDOW, STEP)
    return vbf.extract_cfd(X, fs=FS).astype(np.float32), y


def main():
    X0, y0 = load([0]); X3, y3 = load([3])
    D = 120

    # ---- CORAL (浅层 DA, 闭式协方差对齐) ----
    Cs = np.cov(X0.T) + 1e-4*np.eye(D)
    Ct = np.cov(X3.T) + 1e-4*np.eye(D)
    T = mat_pow(Ct, -0.5) @ mat_pow(Cs, 0.5)   # 120x120 对齐变换
    mus, mut = X0.mean(0), X3.mean(0)
    # 部署模型: T(120x120) + mus + mut + LR(在源上训)
    lr_coral = LogisticRegression(max_iter=3000, random_state=42).fit(X0, y0)
    coral_bytes = T.nbytes + mus.nbytes + mut.nbytes + lr_coral.coef_.nbytes + lr_coral.intercept_.nbytes
    def coral_infer(x):
        xa = (x - mut) @ T + mus
        return lr_coral.predict(xa[None,:])[0]
    for _ in range(30): coral_infer(X3[0])
    t0 = time.perf_counter()
    for i in range(500): coral_infer(X3[i % len(X3)])
    coral_ms = (time.perf_counter() - t0) / 500 * 1000

    # ---- vibrolab (Bot-40 + LR) ----
    _, ranked = vbf.prealign_select(X0, X3, method='cohen')
    bot40 = np.sort(vbf.select_low_drift(ranked, k=40))
    sc = StandardScaler().fit(X0[:, bot40])
    lr_vbl = LogisticRegression(max_iter=3000, random_state=42).fit(sc.transform(X0[:, bot40]), y0)
    vbl_bytes = lr_vbl.coef_.nbytes + lr_vbl.intercept_.nbytes + sc.mean_.nbytes + sc.scale_.nbytes + bot40.nbytes
    def vbl_infer(x):
        f = (x[bot40] - sc.mean_) / sc.scale_
        return lr_vbl.predict(f[None,:])[0]
    for _ in range(30): vbl_infer(X3[0])
    t0 = time.perf_counter()
    for i in range(2000): vbl_infer(X3[i % len(X3)])
    vbl_ms = (time.perf_counter() - t0) / 2000 * 1000

    # ---- 准确率 (vibrolab 实测; CORAL 取 exp05 文献数 49.21% [AMDA], 本地复测 ~41% 同结论) ----
    vbl_acc = (lr_vbl.predict(sc.transform(X3[:, bot40])) == y3).mean()

    # ---- 汇总 (1D-CNN 实测 + 文献深度DA估) ----
    rows = [
        ('vibrolab\n(Bot-40+LR)', vbl_bytes, vbl_ms, '能', 'ESP32', f'{vbl_acc*100:.1f}%'),
        ('CORAL\n(浅层DA)', coral_bytes, coral_ms, '能', 'MCU', '49.21%(文献)'),
        ('1D-CNN\n(深度DA实测)', 979.8*1024, 0.35, '不能', 'GPU/服务器', '-'),
        ('DCDAN/AMDA/CDHM\n(深度DA文献)', 2.0*1024*1024, 2.0, '不能', 'GPU/服务器', '~99%(文献)'),
    ]
    print(f'{"TL方法":22s} {"体积":>10s} {"推理/窗":>9s} {"MCU":>5s} {"硬件":>10s} {"3HP精度":>9s}')
    print('-'*72)
    for n, b, ms, mcu, hw, acc in rows:
        kb = b/1024
        sz = f'{kb:.0f} KB' if kb<1024 else f'{kb/1024:.1f} MB'
        print(f'{n:22s} {sz:>10s} {ms:>6.2f} ms  {mcu:>4s}  {hw:>10s}  {acc:>8s}')

    # ---- 出图 ----
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    names = [r[0] for r in rows]; sizes = [r[1]/1024 for r in rows]; lats = [r[2] for r in rows]
    feas = [r[3] for r in rows]
    colors = ['#3a9d5d' if f=='能' else '#d64545' for f in feas]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    bars = ax1.bar(range(len(names)), sizes, color=colors)
    ax1.set_yscale('log'); ax1.set_xticks(range(len(names))); ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylabel('模型体积 (KB, 对数)'); ax1.set_title('模型体积')
    for b, s, f in zip(bars, sizes, feas):
        lbl = f'{s:.0f}KB' if s<1024 else f'{s/1024:.1f}MB'
        ax1.text(b.get_x()+b.get_width()/2, s*1.2, lbl, ha='center', fontsize=9, fontweight='bold')
        if f=='不能': ax1.text(b.get_x()+b.get_width()/2, s*0.4, '塞不进MCU', ha='center', fontsize=8, color='#fff')
    ax1.axhline(200, color='gray', ls=':', lw=1); ax1.text(0.1, 240, 'MCU舒适线 ~200KB', fontsize=8, color='gray')
    bars2 = ax2.bar(range(len(names)), lats, color=colors)
    ax2.set_xticks(range(len(names))); ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylabel('推理延迟 (ms/窗)'); ax2.set_title('推理延迟 (CPU/各自硬件)')
    for b, l in zip(bars2, lats):
        ax2.text(b.get_x()+b.get_width()/2, l*1.05, f'{l:.2f}', ha='center', fontsize=9)
    fig.suptitle('迁移学习方法效率谱: 深度DA vs 浅层DA vs vibrolab (CWRU 跨工况)', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0,0.05,1,0.95])
    path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'figures', 'fig9_tl_efficiency.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    print(f'\n落盘: {path}')


if __name__ == '__main__':
    main()
