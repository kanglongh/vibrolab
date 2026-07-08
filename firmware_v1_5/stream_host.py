"""v1.5 · PC 流式主机: 自定义数据组合, 串口发 ESP32, 收诊断.

板子用 0+3HP 训死的 40 维 + LR (2KB), 现场流式收窗逐窗诊断.


用法:
  pip install pyserial matplotlib
  # 先按 data/README.md 把 CWRU 放到 data/, 或设环境变量指到别处:
  #   set CWRU_DATA_ROOT=<你的路径>\Data      (Windows)
  #   export CWRU_DATA_ROOT=<你的路径>/Data    (Linux/macOS)
  python firmware_v1_5/stream_host.py
  (或命令行覆盖: python firmware_v1_5/stream_host.py --stream-loads 0 1 2 3 --shuffle)
"""
import os, sys, time, argparse, numpy as np
try:
    import matplotlib
    matplotlib.use('TkAgg')  # 避开 PyCharm 后端的 tostring_rgb 兼容问题
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ============================================================
#  CONFIG —— 流式输入自定义区 (改这里就行, 注释告诉你怎么填)
# ============================================================
# CWRU 数据路径: 若已设 CWRU_DATA_ROOT 环境变量则用它;
# 否则用 vibrolab/io.py 里的默认 (仓库内 data/, 见 data/README.md).

# 串口
PORT = 'COM7'          # 板子的 COM 口 (设备管理器查, 之前是 COM7)
BAUD = 921600          # 波特率, 跟固件一致 (921600)

# 流式输入数据组合 —— 核心自定义项
#
# 方式 A (推荐, 分段模式): 下面 STREAM_SEGMENTS 非空时生效.
#   每段 = (工况HP, 故障类名, 窗数), 段间严格顺序, 段内可打乱 (见 SHUFFLE).
#   模拟真实工业场景: 正常运转 → 故障出现 → 变工况 → 新故障...
#   故障类: 0=N, 1=B007, 2=B014, 3=B021, 4=IR007, 5=IR014, 6=IR021, 7=OR007@6, 8=OR014@6, 9=OR021@6
#   也可直接用类名, 如 'N', 'IR007'. 数字更短, 推荐.
#
#   清空为 [] 则自动退回到方式 B 的快捷模式.
STREAM_SEGMENTS = [
    # (HP, class, N)         含义工况，故障类，窗数

]

# 方式 B (快捷, 仅当 STREAM_SEGMENTS 为空时生效):
STREAM_LOADS = [0,1,2,3]      # 欲流式的工况 (HP). 仅快捷模式生效.
N_PER_CLASS = 10           # 每类窗数. 仅快捷模式生效.
TRAIN_LOADS = [0, 3]   # 板子模型训练工况. 其余工况字幕标 "held-out (训练没见过)".

SHUFFLE = False       # 分段模式: False=保持段内顺序, 叙事不被打断
SEED = 42              # 随机种子 (选窗 + 打乱). 改它换一组窗, 但可复现.

# 传感器降级模拟: 0=不降级, 1=ADXL1002, 2=ADXL355, 3=MPU6050
SENSOR = 2
# (bandwidth_Hz, noise_ug_sqrtHz, adc_bits, full_scale_g, 名称)
_SENSORS = [
    None,                                    # 0: 不降级
    (11000, 25,   12, 50, 'ADXL1002'),       # 1: 预期 ~99.8%
    (1900,  25,   20,  8, 'ADXL355'),        # 2: 预期 ~39%
    (260,   400,  16, 16, 'MPU6050'),        # 3: 预期 ~10%
]
# ============================================================
#  下面不用动
# ============================================================

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vibrolab import io as vio
import serial

WINDOW, STEP, FS = 2048, 2048, vio.FS_12K
MAGIC = b'\xa5\x5a'
LABEL_CN = {0:'正常',1:'滚珠007',2:'滚珠014',3:'滚珠021',4:'内圈007',5:'内圈014',6:'内圈021',7:'外圈007',8:'外圈014',9:'外圈021'}



def degrade(signal, spec, rng):
    """把干净信号降级到指定传感器规格. spec = (bw, noise, bits, full_scale) 或 None (不降级)."""
    if spec is None:
        return signal
    from scipy.signal import butter, filtfilt
    bw, noise_ug, bits, fs_g = spec
    FS = 12000
    x = signal.astype(np.float64).copy()
    if bw is not None and bw < FS / 2:
        b, a = butter(4, bw / (FS/2), 'low')
        x = filtfilt(b, a, x)
    enb = min(bw or FS/2, FS/2)
    noise_std = noise_ug * 1e-6 * np.sqrt(enb)
    x = x + rng.randn(len(x)) * noise_std
    if fs_g is not None and bits < 24:
        x = np.clip(x, -fs_g, fs_g)
        step = 2.0 * fs_g / (2 ** bits)
        x = np.round(x / step) * step
    return x.astype(np.float32)


def load_raw(loads):
    s = vio.load_cwru(loads=loads); X, y, _ = vio.build_dataset(s, WINDOW, STEP)
    return X.astype(np.float32), y


def send_recv(ser, win):
    try:
        ser.write(MAGIC + win.astype('<f4').tobytes()); ser.flush()
    except serial.SerialException:
        return None
    buf = b''; t0 = time.time()
    while time.time() - t0 < 3:
        try:
            if ser.in_waiting:
                buf += ser.read(ser.in_waiting)
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    line = line.decode('utf-8', 'replace').strip()
                    if line.startswith('D '):
                        return line
                    # 非 D 的行打印出来, 方便调试 (READY, DBG 等)
                    if line:
                        print(f'  [板子] {line}')
        except serial.SerialException:
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default=PORT)
    ap.add_argument('--stream-loads', type=int, nargs='+', default=STREAM_LOADS)
    ap.add_argument('--n-per-class', type=int, default=N_PER_CLASS)
    ap.add_argument('--shuffle', action='store_true', default=SHUFFLE)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--baud', type=int, default=BAUD)
    ap.add_argument('--all', action='store_true', help='每类用全部窗 (忽略 n-per-class)')
    ap.add_argument('--no-plot', action='store_true', help='关 PC 波形监视窗')
    args = ap.parse_args()

    print(f'加载 CWRU...'); rng = np.random.RandomState(args.seed)
    use_segments = bool(STREAM_SEGMENTS)
    needed_loads = (set(seg[0] for seg in STREAM_SEGMENTS) if use_segments
                    else set(args.stream_loads))
    data = {L: load_raw([L]) for L in needed_loads}

    def open_serial(port, baud):
        while True:
            try:
                return serial.Serial(port, baud, timeout=0.1)
            except serial.SerialException:
                print(f'  端口 {port} 不可用, 3秒后重试...')
                time.sleep(3)

    ser = open_serial(args.port, args.baud)
    print(f'等板子 READY ({args.port}), 按 RST 复位板子 或 Ctrl+C 退出...')
    buf = b''
    while True:
        try:
            if ser.in_waiting:
                buf += ser.read(ser.in_waiting)
                if b'\n' in buf:
                    line = buf.split(b'\n')[0].decode('utf-8', 'replace').strip()
                    if line:
                        print(f'  板子: {line}')
                        break
                    buf = b''
            time.sleep(0.1)
        except serial.SerialException:
            print('  [断开] 串口丢失, 重连中', end='', flush=True)
            try:
                ser.close()
            except Exception:
                pass
            while True:
                try:
                    time.sleep(1)
                    ser = serial.Serial(args.port, args.baud, timeout=0.1)
                    print(f'\n  重连成功 ({args.port})')
                    buf = b''
                    break
                except serial.SerialException:
                    print('.', end='', flush=True)
                    time.sleep(1)

    # 清空缓冲区内残留的 READY, 确保流式响应干净
    ser.reset_input_buffer()

    # 组装流式播放列表
    playlist = []
    seg_desc = []   # 每段的叙事字幕
    if use_segments:
        prev_hp, prev_cls = None, None
        for si, (load_hp, class_spec, n_windows) in enumerate(STREAM_SEGMENTS):
            X, y = data[load_hp]
            class_idx = class_spec if isinstance(class_spec, int) else vio.LABEL_TO_INT[class_spec]
            class_name = vio.LABEL_NAMES[class_idx]
            idx = np.where(y == class_idx)[0]
            if len(idx) == 0:
                print(f'[WARN] 分段 ({load_hp}HP, {class_name}): 无窗可用, 跳过')
                seg_desc.append(f'[段{si+1}] {load_hp}HP · {LABEL_CN[class_idx]} (跳过)')
                prev_hp, prev_cls = load_hp, class_idx
                continue
            # 过渡标签 (叙事)
            tags = []
            if prev_hp is not None and load_hp != prev_hp: tags.append(f'工况切换 {prev_hp}HP→{load_hp}HP')
            if prev_cls == 0 and class_idx != 0: tags.append('故障出现')
            tag = ' · '.join(tags) if tags else ('正常运转' if class_idx == 0 else '故障持续')
            held = ' (held-out, 训练没见过)' if load_hp not in TRAIN_LOADS else ''
            seg_desc.append(f'[段{si+1}/{len(STREAM_SEGMENTS)}] {load_hp}HP{held} · {LABEL_CN[class_idx]} · {tag}')
            if len(idx) < n_windows:
                print(f'  [注意] {load_hp}HP {class_name}: 请求{n_windows}窗, 仅{len(idx)}窗可用')
            if args.all:
                take = idx
            else:
                take = rng.choice(idx, min(n_windows, len(idx)), replace=False)
            if args.shuffle:
                rng.shuffle(take)
            for i in take:
                playlist.append((X[i], y[i], load_hp, si))
            prev_hp, prev_cls = load_hp, class_idx
    else:
        for L in args.stream_loads:
            X, y = data[L]
            for c in range(10):
                idx = np.where(y == c)[0]
                if args.all:
                    take = idx
                else:
                    take = rng.choice(idx, min(args.n_per_class, len(idx)), replace=False)
                for i in take:
                    playlist.append((X[i], y[i], L, 0))
        if args.shuffle:
            rng.shuffle(playlist)
    if use_segments:
        seg_desc = ' | '.join(f'{hp}HP:{cn}x{nw}' for hp, cn, nw in STREAM_SEGMENTS)
        seg_desc = seg_desc[:90] + ('...' if len(seg_desc) > 90 else '')
        print(f'\n流式: 分段模式 [{seg_desc}], {len(playlist)}窗, shuffle={args.shuffle}, seed={args.seed}\n')
    else:
        print(f'\n流式: 工况{args.stream_loads}, {len(playlist)}窗, shuffle={args.shuffle}, seed={args.seed}\n')
    print(f'{"#":>3} {"工况":>4} {"真实":8s} {"板子诊断":8s} {"置信":>6}  {"":2s}')
    print('-' * 44)

    # PC 波形监视窗 (跟 OLED 比对着看)
    plot_on = HAS_PLOT and not args.no_plot
    if plot_on:
        plt.ion()
        fig, (ax_orig, ax_degr) = plt.subplots(2, 1, figsize=(10, 6))
        line_orig, = ax_orig.plot(np.arange(WINDOW), np.zeros(WINDOW), 'b-', lw=0.5)
        line_degr, = ax_degr.plot(np.arange(WINDOW), np.zeros(WINDOW), 'r-', lw=0.5)
        ax_orig.set_ylabel('原始信号')
        ax_orig.set_title('等待数据...', fontsize=11)
        ax_orig.grid(True, alpha=0.3)
        ax_degr.set_xlabel('采样点 (2048 @ 12kHz = 170ms)')
        ax_degr.set_ylabel('降级后 (发板子)')
        ax_degr.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.pause(0.01)  # 窗口出现后手动拖到副屏即可

    ok = 0
    consecutive_timeouts = 0
    t_stream_start = time.time()
    feat_times, infer_times = [], []
    cur_seg = -1
    sensor_entry = _SENSORS[SENSOR] if SENSOR else None
    sensor_spec = sensor_entry[:4] if sensor_entry else None
    deg_rng = np.random.RandomState(SEED)
    if sensor_entry:
        print(f'>>> 传感器降级模拟: {sensor_entry[4]}  (发给板子的窗已降级)')
    for i, (win, ytrue, L, seg_idx) in enumerate(playlist):
        if use_segments and seg_idx != cur_seg:
            cur_seg = seg_idx
            print(f'\n===== {seg_desc[seg_idx]} =====')
        # 传感器降级 (可选): 让板子体验"接便宜传感器"是什么样
        win_to_send = degrade(win, sensor_spec, deg_rng) if sensor_spec else win
        resp = send_recv(ser, win_to_send)
        if resp is None:
            consecutive_timeouts += 1
            if consecutive_timeouts <= 3:
                print(f'{i:3d} {L}HP  {vio.LABEL_NAMES[ytrue]:8s} [无响应]')
            if consecutive_timeouts == 3:
                print('\n[FATAL] 连续 3 窗无响应, 板子通信失败. 退出.')
                break
            continue
        consecutive_timeouts = 0
        parts = resp.split()
        pred = parts[1] if len(parts) > 1 else '?'
        conf = parts[2] if len(parts) > 2 else '?'
        feat_ms = 0.0
        infer_us = 0.0
        for p in parts[3:]:
            if p.startswith('feat='):
                feat_ms = float(p[5:-2])  # feat=Xms → X
            elif p.startswith('infer='):
                infer_us = float(p[6:-2])  # infer=Yus → Y
        if feat_ms >= 0: feat_times.append(feat_ms)
        if infer_us >= 0: infer_times.append(infer_us)
        correct = pred == vio.LABEL_NAMES[ytrue]
        ok += correct
        mark = 'OK' if correct else 'MISS'
        print(f'{i:3d} {L}HP  {vio.LABEL_NAMES[ytrue]:8s} {pred:8s} {conf:>6}  {mark}')
        if plot_on:
            line_orig.set_ydata(win)
            line_degr.set_ydata(win_to_send)
            ax_orig.relim(); ax_orig.autoscale_view()
            ax_degr.relim(); ax_degr.autoscale_view()
            held = ' (held-out)' if L not in TRAIN_LOADS else ''
            ax_orig.set_title(f'工况 {L}HP{held} | 真实 {vio.LABEL_NAMES[ytrue]} | '
                              f'板子 {pred} | conf {conf} | {"OK" if correct else "MISS"}',
                              color='green' if correct else 'red', fontsize=11, fontweight='bold')
            ax_degr.set_title('')
            plt.pause(0.001)
    t_elapsed = time.time() - t_stream_start
    loads_used = sorted(set(L for _, _, L, _ in playlist))
    received = len(feat_times)              # 实际收到响应的窗数
    dropped = len(playlist) - received
    if received > 0:
        print(f'\n流式结果: {ok}/{received} 正确 ({ok/received*100:.1f}%)')
        if dropped > 0:
            print(f'丢包: {dropped}/{len(playlist)} 窗无响应 (不计入精度)')
        avg_feat = np.mean(feat_times)
        avg_infer = np.mean(infer_times)
        print(f'PC 总耗时: {t_elapsed:.1f}s  ({len(playlist)} 窗, {t_elapsed/len(playlist)*1000:.0f}ms/窗)')
        print(f'板子每窗: 特征 {avg_feat:.1f}ms + 推理 {avg_infer:.0f}us')
        print(f'训练[0,3]HP的模型, 流式诊断{loads_used}HP, '
              f'{"泛化成立" if ok/received>0.95 else "泛化不够"}')
    else:
        print('\n未收到任何诊断结果, 请确认板子固件已更新到 v1.5.')
    ser.close()


if __name__ == '__main__':
    main()
