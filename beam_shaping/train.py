"""高斯光束整形成圆环。在 IDLE 打开本文件，按 F5 运行。"""
import json
import math
import time
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 长度均以米为单位。默认三层，每层之后自由传播 40 mm。
LAYERS_TO_TRAIN = (3,)       # 层数对照改为 (1, 2, 3)，总长保持不变
SEED = 42                     # 每次 F5 只使用这个种子。换种子时修改此处。
ITERATIONS = 20000
LEARNING_RATE = 0.001
LOG_EVERY = 10               # 权重文件中训练历史的记录间隔
PRINT_EVERY = 100            # IDLE 控制台打印间隔
SAMPLES_PER_PIXEL = 2
WINDOW_FACTOR = 2            # 零填充窗口宽度 / 相位板宽度
CHECK_CONVERGENCE = True
PIXELS = 256
PITCH = 16e-6
WAVELENGTH = 640e-9
TOTAL_DISTANCE = 120e-3
WAIST = 0.35e-3
RADIUS = 0.45e-3
WIDTH = 0.06e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent


class OpticalSystem(nn.Module):
    def __init__(self, layers, samples, window):
        super().__init__()
        self.samples = samples
        self.dx = PITCH / samples
        active = PIXELS * samples
        n = active * window
        self.margin = (n - active) // 2
        self.phase = nn.Parameter(torch.empty(layers, PIXELS, PIXELS))
        nn.init.uniform_(self.phase, -math.pi, math.pi)
        axis = (torch.arange(n, dtype=torch.float64) - (n-1)/2) * self.dx
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        r = torch.hypot(x, y)
        target = torch.exp(-0.5*((r-RADIUS)/WIDTH)**2)
        self.register_buffer("x", x.float())
        self.register_buffer("y", y.float())
        self.register_buffer("target", (target/target.sum()).float())
        self.register_buffer("ring", (r-RADIUS).abs() <= 2*WIDTH)

        # 频域相位导数对应的横向位移须小于半个计算窗口。
        # 两轴分别检查，排除倏逝波。双精度计算核后转为 complex64。
        z = TOTAL_DISTANCE/layers
        freq = torch.fft.fftfreq(n, self.dx, dtype=torch.float64)
        fy, fx = torch.meshgrid(freq, freq, indexing="ij")
        kz = (WAVELENGTH**-2-fx.square()-fy.square()).clamp_min(0).sqrt()
        half_window = n*self.dx/2
        valid = (kz > 0) & (z*fx.abs() <= half_window*kz) & (z*fy.abs() <= half_window*kz)
        transfer = torch.exp(2j*math.pi*z*kz)*valid
        self.register_buffer("transfer", transfer.to(torch.complex64))

    def input_field(self, shift_x=0., shift_y=0.):
        field = torch.exp(-((self.x-shift_x).square()+(self.y-shift_y).square())/WAIST**2)
        # 离散场包含 sqrt(采样面积)，因此 sum(|u|²) 表示功率，入射功率为 1。
        return field / torch.linalg.vector_norm(field)

    def forward(self, shift_x=0., shift_y=0., levels=None):
        field = self.input_field(shift_x, shift_y)
        m = self.margin
        for phase in self.phase:
            if levels:
                step = 2*math.pi/levels
                phase = torch.round(phase/step)*step
            phi = phase.repeat_interleave(self.samples, 0).repeat_interleave(self.samples, 1)
            # 每层固定方形口径；末端屏幕保留整个补零窗口。
            size = phi.shape[0]
            active = field[m:m+size,m:m+size] * torch.exp(1j*phi)
            field = torch.nn.functional.pad(active, (m,m,m,m))
            field = torch.fft.ifft2(torch.fft.fft2(field)*self.transfer)
        return field.abs().square()

    def metrics(self, intensity):
        power = intensity.sum()
        p = intensity / power.clamp_min(1e-20)
        shape = (p-self.target).square().sum()/self.target.square().sum().clamp_min(1e-20)
        eta = intensity[self.ring].sum()
        return shape + 0.1*(1-eta), shape, eta, power


def numbers(metrics):
    return torch.stack(metrics).detach().cpu().tolist()


def optimize(layers, seed):
    torch.manual_seed(seed)
    model = OpticalSystem(layers, SAMPLES_PER_PIXEL, WINDOW_FACTOR).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []
    started = time.perf_counter()
    previous = started
    last_print = started
    for step in range(ITERATIONS+1):
        intensity = model()
        metrics = model.metrics(intensity)
        record_now = step % LOG_EVERY == 0 or step == ITERATIONS
        print_now = step % PRINT_EVERY == 0 or step == ITERATIONS
        if record_now or print_now:
            loss, shape, eta, power = numbers(metrics)
            if not math.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            now = time.perf_counter()
            if record_now:
                history.append([step,loss,shape,eta,power,now-previous,now-started])
                previous = now
            if print_now:
                print(f"{layers} 层 seed={seed} | {step}/{ITERATIONS} | loss {loss:.5f} | 入环效率 {eta:.2%} | 本段 {now-last_print:.2f}s | 总计 {now-started:.1f}s",flush=True)
                last_print = now
        if step < ITERATIONS:
            optimizer.zero_grad(set_to_none=True)
            metrics[0].backward()
            optimizer.step()
    phase = model.phase.detach().cpu().numpy().copy()
    return phase, history


def plot_plates(phase, path):
    layers = len(phase)
    fig, axes = plt.subplots(1, layers, squeeze=False, figsize=(3.6*layers+1, 3.6),
                             layout="constrained")
    edge = phase.shape[-1]*PITCH*1000/2
    for i, ax in enumerate(axes[0]):
        image = ax.imshow(np.mod(phase[i], 2*np.pi), cmap="twilight_shifted",
                          vmin=0, vmax=2*np.pi, origin="lower", interpolation="nearest",
                          extent=[-edge,edge,-edge,edge])
        ax.set(title=f"Phase plate {i+1}", xlabel="x (mm)", ylabel="y (mm)")
    fig.colorbar(image, ax=list(axes[0]), label="Phase (rad)",
                 ticks=[0,np.pi,2*np.pi]).ax.set_yticklabels(["0","π","2π"])
    fig.savefig(path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def export(layers, phase, history, config, folder, prefix):
    model = OpticalSystem(layers,SAMPLES_PER_PIXEL,WINDOW_FACTOR).to(DEVICE)
    model.phase.copy_(torch.from_numpy(phase))
    intensity = model()
    axis = model.x[0].cpu().numpy()
    extent = [1000*(axis[0]-model.dx/2),1000*(axis[-1]+model.dx/2)]*2
    limit_mm = 1.2*RADIUS*1000  # 全轴跨度 = 1.2 × 圆环直径
    fig, axes = plt.subplots(1,2,figsize=(9,4),layout="constrained")
    for ax, data, title in zip(axes,[model.target,intensity],["Target","Output"]):
        ax.imshow((data/data.max()).cpu(),origin="lower",extent=extent,cmap="inferno",vmin=0,vmax=1)
        ax.set(xlabel="x (mm)",ylabel="y (mm)",title=title,
               xlim=(-limit_mm,limit_mm), ylim=(-limit_mm,limit_mm), aspect="equal")
    fig.savefig(folder/(prefix+"ring.png"),dpi=160)
    plt.close(fig)
    plot_plates(phase, folder/(prefix+"phase_plates.png"))

    checks = {}
    for label, sx, sy, levels in [("nominal",0,0,None),("16_levels",0,0,16),
                                  ("x_plus",20e-6,0,None),("x_minus",-20e-6,0,None),
                                  ("y_plus",0,20e-6,None),("y_minus",0,-20e-6,None)]:
        checks[label] = numbers(model.metrics(model(sx,sy,levels)))
    if CHECK_CONVERGENCE:
        for label,samples,window in [("finer_grid",2*SAMPLES_PER_PIXEL,WINDOW_FACTOR),
                                     ("larger_window",SAMPLES_PER_PIXEL,WINDOW_FACTOR+1)]:
            check = OpticalSystem(layers,samples,window).to(DEVICE)
            check.phase.copy_(model.phase)
            checks[label] = numbers(check.metrics(check()))
            del check
    np.savez_compressed(folder/(prefix+"weights.npz"), phase_rad=phase, config=json.dumps(config),
                        intensity=intensity.cpu().numpy(), x_m=axis, history=history,
                        history_columns="step loss shape_loss ring_efficiency screen_power interval_s total_s",
                        checks=json.dumps(checks), check_columns="loss shape_loss ring_efficiency screen_power")


def main():
    out = ROOT/"output"
    out.mkdir(exist_ok=True)
    config = {k:v for k,v in globals().items() if k.isupper() and isinstance(v,(tuple,int,float))}
    config["device"] = str(DEVICE)
    print(f"设备 {DEVICE} | 输出 {out}",flush=True)
    # 完成训练和出图后再替换上一轮的文件。
    with TemporaryDirectory(dir=out) as tmp:
        stage = Path(tmp)
        for layers in LAYERS_TO_TRAIN:
            prefix = "" if len(LAYERS_TO_TRAIN)==1 else f"layers_{layers}_"
            phase, history = optimize(layers, SEED)
            export(layers, phase, history, config, stage, prefix)
        ready = {file.name for file in stage.iterdir()}
        for file in stage.iterdir():
            file.replace(out/file.name)
        # 上一轮可选的层数对照和复算图已不对应本轮权重。
        import re
        owned = r"(layers_[0-9]+_)?(weights\.npz|phase_plates\.png|ring\.png|measurements\.csv|propagation\.png|radial_profiles\.png)"
        for file in out.iterdir():
            if file.is_file() and file.name not in ready and re.fullmatch(owned,file.name):
                file.unlink()
    print(f"训练完成，seed={SEED}。结果已保存到 {out}",flush=True)


if __name__ == "__main__":
    main()
