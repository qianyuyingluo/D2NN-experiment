"""高斯光束 → 多层相位板 → 图片光强。在 IDLE 中按 F5 运行。"""
import json
import math
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn
from PIL import Image, ImageOps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

TARGET_FILE = ROOT / "target.jpg"
BRIGHT_SILHOUETTE = True   # True: 原图黑色处发光；False: 原图白色处发光
TARGET_LONG_SIDE_MM = 1.0  # 整张图片在屏幕上的长边，保持宽高比
# 剪影网络的独立设置。所有物理长度均以米为单位（带 _MM 的参数除外）。
LAYERS = 4                 # 相位板数量
PIXELS = 256               # 每块相位板的分辨率：PIXELS × PIXELS
PITCH = 16e-6              # 相位像素宽度
WAVELENGTH = 640e-9
TOTAL_DISTANCE = 120e-3    # 平均分配到各层之后的传播段
WAIST = 0.35e-3
SAMPLES_PER_PIXEL = 2      #训练采样
WINDOW_FACTOR = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
ITERATIONS = 50000
LEARNING_RATE = 0.001
PRINT_EVERY = 100
CHECK_FINER_GRID = True
OUTPUT = ROOT / "output"


class SilhouetteD2NN(nn.Module):
    """相位板调制、带限角谱传播、屏幕光强读出。"""
    def __init__(self, layers, samples, window):
        super().__init__()
        self.samples = samples
        self.dx = PITCH / samples
        self.active = PIXELS * samples
        n = self.active * window
        self.margin = (n - self.active) // 2
        self.phase = nn.Parameter(torch.empty(layers, PIXELS, PIXELS))
        nn.init.uniform_(self.phase, -math.pi, math.pi)

        axis = (torch.arange(n, dtype=torch.float64) - (n-1)/2) * self.dx
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("x", x.float())
        self.register_buffer("y", y.float())
        gaussian = torch.exp(-(self.x.square() + self.y.square())/WAIST**2)
        self.register_buffer("incident", gaussian / torch.linalg.vector_norm(gaussian))

        z = TOTAL_DISTANCE/layers
        freq = torch.fft.fftfreq(n, self.dx, dtype=torch.float64)
        fy, fx = torch.meshgrid(freq, freq, indexing="ij")
        kz = (WAVELENGTH**-2 - fx.square() - fy.square()).clamp_min(0).sqrt()
        half_window = n*self.dx/2
        valid = (kz > 0) & (z*fx.abs() <= half_window*kz) & (z*fy.abs() <= half_window*kz)
        transfer = torch.exp(2j*math.pi*z*kz)*valid
        self.register_buffer("transfer", transfer.to(torch.complex64))

    def forward(self):
        field = self.incident
        lo = self.margin
        hi = lo + self.active
        far_margin = field.shape[-1] - hi
        for phase in self.phase:
            phi = phase.repeat_interleave(self.samples, 0).repeat_interleave(self.samples, 1)
            active = field[lo:hi, lo:hi] * torch.exp(1j*phi)
            field = torch.nn.functional.pad(active, (lo,far_margin,lo,far_margin))
            field = torch.fft.ifft2(torch.fft.fft2(field)*self.transfer)
        return field.abs().square()


def numbers(values):
    return torch.stack(values).detach().cpu().tolist()


def image_target(model):
    n = model.x.shape[0]
    with Image.open(TARGET_FILE) as src:
        gray = ImageOps.exif_transpose(src).convert("L")
        longest = round(TARGET_LONG_SIDE_MM * 1e-3 / model.dx)
        if not 1 <= longest <= n:
            raise ValueError("目标图长边必须落在计算窗口内，且至少覆盖一个采样点。")
        scale = longest / max(gray.size)
        size = tuple(max(1, round(d * scale)) for d in gray.size)
        gray = gray.resize(size, Image.Resampling.LANCZOS)
    # 抑制 JPEG 近黑、近白背景的微小噪声，保留轮廓的灰度过渡。
    light = np.clip((np.asarray(gray, dtype=np.float32)-5)/245, 0, 1)
    if BRIGHT_SILHOUETTE:
        light = 1-light
    # 图片行号向下增加，物理 y 轴向上增加。
    tile = torch.from_numpy(np.flipud(light).copy()).to(model.x.device)
    target = torch.zeros_like(model.x)
    height, width = tile.shape
    row, col = (n-height)//2, (n-width)//2
    target[row:row+height, col:col+width] = tile
    mask = target >= 0.5
    if not mask.any():
        raise ValueError("图片没有目标亮区，请检查 TARGET_FILE 和 BRIGHT_SILHOUETTE。")
    return target/target.sum(), mask


def metrics(intensity, target, mask):
    power = intensity.sum()
    normalized = intensity/power.clamp_min(1e-20)
    shape = (normalized-target).square().sum()/target.square().sum()
    efficiency = intensity[mask].sum()  # 入射总功率已归一化到 1
    return shape + 0.1*(1-efficiency), shape, efficiency, power


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


def plot_result(model, intensity, target, eta, destination):
    ax_m = model.x[0].detach().cpu().numpy()
    extent = [1000*(ax_m[0]-model.dx/2), 1000*(ax_m[-1]+model.dx/2)]*2
    limit = 0.6*TARGET_LONG_SIDE_MM
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    peak = target.max()
    images = (target/peak, intensity/intensity.sum().clamp_min(1e-20)/peak)
    for ax, img, title in zip(axes, images, ("Target", f"Output | efficiency {eta:.1%}")):
        artist = ax.imshow(img.detach().cpu(), extent=extent, origin="lower", cmap="inferno",
                           vmin=0, vmax=1, interpolation="nearest")
        ax.set(xlim=(-limit,limit), ylim=(-limit,limit), xlabel="x (mm)", ylabel="y (mm)", title=title)
    fig.colorbar(artist, ax=axes, extend="max", label="Normalized intensity / target peak")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    samples, window = SAMPLES_PER_PIXEL, WINDOW_FACTOR
    model = SilhouetteD2NN(LAYERS, samples, window).to(DEVICE)
    target, mask = image_target(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"设备 {DEVICE} | seed={SEED} | 目标 {TARGET_FILE.name} | 输出 {OUTPUT}", flush=True)
    history = []
    started = previous = time.perf_counter()
    for step in range(ITERATIONS+1):
        intensity = model()
        values = metrics(intensity,target,mask)
        if step % PRINT_EVERY == 0 or step == ITERATIONS:
            loss, shape, eta, power = numbers(values)
            if not np.isfinite([loss,shape,eta,power]).all():
                raise FloatingPointError(f"第 {step} 步的光强或损失异常")
            now = time.perf_counter()
            history.append((step,loss,shape,eta,power,now-previous,now-started))
            print(f"{step}/{ITERATIONS} | loss {loss:.6f} | 目标亮区效率 {eta:.2%} | "
                  f"本段 {now-previous:.1f}s | 总计 {now-started:.1f}s", flush=True)
            previous = now
        if step < ITERATIONS:
            optimizer.zero_grad(set_to_none=True)
            values[0].backward()
            optimizer.step()

    cfg = dict(seed=SEED, iterations=ITERATIONS, learning_rate=LEARNING_RATE, layers=LAYERS,
               wavelength_m=WAVELENGTH, distance_m=TOTAL_DISTANCE, waist_m=WAIST,
               plate_pixels=PIXELS, pitch_m=PITCH, samples_per_pixel=samples,
               window_factor=window, bright_silhouette=BRIGHT_SILHOUETTE,
               target_long_side_mm=TARGET_LONG_SIDE_MM, target_filename=TARGET_FILE.name)
    checks = {"training_grid": [loss,shape,eta,power]}
    with torch.no_grad():
        if CHECK_FINER_GRID:
            fine = SilhouetteD2NN(LAYERS, 2*samples, window).to(DEVICE)
            fine.phase.copy_(model.phase)
            fine_target, fine_mask = image_target(fine)
            checks["finer_grid"] = numbers(metrics(fine(), fine_target, fine_mask))
            print(f"加密网格亮区效率 {checks['finer_grid'][2]:.2%}", flush=True)
        phase = model.phase.detach().cpu().numpy()
        OUTPUT.mkdir(exist_ok=True, parents=True)
        with TemporaryDirectory(dir=OUTPUT) as tmp:
            stage = Path(tmp)
            plot_result(model,intensity,target,eta,stage/"result.png")
            plot_plates(phase,stage/"phase_plates.png")
            np.savez_compressed(stage/"weights.npz", phase_rad=phase, config=json.dumps(cfg),
                                intensity=intensity.detach().cpu().numpy(), target=target.cpu().numpy(),
                                x_m=model.x[0].cpu().numpy(), history=history,
                                history_columns="step loss shape_loss efficiency screen_power interval_s total_s",
                                checks=json.dumps(checks), check_columns="loss shape_loss efficiency screen_power")
            for file in stage.iterdir():
                file.replace(OUTPUT/file.name)
    print(f"训练完成，结果在 {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
