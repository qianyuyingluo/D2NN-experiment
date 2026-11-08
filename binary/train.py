"""单层二值振幅全息图。IDLE 按 F5；依赖 numpy、Pillow、matplotlib。
透明片放在薄透镜附近，准直高斯光正入射，屏在透镜后焦面。
模型假设黑格不透光、白格完全透光，照明波前平坦。
"""
from pathlib import Path
from time import perf_counter
from datetime import datetime
import numpy as np
from PIL import Image, ImageOps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 在这里改参数；所有长度单位为 mm ----
JPEG_FILE = ""            # 留空：自动读同目录唯一的 JPEG，否则弹出选择框
DARK_FOREGROUND = True    # 白底黑剪影 True，黑底亮目标 False
THRESHOLD = 128
GRID = 96                 # 可打印格数，每边 96 格
PITCH = 0.10              # 每格 0.10 mm，图案总宽 9.60 mm
WAVELENGTH = 0.000640
FOCAL_LENGTH = 200.0
BEAM_RADIUS = 2.5         # 胶片处 1/e² 光强半径，对应 5 mm 直径
CENTER_X = 0.32           # 目标放在光轴右侧，避开零级
TARGET_BOX = 0.40         # 等比例放入此宽高的正方形，保持图案朝向
FFT_SIZE = 256            # 补零细分焦面采样，不增加器件自由度
EPOCHS = 3000
LR = 0.035
EFF_WEIGHT = 0.7
SEED = 42
ROOT = Path(__file__).resolve().parent


def choose_jpeg():
    if JPEG_FILE:
        return ROOT / JPEG_FILE
    files = sorted(p for p in ROOT.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    if len(files) == 1:
        return files[0]
    import tkinter as tk
    from tkinter import filedialog
    window = tk.Tk()
    window.withdraw()
    try:
        name = filedialog.askopenfilename(parent=window, initialdir=ROOT,
                    title="选择目标 JPEG", filetypes=[("JPEG", "*.jpg *.jpeg *.JPG *.JPEG")])
    finally:
        window.destroy()
    return Path(name) if name else None


def target_roi(path, spacing):
    with Image.open(path) as im:
        gray = np.asarray(ImageOps.exif_transpose(im).convert("L"))
    fg = gray < THRESHOLD if DARK_FOREGROUND else gray >= THRESHOLD
    if not fg.any() or fg.all():
        raise ValueError("阈值未分出前景和背景，请检查 THRESHOLD 和 DARK_FOREGROUND。")
    y, x = np.where(fg)
    silhouette = Image.fromarray(fg[y.min():y.max()+1, x.min():x.max()+1].astype("uint8")*255)
    side = round(TARGET_BOX / spacing)
    if side < 3:
        raise ValueError("目标不足三个焦面采样点，请检查尺寸。")
    scale = side / max(silhouette.size)
    w, h = (max(1, round(v*scale)) for v in silhouette.size)
    small = np.asarray(silhouette.resize((w, h), Image.Resampling.BOX))/255
    # 图像首行在上；物理坐标数组 y 随行号递增。
    q = np.zeros((side, side))
    j, i = (side-h)//2, (side-w)//2
    q[j:j+h, i:i+w] = small[::-1]
    if q.sum() == 0:
        raise ValueError("目标在焦面网格上消失，请扩大图案。")
    q /= q.sum()
    mid = FFT_SIZE//2
    left = mid + round(CENTER_X/spacing) - side//2
    if left <= mid or left+side > FFT_SIZE or side > FFT_SIZE:
        raise ValueError("目标窗口应在正半轴且不越过主频带。")
    return (slice(mid-side//2, mid-side//2+side), slice(left,left+side)), q


def main():
    path = choose_jpeg()
    if path is None:
        print("已取消选择。")
        return
    if FFT_SIZE < GRID or min(EPOCHS, PITCH, WAVELENGTH, FOCAL_LENGTH, BEAM_RADIUS) <= 0:
        raise ValueError("请检查网格、光学参数和迭代次数。")
    n = FFT_SIZE
    dx = WAVELENGTH * FOCAL_LENGTH / (n*PITCH)
    roi, target = target_roi(path, dx)
    region = target >= 0.5 * target.max()
    coord = (np.arange(GRID)-(GRID-1)/2)*PITCH
    beam = np.exp(-(coord[:,None]**2+coord[None,:]**2)/BEAM_RADIUS**2)
    # 入射功率以整个有限板面上的入射场为基准；默认口径截断很小。
    beam /= np.linalg.norm(beam)
    freq = np.fft.fftshift(np.fft.fftfreq(n))
    envelope = np.sinc(freq[:,None])*np.sinc(freq[None,:])  # 方格像素的振幅包络
    k = (n-GRID)//2
    plate = (slice(k,k+GRID), slice(k,k+GRID))
    buffer = np.zeros((n,n))
    q2 = np.sum(target**2)

    def forward(mask):
        buffer[plate] = beam*mask
        return envelope*np.fft.fftshift(np.fft.fft2(buffer, norm="ortho"))

    def objective(field):
        intensity = abs(field[roi])**2
        total = intensity.sum() + 1e-18
        p = intensity/total
        error = p-target
        eta = intensity[region].sum()
        loss = np.sum(error**2)/q2 - EFF_WEIGHT*eta
        # d(loss)/d(I)，包含 ROI 内的形状归一化。
        slope = 2*(error-np.sum(error*p))/(q2*total) - EFF_WEIGHT*region
        return loss, eta, slope

    rng = np.random.default_rng(SEED)
    logits = rng.normal(0, 0.3, (GRID,GRID))
    m, v = np.zeros_like(logits), np.zeros_like(logits)
    best_loss = np.inf
    history = []
    start = perf_counter()
    for epoch in range(1, EPOCHS+1):
        soft = 1/(1+np.exp(-np.clip(logits,-20,20)))
        hard = soft >= 0.5
        hard_field = forward(hard)
        loss, eta, slope = objective(hard_field)
        history.append(eta)
        if loss < best_loss:
            best_loss, best_epoch, best = loss, epoch, hard.copy()
        # 前 1/3 轮用连续掩模预热，之后前向用真实二值板；
        # 反向用 sigmoid 的导数近似硬阈值的导数。
        field = hard_field
        if epoch <= EPOCHS//3:
            field = forward(soft)
            slope = objective(field)[2]
        adjoint = np.zeros_like(field)
        adjoint[roi] = 2*field[roi]*slope*envelope[roi]
        grad_mask = beam*np.fft.ifft2(np.fft.ifftshift(adjoint),norm="ortho").real[plate]
        grad = grad_mask*soft*(1-soft)
        m = 0.9*m + 0.1*grad
        v = 0.999*v + 0.001*grad**2
        logits -= LR*(m/(1-0.9**epoch))/(np.sqrt(v/(1-0.999**epoch))+1e-8)
        if epoch % 100 == 0 or epoch in (1,EPOCHS):
            print(f"epoch {epoch:5d} | binary loss {loss:.6f} | eta {eta:.2%} | {perf_counter()-start:.1f} s",flush=True)

    out = ROOT / ("hologram_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir()
    # PNG 每格放大为纯色像素，不做灰度插值。数组首行对应物理 y 最大。
    mask_img = Image.fromarray(best[::-1].astype("uint8")*255)
    dots = 8
    mask_img.resize((GRID*dots,GRID*dots),Image.Resampling.NEAREST).convert("1").save(
        out/"binary_mask.png",dpi=(25.4*dots/PITCH,)*2)
    full = abs(forward(best))**2
    lo, hi = (-n//2-0.5)*dx,(n-n//2-0.5)*dx
    sy,sx = roi
    bounds = [(sx.start-n//2-0.5)*dx,(sx.stop-n//2-0.5)*dx,
              (sy.start-n//2-0.5)*dx,(sy.stop-n//2-0.5)*dx]
    fig, ax = plt.subplots(1,3,figsize=(12,4),layout="constrained")
    for a, data, extent, title in zip(ax,(target,full[roi],full/full.max()),
                    (bounds,bounds,[lo,hi,lo,hi]),("Target (relative)","Binary output (local scale)","Central band (linear)")):
        pic=a.imshow(data,origin="lower",extent=extent,cmap="gray",
                     **({"vmin":0,"vmax":1} if a is ax[2] else {}))
        a.set(title=title,xlabel="x (mm)",ylabel="y (mm)")
        if a is ax[2]:
            fig.colorbar(pic,ax=a,label="Intensity / central peak")
    fig.suptitle(f"Best binary plate: epoch {best_epoch}, target / incident power = {history[best_epoch-1]:.2%}")
    fig.savefig(out/"result.png",dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot(np.arange(1,EPOCHS+1),100*np.asarray(history))
    ax.set(xlabel="Epoch",ylabel="Target power / incident power (%)",title="Binary plate efficiency")
    ax.grid(alpha=0.25)
    fig.savefig(out/"efficiency.png",dpi=160,bbox_inches="tight")
    plt.close(fig)
    print(f"已保存到 {out}\n黑白板实际边长 {GRID*PITCH:g} mm；白色透明，黑色遮光。")


if __name__ == "__main__":
    main()
