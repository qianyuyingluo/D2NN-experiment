"""双圆孔单图反演。用 IDLE 打开本文件后按 F5。长度单位 mm，角度单位度。"""
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "output" / "screen_data.npz"
OUTPUT = ROOT / "double_hole_inverse"
WAVELENGTH = 0.000640
FOCAL_LENGTH = 200.0
INCIDENT_IRRADIANCE = 1.0   # 与实验一致，mW/mm²
INIT_DIAMETER = 0.27        # 初始猜测，不读取生成图像时的孔径参数
INIT_SEPARATION = 0.60
INIT_ANGLE_DEG = 12.0
EPOCHS = 1500
LR = 0.012
PRINT_EVERY = 100
STRIDE = 3                 # 训练时规则抽样，全分辨率用于最终输出
LOSS_MODE = "sqrt"          # sqrt: 平方根强度损失；linear: 线性强度损失
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AiryAmplitude(torch.autograd.Function):
    """圆孔的实振幅 2 J1(t)/t；手写解析导数支持几何参数反传。"""
    @staticmethod
    def forward(ctx, t):
        small = t.abs() < 0.1
        safe = torch.where(small, 1.0, t)
        value = torch.where(small, 1-t.square()/8+t.pow(4)/192,
                            2*torch.special.bessel_j1(safe)/safe)
        ctx.save_for_backward(t, value)
        return value

    @staticmethod
    def backward(ctx, grad):
        t, value = ctx.saved_tensors
        small = t.abs() < 0.1
        safe = torch.where(small, 1.0, t)
        derivative = torch.where(small, -t/4+t.pow(3)/48,
                                 2*(torch.special.bessel_j0(safe)-value)/safe)
        return grad*derivative


class DoubleHole(nn.Module):
    """两个等大、互不重叠的圆孔，中心的中点固定在原点。"""
    def __init__(self):
        super().__init__()
        if not 0 < INIT_DIAMETER < INIT_SEPARATION:
            raise ValueError("初始中心距须大于孔径，且孔径为正。")
        self.log_diameter = nn.Parameter(torch.tensor(math.log(INIT_DIAMETER)))
        self.log_gap = nn.Parameter(torch.tensor(math.log(INIT_SEPARATION-INIT_DIAMETER)))
        self.angle = nn.Parameter(torch.tensor(math.radians(INIT_ANGLE_DEG)))

    def geometry(self):
        diameter = self.log_diameter.exp()
        return diameter, diameter+self.log_gap.exp(), self.angle

    def forward(self, u, v, radial_frequency):
        diameter, separation, angle = self.geometry()
        disk = AiryAmplitude.apply(math.pi*diameter*radial_frequency)
        projected_frequency = u*angle.cos()+v*angle.sin()
        amplitude = (math.pi*diameter.square()/2)*disk*torch.cos(math.pi*separation*projected_frequency)
        # 光屏处先叠加振幅，再取模平方，得到线性光强。
        return INCIDENT_IRRADIANCE*(amplitude/(WAVELENGTH*FOCAL_LENGTH)).square()


def grid(x, y):
    v, u = torch.meshgrid(y/(WAVELENGTH*FOCAL_LENGTH), x/(WAVELENGTH*FOCAL_LENGTH), indexing="ij")
    return u, v, torch.hypot(u,v)


def main():
    if LOSS_MODE not in ("sqrt","linear") or EPOCHS < 1 or STRIDE < 1:
        raise ValueError("检查训练参数")
    if not INPUT.is_file():
        raise FileNotFoundError(f"请先运行 simulate.py 生成 {INPUT}")
    with np.load(INPUT) as archive:
        measured = archive["double_hole"]
        x_mm, y_mm = archive["x_mm"], archive["y_mm"]
    if measured.shape != (len(y_mm),len(x_mm)) or not np.isfinite(measured).all() or measured.min()<0 or measured.max()<=0:
        raise ValueError("光强图或坐标不符合要求")
    x, y, obs = [torch.as_tensor(a,device=DEVICE,dtype=torch.float32) for a in (x_mm,y_mm,measured)]
    peak = obs.max()
    g = grid(x[::STRIDE],y[::STRIDE])
    target = obs[::STRIDE,::STRIDE]/peak
    model = DoubleHole().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),lr=LR)
    history = []
    def transform(i):
        return (i+1e-8).sqrt() if LOSS_MODE=="sqrt" else i
    target_loss = transform(target)
    denominator = target_loss.square().mean()
    start = previous = time.perf_counter()
    print(f"设备 {DEVICE} | 输入 {INPUT} | 损失 {LOSS_MODE}",flush=True)
    for step in range(EPOCHS+1):
        pred = model(*g)/peak
        loss = (transform(pred)-target_loss).square().mean()/denominator
        if step % PRINT_EVERY == 0 or step == EPOCHS:
            diameter,separation,angle = [float(a.detach()) for a in model.geometry()]
            angle_deg = (math.degrees(angle)+90)%180-90
            value=loss.item()
            if not math.isfinite(value):
                raise FloatingPointError(f"第 {step} 步出现非有限损失")
            now=time.perf_counter()
            history.append((step,value,diameter,separation,angle_deg,now-start))
            print(f"{step}/{EPOCHS} | loss {value:.6g} | 孔径 {diameter:.5f} mm | 中心距 {separation:.5f} mm | 角度 {angle_deg:.3f}° | 本段 {now-previous:.2f}s | 总计 {now-start:.2f}s",flush=True)
            previous=now
        if step<EPOCHS:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        fitted=model(*grid(x,y)).cpu().numpy()
    error=float(np.linalg.norm(fitted-measured)/np.linalg.norm(measured))
    OUTPUT.mkdir(exist_ok=True)
    params=dict(diameter_mm=diameter,separation_mm=separation,angle_deg_mod_180=angle_deg,
                relative_linear_intensity_L2=error,loss_mode=LOSS_MODE)
    (OUTPUT/"geometry.json").write_text(json.dumps(params,indent=2),encoding="utf-8")
    np.savetxt(OUTPUT/"history.csv",history,delimiter=",",header="step,loss,diameter_mm,separation_mm,angle_deg,elapsed_s",comments="")
    np.savez_compressed(OUTPUT/"fit.npz",intensity=fitted,x_mm=x_mm,y_mm=y_mm)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout="constrained")
    dx,dy=np.diff(x_mm).mean(),np.diff(y_mm).mean()
    extent=[x_mm[0]-dx/2,x_mm[-1]+dx/2,y_mm[0]-dy/2,y_mm[-1]+dy/2]
    for ax,I,title in zip(axes[:2],(measured,fitted),("Input screen","Fitted screen")):
        db=10*np.log10(np.maximum(I/measured.max(),1e-4))
        im=ax.imshow(db,origin="lower",extent=extent,cmap="inferno",vmin=-40,vmax=0)
        ax.set(title=title,xlabel="x (mm)",ylabel="y (mm)")
    fig.colorbar(im,ax=list(axes[:2]),label="dB",shrink=.75)
    ax=axes[2]; ax.set_facecolor("black")
    for sign in (-1,1):
        offset=sign*separation/2
        center=(offset*math.cos(angle),offset*math.sin(angle))
        ax.add_patch(plt.Circle(center,diameter/2,color="white"))
    limit=.65*(diameter+separation)
    ax.set(xlim=(-limit,limit),ylim=(-limit,limit),aspect="equal",title="Recovered aperture",xlabel="x (mm)",ylabel="y (mm)")
    fig.savefig(OUTPUT/"reconstruction.png",dpi=170)
    plt.close(fig)
    print(f"全图线性光强相对误差 {error:.5g} | 输出 {OUTPUT}")


if __name__=="__main__":
    main()
