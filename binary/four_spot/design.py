"""黑白四点分光板。IDLE F5，生成可打印掩模、焦面图和功率表。长度单位 mm。"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

TILE = 0.127              # 棋盘格边长；600 dpi 下恰好 3 个打印点
CELLS = 76                # 偶数，开口面积占一半
WAVELENGTH = 0.000640
FOCAL_LENGTH = 200.0
BEAM_RADIUS = 2.5
INPUT_MW = 2.0            # 到达有效板面的功率
ROI_RADIUS = 0.08         # 在每个斑点周围积分的半径
OUT = Path(__file__).resolve().parent / "output"


def write_mask():
    # 白色对应透明胶片上不印墨，黑色对应遮光墨层。
    row, col = np.indices((CELLS, CELLS))
    mask = (row + col) % 2 == 0
    side = CELLS*TILE
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{side:.6f}mm" height="{side:.6f}mm" viewBox="0 0 {CELLS} {CELLS}" shape-rendering="crispEdges">',
           f'<rect width="{CELLS}" height="{CELLS}" fill="white"/>']
    for y,x in zip(*np.where(~mask)):
        svg.append(f'<rect x="{x}" y="{y}" width="1" height="1"/>')
    svg.append("</svg>")
    (OUT/"mask_1to1.svg").write_text("\n".join(svg),encoding="utf-8")
    # 整数个打印点精确覆盖一格。
    dots = 6
    img = Image.fromarray(mask.astype("uint8")*255)
    img.resize((CELLS*dots,)*2,Image.Resampling.NEAREST).convert("1").save(
        OUT/"mask_1to1.png",dpi=(25.4*dots/TILE,)*2)
    return mask


def simulate(samples):
    # 二维棋盘可写为 (1 + sx*sy)/2。分离 FFT 等价于完整二维傅里叶传播。
    dx = TILE/samples
    length = CELLS*samples
    x = (np.arange(length)-(length-1)/2)*dx
    g = np.exp(-(x/BEAM_RADIUS)**2)
    g /= np.linalg.norm(g)
    sign = np.repeat((-1.)**np.arange(CELLS), samples)
    n = 8*length
    def spectrum(v):
        f = np.fft.fftshift(np.fft.fftfreq(n,dx))
        # 每个微小采样单元是方形平顶区域，包含其 sinc 振幅包络。
        return f,np.fft.fftshift(np.fft.fft(v,norm="ortho",n=n))*np.sinc(dx*f)
    f,a = spectrum(g)
    _,b = spectrum(sign*g)
    axis = f*WAVELENGTH*FOCAL_LENGTH
    order = WAVELENGTH*FOCAL_LENGTH/(2*TILE)
    keep = np.abs(axis) < 3.6*order
    axis,a,b = axis[keep],a[keep],b[keep]
    intensity = abs(0.5*(a[:,None]*a+b[:,None]*b))**2
    rows=[]
    for sx,sy in [(0,0),(-1,-1),(-1,1),(1,-1),(1,1)]:
        x0,y0=sx*order,sy*order
        region = (axis-x0)**2+(axis[:,None]-y0)**2 <= ROI_RADIUS**2
        power=float(intensity[region].sum())
        rows.append([sx,sy,x0,y0,power,1000*INPUT_MW*power])
    return axis,intensity,rows


def main():
    if CELLS%2:
        raise ValueError("CELLS 应为偶数")
    OUT.mkdir(exist_ok=True)
    mask = write_mask()
    _,_,coarse=simulate(4)
    x,intensity,rows=simulate(8)
    with (OUT/"spot_power.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["order_x","order_y","x_mm","y_mm","incident_fraction","power_uW"])
        writer.writerows(rows)
    print("每点功率占比:",[round(r[4],6) for r in rows])
    print("4 与 8 子采样的最大效率差（百分点）:",100*max(abs(a[4]-b[4]) for a,b in zip(coarse,rows)))
    order=rows[-1][2]
    fig,axes=plt.subplots(1,2,figsize=(9,4.4),layout="constrained")
    axes[0].imshow(mask[:8,:8],cmap="gray",origin="lower",interpolation="nearest",
                   extent=[0,8*TILE,0,8*TILE],vmin=0,vmax=1)
    axes[0].set(title="Mask detail (8 x 8 cells)",xlabel="x (mm)",ylabel="y (mm)")
    half_step=(x[1]-x[0])/2
    floor=x[0]-half_step
    ceil=x[-1]+half_step
    img=axes[1].imshow(intensity/intensity.max(),origin="lower",interpolation="nearest",cmap="inferno",
                       extent=[floor,ceil,floor,ceil],vmin=0,vmax=1)
    axes[1].set(xlim=(-1.5*order,1.5*order),ylim=(-1.5*order,1.5*order),
                title="Focal plane (central crop)",xlabel="x (mm)",ylabel="y (mm)")
    fig.colorbar(img,ax=axes[1],label="Intensity / central peak (linear)",shrink=.85)
    fig.savefig(OUT/"screen_linear.png",dpi=200)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
