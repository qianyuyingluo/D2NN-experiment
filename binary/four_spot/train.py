"""IDLE F5：在透明/遮光两种状态间翻转格子，优化四个衍射级。"""
from pathlib import Path
import json
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"trained_output"
WAVELENGTH = 0.000640     # mm
FOCAL_LENGTH = 200.0
BEAM_RADIUS = 2.5
PERIOD = 0.254            # 与原棋盘的光栅周期一致
REPEATS = 38              # 有效边长 9.652 mm
SIDE = 4                  # 周期单元为 4×4 格，最小格 0.0635 mm
ROI_RADIUS = 0.08
INPUT_MW = 2.0
SEED = 42
RESTARTS = 48
ZERO_WEIGHT = 0.65        # 增大则更重视减弱中央亮点
BALANCE_WEIGHT = 1.0
ORDERS = [(0,0),(-1,-1),(-1,1),(1,-1),(1,1)]


def propagation(samples):
    """用单元中各个格子的线性场响应计算积分功率；保留有限板面和高斯照明。"""
    cells = SIDE*REPEATS
    pitch = PERIOD/SIDE
    dx = pitch/samples
    count = cells*samples
    x = (np.arange(count)-(count-1)/2)*dx
    beam = np.exp(-(x/BEAM_RADIUS)**2)
    beam /= np.linalg.norm(beam)
    n = 8*count
    freq = np.fft.fftshift(np.fft.fftfreq(n,dx))
    order_sep = WAVELENGTH*FOCAL_LENGTH/PERIOD
    axis = freq*WAVELENGTH*FOCAL_LENGTH
    keep = abs(axis)<1.6*order_sep
    parts = []
    for site in range(SIDE):
        opened = np.repeat(np.arange(cells)%SIDE==site,samples)
        field = np.fft.fftshift(np.fft.fft(beam*opened,n=n,norm="ortho"))
        parts.append((field*np.sinc(freq*dx))[keep])
    axis = axis[keep]
    parts = np.asarray(parts)
    grams = []
    for ox,oy in ORDERS:
        y,x = np.where((axis[None,:]-ox*order_sep)**2+(axis[:,None]-oy*order_sep)**2<=ROI_RADIUS**2)
        basis = (parts[:,y][:,None,:]*parts[:,x][None,:,:]).reshape(SIDE**2,-1)
        grams.append((basis.conj()@basis.T).real)
    return np.stack(grams),axis,parts


def powers(mask, grams):
    return np.einsum("i,kij,j->k",mask,grams,mask)


def score(mask, grams):
    zero, *spots = powers(mask, grams)
    return sum(spots)-ZERO_WEIGHT*zero-BALANCE_WEIGHT*(max(spots)-min(spots))


def train(grams, baseline):
    # 硬二值直接搜索：每次接受得分提升最大的单格翻转。
    rng = np.random.default_rng(SEED)
    best = baseline.copy()
    for attempt in range(RESTARTS):
        mask = baseline.copy() if attempt == 0 else rng.integers(0,2,SIDE**2)
        current = score(mask, grams)
        while True:
            candidates = np.tile(mask,(len(mask),1))
            diag = np.diag_indices(len(mask))
            candidates[diag] = 1-candidates[diag]
            scores = [score(candidate,grams) for candidate in candidates]
            idx = np.argmax(scores)
            if scores[idx] <= current+1e-12:
                break
            mask,current = candidates[idx].copy(),scores[idx]
        if current > score(best,grams)+1e-12:
            best = mask.copy()
    return best


def export(mask):
    pattern = np.tile(mask.reshape(SIDE,SIDE),(REPEATS,REPEATS))[::-1]
    cells = pattern.shape[0]
    width = PERIOD*REPEATS
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}mm" height="{width:g}mm" viewBox="0 0 {cells} {cells}" shape-rendering="crispEdges">',
           f'<rect width="{cells}" height="{cells}" fill="white"/>']
    for y,x in zip(*np.where(pattern==0)):
        svg.append(f'<rect x="{x}" y="{y}" width="1" height="1"/>')
    svg.append("</svg>")
    (OUT/"mask_1to1.svg").write_text("\n".join(svg),encoding="utf-8")
    image = Image.fromarray(pattern.astype("uint8")*255)
    dots=3  # 默认格宽恰好为 1200 dpi 的 3 个打印点
    image.resize((cells*dots,)*2,Image.Resampling.NEAREST).convert("1").save(
        OUT/"mask_1to1.png",dpi=(25.4*dots/(PERIOD/SIDE),)*2)


def main():
    if SIDE%2 or ZERO_WEIGHT < 0 or BALANCE_WEIGHT < 0:
        raise ValueError("SIDE 必须为偶数，损失权重必须非负。")
    OUT.mkdir(exist_ok=True)
    yy,xx=np.indices((SIDE,SIDE))
    checker=((yy//(SIDE//2)+xx//(SIDE//2))%2==0).astype(int).ravel()
    coarse,_,_ = propagation(2)
    mask = train(coarse,checker)
    fine,axis,parts = propagation(8)
    export(mask)
    rows=[]
    maps=[]
    for label,design in [("Checkerboard",checker),("Trained",mask)]:
        ps=powers(design,fine)
        print(label, "中央与四点效率(%)",np.round(100*ps,4),flush=True)
        rows.extend([[label,ox,oy,p,p*1000*INPUT_MW] for (ox,oy),p in zip(ORDERS,ps)])
        field=parts.T@design.reshape(SIDE,SIDE)@parts
        maps.append(abs(field)**2)
    with (OUT/"power.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["mask","order_x","order_y","incident_fraction","power_uW"])
        writer.writerows(rows)
    (OUT/"design.json").write_text(json.dumps(dict(unit_cell_bottom_to_top=mask.reshape(SIDE,SIDE).tolist(),
        period_mm=PERIOD, repeats=REPEATS, wavelength_mm=WAVELENGTH,focal_length_mm=FOCAL_LENGTH,
        beam_radius_mm=BEAM_RADIUS,roi_radius_mm=ROI_RADIUS,zero_weight=ZERO_WEIGHT,
        balance_weight=BALANCE_WEIGHT,seed=SEED,restarts=RESTARTS),indent=2))
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout="constrained")
    vmax=maps[0].max()
    edge=(axis[-1]-axis[-2])/2
    for ax,data,title in zip(axes,maps,["Checkerboard","Trained binary mask"]):
        im=ax.imshow(data/vmax,cmap="inferno",origin="lower",vmin=0,vmax=1,interpolation="nearest",
                     extent=[axis[0]-edge,axis[-1]+edge]*2)
        ax.set(title=title,xlabel="x (mm)",ylabel="y (mm)")
    fig.colorbar(im,ax=list(axes),shrink=.8,label="Linear intensity / checkerboard central peak")
    fig.savefig(OUT/"comparison.png",dpi=180)
    plt.close(fig)
    print(OUT)


if __name__=="__main__":
    main()
