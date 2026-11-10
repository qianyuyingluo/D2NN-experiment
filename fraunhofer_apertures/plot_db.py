"""在 IDLE 按 F5，从已保存的线性光强数组绘制相对峰值 dB 图。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

OUTPUT = Path(__file__).resolve().parent / "output"
FLOOR_DB = -40
CASES = ["single_slit", "double_slit", "rectangle", "circle", "double_hole"]


def main():
    font = Path("C:/Windows/Fonts/msyh.ttc")
    titles = ["Single slit", "Double slit", "Rectangle", "Circle", "Double hole"]
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
        titles = ["单缝", "双缝", "矩形孔", "圆孔", "双圆孔"]
    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.45), layout="constrained")
    with np.load(OUTPUT/"screen_data.npz") as archive:
        x, y = archive["x_mm"], archive["y_mm"]
        dx, dy = x[1]-x[0], y[1]-y[0]
        extent = (x[0]-dx/2, x[-1]+dx/2, y[0]-dy/2, y[-1]+dy/2)
        for ax, key, title in zip(axes, CASES, titles):
            intensity = archive[key]
            relative = intensity/intensity.max()
            db = 10*np.log10(np.maximum(relative, 10**(FLOOR_DB/10)))
            im = ax.imshow(db, origin="lower", extent=extent, cmap="inferno",
                           interpolation="nearest", vmin=FLOOR_DB, vmax=0)
            ax.set(title=title, xlabel="x (mm)", xticks=[-2,0,2], yticks=[-2,0,2])
    axes[0].set_ylabel("y (mm)")
    fig.colorbar(im, ax=list(axes), label="dB", ticks=np.arange(FLOOR_DB, 1, 10), shrink=.84)
    fig.savefig(OUTPUT/"overview_db.png", dpi=180)
    plt.close(fig)


if __name__=="__main__":
    main()
