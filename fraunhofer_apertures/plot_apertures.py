"""在 IDLE 按 F5，读取上次仿真的孔径参数，按同一比例绘制正视图。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Rectangle

OUTPUT = Path(__file__).resolve().parent / "output"
LABELS = {"single_slit": "单缝", "double_slit": "双缝", "rectangle": "矩形孔",
          "circle": "圆孔", "double_hole": "双圆孔"}


def main():
    specs = json.loads((OUTPUT/"parameters.json").read_text(encoding="utf-8"))["aperture_parameters"]
    font = Path("C:/Windows/Fonts/msyh.ttc")
    chinese = font.is_file()
    if chinese:
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    largest = max(max(s.get("diameter_mm", s.get("height_mm")),
                      s.get("separation_mm", 0) + s.get("diameter_mm", s.get("width_mm")))
                  for s in specs.values())
    limit = 0.62*largest
    fig, axes = plt.subplots(1, len(specs), figsize=(15, 3.7), layout="constrained")
    for ax, (key, s) in zip(axes, specs.items()):
        ax.set_facecolor("black")
        dist = s.get("separation_mm", 0)
        centers = [-dist/2, dist/2] if dist else [0]
        for x in centers:
            if "diameter_mm" in s:
                diameter = s["diameter_mm"]
                shape = Circle((x,0), diameter/2)
                size = f"Ø {diameter:.2f} mm"
            else:
                w, h = s["width_mm"], s["height_mm"]
                shape = Rectangle((x-w/2,-h/2), w, h)
                size = f"{w:.2f} × {h:.2f} mm"
            shape.set(facecolor="white", edgecolor="none")
            ax.add_patch(shape)
        if dist:
            size += (f"  |  中心距 {dist:.2f} mm" if chinese else f"  |  d = {dist:.2f} mm")
        ax.set(xlim=(-limit,limit), ylim=(-limit,limit), aspect="equal", xlabel="x (mm)")
        ax.set_title(LABELS[key] if chinese else key.replace("_"," ").title(), fontsize=14, pad=18)
        ax.text(.5,1.015,size,transform=ax.transAxes,ha="center",va="bottom",fontsize=9)
    axes[0].set_ylabel("y (mm)")
    fig.suptitle("孔径板正视图    白色：透光开口    黑色：遮光区域" if chinese else
                 "Aperture plane | White: open   Black: opaque", fontsize=15)
    fig.savefig(OUTPUT/"aperture_shapes.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
