"""IDLE/F5：模拟均匀、正入射相干光照亮五种黑白孔径后的透镜后焦面。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.special import j1

WAVELENGTH_MM = 0.000640
FOCAL_MM = 200.0
INCIDENT_MW_MM2 = 1.0       # 单位面积的入射功率，所有孔径用相同照明
SCREEN_STEP_MM = 0.01       # 理想点采样，无噪声、饱和或像素面积平均
HALF_SCREEN_MM = 3.2
CASES = {
    "single_slit": {"width_mm": 0.10, "height_mm": 1.00},
    "double_slit": {"width_mm": 0.10, "height_mm": 1.00, "separation_mm": 0.40},
    "rectangle": {"width_mm": 0.24, "height_mm": 0.40},
    "circle": {"diameter_mm": 0.32},
    "double_hole": {"diameter_mm": 0.20, "separation_mm": 0.48},
}
OUTPUT = Path(__file__).resolve().parent / "output"


def irradiance(x, y, spec):
    # 焦面坐标 -> 孔径面空间频率（周/mm）。np.sinc(t)=sin(pi*t)/(pi*t)。
    scale = WAVELENGTH_MM * FOCAL_MM
    u, v = x/scale, y/scale
    if "diameter_mm" in spec:
        d = spec["diameter_mm"]
        t = np.pi*d*np.hypot(u,v)
        airy = np.divide(2*j1(t), t, out=np.ones_like(t), where=t!=0)
        amplitude = (np.pi*d*d/4)*airy
    else:
        w, h = spec["width_mm"], spec["height_mm"]
        amplitude = w*h*np.sinc(w*u)*np.sinc(h*v)
    if "separation_mm" in spec:
        # 两个相同孔中心位于 (+/- separation/2, 0)，叠加复振幅。
        amplitude = amplitude * (2*np.cos(np.pi*spec["separation_mm"]*u))
    return INCIDENT_MW_MM2 * (amplitude/scale)**2


def main():
    OUTPUT.mkdir(exist_ok=True)
    half = round(HALF_SCREEN_MM/SCREEN_STEP_MM)
    axis = np.arange(-half,half+1)*SCREEN_STEP_MM
    edge = (half+0.5)*SCREEN_STEP_MM
    extent = [-edge,edge,-edge,edge]
    data = {"x_mm":axis,"y_mm":axis}
    metadata = dict(wavelength_mm=WAVELENGTH_MM,focal_mm=FOCAL_MM,
                    incident_irradiance_mW_per_mm2=INCIDENT_MW_MM2,
                    step_mm=SCREEN_STEP_MM, aperture_parameters=CASES,
                    array_order="row increases with y; column increases with x",
                    units="mW/mm^2", png="16-bit linear, normalized per image; top row is positive y")
    fig,axes=plt.subplots(1,len(CASES),figsize=(15,3.6),layout="constrained")
    for ax,(name,spec) in zip(axes,CASES.items()):
        intensity=irradiance(axis[None,:],axis[:,None],spec)
        data[name]=intensity
        relative=intensity/intensity.max()
        Image.fromarray(np.rint(65535*relative[::-1]).astype(np.uint16)).save(OUTPUT/f"{name}.png")
        im=ax.imshow(relative,extent=extent,origin="lower",cmap="gray",
                     interpolation="nearest",vmin=0,vmax=1)
        ax.set(title=name.replace("_"," ").title(),xlabel="x (mm)",xticks=[-2,0,2],yticks=[-2,0,2])
    axes[0].set_ylabel("y (mm)")
    fig.colorbar(im,ax=list(axes),label="I / peak (linear)",shrink=.75)
    fig.suptitle("Fraunhofer diffraction at the back focal plane | 640 nm, f = 200 mm")
    fig.savefig(OUTPUT/"overview.png",dpi=170)
    plt.close(fig)
    np.savez_compressed(OUTPUT/"screen_data.npz",**data)
    (OUTPUT/"parameters.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(f"Saved {len(CASES)} screen images, data and parameters: {OUTPUT}")


if __name__=="__main__":
    main()
