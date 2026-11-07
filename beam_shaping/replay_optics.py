"""IDLE/F5: independently propagate the exported phase maps (CPU, SciPy FFT)."""
from pathlib import Path
import csv
import json
import numpy as np
from scipy import fft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent / "output"


def simulate(plates, samples, cfg, keep_images=False):
    pitch, wavelength = cfg["PITCH"], cfg["WAVELENGTH"]
    dx = pitch / samples
    width = plates.shape[1] * samples
    n = cfg["WINDOW_FACTOR"] * width
    lo = (n-width)//2
    inside = np.s_[lo:lo+width, lo:lo+width]
    axis = (np.arange(n) - (n-1)/2)*dx
    r = np.hypot(axis[:, None], axis)
    ring = abs(r-cfg["RADIUS"]) <= 2*cfg["WIDTH"]
    u = np.exp(-(r/cfg["WAIST"])**2)
    u = (u/np.linalg.norm(u)).astype(np.complex64)
    images = [np.abs(u)**2] if keep_images else []
    freq = fft.fftfreq(n, dx)
    freq2 = freq[:, None]**2 + freq**2
    k0 = 1/wavelength
    kz = np.sqrt(np.maximum(k0**2-freq2, 0))
    z = cfg["TOTAL_DISTANCE"]/len(plates)
    valid = ((kz > 0) & (z*abs(freq[:, None]) <= n*dx*kz/2)
             & (z*abs(freq) <= n*dx*kz/2))
    # kz-k0 = -freq2/(kz+k0); dropping the constant phase does not affect intensity.
    kernel = (np.exp(-2j*np.pi*z*freq2/(kz+k0))*valid).astype(np.complex64)
    for phase in plates:
        field = np.zeros_like(u)
        phi = phase.repeat(samples, axis=0).repeat(samples, axis=1)
        field[inside] = u[inside] * np.exp(1j*phi)
        u = fft.ifft2(fft.fft2(field, workers=4)*kernel, workers=4)
        if keep_images:
            images.append(np.abs(u)**2)
    intensity = np.abs(u)**2
    # Azimuthal mean intensity, not integrated power (which favors larger radii).
    dr = 8e-6
    bins = (r/dr).astype(np.int32)
    counts = np.bincount(bins.ravel())
    radial = np.bincount(bins.ravel(), weights=intensity.ravel())/np.maximum(counts, 1)
    radii = (np.arange(len(radial))+0.5)*dr
    metrics = dict(samples=samples, dx_um=dx*1e6,
                   peak_radius_mm=radii[radial.argmax()]*1000,
                   ring_power=float(intensity[ring].sum(dtype=np.float64)),
                   screen_power=float(intensity.sum(dtype=np.float64)))
    return metrics, images, axis, radii, radial, intensity


def replay(weights):
    with np.load(weights,allow_pickle=False) as archive:
        cfg = json.loads(archive["config"].item())
        plates = archive["phase_rad"]
        saved = archive["intensity"]
    assert plates.shape[1:] == (cfg["PIXELS"],cfg["PIXELS"]) and np.isfinite(plates).all()
    prefix = weights.name.removesuffix("weights.npz")
    base = cfg["SAMPLES_PER_PIXEL"]
    limit_mm = 1.2*cfg["RADIUS"]*1000
    rows = []
    for samples in (base,2*base,4*base):
        row, images, axis, radii, radial, intensity = simulate(plates,samples,cfg,samples==4*base)
        if samples == base:
            relative_error = np.linalg.norm(intensity-saved)/np.linalg.norm(saved)
            print(f"Original grid intensity relative L2 difference: {relative_error:.6g}")
        rows.append(dict(optics="trained",**row))
        print(rows[-1], flush=True)
    control, _, _, _, blank, _ = simulate(np.zeros_like(plates),4*base,cfg)
    rows.append(dict(optics="flat",**control))
    print(rows[-1], flush=True)
    with (OUTPUT/(prefix+"measurements.csv")).open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    fig, axs = plt.subplots(1,len(images),figsize=(3.25*len(images),3.7),layout="constrained")
    mm = axis*1000
    halfpixel = (mm[1]-mm[0])/2
    extent = [mm[0]-halfpixel,mm[-1]+halfpixel]*2
    for i,(ax,img) in enumerate(zip(axs,images)):
        label = "Input" if i==0 else ("Screen" if i==len(plates) else f"After plate {i}")
        title = f"{label} | {1000*i*cfg['TOTAL_DISTANCE']/len(plates):g} mm"
        ax.imshow(img/img.max(),cmap="inferno",origin="lower",extent=extent,vmin=0,vmax=1)
        ax.set(xlim=(-limit_mm,limit_mm),ylim=(-limit_mm,limit_mm),xlabel="x (mm)",title=title)
    axs[0].set_ylabel("y (mm)")
    fig.suptitle("Optical propagation | each plane normalized to its own peak")
    fig.savefig(OUTPUT/(prefix+"propagation.png"),dpi=170)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout="constrained")
    target=np.exp(-0.5*((radii-cfg["RADIUS"])/cfg["WIDTH"])**2)
    for v,label in [(target,"Target"),(radial,"Trained plates"),(blank,"Flat plates")]:
        ax.plot(radii*1e3,v/v.max(),label=label)
    ax.set(xlim=(0,limit_mm),xlabel="Radius (mm)",ylabel="Azimuthal mean intensity / peak")
    ax.legend()
    fig.savefig(OUTPUT/(prefix+"radial_profiles.png"),dpi=160)
    plt.close(fig)


def main():
    files = sorted(OUTPUT.glob("*weights.npz"))
    if not files:
        print("请先在 IDLE 运行 train.py 生成 output 中的权重。")
    for weights in files:
        replay(weights)


if __name__ == "__main__":
    main()
