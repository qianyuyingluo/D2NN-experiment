"""3-layer phase-only diffractive optical neural network (D2NN) for MNIST.
All hyperparameters and physical parameters are constants below.
Run: python train_d2nn.py
Dependencies: torch, torchvision, numpy, matplotlib
"""
from pathlib import Path
import copy
import csv
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==================== All settings ====================
SEED = 42
DATA_DIR = "./data"
OUTPUT_DIR = "./d2nn_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WAVELENGTH = 532e-9        # m
DISTANCE = 20e-3          # m, each layer -> next layer / screen
NUM_LAYERS = 5
PHASE_PIXELS = 64          # 64x64 independently trainable phase pixels per plate
PHASE_PIXEL_SIZE = 10e-6  # m
GRID_SIZE = 256            # propagation field samples along each axis
DX = 5e-6                 # m: 256*5um = 1.28mm field of view
FFT_PAD_FACTOR = 2        # zero-pad each propagation to 512x512 (avoid some wraparound)

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-3
EFFICIENCY_WEIGHT = 0.2
TRAIN_SAMPLES = 60000     # 1..60000, train/validation split below
VAL_SAMPLES = 5000        # held out from TRAIN_SAMPLES
NUM_WORKERS = 0           # portable Windows setting
TEST_BATCHES = None      # None tests all 10,000 MNIST test images

DETECTOR_SIDE = 24        # 24*5um = 0.12mm
DETECTOR_X_CENTERS = (-80, -40, 0, 40, 80)  # pixels from center
DETECTOR_Y_CENTERS = (-30, 30)               # pixels from center, row-major 0..9
EPS = 1e-12
# ======================================================


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class D2NN(nn.Module):
    def __init__(self):
        super().__init__()
        assert GRID_SIZE % PHASE_PIXELS == 0
        assert FFT_PAD_FACTOR >= 1
        assert abs((PHASE_PIXEL_SIZE / DX) - (GRID_SIZE / PHASE_PIXELS) / 2) < 1e-8, (
            "This demo assumes each 10um phase pixel occupies 2x2 samples of 5um")
        # Fix the phase plate at center: 64*10um=0.64mm, 128x128 samples.
        self.phase = nn.Parameter(2 * np.pi * torch.rand(NUM_LAYERS, PHASE_PIXELS, PHASE_PIXELS))
        plate_n = round(PHASE_PIXELS * PHASE_PIXEL_SIZE / DX)
        self.plate_n = plate_n
        self.start = (GRID_SIZE - plate_n) // 2
        # Not trainable: a hard square aperture at each phase plate.
        aperture = torch.zeros((GRID_SIZE, GRID_SIZE), dtype=torch.float32)
        aperture[self.start:self.start+plate_n, self.start:self.start+plate_n] = 1
        self.register_buffer("aperture", aperture)
        # Forward angular spectrum transfer function, using exp(+i kz) convention.
        n = GRID_SIZE * FFT_PAD_FACTOR
        f = torch.fft.fftfreq(n, d=DX)
        fy, fx = torch.meshgrid(f, f, indexing="ij")
        root = (1.0 / WAVELENGTH)**2 - fx**2 - fy**2
        kz = 2 * np.pi * torch.sqrt(root.to(torch.complex64))
        H = torch.exp(1j * DISTANCE * kz).to(torch.complex64)
        self.register_buffer("H", H)
        self.pad = (n - GRID_SIZE) // 2
        assert n - GRID_SIZE == 2 * self.pad
        self.register_buffer("detector_masks", self.make_detectors())

    @staticmethod
    def make_detectors():
        masks = torch.zeros(10, GRID_SIZE, GRID_SIZE)
        s = DETECTOR_SIDE
        assert s % 2 == 0
        for digit in range(10):
            col, row = digit % 5, digit // 5
            cx = GRID_SIZE // 2 + DETECTOR_X_CENTERS[col]
            cy = GRID_SIZE // 2 + DETECTOR_Y_CENTERS[row]
            assert 0 <= cx-s//2 and cx+s//2 <= GRID_SIZE
            assert 0 <= cy-s//2 and cy+s//2 <= GRID_SIZE
            masks[digit, cy-s//2:cy+s//2, cx-s//2:cx+s//2] = 1
        assert masks.sum(0).max().item() == 1, "Overlapping detectors"
        return masks

    def propagate(self, u):
        # 2x zero padding reduces FFT's artificial periodic wraparound.
        p = self.pad
        u = torch.nn.functional.pad(u, (p, p, p, p))
        u = torch.fft.ifft2(torch.fft.fft2(u) * self.H)
        if p > 0:
            u = u[..., p:-p, p:-p]
        return u

    def forward(self, images, return_field=False):
        # 28x28 gray intensity -> field amplitude, then resize to 128x128 aperture.
        # Resizing intensity first makes the encoding explicit.
        intensity = torch.nn.functional.interpolate(
            images, size=(self.plate_n, self.plate_n), mode="bilinear", align_corners=False
        ).clamp_min(0)
        amplitude = intensity.sqrt()
        amplitude = amplitude / amplitude.square().sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
        u = torch.zeros((images.shape[0], GRID_SIZE, GRID_SIZE), device=images.device, dtype=torch.complex64)
        s = self.start
        u[:, s:s+self.plate_n, s:s+self.plate_n] = amplitude[:, 0].to(torch.complex64)
        for layer in range(NUM_LAYERS):
            # nearest-neighbor implements a physically pixelated phase plate.
            phi = self.phase[layer].repeat_interleave(self.plate_n // PHASE_PIXELS, 0).repeat_interleave(self.plate_n // PHASE_PIXELS, 1)
            modulation = torch.exp(1j * phi)
            next_field = torch.zeros_like(u)
            next_field[:, s:s+self.plate_n, s:s+self.plate_n] = (
                u[:, s:s+self.plate_n, s:s+self.plate_n] * modulation
            )
            u = self.propagate(next_field)
        light = u.abs().square()
        # energy proportional to sum of optical intensity over detector pixels
        energies = torch.einsum('bhw,khw->bk', light, self.detector_masks)
        total = light.sum(dim=(-2,-1))
        if return_field:
            return energies, total, light, u
        return energies, total


def loss_fn(energies):
    # Correct-class negative log(relative energy) is calculated externally.
    probs = energies / energies.sum(dim=1, keepdim=True).clamp_min(EPS)
    return probs


def evaluate(model, loader, device, limit_batches=None):
    model.eval()
    count = 0
    correct = 0
    efficiency_sum = 0.0
    loss_sum = 0.0
    with torch.no_grad():
        for b, (images, targets) in enumerate(loader):
            if limit_batches is not None and b >= limit_batches:
                break
            images, targets = images.to(device), targets.to(device)
            energies, _ = model(images)
            p = loss_fn(energies)
            true_p = p.gather(1, targets[:, None]).clamp_min(EPS)
            eff = energies.sum(dim=1).clamp_min(EPS)
            loss = -true_p.log().mean() - EFFICIENCY_WEIGHT * eff.log().mean()
            bs = images.shape[0]
            count += bs
            correct += (energies.argmax(1) == targets).sum().item()
            efficiency_sum += eff.sum().item()  # input field normalized to power 1
            loss_sum += loss.item() * bs
    return loss_sum / count, correct / count, efficiency_sum / count


def save_visuals(model, sample_images, sample_labels, output_dir, device, history):
    model.eval()
    phase = model.phase.detach().cpu().numpy() % (2 * np.pi)
    for i, ph in enumerate(phase, 1):
        np.save(output_dir / f"phase_{i}.npy", ph)
        np.savetxt(output_dir / f"phase_{i}.csv", ph, delimiter=",")
    fig, axs = plt.subplots(1, NUM_LAYERS, figsize=(4 * NUM_LAYERS, 3.5))
    for k, ax in enumerate(np.atleast_1d(axs)):
        im = ax.imshow(phase[k], cmap="twilight", vmin=0, vmax=2*np.pi)
        ax.set_title(f"Phase plate {k+1}")
        ax.axis("off")
    fig.colorbar(im, ax=np.atleast_1d(axs).tolist(), label="Phase (rad)", shrink=.7)
    fig.savefig(output_dir / "phase_plates.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
    axs[0].plot(history["train_loss"], label="train loss")
    axs[0].plot(history["val_loss"], label="val loss")
    axs[0].legend(); axs[0].set_title("Loss")
    axs[1].plot(history["train_accuracy"], label="train")
    axs[1].plot(history["val_accuracy"], label="val")
    axs[1].legend(); axs[1].set_title("Classification accuracy")
    axs[2].plot(history["train_efficiency"], label="train")
    axs[2].plot(history["val_efficiency"], label="val")
    axs[2].legend(); axs[2].set_title("Power in detectors / input")
    fig.tight_layout(); fig.savefig(output_dir / "learning_curves.png", dpi=160); plt.close(fig)

    with torch.no_grad():
        images = sample_images.to(device)
        energies, _, light, _ = model(images, return_field=True)
    fig, axs = plt.subplots(2, images.shape[0], figsize=(2.5*images.shape[0], 5))
    if images.shape[0] == 1: axs = axs.reshape(2, 1)
    for j in range(images.shape[0]):
        axs[0,j].imshow(images[j,0].cpu(), cmap="gray")
        axs[0,j].set_title(f"True: {sample_labels[j].item()}")
        axs[1,j].imshow(light[j].cpu(), cmap="inferno")
        axs[1,j].set_title(f"Pred: {energies[j].argmax().item()}")
        for a in axs[:,j]: a.axis("off")
    fig.tight_layout(); fig.savefig(output_dir / "samples.png", dpi=160); plt.close(fig)


def main():
    seed_everything(SEED)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Device:", DEVICE)
    print("Trainable parameters:", NUM_LAYERS * PHASE_PIXELS**2)
    print("Note: MNIST first run requires network to download data.")

    ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
    assert 0 < VAL_SAMPLES < TRAIN_SAMPLES <= len(ds)
    chosen, _ = random_split(ds, [TRAIN_SAMPLES, len(ds)-TRAIN_SAMPLES], generator=torch.Generator().manual_seed(SEED))
    train_ds, val_ds = random_split(chosen, [TRAIN_SAMPLES-VAL_SAMPLES, VAL_SAMPLES], generator=torch.Generator().manual_seed(SEED+1))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    device = torch.device(DEVICE)
    model = D2NN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {k:[] for k in ["train_loss","val_loss","train_accuracy","val_accuracy","train_efficiency","val_efficiency"]}
    best_acc = -1.0
    best_loss = float('inf')

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *history.keys()])
        for epoch in range(1, EPOCHS+1):
            model.train()
            total_loss = 0.0
            correct = 0
            train_eff = 0.0
            seen = 0
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                energies, _ = model(images)
                p = loss_fn(energies)
                picked = p.gather(1, targets[:, None]).clamp_min(EPS)
                cls_loss = -picked.log().mean()
                # Input power is normalized to one; detector power is the efficiency.
                eff = energies.sum(1).clamp_min(EPS)
                loss = cls_loss - EFFICIENCY_WEIGHT * eff.log().mean()
                loss.backward()
                optimizer.step()
                bs = images.shape[0]
                seen += bs
                total_loss += loss.item() * bs
                correct += (energies.argmax(1) == targets).sum().item()
                train_eff += eff.detach().sum().item()
            tr_loss, tr_acc, tr_eff = total_loss/seen, correct/seen, train_eff/seen
            val_loss, val_acc, val_eff = evaluate(model, val_loader, device)
            for name, val in zip(history, [tr_loss,val_loss,tr_acc,val_acc,tr_eff,val_eff]):
                history[name].append(val)
            writer.writerow([epoch, tr_loss,val_loss,tr_acc,val_acc,tr_eff,val_eff]); f.flush()
            print(f"Epoch {epoch:02d}/{EPOCHS} | train loss {tr_loss:.4f}, acc {tr_acc:.2%}, eff {tr_eff:.2%}"
                  f" | val loss {val_loss:.4f}, acc {val_acc:.2%}, eff {val_eff:.2%}", flush=True)
            if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
                best_acc, best_loss = val_acc, val_loss
                torch.save({"model":model.state_dict(), "epoch":epoch, "val_accuracy":val_acc}, output_dir / "best.pt")

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    test_loss, test_acc, test_eff = evaluate(model, test_loader, device, TEST_BATCHES)
    print(f"Best epoch: {checkpoint['epoch']} | test accuracy {test_acc:.2%} | test detector efficiency {test_eff:.2%}")
    torch.save(model.state_dict(), output_dir / "model_weights.pt")
    sample_images = torch.stack([test_ds[k][0] for k in range(5)])
    sample_labels = torch.tensor([test_ds[k][1] for k in range(5)])
    save_visuals(model, sample_images, sample_labels, output_dir, device, history)
    print("Saved to:", output_dir.resolve())

if __name__ == "__main__":
    main()
