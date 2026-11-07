# -*- coding: utf-8 -*-
"""D2NN 模97加法：IDLE 直接按 F5 运行。

物理模型：相干双点输入 -> 三张仅相位调制板 -> 角谱自由空间传播 -> 97个光强探测器。
这是数值仿真。线性光场传播 + 末端平方律探测，不含非线性光学器件。
"""

# ======================= 可修改参数：全部在这里 =======================
MODULUS = 97
TRAIN_RATIO = 0.50             # 数据集中随机 50% 用于训练
VAL_RATIO = 0.25               # 25% 验证，其余 25% 最终测试
SEED = 2026
EPOCHS = 1500                  # Grokking 可能很慢，需要自行增大
BATCH_SIZE = 128               # 显存不足可以降为 64 / 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0             # 不建议对周期相位角直接使用 L2 权重衰减
EFFICIENCY_WEIGHT = 0.0       # 先只研究泛化，后续可研究效率损失

NUM_LAYERS = 3
PHASE_PIXELS = 64              # 每张板64x64个可训练相位像素
FIELD_PIXELS = 128             # 传播光场128x128；相位像素映射到2x2场采样点
PROPAGATION_PIXELS = 256       # FFT传播时补零，减轻周期边界伪影
FIELD_PIXEL_SIZE = 10e-6      # 光场采样间距：10微米，相位像素间距为20微米
WAVELENGTH = 532e-9           # 532 nm
DISTANCE = 20e-3              # 每一段传播20 mm

INPUT_SPOT_SIGMA = 1.0        # 输入光点的高斯宽度，以场网格像素为单位
DETECTOR_RADIUS = 3           # 探测器半径（采样点），使用非重叠圆形区域
LOG_EPS = 1e-12
PRINT_EVERY = 10              # 每10轮打印一次，包括这10轮耗时
SAVE_EVERY = 100              # 每100轮保存可恢复的最新检查点
OUTPUT_DIR = "d2nn_mod97_results"
DEVICE = "auto"               # auto / cuda / cpu
# ====================================================================

import os
import csv
import time
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 采用非交互后端，IDLE、终端都可自动保存图片
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset():
    """严格划分所有97^2组输入；验证/测试组合不参与梯度训练。"""
    aa, bb = torch.meshgrid(torch.arange(MODULUS), torch.arange(MODULUS), indexing="ij")
    pairs = torch.stack([aa.reshape(-1), bb.reshape(-1)], dim=1).long()
    labels = (pairs[:, 0] + pairs[:, 1]) % MODULUS
    indices = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(SEED))
    n_train = int(len(pairs) * TRAIN_RATIO)
    n_val = int(len(pairs) * VAL_RATIO)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError("训练/验证/测试集至少需要各有一个样本，请修改比例。")
    return (pairs[train_idx], labels[train_idx]), (pairs[val_idx], labels[val_idx]), (pairs[test_idx], labels[test_idx])


def create_input_encoding(device):
    """先生成194张固定实数透射模式：a区97张、b区97张。

    每个模式为二维高斯亮点（不是无面积的Dirac delta）。两个亮点来自相干光场，
    复振幅相加后全场归一化。因此标签不会隐式泄漏到光场之外。
    """
    n = FIELD_PIXELS
    yy, xx = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device), indexing="ij")
    bank = torch.empty((2, MODULUS, n, n), dtype=torch.float32, device=device)
    # 两个固定编码区，各10列x10行，97点，间距5格，完全位于板内
    for which, x_start in enumerate((14, 69)):
        for idx in range(MODULUS):
            row, col = divmod(idx, 10)
            cx = x_start + 5 * col
            cy = 39 + 5 * row
            spot = torch.exp(-((xx-cx)**2+(yy-cy)**2)/(2*INPUT_SPOT_SIGMA**2))
            bank[which, idx] = spot
    return bank


def create_detectors(device):
    """10x10布局共97个圆形探测区，间距10像素，确保互不重叠。"""
    n = FIELD_PIXELS
    yy, xx = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device), indexing="ij")
    masks = torch.zeros((MODULUS, n, n), device=device, dtype=torch.float32)
    centers = []
    for k in range(MODULUS):
        row, col = divmod(k, 10)
        cx, cy = 19 + col*10, 19 + row*10
        masks[k] = (((xx-cx)**2 + (yy-cy)**2) <= DETECTOR_RADIUS**2).float()
        centers.append((cx, cy))
    assert masks.sum(dim=0).max().item() <= 1, "探测器发生重叠"
    return masks, centers


class FreeSpacePropagation(nn.Module):
    """正向传播：角谱法；对应 e^{-i omega t} 时间约定。"""
    def __init__(self):
        super().__init__()
        N = PROPAGATION_PIXELS
        if N < FIELD_PIXELS or N % 2 or FIELD_PIXELS % 2:
            raise ValueError("传播网格必须是偶数，并且不小于光场网格")
        fx = torch.fft.fftfreq(N, d=FIELD_PIXEL_SIZE)
        fy = torch.fft.fftfreq(N, d=FIELD_PIXEL_SIZE)
        fy2, fx2 = torch.meshgrid(fy, fx, indexing="ij")
        inv_lambda2 = (1.0 / WAVELENGTH)**2
        # 复数平方根：传播波为实相位，倏逝波为指数衰减
        kz = 2*math.pi * torch.sqrt((inv_lambda2 - fx2**2 - fy2**2).to(torch.complex64))
        H = torch.exp(1j * kz * DISTANCE).to(torch.complex64)
        self.register_buffer("H", H)
        self.pad = (N - FIELD_PIXELS)//2

    def forward(self, u):
        p = self.pad
        u = F.pad(u, (p, p, p, p))
        u = torch.fft.ifft2(torch.fft.fft2(u) * self.H)
        if p:
            u = u[:, p:-p, p:-p]
        return u


class D2NN(nn.Module):
    def __init__(self, masks):
        super().__init__()
        if FIELD_PIXELS % PHASE_PIXELS != 0:
            raise ValueError("FIELD_PIXELS 必须是 PHASE_PIXELS 的整数倍")
        if NUM_LAYERS < 1:
            raise ValueError("NUM_LAYERS 至少为1")
        self.phases = nn.ParameterList([
            nn.Parameter(torch.rand(PHASE_PIXELS, PHASE_PIXELS)*2*math.pi)
            for _ in range(NUM_LAYERS)
        ])
        self.propagate = FreeSpacePropagation()
        self.register_buffer("masks_flat", masks.reshape(MODULUS, -1))
        self.scale = FIELD_PIXELS // PHASE_PIXELS

    def forward(self, u):
        for phase in self.phases:
            # 相位像素在对应2x2场采样点上保持常数：真实像素化相位调制
            phase_large = phase.repeat_interleave(self.scale, 0).repeat_interleave(self.scale, 1)
            u = u * torch.exp(1j * phase_large)
            u = self.propagate(u)
        intensity = u.abs().square().reshape(u.shape[0], -1)
        # masks @ intensity：每个区域内所有像素的光强求和
        energies = intensity @ self.masks_flat.T
        output_power = intensity.sum(dim=1)
        detector_power = energies.sum(dim=1)
        return energies, detector_power, output_power


def encode(pairs, bank):
    # a和b两组编码光点叠加形成相干复光场
    u = bank[0, pairs[:, 0]] + bank[1, pairs[:, 1]]
    # 输入功率统一为1；两组光点对应的坐标不同，所以不相互重合
    u = u / (u.square().sum(dim=(1, 2), keepdim=True).sqrt() + 1e-12)
    return u.to(torch.complex64)


def compute_loss(energies, labels, detector_power):
    # logits=log(探测区能量)，CE等价于 -log(E_正确/总探测功率)
    logits = torch.log(energies + LOG_EPS)
    loss = F.cross_entropy(logits, labels)
    if EFFICIENCY_WEIGHT:
        loss = loss - EFFICIENCY_WEIGHT * torch.log(detector_power + LOG_EPS).mean()
    return loss


def eval_dataset(model, data, bank):
    model.eval()
    pairs, labels = data
    n, correct, loss_sum, efficiency_sum, out_power_sum = 0, 0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(pairs), BATCH_SIZE):
            p = pairs[i:i+BATCH_SIZE]
            y = labels[i:i+BATCH_SIZE]
            e, dp, op = model(encode(p, bank))
            # 按同样的训练损失计算；报告的效率是相对输入功率1
            loss = compute_loss(e, y, dp)
            bs = len(p)
            n += bs
            correct += int((e.argmax(dim=1) == y).sum())
            loss_sum += float(loss)*bs
            efficiency_sum += float(dp.sum())
            out_power_sum += float(op.sum())
    return {"loss":loss_sum/n, "acc":correct/n*100,
            "eff":efficiency_sum/n*100, "out_power":out_power_sum/n*100}


def save_checkpoint(path, model, optimizer, epoch, history, best_val):
    torch.save({"model":model.state_dict(), "optimizer":optimizer.state_dict(),
                "epoch":epoch, "history":history, "best_val":best_val,
                "config":{"modulus":MODULUS,"layers":NUM_LAYERS,
                          "phase_pixels":PHASE_PIXELS,"field_pixels":FIELD_PIXELS}}, path)


def save_phases(model, out_dir):
    for i, phase in enumerate(model.phases, 1):
        arr = np.mod(phase.detach().cpu().numpy(), 2*np.pi)
        np.save(out_dir / f"phase_{i}.npy", arr)
        np.savetxt(out_dir / f"phase_{i}.csv", arr, delimiter=",")
    fig, axs = plt.subplots(1, NUM_LAYERS, figsize=(4*NUM_LAYERS, 4), squeeze=False)
    for i, ax in enumerate(axs[0]):
        im = ax.imshow(np.mod(model.phases[i].detach().cpu().numpy(), 2*np.pi),
                       cmap="twilight", vmin=0, vmax=2*np.pi)
        ax.set_title(f"Phase plate {i+1}")
        ax.axis("off")
    fig.colorbar(im, ax=axs.ravel().tolist(), label="phase (rad)", shrink=0.75)
    fig.savefig(out_dir / "phase_plates.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_history(history, out_dir):
    if not history:
        return
    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    ep = [r["epoch"] for r in history]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ep, [r["train_acc"] for r in history], label="Train")
    ax.plot(ep, [r["val_acc"] for r in history], label="Validation")
    ax.set(xlabel="Epoch", ylabel="Accuracy (%)", title="D2NN modular addition: generalization")
    ax.set_ylim(0, 101)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ep, [r["train_loss"] for r in history], label="Train")
    ax.plot(ep, [r["val_loss"] for r in history], label="Validation")
    ax.set(xlabel="Epoch", ylabel="Loss", title="D2NN modular addition: loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss.png", dpi=160)
    plt.close(fig)


def main():
    seed_everything(SEED)
    if DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(DEVICE)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    out_dir = Path(__file__).resolve().parent / OUTPUT_DIR
    out_dir.mkdir(exist_ok=True, parents=True)

    (train_cpu, train_y_cpu), (val_cpu, val_y_cpu), (test_cpu, test_y_cpu) = make_dataset()
    print("Device:", device, "| PyTorch:", torch.__version__, flush=True)
    print(f"Mod {MODULUS}: {MODULUS**2} combos | train {len(train_cpu)} | validation {len(val_cpu)} | test {len(test_cpu)}")
    print(f"D2NN: {NUM_LAYERS} phase plates, {PHASE_PIXELS}x{PHASE_PIXELS} phase pixels each")
    print(f"FFT grid: {PROPAGATION_PIXELS}x{PROPAGATION_PIXELS} | field: {FIELD_PIXELS}x{FIELD_PIXELS}")
    print(f"Epochs={EPOCHS}, batch={BATCH_SIZE}, learning_rate={LEARNING_RATE}")
    print("Random guess accuracy: {:.2f}%".format(100/MODULUS), flush=True)

    bank = create_input_encoding(device)
    masks, centers = create_detectors(device)
    model = D2NN(masks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
    train=(train_cpu.to(device),train_y_cpu.to(device))
    val=(val_cpu.to(device),val_y_cpu.to(device))
    test=(test_cpu.to(device),test_y_cpu.to(device))

    best_val = -1.0
    history = []
    overall_start = time.perf_counter()
    block_start = overall_start
    last_print_epoch = 0
    print("Training... (results saved in {})".format(out_dir), flush=True)
    for epoch in range(1, EPOCHS+1):
        model.train()
        order = torch.randperm(len(train[0]), device=device)
        total_loss = 0.0
        correct = 0
        total = 0
        efficiency_total = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start+BATCH_SIZE]
            p, y = train[0][idx], train[1][idx]
            optimizer.zero_grad(set_to_none=True)
            e, dp, _ = model(encode(p, bank))
            loss = compute_loss(e, y, dp)
            loss.backward()
            optimizer.step()
            bs = len(idx)
            total += bs
            total_loss += float(loss.detach())*bs
            correct += int((e.detach().argmax(dim=1)==y).sum())
            efficiency_total += float(dp.detach().sum())

        # 验证集只用于选模型，测试集直到训练结束才评估
        vm = eval_dataset(model, val, bank)
        row = {"epoch":epoch,"train_loss":total_loss/total,
               "train_acc":correct/total*100,"train_eff":efficiency_total/total*100,
               "val_loss":vm["loss"],"val_acc":vm["acc"],
               "val_eff":vm["eff"],"val_output_power":vm["out_power"]}
        history.append(row)
        if vm["acc"] > best_val:
            best_val = vm["acc"]
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, history, best_val)

        if epoch % PRINT_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
            if device.type == "cuda":
                torch.cuda.synchronize()
            now=time.perf_counter()
            print(f"Epoch {epoch:04d}/{EPOCHS} | "
                  f"train loss {row['train_loss']:.4f}, acc {row['train_acc']:.2f}% | "
                  f"val loss {row['val_loss']:.4f}, acc {row['val_acc']:.2f}% | "
                  f"eff {row['val_eff']:.3f}% | "
                  f"last {epoch-last_print_epoch} epoch(s) {now-block_start:.1f}s | "
                  f"total {now-overall_start:.1f}s", flush=True)
            block_start=now
            last_print_epoch=epoch

        if epoch % SAVE_EVERY == 0:
            save_checkpoint(out_dir / "latest.pt", model, optimizer, epoch, history, best_val)
            save_history(history, out_dir)

    print("\nLoading best validation checkpoint for FINAL test...", flush=True)
    try:
        state = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    except TypeError:  # compatible with older PyTorch
        state = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(state["model"])
    final_train = eval_dataset(model, train, bank)
    final_val = eval_dataset(model, val, bank)
    final_test = eval_dataset(model, test, bank)
    print(f"Best epoch {state['epoch']} | train {final_train['acc']:.2f}% | "
          f"val {final_val['acc']:.2f}% | test {final_test['acc']:.2f}%", flush=True)
    print(f"Test detector efficiency {final_test['eff']:.4f}% | "
          f"output window power {final_test['out_power']:.4f}%", flush=True)
    save_phases(model, out_dir)
    save_history(history, out_dir)
    with open(out_dir / "final_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Best epoch: {state['epoch']}\n"
                f"Train accuracy: {final_train['acc']:.4f}%\n"
                f"Validation accuracy: {final_val['acc']:.4f}%\n"
                f"Test accuracy: {final_test['acc']:.4f}%\n"
                f"Test detector efficiency: {final_test['eff']:.6f}%\n"
                f"Test output window power: {final_test['out_power']:.6f}%\n")
    print("Saved to:", out_dir, flush=True)


if __name__ == "__main__":
    main()
