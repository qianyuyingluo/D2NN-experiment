"""在 IDLE 按 F5：生成全部模 97 加法样本及固定训练/测试划分。"""
from pathlib import Path
import numpy as np

MODULUS = 97
TRAIN_FRACTION = 0.5
SEED = 42
DATA_FILE = Path(__file__).resolve().parent / 'mod97_data.npz'

def generate():
    assert 0 < TRAIN_FRACTION < 1
    a, b = np.meshgrid(np.arange(MODULUS), np.arange(MODULUS), indexing='ij')
    pairs = np.column_stack((a.ravel(), b.ravel())).astype(np.int64)
    targets = pairs.sum(axis=1) % MODULUS
    indices = np.random.default_rng(SEED).permutation(len(pairs))
    cut = int(len(pairs) * TRAIN_FRACTION)
    np.savez_compressed(DATA_FILE, pairs=pairs, targets=targets,
                        train_indices=indices[:cut], test_indices=indices[cut:],
                        modulus=MODULUS, seed=SEED, train_fraction=TRAIN_FRACTION)
    print(f'已保存 {DATA_FILE}\n总计 {len(pairs)}，训练 {cut}，测试 {len(pairs)-cut}。')
    print('输入范围 0..96，标签 (a+b)%97；测试样本不参与反向传播。')

if __name__ == '__main__':
    generate()
