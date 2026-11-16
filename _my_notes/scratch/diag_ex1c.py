"""诊断 3：训练/验证误差对比（判断欠拟合还是过拟合）+ 物理单位下的每步误差。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

HW3 = Path(__file__).resolve().parent.parent / "ethz-robot-learning-2026" / "hw3_imitation_learning"
sys.path.insert(0, str(HW3))

from hw3.dataset import Normalizer, SO100ChunkDataset, load_zarr  # noqa: E402
from hw3.eval_utils import load_checkpoint  # noqa: E402

CKPT = HW3 / "checkpoints/single_cube/best_model_ee_xyz_obstacle.pt"
ZARR = HW3 / "datasets/processed/single_cube/processed_ee_xyz.zarr"
H = 16
SEED = 42

model, norm, H_ckpt, sk, ak = load_checkpoint(CKPT, torch.device("cpu"))
model.eval()

states, actions, ep_ends = load_zarr(ZARR, state_keys=sk, action_keys=ak)
ds = SO100ChunkDataset(states, actions, ep_ends, chunk_size=H, normalizer=norm)
n_val = max(1, int(len(ds) * 0.1))
n_train = len(ds) - n_val
train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))

train_loader = DataLoader(train_ds, batch_size=512, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)


def mse(loader) -> float:
    tot, n = 0.0, 0
    with torch.no_grad():
        for s, a in loader:
            p = model(s)
            tot += torch.nn.functional.mse_loss(p, a, reduction="sum").item()
            n += a.numel()
    return tot / n


tr, va = mse(train_loader), mse(val_loader)
print(f"[1] 归一化空间 MSE: train {tr:.4f} | val {va:.4f}  (ckpt 记录的 best val = 0.3193)")
print(f"    -> {'欠拟合（训练/验证都高且接近）' if tr > 0.15 else '拟合'}, 差距 {(va-tr)/tr:+.1%}")

print("[2] 物理单位下的每步动作误差（模型 RMSE vs 动作标准差）")
with torch.no_grad():
    # 用整个数据集算逐维 RMSE（归一化）再乘回 action_std
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    se = np.zeros(actions.shape[1])
    n = 0
    for s, a in loader:
        p = model(s)
        se += ((p - a) ** 2).sum(dim=(0, 1)).numpy()
        n += a.shape[0] * a.shape[1]
    rmse_norm = np.sqrt(se / n)
for i, name in enumerate(["delta_x(3D)", "delta_y(3D)", "delta_z(3D)", "gripper"]):
    rmse_phys = rmse_norm[i] * norm.action_std[i]
    print(f"  {name:12s}: 归一化 RMSE {rmse_norm[i]:.3f} | 物理 RMSE {rmse_phys:.4f} "
          f"(std {norm.action_std[i]:.4f})")
print(f"  参考：示范数据里每步平均位移 ≈ 8 mm；若每步预测误差 6-7 mm，H=16 开环累积就会偏 10 cm+")
