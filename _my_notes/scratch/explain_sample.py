"""打印一个真实训练样本：state 长什么样、action chunk 长什么样、两者如何对应。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HW3 = Path(__file__).resolve().parent.parent / "ethz-robot-learning-2026" / "hw3_imitation_learning"
sys.path.insert(0, str(HW3))

from hw3.dataset import (  # noqa: E402
    Normalizer, SO100ChunkDataset, build_valid_indices, load_zarr,
)

ZARR = HW3 / "datasets/processed/single_cube/processed_ee_xyz.zarr"
STATE_KEYS = ["state_ee_xyz", "state_gripper", "state_cube[:3]", "state_obstacle"]
ACTION_KEYS = ["action_ee_xyz", "action_gripper"]
H = 16

states, actions, ep_ends = load_zarr(ZARR, state_keys=STATE_KEYS, action_keys=ACTION_KEYS)
norm = Normalizer.from_data(states, actions)
ds = SO100ChunkDataset(states, actions, ep_ends, chunk_size=H, normalizer=norm)
idx = build_valid_indices(ep_ends, H)

print("=" * 78)
print(f"数据集: {len(ds)} 个样本 | episode_ends={ep_ends}")
print(f"state 维数={states.shape[1]} | action 维数={actions.shape[1]} | chunk={H}")
print("state 各维含义:", "ee_x, ee_y, ee_z, gripper_state, cube_x, cube_y, cube_z, obs_x, obs_y, obs_z")
print("action 各维含义: delta_ee_x, delta_ee_y, delta_ee_z, gripper_ctrl(绝对指令)")

k = 100                      # 取第 100 个训练样本
t = int(idx[k])
ep = int(np.searchsorted(ep_ends, t))
ep_start = int(ep_ends[ep - 1]) if ep > 0 else 0
print("=" * 78)
print(f"样本 dataset[{k}] -> 全局时刻 t={t}（属于 episode {ep}，该 episode 的 t 范围 {ep_start}..{ep_ends[ep]-1}）")

s_raw = states[t]
a_raw = actions[t:t + H]
print(f"\n--- state (t={t}, 原始物理量) ---")
for name, v in zip(["ee_x", "ee_y", "ee_z", "gripper_state", "cube_x", "cube_y", "cube_z",
                    "obs_x", "obs_y", "obs_z"], s_raw):
    print(f"  {name:14s} = {v: .4f}")
print(f"\n--- 目标 action chunk (t={t} .. {t+H-1}, 原始物理量) ---")
print(f"{'step':>4} {'d_ee_x(mm)':>10} {'d_ee_y(mm)':>10} {'d_ee_z(mm)':>10} {'gripper_ctrl':>13}")
for i in range(H):
    ax, ay, az, g = a_raw[i]
    print(f"{t+i:>4} {ax*1000:>10.2f} {ay*1000:>10.2f} {az*1000:>10.2f} {g:>13.3f}")

# 验证 a_t = s_{t+1} - s_t
print(f"\n--- 验证 action = 下一时刻状态 - 当前状态 ---")
print(f"  state_ee(t)          = {np.round(states[t][:3], 4)}")
print(f"  state_ee(t+1)        = {np.round(states[t+1][:3], 4)}")
print(f"  state_ee(t+1)-state_ee(t) = {np.round(states[t+1][:3] - states[t][:3], 4)}")
print(f"  action_ee_xyz(t)     = {np.round(a_raw[0][:3], 4)}   <- 相等")
print(f"  （夹爪除外：state_gripper 是关节角状态 {states[t][3]:.3f}，"
      f"action_gripper 是控制指令 {a_raw[0][3]:.3f}）")

print("\n--- 归一化后（网络真正看到的数值） ---")
s_norm = norm.normalize_state(s_raw)
a_norm = norm.normalize_action(a_raw)
print("  state_norm =", np.round(s_norm, 3))
print("  chunk_norm[0] =", np.round(a_norm[0], 3), " chunk_norm[-1] =", np.round(a_norm[-1], 3))
print(f"  state 各维 mean/std: {np.round(norm.state_mean,4)} / {np.round(norm.state_std,4)}")
print(f"  action 各维 mean/std: {np.round(norm.action_mean,4)} / {np.round(norm.action_std,4)}")

s0, a0 = ds[k]
print(f"\n--- Dataset 实际返回的 tensors（已归一化） ---")
print(f"  state: {tuple(s0.shape)} float32, 前 5 维 {np.round(s0[:5].numpy(), 3)}")
print(f"  chunk: {tuple(a0.shape)} float32, chunk[0] = {np.round(a0[0].numpy(), 3)}")

print("\n--- DataLoader 一个 batch ---")
loader = DataLoader(ds, batch_size=320, shuffle=False)
bs, ba = next(iter(loader))
print(f"  states {tuple(bs.shape)} | action_chunks {tuple(ba.shape)}")
print(f"  => 模型输入 (320, 10)，输出/标签 (320, 16, 4)")

# 找一个 chunk 里包含夹爪开->合动作的样本，展示阶段变化
for kk in range(len(ds)):
    tt = int(idx[kk])
    g = actions[tt:tt + H, 3]
    if g.max() - g.min() > 1.0:          # 夹爪指令大幅变化
        print("\n--- 另一个样本：chunk 内包含夹爪动作变化 ---")
        print(f"样本 dataset[{kk}] -> t={tt}, episode {int(np.searchsorted(ep_ends, tt))}")
        print(f"{'step':>4} {'d_ee_x(mm)':>10} {'d_ee_y(mm)':>10} {'d_ee_z(mm)':>10} {'gripper_ctrl':>13}")
        for i in range(H):
            ax, ay, az, gg = actions[tt + i]
            print(f"{tt+i:>4} {ax*1000:>10.2f} {ay*1000:>10.2f} {az*1000:>10.2f} {gg:>13.3f}")
        break

print("\n--- episode 边界 ---")
print(f"  episode {ep}: 共 {ep_ends[ep]-ep_start} 步，最后一步 t={ep_ends[ep]-1} 不能作为起点")
print(f"  能作为起点的最大 t = {ep_ends[ep]-1-H+1} = {ep_ends[ep]-H}（chunk 刚好到 episode 末尾）")
