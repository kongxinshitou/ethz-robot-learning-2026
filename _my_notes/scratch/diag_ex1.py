"""诊断 ex1 只有 10% 成功率：检查 checkpoint 元数据 / 示范数据本身是否成功 / 模型拟合程度。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import zarr

HW3 = Path(__file__).resolve().parent.parent / "ethz-robot-learning-2026" / "hw3_imitation_learning"
sys.path.insert(0, str(HW3))

from hw3.dataset import Normalizer, SO100ChunkDataset, load_zarr  # noqa: E402
from hw3.eval_utils import load_checkpoint  # noqa: E402
from hw3.sim_env import SO100SimEnv  # noqa: E402
from so101_gym.constants import ASSETS_DIR  # noqa: E402

CKPT = HW3 / "checkpoints/single_cube/best_model_ee_xyz_obstacle.pt"
ZARR = HW3 / "datasets/processed/single_cube/processed_ee_xyz.zarr"
BIN_XY = np.array([-0.2, 0.7])

print("=" * 70)
print("[A] checkpoint 元数据")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
for k in ("epoch", "val_loss", "chunk_size", "policy_type", "state_keys", "action_keys",
          "d_model", "depth", "state_dim", "action_dim"):
    print(f"  {k:12s} = {ckpt.get(k)}")
model, norm, H, sk, ak = load_checkpoint(CKPT, torch.device("cpu"))
print(f"  checkpoint 文件大小: {CKPT.stat().st_size/1024:.0f} KB")

print("=" * 70)
print("[B] 示范数据本身是否成功（cube 终点是否在 bin 里）")
g = zarr.open_group(str(ZARR), mode="r")
cube = np.asarray(g["data"]["state_cube"][:])
ee = np.asarray(g["data"]["state_ee_xyz"][:])
ends = np.asarray(g["meta"]["episode_ends"][:])
starts = np.concatenate(([0], ends[:-1]))
ok = 0
stats = []
for i, (s, e) in enumerate(zip(starts, ends, strict=True)):
    c0, c1 = cube[s, :3], cube[e - 1, :3]
    e1 = ee[e - 1]
    d_xy = np.linalg.norm(c1[:2] - BIN_XY)
    in_bin = d_xy < 0.05 and 0.0 < c1[2] < 0.04
    ok += in_bin
    stats.append(d_xy)
    print(f"  ep{i:02d}: cube起点 {np.round(c0,3)} -> 终点 {np.round(c1,3)} | 终点距bin_xy {d_xy*100:5.1f} cm "
          f"| EE终点距bin_xy {np.linalg.norm(e1[:2]-BIN_XY)*100:5.1f} cm | {'✓在bin' if in_bin else '✗不在bin'}")
print(f"  => 示范成功率: {ok}/{len(ends)} ({ok/len(ends)*100:.0f}%) | 终点距 bin 平均 {np.mean(stats)*100:.1f} cm")

print("=" * 70)
print("[C] 模型在训练数据上的拟合程度（归一化空间）")
states, actions, ep_ends = load_zarr(ZARR, state_keys=sk, action_keys=ak)
ds = SO100ChunkDataset(states, actions, ep_ends, chunk_size=H, normalizer=norm)
loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False)
model.eval()
tot, n, base_tot = 0.0, 0, 0.0
with torch.no_grad():
    for s, a in loader:
        pred = model(s)
        tot += torch.nn.functional.mse_loss(pred, a, reduction="sum").item()
        base_tot += torch.nn.functional.mse_loss(torch.zeros_like(a), a, reduction="sum").item()
        n += a.numel()
print(f"  模型 MSE = {tot/n:.4f} | 全零预测(均值) MSE = {base_tot/n:.4f}")
print(f"  => 可解释方差 = {1 - (tot/n)/(base_tot/n):.1%}（100% 表示完美拟合动作标签）")

# 预测的动作幅度 vs 标签
with torch.no_grad():
    s, a = ds[0]
    p = model(s.unsqueeze(0)).squeeze(0)
    print(f"  第 0 个样本: 标签 chunk 范围 [{a.min():.3f},{a.max():.3f}] | 预测范围 [{p.min():.3f},{p.max():.3f}]")

print("=" * 70)
print("[D] 评估链路单 episode 诊断（2 条）")
from hw3.eval_utils import apply_action, check_cube_out_of_bounds, check_success, infer_action_chunk, obs_to_state  # noqa: E402

env = SO100SimEnv(xml_path=ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml", seed=None)
for ep in range(2):
    obs = env.reset()
    queue = []
    min_ee_cube, min_cube_bin = 9.9, 9.9
    success = False
    for step in range(800):
        if not queue:
            queue.extend(infer_action_chunk(model, norm, obs, sk, torch.device("cpu")))
        a = queue.pop(0)
        ee_p = env.get_ee_pos()
        cube_p = env.get_cube_state()[:3]
        min_ee_cube = min(min_ee_cube, float(np.linalg.norm(ee_p - cube_p)))
        min_cube_bin = min(min_cube_bin, float(np.linalg.norm(cube_p[:2] - BIN_XY)))
        if step % 100 == 0:
            print(f"  ep{ep} step{step:3d}: ee->cube {np.linalg.norm(ee_p-cube_p)*100:5.1f}cm "
                  f"cube->bin {np.linalg.norm(cube_p[:2]-BIN_XY)*100:5.1f}cm "
                  f"cube_z {cube_p[2]:.3f} gripper {env.get_gripper_angle():.2f}")
        apply_action(env, a, ak)
        obs = env.step()
        if check_success(env):
            success = True
            break
        if check_cube_out_of_bounds(env):
            break
    print(f"  ep{ep}: success={success} steps={step+1} | 最近 ee-cube {min_ee_cube*100:.1f}cm "
          f"| 最近 cube-bin(xy) {min_cube_bin*100:.1f}cm")
