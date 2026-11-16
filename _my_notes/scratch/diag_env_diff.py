"""对比「录制环境」和「评估环境」的实际状态分布与时间节奏。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import zarr

HW3 = Path(__file__).resolve().parent.parent / "ethz-robot-learning-2026" / "hw3_imitation_learning"
sys.path.insert(0, str(HW3))

from hw3.sim_env import SO100SimEnv  # noqa: E402
from so101_gym.constants import ASSETS_DIR  # noqa: E402

XML = ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml"

print("=" * 74)
print("[1] 评估环境 reset 的初始状态（seed 0..4）")
for seed in range(5):
    env = SO100SimEnv(xml_path=XML, seed=seed)
    obs = env.reset()
    print(f"  seed={seed}: ee {np.round(obs['ee_pos'],4)} | cube {np.round(obs['cube'][:3],4)} "
          f"| gripper {obs['gripper'][0]:.3f} | obstacle {np.round(obs['obstacle'],4)}")

print("\n" + "=" * 74)
print("[2] 你的示范数据：每条 episode 起始状态（与 eval reset 对比）")
clean = zarr.open_group(str(HW3 / "datasets/raw_clean/single_cube/teleop/so100_transfer_cube_teleop.zarr"), mode="r")
ee = np.asarray(clean["data"]["state_ee"][:])
cube = np.asarray(clean["data"]["state_cube"][:])
grip = np.asarray(clean["data"]["state_gripper"][:]).ravel()
obs_p = np.asarray(clean["data"]["state_obstacle"][:])
ends = np.asarray(clean["meta"]["episode_ends"][:])
starts = np.concatenate(([0], ends[:-1]))
print(f"  共 {len(ends)} 条，起始 EE 统计:")
for name, i in (("ee_x", 0), ("ee_y", 1), ("ee_z", 2)):
    v = ee[starts, i]
    print(f"    {name}: mean {v.mean():.4f} std {v.std():.4f} range [{v.min():.4f},{v.max():.4f}]")
print(f"  起始 cube_xy: x mean {cube[starts,0].mean():.4f} std {cube[starts,0].std():.4f} | "
      f"y mean {cube[starts,1].mean():.4f} std {cube[starts,1].std():.4f}")
print(f"  起始 gripper: mean {grip[starts].mean():.3f} range [{grip[starts].min():.3f},{grip[starts].max():.3f}]")
print(f"  起始 obstacle_x: mean {obs_p[starts,0].mean():.4f} std {obs_p[starts,0].std():.4f}")

print("\n" + "=" * 74)
print("[3] 时间节奏：raw 录制 vs 8mm 抽稀后（夹爪闭合 -> 方块被提起 的间隔）")


def timing(path: Path, label: str) -> None:
    g = zarr.open_group(str(path), mode="r")
    ag = np.asarray(g["data"]["action_gripper"][:]).ravel()
    cb = np.asarray(g["data"]["state_cube"][:])
    e = np.asarray(g["meta"]["episode_ends"][:])
    s = np.concatenate(([0], e[:-1]))
    gaps = []
    for a, b in zip(s, e, strict=True):
        seg_g = ag[a:b]
        seg_z = cb[a:b, 2]
        close_idx = np.where(seg_g < -0.1)[0]
        lift_idx = np.where(seg_z > 0.06)[0]
        if close_idx.size and lift_idx.size:
            c = close_idx[0]
            later = lift_idx[lift_idx > c]
            if later.size:
                gaps.append(later[0] - c)
    print(f"  {label}: 平均间隔 {np.mean(gaps):.1f} 帧 = {np.mean(gaps)*0.1:.2f} s "
          f"(中位 {np.median(gaps):.0f} 帧, 样本 {len(gaps)} 条) | "
          f"每条 episode 平均 {(e-s).mean():.0f} 帧")


timing(HW3 / "datasets/raw/single_cube/teleop/2026-11-05_02-28-53/so100_transfer_cube_teleop.zarr", "raw 原始录制")
timing(HW3 / "datasets/raw_clean/single_cube/teleop/so100_transfer_cube_teleop.zarr", "clean 抽稀后")

print("\n" + "=" * 74)
print("[4] 夹爪闭合到方块提起之间，EE 是否有停顿（学习时序所需）")
g = zarr.open_group(str(HW3 / "datasets/raw_clean/single_cube/teleop/so100_transfer_cube_teleop.zarr"), mode="r")
ag = np.asarray(g["data"]["action_gripper"][:]).ravel()
ee_c = np.asarray(g["data"]["state_ee"][:])
cb = np.asarray(g["data"]["state_cube"][:])
e = np.asarray(g["meta"]["episode_ends"][:])
s = np.concatenate(([0], e[:-1]))
for k in range(min(3, len(e))):
    a, b = s[k], e[k]
    seg_g, seg_ee, seg_z = ag[a:b], ee_c[a:b], cb[a:b, 2]
    close_idx = np.where(seg_g < -0.1)[0]
    if not close_idx.size:
        continue
    c = close_idx[0]
    lift = np.where(seg_z > 0.06)[0]
    lift = lift[lift > c]
    l = lift[0] if lift.size else b - a - 1
    print(f"  ep{k}: close 帧 {c}, lift 帧 {l}, 间隔 {l-c} 帧；"
          f"该区间 EE 位移 {np.linalg.norm(seg_ee[l]-seg_ee[c])*100:.1f} cm, "
          f"夹爪 ctrl {seg_g[c]:.2f} -> {seg_g[l]:.2f}")
