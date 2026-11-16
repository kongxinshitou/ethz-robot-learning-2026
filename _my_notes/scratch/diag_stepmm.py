"""量化：抽稀程度 step_mm 如何改变「夹爪闭合 -> 抬起」的时序（不写任何数据）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import zarr

HW3 = Path(__file__).resolve().parent.parent / "ethz-robot-learning-2026" / "hw3_imitation_learning"
RAW = HW3 / "datasets/raw/single_cube/teleop/2026-11-05_02-28-53/so100_transfer_cube_teleop.zarr"

g = zarr.open_group(str(RAW), mode="r")
ee = np.asarray(g["data"]["state_ee"])[:, :3]
cube = np.asarray(g["data"]["state_cube"])[:, :3]
grip = np.asarray(g["data"]["action_gripper"]).ravel()
ends = np.asarray(g["meta"]["episode_ends"][:])
starts = np.concatenate(([0], ends[:-1]))


def keep_mask(ep_slice, step_mm: float, grip_step: float = 0.02) -> np.ndarray:
    e = ee[ep_slice]
    c = cube[ep_slice]
    gr = grip[ep_slice]
    n = len(e)
    step = step_mm / 1000.0
    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True
    last = 0
    for i in range(1, n - 1):
        if (np.linalg.norm(e[i] - e[last]) >= step
                or np.linalg.norm(c[i] - c[last]) >= step
                or abs(gr[i] - gr[last]) >= grip_step):
            keep[i] = True
            last = i
    return keep


print(f"{'step_mm':>8} {'帧/条':>7} {'总帧数':>8} {'close->lift 平均(s)':>20} {'中位(s)':>9} {'关键段保留帧':>12}")
for step_mm in (2.0, 3.0, 4.0, 6.0, 8.0, 12.0):
    keep_all = []
    gaps = []
    total = 0
    for s, e in zip(starts, ends, strict=True):
        sl = slice(s, e)
        keep = keep_mask(sl, step_mm)
        idx = np.where(keep)[0]
        total += len(idx)
        keep_all.append(len(idx))
        seg_z = cube[sl][:, 2]
        seg_g = grip[sl]
        close = idx[seg_g[idx] < -0.1]
        if close.size:
            c0 = close[0]
            lift = idx[(seg_z[idx] > 0.06) & (idx > c0)]
            if lift.size:
                pos_c = np.where(idx == c0)[0][0]
                pos_l = np.where(idx == lift[0])[0][0]
                gaps.append(pos_l - pos_c)
    # 原始数据下 close->lift 的原始帧间隔做对照
    print(f"{step_mm:>8.0f} {np.mean(keep_all):>7.0f} {total:>8d} "
          f"{np.mean(gaps)*0.1:>20.2f} {np.median(gaps)*0.1:>9.2f} {np.mean(gaps):>12.1f}")

print("\n原始（不抽稀）:")
gaps_raw = []
lens = []
for s, e in zip(starts, ends, strict=True):
    seg_z = cube[s:e, 2]
    seg_g = grip[s:e]
    close = np.where(seg_g < -0.1)[0]
    if close.size:
        c0 = close[0]
        lift = np.where((seg_z > 0.06) & (np.arange(len(seg_z)) > c0))[0]
        if lift.size:
            gaps_raw.append(lift[0] - c0)
    lens.append(e - s)
print(f"  帧/条 {np.mean(lens):.0f} | close->lift 平均 {np.mean(gaps_raw)*0.1:.2f} s, "
      f"中位 {np.median(gaps_raw)*0.1:.2f} s")

print("\n原始数据里相邻帧 EE 位移（说明录制时人走得多慢）:")
d = np.linalg.norm(np.diff(ee, axis=0), axis=1)
d = d[np.isfinite(d)]
d = d[d < 0.05]
print(f"  中位 {np.median(d)*1000:.2f} mm/帧 | 90分位 {np.percentile(d, 90)*1000:.2f} mm/帧 | "
      f"平均 {d.mean()*1000:.2f} mm/帧")
