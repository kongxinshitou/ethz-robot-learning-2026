"""分析新录制的数据：停顿比例、episode 长度、是否成功、与旧数据的对比。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import zarr

HW3 = Path(__file__).resolve().parent.parent / "ethz-robot-learning-2026" / "hw3_imitation_learning"
BIN_XY = np.array([-0.2, 0.7])


def analyze(path: Path, label: str) -> None:
    g = zarr.open_group(str(path), mode="r")
    ee = np.asarray(g["data"]["state_ee_xyz"] if "state_ee_xyz" in g["data"] else g["data"]["state_ee"])[:, :3]
    cube = np.asarray(g["data"]["state_cube"])[:, :3]
    act = np.asarray(g["data"]["action_ee_xyz"])
    ag = np.asarray(g["data"]["action_gripper"]).ravel()
    ends = np.asarray(g["meta"]["episode_ends"][:])
    starts = np.concatenate(([0], ends[:-1]))
    lens = ends - starts
    d = np.linalg.norm(np.diff(ee, axis=0), axis=1)
    print(f"\n===== {label} =====")
    print(f"  episodes={len(ends)} steps={ends[-1]} | 每条长度 mean {lens.mean():.0f} "
          f"min {lens.min()} max {lens.max()}")
    print(f"  相邻帧 EE 位移: 中位 {np.median(d)*1000:.2f} mm | 平均 {d.mean()*1000:.2f} mm "
          f"| <1mm 的帧占比 {np.mean(d < 0.001):.1%}")
    print(f"  action 均值/标准差: {np.round(act.mean(0),5)} / {np.round(act.std(0),5)}")
    print(f"  |action|<0.5mm 的帧占比: {np.mean(np.linalg.norm(act, axis=1) < 0.0005):.1%}")
    ok = lift = 0
    close_to_lift = []
    for i, (s, e) in enumerate(zip(starts, ends, strict=True)):
        c_end = cube[e - 1]
        in_bin = np.linalg.norm(c_end[:2] - BIN_XY) < 0.05 and 0 < c_end[2] < 0.04
        lifted = cube[s:e, 2].max() > 0.06
        ok += in_bin
        lift += lifted
        seg_g = ag[s:e]
        seg_z = cube[s:e, 2]
        closed = np.where(seg_g < -0.1)[0]
        if closed.size:
            later = np.where((seg_z > 0.06) & (np.arange(len(seg_z)) > closed[0]))[0]
            if later.size:
                close_to_lift.append(later[0] - closed[0])
    print(f"  示范成功(bin) {ok}/{len(ends)} | 抬起过 {lift}/{len(ends)} | "
          f"close->lift 平均 {np.mean(close_to_lift) if close_to_lift else float('nan'):.0f} 帧 "
          f"({np.mean(close_to_lift)*0.1 if close_to_lift else 0:.1f}s)")


analyze(HW3 / "datasets/processed/single_cube/processed_ee_xyz_new.zarr", "新数据 processed_ee_xyz_new (未清洗)")
analyze(HW3 / "datasets/processed/single_cube/processed_ee_xyz.zarr", "旧数据 processed_ee_xyz (8mm 清洗)")

print("\n===== 当前 checkpoint =====")
ck = torch.load(HW3 / "checkpoints/single_cube/best_model_ee_xyz_obstacle.pt", map_location="cpu", weights_only=False)
for k in ("epoch", "val_loss", "chunk_size", "d_model", "depth"):
    print(f"  {k:10s} = {ck.get(k)}")

print("\n===== raw 录制目录 =====")
for p in sorted((HW3 / "datasets/raw/single_cube/teleop").iterdir()):
    if p.is_dir():
        zs = list(p.rglob("*.zarr"))
        try:
            z = zarr.open_group(str(zs[0]), mode="r")
            e = np.asarray(z["meta"]["episode_ends"][:])
            print(f"  {p.name}: {len(e)} eps, {int(e[-1])} steps, {e[-1]/len(e):.0f} steps/ep")
        except Exception as exc:  # noqa: BLE001
            print(f"  {p.name}: {exc}")
