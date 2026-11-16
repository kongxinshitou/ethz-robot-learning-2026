"""Per-episode quality report for HW3 teleop recordings.

Run from ethz-robot-learning-2026/hw3_imitation_learning with its venv:
    .venv\\Scripts\\python.exe ..\\..\\hw3_data_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import zarr

BIN_XY = np.array([-0.2, 0.7])


def analyse_store(path: Path) -> list[dict]:
    root = zarr.open_group(str(path), mode="r")
    try:
        ep_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=int)
    except Exception:
        return []
    if ep_ends.size == 0:
        return []
    data = root["data"]
    ee = np.asarray(data["state_ee"][:])
    cube = np.asarray(data["state_cube"][:])
    grip = np.asarray(data["action_gripper"][:]).reshape(-1)

    rows = []
    start = 0
    for ep_i, end in enumerate(ep_ends):
        end = int(end)
        seg_ee = ee[start:end]
        seg_cube = cube[start:end]
        seg_grip = grip[start:end]
        n = end - start
        steps = max(1, n - 1)
        d_ee = np.linalg.norm(np.diff(seg_ee[:, :3], axis=0), axis=1) if n > 1 else np.zeros(1)
        cube_start = seg_cube[0, :3]
        cube_end = seg_cube[-1, :3]
        in_xy = np.linalg.norm(cube_end[:2] - BIN_XY) < 0.05
        in_z = 0.0 < cube_end[2] < 0.04
        rows.append({
            "store": path.parent.name,
            "ep": ep_i + 1,
            "steps": n,
            "dur_s": n / 10.0,
            "cube_dxy": float(np.linalg.norm(cube_end[:2] - cube_start[:2])),
            "cube_max_z": float(seg_cube[:, 2].max()),
            "cube_end": cube_end,
            "static_frac": float(np.mean(d_ee < 1e-3)),
            "mean_speed": float(np.mean(d_ee)),
            "move_steps": int(np.sum(d_ee > 0.004)),
            "p90_speed": float(np.percentile(d_ee, 90)),
            "grip_min": float(seg_grip.min()),
            "grip_max": float(seg_grip.max()),
            "grip_open_frac": float(np.mean(seg_grip > 1.0)),
            "success": bool(in_xy and in_z),
        })
        start = end
    return rows


def main() -> int:
    base = Path("datasets/raw/single_cube/teleop")
    stores = sorted(p for p in base.rglob("*.zarr") if p.is_dir())
    if not stores:
        print("[ERROR] run from the hw3_imitation_learning folder; no teleop zarr found")
        return 1

    all_rows: list[dict] = []
    print(f"{'store':20s} {'eps':>3s} {'steps':>6s}  episodes")
    for store in stores:
        rows = analyse_store(store)
        all_rows.extend(rows)
        total_steps = sum(r["steps"] for r in rows)
        print(f"{store.parent.name:20s} {len(rows):3d} {total_steps:6d}")

    print(f"\n=== {len(all_rows)} episodes total ===")
    print(
        f"{'store':14s} {'ep':>2s} {'steps':>5s} {'dur':>5s} {'dxy':>6s} "
        f"{'zmax':>6s} {'still':>6s} {'move':>5s} {'p90':>6s} {'grip':>12s} {'ok':>3s} flags"
    )
    n_ok = 0
    for r in all_rows:
        flags = []
        if r["steps"] < 40:
            flags.append("too_short")
        if r["steps"] > 400:
            flags.append("too_long")
        if r["cube_max_z"] < 0.06:
            flags.append("cube_not_lifted")
        if r["static_frac"] > 0.5:
            flags.append("much_still")
        if r["grip_max"] - r["grip_min"] < 0.5:
            flags.append("gripper_unused")
        if r["success"]:
            n_ok += 1
        print(
            f"{r['store']:14s} {r['ep']:2d} {r['steps']:5d} {r['dur_s']:4.1f}s "
            f"{r['cube_dxy']:6.3f} {r['cube_max_z']:6.3f} {r['static_frac']:6.2f} "
            f"{r['move_steps']:5d} {r['p90_speed']:6.4f} "
            f"{r['grip_min']:+.2f}..{r['grip_max']:+.2f} "
            f"{'Y' if r['success'] else '.':>3s} {','.join(flags)}"
        )
    print(f"\nlooks like successful demos: {n_ok}/{len(all_rows)}")
    print(
        f"total steps {sum(r['steps'] for r in all_rows)}, "
        f"moving steps {sum(r['move_steps'] for r in all_rows)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

