"""Clean HW3 teleop recordings: drop failed episodes and thin out static frames.

A frame is kept when the robot has moved about `--step-mm` (default 8 mm) since
the last kept frame, or when the gripper command changed.  Idle stretches are
therefore removed while the motion itself is preserved, so each kept transition
is roughly one 1 cm teleop press.

Output keeps the official zarr schema, so it can be fed straight into
``scripts/compute_actions.py``:

    .venv\\Scripts\\python.exe ..\\..\\hw3_clean_data.py
    .venv\\Scripts\\python.exe scripts\\compute_actions.py ^
        --action-space ee --datasets-dir datasets\\raw_clean\\single_cube
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import zarr

WORKSPACE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE))

from hw3_episode_inspect import load_episodes  # noqa: E402
from hw3.teleop_utils import ZarrEpisodeWriter  # noqa: E402

BIN_XY = np.array([-0.2, 0.7])


def keep_mask(ep: dict, step_mm: float = 8.0, grip_step: float = 0.02) -> np.ndarray:
    """Keep the frames at which the robot has moved ~step_mm since the last kept frame.

    This is distance-based downsampling: it removes idle/duplicate frames but
    keeps the motion itself, so each kept transition is roughly one 1 cm teleop
    press instead of "press + many waiting frames".
    """
    ee = ep["ee"][:, :3]
    cube = ep["cube"][:, :3]
    grip = ep["grip"]
    n = len(ee)
    step = step_mm / 1000.0

    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True
    last = 0
    for i in range(1, n - 1):
        d_ee = float(np.linalg.norm(ee[i] - ee[last]))
        d_cube = float(np.linalg.norm(cube[i] - cube[last]))
        d_grip = abs(float(grip[i] - grip[last]))
        if d_ee >= step or d_cube >= step or d_grip >= grip_step or (i - last) >= 5:
            keep[i] = True
            last = i

    if keep.sum() < 20:  # safety for pathologically short episodes
        keep[:] = False
        keep[np.linspace(0, n - 1, min(n, 20)).astype(int)] = True
    return keep


def episode_metrics(ep: dict) -> dict:
    cube = ep["cube"][:, :3]
    grip = ep["grip"]
    cube_end = cube[-1]
    in_xy = np.linalg.norm(cube_end[:2] - BIN_XY) < 0.05
    in_z = 0.0 < cube_end[2] < 0.04
    lifted = cube[:, 2].max() > 0.06
    tail = cube[-10:]
    stable = bool(np.max(np.linalg.norm(np.diff(tail, axis=0), axis=1)) < 3e-3) if len(tail) > 2 else False
    return {"lift": bool(lifted), "bin_ok": bool(in_xy and in_z), "stable": stable,
            "steps": len(cube), "grip_open_frac": float(np.mean(grip > 1.0))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean HW3 teleop data.")
    parser.add_argument("--input", type=Path, default=Path("datasets/raw/single_cube/teleop"))
    parser.add_argument("--output", type=Path, default=Path("datasets/raw_clean/single_cube/teleop"))
    parser.add_argument("--step-mm", type=float, default=8.0,
                        help="Keep a frame each time the robot has moved this far (mm).")
    parser.add_argument("--drop-bad", action="store_true", default=True)
    parser.add_argument("--keep-bad", dest="drop_bad", action="store_false")
    args = parser.parse_args()

    eps = load_episodes(args.input)
    if not eps:
        print(f"[ERROR] no episodes under {args.input}")
        return 1

    run_dir = args.output
    run_dir.mkdir(parents=True, exist_ok=True)
    out_zarr = run_dir / "so100_transfer_cube_teleop.zarr"

    writer = ZarrEpisodeWriter(
        out_zarr, joint_dim=6, ee_dim=7, cube_dim=7, gripper_dim=1, obstacle_dim=3,
        flush_every=12,
    )
    try:
        src = zarr.open_group(str(next(args.input.rglob("*.zarr"))), mode="r")
        attrs = {k: v for k, v in dict(src.attrs).items()}
        attrs["cleaned_from"] = str(args.input)
        attrs["step_mm"] = float(args.step_mm)
        writer.set_attrs(**attrs)
    except StopIteration:
        pass

    print(f"{'idx':>3s} {'ep':>2s} {'before':>6s} {'after':>5s} {'keep%':>5s} "
          f"{'lift':>4s} {'bin':>3s} {'stbl':>4s}  action")
    kept_eps = 0
    before_total = after_total = 0
    for i, ep in enumerate(eps):
        m = episode_metrics(ep)
        mask = keep_mask(ep, step_mm=args.step_mm)
        n_before, n_after = ep["n"], int(mask.sum())
        drop = args.drop_bad and (not m["lift"] or not m["bin_ok"] or n_before < 40)
        action = "DROP" if drop else "keep"
        print(f"{i:3d} {ep['ep']:2d} {n_before:6d} {n_after:5d} {100.0 * n_after / n_before:5.1f} "
              f"{'Y' if m['lift'] else '.':>4s} {'Y' if m['bin_ok'] else '.':>3s} "
              f"{'Y' if m['stable'] else '.':>4s}  {action}")
        before_total += n_before
        if drop:
            continue
        kept_eps += 1
        after_total += n_after
        k = np.where(mask)[0]
        for f in k:
            writer.append(
                ep["joints"][f].astype(np.float32),
                ep["ee"][f].astype(np.float32),
                ep["cube"][f].astype(np.float32),
                np.array([ep["joints"][f][5]], dtype=np.float32),
                np.array([ep["grip"][f]], dtype=np.float32),
                ep["obstacle"][f].astype(np.float32),
            )
        writer.end_episode()

    writer.flush()
    print(f"\nepisodes: {len(eps)} -> {kept_eps} kept")
    print(f"steps   : {before_total} -> {after_total} "
          f"({100.0 * after_total / max(before_total, 1):.1f}%)")
    print(f"output  : {out_zarr}")

    meta = {
        "source": str(args.input),
        "step_mm": float(args.step_mm),
        "episodes_in": len(eps),
        "episodes_out": kept_eps,
        "steps_in": before_total,
        "steps_out": after_total,
        "control_hz": 10.0,
    }
    (run_dir / "clean_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
