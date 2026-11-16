"""Clean DAgger episodes: remove near-static frames while keeping every episode.

This is the DAgger-specific companion of ``hw3_clean_data.py``:

* it does **not** drop "failed" episodes (DAgger takeover segments are valuable);
* it does **not** use a hard-coded bin position / success check;
* it only thins out frames where the robot, the cube and the gripper are all
  essentially not moving, so the remaining transitions carry motion;
* it preserves the official raw zarr schema, so the output can be fed directly
  into ``scripts/compute_actions.py``.

Frame is kept when, compared with the last kept frame, any of these holds:
  * EE moved >= --step-mm
  * cube moved >= --step-mm
  * gripper command changed >= --grip-step
  * at least --max-gap frames passed (bounds the time gap; 0 disables)

Run from ``ethz-robot-learning-2026/hw3_imitation_learning``:

    .venv\\Scripts\\python.exe ..\\..\\hw3_clean_dagger.py --dry-run
    .venv\\Scripts\\python.exe ..\\..\\hw3_clean_dagger.py ^
        --input datasets\\raw_clean_v3\\single_cube\\dagger ^
        --stage-dir datasets\\raw_clean_static\\single_cube ^
        --teleop datasets\\raw_clean_v3\\single_cube\\teleop

The ``--stage-dir`` mode writes:

    <stage>/dagger/so100_transfer_cube_teleop.zarr   (cleaned)
    <stage>/teleop/...                               (copied from --teleop)

so you can immediately run:

    .venv\\Scripts\\python.exe scripts\\compute_actions.py ^
        --action-space ee --datasets-dir datasets\\raw_clean_static\\single_cube
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import zarr

WORKSPACE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE))

from hw3.teleop_utils import ZarrEpisodeWriter  # noqa: E402

DEFAULT_INPUT = Path("datasets/raw_clean_v3/single_cube/dagger")
DEFAULT_OUTPUT = Path(
    "datasets/raw_clean_dagger/single_cube/dagger/so100_transfer_cube_teleop.zarr"
)

REQUIRED_ARRAYS = (
    "state_joints",
    "state_ee",
    "state_cube",
    "action_gripper",
    "state_obstacle",
)


def discover_stores(path: Path) -> list[Path]:
    """Return the .zarr stores under *path* (or *path* itself if it is a store)."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        raise NotADirectoryError(f"--input must be a directory or a .zarr store: {path}")
    # a zarr v3 store contains zarr.json at its root
    if (path / "zarr.json").exists():
        return [path]
    stores = sorted(p for p in path.rglob("*.zarr") if p.is_dir())
    if not stores:
        raise FileNotFoundError(f"no .zarr stores under {path}")
    return stores


def load_episodes(stores: list[Path]) -> list[dict]:
    """Load every episode from the given raw zarr stores."""
    episodes: list[dict] = []
    for store in stores:
        root = zarr.open_group(str(store), mode="r")
        try:
            ep_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
        except Exception:
            print(f"  [skip] {store} (no meta/episode_ends)")
            continue
        if ep_ends.size == 0:
            print(f"  [skip] {store} (0 episodes; interrupted recording?)")
            continue
        data = root["data"]
        missing = [k for k in REQUIRED_ARRAYS if k not in data]
        if missing:
            print(f"  [skip] {store} (missing arrays: {missing})")
            continue
        arrays = {
            "joints": np.asarray(data["state_joints"][:], dtype=np.float64),
            "ee": np.asarray(data["state_ee"][:], dtype=np.float64),
            "cube": np.asarray(data["state_cube"][:], dtype=np.float64),
            "grip": np.asarray(data["action_gripper"][:], dtype=np.float64).reshape(-1),
            "obstacle": np.asarray(data["state_obstacle"][:], dtype=np.float64),
        }
        n_total = arrays["joints"].shape[0]
        last_end = int(ep_ends[-1])
        if last_end > n_total:
            print(
                f"  [warn] {store}: episode_ends[-1]={last_end} > rows={n_total}; "
                "clipping to available rows"
            )
            ep_ends = ep_ends[ep_ends <= n_total]
            if ep_ends.size == 0:
                continue
        start = 0
        for ep_i, end in enumerate(ep_ends):
            end = int(end)
            if end <= start:
                continue
            ep = {k: v[start:end] for k, v in arrays.items()}
            ep.update(
                {
                    "store": str(store),
                    "store_name": store.parent.name or store.name,
                    "ep": ep_i + 1,
                    "n": end - start,
                }
            )
            episodes.append(ep)
            start = end
    return episodes


def keep_mask(
    ep: dict,
    step_mm: float = 4.0,
    grip_step: float = 0.02,
    max_gap: int = 10,
    min_frames: int = 8,
) -> np.ndarray:
    """Distance-based static-frame filter (keep first/last, no episode dropping)."""
    ee = ep["ee"][:, :3]
    cube = ep["cube"][:, :3]
    grip = ep["grip"]
    n = len(ee)
    if n <= 2:
        return np.ones(n, dtype=bool)

    step = step_mm / 1000.0
    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True
    last = 0
    for i in range(1, n - 1):
        d_ee = float(np.linalg.norm(ee[i] - ee[last]))
        d_cube = float(np.linalg.norm(cube[i] - cube[last]))
        d_grip = abs(float(grip[i] - grip[last]))
        forced = max_gap > 0 and (i - last) >= max_gap
        if d_ee >= step or d_cube >= step or d_grip >= grip_step or forced:
            keep[i] = True
            last = i

    if keep.sum() < min_frames:
        # Very short / almost-static episode: keep evenly spaced frames so the
        # episode still contributes at least a few consecutive transitions.
        idx = np.linspace(0, n - 1, min(n, min_frames)).astype(int)
        keep[:] = False
        keep[idx] = True
        keep[0] = True
        keep[-1] = True
    return keep


def action_stats(ep: dict, keep_idx: np.ndarray) -> tuple[float, float, float]:
    """Mean / p50 / fraction-static of EE deltas between consecutive kept frames."""
    ee = ep["ee"][keep_idx, :3]
    if len(ee) < 2:
        return 0.0, 0.0, 0.0
    d = np.linalg.norm(np.diff(ee, axis=0), axis=1)
    return float(d.mean()), float(np.percentile(d, 50)), float(np.mean(d < 1e-4))


def copy_teleop(teleop: Path, stage_dir: Path) -> None:
    """Copy teleop raw zarr store(s) into <stage>/teleop, preserving layout."""
    dst_root = stage_dir / "teleop"
    dst_root.mkdir(parents=True, exist_ok=True)
    if not teleop.exists():
        raise FileNotFoundError(f"--teleop not found: {teleop}")
    if (teleop / "zarr.json").exists():  # a single .zarr store
        stores = [teleop]
        base = teleop.parent
    else:
        stores = sorted(p for p in teleop.rglob("*.zarr") if p.is_dir())
        base = teleop
    if not stores:
        raise FileNotFoundError(f"no .zarr stores under --teleop {teleop}")
    for store in stores:
        rel = store.relative_to(base) if store != base else Path(store.name)
        dst = dst_root / rel
        if dst.exists():
            print(f"  [teleop] already staged, skip: {dst}")
            continue
        shutil.copytree(store, dst)
        print(f"  [teleop] copied {store} -> {dst}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove near-static frames from DAgger raw data "
        "(keeps every episode, writes official raw schema)."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"Directory containing dagger .zarr stores, or one .zarr store "
                             f"(default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=Path, default=None,
                        help=f"Output .zarr path (default: {DEFAULT_OUTPUT}; "
                             "ignored in --stage-dir mode)")
    parser.add_argument("--stage-dir", type=Path, default=None,
                        help="Write cleaned dagger to <stage>/dagger/... and copy teleop "
                             "to <stage>/teleop/ so compute_actions can run on <stage>")
    parser.add_argument("--teleop", type=Path, default=None,
                        help="Teleop raw dir/store to copy into <stage>/teleop "
                             "(only used with --stage-dir)")
    parser.add_argument("--step-mm", type=float, default=4.0,
                        help="Keep a frame when EE/cube moved this far since last kept "
                             "frame (default: 4 mm)")
    parser.add_argument("--grip-step", type=float, default=0.02,
                        help="Keep a frame when gripper command changed this much "
                             "(default: 0.02)")
    parser.add_argument("--max-gap", type=int, default=10,
                        help="Force-keep a frame at least every N frames, bounding time "
                             "gaps (default: 10; 0 disables)")
    parser.add_argument("--min-frames", type=int, default=8,
                        help="Fallback kept-frame count for almost-static episodes "
                             "(default: 8)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing an existing output .zarr")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print stats; write nothing")
    args = parser.parse_args()

    stores = discover_stores(args.input)
    print(f"Found {len(stores)} store(s):")
    for s in stores:
        print(f"  {s}")
    episodes = load_episodes(stores)
    if not episodes:
        print("[ERROR] no usable episodes")
        return 1

    if args.stage_dir is not None:
        stage_dir = args.stage_dir
        out_zarr = stage_dir / "dagger" / "so100_transfer_cube_teleop.zarr"
    else:
        out_zarr = args.output or DEFAULT_OUTPUT

    print(
        f"\nFilter: step_mm={args.step_mm}, grip_step={args.grip_step}, "
        f"max_gap={args.max_gap}, min_frames={args.min_frames}"
    )
    print(f"Output: {out_zarr}{'  [dry-run]' if args.dry_run else ''}")

    rows = []
    total_before = total_after = 0
    for ep in episodes:
        mask = keep_mask(
            ep,
            step_mm=args.step_mm,
            grip_step=args.grip_step,
            max_gap=args.max_gap,
            min_frames=args.min_frames,
        )
        idx = np.where(mask)[0]
        n_before, n_after = ep["n"], len(idx)
        total_before += n_before
        total_after += n_after
        mean_a, p50_a, static_a = action_stats(ep, idx)
        rows.append((ep, idx, n_before, n_after, mean_a, p50_a, static_a))
        print(
            f"  {Path(ep['store']).name:38s} ep{ep['ep']:02d} "
            f"{n_before:5d} -> {n_after:5d} ({100.0 * n_after / n_before:5.1f}%) "
            f"| kept |dEE| mean={mean_a * 1000:5.2f}mm "
            f"p50={p50_a * 1000:5.2f}mm static={100 * static_a:4.1f}%"
        )

    print(
        f"\nTotal: {len(episodes)} episodes, {total_before} -> {total_after} steps "
        f"({100.0 * total_after / max(total_before, 1):.1f}% kept)"
    )
    if args.dry_run:
        print("Dry run: nothing written.")
        return 0

    if out_zarr.exists():
        if not args.overwrite:
            print(f"[ERROR] output exists: {out_zarr}\n        pass --overwrite to replace it")
            return 1
        shutil.rmtree(out_zarr)

    out_zarr.parent.mkdir(parents=True, exist_ok=True)
    if args.stage_dir is not None and args.teleop is not None:
        copy_teleop(args.teleop, args.stage_dir)

    writer = ZarrEpisodeWriter(
        out_zarr,
        joint_dim=6,
        ee_dim=7,
        cube_dim=7,
        gripper_dim=1,
        obstacle_dim=3,
        flush_every=12,
    )
    writer.set_attrs(
        cleaned_from=[str(s) for s in stores],
        step_mm=float(args.step_mm),
        grip_step=float(args.grip_step),
        max_gap=int(args.max_gap),
        min_frames=int(args.min_frames),
        episodes_in=len(episodes),
        steps_in=int(total_before),
        steps_out=int(total_after),
        source_kind="dagger",
    )

    kept_eps = 0
    for ep, idx, *_ in rows:
        if len(idx) < 2:
            print(f"  [skip] {Path(ep['store']).name} ep{ep['ep']:02d}: <2 frames after filter")
            continue
        joints = ep["joints"]
        for f in idx:
            writer.append(
                joints[f],
                ep["ee"][f],
                ep["cube"][f],
                np.array([joints[f][5]], dtype=np.float32),  # state_gripper = Jaw angle
                np.array([ep["grip"][f]], dtype=np.float32),  # action_gripper = recorded ctrl
                ep["obstacle"][f],
            )
        writer.end_episode()
        kept_eps += 1

    writer.flush()
    print(f"\nWrote {kept_eps} episodes / {writer.num_steps_total} steps -> {out_zarr}")
    if args.stage_dir is not None:
        print(
            "\nNext (explicit --output avoids overwriting the existing processed zarr):\n"
            f"  .venv\\Scripts\\python.exe scripts\\compute_actions.py "
            f"--action-space ee --datasets-dir {args.stage_dir} "
            f"--output datasets\\processed\\single_cube\\processed_ee_xyz_staticclean.zarr"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
