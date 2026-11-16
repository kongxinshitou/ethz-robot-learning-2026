"""Ex3 (multicube) teleop data cleaner: remove static frames, drop bad episodes.

What this fixes
---------------
1. **Static frames** — teleop is recorded at 10 Hz, but keypresses only move the
   arm in ~6-7 mm bursts.  ~70% of the raw frames are "nothing changed" frames
   (``s_{t+1} == s_t``).  Training on them biases the policy towards standing
   still (see HW3 README).  A frame is kept only when, compared to the last
   kept frame:
     - EE position moved >= ``--step-mm``       (default 5 mm)
     - EE orientation rotated >= ``--rot-deg``  (default 3 deg)
     - gripper command changed >= ``--grip-step`` (default 0.02)
     - any cube moved >= ``--cube-mm``          (default 2 mm)
   The first and last frame of every episode are always kept.

2. **Reset-contaminated episodes** — if you press ``record`` off, press
   ``reset`` (or change the goal cube), then press ``record`` on again, the new
   frames land in the *same* episode.  The result is a single episode with two
   different layouts / bin positions / goal labels.  There is no reliable way
   to know where the operator "meant" to start, so such an episode is
   **dropped as a whole** (detected by: goal one-hot change, bin jump,
   >= 2 cubes teleporting, or an end-effector teleport inside the episode).
   Re-record it as a fresh episode instead.

3. **Bad episodes** — episodes that never pick the target cube, put a
   non-target cube into the bin, end while still holding the cube, contain NaN,
   or are too short are dropped (``--keep-bad`` disables this).

Output keeps the official multicube zarr schema (same array names, dtypes,
``meta/episode_ends``, root attrs), so it plugs straight into
``scripts/compute_actions.py``.

Usage (from ``ethz-robot-learning-2026/hw3_imitation_learning``):

    .venv\\Scripts\\python.exe ..\\..\\hw3_ex3_clean_static.py --dry-run   # preview
    .venv\\Scripts\\python.exe ..\\..\\hw3_ex3_clean_static.py

then:

    .venv\\Scripts\\python.exe scripts\\compute_actions.py ^
        --action-space ee --datasets-dir datasets\\raw_clean\\multi_cube
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import zarr

CUBE_COLORS = ("red", "green", "blue")
CUBE_KEYS = tuple(f"pos_cube_{c}" for c in CUBE_COLORS)
REQUIRED_KEYS = (
    "state_joints",
    "state_ee",
    "state_gripper",
    "action_gripper",
    "state_obstacle",
    "pos_cube_red",
    "pos_cube_green",
    "pos_cube_blue",
    "state_goal",
    "goal_pos",
)

# validation thresholds (match hw3/eval_utils.py check_success / wrong cube)
BIN_XY_TOL = 0.04
BIN_Z_TOL = 0.04
LIFT_Z = 0.06
RELEASE_GRIP = 0.5


# ── geometry helpers ─────────────────────────────────────────────────────────

def quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angle (deg) between two unit wxyz quaternions, robust to double cover."""
    d = float(np.clip(abs(np.dot(q1, q2)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(d)))


# ── loading ──────────────────────────────────────────────────────────────────

def load_stores(input_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray, list[Path], dict]:
    """Load and concatenate every .zarr under *input_dir*.

    Episode ends are shifted by the running offset, like
    ``compute_actions.load_and_merge_zarrs`` does.
    """
    stores = sorted(p for p in input_dir.rglob("*.zarr") if p.is_dir())
    if not stores:
        raise SystemExit(f"[ERROR] no .zarr store under {input_dir}")

    merged: dict[str, list[np.ndarray]] = {}
    all_ends: list[np.ndarray] = []
    offset = 0
    attrs: dict = {}
    for store in stores:
        root = zarr.open_group(str(store), mode="r")
        data = root["data"]
        keys = sorted(data.array_keys())
        missing = [k for k in REQUIRED_KEYS if k not in keys]
        if missing:
            raise SystemExit(
                f"[ERROR] {store} is not a multicube recording, missing arrays: {missing}\n"
                "        Record with: python scripts/record_teleop_demos.py --multicube"
            )
        ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
        n = int(ends[-1]) if ends.size else 0
        for k in keys:
            arr = np.asarray(data[k][:n])
            merged.setdefault(k, []).append(arr)
        all_ends.append(ends + offset)
        offset += n
        if not attrs:
            attrs = dict(root.attrs)
        print(f"  loaded {store}  ({ends.size} ep, {n} steps)")

    arrays = {k: np.concatenate(v, axis=0) for k, v in merged.items()}
    episode_ends = np.concatenate(all_ends, axis=0)
    lengths = {k: int(v.shape[0]) for k, v in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise SystemExit(f"[ERROR] array lengths disagree: {lengths}")
    if int(episode_ends[-1]) != next(iter(lengths.values())):
        raise SystemExit(
            f"[ERROR] episode_ends[-1]={episode_ends[-1]} != steps={next(iter(lengths.values()))}"
        )
    return arrays, episode_ends, stores, attrs


# ── reset-contamination detection ────────────────────────────────────────────

def find_discontinuities(
    arrays: dict[str, np.ndarray],
    start: int,
    end: int,
    *,
    bin_jump_mm: float,
    cube_jump_mm: float,
    ee_jump_mm: float,
) -> list[tuple[int, str]]:
    """Return ``[(local_frame, description), ...]`` of reset contamination.

    A jump that only an environment reset can explain means the recording was
    paused, the scene was reset / the goal was switched, and recording resumed
    into the same episode.  Such an episode is dropped as a whole - we do not
    try to split it into segments.
    """
    ee = arrays["state_ee"][:, :3]
    goal = arrays["state_goal"]
    goal_pos = arrays["goal_pos"]
    cubes = [arrays[k][:, :3] for k in CUBE_KEYS]
    bin_jump = bin_jump_mm / 1000.0
    cube_jump = cube_jump_mm / 1000.0
    ee_jump = ee_jump_mm / 1000.0

    events: list[tuple[int, str]] = []
    for i in range(start + 1, end):
        if np.any(np.abs(goal[i] - goal[i - 1]) > 1e-6):
            prev = CUBE_COLORS[int(np.argmax(goal[i - 1]))]
            cur = CUBE_COLORS[int(np.argmax(goal[i]))]
            events.append((i - start, f"goal switched {prev}->{cur}"))
            continue
        d_bin = float(np.linalg.norm(goal_pos[i] - goal_pos[i - 1]))
        if d_bin > bin_jump:
            events.append((i - start, f"bin jumped {d_bin * 1000:.0f} mm"))
            continue
        jumps = [float(np.linalg.norm(c[i] - c[i - 1])) for c in cubes]
        n_jump = sum(d > cube_jump for d in jumps)
        if n_jump >= 2:
            events.append(
                (i - start, f"{n_jump} cubes teleported (max {max(jumps) * 1000:.0f} mm)")
            )
            continue
        d_ee = float(np.linalg.norm(ee[i] - ee[i - 1]))
        if d_ee > ee_jump:
            events.append((i - start, f"EE teleported {d_ee * 1000:.0f} mm"))
    return events


# ── per-episode validation ───────────────────────────────────────────────────

def validate_episode(
    arrays: dict[str, np.ndarray],
    start: int,
    end: int,
    *,
    bin_jump_mm: float,
    cube_jump_mm: float,
    ee_jump_mm: float,
) -> tuple[list[str], dict]:
    """Return (failure codes, info) for one raw episode."""
    goal = arrays["state_goal"][start:end]
    goal_pos = arrays["goal_pos"][start:end]
    grip = arrays["action_gripper"][start:end, 0]
    ee = arrays["state_ee"][start:end, :3]
    cubes = {c: arrays[f"pos_cube_{c}"][start:end, :3] for c in CUBE_COLORS}

    info: dict = {"target": CUBE_COLORS[int(np.argmax(goal[0]))]}
    fail: list[str] = []

    # reset contamination: drop the whole episode, never try to salvage it
    events = find_discontinuities(
        arrays, start, end,
        bin_jump_mm=bin_jump_mm, cube_jump_mm=cube_jump_mm, ee_jump_mm=ee_jump_mm,
    )
    info["reset_events"] = events
    if events:
        fail.append("reset_contamination")

    # goal one-hot must be exactly one-hot and constant
    sums = goal.sum(axis=1)
    if not (np.all(np.abs(sums - 1.0) < 1e-5) and np.allclose(goal, goal[0], atol=1e-6)):
        fail.append("goal_moves")
    # bin must not move inside an episode
    if float(np.max(np.ptp(goal_pos, axis=0))) > 1e-6:
        fail.append("bin_moves")
    # no NaN / inf
    for key in REQUIRED_KEYS:
        if not np.all(np.isfinite(arrays[key][start:end])):
            fail.append("nan")
            break
    # no teleport inside the episode
    if end - start > 1:
        ee_d = np.linalg.norm(np.diff(ee, axis=0), axis=1)
        if float(ee_d.max()) > 0.04:
            fail.append("ee_teleport")

    target = info["target"]
    cube = cubes[target]
    info["lift_z"] = float(cube[:, 2].max())
    info["travel_m"] = float(np.max(np.linalg.norm(cube - cube[0], axis=1)))
    info["final_z"] = float(cube[-1, 2])
    info["final_grip"] = float(grip[-1])

    bin_xy = goal_pos[0, :2]
    info["bin_xy"] = tuple(float(v) for v in bin_xy)
    in_bin = bool(
        np.all(np.abs(cube[-1, :2] - bin_xy) < BIN_XY_TOL)
        and (0.0 < cube[-1, 2] < BIN_Z_TOL)
    )
    info["in_bin"] = in_bin
    info["d_xy"] = float(np.max(np.abs(cube[-1, :2] - bin_xy)))
    if not in_bin:
        fail.append("target_not_in_bin")
    if info["lift_z"] <= LIFT_Z:
        fail.append("target_not_lifted")
    wrong = []
    for c in CUBE_COLORS:
        if c == target:
            continue
        p = cubes[c][-1]
        if np.all(np.abs(p[:2] - bin_xy) < BIN_XY_TOL) and 0.0 < p[2] < BIN_Z_TOL:
            wrong.append(c)
    info["wrong_in_bin"] = wrong
    if wrong:
        fail.append("wrong_cube_in_bin")
    if grip[-1] <= RELEASE_GRIP:
        fail.append("cube_not_released")
    return fail, info


# ── static-frame cleaning ────────────────────────────────────────────────────

def keep_mask(
    arrays: dict[str, np.ndarray],
    start: int,
    end: int,
    *,
    step_mm: float,
    rot_deg: float,
    grip_step: float,
    cube_mm: float,
    max_gap: int,
) -> np.ndarray:
    """Event-based keep mask: keep a frame only if something moved since the last kept frame."""
    n = end - start
    if n <= 2:
        return np.ones(n, dtype=bool)
    ee = arrays["state_ee"][start:end, :3]
    quat = arrays["state_ee"][start:end, 3:7]
    grip = arrays["action_gripper"][start:end, 0]
    cubes = [arrays[k][start:end, :3] for k in CUBE_KEYS]
    step = step_mm / 1000.0
    cube_step = cube_mm / 1000.0

    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True
    last = 0
    for i in range(1, n - 1):
        moved = float(np.linalg.norm(ee[i] - ee[last])) >= step
        rotated = quat_angle_deg(quat[i], quat[last]) >= rot_deg
        grip_changed = abs(float(grip[i] - grip[last])) >= grip_step
        cube_moved = max(
            float(np.linalg.norm(c[i] - c[last])) for c in cubes
        ) >= cube_step
        force = max_gap > 0 and (i - last) >= max_gap
        if moved or rotated or grip_changed or cube_moved or force:
            keep[i] = True
            last = i
    return keep


# ── writing ──────────────────────────────────────────────────────────────────

def write_zarr(
    out_path: Path,
    arrays: dict[str, np.ndarray],
    episode_ends: np.ndarray,
    attrs: dict,
) -> None:
    out = zarr.open_group(str(out_path), mode="w", zarr_format=3)
    data = out.require_group("data")
    meta = out.require_group("meta")
    compressor = zarr.codecs.Blosc(cname="zstd", clevel=3, shuffle=2)
    n = int(episode_ends[-1]) if episode_ends.size else 0
    for key in REQUIRED_KEYS:
        arr = arrays[key]
        if arr.ndim == 2:
            chunks = (min(max(n, 1), 4096), arr.shape[1])
        else:
            chunks = (min(max(n, 1), 4096),)
        data.create_array(
            key,
            data=arr.astype(np.float32),
            chunks=chunks,
            compressors=(compressor,),
        )
    meta.create_array(
        "episode_ends",
        data=episode_ends.astype(np.int64),
        chunks=(min(max(len(episode_ends), 1), 4096),),
        compressors=(compressor,),
    )
    out.attrs.update(attrs)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Clean ex3 multicube teleop data (static frames + bad episodes)."
    )
    ap.add_argument("--input", type=Path, default=Path("datasets/raw/multi_cube/teleop"))
    ap.add_argument("--output", type=Path, default=Path("datasets/raw_clean/multi_cube/teleop"))
    ap.add_argument("--step-mm", type=float, default=5.0,
                    help="Keep a frame each time the EE moved this far (mm).")
    ap.add_argument("--rot-deg", type=float, default=3.0,
                    help="Keep a frame each time the EE rotated this far (deg).")
    ap.add_argument("--grip-step", type=float, default=0.02,
                    help="Keep a frame each time the gripper command changed this much.")
    ap.add_argument("--cube-mm", type=float, default=2.0,
                    help="Keep a frame each time any cube moved this far (mm).")
    ap.add_argument("--max-gap", type=int, default=0,
                    help="If >0, force-keep a frame at least every N frames (0 = disabled).")
    ap.add_argument("--min-raw-steps", type=int, default=40,
                    help="Drop episodes shorter than this before cleaning.")
    ap.add_argument("--min-clean-steps", type=int, default=20,
                    help="Drop episodes whose cleaned length is below this (chunk_size=16 by default).")
    ap.add_argument("--reset-bin-jump-mm", type=float, default=5.0,
                    help="Bin movement inside an episode that marks reset contamination (mm).")
    ap.add_argument("--reset-cube-jump-mm", type=float, default=50.0,
                    help="Cube teleport that marks reset contamination (mm).")
    ap.add_argument("--reset-ee-jump-mm", type=float, default=50.0,
                    help="EE teleport that marks reset contamination (mm).")
    ap.add_argument("--keep-bad", dest="drop_bad", action="store_false", default=True,
                    help="Keep episodes that fail the task validation (default: drop them).")
    ap.add_argument("--dry-run", action="store_true", help="Print decisions, write nothing.")
    ap.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output.")
    ap.add_argument("--report-csv", type=Path, default=None,
                    help="Optional CSV with the per-episode decisions.")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"[ERROR] input not found: {args.input}\n"
                         "        run this from hw3_imitation_learning/")
    out_zarr = args.output / "so100_multicube_teleop.zarr"
    if out_zarr.exists() and not args.dry_run:
        if not args.overwrite:
            raise SystemExit(
                f"[ERROR] {out_zarr} already exists. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(out_zarr)

    print(f"Loading {args.input} ...")
    arrays, episode_ends, stores, attrs = load_stores(args.input)
    n_raw_eps = int(episode_ends.size)
    n_raw_steps = int(episode_ends[-1])
    print(f"raw: {n_raw_eps} episodes, {n_raw_steps} steps "
          f"({n_raw_steps / 10.0 / 60.0:.1f} min @ 10 Hz)")

    starts = [0, *[int(e) for e in episode_ends[:-1]]]
    cleaned: dict[str, list[np.ndarray]] = {k: [] for k in REQUIRED_KEYS}
    new_ends: list[int] = []
    rows: list[dict] = []
    n_dropped = 0
    n_dropped_contaminated = 0

    for ep_i, (start, end) in enumerate(zip(starts, [int(e) for e in episode_ends]), start=1):
        n = end - start
        goal = arrays["state_goal"][start:end]
        target = CUBE_COLORS[int(np.argmax(goal[0]))]
        fail, info = validate_episode(
            arrays, start, end,
            bin_jump_mm=args.reset_bin_jump_mm,
            cube_jump_mm=args.reset_cube_jump_mm,
            ee_jump_mm=args.reset_ee_jump_mm,
        )

        status, reason = "keep", ""
        if n < args.min_raw_steps:
            status, reason = "drop", "too_short"
        elif fail and args.drop_bad:
            status = "drop"
            reason = "+".join(fail)
            if "reset_contamination" in fail:
                frame, desc = info["reset_events"][0]
                reason += f" (frame {frame}: {desc})"
                n_dropped_contaminated += 1
        if status == "drop":
            n_dropped += 1

        mask = keep_mask(
            arrays, start, end,
            step_mm=args.step_mm, rot_deg=args.rot_deg,
            grip_step=args.grip_step, cube_mm=args.cube_mm,
            max_gap=args.max_gap,
        ) if status == "keep" else np.zeros(n, dtype=bool)
        n_keep = int(mask.sum())
        if status == "keep" and n_keep < args.min_clean_steps:
            status, reason = "drop", "too_short_after_cleaning"
            n_dropped += 1

        if status == "keep":
            idx = np.where(mask)[0] + start
            for k in REQUIRED_KEYS:
                cleaned[k].append(arrays[k][idx])
            new_ends.append(new_ends[-1] + n_keep if new_ends else n_keep)

        rows.append({
            "ep": ep_i,
            "frames": n,
            "target": target,
            "kept_frames": n_keep if status == "keep" else 0,
            "kept_pct": 100.0 * n_keep / max(n, 1),
            "status": status,
            "reason": reason,
            "reset_events": "; ".join(f"frame {f}: {d}" for f, d in info["reset_events"]),
            "lift_z": round(info["lift_z"], 3),
            "travel_m": round(info["travel_m"], 3),
            "final_z": round(info["final_z"], 3),
            "final_grip": round(info["final_grip"], 2),
            "in_bin": int(info["in_bin"]),
            "d_xy": round(info["d_xy"], 3),
            "wrong_in_bin": ",".join(info["wrong_in_bin"]) or "-",
        })

    # ── report ──────────────────────────────────────────────────────────
    hdr = (f"{'ep':>3s} {'frames':>6s} {'kept':>5s} {'keep%':>6s} {'goal':>5s} "
           f"{'status':>5s}  reason")
    print(hdr)
    print("-" * (len(hdr) + 16))
    for r in rows:
        print(f"{r['ep']:3d} {r['frames']:6d} {r['kept_frames']:5d} {r['kept_pct']:5.1f}% "
              f"{r['target']:>5s} {r['status']:>5s}  {r['reason']}")

    n_kept_eps = len(new_ends)
    n_cleaned_steps = int(new_ends[-1]) if new_ends else 0
    goal_counts = {c: sum(1 for r in rows if r["status"] == "keep" and r["target"] == c)
                   for c in CUBE_COLORS}
    print(f"\nepisodes        : {n_raw_eps} raw -> {n_kept_eps} kept "
          f"({n_dropped} dropped, {n_dropped_contaminated} due to reset contamination)")
    print(f"steps           : {n_raw_steps} -> {n_cleaned_steps} "
          f"({100.0 * n_cleaned_steps / max(n_raw_steps, 1):.1f}%), "
          f"static frames removed: {100.0 * (1 - n_cleaned_steps / max(n_raw_steps, 1)):.1f}%")
    print(f"dropped         : {[(r['ep'], r['reason']) for r in rows if r['status'] == 'drop']}")
    print(f"goal mix (kept) : {goal_counts}")
    print(f"cleaned lengths : min/mean/max = "
          f"{min((r['kept_frames'] for r in rows if r['status'] == 'keep'), default=0)} / "
          f"{np.mean([r['kept_frames'] for r in rows if r['status'] == 'keep']):.0f} / "
          f"{max((r['kept_frames'] for r in rows if r['status'] == 'keep'), default=0)} steps")

    if args.report_csv:
        args.report_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.report_csv.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"report          : {args.report_csv}")

    if args.dry_run:
        print("\n[dry-run] nothing written.")
        return 0
    if not new_ends:
        raise SystemExit("[ERROR] every episode was dropped - check the thresholds/flags.")

    # ── write cleaned zarr ──────────────────────────────────────────────
    out_arrays = {k: np.concatenate(v, axis=0) for k, v in cleaned.items()}
    new_ends_arr = np.asarray(new_ends, dtype=np.int64)
    attrs = dict(attrs)
    attrs.update({
        "cleaned_from": [str(p) for p in stores],
        "clean_num_episodes": int(len(new_ends_arr)),
        "clean_num_steps": int(new_ends_arr[-1]),
        "clean_static_step_mm": float(args.step_mm),
        "clean_rot_deg": float(args.rot_deg),
        "clean_grip_step": float(args.grip_step),
        "clean_cube_step_mm": float(args.cube_mm),
        "clean_max_gap": int(args.max_gap),
        "clean_drop_bad": bool(args.drop_bad),
        "clean_reset_contamination_dropped": int(n_dropped_contaminated),
        "clean_success_xy_tol": BIN_XY_TOL,
        "clean_success_z_tol": BIN_Z_TOL,
    })
    args.output.mkdir(parents=True, exist_ok=True)
    write_zarr(out_zarr, out_arrays, new_ends_arr, attrs)

    meta = {
        "source": [str(p) for p in stores],
        "output": str(out_zarr),
        "params": {
            "step_mm": args.step_mm, "rot_deg": args.rot_deg,
            "grip_step": args.grip_step, "cube_mm": args.cube_mm,
            "max_gap": args.max_gap, "min_raw_steps": args.min_raw_steps,
            "min_clean_steps": args.min_clean_steps, "drop_bad": args.drop_bad,
            "reset_bin_jump_mm": args.reset_bin_jump_mm,
            "reset_cube_jump_mm": args.reset_cube_jump_mm,
            "reset_ee_jump_mm": args.reset_ee_jump_mm,
        },
        "episodes_in": n_raw_eps, "episodes_out": int(len(new_ends_arr)),
        "steps_in": n_raw_steps, "steps_out": int(new_ends_arr[-1]),
        "dropped_reset_contamination": int(n_dropped_contaminated),
        "goal_counts": goal_counts,
        "decisions": rows,
    }
    (args.output / "clean_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"\nwritten         : {out_zarr}")
    print("verify          : .venv\\Scripts\\python.exe ..\\..\\hw3_ex3_inspect.py "
          f"--input {args.output}")
    print("next            : .venv\\Scripts\\python.exe scripts\\compute_actions.py "
          f"--action-space ee --datasets-dir {args.output.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
