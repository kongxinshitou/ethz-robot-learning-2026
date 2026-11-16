"""Ex3 (multicube) teleop recording inspector.

Checks the recorded ``so100_multicube_teleop.zarr`` data for:
  - structural problems (NaN/inf, episode_ends sanity, non-constant goal)
  - task-level problems (wrong cube picked, target cube not in bin,
    cube still grasped at the end, dropped cubes, teleport jumps)
  - idle/static frames (how many, where: leading / trailing / middle)

Run from ``ethz-robot-learning-2026/hw3_imitation_learning`` with its venv:

    .venv\\Scripts\\python.exe ..\\..\\hw3_ex3_inspect.py
    .venv\\Scripts\\python.exe ..\\..\\hw3_ex3_inspect.py --input datasets/raw/multi_cube/teleop

Only needs numpy + zarr (no mujoco / cv2), so it of any recording.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import zarr

CUBE_COLORS = ("red", "green", "blue")
CUBE_KEYS = {c: f"pos_cube_{c}" for c in CUBE_COLORS}

# static / anomaly thresholds -------------------------------------------------
STATIC_STEP_M = 0.002      # 2 mm: EE displacement below this = "not moving"
STATIC_GRIP = 0.02         # gripper command change below this = "not moving"
STATIC_CUBE_M = 0.002      # cube displacement below this = "not moving"
MAX_EE_JUMP_M = 0.04       # > 4 cm in one 100 ms step = suspicious teleport
MAX_CUBE_JUMP_M = 0.08     # > 8 cm in one step = physics glitch / knock
LIFT_Z = 0.06              # cube z above this = lifted
BIN_XY_TOL = 0.04          # multicube success tolerance
BIN_Z_TOL = 0.04
RELEASE_GRIP = 0.5         # action_gripper above this = gripper open (same convention as ex1 inspect)


def load_store(path: Path) -> dict:
    root = zarr.open_group(str(path), mode="r")
    data = root["data"]
    out = {
        "root": root,
        "attrs": dict(root.attrs),
        "episode_ends": np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64),
        "arrays": {k: np.asarray(data[k][:]) for k in data.array_keys()},
    }
    return out


def quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Angle between wxyz quaternions (deg), robust to double cover."""
    d = np.abs(np.sum(q1 * q2, axis=-1))
    d = np.clip(d, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(d))


def static_mask(ep: dict) -> np.ndarray:
    """A step i is static if nothing (EE, cubes, gripper cmd) moved from i-1 to i."""
    n = len(ep["ee"])
    static = np.zeros(n, dtype=bool)
    if n < 2:
        return static
    ee_d = np.linalg.norm(np.diff(ep["ee"], axis=0), axis=1)
    cube_d = np.stack(
        [np.linalg.norm(np.diff(ep["cubes"][c], axis=0), axis=1) for c in ep["cubes"]],
        axis=1,
    ).max(axis=1)
    grip_d = np.abs(np.diff(ep["grip_cmd"]))
    static[1:] = (ee_d < STATIC_STEP_M) & (cube_d < STATIC_CUBE_M) & (grip_d < STATIC_GRIP)
    return static


def run_lengths(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(start, length), ...] of True runs (length >= 1)."""
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(mask) - start))
    return runs


def slice_episodes(arrs: dict, ep_ends: np.ndarray) -> list[dict]:
    episodes = []
    start = 0
    for ep_i, end in enumerate(ep_ends):
        end = int(end)
        ep = {
            "joints": arrs["state_joints"][start:end],
            "ee_full": arrs["state_ee"][start:end],
            "ee": arrs["state_ee"][start:end, :3],
            "gripper": arrs["state_gripper"][start:end, 0],
            "grip_cmd": arrs["action_gripper"][start:end, 0],
            "goal": arrs["state_goal"][start:end],
            "goal_pos": arrs["goal_pos"][start:end],
            "cubes": {c: arrs[CUBE_KEYS[c]][start:end, :3] for c in CUBE_COLORS},
            "raw_cubes": {c: arrs[CUBE_KEYS[c]][start:end] for c in CUBE_COLORS},
            "ep": ep_i + 1,
            "n": end - start,
            "start": start,
            "end": end,
        }
        episodes.append(ep)
        start = end
    return episodes


def episode_report(ep: dict, first_ee_pose: np.ndarray | None) -> dict:
    n = ep["n"]
    ee = ep["ee"]
    grip_cmd = ep["grip_cmd"]
    goal = ep["goal"]
    goal_pos = ep["goal_pos"]
    cubes = ep["cubes"]

    # ── goal consistency ──────────────────────────────────────────────
    goal_idx_vals = np.argmax(goal, axis=1)
    goal_sum = goal.sum(axis=1)
    goal_ok = (
        np.all(np.abs(goal_sum - 1.0) < 1e-5)
        and np.all(goal_idx_vals == goal_idx_vals[0])
        and np.allclose(goal, goal[0], atol=1e-6)
    )
    target = CUBE_COLORS[int(goal_idx_vals[0])]
    goal_constant = bool(np.max(np.ptp(goal_pos, axis=0)) < 1e-6)

    # ── motion ───────────────────────────────────────────────────────
    ee_disp = np.linalg.norm(np.diff(ee, axis=0), axis=1) if n > 1 else np.array([])
    static = static_mask(ep)
    static_runs = run_lengths(static)
    longest_static = max((length for _, length in static_runs), default=0)
    # leading / trailing idle = static run touching the episode start / end
    lead = next((length for start, length in static_runs if start == 1), 0)
    trail = next(
        (length for start, length in reversed(static_runs) if start + length == n), 0
    )

    # ── cubes ────────────────────────────────────────────────────────
    tgt = cubes[target]
    tgt_disp = np.linalg.norm(tgt - tgt[0], axis=1)
    tgt_final = tgt[-1]
    bin_xy = goal_pos[0, :2]
    d_xy = np.abs(tgt_final[:2] - bin_xy)
    in_bin = bool(np.all(d_xy < BIN_XY_TOL) and (0.0 < tgt_final[2] < BIN_Z_TOL))
    wrong_in_bin = []
    for c in CUBE_COLORS:
        if c == target:
            continue
        p = cubes[c][-1]
        if np.all(np.abs(p[:2] - bin_xy) < BIN_XY_TOL) and 0.0 < p[2] < BIN_Z_TOL:
            wrong_in_bin.append(c)
    others_moved = {
        c: float(np.max(np.linalg.norm(cubes[c] - cubes[c][0], axis=1)))
        for c in CUBE_COLORS
        if c != target
    }
    picked_wrong = bool(others_moved and max(others_moved.values()) > 0.03)

    # distance EE <-> target cube at final frame = "still grasping" indicator
    gap = float(np.linalg.norm(ee[-1] - tgt_final))
    still_grasped = gap < 0.02 and grip_cmd[-1] < RELEASE_GRIP

    # ── anomalies ────────────────────────────────────────────────────
    ee_jumps = np.where(ee_disp > MAX_EE_JUMP_M)[0] + 1 if n > 1 else []
    cube_jumps = []
    for c in CUBE_COLORS:
        d = np.linalg.norm(np.diff(cubes[c], axis=0), axis=1) if n > 1 else np.array([])
        idx = np.where(d > MAX_CUBE_JUMP_M)[0] + 1
        cube_jumps.extend([(c, int(i), float(d[i - 1])) for i in idx])
    any_nan = any(
        not np.all(np.isfinite(v))
        for v in (
            ep["joints"],
            ep["ee_full"],
            ep["gripper"],
            ep["grip_cmd"],
            ep["goal"],
            ep["goal_pos"],
            *cubes.values(),
        )
    )

    # ── gripper ──────────────────────────────────────────────────────
    grip_changes = int(np.sum(np.abs(np.diff((grip_cmd > RELEASE_GRIP).astype(int))) > 0))
    start_pose_dev = (
        float(np.linalg.norm(ee[0] - first_ee_pose)) if first_ee_pose is not None else 0.0
    )

    return {
        "ep": ep["ep"],
        "n": n,
        "dur_s": n / 10.0,
        "goal_ok": goal_ok,
        "target": target,
        "goal_constant": goal_constant,
        "bin_xy": bin_xy,
        "ee_path_m": float(np.sum(ee_disp)),
        "static_frac": float(static.mean()),
        "longest_static": longest_static,
        "lead_static": lead,
        "trail_static": trail,
        "tgt_lift_z": float(tgt[:, 2].max()),
        "tgt_travel_m": float(tgt_disp.max()),
        "in_bin": in_bin,
        "d_xy": float(d_xy.max()),
        "final_z": float(tgt_final[2]),
        "wrong_in_bin": ",".join(wrong_in_bin) or "-",
        "picked_wrong": picked_wrong,
        "others_moved": max(others_moved.values()) if others_moved else 0.0,
        "still_grasped": still_grasped,
        "ee_gap_end": gap,
        "final_grip_open": bool(grip_cmd[-1] > RELEASE_GRIP),
        "grip_changes": grip_changes,
        "nan": any_nan,
        "ee_jumps": len(ee_jumps),
        "cube_jumps": len(cube_jumps),
        "start_pose_dev": start_pose_dev,
    }


def fmt_pos(p: np.ndarray) -> str:
    return f"({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})"


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect ex3 multicube teleop data.")
    ap.add_argument("--input", type=Path, default=Path("datasets/raw/multi_cube/teleop"))
    ap.add_argument("--csv", type=Path, default=None,
                    help="Optional path to write the per-episode report CSV.")
    args = ap.parse_args()

    stores = sorted(p for p in args.input.rglob("*.zarr") if p.is_dir())
    if not stores:
        print(f"[ERROR] no .zarr under {args.input}")
        return 1
    print(f"{len(stores)} store(s) under {args.input}")

    all_reports = []
    for store in stores:
        print(f"\n=== {store} ===")
        d = load_store(store)
        ends = d["episode_ends"]
        arrs = d["arrays"]
        attrs = d["attrs"]
        total = int(ends[-1]) if ends.size else 0
        print(f"control_hz={attrs.get('control_hz', '?')}  "
              f"episodes={ends.size}  steps={total}")
        print(f"arrays: {sorted(arrs)}")
        if ends.size == 0:
            continue
        if np.any(np.diff(ends) <= 0) or ends[0] <= 0 or ends[-1] != min(
            a.shape[0] for a in arrs.values()
        ):
            print("  [ERROR] episode_ends inconsistent with array lengths!")

        eps = slice_episodes(arrs, ends)
        first_pose = eps[0]["ee"][0]
        reports = [episode_report(ep, first_pose) for ep in eps]
        all_reports.extend(reports)

        # ── per-episode table ────────────────────────────────────────
        hdr = (
            f"{'ep':>3s} {'n':>4s} {'dur':>5s} {'goal':>6s} {'bin':>18s} "
            f"{'path':>5s} {'static':>6s} {'lead':>4s} {'trail':>5s} {'maxStill':>8s} "
            f"{'liftZ':>5s} {'travel':>6s} {'inBin':>5s} {'dXY':>5s} {'wrong':>7s} "
            f"{'pickW':>5s} {'graspEnd':>8s} {'gripCh':>6s} {'nan':>3s} {'jumps':>5s}"
        )
        print(hdr)
        print("-" * len(hdr))
        for r in reports:
            print(
                f"{r['ep']:3d} {r['n']:4d} {r['dur_s']:5.1f} "
                f"{('OK' if r['goal_ok'] else 'BAD'):>6s} "
                f"{fmt_pos(np.r_[r['bin_xy'], 0]):>18s} "
                f"{r['ee_path_m']:5.2f} {100 * r['static_frac']:5.1f}% "
                f"{r['lead_static']:4d} {r['trail_static']:5d} {r['longest_static']:8d} "
                f"{r['tgt_lift_z']:5.3f} {r['tgt_travel_m']:6.3f} "
                f"{('Y' if r['in_bin'] else '.'):>5s} {r['d_xy']:5.3f} "
                f"{r['wrong_in_bin']:>7s} {('Y' if r['picked_wrong'] else '.'):>5s} "
                f"{('Y' if r['still_grasped'] else '.'):>8s} {r['grip_changes']:6d} "
                f"{('Y' if r['nan'] else '.'):>3s} {r['ee_jumps'] + r['cube_jumps']:5d}"
            )

        # ── problem summary ─────────────────────────────────────────
        print("\n-- checks --")
        bad_goal = [r["ep"] for r in reports if not r["goal_ok"]]
        bad_bin = [r["ep"] for r in reports if not r["goal_constant"]]
        fail = [r["ep"] for r in reports if not r["in_bin"]]
        wrong_bin = [r["ep"] for r in reports if r["wrong_in_bin"] != "-"]
        wrong_pick = [r["ep"] for r in reports if r["picked_wrong"]]
        grasped = [r["ep"] for r in reports if r["still_grasped"]]
        nan_eps = [r["ep"] for r in reports if r["nan"]]
        jump_eps = [r["ep"] for r in reports if r["ee_jumps"] or r["cube_jumps"]]
        short = [r["ep"] for r in reports if r["n"] < 40]
        not_lifted = [r["ep"] for r in reports if r["tgt_lift_z"] <= LIFT_Z]
        pose_var = max(r["start_pose_dev"] for r in reports) if reports else 0.0

        def line(name: str, eps: list, extra: str = "") -> None:
            print(f"  {'[FAIL]' if eps else '[ok]  '} {name:<34s} {eps}{extra}")

        line("goal one-hot valid & constant", bad_goal)
        line("bin (goal_pos) constant per ep", bad_bin)
        line("target cube ends in bin", fail)
        line("wrong cube ends in bin", wrong_bin)
        line("moved a non-target cube (>3cm)", wrong_pick)
        line("target cube never lifted (>6cm)", not_lifted)
        line("still grasping cube at end", grasped)
        line("NaN / inf values", nan_eps)
        line("teleport jumps (>4cm ee/>8cm cube)", jump_eps)
        line("very short episode (<40 steps)", short)
        print(f"  [info]  first-frame EE pose spread across eps: {pose_var * 1000:.1f} mm")
        print(f"  [info]  goal cube distribution: "
              f"{ {c: sum(1 for r in reports if r['target'] == c) for c in CUBE_COLORS} }")

        idle = [r["ep"] for r in reports if r["static_frac"] > 0.5]
        total_static = sum(
            int(round(r["static_frac"] * r["n"])) for r in reports
        )
        total_steps = sum(r["n"] for r in reports)
        print(f"  [info]  static frames: {total_static}/{total_steps} "
              f"({100 * total_static / max(total_steps, 1):.1f}%)  "
              f"episodes >50% static: {idle}")
        print(f"  [info]  mean ep length {np.mean([r['n'] for r in reports]):.0f} steps "
              f"({np.mean([r['dur_s'] for r in reports]):.1f}s)")

        # ── success stats ──────────────────────────────────────────
        ok = [r for r in reports if r["in_bin"] and not r["picked_wrong"] and r["wrong_in_bin"] == "-"]
        print(f"  [info]  clean successes: {len(ok)}/{len(reports)} "
              f"({100 * len(ok) / max(len(reports), 1):.0f}%)")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = list(all_reports[0].keys())
        with args.csv.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in all_reports:
                row = dict(r)
                row["bin_xy"] = fmt_pos(np.r_[row["bin_xy"], 0])
                w.writerow(row)
        print(f"\nCSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
