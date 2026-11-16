"""Deep per-episode inspection of HW3 teleop recordings.

Metrics: grasp/carry vs push, obstacle clearance, bin placement, idle time.
Plots: trajectory grid + height/gripper timelines.
Optional: replayed videos rendered from the recorded states.

Run from ethz-robot-learning-2026/hw3_imitation_learning with its venv:
    .venv\\Scripts\\python.exe ..\\..\\hw3_episode_inspect.py
    .venv\\Scripts\\python.exe ..\\..\\hw3_episode_inspect.py --replay 7
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import mujoco
import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont

BIN_XY = np.array([-0.2, 0.7])


def load_episodes(base: Path):
    episodes = []
    for store in sorted(p for p in base.rglob("*.zarr") if p.is_dir()):
        root = zarr.open_group(str(store), mode="r")
        try:
            ep_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=int)
        except Exception:
            continue
        if ep_ends.size == 0:
            continue
        data = root["data"]
        arrays = {
            "joints": np.asarray(data["state_joints"][:]),
            "ee": np.asarray(data["state_ee"][:]),
            "cube": np.asarray(data["state_cube"][:]),
            "grip": np.asarray(data["action_gripper"][:]).reshape(-1),
            "obstacle": np.asarray(data["state_obstacle"][:]),
        }
        start = 0
        for ep_i, end in enumerate(ep_ends):
            end = int(end)
            ep = {k: v[start:end] for k, v in arrays.items()}
            ep.update({"store": store.parent.name, "ep": ep_i + 1, "n": end - start})
            episodes.append(ep)
            start = end
    return episodes


def metrics(ep: dict) -> dict:
    ee = ep["ee"][:, :3]
    cube = ep["cube"][:, :3]
    grip = ep["grip"]
    n = ep["n"]
    d_ee = np.linalg.norm(np.diff(ee, axis=0), axis=1)
    closed = grip < 0.5
    lifted_mask = (cube[:, 2] > 0.06) & closed
    carry_frames = int(np.sum(lifted_mask))
    cube_end = cube[-1]
    in_xy = np.linalg.norm(cube_end[:2] - BIN_XY) < 0.05
    in_z = 0.0 < cube_end[2] < 0.04
    # stable at end: cube barely moves in the last 10 frames
    tail = cube[-10:]
    stable = bool(np.max(np.linalg.norm(np.diff(tail, axis=0), axis=1)) < 3e-3) if len(tail) > 2 else False
    # grasp attempts = open->closed transitions
    attempts = int(np.sum(np.diff(closed.astype(int)) == 1))
    # pushing ratio: cube moved while gripper open or cube stays low
    pushed_frames = int(np.sum((np.linalg.norm(np.diff(cube, axis=0), axis=1) > 5e-3) & ~closed[1:]))
    return {
        "steps": n,
        "dur_s": n / 10.0,
        "moving": int(np.sum(d_ee > 4e-3)),
        "still_frac": float(np.mean(d_ee < 1e-3)),
        "cube_start": cube[0],
        "cube_end": cube_end,
        "lift": bool(np.max(cube[:, 2]) > 0.06),
        "carry_frames": carry_frames,
        "grasp_attempts": attempts,
        "pushed_frames": pushed_frames,
        "ee_max_z": float(ee[:, 2].max()),
        "over_obstacle": bool(ee[:, 2].max() > 0.13),
        "bin_ok": bool(in_xy and in_z),
        "stable_end": stable,
        "grip_min": float(grip.min()),
        "grip_max": float(grip.max()),
    }


def plot_grid(eps, metas, path: Path) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplcache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(eps)
    cols, rows = 4, (n + 3) // 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.2))
    axes = np.atleast_2d(axes)
    for i, (ep, m) in enumerate(zip(eps, metas)):
        ax = axes[i // cols, i % cols]
        ee, cube = ep["ee"][:, :3], ep["cube"][:, :3]
        ax.plot(ee[:, 0], ee[:, 1], lw=1.2, color="tab:blue", label="EE")
        ax.plot(cube[:, 0], cube[:, 1], lw=1.2, ls="--", color="tab:orange", label="cube")
        ax.plot(BIN_XY[0], BIN_XY[1], "*", ms=9, color="tab:purple")
        ax.plot(ee[0, 0], ee[0, 1], "o", ms=4, color="tab:green")
        ax.set_title(
            f"{ep['store'][-8:]} #{ep['ep']}  n={m['steps']}\n"
            f"lift={'Y' if m['lift'] else 'N'}  bin={'Y' if m['bin_ok'] else 'N'}",
            fontsize=7,
        )
        ax.tick_params(labelsize=6)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.25)
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle("Recorded episodes: top view (EE solid, cube dashed, bin star)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  plot: {path}")


def plot_timelines(eps, metas, path: Path) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplcache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(eps)
    cols, rows = 4, (n + 3) // 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.0))
    axes = np.atleast_2d(axes)
    for i, (ep, m) in enumerate(zip(eps, metas)):
        ax = axes[i // cols, i % cols]
        t = np.arange(ep["n"]) / 10.0
        ax.plot(t, ep["ee"][:, 2], lw=1.0, color="tab:blue", label="EE z")
        ax.plot(t, ep["cube"][:, 2], lw=1.0, ls="--", color="tab:orange", label="cube z")
        closed = ep["grip"] < 0.5
        ax.fill_between(t, 0, 0.3, where=closed, step="mid", alpha=0.15, color="tab:red")
        ax.set_ylim(-0.02, 0.3)
        ax.set_title(
            f"{ep['store'][-8:]} #{ep['ep']}  carry={m['carry_frames']}f",
            fontsize=7,
        )
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle("Heights over time (shaded = gripper closed)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  plot: {path}")


def render_replay(ep: dict, xml_path: Path, out_path: Path, stride: int = 5) -> None:
    from hw3.teleop_utils import CAMERA_NAMES, compose_camera_views

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    joint_names = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
    qpos_idx = np.array([
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in joint_names
    ])
    cube_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_box_joint")
    cube_qpos = np.arange(model.jnt_qposadr[cube_jid], model.jnt_qposadr[cube_jid] + 7)
    obs_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obstacle")
    up_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "upper_obstacle")

    stride = max(stride, ep["n"] // 240)
    frames = []
    for i in range(0, ep["n"], stride):
        data.qpos[qpos_idx] = ep["joints"][i]
        data.qpos[cube_qpos] = ep["cube"][i]
        model.body_pos[obs_body][0] = ep["obstacle"][i][0]
        model.body_pos[up_body][0] = ep["obstacle"][i][0]
        mujoco.mj_forward(model, data)
        imgs = {}
        for c in CAMERA_NAMES:
            renderer.update_scene(data, camera=c)
            imgs[c] = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        frame = compose_camera_views(imgs, CAMERA_NAMES)
        cv2.putText(
            frame,
            f"{ep['store'][-8:]} ep{ep['ep']}  t={i / 10.0:5.1f}s  step {i}/{ep['n']}",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
        )
        frames.append(frame)
    renderer.close()
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"  replay: {out_path}  ({len(frames)} frames, stride {stride})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("datasets/raw/single_cube/teleop"))
    parser.add_argument("--out", type=Path, default=Path("datasets/inspect"))
    parser.add_argument("--replay", nargs="*", type=int, default=None,
                        help="episode indices (0-based, in the order listed) to replay")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    eps = load_episodes(args.base)
    metas = [metrics(e) for e in eps]
    print(f"{len(eps)} episodes from {args.base}\n")
    print(
        f"{'idx':>3s} {'store':>9s} {'ep':>2s} {'steps':>5s} {'move':>4s} {'still':>5s} "
        f"{'lift':>4s} {'carry':>5s} {'push':>4s} {'eemaxZ':>6s} {'bin':>3s} {'stbl':>4s} {'grasp':>5s}"
    )
    for i, (ep, m) in enumerate(zip(eps, metas)):
        print(
            f"{i:3d} {ep['store'][-8:]:>9s} {ep['ep']:2d} {m['steps']:5d} {m['moving']:4d} "
            f"{m['still_frac']:5.2f} {'Y' if m['lift'] else '.':>4s} {m['carry_frames']:5d} "
            f"{m['pushed_frames']:4d} {m['ee_max_z']:6.3f} {'Y' if m['bin_ok'] else '.':>3s} "
            f"{'Y' if m['stable_end'] else '.':>4s} {m['grasp_attempts']:5d}"
        )

    with (args.out / "episode_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["idx", "store", "ep", "steps", "dur_s", "moving", "still_frac",
                    "lift", "carry_frames", "pushed_frames", "ee_max_z", "over_obstacle",
                    "bin_ok", "stable_end", "grasp_attempts", "grip_min", "grip_max",
                    "cube_start_x", "cube_start_y", "cube_end_x", "cube_end_y", "cube_end_z"])
        for i, (ep, m) in enumerate(zip(eps, metas)):
            w.writerow([i, ep["store"], ep["ep"], m["steps"], f"{m['dur_s']:.1f}",
                        m["moving"], f"{m['still_frac']:.3f}", int(m["lift"]),
                        m["carry_frames"], m["pushed_frames"], f"{m['ee_max_z']:.3f}",
                        int(m["over_obstacle"]), int(m["bin_ok"]), int(m["stable_end"]),
                        m["grasp_attempts"], f"{m['grip_min']:.2f}", f"{m['grip_max']:.2f}",
                        *[f"{v:.4f}" for v in (*m["cube_start"][:2], *m["cube_end"])]])

    plot_grid(eps, metas, args.out / "episode_paths.png")
    plot_timelines(eps, metas, args.out / "episode_heights.png")

    from so101_gym.constants import ASSETS_DIR

    xml = ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml"
    if args.replay:
        for idx in args.replay:
            if 0 <= idx < len(eps):
                ep = eps[idx]
                name = f"replay_{idx:02d}_{ep['store'][-8:]}_ep{ep['ep']}.mp4"
                render_replay(ep, xml, args.out / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
