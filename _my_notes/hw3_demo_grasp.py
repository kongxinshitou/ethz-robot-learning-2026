"""Scripted expert demonstration for HW3 (single cube + obstacle + bin).

This is a *reference* demo: instead of a human pressing arrow keys, a small
scripted controller drives the same mocap teleop interface at the same 10 Hz
control rate, records one episode in the official zarr schema, and produces a
video + trajectory plot so you can see *how* the grasp is done.

It is meant for learning / debugging.  The homework expects your own
teleoperated demonstrations for the graded runs.

Usage (run inside hw3_imitation_learning with its venv):
    .venv\\Scripts\\python.exe ..\\..\\hw3_demo_grasp.py --seed 0

Outputs (default):
    datasets/expert_ref/single_cube/<timestamp>/expert_demo.zarr
    datasets/expert_ref/single_cube/<timestamp>/expert_demo.mp4
    datasets/expert_ref/single_cube/<timestamp>/expert_trajectory.png
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import mujoco
import numpy as np
from hw3.eval_utils import check_cube_out_of_bounds, check_success
from hw3.sim_env import DEFAULT_CUBE_POS_STD, SO100SimEnv
from hw3.teleop_utils import (
    CAMERA_NAMES,
    ZarrEpisodeWriter,
    compose_camera_views,
    fit_for_display,
)
from so101_gym.constants import ASSETS_DIR

XML_PATH = ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml"
CONTROL_HZ = 10.0
JAW_MIN, JAW_MAX = -0.174, 1.75

# grasp is done slightly to the -x side of the cube centre (the fixed finger
# needs to clear the cube's left face)
GRASP_DX = -0.012
GRASP_Z = 0.040          # EE-site height during the grasp
TRANSIT_Z = 0.200        # EE-site height while moving around
AROUND_X = -0.280        # x used to walk *around* the barrier (barrier spans x -0.265..-0.175)
DROP_X = -0.020          # release x offset from the bin centre
DROP_Z = 0.090           # EE-site height when releasing inside the bin


class DemoRecorder:
    """Drives the sim like the teleop recorder does and writes one episode."""

    def __init__(
        self,
        env: SO100SimEnv,
        writer: ZarrEpisodeWriter,
        collect_video: bool = False,
        show: bool = False,
    ) -> None:
        self.env = env
        self.writer = writer
        self.collect_video = collect_video
        self.show = show
        self.frames: list[np.ndarray] = []
        self.ee_track: list[np.ndarray] = []
        self.gripper_track: list[float] = []
        self.cube_track: list[np.ndarray] = []
        self.phase_track: list[str] = []
        self.phase_marks: list[tuple[str, int]] = []
        self.phase = "start"
        self.window = "SO100 expert demo"
        if self.show:
            cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)

    # ── one teleop control tick ──────────────────────────────────────
    def tick(self, mocap_delta: np.ndarray | None = None, jaw: float | None = None) -> None:
        env = self.env
        if mocap_delta is not None:
            env.data.mocap_pos[env.mocap_id] += np.asarray(mocap_delta, dtype=np.float64)
        if jaw is not None:
            env.set_gripper(float(np.clip(jaw, JAW_MIN, JAW_MAX)))
        self._record_step()
        for _ in range(env.substeps):
            mujoco.mj_step(env.model, env.data)
        if self.collect_video or self.show:
            img = self._compose()          # raw frame; annotations are added later
            if self.collect_video:
                self.frames.append(img.copy())
            if self.show:
                cv2.imshow(self.window, fit_for_display(self._overlay(img)))
                cv2.waitKey(1)

    # ── state recording (mirrors record_teleop_demos._record_step) ──
    def _record_step(self) -> None:
        env = self.env
        d = env.data
        state_joints = d.qpos[env.qpos_idx].copy()
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, d.site_xmat[env.ee_site_id])
        state_ee = np.concatenate([d.site_xpos[env.ee_site_id], quat])
        state_cube = d.qpos[env.cube_qpos_idx].copy()
        state_obstacle = d.xpos[env.obstacle_body_id].copy() if env.obstacle_body_id != -1 else np.zeros(3)
        state_gripper = np.array([state_joints[5]], dtype=np.float32)
        action_gripper = np.array([d.ctrl[env.act_ids[5]]], dtype=np.float32)
        if self.writer is not None:
            self.writer.append(
                state_joints,
                state_ee,
                state_cube,
                state_gripper,
                action_gripper,
                state_obstacle,
            )
        self.ee_track.append(state_ee[:3].copy())
        self.gripper_track.append(float(action_gripper[0]))
        self.cube_track.append(state_cube[:3].copy())
        self.phase_track.append(self.phase)

    def _compose(self) -> np.ndarray:
        images = {cam: self.env.render(cam) for cam in CAMERA_NAMES}
        return compose_camera_views(images, CAMERA_NAMES)

    def _overlay(self, img: np.ndarray) -> np.ndarray:
        out = img.copy()
        cv2.putText(
            out, f"EXPERT DEMO | phase: {self.phase} | step {len(self.ee_track)}",
            (660, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        cv2.putText(
            out, "scripted reference trajectory (not human teleop)",
            (660, 552), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
        )
        return out

    # ── mocap servo: make the *actual* EE site reach `target` ───────
    def servo(
        self,
        target: np.ndarray,
        max_ticks: int = 60,
        tol: float = 0.005,
        max_step: float = 0.012,
    ) -> bool:
        target = np.asarray(target, dtype=np.float64)
        for _ in range(max_ticks):
            ee = self.env.get_ee_pos()
            err = target - ee
            if np.linalg.norm(err) < tol:
                return True
            offset = self.env.data.mocap_pos[self.env.mocap_id] - ee
            desired = target + offset
            delta = np.clip(desired - self.env.data.mocap_pos[self.env.mocap_id], -max_step, max_step)
            self.tick(mocap_delta=delta)
        return False

    def hold(self, ticks: int, jaw: float | None = None) -> None:
        for _ in range(ticks):
            self.tick(jaw=jaw)

    def set_phase(self, name: str) -> None:
        self.phase = name
        self.phase_marks.append((name, len(self.ee_track)))
        print(f"  [{len(self.ee_track):3d}] {name}")


def build_writer(out_zarr: Path) -> ZarrEpisodeWriter:
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
        xml=str(XML_PATH),
        source="scripted_expert_reference",
        state_joints_spec="qpos(joints)",
        state_ee_spec="ee_pos(3) + ee_quat_wxyz(4)",
        state_cube_spec="cube_pos(3) + cube_quat_wxyz(4)",
        state_gripper_spec="gripper_angle(1)",
        action_gripper_spec="gripper_ctrl(1)",
        control_hz=CONTROL_HZ,
        cameras_display=list(CAMERA_NAMES),
    )
    return writer


def save_video(
    frames: list[np.ndarray],
    phases: list[str],
    plain_path: Path,
    raw_path: Path | None = None,
) -> None:
    """Write the plain watermarked video and (optionally) a raw copy for annotating."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(plain_path), fourcc, CONTROL_HZ, (w, h))
    vw_raw = cv2.VideoWriter(str(raw_path), fourcc, CONTROL_HZ, (w, h)) if raw_path else None
    n = len(frames)
    for i, f in enumerate(frames):
        out = f.copy()
        phase = phases[i] if i < len(phases) else ""
        cv2.putText(
            out, f"EXPERT DEMO | phase: {phase} | step {i + 1}/{n}",
            (660, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        cv2.putText(
            out, "scripted reference trajectory (not human teleop)",
            (660, 552), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
        )
        vw.write(out)
        if vw_raw is not None:
            vw_raw.write(f)
    vw.release()
    if vw_raw is not None:
        vw_raw.release()
    print(f"  video   : {plain_path}")
    if raw_path is not None:
        print(f"  raw     : {raw_path}")


def save_plot(
    ee: np.ndarray,
    cube: np.ndarray,
    grip: np.ndarray,
    phases: list[str],
    bin_xy: np.ndarray,
    path: Path,
) -> None:
    import os

    cache = Path.cwd() / ".mplcache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(ee[:, 0], ee[:, 1], lw=2, label="EE (grasp point)")
    ax.plot(cube[:, 0], cube[:, 1], lw=2, ls="--", label="cube")
    ax.plot(ee[0, 0], ee[0, 1], "o", ms=6, color="tab:green", label="start")
    ax.plot(ee[-1, 0], ee[-1, 1], "X", ms=7, color="tab:red", label="end")
    ax.plot(bin_xy[0], bin_xy[1], "*", ms=13, color="tab:purple", label="bin")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Top view (XY)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ee[:, 1], ee[:, 2], lw=2, label="EE")
    ax.plot(cube[:, 1], cube[:, 2], lw=2, ls="--", label="cube")
    ax.axhline(0.12, color="0.5", ls=":", label="obstacle wall top")
    ax.axhline(0.24, color="0.5", ls="-.", label="upper obstacle bottom")
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("Side view (YZ)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(ee[:, 2], lw=2, label="EE z")
    ax.plot(cube[:, 2], lw=2, ls="--", label="cube z")
    ax.set_xlabel("control step (10 Hz)")
    ax.set_ylabel("z [m]")
    ax.set_title("Heights")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(grip, lw=2, color="tab:red")
    ax.set_xlabel("control step (10 Hz)")
    ax.set_ylabel("gripper ctrl")
    ax.set_title("Gripper command (+ open / - close)")
    ax.grid(alpha=0.3)

    # shade phases only on the step-based axes
    for a in (axes[1, 0], axes[1, 1]):
        start = 0
        for i in range(1, len(phases) + 1):
            if i == len(phases) or phases[i] != phases[start]:
                a.axvspan(start, i, alpha=0.06)
                start = i

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  plot    : {path}")


def run_demo(args) -> int:
    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = Path.cwd() / out_root
    ts = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = out_root / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    out_zarr = run_dir / "expert_demo.zarr"

    np.random.seed(args.seed)
    env = SO100SimEnv(
        xml_path=XML_PATH,
        control_hz=CONTROL_HZ,
        seed=args.seed,
        cube_pos_std=DEFAULT_CUBE_POS_STD,
    )
    writer = build_writer(out_zarr)
    rec = DemoRecorder(env, writer, collect_video=args.video, show=args.show)

    cube0 = env.get_cube_state()[:3].copy()
    bin_c = env.get_goal_pos().copy()
    print(f"cube    : {np.round(cube0, 4)}")
    print(f"bin     : {np.round(bin_c, 4)}")
    print(f"obstacle: {np.round(env.get_obstacle_pos(), 4)}")

    grasp_xy = cube0[:2] + np.array([GRASP_DX, 0.0])
    ee0 = env.get_ee_pos().copy()

    rec.set_phase("open gripper")
    rec.hold(3, jaw=JAW_MAX)

    rec.set_phase("lift")
    rec.servo([ee0[0], ee0[1], TRANSIT_Z])

    # the barrier sits between the arm base and the cube, so walk around its
    # left edge instead of trying to squeeze through the gap
    rec.set_phase("around barrier (left)")
    rec.servo([AROUND_X, ee0[1], TRANSIT_Z])
    rec.servo([AROUND_X, grasp_xy[1], TRANSIT_Z])

    rec.set_phase("align with cube")
    rec.servo([grasp_xy[0], grasp_xy[1], TRANSIT_Z])

    rec.set_phase("descend")
    rec.servo([grasp_xy[0], grasp_xy[1], GRASP_Z], max_ticks=50)

    rec.set_phase("grasp")
    # close the jaw gradually, like a human tapping the close key
    for jaw in np.linspace(JAW_MAX, 0.0, 12):
        rec.tick(jaw=float(jaw))
    rec.hold(4, jaw=0.0)

    rec.set_phase("lift cube")
    rec.servo([grasp_xy[0], grasp_xy[1], TRANSIT_Z], max_ticks=60)

    rec.set_phase("carry around barrier")
    rec.servo([AROUND_X, grasp_xy[1], TRANSIT_Z], max_ticks=60)
    rec.servo([AROUND_X, bin_c[1], TRANSIT_Z], max_ticks=70)

    rec.set_phase("to bin")
    rec.servo([bin_c[0] + DROP_X, bin_c[1], TRANSIT_Z], max_ticks=70)

    rec.set_phase("insert into bin")
    rec.servo([bin_c[0] + DROP_X, bin_c[1], DROP_Z], max_ticks=60)

    rec.set_phase("release")
    for jaw in np.linspace(0.0, JAW_MAX, 10):
        rec.tick(jaw=float(jaw))
    rec.hold(8, jaw=JAW_MAX)

    rec.set_phase("retreat")
    rec.servo([bin_c[0] + DROP_X, bin_c[1], 0.22], max_ticks=40)
    rec.hold(6)

    writer.end_episode()
    writer.flush()

    env.step()  # let the cube settle
    success = check_success(env)
    oob = check_cube_out_of_bounds(env)
    cube_end = env.get_cube_state()[:3]
    n = writer.num_steps_total
    print(f"\nsteps={n}  success={success}  out_of_bounds={oob}")
    print(f"cube end: {np.round(cube_end, 4)}   bin: {np.round(bin_c, 4)}")
    print(f"zarr    : {out_zarr}")

    # phase boundaries for the annotated slow-motion video
    phases = []
    for i, (name, start) in enumerate(rec.phase_marks):
        end = rec.phase_marks[i + 1][1] if i + 1 < len(rec.phase_marks) else n
        if end > start:
            phases.append({"name": name, "start": int(start), "end": int(end)})
    meta = {
        "seed": int(args.seed),
        "control_hz": CONTROL_HZ,
        "steps": int(n),
        "success": bool(success),
        "cube0": [float(v) for v in cube0],
        "bin_xy": [float(v) for v in bin_c[:2]],
        "obstacle": [float(v) for v in env.get_obstacle_pos()],
        "phases": phases,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.video:
        save_video(
            rec.frames,
            rec.phase_track,
            run_dir / "expert_demo.mp4",
            run_dir / "expert_demo_raw.mp4",
        )
    save_plot(
        np.array(rec.ee_track),
        np.array(rec.cube_track),
        np.array(rec.gripper_track),
        rec.phase_track,
        bin_c[:2],
        run_dir / "expert_trajectory.png",
    )
    if rec.show:
        cv2.destroyAllWindows()
    return 0 if success else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Record one scripted expert demo (reference).")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomisation seed.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("datasets/expert_ref/single_cube"),
        help="Output directory (default: datasets/expert_ref/single_cube).",
    )
    parser.add_argument("--video", action="store_true", help="Also save an mp4 of the episode.")
    parser.add_argument("--show", action="store_true", help="Show the window while running.")
    parser.add_argument("--no-plot", action="store_true", help="Skip the trajectory plot.")
    args = parser.parse_args()
    t0 = time.time()
    rc = run_demo(args)
    print(f"elapsed {time.time() - t0:.1f}s")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
