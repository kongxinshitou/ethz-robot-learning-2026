"""Same expert demo, but driven *exclusively* by single key presses.

Every control tick applies exactly one teleop key press (1 cm move, or one
gripper step of 0.10), using the very same `handle_teleop_key` function as the
real recorder.  This makes the trajectory directly reproducible by hand: the
annotated video shows, tick by tick, which key was pressed.

Usage (run inside hw3_imitation_learning with its venv):
    .venv\\Scripts\\python.exe ..\\..\\hw3_demo_keyed.py --seed 0 --video

Outputs:
    datasets/expert_ref/keyed/<timestamp>/expert_demo.zarr
    datasets/expert_ref/keyed/<timestamp>/expert_demo.mp4 / expert_demo_raw.mp4
    datasets/expert_ref/keyed/<timestamp>/meta.json     (phases + per-step keys)
    datasets/expert_ref/keyed/<timestamp>/expert_trajectory.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import mujoco
import numpy as np

WORKSPACE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE))

import hw3_demo_grasp as demo  # noqa: E402
from hw3.eval_utils import check_cube_out_of_bounds, check_success  # noqa: E402
from hw3.sim_env import SO100SimEnv  # noqa: E402
from hw3.teleop_utils import DEFAULT_KEYMAP_PATH, handle_teleop_key  # noqa: E402

XML_PATH = demo.XML_PATH
CONTROL_HZ = demo.CONTROL_HZ
GRASP_DX = demo.GRASP_DX
GRASP_Z = 0.045
TRANSIT_Z = demo.TRANSIT_Z
AROUND_X = demo.AROUND_X
DROP_X = demo.DROP_X
DROP_Z = demo.DROP_Z
JAW_MAX, JAW_MIN = demo.JAW_MAX, demo.JAW_MIN

RAW_TO_KEY = {
    2424832: "←",
    2490368: "↑",
    2555904: "→",
    2621440: "↓",
    13: "Enter",
    27: "Esc",
    32: "Space",
}


def load_action_labels() -> dict[str, str]:
    """action name -> the key the student actually assigned in keymap.json."""
    path = DEFAULT_KEYMAP_PATH
    labels: dict[str, str] = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        for action, entry in data.items():
            raw = int(entry["raw"])
            ascii_code = int(entry.get("ascii", 0))
            labels[action] = RAW_TO_KEY.get(
                raw,
                chr(ascii_code).upper() if 32 <= ascii_code <= 126 else f"raw:{raw}",
            )
    else:
        labels = {
            "move_up": "↑", "move_down": "↓", "move_left": "←", "move_right": "→",
            "move_forward": "W", "move_backward": "S",
            "gripper_open": "M", "gripper_close": "L",
            "record": "Space", "end_episode": "Enter", "reset": "R", "escape": "Esc",
        }
    return labels


ACTION_TEXT = {
    "move_up": "上移 1 cm",
    "move_down": "下移 1 cm",
    "move_left": "左移 1 cm",
    "move_right": "右移 1 cm",
    "move_forward": "前移 1 cm",
    "move_backward": "后移 1 cm",
    "gripper_open": "张开夹爪 +0.10",
    "gripper_close": "闭合夹爪 -0.10",
    "noop": "停一拍（不按键）",
}


class KeyedRecorder(demo.DemoRecorder):
    """DemoRecorder that can only act through single key presses."""

    def __init__(self, *args, labels: dict[str, str], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.labels = labels
        self.key_track: list[str] = []
        self.action_track: list[str] = []

    def press(self, action: str) -> None:
        handle_teleop_key(
            action,
            self.env.data,
            self.env.model,
            self.env.mocap_id,
            int(self.env.act_ids[5]),
        )
        self.key_track.append(self.labels.get(action, "?"))
        self.action_track.append(action)
        self.tick()

    def noop(self) -> None:
        self.key_track.append("·")
        self.action_track.append("noop")
        self.tick()

    # ── single-axis servo using key presses ─────────────────────────
    def servo_keys(self, target, tol: float = 0.010, max_ticks: int = 160) -> bool:
        target = np.asarray(target, dtype=np.float64)
        for _ in range(max_ticks):
            err = target - self.env.get_ee_pos()
            if np.all(np.abs(err) < tol):
                return True
            axis = int(np.argmax(np.abs(err)))
            if abs(err[axis]) < tol:
                break
            pos = err[axis] > 0
            if axis == 0:
                action = "move_right" if pos else "move_left"
            elif axis == 1:
                action = "move_forward" if pos else "move_backward"
            else:
                action = "move_up" if pos else "move_down"
            self.press(action)
        return bool(np.all(np.abs(np.asarray(target) - self.env.get_ee_pos()) < 2 * tol))

    def jaw_to(self, target: float, tol: float = 0.06, max_presses: int = 40) -> bool:
        for _ in range(max_presses):
            cur = float(self.env.data.ctrl[self.env.act_ids[5]])
            if abs(cur - target) <= tol:
                return True
            action = "gripper_open" if cur < target else "gripper_close"
            self.press(action)
        return False

    def settle(self, ticks: int) -> None:
        for _ in range(ticks):
            self.noop()


def run(args) -> int:
    labels = load_action_labels()
    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = Path.cwd() / out_root
    run_dir = out_root / datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    out_zarr = run_dir / "expert_demo.zarr"

    np.random.seed(args.seed)
    env = SO100SimEnv(xml_path=XML_PATH, control_hz=CONTROL_HZ, seed=args.seed)
    writer = demo.build_writer(out_zarr)
    rec = KeyedRecorder(env, writer, collect_video=args.video, labels=labels)

    cube0 = env.get_cube_state()[:3].copy()
    bin_c = env.get_goal_pos().copy()
    print(f"cube    : {np.round(cube0, 4)}")
    print(f"bin     : {np.round(bin_c, 4)}")
    print(f"obstacle: {np.round(env.get_obstacle_pos(), 4)}")

    grasp_xy = cube0[:2] + np.array([GRASP_DX, 0.0])
    ee0 = env.get_ee_pos().copy()

    rec.set_phase("open gripper")
    rec.jaw_to(JAW_MAX)

    rec.set_phase("lift")
    rec.servo_keys([ee0[0], ee0[1], TRANSIT_Z])

    rec.set_phase("around barrier (left)")
    rec.servo_keys([AROUND_X, ee0[1], TRANSIT_Z])
    rec.servo_keys([AROUND_X, grasp_xy[1], TRANSIT_Z])

    rec.set_phase("align with cube")
    rec.servo_keys([grasp_xy[0], grasp_xy[1], TRANSIT_Z])

    rec.set_phase("descend")
    rec.servo_keys([grasp_xy[0], grasp_xy[1], args.grasp_z], tol=0.008, max_ticks=90)

    rec.set_phase("grasp")
    rec.jaw_to(args.close_jaw)
    rec.settle(4)

    rec.set_phase("lift cube")
    rec.servo_keys([grasp_xy[0], grasp_xy[1], TRANSIT_Z], max_ticks=120)

    rec.set_phase("carry around barrier")
    rec.servo_keys([AROUND_X, grasp_xy[1], TRANSIT_Z], max_ticks=120)
    rec.servo_keys([AROUND_X, bin_c[1], TRANSIT_Z], max_ticks=160)

    rec.set_phase("to bin")
    rec.servo_keys([bin_c[0] + DROP_X, bin_c[1], TRANSIT_Z], max_ticks=160)

    rec.set_phase("insert into bin")
    rec.servo_keys([bin_c[0] + DROP_X, bin_c[1], DROP_Z], max_ticks=120)

    rec.set_phase("release")
    rec.jaw_to(JAW_MAX)
    rec.settle(8)

    rec.set_phase("retreat")
    rec.servo_keys([bin_c[0] + DROP_X, bin_c[1], 0.22], max_ticks=80)
    rec.settle(6)

    writer.end_episode()
    writer.flush()

    env.step()
    success = check_success(env)
    oob = check_cube_out_of_bounds(env)
    cube_end = env.get_cube_state()[:3]
    n = writer.num_steps_total
    print(f"\nsteps={n}  success={success}  out_of_bounds={oob}")
    print(f"cube end: {np.round(cube_end, 4)}   bin: {np.round(bin_c, 4)}")

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
        "keyboard_driven": True,
        "keys": [
            {"step": i, "key": rec.key_track[i], "action": rec.action_track[i]}
            for i in range(len(rec.key_track))
        ],
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.video:
        demo.save_video(
            rec.frames, rec.phase_track,
            run_dir / "expert_demo.mp4", run_dir / "expert_demo_raw.mp4",
        )
    demo.save_plot(
        np.array(rec.ee_track), np.array(rec.cube_track),
        np.array(rec.gripper_track), rec.phase_track, bin_c[:2],
        run_dir / "expert_trajectory.png",
    )
    print(f"zarr    : {out_zarr}")
    return 0 if success else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Key-press-only expert demo.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grasp-z", type=float, default=GRASP_Z)
    parser.add_argument("--close-jaw", type=float, default=0.15)
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/expert_ref/keyed"))
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    t0 = time.time()
    rc = run(args)
    print(f"elapsed {time.time() - t0:.1f}s")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
