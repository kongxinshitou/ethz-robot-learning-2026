"""Headless replica of scripts/evaluate_trajectory.py (Lemniscate keypoints, no viewer).

Measures how the trained RL policy (10 Hz) tracks the 20 Lemniscate keypoints
sequentially: how many keypoints it advances through in 10 s of sim time and
what the mean EE tracking error is. Useful evidence for the bonus question.
"""
import sys
from pathlib import Path

ROOT = Path(r"D:\Users\00796533\Documents\deepseek-harness\default-workspace\ethz-robot-learning-2026\hw2_robot_control_mdps")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import numpy as np
import mujoco
from stable_baselines3 import PPO

from __init__ import XML_PATH, EXP_DIR
from env.so100_tracking_env import SO100TrackEnv
from exercises.ex1 import build_keypoints

CKPT = sys.argv[1] if len(sys.argv) > 1 else "500"
RUN = sys.argv[2] if len(sys.argv) > 2 else "1"

env = SO100TrackEnv(xml_path=XML_PATH, render_mode=None)
model = PPO.load(str(EXP_DIR / f"so100_tracking_{RUN}" / f"model_{CKPT}.zip"), device="cpu")

keypoints = build_keypoints(count=20, width=0.2, x_offset=0.3, z_offset=0.25)
env.data.mocap_pos[0] = keypoints[0]

st = {"step": 0, "kp": 0, "errs": [], "advances": []}


def cb(m, d):
    s = st["step"]
    err = float(np.linalg.norm(d.site("ee_site").xpos - d.mocap_pos[0]))
    st["errs"].append(err)
    if err < 0.05:
        kp = st["kp"]
        st["kp"] = (kp + 1) % len(keypoints)
        st["advances"].append(s)
    d.mocap_pos[0] = keypoints[st["kp"]]
    if s % env.ctrl_decimation == 0:
        obs = env._get_obs()
        a, _ = model.predict(obs, deterministic=True)
        d.ctrl[:] = env._process_action(a)
    st["step"] = s + 1


SIM_S = 10.0
n_steps = int(SIM_S / env.model.opt.timestep)
mujoco.set_mjcb_control(cb)
try:
    for _ in range(n_steps):
        mujoco.mj_step(env.model, env.data)
finally:
    mujoco.set_mjcb_control(None)

errs = np.asarray(st["errs"])
print(f"simulated {SIM_S:.0f} s ({n_steps} steps, {env.ctrl_decimation} decimation)")
print(f"keypoints fully entered: {len(st['advances'])} / 20   (advance times [s]: "
      f"{[round(t * env.model.opt.timestep, 1) for t in st['advances']]})")
print(f"EE->active-keypoint error: mean={errs.mean():.4f} m  median={np.median(errs):.4f}  "
      f"max={errs.max():.4f}  min={errs.min():.4f}")
print(f"fraction of time error < 0.05 m: {(errs < 0.05).mean() * 100:.1f}%")
