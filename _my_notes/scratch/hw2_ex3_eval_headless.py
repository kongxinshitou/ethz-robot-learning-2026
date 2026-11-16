"""Headless replica of scripts/evaluate_rand_targets.py (no MuJoCo viewer window).

Usage:
    python hw2_ex3_eval_headless.py [checkpoint] [load_run] [n_repeats]

It reproduces the official evaluation protocol exactly:
  - 10 episodes, each 2 s = 20 control steps = 1000 mujoco steps
  - reset robot qpos with reset_robot(), resample target with reset_target_position()
  - at the end of each episode print the final ee_tracking_error
  - average over the 10 episodes (full score if < 0.05 m)
"""
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\Users\00796533\Documents\deepseek-harness\default-workspace\ethz-robot-learning-2026\hw2_robot_control_mdps")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import numpy as np
import mujoco
from stable_baselines3 import PPO

from __init__ import XML_PATH, EXP_DIR
from env.so100_tracking_env import SO100TrackEnv
from exercises.ex3 import reset_robot, reset_target_position

CKPT = sys.argv[1] if len(sys.argv) > 1 else "500"
RUN = sys.argv[2] if len(sys.argv) > 2 else "1"
REPEATS = int(sys.argv[3]) if len(sys.argv) > 3 else 1

PATH = EXP_DIR / f"so100_tracking_{RUN}" / f"model_{CKPT}.zip"

env = SO100TrackEnv(xml_path=XML_PATH, render_mode=None)
model = PPO.load(str(PATH), device="cpu")

PLAY_EP_LEN = int(2 / env.ctrl_timestep)          # 20 control steps / episode
DECIM = env.ctrl_decimation                       # 50
MAX_EPISODES = 10
STEPS = MAX_EPISODES * PLAY_EP_LEN * DECIM         # 10000 mujoco steps

state = {"step": 0, "errs": []}


def reset_env():
    env.data.qpos[:] = reset_robot(env.default_qpos)
    env.data.mocap_pos[0] = reset_target_position(env.data.body("Base").xpos.copy())


def cb(m, d):
    s = state["step"]
    if s == 0:
        reset_env()
    elif s % (PLAY_EP_LEN * DECIM) == 0:
        err = float(np.linalg.norm(d.site("ee_site").xpos - d.mocap_pos[0]))
        state["errs"].append(err)
        print(f"  episode {len(state['errs']):2d} / {MAX_EPISODES}:  final EE error = {err:.4f} m")
        reset_env()
    elif s % DECIM == 0:
        obs = env._get_obs()
        action, _ = model.predict(obs, deterministic=True)
        d.ctrl[:] = env._process_action(action)
    state["step"] = s + 1


print(f"Loading policy: {PATH}")
all_avgs = []
for rep in range(REPEATS):
    state["step"] = 0
    state["errs"] = []
    t0 = time.time()
    mujoco.set_mjcb_control(cb)
    try:
        for _ in range(STEPS + 1):
            mujoco.mj_step(env.model, env.data)
    finally:
        mujoco.set_mjcb_control(None)
    errs = np.asarray(state["errs"])
    avg = float(errs.mean())
    all_avgs.append(avg)
    print(f"[repeat {rep + 1}] avg final EE error = {avg:.4f} m "
          f"| max = {errs.max():.4f} | min = {errs.min():.4f} "
          f"| ({time.time() - t0:.1f}s)")
    print(f"             {'PASS' if avg < 0.05 else 'FAIL'} (threshold 0.05)")

print(f"\noverall mean over repeats = {np.mean(all_avgs):.4f} m")
