"""1) 数据里「松手」动作有多稀有；2) 模型到 bin 上方时夹爪指令是什么。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import zarr

HW3 = Path(__file__).resolve().parent.parent / "ethz-robot-learning-2026" / "hw3_imitation_learning"
sys.path.insert(0, str(HW3))

from hw3.eval_utils import apply_action, check_success, infer_action_chunk, load_checkpoint  # noqa: E402
from hw3.sim_env import SO100SimEnv  # noqa: E402
from so101_gym.constants import ASSETS_DIR  # noqa: E402

CLEAN = HW3 / "datasets/raw_clean/single_cube/teleop/so100_transfer_cube_teleop.zarr"
CKPT = HW3 / "checkpoints/single_cube/best_model_ee_xyz_obstacle.pt"
XML = ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml"

print("[1] 示范数据里的夹爪指令分布")
g = zarr.open_group(str(CLEAN), mode="r")
ag = np.asarray(g["data"]["action_gripper"]).ravel()
ends = np.asarray(g["meta"]["episode_ends"][:])
starts = np.concatenate(([0], ends[:-1]))
fracs, opens = [], []
for s, e in zip(starts, ends, strict=True):
    seg = ag[s:e]
    fracs.append(np.mean(seg > 0.5))
    # 松手 = close(-0.1 以下) 之后第一次 >0.5
    closed = np.where(seg < -0.1)[0]
    opened = np.where(seg > 0.5)[0]
    if closed.size and opened.size:
        after = opened[opened > closed[0]]
        opens.append(after[0] - closed[0] if after.size else -1)
print(f"  每条 episode 中「张开指令(>0.5)」占比: mean {np.mean(fracs):.1%} "
      f"(min {np.min(fracs):.1%}, max {np.max(fracs):.1%})")
print(f"  「夹紧 -> 松开」间隔: mean {np.mean(opens):.1f} 帧 = {np.mean(opens)*0.1:.2f} s | "
      f"median {np.median(opens):.0f} 帧")
print(f"  每个 episode 平均 {np.mean(ends-starts):.0f} 帧 -> 松开只占 "
      f"{np.mean([f for f in fracs])*np.mean(ends-starts):.1f} 帧左右")

print("\n[2] 模型在 bin 上方的夹爪指令（3 条 rollout，每 40 步打印）")
model, norm, H, sk, ak = load_checkpoint(CKPT, torch.device("cpu"))
env = SO100SimEnv(xml_path=XML, seed=None)
for ep in range(3):
    obs = env.reset()
    q = []
    print(f"  --- ep{ep} (chunk={H}) ---")
    for step in range(800):
        if not q:
            q.extend(infer_action_chunk(model, norm, obs, sk, torch.device("cpu")))
        a = q.pop(0)
        cube = env.get_cube_state()[:3]
        d_bin = float(np.linalg.norm(cube[:2] - env.get_goal_pos()[:2]))
        if step % 40 == 0 or (d_bin < 0.03 and step % 10 == 0):
            print(f"    step{step:3d} cube_z {cube[2]:.3f} cube-bin {d_bin*100:5.1f}cm "
                  f"grip_cmd {a[3]:+.2f} grip_state {env.get_gripper_angle():+.2f}")
        apply_action(env, a, ak)
        obs = env.step()
        if check_success(env):
            print(f"    step{step}: SUCCESS")
            break
    else:
        print("    timeout")
