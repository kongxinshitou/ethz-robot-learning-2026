"""新模型（chunk=4, 512x3）的失败模式统计：卡在抓取还是搬运？"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HW3 = Path(__file__).resolve().parent.parent / "ethz-robot-learning-2026" / "hw3_imitation_learning"
sys.path.insert(0, str(HW3))

from hw3.eval_utils import (  # noqa: E402
    apply_action, check_cube_out_of_bounds, check_success, infer_action_chunk, load_checkpoint,
)
from hw3.sim_env import SO100SimEnv  # noqa: E402
from so101_gym.constants import ASSETS_DIR  # noqa: E402

CKPT = HW3 / "checkpoints/single_cube/best_model_ee_xyz_obstacle.pt"
XML = ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml"

model, norm, H, sk, ak = load_checkpoint(CKPT, torch.device("cpu"))
print(f"chunk_size={H}")
env = SO100SimEnv(xml_path=XML, seed=None)

rows = []
for ep in range(10):
    obs = env.reset()
    q = []
    max_z, min_bin, min_ee, drop_after = -9, 9, 9, False
    lifted = False
    for step in range(800):
        if not q:
            q.extend(infer_action_chunk(model, norm, obs, sk, torch.device("cpu")))
        a = q.pop(0)
        cube = env.get_cube_state()[:3]
        if cube[2] > 0.06:
            lifted = True
        if lifted and cube[2] < 0.035 and np.linalg.norm(cube[:2] - env.get_goal_pos()[:2]) > 0.08:
            drop_after = True
        max_z = max(max_z, float(cube[2]))
        min_bin = min(min_bin, float(np.linalg.norm(cube[:2] - env.get_goal_pos()[:2])))
        min_ee = min(min_ee, float(np.linalg.norm(env.get_ee_pos() - cube)))
        apply_action(env, a, ak)
        obs = env.step()
        if check_success(env):
            rows.append(("SUCCESS", step + 1, max_z, min_bin, min_ee, False))
            break
        if check_cube_out_of_bounds(env):
            rows.append(("OOB", step + 1, max_z, min_bin, min_ee, drop_after))
            break
    else:
        rows.append(("TIMEOUT", 800, max_z, min_bin, min_ee, drop_after))

print(f"{'结果':>8} {'步数':>5} {'最高cube_z':>10} {'最近cube-bin':>13} {'最近ee-cube':>12} {'提起后掉落':>8}")
for r, steps, mz, mb, me, drop in rows:
    print(f"{r:>8} {steps:>5} {mz:>10.3f} {mb*100:>11.1f}cm {me*100:>10.1f}cm {str(drop):>8}")
ok = sum(1 for r in rows if r[0] == "SUCCESS")
print(f"\nSR = {ok}/{len(rows)} | 抓起过 {sum(1 for r in rows if r[2] > 0.06)}/{len(rows)} "
      f"| 运到bin 8cm 内 {sum(1 for r in rows if r[3] < 0.08)}/{len(rows)}")
