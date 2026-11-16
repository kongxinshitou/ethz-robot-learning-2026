"""当前模型（merged 数据, chunk8/512）诊断：夹爪分布、失败模式、chunk 长度敏感性。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HW3 = Path(__file__).resolve().parent.parent / "ethz-robot-learning-2026" / "hw3_imitation_learning"
sys.path.insert(0, str(HW3))

from hw3.dataset import load_zarr  # noqa: E402
from hw3.eval_utils import (  # noqa: E402
    apply_action, check_cube_out_of_bounds, check_success, infer_action_chunk, load_checkpoint,
)
from hw3.sim_env import SO100SimEnv  # noqa: E402
from so101_gym.constants import ASSETS_DIR  # noqa: E402

CKPT = HW3 / "checkpoints/single_cube/best_model_ee_xyz_obstacle.pt"
OLD = HW3 / "datasets/processed/single_cube/processed_ee_xyz.zarr"
NEW = HW3 / "datasets/processed/single_cube/processed_ee_xyz_new_clean.zarr"
XML = ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml"
SK = ["state_ee_xyz", "state_gripper", "state_cube[:3]", "state_obstacle"]
AK = ["action_ee_xyz", "action_gripper"]

print("=" * 70)
print("[1] 两份数据集的 action_gripper 分布（旧 vs 新）")
for path, name in ((OLD, "旧 24 条"), (NEW, "新 19 条")):
    _, acts, _ = load_zarr(path, state_keys=SK, action_keys=AK)
    ag = acts[:, 3]
    print(f"  {name}: min {ag.min():+.2f} max {ag.max():+.2f} mean {ag.mean():+.2f} "
          f"| p5/p50/p95 = {np.percentile(ag,5):+.2f}/{np.percentile(ag,50):+.2f}/{np.percentile(ag,95):+.2f}")

model, norm, H, sk, ak = load_checkpoint(CKPT, torch.device("cpu"))
print(f"\n[2] 当前模型: chunk={H} | 训练时归一化 action mean/std = {np.round(norm.action_mean,3)} / {np.round(norm.action_std,3)}")

env = SO100SimEnv(xml_path=XML, seed=None)


def run(ep_seed: int, execute: int) -> tuple[bool, dict]:
    env.rng = np.random.default_rng(ep_seed)
    obs = env.reset()
    q = []
    st = {"max_z": -9, "min_bin": 9, "dropped": False, "grip_cmds": []}
    lifted = False
    for step in range(800):
        if not q:
            q.extend(infer_action_chunk(model, norm, obs, sk, torch.device("cpu")))
        a = q.pop(0)
        cube = env.get_cube_state()[:3]
        z = float(cube[2])
        bin_d = float(np.linalg.norm(cube[:2] - env.get_goal_pos()[:2]))
        if z > 0.06:
            lifted = True
        if lifted and z < 0.035 and bin_d > 0.08:
            st["dropped"] = True
        st["max_z"] = max(st["max_z"], z)
        st["min_bin"] = min(st["min_bin"], bin_d)
        if 0.08 < bin_d < 0.25:
            st["grip_cmds"].append(float(a[3]))
        apply_action(env, a, ak)
        obs = env.step()
        if check_success(env):
            return True, st
        if check_cube_out_of_bounds(env):
            break
        if execute > 0 and len(q) <= H - execute:
            q.clear()
    return False, st


print("\n[3] 失败模式（默认开环执行整条 chunk=8，10 条）")
res = []
for ep in range(10):
    ok, st = run(ep, execute=H)
    res.append((ok, st))
    print(f"  ep{ep}: {'SUCCESS' if ok else 'FAIL'} | max_cube_z {st['max_z']:.3f} "
          f"| min_cube-bin {st['min_bin']*100:5.1f}cm | dropped={st['dropped']}")

print(f"  SR {sum(r[0] for r in res)}/10 | 抓起过 {sum(r[1]['max_z']>0.06 for r in res)}/10 "
      f"| 到 bin 8cm 内 {sum(r[1]['min_bin']<0.08 for r in res)}/10 "
      f"| 提起后掉落 {sum(r[1]['dropped'] for r in res)}/10")

print("\n[4] 执行策略敏感性：同一模型，不同开环长度（10 条同 seed）")
for ex in (8, 4, 2, 1):
    oks = [run(ep, execute=ex)[0] for ep in range(10)]
    print(f"  执行 {ex} 步后重推理: SR {sum(oks)}/10 ({sum(oks)/10:.0%})")
