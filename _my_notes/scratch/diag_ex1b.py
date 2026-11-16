"""诊断 2：动作预测质量 + 不同 chunk 执行策略对成功率的影响。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HW3 = Path(__file__).resolve().parent.parent / "ethz-robot-learning-2026" / "hw3_imitation_learning"
sys.path.insert(0, str(HW3))

from hw3.dataset import SO100ChunkDataset, load_zarr  # noqa: E402
from hw3.eval_utils import (  # noqa: E402
    apply_action, check_cube_out_of_bounds, check_success, infer_action_chunk, load_checkpoint,
)
from hw3.sim_env import SO100SimEnv  # noqa: E402
from so101_gym.constants import ASSETS_DIR  # noqa: E402

CKPT = HW3 / "checkpoints/single_cube/best_model_ee_xyz_obstacle.pt"
ZARR = HW3 / "datasets/processed/single_cube/processed_ee_xyz.zarr"
XML = ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml"

model, norm, H, sk, ak = load_checkpoint(CKPT, torch.device("cpu"))
model.eval()

print("=" * 72)
print("[A] 逐维预测质量（归一化空间，在全部训练数据上）")
states, actions, ep_ends = load_zarr(ZARR, state_keys=sk, action_keys=ak)
ds = SO100ChunkDataset(states, actions, ep_ends, chunk_size=H, normalizer=norm)
loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False)
num = np.zeros(actions.shape[1])
den = np.zeros(actions.shape[1])
n = 0
with torch.no_grad():
    for s, a in loader:
        p = model(s)
        num += ((p - a) ** 2).sum(dim=(0, 1)).numpy()
        den += (a ** 2).sum(dim=(0, 1)).numpy()   # 归一化后均值为 0，这就是方差
        n += a.shape[0] * a.shape[1]
names = ["delta_x", "delta_y", "delta_z", "gripper"]
for i, name in enumerate(names):
    mse = num[i] / n
    var = den[i] / n
    print(f"  {name:8s}: model MSE {mse:.4f} | 方差 {var:.4f} | 可解释 {1-mse/var:6.1%} | "
          f"RMSE {np.sqrt(mse):.3f} (归一化单位)")

# 夹爪开合分类
with torch.no_grad():
    s_all = torch.from_numpy(norm.normalize_state(states)).float()
    p_all = model(s_all)[:, 0, 3].numpy()   # chunk 第一步的 gripper（归一化）
true_g = actions[:, 3]
pred_g = norm.denormalize_action(np.stack([np.zeros_like(p_all), np.zeros_like(p_all),
                                           np.zeros_like(p_all), p_all], axis=1))[:, 3]
thr = np.median(true_g)
acc = np.mean((pred_g > thr) == (true_g > thr))
print(f"  夹爪动作: 真实范围 [{true_g.min():.3f},{true_g.max():.3f}] | 预测范围 [{pred_g.min():.3f},{pred_g.max():.3f}]")
print(f"  夹爪开/合分类准确率（以中位数 {thr:.3f} 为阈值）: {acc:.1%}")

print("=" * 72)
print("[B] 不同执行策略的成功率（同一组 seed，8 条）")
env = SO100SimEnv(xml_path=XML, seed=None)


def run(ep_seed: int, execute: str) -> tuple[bool, dict]:
    env.rng = np.random.default_rng(ep_seed)   # 固定 cube/obstacle 随机化
    obs = env.reset()
    queue = []
    stats = {"max_cube_z": -9, "min_cube_bin": 9, "min_ee_cube": 9, "steps": 0}
    for step in range(800):
        if not queue:
            queue.extend(infer_action_chunk(model, norm, obs, sk, torch.device("cpu")))
        a = queue.pop(0)
        cube = env.get_cube_state()[:3]
        stats["max_cube_z"] = max(stats["max_cube_z"], float(cube[2]))
        stats["min_cube_bin"] = min(stats["min_cube_bin"], float(np.linalg.norm(cube[:2] - env.get_goal_pos()[:2])))
        stats["min_ee_cube"] = min(stats["min_ee_cube"], float(np.linalg.norm(env.get_ee_pos() - cube)))
        apply_action(env, a, ak)
        obs = env.step()
        stats["steps"] = step + 1
        if check_success(env):
            return True, stats
        if check_cube_out_of_bounds(env):
            break
        if execute == "one" and len(queue) >= H - 1:
            queue.clear()           # 只执行 chunk 第一步（每步重推理）
        elif execute == "four" and len(queue) <= H - 4:
            queue.clear()           # 执行前 4 步后重推理
    return False, stats


for mode, label in (("all", "开环执行整条 chunk(16)"), ("four", "执行 4 步后重推理"), ("one", "每步重推理")):
    succ = []
    lifted = 0
    moved = 0
    for ep in range(8):
        ok, st = run(ep, mode)
        succ.append(ok)
        lifted += st["max_cube_z"] > 0.05
        moved += st["min_cube_bin"] < 0.25
    print(f"  {label:20s}: SR {sum(succ)}/8 ({sum(succ)/8:.0%}) | 抓起过({'>'}5cm) {lifted}/8 "
          f"| 运到bin附近(25cm) {moved}/8")
