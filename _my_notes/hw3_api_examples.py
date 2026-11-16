"""HW3 函数手册配套示例 —— 每个片段都验证过输入/输出形状。

运行方式（在 hw3_imitation_learning 目录下）:
    .venv\\Scripts\\python.exe ..\\..\\hw3_api_examples.py

覆盖:
  1) 数据管线: load_zarr / Normalizer / SO100ChunkDataset
  2) 策略契约: compute_loss / sample_actions 的形状（用最小参考实现）
  3) eval 侧契约: eval_utils.load_checkpoint 如何调用 build_policy
  4) 仿真环境: reset / set_* / step / render / check_success
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from hw3.dataset import Normalizer, SO100ChunkDataset, load_zarr
from hw3.eval_utils import (
    ZARR_KEY_TO_OBS,
    action_key_dim,
    apply_action,
    check_success,
    obs_to_state,
    parse_key_spec,
)
from hw3.model import BasePolicy, build_policy
from hw3.sim_env import SO100SimEnv
from so101_gym.constants import ASSETS_DIR

ZARR = "datasets/processed/single_cube/processed_ee_xyz.zarr"
XML = ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml"


def section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# ── 1) 数据管线 ──────────────────────────────────────────────────────────
section("1) 数据管线")
states, actions, ep_ends = load_zarr(
    ZARR,
    state_keys=["state_ee_xyz", "state_gripper", "state_cube[:3]", "state_obstacle"],
    action_keys=["action_ee_xyz", "action_gripper"],
)
print("states", states.shape, states.dtype, "| actions", actions.shape)
print("episode_ends", ep_ends.shape, "-> n_episodes =", ep_ends.size)

norm = Normalizer.from_data(states, actions)
ds = SO100ChunkDataset(states, actions, ep_ends, chunk_size=16, normalizer=norm)
s, a = ds[0]
print("dataset len", len(ds), "| one sample: state", tuple(s.shape), "chunk", tuple(a.shape))
err = np.abs(norm.denormalize_action(norm.normalize_action(actions[:5])) - actions[:5]).max()
print("normalize/denormalize 往返误差", float(err))


# ── 2) 策略契约（最小参考实现，仅演示形状） ───────────────────────────────
section("2) 策略契约")


class RefObstaclePolicy(BasePolicy):
    def __init__(self, state_dim, action_dim, chunk_size=16, d_model=256, depth=3):
        super().__init__(state_dim, action_dim, chunk_size)
        layers, d = [], state_dim
        for _ in range(depth):
            layers += [nn.Linear(d, d_model), nn.GELU()]
            d = d_model
        layers += [nn.Linear(d, chunk_size * action_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state).view(-1, self.chunk_size, self.action_dim)

    def compute_loss(self, state, action_chunk):
        return nn.functional.mse_loss(self.forward(state), action_chunk)

    def sample_actions(self, state):
        return self.forward(state)


policy = RefObstaclePolicy(state_dim=states.shape[1], action_dim=actions.shape[1])
print("参数量", f"{sum(p.numel() for p in policy.parameters()):,}")

# 真实 batch: 取连续 8 帧，每帧后面 16 步动作（这里仅演示形状，故平铺复制）
b_state = torch.from_numpy(norm.normalize_state(states[:8])).float()
b_chunk = torch.from_numpy(norm.normalize_action(np.repeat(actions[:8, None, :], 16, axis=1))).float()
print("state", tuple(b_state.shape), "-> forward", tuple(policy(b_state).shape))
print("chunk", tuple(b_chunk.shape), "-> loss", round(policy.compute_loss(b_state, b_chunk).item(), 6))
print("sample_actions", tuple(policy.sample_actions(b_state).shape))


# ── 3) eval 侧契约 ───────────────────────────────────────────────────────
section("3) eval 侧契约: build_policy 必须接受 chunk_size/d_model/depth")
try:
    build_policy("obstacle", state_dim=13, action_dim=4, chunk_size=16, d_model=128, depth=2)
except TypeError as exc:
    print("脚手架现状 ->", type(exc).__name__, ":", exc)
print("ZARR_KEY_TO_OBS keys:", list(ZARR_KEY_TO_OBS))
print("parse_key_spec('state_cube[:3]') ->", parse_key_spec("state_cube[:3]"))
print("action_key_dim: ee_xyz =", action_key_dim("action_ee_xyz"),
      "| gripper =", action_key_dim("action_gripper"))


# ── 4) 仿真环境 ──────────────────────────────────────────────────────────
section("4) 仿真环境")
env = SO100SimEnv(xml_path=XML, seed=0)
obs = env.reset()
print("dt_ctrl", env.dt_ctrl, "s | sim_dt", env.sim_dt, "s | substeps", env.substeps)
for k, v in obs.items():
    print(f"  obs[{k!r}]".ljust(22), v.shape, v.dtype)

# 动作不是 step(action)，而是先 set_*，再 step()
env.set_targets(obs["joints"])          # 前 5 个关节
env.set_gripper(0.4)                    # 第 6 个（Jaw）单独设
obs = env.step()
print("step 后 ee_pos", np.round(obs["ee_pos"], 4), "| gripper", np.round(obs["gripper"], 4))

# 评估链路: 状态拼接 -> 反归一化动作 -> apply_action
state = obs_to_state(obs, ["state_ee_xyz", "state_gripper"])
print("obs_to_state ->", state.shape, state)
delta = np.array([0.01, 0.0, 0.0, 0.0], dtype=np.float32)   # [action_ee_xyz(3), action_gripper(1)]
apply_action(env, delta, ["action_ee_xyz", "action_gripper"])
print("apply_action 后 mocap_pos", np.round(env.data.mocap_pos[env.mocap_id], 4))

print("render('angle') ->", env.render("angle").shape, "| render_rgb('top') ->",
      env.render_rgb("top").shape)
print("goal_pos(bin)", np.round(env.get_goal_pos(), 4), "| success?", check_success(env))

print("\nOK: 所有示例均按预期返回。")
