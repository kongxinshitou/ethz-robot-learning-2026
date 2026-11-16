"""HW3 ex1 代码验证（不训练、不更新参数、不生成作业 checkpoint）。

只做：
  1) 用真实 processed 数据走一遍 load_zarr -> Normalizer -> Dataset -> DataLoader
  2) build_policy + forward + compute_loss + backward（不做 optimizer.step）
  3) 按 train.py 的 save dict 存一个临时 ckpt，用 eval_utils.load_checkpoint 读回
  4) 用仿真 obs 走 obs_to_state -> normalize -> sample_actions -> denormalize -> apply_action
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HW3 = Path(__file__).resolve().parent.parent / "ethz-robot-learning-2026" / "hw3_imitation_learning"
sys.path.insert(0, str(HW3))

from hw3.dataset import Normalizer, SO100ChunkDataset, load_zarr  # noqa: E402
from hw3.eval_utils import apply_action, load_checkpoint, obs_to_state  # noqa: E402
from hw3.model import build_policy  # noqa: E402
from hw3.sim_env import SO100SimEnv  # noqa: E402
from so101_gym.constants import ASSETS_DIR  # noqa: E402

ZARR = HW3 / "datasets/processed/single_cube/processed_ee_xyz.zarr"
STATE_KEYS = ["state_ee_xyz", "state_gripper", "state_cube[:3]", "state_obstacle"]
ACTION_KEYS = ["action_ee_xyz", "action_gripper"]
CHUNK = 16

print("=" * 60)
states, actions, ep = load_zarr(ZARR, state_keys=STATE_KEYS, action_keys=ACTION_KEYS)
print(f"[1] load_zarr: states{states.shape} actions{actions.shape} ep{ep.shape}")
norm = Normalizer.from_data(states, actions)
ds = SO100ChunkDataset(states, actions, ep, chunk_size=CHUNK, normalizer=norm)
loader = DataLoader(ds, batch_size=320, shuffle=True, num_workers=0)
s, a = next(iter(loader))
print(f"    batch: states{tuple(s.shape)} chunks{tuple(a.shape)}")

print("=" * 60)
model = build_policy("obstacle", state_dim=states.shape[1], action_dim=actions.shape[1],
                     chunk_size=CHUNK, d_model=256, depth=3)
n_params = sum(p.numel() for p in model.parameters())
pred = model(s)
loss = model.compute_loss(s, a)
loss.backward()
has_grad = all(p.grad is not None for p in model.parameters())
print(f"[2] model params={n_params:,} | pred{tuple(pred.shape)} | loss={loss.item():.6f} | grads={has_grad}")

print("=" * 60)
ckpt = {
    "epoch": 0,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": {},
    "normalizer": {
        "state_mean": norm.state_mean, "state_std": norm.state_std,
        "action_mean": norm.action_mean, "action_std": norm.action_std,
    },
    "chunk_size": CHUNK, "policy_type": "obstacle",
    "state_keys": STATE_KEYS, "action_keys": ACTION_KEYS,
    "d_model": 256, "depth": 3,
    "state_dim": int(states.shape[1]), "action_dim": int(actions.shape[1]),
    "val_loss": float(loss.item()),
}
tmp_path = Path(__file__).resolve().parent / "verify_ex1.pt"
torch.save(ckpt, tmp_path)
model2, norm2, h, sk, ak = load_checkpoint(tmp_path, torch.device("cpu"))
print(f"[3] load_checkpoint OK: chunk={h} state_keys={sk} action_keys={ak}")

print("=" * 60)
env = SO100SimEnv(xml_path=ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml", seed=0)
obs = env.reset()
st = obs_to_state(obs, STATE_KEYS)
print(f"[4] obs_to_state: {st.shape} (应与训练 state_dim={states.shape[1]} 一致)")
st_norm = norm2.normalize_state(st)
with torch.no_grad():
    chunk = model2.sample_actions(torch.from_numpy(st_norm).float().unsqueeze(0))
chunk = chunk.squeeze(0).numpy()
for i in range(chunk.shape[0]):
    chunk[i] = norm2.denormalize_action(chunk[i])
print(f"    sample_actions -> {chunk.shape} | first action {np.round(chunk[0], 4)}")
apply_action(env, chunk[0], ACTION_KEYS)
obs2 = env.step()
print(f"    apply_action + step OK | ee_pos {np.round(obs2['ee_pos'], 3)}")

tmp_path.unlink(missing_ok=True)
print("=" * 60)
print("全部通过：数据管线 / 模型形状与梯度 / checkpoint 读写 / 评估推理链路 都可用。")
