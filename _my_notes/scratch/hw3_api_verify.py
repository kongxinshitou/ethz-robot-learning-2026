"""Verify the HW3 API surface documented in the handbook.

Run with the hw3 venv python from the hw3_imitation_learning directory.
"""

import numpy as np
import torch
import torch.nn as nn

from hw3.dataset import (
    Normalizer,
    SO100ChunkDataset,
    build_valid_indices,
    load_and_merge_zarrs,
    load_zarr,
)
from hw3.eval_utils import (
    ZARR_KEY_TO_OBS,
    action_key_dim,
    check_success,
    infer_action_chunk,
    load_checkpoint,
    obs_to_state,
    parse_key_spec,
)
from hw3.model import BasePolicy, build_policy
from hw3.sim_env import SO100SimEnv
from so101_gym.constants import ASSETS_DIR

ZARR = "datasets/processed/single_cube/processed_ee_xyz.zarr"
XML = ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml"

print("=" * 70)
print("1) data pipeline")
states, actions, ep_ends = load_zarr(
    ZARR,
    state_keys=["state_ee_xyz", "state_gripper", "state_cube[:3]", "state_obstacle"],
    action_keys=["action_ee_xyz", "action_gripper"],
)
print("states", states.shape, states.dtype, "actions", actions.shape, "ep_ends", ep_ends.shape)
norm = Normalizer.from_data(states, actions)
print("normalizer state_mean", norm.state_mean.shape)
ds = SO100ChunkDataset(states, actions, ep_ends, chunk_size=16, normalizer=norm)
s, a = ds[0]
print("dataset len", len(ds), "sample state", tuple(s.shape), "action_chunk", tuple(a.shape))
print("valid idx first/last", build_valid_indices(ep_ends, 16)[:3], build_valid_indices(ep_ends, 16)[-1])
print("normalize->denormalize roundtrip max err",
      float(np.abs(norm.denormalize_action(norm.normalize_action(actions[:5])) - actions[:5]).max()))

print("=" * 70)
print("2) minimal reference ObstaclePolicy (shape check only)")


class RefObstaclePolicy(BasePolicy):
    def __init__(self, state_dim, action_dim, chunk_size=16, d_model=256, depth=2):
        super().__init__(state_dim, action_dim, chunk_size)
        layers, d = [], state_dim
        for _ in range(depth):
            layers += [nn.Linear(d, d_model), nn.ReLU()]
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
print("params", sum(p.numel() for p in policy.parameters()))
batch_s = torch.from_numpy(norm.normalize_state(states[:8])).float()
batch_a = torch.from_numpy(norm.normalize_action(actions[:8, None, :].repeat(16, axis=1)[:8])).float()
print("state batch", tuple(batch_s.shape), "-> forward", tuple(policy(batch_s).shape))
print("chunk (repeated) ", tuple(batch_a.shape), "-> loss", float(policy.compute_loss(batch_s, batch_a)))
print("sample_actions", tuple(policy.sample_actions(batch_s).shape))

print("=" * 70)
print("3) eval-side contract: eval_utils.load_checkpoint calls build_policy like this")
try:
    m = build_policy("obstacle", state_dim=13, action_dim=4, chunk_size=16, d_model=128, depth=2)
    print("build_policy ok:", type(m).__name__)
except Exception as e:  # noqa: BLE001
    print("build_policy(...) ->", type(e).__name__, ":", e)

print("=" * 70)
print("4) sim env")
env = SO100SimEnv(xml_path=XML, seed=0)
obs = env.reset()
print("dt_ctrl", env.dt_ctrl, "sim_dt", env.sim_dt, "substeps", env.substeps)
for k, v in obs.items():
    print(f"  obs[{k!r}] = {v.shape} {v.dtype}")
env.set_targets(obs["joints"])
env.set_gripper(0.4)
obs2 = env.step()
print("after step: ee_pos", np.round(obs2["ee_pos"], 4), "gripper", np.round(obs2["gripper"], 4))
print("render(angle) ->", env.render("angle").shape, "| render_rgb('top') ->", env.render_rgb("top").shape)
print("success now?", check_success(env), "| goal_pos", np.round(env.get_goal_pos(), 4))
print("ZARR_KEY_TO_OBS keys:", list(ZARR_KEY_TO_OBS))
print("state from obs (ee_xyz+gripper):", obs_to_state(obs, ["state_ee_xyz", "state_gripper"]).shape)
print("action_key_dim action_ee_xyz/action_gripper:", action_key_dim("action_ee_xyz"), action_key_dim("action_gripper"))
print("parse_key_spec('state_cube[:3]'):", parse_key_spec("state_cube[:3]"))
print("=" * 70)
print("done")
