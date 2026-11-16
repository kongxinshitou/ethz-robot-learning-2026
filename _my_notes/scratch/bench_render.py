"""测量不 headless 评估时每步的渲染开销（不改作业文件）。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HW3 = ROOT / "ethz-robot-learning-2026" / "hw3_imitation_learning"
sys.path.insert(0, str(HW3))

from hw3.sim_env import SO100SimEnv  # noqa: E402
from hw3.teleop_utils import CAMERA_NAMES, compose_camera_views  # noqa: E402
from so101_gym.constants import ASSETS_DIR  # noqa: E402

env = SO100SimEnv(xml_path=ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml", seed=0)
env.reset()

n = 20
t0 = time.perf_counter()
for _ in range(n):
    imgs = {cam: env.render(cam) for cam in CAMERA_NAMES}
    compose_camera_views(imgs, CAMERA_NAMES)
dt = (time.perf_counter() - t0) / n
print(f"三路渲染 + 拼图: {dt*1000:.1f} ms/步")
print(f"不 headless 时每步还有 time.sleep(0.1s) -> 合计约 {0.1 + dt:.3f} s/步")
print(f"800 步一条 -> 约 {800*(0.1+dt):.0f} s；10 条 -> 约 {10*800*(0.1+dt)/60:.1f} 分钟")
