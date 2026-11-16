"""本机 HW3 训练/评估耗时基准（不改任何作业文件）。

在 ethz-robot-learning-2026/hw3_imitation_learning 目录下运行。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent  # 工作区根目录（脚本放在 .tmp/ 下）
HW3 = ROOT / "ethz-robot-learning-2026" / "hw3_imitation_learning"
sys.path.insert(0, str(HW3))

from hw3.dataset import Normalizer, SO100ChunkDataset, load_zarr  # noqa: E402
from hw3.sim_env import SO100SimEnv  # noqa: E402
from so101_gym.constants import ASSETS_DIR  # noqa: E402

ZARR = HW3 / "datasets" / "processed" / "single_cube" / "processed_ee_xyz.zarr"
XML = ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml"
H, BATCH = 16, 256


def load_dataset():
    states, actions, ep = load_zarr(
        ZARR,
        state_keys=["state_ee_xyz", "state_gripper", "state_cube[:3]", "state_obstacle"],
        action_keys=["action_ee_xyz", "action_gripper"],
    )
    norm = Normalizer.from_data(states, actions)
    ds = SO100ChunkDataset(states, actions, ep, chunk_size=H, normalizer=norm)
    return states, actions, ds


def make_model(d_state: int, d_action: int, width: int, depth: int) -> torch.nn.Sequential:
    layers, d = [], d_state
    for _ in range(depth):
        layers += [torch.nn.Linear(d, width), torch.nn.GELU()]
        d = width
    layers += [torch.nn.Linear(d, H * d_action)]
    return torch.nn.Sequential(*layers)


def bench_epochs(model, ds, batch_size: int, epochs: int = 2) -> tuple[float, int]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # warmup
    for i, (s, a) in enumerate(loader):
        loss = F.mse_loss(model(s).view(-1, H, 4), a)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i >= 2:
            break

    t0 = time.perf_counter()
    for _ in range(epochs):
        for s, a in loader:
            loss = F.mse_loss(model(s).view(-1, H, 4), a)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    dt = time.perf_counter() - t0
    return dt / epochs, len(ds)


def bench_fwd_only(model, ds, n: int = 32) -> tuple[float, float]:
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False)
    t0 = time.perf_counter()
    seen = 0
    with torch.no_grad():
        for s, a in loader:
            model(s)
            seen += 1
            if seen >= n:
                break
    return (time.perf_counter() - t0) / n, seen


def main() -> None:
    torch.manual_seed(0)
    states, actions, ds = load_dataset()
    d_s, d_a = states.shape[1], actions.shape[1]
    print(f"CPU threads: {torch.get_num_threads()} | cuda: {torch.cuda.is_available()}")
    print(f"数据: {len(ds)} 个样本 | state_dim={d_s} action_dim={d_a} chunk={H} | batch={BATCH}")
    print(f"每 epoch batch 数: {(len(ds) + BATCH - 1) // BATCH}\n")

    print("=== A. 训练一个 epoch 的耗时（含 forward+backward+step）===")
    print(f"{'模型':<22}{'参数量':>10}{'ms/batch':>10}{'s/epoch':>10}{'100ep':>9}{'200ep':>9}{'300ep':>9}")
    rows = []
    for width, depth in [(256, 3), (512, 3), (1024, 2)]:
        model = make_model(d_s, d_a, width, depth)
        n_params = sum(p.numel() for p in model.parameters())
        per_epoch, n = bench_epochs(model, ds, BATCH, epochs=2)
        n_batch = (n + BATCH - 1) // BATCH
        ms = per_epoch / n_batch * 1000
        rows.append((width, depth, n_params, ms, per_epoch))
        print(f"{f'{width}x{depth}':<22}{n_params:>10,}{ms:>9.1f}ms{per_epoch:>9.1f}s"
              f"{per_epoch*100:>8.0f}s{per_epoch*200:>8.0f}s{per_epoch*300:>8.0f}s")

    print("\n=== B. batch 大小 / 线程数对 epoch 时间的影响（256x3）===")
    for bs in (128, 256, 512):
        model = make_model(d_s, d_a, 256, 3)
        per_epoch, _ = bench_epochs(model, ds, bs, epochs=2)
        print(f"  batch={bs:<4} -> {per_epoch:6.1f} s/epoch")
    for nt in (4, 6, 10):
        torch.set_num_threads(nt)
        model = make_model(d_s, d_a, 256, 3)
        per_epoch, _ = bench_epochs(model, ds, BATCH, epochs=2)
        print(f"  threads={nt:<3} -> {per_epoch:6.1f} s/epoch")
    torch.set_num_threads(10)

    print("\n=== C. 验证集前向（no_grad）成本 ===")
    model = make_model(d_s, d_a, 256, 3)
    ms_batch, _ = bench_fwd_only(model, ds)
    print(f"  {ms_batch*1000:.1f} ms / batch({BATCH}) -> 全验证集约 {ms_batch*((len(ds)+BATCH-1)//BATCH):.2f} s/epoch")

    print("\n=== D. MuJoCo 评估单步成本（不含渲染）===")
    env = SO100SimEnv(xml_path=XML, seed=0)
    obs = env.reset()
    env.set_targets(obs["joints"])
    env.set_gripper(0.3)
    n_steps = 100
    t0 = time.perf_counter()
    for _ in range(n_steps):
        env.step()
    dt = (time.perf_counter() - t0) / n_steps
    print(f"  env.step() = {dt*1000:.2f} ms/步（内部 {env.substeps} 个物理子步）")
    print(f"  单条 episode 800 步上限 -> {dt*800:.1f} s（100 条上限 {dt*800*100/60:.0f} 分钟；"
          f"若平均 300 步成功/失败则约 {dt*300*100/60:.0f} 分钟）")


if __name__ == "__main__":
    main()
