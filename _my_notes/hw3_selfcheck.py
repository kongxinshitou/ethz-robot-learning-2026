#!/usr/bin/env python3
"""HW3 自检脚本 —— 边写边跑，确认每个函数“形状/契约”对了。

定位：这不是作业答案，也不修改任何作业文件。它只 import 你的代码、跑断言、
告诉你“还差什么、错在哪、下一步做什么”。

用法（在任意目录都行，脚本会自动找作业目录）：
    python hw3_selfcheck.py                  # 全部核心检查（数据 + 模型 + 训练 + 环境，约 10 秒）
    python hw3_selfcheck.py --smoke          # 额外用真实数据训练 3 个 epoch，看 loss 是否下降
    python hw3_selfcheck.py --skip-env       # 跳过 MuJoCo 环境检查
    python hw3_selfcheck.py --root <目录>     # 手动指定 ethz-.../hw3_imitation_learning

建议用作业自己的解释器运行：
    ethz-robot-learning-2026\\hw3_imitation_learning\\.venv\\Scripts\\python.exe hw3_selfcheck.py

退出码：0 = 核心项全部通过；1 = 还有未实现/失败的核心项（MultiTaskPolicy 等 ex3 项不计入）。
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

# ── 结果记录 ────────────────────────────────────────────────────────────────


class Report:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.todo: list[str] = []          # 核心 TODO（影响退出码）
        self.optional: list[str] = []      # 可选（ex3），不影响退出码
        self.failed: list[str] = []        # 真 bug（影响退出码）

    def ok(self, title: str) -> None:
        self.passed.append(title)
        print(f"  ✓ {title}")

    def not_yet(self, title: str, detail: str = "", optional: bool = False) -> None:
        (self.optional if optional else self.todo).append(title)
        tag = "（可选，ex3 用）" if optional else ""
        print(f"  ○ {title}  —— 还没实现{tag}")
        if detail:
            print(f"      {detail}")

    def bad(self, title: str, exc: BaseException, hint: str = "") -> None:
        self.failed.append(title)
        print(f"  ✗ {title}")
        print(f"      {type(exc).__name__}: {exc}")
        if hint:
            print(f"      提示：{hint}")

    def summary(self) -> int:
        print("\n" + "=" * 74)
        print(f"通过 {len(self.passed)} 项 | 待实现 {len(self.todo)} 项 | "
              f"失败 {len(self.failed)} 项 | 可选未做 {len(self.optional)} 项")
        if self.todo:
            print("\n还没写完的核心部分：")
            for t in self.todo:
                print(f"  ○ {t}")
        if self.failed:
            print("\n报错的部分（按提示修）：")
            for t in self.failed:
                print(f"  ✗ {t}")
        if not self.todo and not self.failed:
            print("\n核心检查全部通过。下一步：")
            print("  1) 真实训练:  python scripts\\train.py --zarr <processed.zarr> --state-keys ... --action-keys ...")
            print("  2) 看效果:    python scripts\\eval.py --checkpoint <ckpt> --headless --num-episodes 20")
        print("=" * 74)
        return 0 if not self.failed and not self.todo else 1


def is_unimplemented(exc: BaseException) -> bool:
    """区分“脚手架还没写”和“真 bug”。"""
    if isinstance(exc, NotImplementedError):
        return True
    if isinstance(exc, (TypeError, AttributeError)):
        s = str(exc)
        if "keyword argument" in s or "missing" in s or "has no attribute" in s:
            return True
    return False


def try_check(rep: Report, title: str, fn, hint: str = "", optional: bool = False) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - 自检脚本要兜住一切
        if is_unimplemented(exc):
            rep.not_yet(title, optional=optional)
        else:
            rep.bad(title, exc, hint)
    else:
        rep.ok(title)


# ── 定位作业目录 / 数据 ─────────────────────────────────────────────────────


def find_project_root(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if (p / "hw3").is_dir() and (p / "scripts" / "train.py").is_file():
            return p
        raise SystemExit(f"--root 不是有效作业目录（找不到 hw3/ 和 scripts/train.py）：{p}")

    here = Path(__file__).resolve().parent
    for cand in (Path.cwd(), here, here / "ethz-robot-learning-2026" / "hw3_imitation_learning",
                 Path.cwd() / "ethz-robot-learning-2026" / "hw3_imitation_learning"):
        if (cand / "hw3").is_dir() and (cand / "scripts" / "train.py").is_file():
            return cand.resolve()
    raise SystemExit(
        "找不到作业目录。请在 ethz-robot-learning-2026/hw3_imitation_learning 下运行，或用 --root 指定。"
    )


def find_zarr(root: Path, explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if p.exists() else None
    default = root / "datasets" / "processed" / "single_cube" / "processed_ee_xyz.zarr"
    if default.exists():
        return default
    for p in (root / "datasets" / "processed").rglob("*.zarr"):
        return p
    return None


def sig_default(builder, name: str, fallback):
    """读 build_policy 某个参数的默认值；支持 **kwargs 写法。"""
    params = inspect.signature(builder).parameters
    if name in params:
        d = params[name].default
        return fallback if d is inspect.Parameter.empty else d
    return fallback


def accepts_var_kwargs(builder) -> bool:
    return any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in inspect.signature(builder).parameters.values())


# ── [1/4] 数据管线 ──────────────────────────────────────────────────────────


def check_data(rep: Report, zarr_path: Path):
    print("\n[1/4] 数据管线（hw3/dataset.py —— 只调用，不用改）")

    import numpy as np

    from hw3.dataset import Normalizer, SO100ChunkDataset, load_zarr

    state_keys = ["state_ee_xyz", "state_gripper", "state_cube[:3]", "state_obstacle"]
    action_keys = ["action_ee_xyz", "action_gripper"]

    states, actions, ep_ends = load_zarr(zarr_path, state_keys=state_keys, action_keys=action_keys)
    d_state, d_action = states.shape[1], actions.shape[1]

    def _shapes() -> None:
        assert states.ndim == 2 and actions.ndim == 2, "load_zarr 应返回二维 (N, D) 数组"
        assert states.shape[0] == actions.shape[0] == ep_ends[-1], "状态/动作/步数对不上"
        assert d_state == 10 and d_action == 4, \
            f"预期 state_dim=10 / action_dim=4，实际 {d_state}/{d_action}（检查 key 顺序与列数）"

    try_check(rep, f"load_zarr -> states{states.shape}, actions{actions.shape}, ep{ep_ends.shape}", _shapes)

    normalizer = Normalizer.from_data(states, actions)

    def _norm() -> None:
        err = np.abs(normalizer.denormalize_action(normalizer.normalize_action(actions[:5])) - actions[:5]).max()
        assert err < 1e-5, f"归一化/反归一化往返误差过大：{err}"

    try_check(rep, "Normalizer 归一化后能无损还原", _norm)

    dataset = SO100ChunkDataset(states, actions, ep_ends, chunk_size=16, normalizer=normalizer)

    def _dataset() -> None:
        s, a = dataset[0]
        assert tuple(s.shape) == (d_state,), f"state 形状应为 ({d_state},)，实际 {tuple(s.shape)}"
        assert tuple(a.shape) == (16, d_action), f"chunk 形状应为 (16, {d_action})，实际 {tuple(a.shape)}"
        assert len(dataset) > 0, "数据集为空"

    try_check(rep, f"SO100ChunkDataset(H=16) -> {len(dataset)} 个样本，sample {tuple(dataset[0][0].shape)} / {tuple(dataset[0][1].shape)}",
              _dataset, "一定要传 normalizer，否则各维量纲差异巨大")

    return d_state, d_action, state_keys, action_keys


# ── [2/4] 策略 ──────────────────────────────────────────────────────────────


def check_model(rep: Report, d_state: int, d_action: int,
                state_keys: list[str], action_keys: list[str]) -> None:
    print("\n[2/4] 策略（hw3/model.py —— 你要写的部分）")

    import torch

    try:
        from hw3.model import BasePolicy, build_policy
    except Exception as exc:  # noqa: BLE001
        rep.bad("import hw3.model", exc, "先修语法/import 错误")
        return

    # 2.1 签名契约：eval_utils.load_checkpoint 会传这 5 个关键字参数
    def _sig() -> None:
        params = inspect.signature(build_policy).parameters
        missing = [p for p in ("state_dim", "action_dim", "chunk_size", "d_model", "depth")
                   if p not in params]
        if set(missing) == {"chunk_size", "d_model", "depth"}:
            # 脚手架的原始状态：只有 state_dim/action_dim —— 属于“还没写”
            raise NotImplementedError("build_policy 还只有 state_dim/action_dim")
        assert not missing and not accepts_var_kwargs(build_policy), \
            f"build_policy 少参数：{missing}（eval_utils.py:120-127 会传 chunk_size/d_model/depth）"

    try_check(rep, "build_policy 接受 state_dim/action_dim/chunk_size/d_model/depth", _sig,
              "参考 eval_utils.load_checkpoint 的调用方式，不要只留 state_dim/action_dim")

    # 2.2 构造（用 build_policy 自己的默认宽度/深度，保证与 ckpt 重建一致）
    chunk, batch = 16, 8
    d_model = sig_default(build_policy, "d_model", 128)
    depth = sig_default(build_policy, "depth", 2)
    model = None

    def _build() -> None:
        nonlocal model
        model = build_policy("obstacle", state_dim=d_state, action_dim=d_action,
                             chunk_size=chunk, d_model=d_model, depth=depth)
        assert isinstance(model, BasePolicy), "build_policy 应返回 BasePolicy 子类"
        assert (model.state_dim, model.action_dim, model.chunk_size) == (d_state, d_action, chunk), \
            "BasePolicy 的三个属性没设对：__init__ 里要 super().__init__(state_dim, action_dim, chunk_size)"

    try_check(rep, "build_policy('obstacle', ...) 能构造出模型", _build,
              "ObstaclePolicy.__init__ 必须接收 chunk_size/d_model/depth 并调用 super().__init__")

    if model is None:
        rep.not_yet("ObstaclePolicy 的 forward / compute_loss / sample_actions（先修构造）")
        return

    # 2.3 forward 形状
    x = torch.randn(batch, d_state)

    def _forward() -> None:
        y = model(x)
        assert tuple(y.shape) == (batch, chunk, d_action), (
            f"forward 输出应为 ({batch}, {chunk}, {d_action})，实际 {tuple(y.shape)}；"
            "线性层输出 (B, H*A) 后要 view(-1, H, A)")

    try_check(rep, f"forward: {tuple(x.shape)} -> ({batch}, {chunk}, {d_action})", _forward,
              "view(self.chunk_size * self.action_dim) 的思路：先展平输出再 reshape")

    # 2.4 compute_loss 标量 + 可反传
    target = torch.randn(batch, chunk, d_action)

    def _loss() -> None:
        loss = model.compute_loss(x, target)
        assert isinstance(loss, torch.Tensor) and loss.dim() == 0, "compute_loss 必须返回 0 维标量 tensor"
        loss.backward()
        assert any(p.grad is not None for p in model.parameters()), \
            "backward 后没有梯度：别对预测 detach / 别把 loss 变成 python float"

    try_check(rep, "compute_loss 返回标量且能 backward（梯度已产生）", _loss,
              "loss = F.mse_loss(model(state), action_chunk)，不要 .item()")

    # 2.5 sample_actions
    def _sample() -> None:
        with torch.no_grad():
            y = model.sample_actions(torch.randn(1, d_state))
        assert tuple(y.shape) == (1, chunk, d_action), \
            f"sample_actions 应为 (1, {chunk}, {d_action})，实际 {tuple(y.shape)}"

    try_check(rep, "sample_actions: (1, D) -> (1, H, A)", _sample)

    # 2.6 与 eval_utils.load_checkpoint 的兼容性（最容易翻车）
    def _roundtrip() -> None:
        import numpy as np

        from hw3.eval_utils import load_checkpoint

        ckpt = {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "normalizer": {
                "state_mean": np.zeros(d_state, np.float32),
                "state_std": np.ones(d_state, np.float32),
                "action_mean": np.zeros(d_action, np.float32),
                "action_std": np.ones(d_action, np.float32),
            },
            "chunk_size": chunk,
            "policy_type": "obstacle",
            "state_keys": state_keys,
            "action_keys": action_keys,
            "state_dim": d_state,
            "action_dim": d_action,
            "val_loss": 0.0,
            "d_model": sig_default(build_policy, "d_model", 128),
            "depth": sig_default(build_policy, "depth", 2),
        }
        # 不用 TemporaryDirectory：某些受限环境里临时目录不可写
        path = Path(__file__).resolve().parent / ".hw3_selfcheck_ckpt.pt"
        try:
            torch.save(ckpt, path)
            loaded, _, h, sk, ak = load_checkpoint(path, torch.device("cpu"))
            assert (h, sk, ak) == (chunk, state_keys, action_keys), "load_checkpoint 返回的元数据不对"
            with torch.no_grad():
                y = loaded.sample_actions(torch.randn(1, d_state))
            assert tuple(y.shape) == (1, chunk, d_action), f"加载后前向形状不对：{tuple(y.shape)}"
        finally:
            path.unlink(missing_ok=True)

    try_check(rep, "checkpoint 能被 eval_utils.load_checkpoint 重建并推理", _roundtrip,
              "ckpt 必须有 state_dim/action_dim/chunk_size/state_keys/action_keys/model_state_dict/normalizer")

    # 2.7 提醒：默认宽度与 load_checkpoint 缺省值不一致时，必须把 d_model/depth 存进 ckpt
    if (d_model, depth) != (128, 2):
        print(f"  ! 提醒：你的 build_policy 默认 (d_model, depth) = ({d_model}, {depth})，"
              "而 load_checkpoint 缺省按 (128, 2) 重建网络。")
        print("         → 训练保存 checkpoint 时必须写入 \"d_model\" 和 \"depth\"（自检已替你写入并验证）。")

    # 2.8 MultiTaskPolicy（ex3；不算核心 TODO）
    def _multitask() -> None:
        m = build_policy("multitask", state_dim=27, action_dim=d_action,
                         chunk_size=8, d_model=64, depth=2)
        y = m(torch.randn(4, 27))
        assert tuple(y.shape) == (4, 8, d_action), f"MultiTaskPolicy 输出形状不对：{tuple(y.shape)}"

    try_check(rep, "MultiTaskPolicy（ex3）: 27 维状态 -> (B, 8, A)", _multitask, optional=True)


# ── [3/4] 训练脚本 ──────────────────────────────────────────────────────────


def check_train(rep: Report, d_state: int, d_action: int, root: Path) -> None:
    print("\n[3/4] 训练脚本（scripts/train.py —— 你要写的部分）")

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    try:
        import scripts.train as tr
    except Exception as exc:  # noqa: BLE001
        rep.bad("import scripts.train", exc, "先修模块级语法/import 错误")
        return

    def _consts() -> None:
        if any(isinstance(v, type(Ellipsis)) for v in (tr.EPOCHS, tr.BATCH_SIZE, tr.LR)):
            raise NotImplementedError("EPOCHS/BATCH_SIZE/LR 还是 ...")
        assert isinstance(tr.EPOCHS, int) and tr.EPOCHS > 0, f"EPOCHS 要正整数，现在是 {tr.EPOCHS!r}"
        assert isinstance(tr.BATCH_SIZE, int) and tr.BATCH_SIZE > 0, f"BATCH_SIZE 要正整数，现在是 {tr.BATCH_SIZE!r}"
        assert isinstance(tr.LR, (int, float)) and tr.LR > 0, f"LR 要正数，现在是 {tr.LR!r}"

    try_check(rep, f"超参常量已填（EPOCHS={tr.EPOCHS}, BATCH_SIZE={tr.BATCH_SIZE}, LR={tr.LR}）",
              _consts, "把 ... 换成真实数字，例如 200 / 256 / 1e-3")

    def _extra_zarr() -> None:
        src = (root / "scripts" / "train.py").read_text(encoding="utf-8")
        if '"--extra-zarr"' not in src and "'--extra-zarr'" not in src:
            raise NotImplementedError(
                "还缺 --extra-zarr：main() 第 115 行会读 args.extra_zarr，现在会 AttributeError"
            )

    try_check(rep, "main() 已定义 --extra-zarr 参数", _extra_zarr)

    class ToyPolicy(torch.nn.Module):
        """只用来测试你写的训练循环，不参与作业评分。"""

        def __init__(self) -> None:
            super().__init__()
            self.h, self.a = 4, d_action
            self.net = torch.nn.Linear(d_state, self.h * self.a)

        def compute_loss(self, state, action_chunk):
            pred = self.net(state).view(state.shape[0], self.h, self.a)
            return torch.nn.functional.mse_loss(pred, action_chunk)

    def _loop() -> None:
        torch.manual_seed(0)
        toy = ToyPolicy()
        loader = DataLoader(
            TensorDataset(torch.randn(32, d_state), torch.randn(32, 4, d_action)),
            batch_size=8, shuffle=True,
        )
        opt = torch.optim.AdamW(toy.parameters(), lr=1e-3)
        before = [p.detach().clone() for p in toy.parameters()]

        loss = tr.train_one_epoch(toy, loader, opt, torch.device("cpu"))
        assert isinstance(loss, float), \
            f"train_one_epoch 应返回 float，实际 {type(loss).__name__}（是不是忘了 return，或返回了 tensor？）"
        assert loss == loss, f"loss 是 NaN：{loss}"
        changed = any(not torch.equal(b, p) for b, p in zip(before, toy.parameters()))
        if not changed and loss == 0.0:
            # 脚手架的循环体只有注释：直接返回 0.0、参数不动
            raise NotImplementedError("训练循环还只有注释，没写更新步骤")
        assert changed, "参数没更新：检查 optimizer.step() 是否被调用、zero_grad 是否放在 backward 前"

        val = tr.evaluate(toy, loader, torch.device("cpu"))
        assert isinstance(val, float), f"evaluate 应返回 float，实际 {type(val).__name__}"

    try_check(rep, "train_one_epoch / evaluate 能跑通（toy 模型，测你的循环）", _loop,
              "batch → .to(device) → compute_loss → zero_grad → backward → step；累计 loss.item() 再取平均")


# ── [4/4] 仿真环境 ──────────────────────────────────────────────────────────


def check_env(rep: Report, root: Path, assets: Path | None) -> None:
    print("\n[4/4] 仿真环境（hw3/sim_env.py —— 只调用，不用改）")

    assets_dir = assets or (root / "so101_gym" / "assets")
    xml = assets_dir / "so100_transfer_cube_obstacle_ee.xml"
    if not xml.exists():
        rep.not_yet("仿真环境检查（找不到 xml）", str(xml))
        return

    from hw3.sim_env import SO100SimEnv

    env = None

    def _construct() -> None:
        nonlocal env
        env = SO100SimEnv(xml_path=xml, seed=0)
        obs = env.reset()
        expected = {"joints": (6,), "ee": (7,), "ee_pos": (3,), "gripper": (1,),
                    "cube": (7,), "obstacle": (3,), "goal_pos": (3,)}
        for k, shape in expected.items():
            assert k in obs, f"缺少观测 key：{k}"
            assert tuple(obs[k].shape) == shape, f"obs['{k}'] 应为 {shape}，实际 {tuple(obs[k].shape)}"

    try_check(rep, "SO100SimEnv 构造 + reset + 观测 key/形状", _construct,
              "xml 用 ASSETS_DIR / 'so100_transfer_cube_obstacle_ee.xml'")

    if env is None:
        return

    def _step() -> None:
        obs = env.get_obs()
        env.set_targets(obs["joints"])   # 只写前 5 关节，Jaw 被跳过
        env.set_gripper(0.3)             # 夹爪单独设
        obs2 = env.step()                # step() 不接受动作参数
        assert tuple(obs2["ee_pos"].shape) == (3,)
        assert abs(env.dt_ctrl - 0.1) < 1e-9, f"dt_ctrl 应为 0.1s，实际 {env.dt_ctrl}"

    try_check(rep, f"set_targets/set_gripper/step 推进（dt={env.dt_ctrl}s, substeps={env.substeps}）",
              _step, "动作先 set_* 再无参 step()，不要写 env.step(action)")

    def _render() -> None:
        img = env.render("angle")
        assert tuple(img.shape) == (480, 640, 3), f"render 形状应为 (480, 640, 3)，实际 {tuple(img.shape)}"

    try_check(rep, "render('angle') -> (480, 640, 3) BGR", _render)


# ── 额外：真实数据小训练 ────────────────────────────────────────────────────


def smoke_train(rep: Report, zarr_path: Path, d_state: int, d_action: int,
                state_keys: list[str], action_keys: list[str]) -> None:
    print("\n[额外] 真实数据小训练（--smoke）：3 个 epoch，看 val loss 是否下降")

    import torch
    from torch.utils.data import DataLoader, random_split

    from hw3.dataset import Normalizer, SO100ChunkDataset, load_zarr
    from hw3.model import build_policy

    import scripts.train as tr

    states, actions, ep_ends = load_zarr(zarr_path, state_keys=state_keys, action_keys=action_keys)
    normalizer = Normalizer.from_data(states, actions)
    dataset = SO100ChunkDataset(states, actions, ep_ends, chunk_size=16, normalizer=normalizer)

    n_val = max(1, int(len(dataset) * 0.1))
    train_ds, val_ds = random_split(
        dataset, [len(dataset) - n_val, n_val], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = build_policy("obstacle", state_dim=d_state, action_dim=d_action,
                         chunk_size=16, d_model=128, depth=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses: list[float] = []
    for epoch in range(1, 4):
        tr.train_one_epoch(model, train_loader, opt, torch.device("cpu"))
        losses.append(tr.evaluate(model, val_loader, torch.device("cpu")))
        print(f"      epoch {epoch}: val_loss = {losses[-1]:.6f}")

    def _converge() -> None:
        assert losses[-1] < losses[0], (
            f"3 epoch 后 val_loss 没下降（{losses[0]:.6f} → {losses[-1]:.6f}）；"
            "常见原因：没归一化、学习率过大、forward 的 reshape 顺序错")

    try_check(rep, "真实数据 3 epoch 小训练，val loss 下降", _converge)


# ── main ───────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="HW3 自检脚本")
    parser.add_argument("--root", default=None, help="作业目录（含 hw3/ 和 scripts/train.py）")
    parser.add_argument("--zarr", default=None, help="processed zarr 路径（默认自动找）")
    parser.add_argument("--assets", default=None, help="so101_gym/assets 路径")
    parser.add_argument("--skip-env", action="store_true", help="跳过 MuJoCo 环境检查")
    parser.add_argument("--smoke", action="store_true", help="额外做真实数据 3-epoch 小训练")
    args = parser.parse_args()

    root = find_project_root(args.root)
    sys.path.insert(0, str(root))
    print(f"作业目录: {root}")

    try:
        import torch  # noqa: F401
    except ImportError:
        print("✗ 没找到 torch。请用作业自己的解释器：")
        print("  .venv\\Scripts\\python.exe hw3_selfcheck.py")
        return 1

    rep = Report()

    zarr_path = find_zarr(root, args.zarr)
    if zarr_path is None:
        print("\n[1/4] 数据管线：跳过（还没生成 processed .zarr）")
        print("      先录数据 → scripts/compute_actions.py --action-space ee")
        d_state, d_action = 10, 4
        state_keys = ["state_ee_xyz", "state_gripper", "state_cube[:3]", "state_obstacle"]
        action_keys = ["action_ee_xyz", "action_gripper"]
    else:
        print(f"数据: {zarr_path}")
        d_state, d_action, state_keys, action_keys = check_data(rep, zarr_path)

    check_model(rep, d_state, d_action, state_keys, action_keys)
    check_train(rep, d_state, d_action, root)
    if not args.skip_env:
        check_env(rep, root, Path(args.assets) if args.assets else None)
    if args.smoke and zarr_path is not None:
        smoke_train(rep, zarr_path, d_state, d_action, state_keys, action_keys)

    return rep.summary()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断")
        sys.exit(130)
