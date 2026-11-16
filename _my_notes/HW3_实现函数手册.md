# HW3（模仿学习）实现函数手册

> 课程：ETH Zürich《Robot Learning》2026 · HW3 Imitation Learning（截止 2026-03-26）
> 代码位置：`ethz-robot-learning-2026/hw3_imitation_learning/`
> 解释器：`ethz-robot-learning-2026/hw3_imitation_learning/.venv/Scripts/python.exe`
> 本手册基于仓库 commit `2e089e9` 的实际源码 + 本机实测（2026-11-05）。凡是标 **【实测】** 的形状/数值都是当场跑出来的，不是猜的。
> 配套可运行示例：`hw3_api_examples.py`（在 `hw3_imitation_learning/` 目录下运行）

---

## 0. 先明确：真正要你写的只有两个文件

| 文件 | 你必须实现的函数 | 分值 |
|---|---|---|
| `hw3/model.py` | `ObstaclePolicy.__init__/forward/compute_loss/sample_actions`、`MultiTaskPolicy` 同名 4 个、`build_policy` | ex1/ex2/ex3 的全部性能分 |
| `scripts/train.py` | `train_one_epoch`、`evaluate`、`EPOCHS/BATCH_SIZE/LR`、`optimizer/scheduler`、补 `--extra-zarr` | 决定你能不能训出好模型 |

其余所有模块（`hw3/dataset.py`、`hw3/sim_env.py`、`hw3/eval_utils.py`、`hw3/teleop_utils.py`、`scripts/*.py`）都是**给你调用**的，不用改（`compute_actions.py` 明确写了 “do NOT modify”）。读懂它们的输入输出，比背公式更重要。

---

## 1. 数据是怎么流动的（记住这张图）

```
① 遥操作录制  scripts/record_teleop_demos.py
      → datasets/raw/single_cube/teleop/<时间戳>/so100_transfer_cube_teleop.zarr
        data/: state_joints(6) state_ee(7) state_cube(7) state_gripper(1)
               action_gripper(1) state_obstacle(3)
        meta/: episode_ends(E,)

② 算动作标签  scripts/compute_actions.py --action-space {ee|ee_full|joints}
      → datasets/processed/single_cube/processed_<space>.zarr
        data/: state_<space>  action_<space>  action_gripper  + 辅助状态
        meta/: episode_ends(E,)          （每条 episode 丢最后一步）

③ 训练       scripts/train.py
      load_zarr → Normalizer → SO100ChunkDataset → DataLoader
      → model.compute_loss  → checkpoints/<task>/best_model_<space>_<policy>.pt

④ 评估       scripts/eval.py / student_eval/run_eval.py
      load_checkpoint → obs_to_state → normalize → model.sample_actions
      → denormalize → apply_action → env.step() → check_success
```

关键约定，先背下来再看代码：

- **动作是 delta**：`a_t = s_{t+1} - s_t`（`compute_actions_for_episodes` 里就是减法）。
- **夹爪是例外**：`action_gripper` 存的是**录制时的控制指令 `data.ctrl[Jaw]`**（绝对量），不是状态差分。
- **四元数一律 wxyz**（`pos(3)+quat_wxyz(4)`），不是 scipy 的 xyzw。
- **动作 chunk 是开环执行**：一次预测 `H` 步，攒进队列，每一步 `pop(0)` 执行，队列空了才重新推理。
- **评估时 `state_joints` 会被截成前 5 维**（Jaw 由 `state_gripper`/`action_gripper` 负责）。

---

## 2. 必做 A：`hw3/model.py`

### 2.1 `BasePolicy`（基类，不要改）

```python
class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None
```

- **输入**：`state_dim` 状态向量维数；`action_dim` 动作向量维数；`chunk_size` 一次预测的动作步数 H。
- **输出**：无返回值；保存为 `self.state_dim / self.action_dim / self.chunk_size`。
- **作用**：定义策略契约。`train.py` 只调用 `compute_loss`，`eval_utils.infer_action_chunk` 只调用 `sample_actions`。
- **注意**：两个抽象方法签名固定，子类必须实现，否则实例化直接 `TypeError`。

### 2.2 `ObstaclePolicy` —— 4 个函数

#### `__init__(self, state_dim, action_dim, chunk_size=16, d_model=256, depth=3, ...)`

- **输入**：见上；`d_model` 隐藏层宽度、`depth` 线性层数（这两个名字是 `eval_utils.load_checkpoint` 约定的，**必须叫这个名**）。
- **输出**：无返回值，构造出 MLP。
- **作用**：把 `(B, state_dim)` 映射成 `(B, chunk_size*action_dim)` 再 reshape。
- **示例**：

```python
import torch, torch.nn as nn
from hw3.model import BasePolicy

p = ObstaclePolicy(state_dim=10, action_dim=4, chunk_size=16, d_model=256, depth=3)
print(p)                      # ObstaclePolicy(...)
```

#### `forward(self, state) -> torch.Tensor`

- **输入**：`state: (B, state_dim)` float32 **已归一化**的状态。
- **输出**：`(B, chunk_size, action_dim)` **已归一化**的动作 chunk。
- **作用**：一次前向，是所有其他方法的底座。
- **示例**：

```python
x = torch.randn(8, 10)
print(p(x).shape)    # 【实测】torch.Size([8, 16, 4])
```

#### `compute_loss(self, state, action_chunk) -> torch.Tensor`

- **输入**：`state: (B, state_dim)`；`action_chunk: (B, chunk_size, action_dim)`（都是 `SO100ChunkDataset` 吐出的**归一化**张量）。
- **输出**：标量 tensor（MSE），可直接 `.backward()`。
- **作用**：训练目标。行为克隆就是最简单的 MSE 回归。
- **示例**：

```python
s = torch.randn(8, 10); a = torch.randn(8, 16, 4)
loss = p.compute_loss(s, a)          # 【实测】tensor(0.80xx, grad_fn=<MseLossBackward0>)
loss.backward()                      # 梯度能回传
```

#### `sample_actions(self, state) -> torch.Tensor`

- **输入**：`state: (B, state_dim)`，评估时 `B=1`。
- **输出**：`(B, chunk_size, action_dim)`。
- **作用**：评估/DAgger 用的推理入口；`infer_action_chunk` 会 `squeeze(0)` 后逐行反归一化。
- **示例**：

```python
print(p.sample_actions(torch.randn(1, 10)).shape)   # torch.Size([1, 16, 4])
```

**完整参考实现**（可直接替换占位类，形状已实测）：

```python
class ObstaclePolicy(BasePolicy):
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
```

> **为什么一定要 `view`？** `nn.Linear` 只能输出 2D；`(B, H*A)` 必须 reshape 成 `(B, H, A)`，否则 `mse_loss` 与 `(B,H,A)` 的标签广播错，loss 看着能算但训练无效。

### 2.3 `MultiTaskPolicy` —— 同样是 4 个函数

契约与 `ObstaclePolicy` 完全一致（`__init__(state_dim, action_dim, chunk_size, d_model, depth)`、`forward`、`compute_loss`、`sample_actions`），区别只在**输入的 state 里包含目标信息**：`original_pos_cube_red/green/blue`、`state_goal`（3 维 one-hot）、`goal_pos`（bin 位置）。

- **输入**：`state: (B, state_dim)`，`state_dim` 由 `--state-keys` 决定（例如 `7+7+7+3+3=27`，再做切片可更小）。
- **输出**：同 `ObstaclePolicy`，`(B, chunk_size, action_dim)`。
- **示例（训练时命令行）**：

```bat
python scripts/train.py --policy multitask ^
  --state-keys original_pos_cube_red original_pos_cube_green original_pos_cube_blue state_goal goal_pos state_ee_xyz state_gripper ^
  --action-keys action_ee_xyz action_gripper
```

- **注意**：`MultiTaskPolicy` 本身不解析 goal，它只是“输入里多几列”。想让网络真正利用目标，可以加 FiLM/拼接/attention，但**输入维度必须与训练命令一致**，否则 `load_state_dict` 或前向报 shape 错。

### 2.4 `build_policy` —— 全局最容易翻车的一处

`eval_utils.load_checkpoint` 是这样调它的（源码 `hw3/eval_utils.py:120-127`）：

```python
model = build_policy(
    policy_type,
    state_dim=state_dim,
    action_dim=action_dim,
    chunk_size=chunk_size,   # ← 脚手架没有
    d_model=d_model,         # ← 脚手架没有（默认 128）
    depth=depth,             # ← 脚手架没有（默认 2）
)
```

而当前脚手架的签名只有 `(policy_type, *, state_dim, action_dim)`，**实测直接报错**：

```text
TypeError: build_policy() got an unexpected keyword argument 'chunk_size'
```

所以必须改成：

```python
def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int = 16,
    d_model: int = 256,
    depth: int = 3,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(state_dim=state_dim, action_dim=action_dim,
                              chunk_size=chunk_size, d_model=d_model, depth=depth)
    if policy_type == "multitask":
        return MultiTaskPolicy(state_dim=state_dim, action_dim=action_dim,
                               chunk_size=chunk_size, d_model=d_model, depth=depth)
    raise ValueError(f"Unknown policy type: {policy_type}")
```

- **输入**：策略名 + 5 个关键字参数。
- **输出**：`BasePolicy` 子类实例。
- **示例**：

```python
m = build_policy("obstacle", state_dim=10, action_dim=4, chunk_size=16, d_model=128, depth=2)
print(type(m).__name__)    # ObstaclePolicy
```

- **注意 1**：`load_checkpoint` 用 `ckpt.get("d_model", 128)` / `ckpt.get("depth", 2)` 重建网络。**如果你训练时用了非默认的 `d_model/depth`，就必须把它们写进 checkpoint**（见 3.4 第 4 条），否则加载时按默认值建网、`load_state_dict` 形状不匹配。
- **注意 2**：类名 `ObstaclePolicy` / `MultiTaskPolicy` 被 autograder 直接 import，**不能改名**；默认构造参数也要和你提交的 ckpt 一致。

---

## 3. 必做 B：`scripts/train.py`

### 3.1 三个常量

```python
EPOCHS = ...        # 建议 100~300
BATCH_SIZE = ...    # 建议 128~512（CPU 上 256 比较稳）
LR = ...            # 建议 1e-3（AdamW）
VAL_SPLIT = 0.1     # 已给
```

- **输入**：无。
- **输出**：无（模块级常量）。
- **注意**：现在是 `...`（Ellipsis），不填会在 `range(1, EPOCHS+1)` 处 `TypeError`。

### 3.2 `train_one_epoch`

```python
def train_one_epoch(model, loader, optimizer, device) -> float
```

- **输入**：`model: BasePolicy`；`loader: DataLoader`，每个 batch 是 `(states (B,state_dim), chunks (B,H,A))`；`optimizer`；`device`。
- **输出**：`float`，本 epoch 平均 loss。
- **作用**：一个 epoch 的参数更新。
- **参考实现**：

```python
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        states, action_chunks = batch
        states = states.to(device)
        action_chunks = action_chunks.to(device)

        loss = model.compute_loss(states, action_chunks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)
```

- **示例输出**：`train 0.012345`（loss 是**归一化空间**的 MSE，量级 1e-3~1e-2 都正常）。
- **注意**：`model.train()` 别丢；`zero_grad` 必须写在 `backward` 前；`loss.item()` 自动 detach。

### 3.3 `evaluate`

```python
@torch.no_grad()
def evaluate(model, loader, device) -> float
```

- **输入**：同 `train_one_epoch`（`optimizer` 不需要）。
- **输出**：`float`，验证集平均 loss。
- **作用**：挑 best checkpoint 的依据。
- **参考实现**：

```python
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        states, action_chunks = batch
        states = states.to(device)
        action_chunks = action_chunks.to(device)
        total_loss += model.compute_loss(states, action_chunks).item()
        n_batches += 1
    return total_loss / max(n_batches, 1)
```

- **注意**：`@torch.no_grad()` + `model.eval()` 缺一不可（Dropout/BN 行为 + 显存）。

### 3.4 `main()` 里必须补的 4 处（不改会直接崩）

| # | 位置 | 现状 | 必须改成 |
|---|---|---|---|
| 1 | 第 115 行 `if args.extra_zarr:` | parser **没有**定义这个参数，**实测** `AttributeError: 'Namespace' object has no attribute 'extra_zarr'` | 加 `parser.add_argument("--extra-zarr", type=Path, nargs="*", default=[])` |
| 2 | 第 158 行 `build_policy(...)` | 没传 `chunk_size` | 补 `chunk_size=args.chunk_size` |
| 3 | 第 168-170 行 | `optimizer =` / `scheduler =` 是注释 | `optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)`；`scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)` |
| 4 | 第 208-228 行 `torch.save({...})` | 不含 `d_model/depth` | 加 `"d_model": ..., "depth": ...`，且与 `build_policy` 实参一致 |

参考片段：

```python
parser.add_argument("--extra-zarr", type=Path, nargs="*", default=[],
                    help="Additional processed .zarr stores to merge (e.g. DAgger data).")

model = build_policy(
    args.policy,
    state_dim=states.shape[1],
    action_dim=actions.shape[1],
    chunk_size=args.chunk_size,
    d_model=D_MODEL,
    depth=D_DEPTH,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
```

保存时补充：

```python
torch.save({
    ...,
    "d_model": D_MODEL,
    "depth": D_DEPTH,
}, save_path)
```

### 3.5 checkpoint 里必须有哪些字段

`load_checkpoint` 反序列化时**硬性需要**：

| 字段 | 类型 | 来源 |
|---|---|---|
| `state_dim` / `action_dim` | int | `states.shape[1]` / `actions.shape[1]` |
| `chunk_size` | int | `args.chunk_size` |
| `state_keys` | list[str] | 训练时命令行参数（`obs_to_state` 必需） |
| `action_keys` | list[str] | 训练时命令行参数（`.get` 读取，缺省会导致 `apply_action` 无法切段） |
| `model_state_dict` | dict | `model.state_dict()` |
| `normalizer.{state_mean,state_std,action_mean,action_std}` | np.ndarray | `Normalizer` |

可选但强烈建议：`policy_type`、`epoch`、`val_loss`、`d_model`、`depth`（默认 128/2）。

### 3.6 训练命令（本机已验证可直接跑通数据加载部分）

```bat
cd ethz-robot-learning-2026\hw3_imitation_learning
.venv\Scripts\python.exe scripts\train.py ^
  --zarr datasets/processed/single_cube/processed_ee_xyz.zarr ^
  --policy obstacle --chunk-size 16 ^
  --state-keys state_ee_xyz state_gripper "state_cube[:3]" state_obstacle ^
  --action-keys action_ee_xyz action_gripper
```

- `--state-keys` 支持切片：`state_cube[:3]`（只要位置）、`state_cube[:5]`、`state_ee_full[:3]` 等。
- `--state-keys`/`--action-keys` 省略时回落到 zarr attrs 里的 `state_key`/`action_key`（本例只有 `state_ee_xyz`、`action_ee_xyz`，**不含夹爪**，一般不要省略）。
- 输出 checkpoint：`checkpoints/single_cube/best_model_ee_xyz_obstacle.pt`；含 dagger 数据时文件名带 `_dagger<N>ep`。
- 打印形如：`Epoch   1/200 | train 0.012345 | val 0.013456 ✓ saved`。

---

## 4. 数据管线 API：`hw3/dataset.py`

### 4.1 `load_zarr(zarr_path, state_keys=None, action_keys=None) -> (states, actions, episode_ends)`

- **输入**：
  - `zarr_path: Path` 处理后的 `.zarr` 目录；
  - `state_keys: list[str] | None`，每项可为 `"state_ee_xyz"`、`"state_cube[:3]"`、`"state_joints[2:]"`、`"state_cube[2:5]"`；
  - `action_keys: list[str] | None`，同上。
- **输出**：`states (N, D_state) float32`、`actions (N, D_action) float32`、`episode_ends (E,) int64`（累计终点、exclusive）。
- **作用**：按 spec 拼列，作为训练输入。
- **示例【实测】**（现有 24 条 episode / 4382 步的数据）：

```python
from hw3.dataset import load_zarr
states, actions, ep = load_zarr(
    "datasets/processed/single_cube/processed_ee_xyz.zarr",
    state_keys=["state_ee_xyz", "state_gripper", "state_cube[:3]", "state_obstacle"],
    action_keys=["action_ee_xyz", "action_gripper"],
)
print(states.shape, actions.shape, ep.shape)
# (4382, 10) (4382, 4) (24,)
```

- **注意**：`state_keys=None` 时用 `root.attrs["state_key"]`，只给一列，通常不够用。

### 4.2 `load_and_merge_zarrs(zarr_paths, state_keys=None, action_keys=None)`

- **输入**：多个 `.zarr` 路径列表（teleop + dagger，或 ex2 的多份数据）。
- **输出**：合并后的 `(states, actions, episode_ends)`，`episode_ends` 已按偏移平移。
- **作用**：DAgger 之后必须用它（或 `train.py --extra-zarr`）把新数据并进来。
- **示例**：

```python
from hw3.dataset import load_and_merge_zarrs
s, a, e = load_and_merge_zarrs(
    [Path("datasets/processed/single_cube/processed_ee_xyz.zarr"),
     Path("datasets/processed/single_cube/processed_ee_xyz_dagger.zarr")],
    state_keys=["state_ee_xyz", "state_gripper"],
    action_keys=["action_ee_xyz", "action_gripper"],
)
```

- **注意**：各 zarr 的 `state_keys`/`action_keys` 必须兼容（列数一致），否则 `concatenate` 报错。

### 4.3 `Normalizer`

| 方法 | 签名 | 输入 | 输出 |
|---|---|---|---|
| `from_data` | `@classmethod from_data(states, actions)` | `(N,D)` 数组 | `Normalizer` |
| `normalize_state` | `normalize_state(state)` | `(D,)` 或 `(...,D)` | 同形 `(x-mean)/std` |
| `normalize_action` | `normalize_action(action)` | `(A,)` 或 `(H,A)` | 同形 |
| `denormalize_action` | `denormalize_action(action)` | 归一化动作 | `a*std+mean` |

- **作用**：`std` 有 `1e-6` 下限（`_safe_std`），防止除零。
- **示例【实测】**：

```python
from hw3.dataset import Normalizer
norm = Normalizer.from_data(states, actions)
print(norm.state_mean.shape, norm.state_std.shape)   # (10,) (10,)
x = norm.normalize_action(actions[:5])
print(abs(norm.denormalize_action(x) - actions[:5]).max())   # 5.8e-11（往返几乎无损）
```

- **注意**：**评估时必须用 checkpoint 里存的 normalizer**，不能重新 `from_data`（否则分布不一致，动作全错）。

### 4.4 `build_valid_indices(episode_ends, chunk_size) -> np.ndarray`

- **输入**：`episode_ends (E,)`；`chunk_size`。
- **输出**：`(N_valid,) int64`，只保留“后面还有完整 H 步”的起点。
- **作用**：保证 `actions[t:t+H]` 不跨 episode 边界（不 padding）。
- **示例【实测】**：24 条 episode、4382 步、H=16 → 有效样本 4022（=4382 − 24×15），首三个 `[0 1 2]`，最后一个 `4366`。
- **注意**：短于 `chunk_size` 的 episode 会被整体跳过。

### 4.5 `SO100ChunkDataset`

```python
SO100ChunkDataset(states, actions, episode_ends, chunk_size, normalizer=None)
```

| 方法 | 输入 | 输出 |
|---|---|---|
| `__len__()` | — | `int`，有效样本数（【实测】4022） |
| `__getitem__(idx)` | `int` | `(state, action_chunk)`：`state (D,) float32`、`action_chunk (H, A) float32`，**均已归一化** |

- **示例【实测】**：

```python
from hw3.dataset import SO100ChunkDataset
ds = SO100ChunkDataset(states, actions, ep, chunk_size=16, normalizer=norm)
s, a = ds[0]
print(len(ds), s.shape, a.shape)    # 4022 torch.Size([10]) torch.Size([16, 4])
```

- **注意**：`normalizer=None` 时返回原始物理量；训练时**一定要传**，否则各维量纲差异巨大、网络学不动。

---

## 5. 动作生成 API：`scripts/compute_actions.py`

> `scripts/` 不是常规包（无 `__init__.py`），但在项目根目录下 `from scripts.compute_actions import ...` 可以 import（PEP 420 命名空间包，**实测可用**）。

### 5.1 四元数工具（wxyz）

| 函数 | 签名 | 输入 | 输出 |
|---|---|---|---|
| `quat_conjugate` | `(q)` | `(...,4)` wxyz | `(...,4)` 共轭（单位四元数即逆） |
| `quat_multiply` | `(q1, q2)` | 两个 `(...,4)` | Hamilton 积 `(...,4)` |
| `quat_to_euler` | `(q)` | `(...,4)` | `(...,3)` 的 `(roll,pitch,yaw)` |
| `_ee_full_delta` | `(s_curr, s_next)` | `(L,7)` 两个 | `(L,6)` = 位置差(3)+欧拉角差(3) |

- **示例**：

```python
import numpy as np
from scripts.compute_actions import quat_multiply, quat_conjugate, quat_to_euler
q = np.array([1.0, 0, 0, 0])                 # 单位四元数 wxyz
print(quat_multiply(q, quat_conjugate(q)))   # [1. 0. 0. 0.]
print(quat_to_euler(q))                      # [0. 0. 0.]
```

- **注意**：**只有 `--action-space ee_full` 才用到姿态差分**。`ee` / `joints` 是纯减法。

### 5.2 `select_action_space(action_space, merged) -> (raw_states, action_label, state_label, sa_suffix)`

- **输入**：`"ee"` / `"ee_full"` / `"joints"`；`merged` 是 `load_and_merge_zarrs` 返回的 dict（含 `state_ee`、`state_joints`）。
- **输出**：`raw_states (N,D)`；三个说明字符串；`sa_suffix` 用于输出数组名（`ee_xyz` / `ee_full` / `joints`）。
- **示例**：

```python
raw, a_lbl, s_lbl, suffix = select_action_space("ee", merged)
print(raw.shape, a_lbl, suffix)    # (N,3) 'ee_pos_xyz(3)' ee_xyz
```

| 选择 | state 用哪几列 | action 维数 | 输出文件名 |
|---|---|---|---|
| `ee` | `state_ee[:, :3]` | 3 | `processed_ee_xyz.zarr` |
| `ee_full` | `state_ee`（7 列全用） | 6 | `processed_ee_full.zarr` |
| `joints` | `state_joints[:, :5]`（去 Jaw） | 5 | `processed_joints.zarr` |

### 5.3 `get_episode_ranges(episode_ends) -> list[tuple[int,int]]`

- **输入**：`episode_ends (E,)`。
- **输出**：每条的 `(start, end)`（左闭右开）。
- **示例**：`[(0,154),(154,300),...]`（对应现有数据）。

### 5.4 `compute_actions_for_episodes(states, episode_ranges, action_fn=None)`

- **输入**：`states (N,D)`；`episode_ranges`；可选 `action_fn`（`ee_full` 传 `_ee_full_delta`，默认纯减）。
- **输出**：`out_states (N',D)`、`out_actions (N',D)`、`out_episode_ends (E',)`、`keep_idx (N',)`。
- **作用**：每条 episode 保留 `L-1` 个转移，`action = s[t+1]-s[t]`；`keep_idx` 用于把夹爪等其它数组对齐裁剪。
- **示例**：

```python
ranges = get_episode_ranges(merged["episode_ends"])
st, ac, new_ends, keep = compute_actions_for_episodes(raw, ranges, action_fn=None)
print(st.shape, ac.shape, new_ends.shape, keep.shape)
```

- **注意**：默认 `action_fn=None` 时 `out_actions` 与 `raw_states` 同维（纯减法）；`--action-space ee_full` 例外——state 7 维、动作 6 维（`_ee_full_delta`）。`action_gripper` 必须用 `keep_idx` 单独裁（下一步）。

### 5.5 `trim_to_transitions(merged, keep_idx, *, skip_keys)`

- **输入**：`merged` dict；`keep_idx`；已经手写过的 key 集合（避免重复写）。
- **输出**：`dict[str, np.ndarray]`，每个辅助数组按 `keep_idx` 裁剪并**改名**：`state_ee → state_ee_full`、`pos_cube_* → original_pos_cube_*`。
- **示例**：

```python
aux = trim_to_transitions(merged, keep_idx, skip_keys={"state_ee_xyz","action_ee_xyz","action_gripper"})
print(sorted(aux))    # ['state_cube', 'state_ee_full', 'state_gripper', 'state_joints', 'state_obstacle']
```

- **注意**：改名是为了避免与 `state_ee_xyz` 冲突；所以数据集里同时存在 `state_ee_full` 和 `state_ee_xyz`。

### 5.6 `load_and_merge_zarrs(zarr_paths) -> dict`

- **输入**：`.zarr` 路径列表（会自动排序）。
- **输出**：dict，键有 `state_joints/state_ee/state_cube/state_gripper/action_gripper/state_obstacle/episode_ends`（multicube 另加 `state_goal/goal_pos/pos_cube_red/green/blue`），以及内部字段 `_num_dagger_episodes`。
- **作用**：把 `datasets/raw/single_cube/` 下所有 `*.zarr`（teleop + dagger）拼成一份。
- **注意**：它打印每个 store 的 episode 数与步数；dagger 目录会标 `[dagger]` 并计入 `num_dagger_episodes`。

### 5.7 CLI 与输出

```bat
.venv\Scripts\python.exe scripts\compute_actions.py --action-space ee
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--action-space` | 必填 | `ee` / `ee_full` / `joints` |
| `--datasets-dir` | `./datasets/raw/single_cube` | 递归找 `*.zarr` |
| `--output` | `datasets/processed/<task>/processed_<suffix>.zarr` | 输出路径 |

输出 zarr（v3）结构【实测】：

```
attrs: action_space=ee, action_dim=3, action_spec='ee_pos_xyz(3)',
       state_spec='ee_pos_xyz(3)', state_key='state_ee_xyz',
       action_key='action_ee_xyz', num_episodes=24,
       num_dagger_episodes=0, num_transitions=4382, source_zarrs=[...]
data/: action_ee_xyz(4382,3) state_ee_full(4382,7) state_cube(4382,7)
       state_ee_xyz(4382,3) state_joints(4382,6) state_gripper(4382,1)
       state_obstacle(4382,3) action_gripper(4382,1)
meta/: episode_ends(24,) = [154, 300, ..., 4382]
```

---

## 6. 仿真环境 API：`hw3/sim_env.py`

### 6.1 构造

```python
SO100SimEnv(xml_path, control_hz=10.0, render_w=640, render_h=480,
            keyframe="student_start", use_mocap=True, seed=None,
            cube_pos_std=0.006, obstacle_pos_std=0.01,
            adversarial_obstacle_pos_std=0.005, obstacle_mode="train",
            obstacle_shift_x=0.08)

SO100MulticubeSimEnv(xml_path, ..., goal_cube="red", shuffle_cubes=True)
```

- **输入**：`xml_path` 用 `so101_gym.constants.ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml"`（多 cube 用 `so100_multicube_ee.xml`）。
- **输出**：环境对象（构造时已自动 `reset()` 一次）。
- **示例【实测】**：

```python
from so101_gym.constants import ASSETS_DIR
from hw3.sim_env import SO100SimEnv
env = SO100SimEnv(xml_path=ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml", seed=0)
print(env.dt_ctrl, env.sim_dt, env.substeps)    # 0.1 0.002 50
```

- **注意**：`control_hz=10` → 每步 0.1 s，内部 50 个物理子步；**评估循环里 `time.sleep(env.dt_ctrl)` 只是为了让画面实时**。

### 6.2 方法速查

| 方法 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `reset(keyframe=None)` | 可选 keyframe 名 | `obs dict` | 重置 + 场景随机化 + 返回观测 |
| `get_obs()` | — | `dict`（见 6.3） | 观测组装 |
| `step()` | **无参数** | `obs dict` | 推进一个控制周期（内部 50 个 `mj_step`） |
| `render(camera_name="angle")` | `left_wrist`/`angle`/`top` | `(480,640,3) uint8` **BGR** | 有窗口/离屏渲染 |
| `render_rgb(camera_name)` | 同上 | 同形 **RGB** | 录视频用 |
| `get_joint_angles()` | — | `(6,)` | 含 Jaw |
| `get_ee_pos()` / `get_ee_quat()` / `get_ee_state()` | — | `(3,)` / `(4,) wxyz` / `(7,)` | |
| `get_gripper_angle()` | — | `float` | Jaw qpos |
| `get_cube_state()` / `get_obstacle_pos()` / `get_goal_pos()` | — | `(7,)` / `(3,)` / `(3,)` | 单 cube |
| `set_targets(joint_targets)` | `(6,)` 或 `(5,)` | `None` | **只写前 5 个关节，跳过 Jaw** |
| `set_gripper(angle)` | `float` | `None` | 写 Jaw 的 ctrl |
| `set_mocap_pos/set_mocap_quat/set_mocap_pose` | `(3,)` / `(4,) wxyz` / 两者 | `None` | EE 目标由 mocap 驱动 |

- **示例【实测】**：

```python
obs = env.reset()
env.set_targets(obs["joints"])     # 5 个关节目标
env.set_gripper(0.4)               # 夹爪单独
obs = env.step()
print(obs["ee_pos"], obs["gripper"])   # [-0.2313 0.5 0.1153] [0.1678]
print(env.render("angle").shape)       # (480, 640, 3)
```

### 6.3 观测 dict（【实测】形状）

| key | shape | dtype | 含义 |
|---|---|---|---|
| `joints` | (6,) | float64 | 6 关节角（含 Jaw） |
| `ee` | (7,) | float64 | EE 位置(3) + 四元数 wxyz(4) |
| `ee_pos` | (3,) | float64 | EE 世界坐标 |
| `gripper` | (1,) | float64 | `[Jaw qpos]` |
| `cube` | (7,) | float64 | 方块 pos(3)+quat wxyz(4) |
| `obstacle` | (3,) | float64 | 障碍物位置；无则全零 |
| `goal_pos` | (3,) | float64 | bin 中心 |

Multicube 额外：`cubes (21,)`（红绿蓝各 7）、`cubes_xyz (9,)`、`goal (3,)` one-hot。

### 6.4 Multicube 专属

| 方法 | 输入 | 输出 |
|---|---|---|
| `set_goal(cube_color)` | `"red"` / `"green"` / `"blue"` | `None`（非法值 `ValueError`） |
| `get_goal_onehot()` | — | `(3,)` |
| `get_all_cubes_state()` | — | `(21,)` |
| `get_target_cube_state()` | — | `(7,)` |

---

## 7. 评估与推理 API：`hw3/eval_utils.py`

### 7.1 `load_checkpoint(ckpt_path, device) -> (model, normalizer, chunk_size, state_keys, action_keys)`

- **输入**：`.pt` 路径、`torch.device`。
- **输出**：5 元组；model 已 `.eval()`。
- **作用**：评估/DAgger 的统一入口；**会用 3.5 的字段重建网络**。
- **示例**：

```python
import torch; from pathlib import Path
from hw3.eval_utils import load_checkpoint
model, norm, H, sk, ak = load_checkpoint(
    Path("checkpoints/single_cube/best_model_ee_xyz_obstacle.pt"), torch.device("cpu"))
print(H, sk, ak)    # 16 ['state_ee_xyz', ...] ['action_ee_xyz', 'action_gripper']
```

- **注意**：`chunk_size` 虽返回，但 `eval.py` 不用它；实际开环长度 = `sample_actions` 输出的行数。

### 7.2 `obs_to_state(obs, state_keys) -> np.ndarray`

- **输入**：`obs` dict；训练时的 key specs。
- **输出**：`(D,) float32` 拼好的状态。
- **作用**：把仿真观测按训练顺序拼成网络输入；`state_joints` 自动截前 5 维。
- **示例【实测】**：

```python
from hw3.eval_utils import obs_to_state
print(obs_to_state(obs, ["state_ee_xyz", "state_gripper"]))
# [-0.23127867 0.5000009 0.11526269 0.16780499]
```

- **注意**：未知 key 抛 `ValueError`；key 映射见 `ZARR_KEY_TO_OBS`（`state_ee_xyz→obs["ee_pos"]`、`state_cube→obs["cube"]`、`state_goal→obs["goal"]` 等）。

### 7.3 `infer_action_chunk(model, normalizer, obs, state_keys, device) -> np.ndarray`

- **输入**：模型、normalizer、obs、keys、device。
- **输出**：`(chunk_size, action_dim)` **物理量**（已反归一化）。
- **示例**：

```python
from hw3.eval_utils import infer_action_chunk
chunk = infer_action_chunk(model, norm, env.get_obs(), sk, torch.device("cpu"))
print(chunk.shape)      # (16, 4)
```

- **注意**：内部 `torch.no_grad()`；逐行 `denormalize_action`，所以返回的每行都是可执行 delta。

### 7.4 `apply_action(env, action, action_keys) -> None`

- **输入**：`env`；单步 `action (A,)`；训练时的 `action_keys`。
- **输出**：无（直接改仿真里的 mocap / ctrl）。
- **作用**：把“delta 动作”变成“绝对目标”再写进仿真。
- **规则**：
  - `action_ee_xyz`：`mocap_pos += delta`；
  - `action_ee_full`：位置同上；姿态 `new = delta_quat ⊗ current_quat`（Hamilton 积、delta 在左，然后归一化）；
  - `action_gripper`：**绝对** `set_gripper(value)`（不是加）；
  - `action_joints`：`ctrl[:5] += delta` → `set_targets`。
- **示例【实测】**：

```python
from hw3.eval_utils import apply_action
apply_action(env, np.array([0.01, 0, 0, 0], dtype=np.float32),
             ["action_ee_xyz", "action_gripper"])
print(env.data.mocap_pos[0])   # x 增加 0.01
```

- **注意**：`action_keys` 的顺序必须与训练时**完全一致**（切段靠顺序）。

### 7.5 成功/失败判定

#### `check_success(env, xy_thresh=0.05, z_thresh=0.04) -> bool`

- **输入**：环境；阈值。
- **输出**：`bool`。
- **规则**：`|cube_xy - bin_xy| < 0.05`（multicube 收紧为 0.04）且 `0 < cube_z < 0.04`。
- **示例**：

```python
from hw3.eval_utils import check_success
print(check_success(env))    # False（刚 reset）
```

- **注意**：代码**不检查夹爪是否松开**；但 README 要求“方块必须真正落进 bin，不能夹在爪里达到 z 阈值”，官方会额外核验。

#### `check_cube_out_of_bounds(env, x_range=(-0.6,0.6), y_range=(0.1,1.1), z_min=-0.01) -> bool`

- **输出**：掉下桌子/出工作区 → `True`，评估脚本视作失败并提前终止。

#### `check_wrong_cube_in_bin(env, ...) -> str | None`

- 仅 multicube：返回误入 bin 的方块颜色名，用于警告。

### 7.6 评估循环（`scripts/eval.py: run_episode`）

```python
obs = env.reset()
while step < max_steps:
    if not action_queue:                       # 队列空 → 推理一次
        action_queue.extend(infer_action_chunk(model, norm, obs, state_keys, device))
    action = action_queue.pop(0)               # 开环执行 chunk[0], chunk[1], ...
    apply_action(env, action, action_keys)
    obs = env.step()
    step += 1
    if check_success(env): return True, ...
    if check_cube_out_of_bounds(env): return False, ...
```

- 按键：`ESC` 中止整个评估、`r` 放弃本 episode（计失败）、空格暂停。
- 注意 `use_mocap = not any("action_joints" in k for k in action_keys)`：**用关节动作策略时自动关闭 mocap 驱动**。

---

## 8. 脚本层：评估、DAgger、遥操作

### 8.1 `scripts/eval.py`

| 参数 | 默认 | 说明 |
|---|---|---|
| `--checkpoint` | 必填 | `.pt` 路径 |
| `--multicube` | off | 3 cube 场景 |
| `--num-episodes` | 10 | 评估条数（最终评分用 100） |
| `--max-steps` | 800 | 每条最多步数 |
| `--headless` | off | 不渲染（快很多） |
| `--seed` | None | 固定场景随机化 |
| `--adversarial-obstacle` | off | 障碍物对抗分布（ex2 用；README 写的 `--adversarial` 是错的） |
| `--goal-cube` / `--no-shuffle` | `all` / off | multicube 目标色与布局洗牌 |

```bat
.venv\Scripts\python.exe scripts\eval.py --checkpoint checkpoints\single_cube\best_model_ee_xyz_obstacle.pt --headless --num-episodes 20
.venv\Scripts\python.exe scripts\eval.py --checkpoint ... --multicube --goal-cube all --headless
```

### 8.2 `scripts/dagger_eval.py`

- **作用**：策略跑偏时**人接管**，把专家动作录进 `datasets/raw/single_cube/dagger/<时间戳>/...zarr`，再 `compute_actions.py` → 重新训练。
- **场景**：固定 `obstacle_mode="adversarial"`。
- **CLI**：`--checkpoint`（必填）、`--num-episodes`(10)、`--max-steps`(500)、`--seed`、`--output-dir`、`--keymap`、`--headless`。
- **按键**：`record`(Space)=接管/交还；`reset`(r)=丢弃并**重放同一场景**；`end_episode`(Enter)=丢弃进下一条；`ESC`=退出；移动键在接管态生效。
- **宽限期**：接管状态下检测到成功后，再录 `GRACE_SECS=1.7 s`（10 Hz 下 17 步），保证 action chunk 不被截断。
- **示例流程**：

```bat
.venv\Scripts\python.exe scripts\dagger_eval.py --checkpoint checkpoints\single_cube\best_model_ee_xyz_obstacle.pt
.venv\Scripts\python.exe scripts\compute_actions.py --action-space ee
.venv\Scripts\python.exe scripts\train.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr
```

### 8.3 `hw3/teleop_utils.py`

| 函数/类 | 签名 | 输入 | 输出 |
|---|---|---|---|
| `load_keymap` | `(km_path=None)` | 默认 `hw3/keymap.json` | `dict[int, str]`：`raw 键码 → 动作名` |
| `handle_teleop_key` | `(action_name, data, model, mocap_id, jaw_act_idx)` | 动作名 + 就地修改的 MuJoCo data | `None`：改 mocap/ctrl |
| `compose_camera_views` | `(images, camera_names=CAMERA_NAMES)` | 3 路 BGR 图 | `(2H, 2W, 3)` 拼图 |
| `fit_for_display` | `(img, margin=(24,96))` | BGR | 缩小后的显示副本（仅影响窗口，不影响录制） |
| `rotate_quaternion` | `(quat_wxyz, axis, angle_deg)` | wxyz + 轴 + 角度制 | `(4,)` |
| `ZarrEpisodeWriter` | 见下 | | |

`ZarrEpisodeWriter(path, joint_dim=6, ee_dim=7, cube_dim=7, gripper_dim=1, obstacle_dim=3, flush_every=12)`：

| 方法 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `append(state_joints, state_ee, state_cube, state_gripper, action_gripper, state_obstacle)` | 6 个数组 | `None` | 每帧调一次 |
| `end_episode()` | — | `None` | flush + 追加 `episode_ends` |
| `discard_episode()` | — | `None` | 回滚到上一 episode 末尾 |
| `flush()` | — | `None` | 落盘 |
| `set_attrs(**attrs)` | 任意元信息 | `None` | 写 zarr attrs |

- **注意**：`mode="w"` **会覆盖同名 zarr**；`discard_episode` 不删已经写入的 `episode_ends`。

### 8.4 `scripts/record_teleop_demos.py` / `configure_keys.py`

- `configure_keys.py`：交互式按键采集，写 `hw3/keymap.json`，格式：

```json
{
  "move_up": {"raw": 2490368, "ascii": 0, "label": "raw:2490368", "description": "Move EE up (+Z) ..."},
  "record":  {"raw": 32, "ascii": 32, "label": " ", "description": "Toggle recording on/off ..."}
}
```

- 录制输出：`datasets/raw/single_cube/teleop/<Europe/Berlin 时间戳>/so100_transfer_cube_teleop.zarr`（multicube 在 `multi_cube/`）。
- 录制频率 `control_hz=10`；每按一次方向键 EE 移动 **0.01 m**，旋转 **10°**，夹爪 **±0.10**（ctrl 单位）。
- 数据落盘顺序：`_record_step()` 在 `mj_step()` **之前**执行，所以 `state_*` 是“动作生效前”的状态。

---

## 9. 提交：`student_eval/run_eval.py`

```bat
.venv\Scripts\python.exe student_eval\run_eval.py --exercise 1 --checkpoint ex1.pt
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--exercise` | 必填 1/2/3 | 对应 ex1/ex2/ex3 |
| `--checkpoint` | `./ex{N}.pt` | 相对**当前工作目录** |
| `--output` | `./ex{N}_result.hwresult` | 签名后的提交文件 |
| `--num-episodes` | 100 | 官方评分就是 100 |
| `--seed` | 42 | **不要改** |

- 输出 `ex{N}_result.hwresult`，上传 Gradescope；同时提交 `hw3/model.py` + `ex{N}.pt`。
- **Windows 上一定失败**：`student_eval/` 只有 `eval_harness.cpython-312-darwin.so` 和 `...-x86_64-linux-gnu.so`，没有 `.pyd`，`import eval_harness` 必抛 `ImportError`。要算本地成功率请用 **WSL / Linux 机器**，或先用 `scripts/eval.py` 估。
- **不要修改 `run_eval.py`**（作业明确禁止）。

---

## 10. 端到端最小流程（本机可直接复制）

```bat
cd ethz-robot-learning-2026\hw3_imitation_learning
set PY=.venv\Scripts\python.exe

:: ① 配键（只需一次；仓库里已有 hw3/keymap.json，可跳过）
%PY% scripts\configure_keys.py

:: ② 录制 15-20 条（图形窗口；Space 开始，Enter 存一条，R 丢弃重来，ESC 退出）
%PY% scripts\record_teleop_demos.py

:: ③ 生成动作标签（三选一，推荐先 ee）
%PY% scripts\compute_actions.py --action-space ee

:: ④ 训练（先补完 model.py / train.py 的 TODO）
%PY% scripts\train.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr ^
  --state-keys state_ee_xyz state_gripper "state_cube[:3]" state_obstacle ^
  --action-keys action_ee_xyz action_gripper --chunk-size 16

:: ⑤ 评估（Windows 上只能用 eval.py）
%PY% scripts\eval.py --checkpoint checkpoints\single_cube\best_model_ee_xyz_obstacle.pt --headless --num-episodes 20

:: ⑥ ex2：DAgger 补数据 → 重算动作 → 重训 → 对抗评估
%PY% scripts\dagger_eval.py --checkpoint checkpoints\single_cube\best_model_ee_xyz_obstacle.pt
%PY% scripts\compute_actions.py --action-space ee
%PY% scripts\train.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr
%PY% scripts\eval.py --checkpoint ... --adversarial-obstacle --headless

:: ⑦ ex3：多 cube 录制 + 训练 + 评估
%PY% scripts\record_teleop_demos.py --multicube --seed 0
%PY% scripts\compute_actions.py --action-space ee --datasets-dir ./datasets/raw/multi_cube
%PY% scripts\train.py --policy multitask --zarr datasets/processed/multi_cube/processed_ee_xyz.zarr ^
  --state-keys original_pos_cube_red original_pos_cube_green original_pos_cube_blue state_goal goal_pos state_ee_xyz state_gripper ^
  --action-keys action_ee_xyz action_gripper
%PY% scripts\eval.py --checkpoint ... --multicube --headless
```

> 本机的 `pip install -e .` 已被 `_hw_local.pth` 替代，**不需要**再装包；直接跑上面的 `.venv\Scripts\python.exe` 即可。

---

## 11. 易错点（每条都有源码/实测证据）

1. **`build_policy` 必须吃 `chunk_size/d_model/depth`**：`eval_utils.load_checkpoint:120` 会传；不改就 `TypeError`（**实测**）。
2. **`train.py` 缺 `--extra-zarr` 参数**：第 115 行引用，**实测** `AttributeError: 'Namespace' object has no attribute 'extra_zarr'`。
3. **checkpoint 不含 `d_model/depth`**：`load_checkpoint` 回落到 128/2；若你用了别的宽度，加载时形状不匹配。训练时把它们写进 ckpt。
4. **`sample_actions` 返回形状必须是 `(B,H,A)`**：`infer_action_chunk` 直接 `squeeze(0)` 后按行反归一化；返回 `(B, H*A)` 会炸。
5. **动作是 delta，夹爪是绝对值**：`apply_action` 里 `action_ee_xyz/joints` 用加法，`action_gripper` 是 `set_gripper(value)`。
6. **`env.step()` 不接受动作**：必须 `set_targets/set_gripper/set_mocap_*` 写入后无参调用；写 `env.step(action)` 直接 `TypeError`。
7. **`set_targets` 跳过 Jaw**：`sim_env.py:265-269` 循环里 `continue`；夹爪必须单独 `set_gripper`。
8. **`state_joints` 评估时被截成 5 维**：`obs_to_state` 里 `raw[:5]`；训练时用 `state_joints` 全 6 维会导致评估形状不一致（差 1 维）。
9. **四元数 wxyz**：`mju_mat2Quat`、`CUBE_DIM=7 # pos+quat_wxyz`；用 scipy xyz 顺序静默错。
10. **normalizer 必须从 checkpoint 恢复**：评估时重新 `from_data` 会用评估数据分布，动作全错。
11. **`--adversarial` 是文档错误**：真实 flag 是 `--adversarial-obstacle`（`eval.py:172`）。
12. **episode 末尾不能手动太早结束**：最后一步没有后继状态，`compute_actions` 会丢最后 1 步；动作 chunk 会缺尾。
13. **训练/评估的 key 顺序必须一致**：`apply_action` 按 `action_keys` 顺序切段；顺序变了动作就错位。
14. **`record_teleop_demos` 的 zarr 是 `mode="w"`**：同一路径重跑会覆盖。
15. **DAgger 数据要重新 `compute_actions`**：dagger 目录的 zarr 只有原始状态 + `action_gripper`，没有 `action_ee_xyz`。

---

## 12. 一页速查表

| 想做这件事 | 用这个函数 |
|---|---|
| 读处理好的数据 | `dataset.load_zarr(path, state_keys, action_keys)` |
| 合并 teleop+dagger | `dataset.load_and_merge_zarrs([...])` 或 `train.py --extra-zarr` |
| 归一化 | `Normalizer.from_data` / `normalize_state` / `denormalize_action` |
| 训练样本 | `SO100ChunkDataset(...)[i] -> (state (D,), chunk (H,A))` |
| 你的模型 | `model.ObstaclePolicy.compute_loss / sample_actions` |
| 从 ckpt 恢复 | `eval_utils.load_checkpoint(path, device)` |
| obs → 网络输入 | `eval_utils.obs_to_state(obs, state_keys)` |
| 推理 | `eval_utils.infer_action_chunk(model, norm, obs, keys, device)` |
| 动作 → 仿真 | `eval_utils.apply_action(env, action, action_keys)` |
| 成功判定 | `eval_utils.check_success(env)` |
| 环境 | `sim_env.SO100SimEnv(xml_path=...)` → `reset/step/render` |
| 遥操作按键 | `teleop_utils.load_keymap` + `handle_teleop_key` |
| 录数据 | `teleop_utils.ZarrEpisodeWriter` → `append/end_episode/flush` |
| 出提交文件 | `student_eval/run_eval.py --exercise N --checkpoint exN.pt` |

---

## 附：本机实测记录（2026-11-05）

```
数据: datasets/processed/single_cube/processed_ee_xyz.zarr
  24 episodes, 4382 transitions, action_space=ee, action_dim=3
  episode_ends = [154,300,...,4382]
  示例 state_keys -> states (4382,10), actions (4382,4)
  dataset(H=16) -> 4022 samples, sample state (10,), chunk (16,4)

模型契约: state (8,10) -> forward (8,16,4); MSE loss 0.80; 参数量示例 150,848

eval 契约: build_policy(..., chunk_size=16, d_model=128, depth=2)
  当前脚手架 -> TypeError: got an unexpected keyword argument 'chunk_size'

环境: dt_ctrl=0.1s, sim_dt=0.002s, substeps=50
  obs shapes: joints(6) ee(7) ee_pos(3) gripper(1) cube(7) obstacle(3) goal_pos(3)
  render(angle) -> (480,640,3); goal_pos(bin) = [-0.2, 0.7, 0.021]; success=False
```
