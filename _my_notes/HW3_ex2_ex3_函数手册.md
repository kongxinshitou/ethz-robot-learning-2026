# HW3 ex2 / ex3 实现函数手册（只讲接口，不给实现）

> 课程：ETH Zürich《Robot Learning》2026 · HW3 Imitation Learning（截止 2026-03-26）
> 代码位置：`ethz-robot-learning-2026/hw3_imitation_learning/`
> 解释器：`ethz-robot-learning-2026/hw3_imitation_learning/.venv/Scripts/python.exe`
> 实测环境：Python 3.12.4、torch 2.15.0+cpu、zarr 3.4.2、mujoco 3.17.0（2026-11-06 实测）
> **本文只讲每个函数的输入/输出/调用方式，不提供 `model.py`、`train.py` 的函数体实现。**
> 标【你要写】= 作业要求你补的函数；标【已给】= 仓库里现成、直接调用；标【只用不改】= 官方脚本。

---

## 0. 先看清楚：ex2 / ex3 到底要你写什么

### 0.1 分工表

| | **ex2（DAgger）** | **ex3（Multicube，竞赛）** |
|---|---|---|
| 新增模型 | **没有**（复用 ex1 的 `ObstaclePolicy`） | **有**：`MultiTaskPolicy`（goal-conditioned） |
| 新增数据 | `datasets/raw/single_cube/dagger/` | `datasets/raw/multi_cube/teleop/` |
| 训练 | `train.py`（ex1 就写好的那套） | `train.py --policy multitask` + goal 键 |
| 评估 | `eval.py --adversarial-obstacle` | `eval.py --multicube --goal-cube all` |
| 提交 | `model.py` + `ex2.pt` + `ex2_result.hwresult` | `model.py` + `ex3.pt` + `ex3_result.hwresult` + **≤60s 视频** |

结论：**ex2 不要求你写新模型代码**，它考的是「DAgger 数据循环 + 数据合并重算 + 重训」；**ex3 唯一新增的模型代码是 `MultiTaskPolicy`**（外加 `build_policy` 支持它）。

### 0.2 真正需要改的文件

| 文件 | 内容 | ex2 | ex3 |
|---|---|---|---|
| `hw3/model.py` | `BasePolicy`（不改）、`ObstaclePolicy`、`MultiTaskPolicy`、`build_policy` | 复用 | **要写 `MultiTaskPolicy`** |
| `scripts/train.py` | `train_one_epoch`、`evaluate`、超参、optimizer/scheduler | 复用 | 复用（换 `--policy multitask`） |
| 其余所有 `hw3/*.py`、`scripts/*.py` | 数据集、环境、评估、遥操作、DAgger 循环 | 只用不改 | 只用不改 |

### 0.3 最容易踩的 5 个坑（先看）

1. **README 里的 `eval.py --adversarial` 是旧名字**；代码里实际是 `--adversarial-obstacle`（见 `scripts/eval.py`）。
2. **`state_joints` 训练时是 6 维，评估时会被截成 5 维**（Jaw 由 `state_gripper` 负责）。训练若要用它，请写 `"state_joints[:5]"`，否则训练/评估维度对不上。
3. **动作是 delta，夹爪是绝对量**：`a_t = s_{t+1} - s_t`，只有 `action_gripper` 存的是录制时的 `data.ctrl[Jaw]`。
4. **checkpoint 里的 `d_model/depth` 默认值是 128/2**（`hw3/eval_utils.py:117-118`）。如果你用 `d_model=256, depth=3` 训练，又没把这两个值存进 ckpt，`load_checkpoint` 会用 128/2 重建 → `load_state_dict` 尺寸不匹配。
5. **ex3 的 `MultiTaskPolicy` 默认构造参数必须和你训练用的 checkpoint 一致**（README 原文要求），autograder 会按默认参数重建模型。

---

## 1. 模型层：`hw3/model.py`【你要写】

### 1.1 `BasePolicy`（基类，给你不改）

```python
class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None
```

- **输入**：`state_dim` 状态维数；`action_dim` 动作维数；`chunk_size` 一次预测多少步（H）。
- **输出**：无返回值；保存 `self.state_dim / self.action_dim / self.chunk_size`。
- **作用**：定义契约。`train.py` 只调用 `compute_loss`；`eval_utils.infer_action_chunk` 只调用 `sample_actions`。
- **注意**：两个抽象方法必须实现，否则实例化就 `TypeError`。

```python
from hw3.model import BasePolicy
print(BasePolicy.__abstractmethods__)   # {'sample_actions', 'compute_loss'}
```

### 1.2 `ObstaclePolicy`（ex1 用；ex2 直接复用）

ex2 **不需要改它**，但它是 ex2 的基础。需要实现 4 个方法，签名如下：

#### `__init__(self, state_dim, action_dim, chunk_size=16, d_model=256, depth=3)`

- **输入**：`state_dim / action_dim / chunk_size`（来自 `BasePolicy`）；`d_model` 隐藏层宽度；`depth` 隐藏层数。
- **输出**：无返回值，构造出 MLP（输入 `(B, state_dim)`，输出 `(B, chunk_size*action_dim)`）。
- **必须叫这些名字**：`eval_utils.load_checkpoint` 会按 `chunk_size`、`d_model`、`depth` 三个关键字参数调用 `build_policy`，见 `hw3/eval_utils.py:117-127`。
- **示例**：

```python
p = ObstaclePolicy(state_dim=10, action_dim=4, chunk_size=16, d_model=256, depth=3)
sum(x.numel() for x in p.parameters())     # 参数量（<1M 足够 ex1/ex2）
```

#### `forward(self, state) -> torch.Tensor`

- **输入**：`state: (B, state_dim)` float32，**已归一化**。
- **输出**：`(B, chunk_size, action_dim)`，**已归一化**的动作 chunk。
- **示例**：

```python
p(torch.randn(8, 10)).shape      # -> torch.Size([8, 16, 4])
```

#### `compute_loss(self, state, action_chunk) -> torch.Tensor`

- **输入**：`state: (B, state_dim)`；`action_chunk: (B, chunk_size, action_dim)`（都是 `SO100ChunkDataset` 吐出的归一化张量）。
- **输出**：标量 tensor（MSE），可直接 `.backward()`。
- **示例**：

```python
loss = p.compute_loss(torch.randn(8, 10), torch.randn(8, 16, 4))
loss.backward()                  # tensor(0.9x, grad_fn=<MseLossBackward0>)，梯度可回传
```

#### `sample_actions(self, state) -> torch.Tensor`

- **输入**：`state: (B, state_dim)`，评估时 `B=1`。
- **输出**：`(B, chunk_size, action_dim)`。
- **作用**：`infer_action_chunk` 的推理入口；输出会被逐行反归一化。
- **示例**：

```python
p.sample_actions(torch.randn(1, 10)).shape    # -> torch.Size([1, 16, 4])
```

### 1.3 `MultiTaskPolicy`（ex3 新增）【你要写】

- **契约与 `ObstaclePolicy` 完全相同**：`__init__ / forward / compute_loss / sample_actions` 四个方法，输入输出形状一致。
- **区别只在“输入向量的含义”**：`state` 里额外拼进了目标信息（`original_pos_cube_red/green/blue`、`state_goal`、`goal_pos`），所以 `state_dim` 变大，网络第一层输入维度跟着变。
- **架构完全可以复用同一个 MLP**（README 允许“same policy as earlier”）；也可以自己设计（FiLM、attention 等），但不许引入新库。
- **默认构造参数**（`chunk_size/d_model/depth`）必须与训练 checkpoint 一致。
- **示例**（31 维输入 = 3+1+7+7+7+3+3）：

```python
m = MultiTaskPolicy(state_dim=31, action_dim=4, chunk_size=16, d_model=256, depth=3)
m(torch.randn(2, 31)).shape          # -> torch.Size([2, 16, 4])
m.compute_loss(torch.randn(2, 31), torch.randn(2, 16, 4))   # 标量
```

### 1.4 `build_policy`

```python
def build_policy(policy_type, *, state_dim, action_dim, chunk_size=16, d_model=256, depth=3) -> BasePolicy
```

- **输入**：`policy_type ∈ {"obstacle", "multitask"}`；其余为维度/容量参数。
- **输出**：`ObstaclePolicy` 或 `MultiTaskPolicy` 实例。
- **为什么必须接受这些关键字**：`hw3/eval_utils.py:120` 就是这么调的（`build_policy(policy_type, state_dim=..., action_dim=..., chunk_size=..., d_model=..., depth=...)`）。原 skeleton 里 `chunk_size/d_model/depth` 是 TODO（`git show HEAD:.../model.py`）。
- **示例**：

```python
build_policy("obstacle",  state_dim=10, action_dim=4, chunk_size=16, d_model=256, depth=3)
build_policy("multitask", state_dim=31, action_dim=4, chunk_size=16, d_model=256, depth=3)
```

### 1.5 checkpoint 里必须有的键（`train.py` 保存 / `load_checkpoint` 读取）

| 键 | 形状/类型 | 用途 |
|---|---|---|
| `model_state_dict` | dict | 权重 |
| `normalizer.state_mean/std` | `(state_dim,)` | 评估时还原状态 |
| `normalizer.action_mean/std` | `(action_dim,)` | 评估时反归一化动作 |
| `chunk_size` | int | 重建模型 |
| `policy_type` | `"obstacle"`/`"multitask"` | 重建模型 |
| `state_keys` | `list[str]` | 评估时拼 obs |
| `action_keys` | `list[str]` | 评估时拆动作 |
| `state_dim/action_dim` | int | 重建模型 |
| `d_model/depth`（**建议存**） | int | 否则 `load_checkpoint` 用默认 128/2 |

> 官方 skeleton 的 `train.py` 没存 `d_model/depth`（`git show HEAD:.../train.py`），而 `eval_utils.load_checkpoint` 缺省用 128/2。所以要么训练就用 128/2，要么自己在保存时补上这两个键（train.py 允许自由改）。

### 1.6 张量约定速记

```
state:        (B, state_dim)              float32, 归一化
action_chunk: (B, chunk_size, action_dim) float32, 归一化
forward/sample_actions: state -> action_chunk
compute_loss: (state, action_chunk) -> 标量
```

---

## 2. 数据加载与归一化：`hw3/dataset.py`【已给】

### 2.1 `load_zarr(zarr_path, state_keys=None, action_keys=None)`

- **输入**：处理后的 `.zarr` 路径；`state_keys` 是数组名 + 可选切片（`"state_cube[:3]"`）。
- **输出**：`(states, actions, episode_ends)`
  - `states: (N, state_dim) float32`
  - `actions: (N, action_dim) float32`
  - `episode_ends: (E,) int64`（每段 episode 的结束下标，累加值）
- **示例**（用你本机已有的 processed 数据）：

```python
states, actions, ep = load_zarr(
    "datasets/processed/single_cube/processed_ee_xyz.zarr",
    state_keys=["state_ee_xyz", "state_gripper", "state_cube[:3]", "state_obstacle"],
    action_keys=["action_ee_xyz", "action_gripper"],
)
# 【实测】states (4382, 10) / actions (4382, 4) / ep_ends (24,)
```

### 2.2 `Normalizer`

- **构造**：`Normalizer.from_data(states, actions)` → `Normalizer(state_mean, state_std, action_mean, action_std)`。
- **方法**：
  - `normalize_state(state) -> np.ndarray`：`(D,) -> (D,)`
  - `normalize_action(action) -> np.ndarray`：`(A,) 或 (H,A) -> 同形状`
  - `denormalize_action(action) -> np.ndarray`：同上
- **示例**：

```python
norm = Normalizer.from_data(states, actions)
norm.normalize_state(states[0]).shape        # (10,)
norm.denormalize_action(actions[0]).shape    # (4,)
```

### 2.3 `SO100ChunkDataset(states, actions, episode_ends, chunk_size, normalizer=None)`

- **输入**：上面三个数组 + `chunk_size`（H）。
- **输出（`__getitem__`）**：`(state_t, action_chunk)` → `(state_dim,)`、`(H, action_dim)` 两个 `torch.float32`。
- **`__len__`**：所有 episode 里能放下完整 chunk 的起点数（每条 episode 丢最后 `H-1` 个起点）。
- **边界**：`build_valid_indices(episode_ends, H)` 返回可用扁平下标；chunk 不跨 episode。
- **示例**：

```python
ds = SO100ChunkDataset(states, actions, ep, chunk_size=16, normalizer=norm)
len(ds)             # 【实测】4022（4382 个 transition 里去掉每条 episode 末尾不够 H 的）
s, a = ds[0]
s.shape, a.shape    # (10,), (16, 4)
```

### 2.4 `DataLoader` / `random_split`（PyTorch 自带）

- **输出**：一个 batch `states (B, state_dim)`、`action_chunks (B, H, action_dim)`。
- **示例**：

```python
from torch.utils.data import DataLoader, random_split
tr, va = random_split(ds, [len(ds)-100, 100], generator=torch.Generator().manual_seed(0))
bs, ba = next(iter(DataLoader(tr, batch_size=8)))
bs.shape, ba.shape   # 【实测】(8, 10), (8, 16, 4)
```

### 2.5 `load_and_merge_zarrs(zarr_paths, state_keys, action_keys)`（`hw3/dataset.py` 版本）

- **用途**：合并**多个 processed `.zarr`**（配合 `train.py --extra-zarr`）。
- **输出**：`(states, actions, episode_ends)`，`episode_ends` 会按偏移量平移。
- **注意**：`scripts/compute_actions.py` 里有一个**同名但签名不同**的函数（见 §3.2），别混。

```python
states, actions, ep = load_and_merge_zarrs(
    [Path("processed_a.zarr"), Path("processed_b.zarr")],
    state_keys=["state_ee_xyz"], action_keys=["action_ee_xyz"],
)
```

---

## 3. 动作标签与 DAgger 数据合并：`scripts/compute_actions.py`【只用不改】

### 3.1 动作定义（先理解）

- `a_t = s_{t+1} - s_t`；每条 episode **丢最后一步**（没有 next state）。
- `action_gripper` 例外：存录制时的 `data.ctrl[Jaw]`（绝对量），因为它要“再夹紧一点”施加力，状态差分看不出来。
- 三种动作空间：`ee`（3 维 xyz）、`ee_full`（6 维 pos+euler）、`joints`（5 维，不含 Jaw）；夹爪永远单独一维。
- 三个空间各有配套的 state 数组。

### 3.2 `load_and_merge_zarrs(zarr_paths)` → `dict`（**脚本版**）

- **输入**：**raw** `.zarr` 路径列表（teleop 和 dagger 可混在一起）。
- **输出**：合并后的 dict，键包括：
  - 单方块场景：`state_joints (N,6)`、`state_ee (N,7)`、`state_cube (N,7)`、`state_gripper (N,1)`、`state_obstacle (N,3)`、`action_gripper (N,1)`
  - 多立方体场景：没有 `state_cube`，换成 `pos_cube_red/green/blue (N,7)`，另加 `state_goal (N,3)`、`goal_pos (N,3)`
  - `episode_ends (E,)`（已按 store 顺序平移）
  - `_num_dagger_episodes`：`int`，统计路径里带 `dagger` 的 store 的 episode 数
- **示例**（用本机 raw 数据，只读）：

```python
from scripts.compute_actions import load_and_merge_zarrs
merged = load_and_merge_zarrs([Path("datasets/raw_clean/single_cube/teleop/so100_transfer_cube_teleop.zarr")])
sorted(merged)        # ['_num_dagger_episodes','action_gripper','episode_ends','state_cube','state_ee',
                      #  'state_gripper','state_joints','state_obstacle']
merged["state_ee"].shape, merged["episode_ends"].shape   # 【实测】(4406, 7), (24,)
```

### 3.3 `get_episode_ranges(episode_ends)`

- **输入**：`episode_ends (E,)`。
- **输出**：`list[tuple[int,int]]`，每个 episode 的 `(start, end)`（左闭右开）。
- **示例**：`get_episode_ranges(merged["episode_ends"])[0]  # (0, 155)`

### 3.4 `select_action_space(action_space, merged)`

- **输入**：`"ee" | "ee_full" | "joints"`；上一步的 `merged`。
- **输出**：`(raw_states, action_label, state_label, suffix)`
  - `ee` → `merged["state_ee"][:, :3]`，suffix `"ee_xyz"`
  - `ee_full` → `merged["state_ee"]`（7 维），suffix `"ee_full"`
  - `joints` → `merged["state_joints"][:, :5]`，suffix `"joints"`
- **示例**：`raw, _, _, suffix = select_action_space("ee", merged)  # (4406, 3), 'ee_xyz'`

### 3.5 `compute_actions_for_episodes(states, episode_ranges, action_fn=None)`

- **输入**：`raw_states (N,D)`、`get_episode_ranges` 的结果；`action_fn` 可传 `_ee_full_delta`（ee_full 用），默认就是减法。
- **输出**：4 元组 `(out_states, out_actions, out_episode_ends, keep_idx)`
  - `out_states: (N', D)`（每条 episode 去掉最后一步）
  - `out_actions: (N', D)`（默认 `s_{t+1}-s_t`）
  - `out_episode_ends: (E,)`
  - `keep_idx: (N',)`（保留时刻在原始数组里的下标，用来对齐其他辅助数组）
- **示例**：

```python
st, ac, ee, keep = compute_actions_for_episodes(raw, get_episode_ranges(merged["episode_ends"]))
# 【实测】st (4382, 3), ac (4382, 3), ee (24,), keep (4382,)
np.allclose(ac[0], st[1] - st[0])   # True
```

### 3.6 `trim_to_transitions(merged, keep_idx, skip_keys=...)`

- **输入**：raw merged dict、`keep_idx`、已写过的键集合 `skip_keys`。
- **输出**：`dict[str, np.ndarray]`，每个辅助数组按 `keep_idx` 截取，并做重命名：
  - `state_ee → state_ee_full`
  - `pos_cube_red/green/blue → original_pos_cube_red/green/blue`（**ex3 训练时用的就是 `original_*` 这个名字**）
- **示例**：

```python
aux = trim_to_transitions(merged, keep, skip_keys={"state_ee_xyz","action_ee_xyz","action_gripper"})
sorted(aux)   # ['state_cube','state_ee_full','state_gripper','state_joints','state_obstacle']
```

### 3.7 CLI 输出（实测）

```bash
python scripts/compute_actions.py --action-space ee
# 默认 --datasets-dir ./datasets/raw/single_cube（会递归找到 teleop 和 dagger 两个子目录）
# 输出：datasets/processed/single_cube/processed_ee_xyz.zarr
# 实测：24 episodes / 4406 steps -> 4382 transitions
# data/: state_ee_xyz(4382,3) action_ee_xyz(4382,3) action_gripper(4382,1)
#         state_ee_full(4382,7) state_cube(4382,7) state_gripper(4382,1)
#         state_joints(4382,6) state_obstacle(4382,3)
# attrs: num_episodes=24, num_dagger_episodes=0, num_transitions=4382
```

- **ex3 必须显式换目录**：`--datasets-dir ./datasets/raw/multi_cube`
  （multicube 录制落在 `datasets/raw/multi_cube/teleop/`，默认目录是 `single_cube`，不改会找不到数据）。

### 3.8 ex2 的合并逻辑（重点）

- `compute_actions.py` 用 `rglob("*.zarr")` 扫整个 `--datasets-dir`，所以：
  - `datasets/raw/single_cube/teleop/*.zarr`（ex1 数据）
  - `datasets/raw/single_cube/dagger/*/*.zarr`（ex2 新采的数据）
  会**自动合并**成一份 processed 数据；不用手动拼。
- 路径里含 `dagger` 的 store 会被计入 `_num_dagger_episodes`，`train.py` 因此给 checkpoint 起名 `best_model_*_dagger{N}ep.pt`。
- 合并后**重新算动作**是必须的：DAgger 新数据也只有状态 + `action_gripper`，delta 动作由 `s_{t+1}-s_t` 现算。

---

## 4. 推理与评估：`hw3/eval_utils.py`【已给】

### 4.1 `load_checkpoint(ckpt_path, device)`

- **输入**：`.pt` 路径、`torch.device`。
- **输出**：`(model, normalizer, chunk_size, state_keys, action_keys)`
- **示例**：

```python
model, norm, H, sk, ak = load_checkpoint(
    Path("checkpoints/single_cube/best_model_ee_xyz_obstacle.pt"), torch.device("cpu"))
# 【实测】policy_type=obstacle, epoch=60, val_loss=0.358794
#         state_dim=10, action_dim=4, chunk_size=16
#         sk=['state_ee_xyz','state_gripper','state_cube[:3]','state_obstacle']
#         ak=['action_ee_xyz','action_gripper']
```

### 4.2 `ZARR_KEY_TO_OBS`（键 → obs 字段的映射表）

| 键 | 从 obs 取什么 | 维数 |
|---|---|---|
| `state_ee_xyz` | `obs["ee_pos"]` | 3 |
| `state_ee_full` | `obs["ee"]` | 7 |
| `state_joints` | `obs["joints"]`（**评估时自动取 `[:5]`**） | 6→5 |
| `state_gripper` | `obs["gripper"]` | 1 |
| `state_cube` | `obs["cube"]`（multicube 下是目标方块） | 7 |
| `state_obstacle` | `obs["obstacle"]` | 3 |
| `goal_pos` | `obs["goal_pos"]`（箱子中心） | 3 |
| `original_pos_cube_red` | `obs["cubes"][:7]` | 7 |
| `original_pos_cube_green` | `obs["cubes"][7:14]` | 7 |
| `original_pos_cube_blue` | `obs["cubes"][14:21]` | 7 |
| `state_goal` | `obs["goal"]`（one-hot） | 3 |

- **作用**：训练时你选了哪些 state key，评估时就按同样的顺序/切片从仿真 obs 里拼回来。

### 4.3 `obs_to_state(obs, state_keys)`

- **输入**：仿真 `obs` dict、训练时的 `state_keys`。
- **输出**：`(state_dim,) float32`。
- **注意**：`state_joints` 会先 `raw = raw[:5]` 再套切片，所以训练必须写 `"state_joints[:5]"`。
- **示例**：

```python
obs = env.reset()
obs_to_state(obs, sk).shape                       # 【实测】(10,)
mst = obs_to_state(mobs, ["state_ee_xyz","state_gripper",
                          "original_pos_cube_red","original_pos_cube_green",
                          "original_pos_cube_blue","state_goal","goal_pos"])
mst.shape                                         # 【实测】(31,)
```

### 4.4 `infer_action_chunk(model, normalizer, obs, state_keys, device)`

- **输入**：模型、归一化器、obs、state_keys、device。
- **输出**：`(chunk_size, action_dim)` **已反归一化**的动作 chunk。
- **内部流程**：`obs_to_state → normalize_state → sample_actions → squeeze → 逐行 denormalize`。
- **示例**：

```python
chunk = infer_action_chunk(model, norm, obs, sk, torch.device("cpu"))
chunk.shape          # 【实测】(16, 4)
chunk[0]             # 【实测】[-0.0067 -0.0090  0.0077 -0.5327]
```

### 4.5 `apply_action(env, action, action_keys)`

- **输入**：环境、**单步**动作 `(action_dim,)`、训练时的 `action_keys`。
- **输出**：无（就地修改仿真的 mocap/ctrl）。
- **语义**：delta 动作拆段后加在当前目标上；`action_gripper` 直接 `env.set_gripper(...)`。
- **示例**：

```python
apply_action(env, chunk[0], ak)
obs = env.step()
obs["ee_pos"]       # 【实测】[-0.2371  0.4922  0.1222]
```

### 4.6 成功/越界判断

| 函数 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `check_success(env, xy_thresh=0.05, z_thresh=0.04)` | env | `bool` | 方块在箱内（multicube 自动把 xy 阈值收紧到 0.04） |
| `check_cube_out_of_bounds(env)` | env | `bool` | 方块掉出桌面/工作区 |
| `check_wrong_cube_in_bin(env)` | multicube env | `str` 或 `None` | 非目标方块进箱，返回颜色名 |

```python
check_success(env)                 # 【实测】False
check_cube_out_of_bounds(env)      # 【实测】False
check_wrong_cube_in_bin(menv)      # None 或 'red'/'green'/'blue'
```

---

## 5. 仿真环境：`hw3/sim_env.py`【已给】

### 5.1 `SO100SimEnv`（单方块 + 障碍）

**构造**：`SO100SimEnv(xml_path, control_hz=10.0, render_w=640, render_h=480, use_mocap=True, obstacle_mode="train", seed=None)`

- `obstacle_mode`：`"train"`（中心区小噪声，ex1）或 `"adversarial"`（三区偏移，ex2）。
- `use_mocap=False` 会禁用 weld（joint 动作空间时用；`eval.py` 里由 `use_mocap = not any("action_joints" in k for k in action_keys)` 决定）。

**常用方法**：

| 方法 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `reset(keyframe=None)` | — | `obs: dict` | 复位 + 随机化 + 返回首帧 obs |
| `step()` | — | `obs: dict` | 推进一个控制步（10 Hz） |
| `get_obs()` | — | `dict` | 见下表 |
| `set_mocap_pos(pos)` / `set_mocap_quat(q)` | `(3,)`/`(4,)` | None | 末端目标 |
| `set_gripper(angle)` | float | None | 夹爪控制量 |
| `get_joint_angles()` | — | `(6,)` | 6 个关节 |
| `get_ee_pos()` / `get_ee_quat()` / `get_ee_state()` | — | `(3,)`/`(4,)`/`(7,)` | wxyz |
| `get_cube_state()` | — | `(7,)` | pos(3)+quat_wxyz(4) |
| `get_obstacle_pos()` | — | `(3,)` | 无障碍时全 0 |
| `get_goal_pos()` | — | `(3,)` | 箱子中心 |
| `render(cam)` | `"left_wrist"`/`"angle"`/`"top"` | BGR `(H,W,3)` | 渲染 |

**obs 键（实测）**：`cube (7,)`、`ee (7,)`、`ee_pos (3,)`、`goal_pos (3,)`、`gripper (1,)`、`joints (6,)`、`obstacle (3,)`

```python
env = SO100SimEnv(xml_path=ASSETS_DIR/"so100_transfer_cube_obstacle_ee.xml",
                  use_mocap=True, obstacle_mode="adversarial", seed=0)
obs = env.reset()
obs["ee_pos"].shape, obs["cube"].shape      # (3,), (7,)
```

### 5.2 `SO100MulticubeSimEnv`（ex3）

**构造**：`SO100MulticubeSimEnv(xml_path, ..., goal_cube="red", shuffle_cubes=True, seed=None)`

**额外方法**：

| 方法 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `set_goal(color)` | `"red"/"green"/"blue"` | None | 每个 episode 前调用 |
| `get_goal_onehot()` | — | `(3,)` | `[red, green, blue]` one-hot |
| `get_all_cubes_state()` | — | `(21,)` | 三个方块各 pos(3)+quat_wxyz(4) |
| `get_all_cubes_xyz()` | — | `(9,)` | 三个方块的 xyz |
| `get_target_cube_state()` | — | `(7,)` | 目标方块的位姿 |
| `get_cube_state()` | — | `(7,)` | **等价于目标方块**（成功判定用） |
| `get_obs()` | — | dict | 在单方块 obs 基础上多 `cubes (21,)`、`cubes_xyz (9,)`、`goal (3,)` |

```python
menv = SO100MulticubeSimEnv(xml_path=ASSETS_DIR/"so100_multicube_ee.xml",
                            goal_cube="red", shuffle_cubes=True, seed=0)
mobs = menv.reset()
menv.set_goal("green")
menv.get_goal_onehot()     # 【实测】[0. 1. 0.]
mobs["cubes"].shape, mobs["goal"].shape, mobs["goal_pos"].shape   # (21,), (3,), (3,)
```

### 5.3 相关常量

```python
CUBE_COLORS   = ("red", "green", "blue")
GOAL_DIM      = 3
NUM_CUBES     = 3
CUBE_FREE_DIM = 7
ALL_CUBES_DIM = 21
```

---

## 6. 遥操作与写盘：`hw3/teleop_utils.py` + `scripts/record_teleop_demos.py`【已给】

### 6.1 `load_keymap(km_path=None)` / `handle_teleop_key(...)`

- `load_keymap` → `dict[int, str]`：`{raw_keycode: action_name}`（`hw3/keymap.json`）。
- `handle_teleop_key(action_name, data, model, mocap_id, jaw_act_idx)` → None；按 action 名称改 mocap/ctrl。
- **ex3 的 goal 切换键**：`goal_cube_red/green/blue`（当前 keymap 里是 h/g/f），录制时按一下切目标，**录制中不能切**。

```python
km = load_keymap()              # {32:'record', 13:'end_episode', 27:'escape', ...}
handle_teleop_key("move_up", env.data, env.model, env.mocap_id, env.act_ids[5])
```

### 6.2 `ZarrEpisodeWriter(path, joint_dim=6, ee_dim=7, cube_dim=7, gripper_dim=1, obstacle_dim=3, flush_every=12)`

| 方法/属性 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `append(state_joints, state_ee, state_cube, state_gripper, action_gripper, state_obstacle)` | `(6,)(7,)(7,)(1,)(1,)(3,)` | None | 缓冲一步；每 12 步 flush |
| `end_episode()` | — | None | flush 并写 `meta/episode_ends` |
| `discard_episode()` | — | None | 回滚到上一个 episode 末尾 |
| `flush()` | — | None | 落盘 |
| `set_attrs(**kw)` | — | None | 写 zarr 根属性 |
| `num_steps_total` / `num_episodes` | — | int | 计数 |

- **写入的数组**（实测）：`state_joints (N,6)`、`state_ee (N,7)`、`state_cube (N,7)`、`state_gripper (N,1)`、`action_gripper (N,1)`、`state_obstacle (N,3)`、`meta/episode_ends (E,)`。

```python
w = ZarrEpisodeWriter(Path(".tmp/demo.zarr"))
for _ in range(3):
    w.append(np.zeros(6), np.zeros(7), np.zeros(7), np.zeros(1), np.zeros(1), np.zeros(3))
w.end_episode()
w.num_steps_total, w.num_episodes     # 【实测】3, 1
```

### 6.3 `MulticubeZarrWriter.append_with_goal(...)`（ex3 专用）

```python
append_with_goal(state_joints, state_ee, state_cube, state_gripper,
                 action_gripper, state_obstacle, state_goal, goal_pos)
```

- `state_cube` 传 `(21,)`（三个方块拼起来），writer 会拆成 `pos_cube_red/green/blue` 三个 `(7,)`。
- 额外写：`state_goal (N,3)`、`goal_pos (N,3)`。
- **注意**：官方 `MulticubeTeleopRecorder` 构造 writer 时传 `cube_dim=0`，所以 raw 数据**没有 `state_cube` 数组**，取而代之是 `pos_cube_*`。

```python
mw = MulticubeZarrWriter(path, cube_dim=0)   # 官方 recorder 就是这么传的
mw.append_with_goal(np.zeros(6), np.zeros(7), np.zeros(21), np.zeros(1),
                    np.zeros(1), np.zeros(3), np.array([0,1,0], np.float32), np.zeros(3))
mw.end_episode()
# 【实测】data/: state_goal(1,3) goal_pos(1,3) pos_cube_red(1,7) pos_cube_green(1,7)
#               pos_cube_blue(1,7) state_joints(1,6) state_ee(1,7) state_gripper(1,1)
#               action_gripper(1,1) state_obstacle(1,3)
```

### 6.4 raw → processed 的键名对照（ex3 重点）

| 录制时（raw） | 处理后（processed，`compute_actions` 重命名） | 训练 `--state-keys` 用哪个 |
|---|---|---|
| `pos_cube_red` | `original_pos_cube_red` | ✅ `original_pos_cube_red` |
| `pos_cube_green` | `original_pos_cube_green` | ✅ `original_pos_cube_green` |
| `pos_cube_blue` | `original_pos_cube_blue` | ✅ `original_pos_cube_blue` |
| `state_goal` | `state_goal` | ✅ `state_goal` |
| `goal_pos` | `goal_pos` | ✅ `goal_pos` |
| `state_ee` | `state_ee_full` | 需要位姿时用 |
| `state_cube` | 不写（multicube） | — |

---

## 7. ex2 / ex3 的执行链（函数级）

### 7.1 ex2（DAgger）标准闭环

```
① 训练 ex1 模型（含 state_obstacle）          scripts/train.py
② 看 adversarial 失败                         scripts/eval.py --adversarial-obstacle
③ 人接管采数据                                scripts/dagger_eval.py
④ 合并 teleop+dagger、重算动作                scripts/compute_actions.py
⑤ 重训（回到 ②，可循环）                       scripts/train.py
⑥ 官方评估                                    student_eval/run_eval.py --exercise 2
```

**① 训练（例）**

```bash
python scripts/train.py \
  --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
  --policy obstacle \
  --state-keys state_ee_xyz state_gripper "state_cube[:3]" state_obstacle \
  --action-keys action_ee_xyz action_gripper
```

**② 看失败（注意参数名）**

```bash
python scripts/eval.py --checkpoint checkpoints/single_cube/best_model_ee_xyz_obstacle.pt \
  --adversarial-obstacle --headless --num-episodes 100
# README 写的 --adversarial 是旧名字，实际用 --adversarial-obstacle
```

`run_episode(env, model, normalizer, state_keys, action_keys, device, max_steps, successes, total, headless, multicube) -> (success, aborted, wrong_cube_color)`（`scripts/eval.py:41`）是每个 episode 的主循环；内部就是 `infer_action_chunk → apply_action → env.step → check_success`。

**③ DAgger 采集**

```bash
python scripts/dagger_eval.py \
  --checkpoint checkpoints/single_cube/best_model_ee_xyz_obstacle.pt \
  --num-episodes 10
```

`run_dagger_episode(...)` 的签名（`scripts/dagger_eval.py:57`）：

```python
run_dagger_episode(env, model, normalizer, state_keys, action_keys, device,
                   writer, key_to_action, *, max_steps=800,
                   successes=0, total=0, headless=False)
# 返回：(success: bool, n_takeover_steps: int, aborted: bool, replay: bool)
```

- **输入**：仿真实例、模型、归一化器、训练时的 keys、`ZarrEpisodeWriter`、`load_keymap()` 的 dict。
- **输出**：`success` 是否成功；`n_takeover_steps` 本 ep 人接管步数；`aborted` 用户按 Esc 退出；`replay` 用户按 R 要求重放同一随机场景。

行为细节：

| 按键/事件 | 行为 |
|---|---|
| `record` 键 | 切换「人接管/策略控制」；接管时清空动作队列并开始记录 |
| `reset` 键 | 丢弃本 ep 数据，**恢复 RNG 后重放同一个随机场景** |
| `Enter` | 丢弃本 ep 数据，跳到下一个 ep |
| `Esc` | 丢弃本 ep 并退出（aborted） |
| 成功且人在控制 | 继续录 1.7s（`GRACE_SECS`）再收尾，保证 chunk 完整 |
| 策略模式下成功 | 立即结束；若本 ep **从未接管**则不保存任何数据 |

- 输出：`datasets/raw/single_cube/dagger/<时间戳>/so100_transfer_cube_teleop.zarr`
- 只有「人接管期间」的步才会 append；策略自己跑的步不记。
- `--headless` 只是快速跑评估，不能接管、也不会有数据。

**④ 重算动作（自动合并 teleop+dagger）**

```bash
python scripts/compute_actions.py --action-space ee
# 默认 datasets/raw/single_cube 下 rglob，teleop 与 dagger 一起合并
# 输出 datasets/processed/single_cube/processed_ee_xyz.zarr
# attrs.num_dagger_episodes = 新增的 dagger 条数
```

**⑤ 重训**：同 ①，`--zarr` 仍指 processed（已含 dagger）。
checkpoint 名会自动带 `_dagger{N}ep`（`train.py` 里 `num_dagger_episodes` 逻辑）。

**⑥ 官方评估**

```bash
python student_eval/run_eval.py --exercise 2 --checkpoint ex2.pt
# 100 episodes, seed=42, 生成 ex2_result.hwresult
```

### 7.2 ex3（Multicube）标准闭环

```
① 采数据（按 h/g/f 切目标）     scripts/record_teleop_demos.py --multicube
② 算动作（换目录！）            scripts/compute_actions.py --datasets-dir ./datasets/raw/multi_cube
③ 训练 multitask                scripts/train.py --policy multitask + 5 个 goal 键
④ 评估                          scripts/eval.py --multicube --goal-cube all
⑤ 官方评估                      student_eval/run_eval.py --exercise 3
⑥ 录 ≤60s 视频（讲方法/数据/最终 SR）
```

**① 录制**

```bash
python scripts/record_teleop_demos.py --multicube
# 输出 datasets/raw/multi_cube/teleop/<ts>/so100_multicube_teleop.zarr
# h/g/f（keymap）在录制前切换 red/green/blue 目标；录制中不能切
```

**② 算动作**

```bash
python scripts/compute_actions.py --action-space ee \
  --datasets-dir ./datasets/raw/multi_cube
# 输出 datasets/processed/multi_cube/processed_ee_xyz.zarr
# 多出：original_pos_cube_red/green/blue (N,7)、state_goal (N,3)、goal_pos (N,3)
```

**③ 训练（state_dim 算法）**

```bash
python scripts/train.py \
  --zarr datasets/processed/multi_cube/processed_ee_xyz.zarr \
  --policy multitask \
  --state-keys state_ee_xyz state_gripper \
               "original_pos_cube_red[:3]" "original_pos_cube_green[:3]" "original_pos_cube_blue[:3]" \
               state_goal goal_pos \
  --action-keys action_ee_xyz action_gripper
```

- 该例 `state_dim = 3 + 1 + 3 + 3 + 3 + 3 + 3 = 19`；不切片则 `3+1+7+7+7+3+3 = 31`。
- `state_goal` 顺序固定 `[red, green, blue]`。
- 目标信息（三个方块位置 + one-hot + 箱子位置）**必须**进 `state`，否则策略无法区分任务。

**④ 评估**

```bash
python scripts/eval.py --checkpoint checkpoints/multi_cube/best_model_ee_xyz_multitask.pt \
  --multicube --goal-cube all --headless --num-episodes 100
# --goal-cube all 会按 red→green→blue 轮换（build_goal_schedule）
```

**⑤ 官方评估 + 视频**

```bash
python student_eval/run_eval.py --exercise 3 --checkpoint ex3.pt
# 100 episodes, seed=42, 生成 ex3_result.hwresult
```

视频要求（README）：≤60s `.mp4`，讲清楚：你的 ex3 思路、实现过程（成功/失败原因）、用了多少数据、最终 SR。

---

## 8. 对齐检查清单

### ex2

- [ ] 训练时 `--state-keys` 里包含 `state_obstacle`（否则 adversarial 必崩）。
- [ ] checkpoint 的 `state_keys/action_keys/normalizer/d_model/depth` 能被 `load_checkpoint` 正确还原。
- [ ] DAgger 数据确实采到了（`writer.num_episodes > 0`；策略自己成功的 ep 不会存数据）。
- [ ] `compute_actions.py` 的 `--datasets-dir` 能同时覆盖 `teleop/` 和 `dagger/`（默认就是 `single_cube`）。
- [ ] 重算后 `attrs.num_dagger_episodes > 0`，checkpoint 名带 `_dagger`。
- [ ] 评估用 `--adversarial-obstacle`（不是 `--adversarial`）。
- [ ] `run_eval.py --exercise 2` 前，模型默认参数与 ckpt 一致，文件名 `ex2.pt`。

### ex3

- [ ] 录制用了 `--multicube`，数据在 `datasets/raw/multi_cube/teleop/`。
- [ ] `compute_actions.py` 用了 `--datasets-dir ./datasets/raw/multi_cube`。
- [ ] 训练 `--policy multitask`，且 `--state-keys` 含 `original_pos_cube_*`、`state_goal`、`goal_pos`（可切片）。
- [ ] `state_goal` 的 one-hot 顺序与 `CUBE_COLORS` 一致：red, green, blue。
- [ ] 评估用 `--multicube`；`run_eval.py --exercise 3`。
- [ ] `MultiTaskPolicy` 的默认构造参数 = 训练 ckpt 的重建参数。
- [ ] 视频 ≤60s，含方法/数据量/最终 SR。
- [ ] 100 次评估 seed=42，不要改 `run_eval.py`。

### 通用（ex2/ex3 都适用）

- [ ] 四元数一律 **wxyz**（不是 scipy 的 xyzw）。
- [ ] `state_joints` 要写 `[:5]`；Jaw 用 `state_gripper`。
- [ ] 动作用 delta；`action_gripper` 是绝对控制量。
- [ ] chunk 开环执行：一次推 H 步，`pop(0)` 执行，队列空才重新推理。
- [ ] 成功判定：方块在箱内、z 在阈值内，且不能还在夹爪里。
- [ ] 评估上限 800 步（`--max-steps`）。
- [ ] 归一化只在训练数据上算，存进 ckpt，评估时用同一个。

---

## 9. 函数速查表

| 函数 | 所在文件 | 输入 | 输出 |
|---|---|---|---|
| `BasePolicy.__init__` | `hw3/model.py` | `state_dim, action_dim, chunk_size` | — |
| `ObstaclePolicy.__init__` | `hw3/model.py` | 上述 + `d_model, depth` | 实例 |
| `ObstaclePolicy.forward` | `hw3/model.py` | `(B, state_dim)` | `(B, H, A)` |
| `ObstaclePolicy.compute_loss` | `hw3/model.py` | `(B,state_dim), (B,H,A)` | 标量 |
| `ObstaclePolicy.sample_actions` | `hw3/model.py` | `(B, state_dim)` | `(B, H, A)` |
| `MultiTaskPolicy.*` | `hw3/model.py` | 同上（state 含 goal） | 同上 |
| `build_policy` | `hw3/model.py` | `policy_type, state_dim, action_dim, chunk_size, d_model, depth` | 策略实例 |
| `load_zarr` | `hw3/dataset.py` | 路径 + key 列表 | `(states, actions, episode_ends)` |
| `Normalizer.from_data` | `hw3/dataset.py` | `states, actions` | `Normalizer` |
| `normalize_state/normalize_action/denormalize_action` | `hw3/dataset.py` | 数组 | 同形状 |
| `SO100ChunkDataset.__getitem__` | `hw3/dataset.py` | 下标 | `((state_dim,), (H,A))` |
| `load_and_merge_zarrs` | `hw3/dataset.py` | processed 路径列表 + keys | `(states, actions, episode_ends)` |
| `load_and_merge_zarrs` | `scripts/compute_actions.py` | raw 路径列表 | `dict`（含 `_num_dagger_episodes`） |
| `get_episode_ranges` | `scripts/compute_actions.py` | `episode_ends` | `[(start,end)]` |
| `select_action_space` | `scripts/compute_actions.py` | `space, merged` | `(raw_states, labels, suffix)` |
| `compute_actions_for_episodes` | `scripts/compute_actions.py` | `states, ranges, action_fn` | `(states, actions, ep_ends, keep_idx)` |
| `trim_to_transitions` | `scripts/compute_actions.py` | `merged, keep_idx, skip_keys` | `dict`（含重命名） |
| `load_checkpoint` | `hw3/eval_utils.py` | `ckpt_path, device` | `(model, normalizer, H, state_keys, action_keys)` |
| `obs_to_state` | `hw3/eval_utils.py` | `obs, state_keys` | `(state_dim,)` |
| `infer_action_chunk` | `hw3/eval_utils.py` | `model, normalizer, obs, state_keys, device` | `(H, A)` 反归一化 |
| `apply_action` | `hw3/eval_utils.py` | `env, action, action_keys` | None |
| `check_success` | `hw3/eval_utils.py` | `env` | `bool` |
| `check_cube_out_of_bounds` | `hw3/eval_utils.py` | `env` | `bool` |
| `check_wrong_cube_in_bin` | `hw3/eval_utils.py` | multicube env | `str` 或 `None` |
| `SO100SimEnv.reset/step` | `hw3/sim_env.py` | — | `obs: dict` |
| `SO100MulticubeSimEnv.set_goal` | `hw3/sim_env.py` | `"red"/"green"/"blue"` | None |
| `SO100MulticubeSimEnv.get_goal_onehot` | `hw3/sim_env.py` | — | `(3,)` |
| `ZarrEpisodeWriter.append` | `hw3/teleop_utils.py` | 6 个数组 | None |
| `ZarrEpisodeWriter.end_episode` | `hw3/teleop_utils.py` | — | None |
| `MulticubeZarrWriter.append_with_goal` | `scripts/record_teleop_demos.py` | 8 个数组 | None |
| `load_keymap` | `hw3/teleop_utils.py` | 路径 | `dict[int,str]` |
| `handle_teleop_key` | `hw3/teleop_utils.py` | `action_name, data, model, mocap_id, jaw_act_idx` | None |

---

### 附：本文实测用的命令（可复现）

```bash
cd ethz-robot-learning-2026/hw3_imitation_learning
.venv/Scripts/python.exe -c "import torch, zarr, mujoco; print(torch.__version__, zarr.__version__, mujoco.__version__)"
# 2.15.0+cpu 3.4.2 3.17.0
```

> 提醒：仓库里 `hw3/model.py`、`scripts/train.py` 当前已有一版可运行的实现（之前会话留下的）。本文只按作业要求讲**接口和契约**；如果你要自己从零写，按 §1 的签名和 §8 的检查清单来即可。
