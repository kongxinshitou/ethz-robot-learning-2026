# HW3 ex2 / ex3 代码实现 + 数据采集 / 训练 / 测试 实操指南

> 路径：`ethz-robot-learning-2026/hw3_imitation_learning/`
> Python：`.venv/Scripts/python.exe`（Python 3.12.4 / torch 2.15.0+cpu / zarr 3.4.2 / mujoco 3.17.0）
> 下文的 `python` 都指 `.venv/Scripts/python.exe`（或先激活 venv 再直接用 `python`）。
> 本指南的命令都在这台 Windows 机器上实测跑过；**官方 `student_eval/run_eval.py` 需要 Linux**（见 §7）。

---

## 0. 先看结论

- ex1 的 `ObstaclePolicy` + `train.py` 之前已经有完整实现；**ex2 不需要新模型代码**，只需要「DAgger 采数据 → 合并重算动作 → 重训 → adversarial 测试」。
- ex3 的新模型 `MultiTaskPolicy` 已经实现：它和 `ObstaclePolicy` 共用同一个 MLP，goal 信息作为 `state` 的额外列输入（`original_pos_cube_*` + `state_goal` + `goal_pos`）。
- 本轮我把 `train.py` 补成了「可调超参 + 可 warm start + 可指定 checkpoint 目录」的版本，并给模型构造函数加了默认参数；然后跑通了：
  - 模型 forward/loss 形状
  - 单方块 2 epoch 训练 + 评估
  - DAgger headless 循环
  - teleop+dagger 数据合并（48 episodes / 24 dagger）
  - multicube 数据写盘 → compute_actions → 训练 → 评估 全链路（用合成数据验证）
- 你接下来真正要做的只有两件事：**录数据** + **跑命令**。

---

## 1. 本轮改了哪些代码

### 1.1 `hw3/model.py`

（原本已有 `ObstaclePolicy` / `MultiTaskPolicy` / `build_policy` 的实现，本轮只加默认参数）

```python
class _MLPChunkPolicy(BasePolicy):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 16,   # ← 新增默认值
        d_model: int = 256,     # ← 新增默认值
        depth: int = 3,         # ← 新增默认值
    ) -> None:
        ...
```

- 作用：即使别人不显式传 `chunk_size/d_model/depth` 也能构造；`build_policy` 与 `eval_utils.load_checkpoint` 仍然按 checkpoint 里的值显式传入。
- 两个类的契约保持一致：
  - `forward(state (B,D)) -> (B,H,A)`
  - `compute_loss(state, action_chunk) -> 标量 MSE`
  - `sample_actions(state) -> (B,H,A)`

### 1.2 `scripts/train.py`（主要改动）

新增命令行参数（默认值沿用当前文件里的 600 / 320 / 1e-3 / 0.1）：

| 参数 | 作用 |
|---|---|
| `--epochs` | 训练轮数（默认 600） |
| `--batch-size` | batch 大小（默认 320） |
| `--lr` | 学习率（默认 1e-3） |
| `--weight-decay` | AdamW 权重衰减（默认 1e-4） |
| `--val-split` | 验证集比例（默认 0.1） |
| `--num-workers` | DataLoader 进程数（默认 0） |
| `--checkpoint-dir` | 覆盖 checkpoint 输出目录（默认 `checkpoints/single_cube` 或 `checkpoints/multi_cube`） |
| `--init-from` | 可选热启动：从已有 `.pt` 加载权重（ex2 可从 ex1 checkpoint 继续） |

其他行为：
- 训练循环、scheduler 的 `T_max`、epoch 打印都改用 `args.epochs` 等参数（不再写死常量）。
- checkpoint 里额外保存 `"epochs"`；`d_model/depth/chunk_size/state_keys/action_keys` 原本就会保存。
- 若 `--init-from` 的模型尺寸对不上，会打印 warning 并自动从零训练（不会崩）。
- 若传入 `--checkpoint-dir`，实验不会覆盖 `checkpoints/` 下的正式模型（方便 smoke test）。

> 其余文件（`dataset.py`、`eval_utils.py`、`sim_env.py`、`teleop_utils.py`、`eval.py`、`dagger_eval.py`、`compute_actions.py`、`record_teleop_demos.py`）本轮**没有改**，按官方逻辑直接使用。

---

## 2. 我实际跑过的冒烟测试（结果）

| # | 测试 | 命令/方式 | 结果 |
|---|---|---|---|
| 1 | 模型契约 | 构造 `ObstaclePolicy(10,4)` / `MultiTaskPolicy(31,4)` | forward `(2,16,4)`、sample `(1,16,4)`、loss 标量 ✅ |
| 2 | 单方块训练 | 2 epochs，chunk=4，d_model=512，batch=320 | 约 **6.2 s**，val 0.19，checkpoint 正常保存 ✅ |
| 3 | warm start | `--init-from` 加载 ex1 checkpoint | 尺寸匹配时打印 `Warm-started...`；不匹配时自动从零训练 ✅ |
| 4 | 单方块评估 | `eval.py --adversarial-obstacle --headless` | 正常跑完并打印 SR（2 epoch 模型当然会失败） ✅ |
| 5 | DAgger 循环 | `dagger_eval.py --headless --num-episodes 1 --max-steps 50` | 正常跑完；没接管 → 不写数据（符合逻辑） ✅ |
| 6 | teleop+dagger 合并 | 复制 24-ep teleop 到 `teleop/`，另复制一份到 `dagger/`，跑 `compute_actions.py` | `num_episodes=48`、`num_dagger_episodes=24`、`num_transitions=8764` ✅ |
| 7 | multicube 数据链路 | 用 `MulticubeZarrWriter` 写 3×25 步合成数据 → `compute_actions.py --datasets-dir .../multi_cube` | 输出含 `original_pos_cube_red/green/blue (72,7)`、`state_goal (72,3)`、`goal_pos (72,3)` ✅ |
| 8 | ex3 训练/评估 | `train.py --policy multitask`（state_dim=19，2 epochs）→ `eval.py --multicube --headless` | 正常训练、加载、跑 episode ✅ |
| 9 | 官方 harness | `student_eval/run_eval.py` | ❌ Windows 上没有 `.pyd`（只有 darwin/linux 的 `.so`），需要 Linux/WSL/学生机 |

> 合成数据只用于验证管道，测完已删除；你的真实 multicube 数据仍需自己遥操作录制。

---

## 3. 你现有 ex1 checkpoint 的规格（很重要）

```text
checkpoints/single_cube/best_model_ee_xyz_obstacle.pt
  epoch        = 378
  val_loss     = 0.1857
  chunk_size   = 4        ← 注意不是默认的 16
  d_model      = 512      ← 注意不是默认的 256
  depth        = 3
  state_dim    = 10
  action_dim   = 4
  state_keys   = state_ee_xyz, state_gripper, "state_cube[:3]", state_obstacle
  action_keys  = action_ee_xyz, action_gripper
```

所以 ex2 想 warm start / 复现 ex1 性能，训练命令必须带：

```bash
--chunk-size 4 --d-model 512 --depth 3
```

检查任意 checkpoint 的规格：

```bash
.venv/Scripts/python.exe -c "import torch; c=torch.load('checkpoints/single_cube/best_model_ee_xyz_obstacle.pt', map_location='cpu', weights_only=False); print({k:c.get(k) for k in ['epoch','val_loss','chunk_size','d_model','depth','state_dim','action_dim','policy_type','state_keys','action_keys']})"
```

---

## 4. ex2：DAgger 数据采集 + 训练 + 测试

### 4.1 先看 baseline（可选，但建议）

```bash
cd ethz-robot-learning-2026/hw3_imitation_learning

# 可视化跑（会开窗口，按 ESC 退出）
.venv/Scripts/python.exe scripts/eval.py \
  --checkpoint checkpoints/single_cube/best_model_ee_xyz_obstacle.pt \
  --adversarial-obstacle

# 快速跑 100 条，看成功率
.venv/Scripts/python.exe scripts/eval.py \
  --checkpoint checkpoints/single_cube/best_model_ee_xyz_obstacle.pt \
  --adversarial-obstacle --headless --num-episodes 100
```

> README 里写的 `--adversarial` 是旧名字，代码里实际是 **`--adversarial-obstacle`**。

### 4.2 用 DAgger 采数据

**推荐把 dagger 数据写进 `datasets/raw_clean/single_cube/dagger/`**，这样可以和 ex1 用的清洗版 teleop 数据（24 episodes / 4406 steps）一起重算动作，避免混入 `datasets/raw/single_cube` 里那两条很长（约 2.9 万步）的原始 teleop。

```bash
.venv/Scripts/python.exe scripts/dagger_eval.py \
  --checkpoint checkpoints/single_cube/best_model_ee_xyz_obstacle.pt \
  --num-episodes 10 \
  --output-dir datasets/raw_clean/single_cube/dagger/round1
```

> `dagger_eval.py` 目前是**原版**（没有 `--cameras` 等参数）。卡的话参考 §10：少量多次、提前接管、关掉其他程序。

**操作方式**（`scripts/dagger_eval.py`）：

| 按键 | 行为 |
|---|---|
| `Space`（你 keymap 里的 `record`） | **切换人接管/策略控制**；接管时开始记录 |
| `R` | 丢弃本 ep 数据，**恢复随机种子后重放同一个场景**（方便你提前接管重试） |
| `Enter` | 丢弃本 ep 数据，跳到下一个 ep |
| `Esc` | 丢弃本 ep 并退出 |

关键细节：
- **只有人接管的那些步才会写盘**；策略自己跑的步不记录。如果整条 episode **从头到尾都没接管**，即使策略自己成功了也不会产生数据。
- 人接管完成放置后，成功检测到会再录 **1.7 s** 然后保存（保证 chunk 完整）；不要按 Enter 结束，否则会丢弃。
- 站在策略视角观察：快偏出轨迹/快撞障碍时按 Space 接管，手动完成，比失败后再救更有效。
- 不要长时间静止：DAgger 也会把静止步录进去。
- 建议先采 **10~25 条**；如果失败模式还在，重复「采 → 重算 → 重训」2~3 轮。
- 屏幕会显示 `DAgger steps` 和 `Episodes saved`，结束时打印保存路径。

数据检查：

```bash
.venv/Scripts/python.exe -c "import glob,zarr; [print(p, zarr.open_group(p,mode='r')['meta']['episode_ends'].shape[0], 'episodes') for p in glob.glob('datasets/raw_clean/single_cube/dagger/*/*.zarr')]"
```

### 4.3 合并 teleop + dagger 并重算动作

```bash
.venv/Scripts/python.exe scripts/compute_actions.py \
  --action-space ee \
  --datasets-dir datasets/raw_clean/single_cube
```

- `compute_actions.py` 会递归扫描该目录下所有 `.zarr`，所以 `teleop/` 和 `dagger/` 自动合并。
- 会重新计算 delta 动作（`a_t = s_{t+1} - s_t`），最后一步丢弃。
- 输出：`datasets/processed/single_cube/processed_ee_xyz.zarr`。

检查：

```bash
.venv/Scripts/python.exe -c "import zarr; r=zarr.open_group('datasets/processed/single_cube/processed_ee_xyz.zarr',mode='r'); print(dict(r.attrs)); print({k:r['data'][k].shape for k in r['data']})"
```

重点看：
- `num_episodes` 增加（原 24 + 新增 dagger 条数）
- `num_dagger_episodes > 0`
- `action_ee_xyz`、`action_gripper` 存在

### 4.4 重训

```bash
.venv/Scripts/python.exe scripts/train.py \
  --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
  --policy obstacle \
  --state-keys state_ee_xyz state_gripper "state_cube[:3]" state_obstacle \
  --action-keys action_ee_xyz action_gripper \
  --chunk-size 4 --d-model 512 --depth 3 \
  --epochs 600 --batch-size 320 \
  --init-from checkpoints/single_cube/best_model_ee_xyz_obstacle.pt
```

- 不想热启动就去掉 `--init-from`（从头训练）。
- 输出 checkpoint 会带 dagger 数量，例如：
  `checkpoints/single_cube/best_model_ee_xyz_obstacle_dagger10ep.pt`
- CPU 时间参考：4.3k 样本、chunk=4、batch=320 时 **2 epoch ≈ 6.2 s**；600 epoch ≈ 30~40 分钟（数据变多会更久）。
- 想快速验证流程可以先跑 `--epochs 5`。

### 4.5 本地测试

```bash
.venv/Scripts/python.exe scripts/eval.py \
  --checkpoint checkpoints/single_cube/best_model_ee_xyz_obstacle_dagger10ep.pt \
  --adversarial-obstacle --headless --num-episodes 100
```

- 输出总 SR；想可视化就去掉 `--headless`。
- 建议同时跑训练分布（`--headless` 不加 `--adversarial-obstacle`）看有没有掉点。

### 4.6 官方评估（需要 Linux）

```bash
# 在 Linux / WSL / 学生机上
cp checkpoints/single_cube/best_model_ee_xyz_obstacle_dagger10ep.pt ex2.pt
python student_eval/run_eval.py --exercise 2 --checkpoint ex2.pt
# 生成 ex2_result.hwresult
```

提交物：`model.py` + `ex2.pt` + `ex2_result.hwresult`。

---

## 5. ex3：Multicube 数据采集 + 训练 + 测试

### 5.1 录制 multicube 数据

```bash
.venv/Scripts/python.exe scripts/record_teleop_demos.py --multicube
```

> 已按你的要求把 `record_teleop_demos.py` 也恢复成原版（没有 `--cameras` 等参数）；卡的话参考 §10 的“少量多次、提前结束、关掉其他程序”。

- 场景里红/绿/蓝三个方块位置随机，箱子位置也随机；目标颜色由你指定。
- **目标切换键**：`h`=红、`g`=绿、`f`=蓝（你当前 `keymap.json` 里是这三个；录制中不能切）。
- 操作：`Space` 开始/暂停录制 → 抓目标方块 → 放进箱子 → `Enter` 结束本 ep 并重置。`R` 丢弃本 ep，`Esc` 退出并保存。
- 建议：**每种颜色 15~20 条**（共 45~60 条），至少先录 30 条跑通流程。
- 不要把 episode 拖太长（静止步会污染数据）。

输出：

```text
datasets/raw/multi_cube/teleop/<时间戳>/so100_multicube_teleop.zarr
```

检查：

```bash
.venv/Scripts/python.exe -c "import glob,zarr; p=sorted(glob.glob('datasets/raw/multi_cube/teleop/*/*.zarr'))[-1]; r=zarr.open_group(p,mode='r'); print(p); print('episodes=', r['meta']['episode_ends'].shape[0]); print({k:r['data'][k].shape for k in r['data']})"
```

应看到：`state_goal (N,3)`、`goal_pos (N,3)`、`pos_cube_red/green/blue (N,7)`、`state_ee (N,7)`、`state_joints (N,6)`、`action_gripper (N,1)`。

### 5.2 算动作

```bash
.venv/Scripts/python.exe scripts/compute_actions.py \
  --action-space ee \
  --datasets-dir datasets/raw/multi_cube
```

输出：`datasets/processed/multi_cube/processed_ee_xyz.zarr`。

检查是否会多出 ex3 需要的关键数组：

```bash
.venv/Scripts/python.exe -c "import zarr; r=zarr.open_group('datasets/processed/multi_cube/processed_ee_xyz.zarr',mode='r'); print({k:r['data'][k].shape for k in r['data'] if 'cube' in k or k in ('state_goal','goal_pos')})"
```

应看到 `original_pos_cube_red/green/blue (N,7)`（注意已从 `pos_cube_*` 改名）、`state_goal (N,3)`、`goal_pos (N,3)`。

### 5.3 训练

```bash
.venv/Scripts/python.exe scripts/train.py \
  --zarr datasets/processed/multi_cube/processed_ee_xyz.zarr \
  --policy multitask \
  --state-keys state_ee_xyz state_gripper \
               "original_pos_cube_red[:3]" "original_pos_cube_green[:3]" "original_pos_cube_blue[:3]" \
               state_goal goal_pos \
  --action-keys action_ee_xyz action_gripper \
  --chunk-size 4 --d-model 512 --depth 3 \
  --epochs 600 --batch-size 320
```

- 该例 `state_dim = 3 + 1 + 3 + 3 + 3 + 3 + 3 = 19`；如果三个方块都用完整 7 维，则 `31`。
- 必须包含 `state_goal` 和 `goal_pos`，否则策略不知道「抓哪个、放哪」。
- 输出：`checkpoints/multi_cube/best_model_ee_xyz_multitask.pt`。
- 首次建议先用 `--epochs 20` 确认能跑，再拉满 600。

### 5.4 本地测试

```bash
.venv/Scripts/python.exe scripts/eval.py \
  --checkpoint checkpoints/multi_cube/best_model_ee_xyz_multitask.pt \
  --multicube --goal-cube all --headless --num-episodes 100
```

- 会打印总 SR + 每种颜色的 SR。
- 可视化：去掉 `--headless`。
- 想固定颜色测试：`--goal-cube red`。

### 5.5 官方评估 + 视频（需要 Linux）

```bash
cp checkpoints/multi_cube/best_model_ee_xyz_multitask.pt ex3.pt
python student_eval/run_eval.py --exercise 3 --checkpoint ex3.pt
# 生成 ex3_result.hwresult
```

视频（README 要求）：
- ≤60s `.mp4`
- 讲：你的 ex3 思路、实现过程（哪里成功/失败）、用了多少数据、最终 SR

---

## 6. 常用检查命令

### 6.1 模型形状自检（不训练）

```bash
.venv/Scripts/python.exe -c "
import torch
from hw3.model import ObstaclePolicy, MultiTaskPolicy, build_policy
p = ObstaclePolicy(10, 4)
print(p(torch.randn(2,10)).shape)          # (2,16,4)
print(p.sample_actions(torch.randn(1,10)).shape)
print(float(p.compute_loss(torch.randn(2,10), torch.randn(2,16,4))))
m = MultiTaskPolicy(31, 4)
print(m(torch.randn(2,31)).shape)          # (2,16,4)
"
```

### 6.2 checkpoint 自检

```bash
.venv/Scripts/python.exe -c "
import torch
from hw3.eval_utils import load_checkpoint
m, n, H, sk, ak = load_checkpoint('checkpoints/single_cube/best_model_ee_xyz_obstacle.pt', torch.device('cpu'))
print(H, sk, ak)
"
```

### 6.3 zarr 自检

```bash
.venv/Scripts/python.exe -c "
import zarr
r = zarr.open_group('datasets/processed/single_cube/processed_ee_xyz.zarr', mode='r')
print(dict(r.attrs))
print({k: r['data'][k].shape for k in r['data']})
"
```

---

## 7. Windows 上跑官方 `run_eval.py` 的问题

`student_eval/` 里只有两个编译好的 harness：

```text
eval_harness.cpython-312-darwin.so          (macOS)
eval_harness.cpython-312-x86_64-linux-gnu.so (Linux)
```

没有 Windows 的 `.pyd`，所以本机直接跑 `python student_eval/run_eval.py ...` 会 `ImportError`。本机 WSL 目前也没装 Linux 发行版（`wsl -l -v` 为空），Docker 也未安装。

可行方案（三选一）：
1. **安装 WSL Ubuntu**：`wsl --install -d Ubuntu`，在 Ubuntu 里建 Python 3.12 环境、`uv pip install -e .`，再跑 `run_eval.py`。
2. 用课程提供的 **student PC / Linux 机器**跑官方评估。
3. 暂时用 `scripts/eval.py --headless --num-episodes 100` 作为本地指标（不是官方 `.hwresult`，但足够调模型）。

> 注意：不改 `run_eval.py`；Windows 上也别试图把 `.so` 改名成 `.pyd`，架构不对。

---

## 8. 提分/调参建议

### ex2

- 先确保 **训练时包含 `state_obstacle`**（否则 adversarial 分布直接崩）。
- DAgger 的收益在于「在策略会犯错的状态附近补专家动作」：接管点越靠近失败状态越好。
- 数据量：10~25 条 dagger 通常够；如果不够，优先增加模型容量而不是盲目加数据。
- 模型：ex1 用的 `chunk_size=4, d_model=512, depth=3` 是已验证的配置；可以试 `d_model=768`、`depth=4`（<1M 参数）。
- 评估：`--num-episodes 100` 才是接近最终评分的数字；10 条波动很大。

### ex3

- 每种颜色都要有足够数据；如果红色录得多、蓝绿少，按颜色看 SR 会很不均。
- 输入特征尽量精简：`original_pos_cube_*[:3]`（位置）通常比完整 7 维（含四元数）更好学。
- 目标信息：`state_goal`（one-hot）+ `goal_pos`（箱子中心）必须保留。
- 模型容量：ex3 比 ex1/ex2 难，可以增大到 `d_model=768, depth=4`，或增大 chunk（8/16）看效果。
- 数据质量 > 数据数量：录慢一点、轨迹干净一点，比多录 20 条有用。
- 评估时一定要看 `--goal-cube all` 的 per-color SR；只测红色会误判。

---

## 9. 命令速查

```bash
# ===== ex2 =====
# 1) baseline
python scripts/eval.py --checkpoint checkpoints/single_cube/best_model_ee_xyz_obstacle.pt --adversarial-obstacle --headless --num-episodes 100
# 2) DAgger
python scripts/dagger_eval.py --checkpoint checkpoints/single_cube/best_model_ee_xyz_obstacle.pt --num-episodes 10 --output-dir datasets/raw_clean/single_cube/dagger/round1
# 3) 合并+重算
python scripts/compute_actions.py --action-space ee --datasets-dir datasets/raw_clean/single_cube
# 4) 重训
python scripts/train.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr --policy obstacle --state-keys state_ee_xyz state_gripper "state_cube[:3]" state_obstacle --action-keys action_ee_xyz action_gripper --chunk-size 4 --d-model 512 --depth 3 --epochs 600 --init-from checkpoints/single_cube/best_model_ee_xyz_obstacle.pt
# 5) 测试
python scripts/eval.py --checkpoint checkpoints/single_cube/best_model_ee_xyz_obstacle_dagger10ep.pt --adversarial-obstacle --headless --num-episodes 100

# ===== ex3 =====
# 1) 录制
python scripts/record_teleop_demos.py --multicube
# 2) 算动作
python scripts/compute_actions.py --action-space ee --datasets-dir datasets/raw/multi_cube
# 3) 训练
python scripts/train.py --zarr datasets/processed/multi_cube/processed_ee_xyz.zarr --policy multitask --state-keys state_ee_xyz state_gripper "original_pos_cube_red[:3]" "original_pos_cube_green[:3]" "original_pos_cube_blue[:3]" state_goal goal_pos --action-keys action_ee_xyz action_gripper --chunk-size 4 --d-model 512 --depth 3 --epochs 600
# 4) 测试
python scripts/eval.py --checkpoint checkpoints/multi_cube/best_model_ee_xyz_multitask.pt --multicube --goal-cube all --headless --num-episodes 100

# ===== 官方（Linux） =====
python student_eval/run_eval.py --exercise 2 --checkpoint ex2.pt
python student_eval/run_eval.py --exercise 3 --checkpoint ex3.pt
```

---

## 10. 电脑卡 / DAgger 太慢怎么办（已实测）

### 10.1 先定位：瓶颈在“渲染”，不在物理/推理

在本机（i7-1355U，无独显）用 `mujoco.Renderer` 实测：

| 项目 | 耗时 |
|---|---|
| `env.step()`（10 Hz 控制，50 个物理子步） | ≈ 0.6 ms |
| 策略推理 `infer_action_chunk`（chunk=4） | ≈ 0.2 ms |
| 渲染 **1 个相机** 640×480 | **≈ 11 ms** |
| 渲染 **3 个相机** 640×480 | **≈ 33 ms** |
| 渲染 1 个相机 320×240 | ≈ 10 ms（**几乎不降**） |

结论：
- **降分辨率没用**（每帧固定开销主要在 GPU→CPU 回读，不在像素数）；
- **减少相机数量最有效**：3 相机 → 1 相机，每帧省 ~22 ms；
- 原版 DAgger 循环在渲染后又固定 `sleep(0.1)`，所以一帧 ≈ 33+100 = **133 ms ≈ 7.5 fps**，输入也会延迟。

### 10.2 现在的代码状态

- **UI 相关脚本已全部恢复原版**：`dagger_eval.py`、`record_teleop_demos.py`、`eval.py`、`teleop_utils.py` 都回到“没加性能参数”的状态（保留了你之前加的 `fit_for_display` 窗口缩放）。
- 因此：没有 `--cameras` / `--render-every` / `--render-w` / `--render-h` 这些参数；`dagger_eval.py` 也没有“中断自动保存”。
- 行为与课程原版一致：3 相机 640×480，渲染后固定 `sleep(0.1)`。

### 10.3 用原版 dagger 时的实用建议

```bash
# 原版命令不变
python scripts/dagger_eval.py \
  --checkpoint checkpoints/single_cube/best_model_ee_xyz_obstacle.pt \
  --num-episodes 5 \
  --max-steps 400 \
  --output-dir datasets/raw_clean/single_cube/dagger/round2
```

- 一次采 5 条，分多次采；机器热了/卡了就休息一下。
- 提前接管：在策略明显要失败之前按 Space，不要等方块掉了再救。
- **不要用 Ctrl+C 或直接关窗口退出正在录制的 episode**：原版不会保存当前 episode；要保存就让它以成功 / 失败 / 到 `--max-steps` 结束。按 Esc 会丢弃当前 episode。
- 原版 UI 是 ~7.5 fps（33ms 渲染 + 100ms 固定 sleep）；接受这个节奏，靠“少量多次 + 提前接管”来采。

### 10.4 系统层面

- **插电源**，电源模式改成“高性能”：`powercfg /setactive SCHEME_MIN`（当前是某个节能方案）；
- 采集时关掉浏览器/视频/其他 Python、IDE debugger（用普通终端跑脚本）；
- 把 OpenCV 窗口保持普通大小，不要最大化；
- 一次只采 5~10 条，分多次采，避免机器过热降频；
- 如果 `episode` 太长（500 步 = 50 秒），可以适当 `--max-steps 400`，但别太短导致动作 chunk 被截断。

### 10.5 你现在的 round1 数据说明

`datasets/raw_clean/single_cube/dagger/round1/...zarr` 目前是 **177 步、0 个 episode**——上次进程被中断（窗口关闭/Ctrl+C），没写入 `episode_ends`，所以 `compute_actions` 会把它当空数据跳过。

处理方式：
- 直接删掉重录（推荐，数据量很小）：
  `Remove-Item -Recurse -Force datasets\raw_clean\single_cube\dagger\round1`
- 或者保留不动（不会影响训练，只是白占空间）。
- 原版 `dagger_eval.py` 在中断时不会自动保存，所以录制时请让 episode 正常结束（成功/失败/到 max_steps）。

---

## 11. Dagger 去静止帧：`hw3_clean_dagger.py`

这是 DAgger 专用的清洗脚本（和 `hw3_clean_data.py` 不同）：

- **不丢弃“失败”episode**（接管片段也要）；
- **不做 bin/success 判定**（没有写死的 `BIN_XY`）；
- 只按“EE / cube / 夹爪是否真的动了”抽帧，保留首尾帧；
- 输出仍是官方 raw schema，可直接喂给 `compute_actions.py`；
- 支持 `--dry-run` 和 `--stage-dir`（自动把 teleop 复制到同一个 staging 根目录）。

### 11.1 推荐流程

```bash
cd ethz-robot-learning-2026/hw3_imitation_learning

# 1) 先 dry-run 看压缩比例
.venv\Scripts\python.exe ..\..\hw3_clean_dagger.py `
  --input datasets\raw_clean_v3\single_cube\dagger --dry-run

# 2) 正式清理 + 搭 staging（cleaned dagger + teleop 放在同一根目录）
.venv\Scripts\python.exe ..\..\hw3_clean_dagger.py `
  --input datasets\raw_clean_v3\single_cube\dagger `
  --stage-dir datasets\raw_clean_static\single_cube `
  --teleop datasets\raw_clean_v3\single_cube\teleop

# 3) 重算动作（显式 --output，避免覆盖原来的 processed_ee_xyz.zarr）
.venv\Scripts\python.exe scripts\compute_actions.py --action-space ee `
  --datasets-dir datasets\raw_clean_static\single_cube `
  --output datasets\processed\single_cube\processed_ee_xyz_staticclean.zarr

# 4) 重新训练（建议先回到 ex1 的 chunk=4 / depth=3 做对照）
.venv\Scripts\python.exe scripts\train.py `
  --zarr datasets\processed\single_cube\processed_ee_xyz_staticclean.zarr `
  --policy obstacle `
  --state-keys state_ee_xyz state_gripper "state_cube[:3]" state_obstacle `
  --action-keys action_ee_xyz action_gripper `
  --chunk-size 4 --d-model 512 --depth 3 --epochs 200 `
  --init-from checkpoints\single_cube\best_model_ee_xyz_obstacle.pt `
  --checkpoint-dir checkpoints\single_cube_staticclean

# 5) 评估
.venv\Scripts\python.exe scripts\eval.py `
  --checkpoint checkpoints\single_cube_staticclean\best_model_ee_xyz_obstacle_dagger40ep.pt `
  --adversarial-obstacle --headless --num-episodes 100 --seed 42
```

### 11.2 在本机数据上的实测（40 条 dagger）

- 40 episodes，**15945 → 6109 steps（38.3% 保留）**；
- 合并 teleop 后 processed 从 **20768 → 10932 transitions**；
- 整体 near-static（`|a|<5e-4`）从 **60.6% → 24.8%**；
- 2 epoch 冒烟训练正常，输出 `_dagger40ep.pt` 命名正常。

### 11.3 调参

| 参数 | 默认 | 说明 |
|---|---|---|
| `--step-mm` | 4.0 | 越小保留越多；6~8 更激进 |
| `--grip-step` | 0.02 | 夹爪命令变化阈值 |
| `--max-gap` | 10 | 最多隔 10 帧强制保留一张，限制时间跳跃；0 = 不强制 |
| `--min-frames` | 8 | 几乎静止的 episode 的兜底帧数 |
| `--overwrite` | 关 | 覆盖已有输出 |
| `--dry-run` | 关 | 只统计不写盘 |

> 抽帧会改变相邻帧的时间间隔：动作仍是 `s_next_kept - s_current`，训练/评估仍按 10Hz 执行，相当于“重定时”。这和 `raw_clean` 的思路一致，但需要重新训练并用 SR 验证，不能直接拿旧 checkpoint 套新数据。

