# HW3 从零实现教程（自己动手版）

> 先明确：**我没有改你的 `hw3/model.py` 和 `scripts/train.py`**，它们仍是原始 TODO 状态（已用 `git status` 确认）。
> 上一份 [HW3_实现函数手册.md](HW3_实现函数手册.md) 是“项目 API 地图”；这一份是**从零动手教程**：你写的每个函数、要用到的每个库函数，都有输入/输出/解释/最小例子，**但不给你拼装好的最终答案**。
> 配套自检：`hw3_selfcheck.py`（每写完一个函数跑一次，会告诉你过了没有、错在哪）。

---

## 0. 你的任务清单（写完一项就跑一次自检）

| # | 文件 / 位置 | 要写的东西 | 自检会检查 |
|---|---|---|---|
| 1 | `hw3/model.py` `ObstaclePolicy.__init__` | 搭 MLP 结构 | 构造成功 + 三个属性 |
| 2 | `hw3/model.py` `ObstaclePolicy.forward` | `(B,D) -> (B,H,A)` | 输出形状 |
| 3 | `hw3/model.py` `ObstaclePolicy.compute_loss` | MSE 标量 | 标量 + 能 backward |
| 4 | `hw3/model.py` `ObstaclePolicy.sample_actions` | 推理入口 | `(1,D) -> (1,H,A)` |
| 5 | `hw3/model.py` `build_policy` | 补 3 个关键字参数 | 签名 + ckpt 重建 |
| 6 | `hw3/model.py` `MultiTaskPolicy` | 同 1–4（ex3 才需要） | 27 维输入形状 |
| 7 | `scripts/train.py` 常量 | `EPOCHS/BATCH_SIZE/LR` | 不再是 `...` |
| 8 | `scripts/train.py` `train_one_epoch` | 一个 batch 的训练步 | 参数真的变了 |
| 9 | `scripts/train.py` `evaluate` | 验证集 loss | 返回 float |
| 10 | `scripts/train.py` `main()` | 4 处修补（见 §5.9） | `--extra-zarr` 存在 |

**不用改**：`hw3/dataset.py`、`hw3/sim_env.py`、`hw3/eval_utils.py`、`hw3/teleop_utils.py`、`scripts/*.py`（`compute_actions.py` 明确写着 do NOT modify）。

---

## 1. 动手前，先吃透 3 个契约（它们决定了每一行的形状）

### 1.1 数据契约（`SO100ChunkDataset[i]`）

```python
state, action_chunk = dataset[i]
# state:        (state_dim,)        float32，已归一化
# action_chunk: (chunk_size, act_dim) float32，已归一化
```

DataLoader 加一个 batch 维后：

```
states:  (B, state_dim)
chunks:  (B, chunk_size, action_dim)
```

本机实测（`processed_ee_xyz.zarr`）：`state_dim=10`、`action_dim=4`、`chunk_size=16` → batch `(B,10)` 和 `(B,16,4)`。

### 1.2 模型契约（`BasePolicy`）

| 方法 | 输入 | 输出 | 谁调用它 |
|---|---|---|---|
| `forward(state)` | `(B, D)` | `(B, H, A)` | 你自己（`compute_loss`/`sample_actions` 里） |
| `compute_loss(state, action_chunk)` | `(B, D)`、`(B, H, A)` | **0 维标量 Tensor** | `train.py` 的训练/验证循环 |
| `sample_actions(state)` | `(B, D)`，评估时 B=1 | `(B, H, A)` | `eval_utils.infer_action_chunk` |

记住：**`H` 和 `A` 直接来自 `dataset`**，不是猜的；`D` 来自你的 `--state-keys` 拼出来的列数。

### 1.3 评估契约（最容易翻车的地方）

`hw3/eval_utils.py` 的 `load_checkpoint` 会这样重建你的模型：

```python
build_policy(policy_type, state_dim=..., action_dim=...,
             chunk_size=..., d_model=128, depth=2)
```

所以 `build_policy` 的签名**必须**含这 5 个关键字参数，且默认 `d_model/depth` 要和 checkpoint 里存的一致。你的 `train.py` 保存 ckpt 时至少要写：

```python
{"state_dim", "action_dim", "chunk_size", "state_keys", "action_keys",
 "model_state_dict", "normalizer": {"state_mean","state_std","action_mean","action_std"}}
```

### 1.4 项目函数速查（直接调用，不用改）

| 函数 | 输入 | 输出 |
|---|---|---|
| `dataset.load_zarr(path, state_keys, action_keys)` | zarr 路径 + 列 spec | `(states, actions, episode_ends)` |
| `dataset.Normalizer.from_data(states, actions)` | 两个 `(N,D)` | `Normalizer` |
| `normalizer.normalize_state(x)` / `normalize_action(a)` / `denormalize_action(a)` | 数组 | 同形数组 |
| `dataset.SO100ChunkDataset(...)` | 见上 | `dataset[i] -> (state, chunk)` |
| `eval_utils.load_checkpoint(path, device)` | `.pt` | `(model, normalizer, H, state_keys, action_keys)` |
| `eval_utils.obs_to_state(obs, state_keys)` | 仿真 obs dict | `(D,)` |
| `eval_utils.infer_action_chunk(...)` | model/norm/obs/keys | `(H, A)` 物理量动作 |
| `sim_env.SO100SimEnv(xml_path=...)` | xml 路径 | env：`reset/step/set_targets/set_gripper/render` |

---

## 2. 函数目录 A：写 `model.py` 会用到的（PyTorch）

> 下面每个例子都是**独立最小演示**，可以直接在 `.venv\Scripts\python.exe` 里跑；它们不构成作业答案。

### 2.1 Python 类与继承

```python
class BasePolicy(nn.Module):          # 父类已定义好 __init__ 的签名
    ...
class ObstaclePolicy(BasePolicy):     # 子类继承并实现抽象方法
    def __init__(self, ...):
        super().__init__(...)         # 先初始化父类（父类会存 state_dim/action_dim/chunk_size）
```

- **输入/输出**：`super().__init__(...)` 无返回；它的作用是把 `nn.Module` 的内部状态准备好。
- **解释**：`nn.Module` 要求“先 `super().__init__()`，再 `self.xxx = nn.Linear(...)`”，否则参数不会被注册。
- **实测反例**：漏掉 `super().__init__()` 就赋值子模块 → `AttributeError: cannot assign module before Module.__init__() call`。
- **在本作业**：`ObstaclePolicy.__init__` 第一行必须是 `super().__init__(state_dim, action_dim, chunk_size)`。

### 2.2 `torch.nn.Module`（基类）

```python
class MLP(nn.Module):
    def __init__(self, d_in, d_h, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(), nn.Linear(d_h, d_out))

    def forward(self, x):
        return self.net(x)

m = MLP(8, 32, 48)
print(m(torch.randn(4, 8)).shape)      # torch.Size([4, 48])  【实测】
print(sum(p.numel() for p in m.parameters()))   # 1872  【实测】(8*32+32 + 32*48+48)
```

- **输入**：`forward(x)` 的 `x` 形状由你定；**输出**：你 `return` 的 Tensor。
- **解释**：`m(x)` 会自动调用 `forward`；`parameters()` 是生成器，给优化器用。
- **注意**：脚手架里 `def forward(self)` **漏了参数**，你要改成 `def forward(self, state)`。

### 2.3 `nn.Linear(in_features, out_features)`

```python
lin = nn.Linear(4, 3)
print(lin(torch.randn(5, 16, 4)).shape)   # torch.Size([5, 16, 3])  【实测】
```

- **输入**：`(..., in_features)`（可以带任意前导维）→ **输出**：`(..., out_features)`。
- **解释**：对**最后一维**做 `y = xWᵀ + b`；权重形状 `(out, in)`、偏置 `(out,)`。
- **在本作业**：第一层 `nn.Linear(state_dim, d_model)`，最后一层 `nn.Linear(d_model, chunk_size * action_dim)`。
- **典型报错**：`RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x6 and 8x48)` —— `state_dim` 传错了。

### 2.4 `nn.Sequential(*layers)`

```python
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
print(net(torch.randn(2, 4)).shape)      # torch.Size([2, 3])  【实测】
print(list(net.state_dict().keys()))     # ['0.weight','0.bias','2.weight','2.bias']  【实测】
```

- **输入**：上一层输出 → **输出**：最后一层输出。
- **解释**：顺序容器，省掉手写 `forward`。
- **坑**：`state_dict` 的键名是**按层下标**命名的（`0`、`2`）；之后你调整层顺序/数量，已保存的 ckpt 就加载不上（`Missing keys` / `Unexpected keys`）。

### 2.5 激活函数：`nn.ReLU()` / `nn.GELU()`（或 `F.relu` / `F.gelu`）

```python
x = torch.tensor([-2., -0.5, 0., 0.5, 2.])
print(F.relu(x))     # tensor([0.0000, 0.0000, 0.0000, 0.5000, 2.0000])   【实测】
print(F.gelu(x))     # tensor([-0.0455, -0.1543, 0.0000, 0.3457, 1.9545]) 【实测】
```

- **输入/输出**：形状不变，逐元素。
- **解释**：放在两个 `Linear` 之间提供非线性；没有它，多层线性叠起来仍等价于一层线性。
- **在本作业**：`H` 层 MLP 里，每层 `Linear` 后面跟一个激活。

### 2.6 `Tensor.view(*shape)` / `reshape`

```python
flat = torch.randn(4, 48)              # 假设 chunk_size=16, action_dim=3
print(flat.view(4, 16, 3).shape)       # torch.Size([4, 16, 3])  【实测】
print(flat.view(16, -1).shape)         # torch.Size([16, 3])     【实测】-1 = 自动算
```

- **输入**：元素总数必须相等；**输出**：新形状视图（共享内存）。
- **解释**：`-1` 表示“这一维你自己算”；`reshape` 在需要时会复制，`view` 不行。
- **典型报错**：`RuntimeError: shape '[4, 16, 4]' is invalid for input of size 192` —— `H*A` 或 `A` 写错了。
- **在本作业**：`forward` 里把 `(B, H*A)` 变成 `(B, H, A)`，就是这一行。

### 2.7 `F.mse_loss(pred, target)`

```python
pred = torch.zeros(2, 16, 3)
tgt  = torch.ones(2, 16, 3)
print(F.mse_loss(pred, tgt))           # tensor(1.)  【实测】
print(type(F.mse_loss(pred, tgt)))     # <class 'torch.Tensor'>，0 维
```

- **输入**：两个**完全相同形状**的张量 → **输出**：0 维标量 Tensor（对所有元素求平均）。
- **解释**：行为克隆的目标函数；`nn.MSELoss()` 完全等价。
- **坑**：形状不一致会先 broadcast warning 再 `RuntimeError`；`(B,H,A)` 和 `(B,H*A)` 别混。

### 2.8 `Tensor.backward()` / `requires_grad`

```python
pred = m(torch.randn(4, 8))
loss = F.mse_loss(pred, torch.randn(4, 48))
loss.backward()
print(m.net[0].weight.grad is not None)   # True
```

- **输入**：0 维 Tensor → **输出**：无；副作用是往每个参数的 `.grad` **累加**梯度。
- **在本作业**：`train_one_epoch` 里调用；`compute_loss` 只负责返回 loss，不要在里面 `backward`。

### 2.9 `with torch.no_grad():` / `@torch.no_grad()`

```python
with torch.no_grad():
    y = m(torch.randn(1, 8))
print(y.requires_grad)      # False  【实测】
```

- **解释**：不建计算图、省内存。推理/验证时用；训练时**不要**用（否则没梯度）。
- **注意**：`model.eval()` 不等于关梯度——实测 `eval()` 下前向的 `requires_grad` 仍是 `True`，两者要配合。

### 2.10 `state_dict()` / `load_state_dict()`、`sum(p.numel() ...)`

```python
sd = m.state_dict()            # 有序字典：'net.0.weight', ...
print(sum(p.numel() for p in m.parameters() if p.requires_grad))   # 可训练参数个数
```

- **输入**：无 → **输出**：`dict[str, Tensor]` / `int`。
- **在本作业**：`train.py` 里 `torch.save({... "model_state_dict": model.state_dict() ...})`；打印参数量。

**model.py 需要的函数清单汇总**：`class` + `super().__init__`、`nn.Module`、`nn.Linear`、`nn.Sequential`、`nn.ReLU/GELU`、`view/reshape`、`F.mse_loss`、`torch.randn`（自测）、`no_grad`（自测）。**就这些**，不需要卷积/Attention。

---

## 3. 函数目录 B：写 `train.py` 会用到的

### 3.1 `DataLoader(dataset, batch_size=, shuffle=, num_workers=0)`

```python
from torch.utils.data import DataLoader, TensorDataset
ds = TensorDataset(torch.randn(40, 8), torch.randn(40, 16, 3))
loader = DataLoader(ds, batch_size=4, shuffle=True)
states, chunks = next(iter(loader))
print(states.shape, chunks.shape, len(loader))   # torch.Size([4, 8]) torch.Size([4, 16, 3]) 10  【实测】
```

- **输入**：Dataset → **输出**：可迭代对象，每次给一个 batch；`len(loader)` = batch 数。
- **解释**：`for batch in loader: states, action_chunks = batch` 就是 `train.py` 里的写法。
- **注意**：Windows 上 `num_workers=0` 最稳（`train.py` 已写 0）。

### 3.2 `random_split(dataset, lengths, generator=)`

```python
from torch.utils.data import random_split
train_ds, val_ds = random_split(ds, [36, 4], generator=torch.Generator().manual_seed(42))
print(len(train_ds), len(val_ds))    # 36 4  【实测】
```

- **输入**：长度列表（和必须等于 `len(dataset)`）→ **输出**：`Subset` 列表。
- **在本作业**：脚手架已经写了，`VAL_SPLIT=0.1`；你只要保证数据集长度算对。
- **注意**：它是**按样本**随机切，同一 episode 的相邻帧可能同时进训练/验证（验证 loss 会偏乐观，作业不追究）。

### 3.3 优化器三件套：`zero_grad` → `backward` → `step`

```python
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

for states, action_chunks in loader:
    loss = model.compute_loss(states, action_chunks)
    opt.zero_grad(set_to_none=True)   # 1) 清掉上一轮梯度
    loss.backward()                   # 2) 反向传播，梯度累加进 p.grad
    opt.step()                        # 3) 用梯度更新参数
```

- **`zero_grad(set_to_none=True)`**：无输入/输出；**位置必须在 `backward()` 之前**。
  - 实测不清零连做两次 backward：`grad` 从 `-0.216` 变成 `-0.432`（**梯度是累加的**）。
- **`step()`**：无输入/输出；**必须在 `backward()` 之后**。
  - 实测没 backward 就 step：`grad is None`，参数**静默不变**、不报错（loss 不降时先查这里）。
- **在本作业**：这段就是 `train_one_epoch` 循环体的核心 3 行。

### 3.4 `torch.optim.AdamW(params, lr=, weight_decay=)`

```python
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
print(opt.param_groups[0]["lr"])   # 0.001  【实测】
```

- **输入**：`model.parameters()`（可迭代）+ 超参 → **输出**：优化器对象。
- **解释**：AdamW = Adam + 解耦权重衰减；这个任务上比 SGD 好调。
- **坑**：别把 `model.parameters()` 写成 `model` 或 `model.state_dict()`。

### 3.5 `CosineAnnealingLR(optimizer, T_max=)`

```python
opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
lrs = []
for _ in range(6):
    lrs.append(round(opt.param_groups[0]["lr"], 6))
    opt.step()
    sched.step()
print(lrs)   # [0.01, 0.009045, 0.006545, 0.003455, 0.000955, 0.0]  【实测】
```

- **输入**：优化器 + `T_max`（= 你的 EPOCHS）→ **输出**：scheduler。
- **解释**：按余弦曲线把 lr 从初值降到 `eta_min`；`train.py` 里在**每个 epoch 末尾**调 `scheduler.step()`。
- **坑**：先 `scheduler.step()` 再 `optimizer.step()` 会警告 `Detected call of lr_scheduler.step() before optimizer.step()`，并跳过第一个 lr 值。

### 3.6 `clip_grad_norm_(params, max_norm)`

```python
raw = torch.nn.utils.clip_grad_norm_(m.parameters(), 0.5)
print(float(raw))    # 返回裁剪【前】的范数；裁剪后总范数恰为 0.5  【实测】
```

- **输入**：参数迭代器 + 阈值 → **输出**：裁剪前梯度范数（Tensor，打印用 `float()`）。
- **解释**：防止梯度爆炸；训练循环里 `backward()` 之后、`step()` 之前调用。
- **注意**：可选，不加也能训；但加了更稳。

### 3.7 `loss.item()` 与累计平均

```python
total += loss.item()      # Python float
return total / max(n_batches, 1)
```

- **`loss.item()`**：Tensor → `float`，与计算图脱钩。
- **反例**：`total_loss += loss`（少了 `.item()`）→ 得到一个 `requires_grad=True` 的 Tensor，整个循环的计算图都被留住（显存飙升），而且 `f"{loss:.6f}"` 不报错，非常隐蔽。
- **在本作业**：`train_one_epoch`/`evaluate` 都返回**平均** loss，单位是归一化空间的 MSE（数量级 1e-3～1e-1）。

### 3.8 `model.train()` / `model.eval()`

```python
model.train()   # 训练模式；model.training == True
model.eval()    # 评估模式；model.training == False
```

- **在本作业**：`train_one_epoch` 开头 `model.train()`；`evaluate` 开头 `model.eval()`。
- **注意**：纯 MLP 没有 Dropout/BN 时影响不大，但这是标准写法；而且 `eval()` **不关梯度**。

### 3.9 `torch.save` / `torch.load`

```python
torch.save({"epoch": 1, "model_state_dict": m.state_dict()}, "ckpt.pt")
ckpt = torch.load("ckpt.pt", map_location="cpu", weights_only=False)
print(ckpt.keys())      # dict_keys(['epoch', 'model_state_dict'])  【实测】
```

- **坑（实测）**：`torch.load` 默认 `weights_only=True`，dict 里有 numpy 数组（normalizer 的 mean/std）会报 `UnpicklingError: Weights only load failed`。**必须 `weights_only=False`**（`eval_utils.load_checkpoint` 已经这么写了，你照抄即可）。

### 3.10 `Tensor.to(device)` / `Module.to(device)`

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
states = states.to(device)     # Tensor 版：返回新 Tensor（CPU→CPU 实测返回自身）
model = model.to(device)       # Module 版：原地改参数，返回 self
```

- **本机实测**：`torch.cuda.is_available()` → `False`，`device` 就是 `cpu`。写 `.to(device)` 是习惯，将来换 GPU 不用改代码。
- **在本作业**：`train_one_epoch`/`evaluate` 里把 batch 搬过去；`build_policy(...).to(device)`。

### 3.11 `argparse` + `Path`（给 `main()` 补参数用）

```python
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--zarr", type=Path, required=True)
parser.add_argument("--chunk-size", type=int, default=16)
parser.add_argument("--extra-zarr", type=Path, nargs="*", default=[])
args = parser.parse_args(["--zarr", "datasets/x.zarr"])
print(args.zarr, args.chunk_size, args.extra_zarr)   # datasets\x.zarr 16 []  【实测】
```

- **解释**：`--chunk-size` 自动变成 `args.chunk_size`（连字符 → 下划线）；`nargs="*"` 收集多个路径。
- **在本作业**：补 `--extra-zarr`（第 115 行会读它）；`Path` 用来拼 checkpoint 路径：

```python
Path("./checkpoints/single_cube") / "best_model_ee_xyz_obstacle.pt"
# checkpoints\single_cube\best_model_ee_xyz_obstacle.pt  【实测】
```

### 3.12 f-string 打印

```python
print(f"Epoch {3:3d}/{100} | train {0.0123456:.6f} | val {0.0234:.6f}")
# Epoch   3/100 | train 0.012346 | val 0.023400  【实测】
print(f"{1234567:,}")      # 1,234,567
```

- **在本作业**：`train.py` 里已经写好了打印格式，你把自己的 loss 变量喂进去即可。

**train.py 需要的函数清单汇总**：`DataLoader`、`random_split`、`AdamW`、`zero_grad/backward/step`、`CosineAnnealingLR`、`clip_grad_norm_`、`loss.item()`、`train()/eval()`、`to(device)`、`torch.save`、`argparse`、`Path`、f-string。

---

## 4. 逐 TODO 拆解（目标 → 用什么 → 输出要求 → 思路提示 → 自检）

> 这里给的是**思考路径和验收标准**，不是最终代码。你写完后跑 `hw3_selfcheck.py` 就知道对不对。

### TODO 1：`ObstaclePolicy.__init__`

- **目标**：构造一个 MLP：`state_dim → ... → chunk_size * action_dim`。
- **用到**：`super().__init__(state_dim, action_dim, chunk_size)`、`nn.Sequential`、`nn.Linear`、`nn.ReLU/GELU`。
- **输出要求**：`self.net(x)` 能吃 `(B, state_dim)`、吐 `(B, chunk_size*action_dim)`。
- **思路提示**：
  1. 先 `super().__init__(...)`（否则后面赋值报错）；
  2. 存 `self.state_dim/action_dim/chunk_size` 由父类完成，你不用再写；
  3. 层结构：第一层 `Linear(state_dim, d_model)` → 激活 →（可重复 `depth` 次 `Linear(d_model, d_model)` + 激活）→ 最后一层 `Linear(d_model, chunk_size*action_dim)`；
  4. 最后一层**不要**接激活（回归任务，要线性输出）。
- **自检**：`hw3_selfcheck.py` 的 “build_policy('obstacle', ...) 能构造出模型”。

### TODO 2：`forward(self, state)`

- **目标**：`(B, D) -> (B, H, A)`。
- **用到**：`self.net(state)`、`view(-1, self.chunk_size, self.action_dim)`。
- **输出要求**：形状严格 `(B, H, A)`；batch 维用 `-1` 自动算。
- **思路提示**：`y = self.net(state)` 得到 `(B, H*A)`，再 `y.view(-1, self.chunk_size, self.action_dim)`。
- **顺手检查**：脚手架里写的是 `def forward(self)`，记得加参数 `state`。
- **自检**：“forward: (8, 10) -> (8, 16, 4)”。

### TODO 3：`compute_loss(self, state, action_chunk)`

- **目标**：返回标量 MSE。
- **用到**：`F.mse_loss(self.forward(state), action_chunk)`。
- **输出要求**：0 维 Tensor，且 `loss.backward()` 后参数有梯度。
- **思路提示**：不要 `detach()`、不要 `.item()`、不要 `torch.no_grad()`；预测和标签都是**归一化空间**的值，直接比。
- **自检**：“compute_loss 返回标量且能 backward（梯度已产生）”。

### TODO 4：`sample_actions(self, state)`

- **目标**：推理入口。
- **用到**：`self.forward(state)`（或 `self(state)`）。
- **输出要求**：`(B, H, A)`；`eval_utils` 会 `squeeze(0)` 后逐行反归一化。
- **思路提示**：与 `forward` 等价即可；不需要 `no_grad`（调用方 `infer_action_chunk` 已经包了）。
- **自检**：“sample_actions: (1, D) -> (1, H, A)”。

### TODO 5：`build_policy`

- **目标**：让 `eval_utils` 能按 ckpt 重建模型。
- **用到**：`inspect` 不用写，直接改签名：

```python
def build_policy(policy_type, *, state_dim, action_dim,
                 chunk_size=16, d_model=128, depth=2) -> BasePolicy:
```

- **输出要求**：`obstacle` / `multitask` 两个分支各自把 5 个参数传进构造函数；未知类型 `raise ValueError`。
- **思路提示**：`train.py` 调用时也要补 `chunk_size=args.chunk_size`（默认值兜底也行，但显式更清楚）。
- **自检**：“build_policy 接受 …” + “checkpoint 能被 eval_utils.load_checkpoint 重建并推理”。

### TODO 6：`MultiTaskPolicy`（ex3）

- **目标**：同 ObstaclePolicy，但输入里含目标信息（3 个 cube 位置、one-hot 目标色、bin 位置）。
- **输出要求**：`(B, H, A)`，`state_dim` 由 `--state-keys` 决定（例如 27）。
- **思路提示**：先把 ex1 跑通再来；最简单的做法是复用同一个 MLP 结构，只改输入维度；想要更高分再考虑 goal 编码/注意力。
- **自检**：“MultiTaskPolicy（ex3）: 27 维状态 -> (B, 8, A)”（**可选**，不影响 ex1/ex2 的通过）。

### TODO 7：`train.py` 的三个常量

```python
EPOCHS = ...        # 建议 100 ~ 300
BATCH_SIZE = ...    # CPU 上 128 ~ 512
LR = ...            # 1e-3 起步
```

- **自检**：“超参常量已填”。

### TODO 8：`train_one_epoch`

- **目标**：遍历一个 epoch，更新参数，返回平均 loss。
- **用到**：`model.train()`、`states.to(device)`、`model.compute_loss`、`optimizer.zero_grad`、`loss.backward`、`optimizer.step`、`loss.item()`。
- **输出要求**：`float`（平均 loss），参数**确实变了**。
- **每行该做什么**（按这个顺序填，别跳）：

| 顺序 | 做什么 | 为什么 |
|---|---|---|
| 1 | `model.train()` | 切训练模式 |
| 2 | 循环 `for states, action_chunks in loader:` | 拿 batch |
| 3 | 两个张量 `.to(device)` | 设备一致 |
| 4 | `loss = model.compute_loss(states, action_chunks)` | 前向 + 损失 |
| 5 | `optimizer.zero_grad(set_to_none=True)` | 清上一轮梯度（**必须在 backward 前**） |
| 6 | `loss.backward()` | 反传 |
| 7 | `optimizer.step()` | 更新参数 |
| 8 | `total += loss.item()`；`n += 1` | 记录标量 |
| 9 | `return total / max(n, 1)` | 平均 |

- **可选**：`clip_grad_norm_` 放在 6 和 7 之间。
- **自检**：“train_one_epoch / evaluate 能跑通（toy 模型，测你的循环）”——它用假模型测你的循环，参数没变就会报错。

### TODO 9：`evaluate`

- **目标**：验证集平均 loss。
- **用到**：`@torch.no_grad()`、`model.eval()`、`compute_loss`、`loss.item()`。
- **输出要求**：`float`。
- **思路提示**：和训练循环一样，但**去掉** `zero_grad/backward/step`，加上 `no_grad`。
- **自检**：同上一条自检项（会同时调用 `evaluate`）。

### TODO 10：`main()` 的 4 处修补

| # | 位置 | 现状 | 你要改成 | 不改的后果 |
|---|---|---|---|---|
| 1 | 第 115 行 `if args.extra_zarr:` | parser 里没有这个参数 | 加 `parser.add_argument("--extra-zarr", type=Path, nargs="*", default=[])` | `AttributeError: 'Namespace' object has no attribute 'extra_zarr'` |
| 2 | `build_policy(...)` 调用 | 只传了 2 个参数 | 补 `chunk_size=args.chunk_size`（以及你的 `d_model/depth`） | 一旦 `build_policy` 加上必填关键字就会 `TypeError` |
| 3 | 第 168-170 行 | `optimizer =` / `scheduler =` 是注释 | `AdamW` + `CosineAnnealingLR(T_max=EPOCHS)` | 训练循环里 `optimizer.step()` 报 `NameError` |
| 4 | `torch.save({...})` | 没存 `d_model/depth` | 加上（与 `build_policy` 一致） | 若默认值不是 128/2，评估重建网络时形状不匹配 |

- **自检**：“main() 已定义 --extra-zarr 参数”。

---

## 5. 自检脚本怎么用

```bat
cd ethz-robot-learning-2026\hw3_imitation_learning
.venv\Scripts\python.exe ..\..\hw3_selfcheck.py
```

（或从工作区根目录运行，用同一个解释器）

```bat
ethz-robot-learning-2026\hw3_imitation_learning\.venv\Scripts\python.exe hw3_selfcheck.py
```

输出符号含义：

| 符号 | 含义 | 你该做什么 |
|---|---|---|
| `✓` | 这一项通过 | 继续下一项 |
| `○` | 还没实现（脚手架原状） | 正常，去写那一项 |
| `✗` | 有 bug | 看下面的“提示”一行 |

**现在的预期**（你什么都没写时，实测就是这样）：

```
通过 6 项 | 待实现 6 项 | 失败 0 项
```

6 项 ✓ 是数据管线和仿真环境（这些不用你写）；6 项 ○ 正好对应你的 TODO 清单——**○ 就是进度条**。

**每完成一项后应该新增哪些 ✓**：

| 你写完 | 自检应通过 |
|---|---|
| `__init__` | `build_policy('obstacle', ...) 能构造出模型` |
| `forward` | `forward: (8, 10) -> (8, 16, 4)` |
| `compute_loss` | `compute_loss 返回标量且能 backward` |
| `sample_actions` | `sample_actions: (1, D) -> (1, H, A)` |
| `build_policy` + ckpt 字段 | `checkpoint 能被 eval_utils.load_checkpoint 重建并推理` |
| 三个常量 | `超参常量已填` |
| `--extra-zarr` | `main() 已定义 --extra-zarr 参数` |
| `train_one_epoch` + `evaluate` | `train_one_epoch / evaluate 能跑通` |
| 全部 | `通过 16 项 | 待实现 0 项 | 失败 0 项`（加 `--smoke` 是 17 项） |

最后再跑一次加强版（用真实数据训练 3 个 epoch）：

```bat
.venv\Scripts\python.exe hw3_selfcheck.py --smoke
```

实测参考（用一个 2 层小 MLP，仅证明管线通）：`val_loss 0.883 → 0.767 → 0.682`。你的数字不必一样，但应当**下降**；不降就按 §6 排查。

---

## 6. 常见报错 → 原因 → 修法

| 报错 / 现象 | 原因 | 修法 |
|---|---|---|
| `TypeError: build_policy() got an unexpected keyword argument 'chunk_size'` | `build_policy` 没加 3 个关键字参数 | 改签名（§4 TODO 5） |
| `TypeError: ObstaclePolicy.__init__() missing 1 required positional argument: 'chunk_size'` | 子类没定义 `__init__` 或没调 `super().__init__` | 加 `__init__` + `super().__init__(state_dim, action_dim, chunk_size)` |
| `AttributeError: cannot assign module before Module.__init__() call` | 先赋值 `self.xxx` 后调 `super().__init__()` | `super().__init__()` 放第一行 |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | 第一层 `in_features` ≠ `state_dim` | 检查 `--state-keys` 的总列数 |
| `RuntimeError: shape '[B, H, A]' is invalid for input of size ...` | `H/A` 与最后一层输出维不匹配 | 最后一层必须是 `chunk_size*action_dim` |
| `RuntimeError: The size of tensor a (3) must match tensor b (48)` | `compute_loss` 里预测没 reshape 就比 | 先 `view(-1, H, A)` 再 MSE |
| loss 完全不降、参数没变 | `optimizer.step()` 没调、或没 `backward`、或 `zero_grad` 位置错 | 按 §3.3 的顺序检查 |
| `NameError: name 'optimizer' is not defined` | `train.py` 第 168 行仍是注释 | §4 TODO 10 第 3 条 |
| `AttributeError: 'Namespace' object has no attribute 'extra_zarr'` | 没定义 `--extra-zarr` | §4 TODO 10 第 1 条 |
| `UnpicklingError: Weights only load failed` | `torch.load` 默认 `weights_only=True`，ckpt 里有 numpy | `torch.load(..., weights_only=False)` |
| 加载 ckpt 报 `Missing keys` / `Unexpected keys` | 你改了层结构/顺序但 ckpt 是旧结构 | 重新训练；或保持结构一致 |
| `Detected call of lr_scheduler.step() before optimizer.step()` | scheduler 顺序错 | epoch 末尾再 `scheduler.step()` |
| loss 变成 `nan` | lr 太大 / 没归一化 / 除零 | 降 lr 到 1e-3、确认 dataset 传了 normalizer |

---

## 7. 建议节奏（每天 1–2 小时，每步都有可验证结果）

**Day 1（约 60–90 分钟）——把形状搞对**
1. 读 `hw3/dataset.py` 的 `__getitem__`（5 分钟）；
2. 跑一次未实现的 `hw3_selfcheck.py`，认识 ○/✗（5 分钟）；
3. 写 `__init__` + `forward` → 自检出现 2 个 ✓（30 分钟）；
4. 写 `compute_loss` + `sample_actions` → 再 2 个 ✓（20 分钟）。

**Day 2（约 60 分钟）——把训练跑通**
5. 写 `build_policy` → 自检出现 ckpt 重建 ✓（20 分钟）；
6. 填 3 个常量 + `train_one_epoch` + `evaluate`（30 分钟）；
7. `--extra-zarr` + optimizer/scheduler（10 分钟）；
8. 自检 17 项全过 → 跑 `--smoke` 看 loss 下降。

**Day 3（约 60 分钟）——真训练 + 评估**
9. `python scripts\train.py ...`（第一次先 50 epoch）；
10. `python scripts\eval.py --checkpoint ... --headless --num-episodes 20`；
11. 记下 SR，决定要不要加数据/加宽网络。

> 学习重点不是“填对 3 行代码”，而是：**形状对了吗 → 梯度流了吗 → loss 降了吗 → 评估 SR 是多少**。这四问能定位 90% 的问题。

---

## 8. 训练 / 评估命令（复制即用）

```bat
cd ethz-robot-learning-2026\hw3_imitation_learning
set PY=.venv\Scripts\python.exe

:: 训练（状态取 ee 位置 + 夹爪 + 方块位置 + 障碍物位置）
%PY% scripts\train.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr ^
  --state-keys state_ee_xyz state_gripper "state_cube[:3]" state_obstacle ^
  --action-keys action_ee_xyz action_gripper --chunk-size 16 --policy obstacle

:: 评估
%PY% scripts\eval.py --checkpoint checkpoints\single_cube\best_model_ee_xyz_obstacle.pt ^
  --headless --num-episodes 20

:: 自检（写完代码随时跑）
%PY% ..\..\hw3_selfcheck.py
%PY% ..\..\hw3_selfcheck.py --smoke
```

---

## 9. 本机实测：训练/评估要多久

机器：**Intel i7-1355U**（10 核 12 线程，笔记本 U 系列），`torch 2.15.0+cpu`、`torch.cuda.is_available() = False`（纯 CPU）。
数据：本机现有 `processed_ee_xyz.zarr` = 24 条 episode / 4022 个训练样本，`state_dim=10, action_dim=4, chunk=16`，batch=256 → **每个 epoch 只有 16 个 batch**。

### 9.1 训练（含 forward + backward + step）

| 模型 | 参数量 | 每 epoch | 100 epochs | 200 epochs | 300 epochs |
|---|---|---|---|---|---|
| `256×3` | 150,848 | **0.19 s** | ~19 s | ~40 s | ~60 s |
| `512×3` | 563,776 | **0.30 s** | ~30 s | ~60 s | ~90 s |
| `1024×2` | 1,126,464 | **0.41 s** | ~41 s | ~82 s | ~124 s |

（验证集前向另加 ~0.08 s/epoch，几乎可忽略；实测 batch 128/256/512、线程 4/6/10 差别都在 0.1 s 级。）

**结论：训练不是瓶颈。** 一个 15 万参数的小 MLP 跑 300 epoch 不到 1.5 分钟；按 README 说的“<1M 参数”上限，也就 2 分钟左右。所以你可以放心多试几组配置/随机种子。

### 9.2 评估

| 方式 | 单步成本 | 单条 episode（最多 800 步） | 100 条 |
|---|---|---|---|
| `--headless` | ~1.1 ms（`env.step()` 内含 50 个物理子步） | ~1 s | **~2 分钟** |
| 带窗口渲染 | ~58 ms 渲染 + `sleep(0.1 s)` = 0.16 s | ~2 分钟 | 不现实（3 小时+） |

- 结论：**跑 100 条看成功率一定加 `--headless`**；带窗口只用来肉眼检查 2–3 条。
- 带窗口时 `eval.py` 每步都有 `time.sleep(env.dt_ctrl)`，这是“实机速度”的模拟，不是算力问题。

### 9.3 端到端时间预算（ex1，从零开始）

| 环节 | 谁在做 | 大致耗时 |
|---|---|---|
| 配键 | 人 | 5 分钟（只需一次） |
| 遥操作录 15–20 条 | 人 | **15–30 分钟**（10 Hz，一条约 20–40 s + 重录/重置） |
| `compute_actions.py` | 机器 | 秒级 |
| 训练 200 epoch | 机器 | **40 秒 – 1 分钟** |
| headless 评估 20 条 | 机器 | ~20 秒 |
| 反复调参（5 轮） | 机器 | 3–5 分钟 |

**真正花时间的是人录数据**，不是训练。ex2 的 DAgger 更是人类在环（10–25 条 × 1–2 分钟），机器部分依旧很快。

> 想自己复测：跑完 `train.py` 后看每行 `Epoch` 的时间戳间隔即可；或复用 `.tmp/bench_hw3.py`（不修改作业文件）。

---

## 10. 下一步（ex1 跑通后）

- **ex2（DAgger）**：`scripts/dagger_eval.py` 接管录数据 → `compute_actions.py` 重算 → `train.py --extra-zarr` 合并重训 → `eval.py --adversarial-obstacle`。
- **ex3（多 cube）**：`record_teleop_demos.py --multicube` + `MultiTaskPolicy` + 目标条件状态。
- 提交前：`student_eval/run_eval.py` 只能在 WSL/Linux 跑（Windows 缺 `.pyd`）。
