# 搬到 Windows 显卡机：完整操作手册

> 2026-11-16 · 目标机：**Windows + NVIDIA** · 已推到私有仓库
> 👉 <https://github.com/kongxinshitou/ethz-robot-learning-2026>（默认分支 `my-work`）

---

## 0. 结论：一条 `git clone` 就够，`venv` 绝对不能搬

| 内容 | 怎么搬 | 状态 |
|---|---|---|
| 代码 + 你的 18 处改动 + 自检脚本 | **GitHub 私有仓库** | ✅ 已推（25 MB） |
| hw3 数据集 74 MB / 65136 文件 | 同上（一次性搬，之后别再改它） | ✅ 已推 |
| hw3 训练权重 61.8 MB（6 个 `.pt`） | 同上 | ✅ 已推 |
| hw2 PPO checkpoint 2 MB | 同上 | ✅ 已推 |
| 课件 `slides/` 35 MB | 同上 | ✅ 已推 |
| **4 个 `.venv`（3.29 GB）** | **不搬，在目标机上重建** | ❌ 别传 |
| `.uv-cache`（1.94 GB） | 不搬，自动重建 | ❌ 别传 |
| `证件照_祝鸿志.*` | 不搬（隐私） | ❌ 别传 |

**为什么 `.venv` 连 Windows→Windows 都不能直接拷**：里面是 CPU 版的 `torch==2.15.0+cpu`，
拷过去也能 import，但 `torch.cuda.is_available()` 永远 `False`，等于白搬 3.3 GB。
必须用 `--torch-backend` 重新装 CUDA 版。这是这次迁移唯一"必须重装"的东西。

仓库现在 **≈212 MB**（其中数据 187 MB），clone 下来就全了，**不需要再单独传 zip**。

---

## 1. 目标机上只要 3 步

```powershell
# 0) 装 uv（如果目标机没有；它还会自动下载 Python 3.12）
winget install --id=astral-sh.uv -e

# 1) 克隆到"短路径"（重要：MuJoCo/数据路径很深，D:\work 这种能避开 260 字符限制）
git clone https://github.com/kongxinshitou/ethz-robot-learning-2026.git D:\work\ethz-robot-learning-2026
cd D:\work\ethz-robot-learning-2026

# 2) 一键重建 4 个 venv（10~25 分钟，最后自动验收）
powershell -ExecutionPolicy Bypass -File .\_gpu_tools\setup_gpu_env.ps1
```

想先看脚本要执行什么（不真装）：

```powershell
powershell -ExecutionPolicy Bypass -File .\_gpu_tools\setup_gpu_env.ps1 -DryRun
```

CUDA 版本想锁死（默认 `auto`，由 uv 按显卡驱动自动挑）：

```powershell
... -TorchBackend cu124      # 或 cu121 / cu126 / cpu（没显卡时）
```

---

## 2. 仓库里有什么 / 没有什么

```
ethz-robot-learning-2026/            ← git clone 得到的
├─ hw1_pytorch_tutorial/ … hw4_reinforcement_learning/   代码（含你的实现 + .gitignore）
│   ├─ hw3_imitation_learning/datasets/     ✅ 已提交（74 MB，65136 文件）
│   ├─ hw3_imitation_learning/checkpoints/  ✅ 已提交（61.8 MB）
│   └─ hw2_robot_control_mdps/logs/          ✅ 已提交（PPO checkpoint）
├─ slides/                                  ✅ 已提交（35 MB 课件）
└─ _gpu_tools/                              ← 迁移工具（也在仓库里，所以 clone 完就有）
    ├─ setup_gpu_env.ps1                    Windows 一键重建（主用）
    ├─ setup_gpu_env.sh                     Linux/WSL 版（备用）
    ├─ GPU迁移_操作手册.md                   本文件
    ├─ MIGRATE_TO_GPU.md                    文件来源盘点
    └─ env-lock/                            各环境精确版本快照
```

**没有的东西**（故意的）：`.venv/`、`.uv-cache/`、`证件照_*`、`hw3_reference/`（14 MB 演示视频，
要的话从旧机器单独拷）、`.vscode/`、`__pycache__/`。

**根目录的笔记工具**（`hw3_*.py`、`HW*.md`、`pid-teach/`、`env-lock/`）目前还在旧机器的工作区根，
没有进仓库。要一起带走，在旧机器上执行：

```powershell
cd "D:\Users\00796533\Documents\deepseek-harness\default-workspace"
$dst = "ethz-robot-learning-2026\_my_notes"; New-Item -ItemType Directory $dst -Force | Out-Null
Copy-Item *.md,*.py,*.cmd,*.ps1,*.sh -Destination $dst -Force
Copy-Item env-lock,pid-teach -Destination $dst -Recurse -Force
cd ethz-robot-learning-2026; git add -A; git commit -m "笔记与工具"; git push
```

---

## 3. 重建脚本具体做什么（这是 `-DryRun` 的真实输出）

```text
[1/4] hw1: uv venv --python 3.12 .venv → torch torchvision --torch-backend auto → jupyterlab ipykernel
[2/4] hw2: → -r requirements.txt → pip install -e . --torch-backend auto
[3/4] hw3: → torch torchvision --torch-backend auto → -e . → matplotlib
[4/4] hw4: → torch --torch-backend auto → -r requirements.txt
随后：4 个 torch.cuda.is_available() 冒烟 + check_ex1/2/3 + hw4 两条 smoke + 数据是否到位
```

要点：
- **Windows 的 PyPI torch 是 CPU-only**，所以每条都带 `--torch-backend`，uv 会去 PyTorch 官方 CUDA 索引取带 runtime 的 wheel。
- **4 个独立 venv**：hw2 和 hw3 都提供同名 `so101_gym` 包，合并会互相覆盖。
- hw3 额外补 `matplotlib`（旧 `env-lock` 里没有，根目录的 `hw3_*.py` 画图脚本要用）。
- `pip install -e .` 万一失败，脚本会自动写 `_hw_local.pth` 兜底。

---

## 4. 验收（脚本最后自动跑，也可以手动）

```powershell
cd D:\work\ethz-robot-learning-2026
foreach ($hw in 'hw1_pytorch_tutorial','hw2_robot_control_mdps','hw3_imitation_learning','hw4_reinforcement_learning') {
  "== $hw"
  & "$hw\.venv\Scripts\python.exe" -c "import torch;print(torch.__version__, 'cuda=', torch.cuda.is_available())"
}
```

**四行必须都是 `cuda= True`**（`True` = 显卡真的用上了）。是 `False` 就先修这个，别急着跑作业。

```powershell
cd hw2_robot_control_mdps
.venv\Scripts\python.exe check_ex1.py
.venv\Scripts\python.exe check_ex2.py
.venv\Scripts\python.exe check_ex3.py

cd ..\hw4_reinforcement_learning
.venv\Scripts\python.exe -c "from envs.grid_world import CliffWalkingEnv; from envs.cartpole_wrapper import CartPoleWrapper; g=CliffWalkingEnv(); c=CartPoleWrapper(seed=0); s=c.reset(); c.step(c.sample_action()); print('A OK', g.n_states)"
.venv\Scripts\python.exe -c "from pathlib import Path; import numpy as np; from envs.so100_rl_env import SO100RLEnv; e=SO100RLEnv(xml_path=Path('assets/mujoco/so100_pos_ctrl.xml').resolve(), render_mode=None); o,_=e.reset(seed=0); e.step(np.zeros(e.action_dim, dtype=np.float32)); print('B OK', o.shape)"
```

数据到位确认：

```powershell
Get-ChildItem hw3_imitation_learning\datasets -Recurse -File | Measure-Object Length -Sum
Get-ChildItem hw3_imitation_learning\checkpoints -File | Select-Object Name, Length
```

---

## 5. Windows 目标机专属的坑（按概率排序）

| # | 坑 | 现象 | 处理 |
|---|---|---|---|
| 1 | **torch 是 CPU 版** | `cuda=False`，训练慢十倍 | 重跑脚本（带 `--torch-backend`），或手动 `uv pip install torch torchvision --torch-backend auto --python .venv\Scripts\python.exe` |
| 2 | **hw3 本地评分器在 Windows 用不了** | `student_eval\run_eval.py` 报 import 错 | `student_eval\` 里只有 `...-linux-gnu.so` 和 `...-darwin.so`，**没有 Windows 版**。用 `scripts\eval.py`，或 WSL2 里跑，最终分数以 Gradescope 为准 |
| 3 | **路径太长（260 字符）** | `git clone` / 解压报 "Filename too long" | 克隆到 `D:\work\` 这种短路径；并 `git config --global core.longpaths true` |
| 4 | **复制了旧 `.venv`** | `ModuleNotFoundError` 或 cuda=False | 删掉 `hw*.venv` 重新跑脚本 |
| 5 | **hw2、hw3 共用一个环境** | `import so101_gym` 行为诡异 | 必须 4 个独立 venv |
| 6 | **MuJoCo 窗口黑屏** | `scripts\interactive.py` 黑窗 | 有桌面时通常是显卡驱动太旧，更新 NVIDIA 驱动；远程桌面连接时换成 VcXsrv 或直接本地登录 |
| 7 | **`setup_gpu_env.ps1` 被执行策略拦** | "禁止运行脚本" | 用 `powershell -ExecutionPolicy Bypass -File ...`（命令里已带），或 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| 8 | **uv 没装** | `uv : 无法将"uv"项识别为...` | `winget install --id=astral-sh.uv -e`，然后**重开一个终端** |
| 9 | **hw2 的 PPO 用不上 GPU** | 训练还是吃 CPU | 正常：hw2 是 16 个并行 CPU 环境。GPU 受益的是 hw3 行为克隆、hw4 的 PPO/SAC/DQN |
| 10 | **磁盘不够** | — | 预留 **≥8 GB**（4 个 venv 3.4 GB + pip 缓存） |

---

## 6. 备用方案（没网/U 盘场景）

**① 离线 zip 备份**：旧机器上已生成 `export\gpu_data_2026-11-16.zip`（210 MB，数据集+权重+课件+`_gpu_tools`）。
目标机解压到仓库根即可归位（和 clone 的内容重复，纯备份用）：

```powershell
powershell -ExecutionPolicy Bypass -File .\pack_data_for_gpu.ps1 -DryRun   # 看清单
powershell -ExecutionPolicy Bypass -File .\pack_data_for_gpu.ps1           # 重新打
```

**② 局域网直传**（比走 GitHub 快）：同网段直接 rsync 整个目录，跳过 venv。

---

## 7. 以后在目标机上怎么继续干活

```powershell
cd D:\work\ethz-robot-learning-2026

# 改完作业后提交（.gitignore 会挡住 .venv / __pycache__ / 证件照）
git add -A
git status --short
git commit -m "hw4 ex2 DQN"
git push

# 拉课程上游的更新（main 分支保持和上游一致，别在上面写作业）
git fetch origin
git rebase origin/main        # 在 my-work 分支上操作
```

两条纪律：
1. **不要再 `git add` 数据集/权重**。它们已经以"一次性快照"的方式提交了；之后每次改动都会让仓库膨胀。
   新数据集放 `hw3_imitation_learning/datasets/` 以外的位置，或加进 `.gitignore`。
2. **`main` 分支别动**，你的工作全在 `my-work`。

---

## 8. 一句话总结

> 目标机 Windows：`winget install uv` → `git clone` → 跑 `_gpu_tools\setup_gpu_env.ps1` → 看四行 `cuda= True`。
> 唯一必须"重做"的是 3.29 GB 的虚拟环境；代码 + 数据 + 权重 212 MB 已在私有仓库里，一条命令拿全。
