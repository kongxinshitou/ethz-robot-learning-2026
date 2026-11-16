# 迁移到 GPU 机器：代码来源清单 + 打包/重建指南

> **⚠️ 2026-11-16 更新：本文是"来源盘点"，具体的迁移操作请以 [GPU迁移_操作手册.md](GPU迁移_操作手册.md) 为准。**
> 自本文写成后有这些变化，下文第二、三节的体积数字已过期：
> - `hw3/datasets/` 从 57 MB / 19599 文件 → **74 MB / 65136 文件**
> - 新增 `hw3/checkpoints/` **61.8 MB**（6 个 `.pt`，最好的 21 MB）
> - `hw4` 的 `ex1_mdp.py`、`ex2_dqn.py` 已改动（hw4 正在做）
> - 已添加 `ethz-robot-learning-2026/.gitignore`，`git add -A` 只会收 18 个代码文件
> - 需要搬走的总量：代码 **25 MB**（GitHub）+ 数据 **160 MB**（zip）+ 环境（目标机重建 3.4 GB）

> 生成时间：2026-11-05 · 面向：把本地（无 GPU）的 hw2/hw3 工作搬到有 CUDA 的机器

---

## 一、这个工作区里，哪些是谁写的

### A. 上游仓库原样拷贝（git clone，未改动）

`ethz-robot-learning-2026/` 下的四套作业，来自
<https://github.com/mees-robot-learning-course/ethz-course-2026>，commit `2e089e9`（2026-04-12，**shallow clone，depth=1**）：

```
hw1_pytorch_tutorial/      hw2_robot_control_mdps/
hw3_imitation_learning/    hw4_reinforcement_learning/
```

这些是**课程作者**的代码（README、exercises 骨架、环境、脚本、XML、STL 模型）。

### B. 我从课程主页下载的（不在 git 仓库里）

`ethz-robot-learning-2026/slides/` —— 11 个讲课 PDF（35 MB），来自
<https://cvg.ethz.ch/lectures/Robot-Learning/>。

### C. 我（本会话）手写的脚本/文档

| 文件 | 作用 |
|---|---|
| `README.md` | 环境与作业总说明（含路径、命令、已知问题） |
| `shell.cmd`、`ethz-robot-learning-2026/shell.cmd` | 一键 cd + 激活某个作业的 venv |
| `hw1_jupyter.cmd`、`hw2_viewer.cmd` | 启动 JupyterLab / MuJoCo 窗口 |
| `whichenv.py`、`whichenv.cmd`（+课程目录副本） | 显示当前激活的是哪个 venv |
| `env-lock/hw*.txt` | 四个环境的 `uv pip freeze` 快照 |
| `ethz-robot-learning-2026/hw2_robot_control_mdps/check_ex1.py` | ex1 自检（只验证实现，不含答案） |
| `site_demo.png` | 讲解 site 时渲染的截图（可删） |

### D. 我生成但不是“代码”

- 四个 `.venv/`（安装产物，共 **3.34 GB**）
- `.uv-cache/`（uv 包缓存，**1.94 GB**）
- 每个 venv 里的 `_hw_local.pth`（因本机沙箱无法 `pip install -e .` 而做的等价替代，**venv 内文件，不迁移**）

### E. 你自己改的（`git status` 可见）

```
 M hw2.../exercises/ex1.py  M ex2.py  M ex3.py        ← 你的作业答案
 M hw3.../hw3/teleop_utils.py, scripts/{eval,dagger_eval,record_teleop_demos}.py
?? hw3.../datasets/  hw3.../keymap.json  hw3.../.mplcache/
```

### F. 不是我写的（你其它会话 / 你自己整理的材料）

`hw3_*.py`（8 个）、`HW3_从零实现教程.md`、`HW3_实现函数手册.md`、`hw3_reference/`、`pid-teach/`、
`check_ex2.py`/`check_ex3.py`、`eth-robot-learning/`（**你自己的 GitHub 仓库** kongxinshitou/eth-robot-learning）、
`ik6_*.py`、`scratch_dbg*.py`、`figure_planar.png`、`证件照_*.jpg/png`、`downloads/`。

**怎么自己核对**：`git -C ethz-robot-learning-2026 status` 看“改了什么”，
`git -C ethz-robot-learning-2026 diff` 看“改的具体内容”，文件时间戳区分批次。

---

## 二、带走 / 不带 / 别外传

| 类别 | 内容 | 体积 | 处理 |
|---|---|---|---|
| ✅ **带走** | `ethz-robot-learning-2026/`（**排除** `.venv`、`__pycache__`、`.mplcache`） | ≈ **153 MB** | 主要载荷 |
| ✅ 带走 | 其中 `hw2/logs/`（PPO 权重 11 个文件） | 2 MB | 已训练的模型 |
| ✅ 带走 | 其中 `hw3/datasets/`（zarr 遥操作数据） | 57 MB / 19599 文件 | 不想重录就带上 |
| ✅ 带走 | 其中 `slides/` | 35 MB | 课件，也可重新下 |
| ⭕ 可选 | `hw3_reference/`（专家演示视频） | 14 MB | 教学参考 |
| ⭕ 可选 | `pid-teach/`、`HW3_*.md`、`hw3_*.py` | 1 MB | 学习资料 |
| ⭕ 可选 | `eth-robot-learning/` | <1 MB | **已推到 GitHub，新机直接 clone 更干净** |
| ❌ **不带** | 4 个 `.venv/` | 3.34 GB | 跨平台二进制，**必然重建** |
| ❌ 不带 | `.uv-cache/` | 1.94 GB | 缓存，重建会自动生成 |
| ❌ 不带 | `__pycache__/`、`.mplcache/`、`.tmp/` | ~1 MB | 缓存 |
| 🚫 **别打包外传** | `证件照_祝鸿志.*`、`downloads/id-photo-*` | 0.3 MB | 个人证件照，别进 zip / 别传 GitHub |

**结论：真正要传的只有约 155 MB**（153 + 2），而不是 5.4 GB。

---

## 三、迁移前必须做的 5 件事

1. **备份你的作业改动。** 课程仓库里 `ex1/2/3.py` 等 5 个文件是 `M`（已改未提交），
   `datasets/`、`keymap.json` 是未跟踪。建议二选一：
   - 在课程仓库建本地分支提交：`git checkout -b my-hw2 && git add -A && git commit -m "hw2 work"`
   - 或直接用下面的打包脚本整体复制（含 `.git`，最省心）
2. **确认训练产物要不要**：`hw2_robot_control_mdps/logs/` 里有 checkpoint，带走就不用重训。
3. **确认 hw3 数据要不要**：`hw3/datasets/*.zarr` 57 MB，带走比重录快。
4. **确认没有硬编码路径**：我已经 grep 过，你的脚本里**没有** `D:\` 之类绝对路径，可以直接搬。
5. **记下版本**：Python 3.12.4、mujoco 3.5.0(hw2)/3.17(hw3)、torch CPU 2.15、zarr 3.4.2 —— 见 `env-lock/`。

---

## 四、打包（Windows 本机执行）

```bat
powershell -ExecutionPolicy Bypass -File .\pack_for_gpu.ps1
```

会在 `export\` 下生成 `ethz-robot-learning-gpu.zip`（约 120–150 MB），**自动排除**
`.venv`、`.uv-cache`、`__pycache__`、`.mplcache`、`.tmp`、`*.pyc`、`证件照*`、`downloads\`。

先看会打包什么、多大（不实际复制）：

```bat
powershell -ExecutionPolicy Bypass -File .\pack_for_gpu.ps1 -DryRun
```

---

## 五、GPU 机器上重建

### 5.1 Linux（推荐）

```bash
unzip ethz-robot-learning-gpu.zip -d ~/work
cd ~/work
bash setup_gpu_env.sh                 # 建 4 个 venv + 装依赖 + 冒烟测试
# 或指定 CUDA 版本：
CUDA_INDEX=https://download.pytorch.org/whl/cu124 bash setup_gpu_env.sh
```

> **好消息**：在 Linux 上 `pip install torch` 装到的就是**带 CUDA runtime** 的 wheel，
> 不需要额外指定 index；指定 index 只是锁定 CUDA 版本。
> 另外 hw3 的 `student_eval/*.so` 只有 linux/mac 版 —— 在 Linux GPU 机上反而**能用**了。

### 5.2 Windows + NVIDIA

Windows 上 PyPI 的 torch 是 **CPU-only**，必须显式装 CUDA 版：

```bat
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch torchvision ^
    --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

四个作业分别做一遍（或用 `shell.cmd` 之前的老办法）。

### 5.3 MuJoCo 的无头渲染

- 有桌面：默认 GLFW，直接 `python scripts/inverse_kinematics.py` 会弹窗
- 服务器无桌面：`sudo apt install -y libegl1 libgl1 libglvnd0 libglew2.2`，
  然后 `export MUJOCO_GL=egl`（只对离屏渲染有效；`viewer.launch_passive` 仍需要 X 显示，用 Xvfb 或 MobaXterm）

### 5.4 迁移后自检（按顺序）

```bash
cd ethz-robot-learning-2026/hw2_robot_control_mdps
.venv/bin/python -c "import torch,mujoco,gymnasium,stable_baselines3 as sb3; print(torch.__version__, torch.cuda.is_available(), mujoco.__version__, sb3.__version__)"
.venv/bin/python check_ex1.py            # 应打印 ALL CHECKS PASSED
.venv/bin/python -c "from envs..."        # (hw4) 见 README 的 smoke test A/B
```

hw4 的两条官方 smoke test：

```bash
cd ../hw4_reinforcement_learning
.venv/bin/python -c "from envs.grid_world import CliffWalkingEnv; from envs.cartpole_wrapper import CartPoleWrapper; g=CliffWalkingEnv(); c=CartPoleWrapper(seed=0); s=c.reset(); c.step(c.sample_action()); print('OK A', g.n_states)"
.venv/bin/python -c "from pathlib import Path; import numpy as np; from envs.so100_rl_env import SO100RLEnv; e=SO100RLEnv(xml_path=Path('assets/mujoco/so100_pos_ctrl.xml').resolve(), render_mode=None); o,_=e.reset(seed=0); e.step(np.zeros(e.action_dim, dtype=np.float32)); print('OK B', o.shape)"
```

---

## 六、跨平台迁移的常见坑

| 坑 | 说明 / 处理 |
|---|---|
| 复制 `.venv` | Windows 的 `python.exe` 在 Linux 跑不了；即使同为 Windows，CPU→CUDA 也要重装 torch |
| `_hw_local.pth` | venv 内的绝对路径文件，venv 不迁移就不用管；新机若 `pip install -e .` 成功就不需要它 |
| `torch` 版本对不上 | `env-lock` 里是 CPU 版（`+cpu` 后缀），GPU 机装对应 `cu124` 版本即可，其余包版本照抄 |
| `.cmd` 脚本 | Linux 上不能用 → 用 `setup_gpu_env.sh`，或用 `bash` 手敲等价命令 |
| CRLF | 你在 Windows 改过的 `.py` 是 CRLF，Python 无所谓；但 `.sh` 必须 LF（`dos2unix setup_gpu_env.sh`） |
| shallow clone | 只有 1 个 commit，`git pull` 拿不到历史；想要完整历史：`git fetch --unshallow` |
| hw3 `.so` | 只有 linux/mac；GPU 机是 Windows 的话本地评测仍然跑不了 |
| Jupyter | 新机首次 `jupyter lab` 会重建 kernelspec；kernel 选 `hw1_pytorch_tutorial/.venv` 那个 |
| `datasets/` 与 `logs/` | 都是未跟踪文件，`git clone` 拿不到，必须随包带走 |
| 内存/显存 | hw2 的 PPO 是 16 个并行 CPU 环境，GPU 帮不上多少；hw3/hw4 训练才是 GPU 受益点 |

---

## 七、一句话总结

> 代码层面：**只有 `slides/`、`check_ex1.py` 和几个 `.cmd`/`.py` 小工具是我加的**，
> 四套作业本体全部来自课程仓库；你的作业答案在 `exercises/ex1-3.py`（未提交）。
> 迁移时**丢掉 3.34 GB 的 venv 和 1.94 GB 缓存，只带 155 MB 代码+数据**，到 GPU 机上跑
> `setup_gpu_env.sh` 重建环境，再用 `check_ex1.py` 和 hw4 的两条 smoke test 验收。
