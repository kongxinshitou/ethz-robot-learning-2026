# ETH Zürich · Robot Learning 作业与环境

课程：**Robot Learning: From Fundamentals to Foundation Models**（263-5911-00L，Spring 2026，讲师 Oier Mees）
状态：4 套作业（hw1–hw4）已全部下载到本地，每套作业各自配好独立 Python 3.12 虚拟环境，并已跑通冒烟测试。

## 1. 目录结构

```
default-workspace/
├─ ethz-robot-learning-2026/        # 官方仓库 clone（main @ 2e089e9，2026-04-12）
│  ├─ hw1_pytorch_tutorial/         # 作业1：PyTorch 教程（src/ex1..ex4.ipynb）
│  │  └─ .venv/                     #   torch + torchvision + jupyter
│  ├─ hw2_robot_control_mdps/       # 作业2：机器人控制与 MDP（SO-101 机械臂 / MuJoCo）
│  │  └─ .venv/
│  ├─ hw3_imitation_learning/       # 作业3：模仿学习（遥操作采集 + 行为克隆）
│  │  └─ .venv/
│  ├─ hw4_reinforcement_learning/   # 作业4：强化学习（DP / DQN / PPO / SAC）
│  │  └─ .venv/
│  └─ slides/                       # 额外下载：11 讲课件 PDF
├─ env-lock/                        # 各环境导出的精确依赖版本（uv pip freeze）
├─ shell.cmd                        # 一键进入某个作业的已激活环境
├─ hw1_jupyter.cmd                  # 一键打开作业1 的 JupyterLab
└─ hw2_viewer.cmd                   # 一键打开作业2 的 MuJoCo 机械臂窗口
```

来源：课程主页 <https://cvg.ethz.ch/lectures/Robot-Learning/> · 代码仓库 <https://github.com/mees-robot-learning-course/ethz-course-2026>

## 2. 环境一览

| 作业 | 环境 | 关键依赖（实测版本） | 验证结果 |
|---|---|---|---|
| hw1 | `hw1_pytorch_tutorial/.venv` | torch 2.15.0+cpu、torchvision 0.29.2、jupyterlab 4.6.5、numpy 2.5.4 | ✅ 张量运算、Jupyter 正常 |
| hw2 | `hw2_robot_control_mdps/.venv` | mujoco 3.5.0、gymnasium 1.2.3、stable-baselines3 2.7.1、numpy 2.4.2、tensorboard 2.20.0 | ✅ 机械臂模型加载 + 仿真推进、`train.py --help` |
| hw3 | `hw3_imitation_learning/.venv` | torch 2.15.0+cpu、mujoco 3.17.0、dm-control 1.0.50、zarr 3.4.2、opencv 5.0.0、pyquaternion | ✅ 单立方体场景 `reset()` 成功（含离屏渲染） |
| hw4 | `hw4_reinforcement_learning/.venv` | torch 2.15.0+cpu、gymnasium 1.4.0、mujoco 3.17.0、moviepy 2.2.1、pygame 2.6.1 | ✅ README 两条 smoke test 均打印 OK |

全部为 **CPU 版**（课程说明作业在 CPU 上即可完成，无需 GPU）。四个环境互相隔离，因为 hw2 与 hw3 都提供同名 `so101_gym` 包，共用环境会冲突。

## 3. 怎么用

### 进入某个作业的环境

双击 `shell.cmd`，或在命令行里：

```bat
shell.cmd hw2            :: hw1 / hw2 / hw3 / hw4
```

它会 `cd` 到对应目录并激活 `.venv`，之后 `python`、`pip` 都指向该环境（`deactivate` 退出）。

> PowerShell 里请改用 `.venv\Scripts\Activate.ps1`；若提示"禁止运行脚本"，先执行一次
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`，或直接用 `.venv\Scripts\activate.bat`。

### 作业 1：PyTorch 教程（截止 2026-03-05）

```bat
hw1_jupyter.cmd
```

等价于 `cd hw1_pytorch_tutorial && .venv\Scripts\python.exe -m jupyter lab src`。
四个 notebook 在 `hw1_pytorch_tutorial/src/`（`ex1.ipynb` … `ex4.ipynb`），填完所有 `TODO` 即可。
注意：不要改动函数名和签名，自动评分器会直接 import 这些函数。

### 作业 2：机器人控制与 MDP（截止 2026-03-12）

```bat
cd ethz-robot-learning-2026\hw2_robot_control_mdps
.venv\Scripts\python.exe scripts\interactive.py      :: 打开 MuJoCo 观察窗（= hw2_viewer.cmd）
.venv\Scripts\python.exe scripts\inverse_kinematics.py
.venv\Scripts\python.exe scripts\pid_control.py
.venv\Scripts\python.exe scripts\train.py             :: PPO 训练
```

需要改的文件：`exercises/ex1.py`、`exercises/ex2.py`、`exercises/ex3.py`。
写完作业1 可以先跑自检脚本 `python check_ex1.py`（只检查你的实现，不含答案）。

### 作业 3：模仿学习（截止 2026-03-26）

```bat
cd ethz-robot-learning-2026\hw3_imitation_learning
.venv\Scripts\python.exe scripts\configure_keys.py    :: 先配键盘，生成 keymap.json
.venv\Scripts\python.exe scripts\record_teleop_demos.py   :: 遥操作采集（十几条即可）
.venv\Scripts\python.exe scripts\compute_actions.py       :: 生成动作标签
.venv\Scripts\python.exe scripts\train.py --zarr <录制的.zarr> --policy obstacle ...
```

### 作业 4：强化学习（截止 2026-04-16）

```bat
cd ethz-robot-learning-2026\hw4_reinforcement_learning
.venv\Scripts\python.exe scripts\run_policy_iteration.py   :: ex1 策略迭代（结果在 logs/mdp/）
.venv\Scripts\python.exe scripts\run_value_iteration.py
.venv\Scripts\python.exe scripts\train_dqn.py              :: ex2
.venv\Scripts\python.exe scripts\train_ppo.py              :: ex3
.venv\Scripts\python.exe scripts\train_sac.py              :: ex4
```

## 4. 已验证的冒烟测试

```text
hw1  torch 2.15.0+cpu / torchvision 0.29.2 / jupyterlab 4.6.5        -> OK
hw2  MjModel.from_xml_path(so100_pos_ctrl.xml) + 100 步 mj_step      -> nq=6 nu=6  OK
hw3  SO100SimEnv(reset) 观测 = cube/ee/ee_pos/goal_pos/gripper/joints/obstacle -> OK
hw4  smoke test A: OK ex1/ex2: 48 (4,) float                        -> OK
hw4  smoke test B: OK ex3/ex4: (19,) -0.0236                        -> OK
```

## 5. 已知问题与说明

1. **`pip install -e .` 的替代方案。** 本机文件沙箱禁止构建后端在临时目录里创建/删除文件，因此 hw2、hw3 的
   `pip install -e .` 无法完成（`WinError 5`）。我改用等价做法：在每个 venv 的
   `Lib/site-packages/_hw_local.pth` 里写入作业根目录，效果同样是 `import so101_gym` / `import hw3` 可用。
   你在自己的普通终端里（沙箱之外）可以照常执行 `pip install -e .`，不受影响。
2. **hw3 的本地评分器只有 Linux/macOS 版本。** `hw3_imitation_learning/student_eval/` 里是
   `eval_harness.cpython-312-x86_64-linux-gnu.so` 和 `...-darwin.so`，Windows 无法加载，
   `python student_eval/run_eval.py` 会导入失败。要在本地算成功率，请用 WSL / Linux 机器，
   或直接用 `scripts/eval.py`（MuJoCo 窗口里跑），最终分数以 Gradescope 为准。
3. **MuJoCo 窗口需要图形界面。** 远程无头环境需 EGL/虚拟显示；本机有桌面，直接跑即可。
   若窗口黑屏，先更新显卡驱动。
4. **`.uv-cache/`（约 1–2 GB）** 是 uv 的包缓存（因沙箱限制放在工作区内）。想省空间可删除：
   `Remove-Item -Recurse -Force .uv-cache`，只是以后重装依赖会重新下载。
5. **作业需提交到 Gradescope**（不是邮件/仓库 PR）：hw1 交 4 个 ipynb + 视频；hw2 交 `exercises/*.py` +
   视频；hw3 交 `.hwresult` 文件；hw4 交代码 + 视频。视频与代码各自计分，具体见各 `README.md`。

## 6. 备注

- 仓库 `README.md` 里写的路径（如 `src/ex1_tensor_basics/ex1.ipynb`）与实际仓库略有出入，实际文件是
  `hw1_pytorch_tutorial/src/ex1.ipynb`。
- 依赖版本如需复现：见 `env-lock/*.txt`；从零重建可执行
  `uv venv --python 3.12 .venv` + `uv pip install -r env-lock/<hw>.txt`。
