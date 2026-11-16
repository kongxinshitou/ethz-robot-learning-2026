# 作业 4：强化学习（Homework 4: Reinforcement Learning）

- **截止日期：2026 年 4 月 16 日 23:59（CET，中欧时间）**
- **需独立完成。将进行抄袭检查。**

> 译注：本文件是官方英文原文 `ethz-robot-learning-2026/hw4_reinforcement_learning/README.md` 的逐节忠实中文翻译，章节顺序与原文一致；公式、命令、文件名保留原样。原文与仓库实际文件不一致之处已用「译注」标出。

### 安装

所有练习共用一个干净的虚拟环境。该安装流程已在 Linux 发行版和 Windows + Python 3.12 上测试过。你也可以使用 CAB H 56 / CAB H 57 的 Linux 计算机。

### 1) 创建并激活虚拟环境

在仓库根目录下执行：

**Linux / macOS**

```bash
cd hw4_reinforcement_learning
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows（PowerShell）**

```powershell
cd hw4_reinforcement_learning
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) 安装依赖

```bash
pip install -r requirements.txt
```

### 3) 运行快速验证测试

**验证测试 A（ex1/ex2 技术栈：GridWorld + CartPole）**

```bash
python -c "from envs.grid_world import CliffWalkingEnv; from envs.cartpole_wrapper import CartPoleWrapper; g=CliffWalkingEnv(); c=CartPoleWrapper(seed=0); s=c.reset(); ns,r,d,i=c.step(c.sample_action()); c.close(); print('OK ex1/ex2:', g.n_states, s.shape, type(r).__name__)"
```

**验证测试 B（ex3/ex4 技术栈：MuJoCo + SO100）**

```bash
python -c "from pathlib import Path; import numpy as np; from envs.so100_rl_env import SO100RLEnv; env=SO100RLEnv(xml_path=Path('assets/mujoco/so100_pos_ctrl.xml').resolve(), render_mode=None); obs,_=env.reset(seed=0); obs2,reward,term,trunc,info=env.step(np.zeros(env.action_dim, dtype=np.float32)); env.close(); print('OK ex3/ex4:', obs.shape, float(reward))"
```

如果两条命令都打印出 `OK ...`，说明安装正常。

### 通用准则

- 完成所提供代码中的 **TODO**。
- 除非明确说明，**不要修改函数签名**。
- 使用 `scripts/` 文件夹中的脚本训练和测试你的实现。
- 结果（图、日志）将保存在 `logs/` 目录中。

保持代码整洁、可读。提交前确保所有代码都能运行。

### 视频准则

视频的目的是展示你对所实现算法的理解。

- 讲解要**清晰简洁**
- 聚焦**关键思想与结果**
- 超过时长限制（5 分钟）可能导致扣分

### 评分

每个练习可获得的最高分：

- **练习 1：MDP**：10 分
- **练习 2：DQN**：10 分
- **练习 3：PPO**：15 分
- **练习 4：SAC**：15 分
- **视频提交**：50 分

## 练习 1：GridWorld 中的动态规划

在本练习中，你将在**悬崖行走（Cliff Walking）**环境上，用表格型 MDP 实现**策略迭代（policy iteration）**和**价值迭代（value iteration）**。

### 悬崖行走环境

悬崖行走环境是一个二维网格世界：智能体从左下角出发，必须到达右下角的目标，同时避开悬崖。该环境是**随机**的：当智能体选择一个动作（上、下、左、右）时，有 `1 - slip_chance` 的概率按意图执行；另有 `slip_chance` 的概率随机均匀地执行另一个动作（即智能体可能"打滑"）。移出网格边界的动作会让智能体停留在原状态。每走一步奖励为 −1；掉入悬崖奖励为 −100 并终止回合；到达目标同样结束回合。

![图 1-1](ethz-robot-learning-2026/hw4_reinforcement_learning/assets/imgs/CliffWalking.png)

*图 1-1：悬崖行走环境（AI 生成）*

### 算法

在本练习中，你将实现两个经典的动态规划算法：**策略迭代**和**价值迭代**。

下面给出了两个算法的伪代码。

![图 1-2](ethz-robot-learning-2026/hw4_reinforcement_learning/assets/imgs/policy_iteration.png)

*图 1-2：策略迭代*

![图 1-3](ethz-robot-learning-2026/hw4_reinforcement_learning/assets/imgs/value_iteration.png)

*图 1-3：价值迭代*

两种方法都应收敛到最优策略。

## TODO（待完成项）

### 代码实现

填写 `exercises/ex1_mdp.py` 中的 TODO：

1. **策略迭代：** 实现**策略评估**、**策略改进**以及**策略迭代主循环**。
2. **价值迭代：** 实现**价值迭代更新**和**贪心策略提取**。

关于 GridWorld 环境更详细的实现与说明，见 `envs/grid_world.py`。
实现完 TODO 后，可运行以下命令测试算法结果：

```bash
python scripts/run_policy_iteration.py
```

```bash
python scripts/run_value_iteration.py
```

结果将保存在 `logs/mdp/` 中。

你应该能看到**状态值**可视化和**最优策略**可视化。

### 随机性的影响

在该环境中，动作可能由于"打滑"而无法按意图执行。为研究其影响，请通过设置参数 `--slip_chance` 用不同的取值做实验。

### 理论问题

1. 从更新过程的角度看，策略迭代与价值迭代有什么区别？
   策略迭代在计算V的过程中，考虑到了pi的概率，是分别更新V和pi，等pi收敛不再更新。 价值迭代直接贪心地更新V，得到的V与pi的概率无关，是状况s下的最优回馈，然后根据V更新pi，等v收敛后停止更新v和pi
2. 如果折扣因子 `gamma` 接近 0 或接近 1，会发生什么？
   折扣因子接近0，策略只会选择当前收益最大的动作，其reward则为V的值，在当前的条件下，学得的策略无法使用。如果接近1则未来的每一步reward不会递减，容易受到随机初始化的影响，策略会收敛的比较慢
3. 增大打滑概率（`slip_chance`）会如何影响最优策略？
   - 对比 `slip_chance = 0.0`、`0.01`、`0.2` 三种情况。
   - 为什么随着随机性增大，智能体的行为会趋于更加保守？

### 交付物

1. **代码：** 你在 `exercises/ex1_mdp.py` 中填好 TODO 的代码。
2. **理论问题：** 视频中必须包含你对理论问题的回答。
3. **图片：** 将 `slip_chance` 分别设为 `0`、`0.01`、`0.2`，运行 `scripts/run_policy_iteration.py` 与 `scripts/run_value_iteration.py` 得到的状态值函数可视化与最优策略可视化。

## 练习 2：CartPole 上的深度 Q 网络（DQN）

在本练习中，你将实现**深度 Q 网络（DQN）**，用于解决一个具有**连续状态空间**的控制问题。

### 为什么要用 DQN？

在之前的练习中，我们使用表格型强化学习，用一张表存储每个状态-动作对的价值。但当状态空间很大或连续时，这种做法不再可行。

为解决该问题，我们使用**函数逼近**来估计 Q 值。DQN 用神经网络代替表格，从而能够处理**连续状态空间 + 离散动作**的问题。

### CartPole 环境

CartPole 是经典控制问题：一根杆子通过铰链连接在小车上，小车沿轨道移动。目标是对小车施加力，使杆子保持竖直。

![图 2-1](ethz-robot-learning-2026/hw4_reinforcement_learning/assets/gifs/cartpole.gif)

*图 2-1：CartPole 环境*

在每个时间步，智能体接收到**4 维连续状态**，并选择一个**离散动作**。

#### 状态空间

| 索引 | 描述       | 范围                 |
| ---- | ---------- | -------------------- |
| 0    | 小车位置   | [-2.4, 2.4]          |
| 1    | 小车速度   | (-inf, inf)          |
| 2    | 杆子角度   | 约 [-41.8°, 41.8°] |
| 3    | 杆子角速度 | (-inf, inf)          |

#### 动作空间

| 动作 | 描述       |
| ---- | ---------- |
| 0    | 向左推小车 |
| 1    | 向右推小车 |

智能体在**每个时间步获得 +1 奖励**。回合结束条件：

- 杆子倾斜超过阈值角度，
- 小车偏离中心过远，
- 或达到最大回合长度。

### DQN 算法

DQN 扩展了 Q-learning，用神经网络来逼近 Q 函数。

DQN 的关键组成部分：

- **Q-learning 更新：** 从贝尔曼方程学习
- **经验回放（Experience Replay）：** 随机采样小批量数据以打破相关性
- **目标网络（Target Network）：** 使用缓慢更新的网络来稳定训练

更多细节见原始论文：https://arxiv.org/abs/1312.5602 。下面是提供的伪代码。

![图 2-2](ethz-robot-learning-2026/hw4_reinforcement_learning/assets/imgs/DQN.png)

*图 2-2：深度 Q 网络（DQN）*

## TODO（待完成项）

### 代码实现与超参数调优

填写 `exercises/ex2_dqn.py` 和 `exercises/ex2_dqn_config.py` 中的 TODO：

1. **实现 DQN：** 完成 DQN 的核心组件，包括将转移存入回放缓冲区、实现 Q 网络前向传播、用 ε-greedy 策略选择动作，以及计算训练用的 TD 目标。
2. **超参数调优：** 在 `exercises/ex2_dqn_config.py` 中调优关键超参数（`lr`、`epsilon`、`target_update`、`hidden_dim`）以提升性能。

### 训练

实现完 TODO 后，可用以下命令训练智能体：

```bash
python scripts/train_dqn.py
```

训练结果（模型与曲线）将保存在：

```bash
python logs/dqn
```

> 译注：上一行的命令是原文笔误，实际是 `logs/dqn` 目录，不是一条 `python` 命令。

### 评估

评估训练好的模型：

```bash
python scripts/eval_dqn.py
```

也可以在 GUI 窗口中可视化学到的策略：

```bash
python scripts/eval_dqn.py --play
```

这会打开一个交互窗口，你可以直接观察智能体的行为。

你也可以选择录制智能体视频：

```bash
python scripts/eval_dqn.py --record_video
```

注意：`--play` 与 `--record_video` 不能同时使用，因为二者需要不同的渲染模式。

### 理论问题

1. 为什么经验回放在 DQN 中很重要？
2. 目标网络在 DQN 中的作用是什么？它如何提升稳定性？
3. 什么是 Double DQN？与标准 DQN 相比，它如何减轻过高估计偏差？（参见：[Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)）

### 交付物

1. **代码：** 你在 `exercises/ex2_dqn.py` 和 `exercises/ex2_dqn_config.py` 中完成 TODO 后的实现。
2. **结果：** 运行所提供脚本得到的训练曲线和打印出的评估摘要。将所有图放入同一个 PDF 文件。
3. **理论问题：** 视频中必须包含你对理论问题的回答。

## 练习 3：SO100 上的近端策略优化（PPO）

在本练习中，你将针对 SO100 机器人的连续控制末端跟踪任务实现 PPO。我们重新审视作业 2 中介绍过的 SO100 MuJoCo 环境，这次从强化学习的视角出发。

### 引言

练习 1 和 2 聚焦低维玩具环境（表格型网格世界和 CartPole），非常适合学习核心 RL 思想，但离真实机器人控制还很远。练习 3 和 4 将转向基于 MuJoCo 的 SO100 跟踪任务：策略需要输出连续动作（关节空间），并处理更丰富的动力学。

练习 3（on-policy）与练习 4（off-policy）的结构刻意保持一致。你将在相同的环境、奖励结构和评估协议下比较两大 RL 范式。目标不仅是实现两种算法，还要理解连续控制中稳定性与样本效率之间的权衡。

建议在开始实现前阅读下面两篇指定论文。我们的建议实现与论文略有偏差。我们所用实现（本作业）的伪代码见下文。

### 状态空间

观测是在机器人基座坐标系下构造的 19 维向量：

| 索引   | 维度 | 描述                                 |
| ------ | ---- | ------------------------------------ |
| 0–5   | 6    | 关节位置（`qpos`）                 |
| 6–8   | 3    | 末端执行器位置（基座坐标系）         |
| 9–12  | 4    | 末端执行器姿态四元数（基座坐标系）   |
| 13–15 | 3    | 目标位置（基座坐标系）               |
| 16–18 | 3    | 位置误差：目标 − 末端（基座坐标系） |

### 动作空间

策略输出 **6 维连续动作**，取值在 \([-1, 1]^6\)。每个分量通过 `process_action` 线性映射到对应关节的物理位置范围。

### 关节空间

SO100 机械臂有 6 个转动关节（位置控制）：

| 索引 | 关节名      | 轴 | 范围（rad）     |
| ---- | ----------- | -- | --------------- |
| 0    | Rotation    | Y  | [−1.92, 1.92]  |
| 1    | Pitch       | X  | [−3.32, 0.174] |
| 2    | Elbow       | X  | [−0.174, 3.14] |
| 3    | Wrist_Pitch | X  | [−1.66, 1.66]  |
| 4    | Wrist_Roll  | Y  | [−2.79, 2.79]  |
| 5    | Jaw         | Z  | [−0.174, 1.75] |

### 建议阅读

- PPO 论文：https://arxiv.org/abs/1707.06347
- 广义优势估计（GAE，PPO 中使用）：https://arxiv.org/abs/1506.02438

## TODO（待完成项）

### 代码实现

填写 `exercises/ex3_ppo.py` 中的 TODO（原文写作 `ex3_ppo_student.py`，见文末译注）：

1. **动作/价值/对数概率计算：** 在 `select_action` 中实现策略动作采样、动作裁剪、对数概率获取，以及动作均值与标准差的获取。
2. **KL 散度：** 在 `compute_kl_mean` 中计算新旧高斯策略分布之间的 KL 散度。
3. **代理损失（surrogate loss）：** 在 `compute_surrogate_loss` 中实现 PPO 的代理损失项。
4. **价值损失：** 在 `compute_value_loss` 中实现 PPO 的价值损失。
5. **熵损失：** 在 `compute_entropy_loss` 中实现 PPO 的熵损失。
6. **优化步骤：** 在 `update` 中完成 actor 与 critic 更新循环的一步。

### 训练

实现完 TODO 后，可用以下命令训练智能体：

```bash
python scripts/train_ppo.py
```

训练产物保存在：

- `logs/ppo/YY_MM_DD_HH_MM_SS_model`（checkpoint 与 TensorBoard 事件日志）

训练曲线保存在 TensorBoard 中。运行以下命令查看：

```bash
tensorboard --logdir="your tensorboard event file path"
```

### 评估

评估训练好的模型：

```bash
python scripts/eval_ppo.py
```

在 GUI 窗口中回放策略：

```bash
python scripts/eval_ppo.py --play
```

在 macOS 上，若出现 MuJoCo/OpenGL 渲染问题，请用以下命令评估：

```bash
.venv/bin/mjpython scripts/eval_ppo.py --play
```

评估摘要会打印在终端。示例如下：

```
===== Evaluation Summary =====
Number of episodes   : 20
Mean return          : 60.058
Std return           : 12.936
Min return           : 21.256
Max return           : 69.988
Median return        : 65.168
Mean length          : 30.00
Std length           : 0.00
Mean tracking error  : 0.015097
Std tracking error   : 0.015067
Min tracking error   : 0.003123
Max tracking error   : 0.064921
```

### 理论问题

1. PPO 为什么对概率比（probability ratio）做裁剪，而不是像 TRPO 那样直接约束 KL 散度？如果完全去掉裁剪会出什么问题？
2. PPO 在每次更新后丢弃所有已收集的数据。为什么不能把旧 rollout 简单地重复用于更多梯度步？
3. GAE 参数 \(\lambda\) 控制什么？在极端情况 \(\lambda = 0\) 与 \(\lambda = 1\) 时会怎样？

### 交付物

1. **代码：** 你在 `exercises/ex3_ppo.py` 中完成 TODO 后的实现。
2. **结果：** 所有 TensorBoard 训练曲线，以及运行所提供脚本打印出的评估摘要。将所有图放入同一个 PDF 文件。
3. **理论问题：** 视频中必须包含你对理论问题的回答。

## 练习 4：SO100 上的软演员-评论家（SAC）

在本练习中，你将针对相同的连续控制 SO100 跟踪任务实现 SAC。与练习 3 一样，这将重新审视作业 2 的 SO100 MuJoCo 环境。

### 建议阅读

- SAC 论文：https://arxiv.org/abs/1812.05905

### 与 SAC 原文的差异

我们的实现遵循 SAC 核心算法（Haarnoja 等，2018），但额外增加了两个特性。

1. **自动温度调节（automatic temperature tuning）** 默认开启。
2. 我们的实现在最初 `start_steps = 1000` 个环境步中使用**均匀随机动作**，以便在学习开始前用多样化的初始数据填充回放缓冲区。

## TODO（待完成项）

### 代码实现

填写 `exercises/ex4_sac.py` 中的 TODO（原文写作 `ex4_sac_student.py`，见文末译注）：

1. **策略动作采样：** 在 `select_action` 中实现动作采样。
2. **Critic 损失：** 在 `compute_critic_loss` 中实现 SAC 贝尔曼目标和 critic 损失。
3. **Actor 损失：** 在 `compute_actor_loss` 中实现 actor 目标。
4. **熵损失：** 在 `compute_actor_loss` 中实现 alpha 损失。
5. **目标更新：** 在 `soft_update_targets` 中实现目标网络的软更新。
6. **优化步骤：** 在 `update` 中完成 actor、critic 与 alpha 更新循环的一步。

### 训练

实现完 TODO 后，可用以下命令训练智能体：

```bash
python scripts/train_sac.py
```

训练产物保存在：

- `logs/sac/YY_MM_DD_HH_MM_SS_model`（checkpoint 与 TensorBoard 事件日志）

训练曲线保存在 TensorBoard 中。运行以下命令查看：

```bash
tensorboard --logdir="your tensorboard event file path"
```

### 评估

评估训练好的模型：

```bash
python scripts/eval_sac.py
```

运行 GUI 窗口回放策略：

```bash
python scripts/eval_sac.py --play
```

与练习 3 相同，评估摘要会打印在终端。

### 理论问题

1. SAC 在奖励中加入熵奖励（entropy bonus），这有什么好处？
2. SAC 通过 tanh 对动作做压缩。为什么这需要对对数概率做修正？
3. 温度 \(\alpha\) 是自动调节的。当策略熵高于/低于目标熵时分别会发生什么？
4. 从更新-数据比（update-to-data，UTD）的角度，SAC 与 PPO 相比如何？（UTD = 梯度更新步数 / 环境步数）
5. 简要讨论 on-policy 与 off-policy 算法的优缺点。

### 交付物

1. **代码：** 你在 `exercises/ex4_sac.py` 中完成 TODO 后的实现。
2. **结果：** 所有 TensorBoard 训练曲线，以及运行所提供脚本打印出的评估摘要。将所有图放入同一个 PDF 文件。
3. **理论问题：** 视频中必须包含你对理论问题的回答。

## 提交

### 代码提交

在 Gradescope 的 **Homework 4** 作业中提交以下文件：

- `exercises/ex1_mdp_student.py`
- `exercises/ex2_dqn_student.py`
- `exercises/ex2_dqn_config.py`
- `exercises/ex3_ppo_student.py`
- `exercises/ex4_sac_student.py`

**不要重命名任何文件，也不要修改函数签名。** 自动评分器会在上传时验证你的实现。

### 视频提交

提交一个覆盖所有练习的单个视频文件（`.mp4`）。如果文件太大无法提交，可以拆分为多个文件，但要清楚标明顺序。

- **最长时长：5 分钟。** 超过时限将导致扣分。不允许加速（如 1.5 倍、2 倍速播放）。
- 粗略参考：练习 1–3 各约 1 分钟，练习 4 约 2 分钟。
- 可以使用幻灯片、图表或录屏来辅助讲解。
- 讲解时结果必须**显示在屏幕上**。
- 提交前检查视频与音频。

### PDF 提交

为练习 2、3、4 各提交一份 PDF 文件。对于练习 1，你应在视频提交中可视化所有图片以回答理论问题，因此不需要 PDF 文件。

---

## 译注（译者补充，非原文）

1. **文件名不一致。** 原文多处写 `ex3_ppo_student.py`、`ex4_sac_student.py`，提交清单也写 `ex1_mdp_student.py` 等；但仓库（含上游 GitHub main 分支）实际文件是 `exercises/ex1_mdp.py`、`ex2_dqn.py`、`ex3_ppo.py`、`ex4_sac.py`，**没有** `_student` 后缀。以 Gradescope 页面实际要求的文件名为准。
2. **配图。** 原文伪代码图在仓库 `assets/imgs/` 下（`policy_iteration.png`、`value_iteration.png`、`DQN.png`、`CliffWalking.png`），本译文已用相对路径引用，可在支持 Markdown 的预览器里直接显示。
3. **原文中两处被注释掉的伪代码小节。** 原文在练习 3、练习 4 中各有一段被 HTML 注释包裹的"Implementation Pseudocode"小节（引用 `ppo_pseudocode.png`、`sac_pseudocode.png`），但仓库中并不存在这两张图，渲染时不可见，故译文未收录。
