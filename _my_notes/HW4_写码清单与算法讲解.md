# HW4 写码清单与算法讲解（自己动手版）

> 对应仓库：`ethz-robot-learning-2026/hw4_reinforcement_learning/`
> 目标：你只写 **5 个文件里的 23 处 TODO**，其余全部是给你的（环境、网络、buffer、训练脚本、评估脚本）。
> 本文只给「要写什么 + 用什么公式 + 用哪个 API + 哪里有坑」，**不给可直接粘贴的答案**，代码由你自己落笔。

---

## 0. 先看清楚：交什么、值多少分

| 练习 | 算法 | 要改的文件 | 代码分 | 视频分 |
|---|---|---|---|---|
| Ex1 | 表格型 DP（策略迭代 / 价值迭代） | `exercises/ex1_mdp.py` | 10 | 合计 50 |
| Ex2 | DQN（连续状态 + 离散动作） | `exercises/ex2_dqn.py` + `exercises/ex2_dqn_config.py` | 10 | （一个 5 分钟视频覆盖全部 4 个练习） |
| Ex3 | PPO（连续动作，on-policy） | `exercises/ex3_ppo.py` | 15 | |
| Ex4 | SAC（连续动作，off-policy） | `exercises/ex4_sac.py` | 15 | |

**关键事实：代码只占 50 分，视频占 50 分。** 视频里必须把结果**显示在屏幕上**（曲线、策略可视化、评估摘要），并且口头回答每个练习的理论题。所以别把时间全砸在调参上。

> 文件名的坑：README 写要交 `ex1_mdp_student.py` 之类，但仓库里实际是 `ex1_mdp.py` / `ex2_dqn.py` / `ex3_ppo.py` / `ex4_sac.py`（没有 `_student` 后缀）。以 Gradescope 页面实际要求为准，**不要重命名，不要改函数签名**。

---

## 1. 仓库地图：哪些是"给"的，哪些是"你写"的

```
hw4_reinforcement_learning/
├─ envs/                      ← 全给
│  ├─ grid_world.py           CliffWalkingEnv：P[s][a] = [(prob, next_state, reward, done)]
│  ├─ cartpole_wrapper.py     CartPole 薄封装
│  ├─ so100_rl_env.py         SO100 MuJoCo 环境（19 维观测 / 6 维动作）
│  └─ so100_mdp_utils.py      process_action：把 [-1,1] 映射到关节角范围
├─ rl/                        ← 全给（这是你最该先读的部分）
│  ├─ networks.py             ValueNet / QNet / DoubleQNet / GaussianActor / SquashedGaussianActor
│  ├─ buffers.py              ReplayBuffer（SAC 用）/ RolloutBatch / RolloutBuffer（PPO 用，GAE 已实现）
│  └─ common.py               set_seed / ensure_dir
├─ exercises/                 ← ★你的战场
│  ├─ ex1_mdp.py              ★ 7 处 TODO
│  ├─ ex2_dqn.py              ★ 4 处 TODO
│  ├─ ex2_dqn_config.py       ★ 4 个超参数
│  ├─ ex3_ppo.py              ★ 6 处 TODO
│  └─ ex4_sac.py              ★ 6 处 TODO
└─ scripts/                   ← 全给，训练/评估/画图入口
   ├─ run_policy_iteration.py / run_value_iteration.py
   ├─ train_dqn.py / eval_dqn.py
   ├─ train_ppo.py / eval_ppo.py
   └─ train_sac.py / eval_sac.py
```

**先读 `rl/networks.py` 和 `rl/buffers.py`（约 350 行，20 分钟）。** 因为 Ex3/Ex4 有一半的数学已经在里面写好了：

- `GaussianActor` 已经实现 `act()`、`get_actions_log_prob()`、`action_mean`、`action_std`、`entropy`
- `SquashedGaussianActor.act()` 已经做了 **tanh + log-prob 修正**（自己别重复做）
- `RolloutBuffer.compute_returns()` 已经实现 **GAE(λ) 和回报，并做了 advantage 归一化**
- `DoubleQNet` 已经封装两个 critic

---

## 2. 总清单：23 处 TODO 速查表

| # | 文件:函数 | 你要写什么 | 算法关键词 | 难度 |
|---|---|---|---|---|
| 1 | `ex1_mdp.py:policy_evaluation` | 用 π 加权的贝尔曼期望更新 + 收敛判据 | 贝尔曼期望方程 | ★ |
| 2 | `ex1_mdp.py:policy_improvement` | 每个 s 的 Q(s,a) → 贪心（tie 均分已写） | 策略改进 | ★ |
| 3 | `ex1_mdp.py:policy_iteration` | 评估→改进→策略不变则停 | 策略迭代主循环 | ★ |
| 4 | `ex1_mdp.py:value_iteration` | 贝尔曼最优更新 + 收敛判据 | 贝尔曼最优方程 | ★ |
| 5 | `ex1_mdp.py:get_policy` | 再算一遍 Q → greedy 提取策略 | 策略提取 | ★ |
| 6 | `ex2_dqn.py:ReplayBuffer.add` | 存一条 transition | 经验回放 | ★ |
| 7 | `ex2_dqn.py:QNet.forward` | Linear→ReLU→Linear | 函数逼近 | ★ |
| 8 | `ex2_dqn.py:DQN.take_action` | ε-greedy | 探索/利用 | ★ |
| 9 | `ex2_dqn.py:DQN.update` | TD target | Q-learning 目标 | ★★ |
| 10 | `ex2_dqn_config.py` | 调 lr / epsilon / target_update / hidden_dim | 超参 | ★★ |
| 11 | `ex3_ppo.py:select_action` | 采样+裁剪+logp+mu/std+value | 高斯策略采样 | ★★ |
| 12 | `ex3_ppo.py:compute_kl_mean` | 两个高斯的解析 KL | KL 散度 | ★★ |
| 13 | `ex3_ppo.py:compute_surrogate_loss` | ratio / clip / min / 负号 | PPO-Clip 目标 | ★★ |
| 14 | `ex3_ppo.py:compute_value_loss` | critic 的 clipped loss | 价值函数裁剪 | ★★ |
| 15 | `ex3_ppo.py:compute_entropy_loss` | 熵正则项（带负号） | 最大熵 | ★ |
| 16 | `ex3_ppo.py:update` | 串起一步更新 + lr 自适应 | 信任域/早停 | ★★ |
| 17 | `ex4_sac.py:sample_action` | 从 squashed 高斯采样 | 随机策略 | ★ |
| 18 | `ex4_sac.py:compute_critic_loss` | 双 Q + 熵项的 Bellman target | soft Bellman | ★★★ |
| 19 | `ex4_sac.py:compute_actor_loss` | α·logπ − min Q | 重参数化策略梯度 | ★★★ |
| 20 | `ex4_sac.py:compute_alpha_loss` | log_alpha 的温度损失 | 自动温度调节 | ★★ |
| 21 | `ex4_sac.py:soft_update_targets` | Polyak 平均 | 目标网络软更新 | ★ |
| 22 | `ex4_sac.py:update` | critic→actor→alpha→soft update | 三元更新循环 | ★★ |

**建议动手顺序：Ex1 → Ex2 → Ex3 → Ex4。** 前两个是"一小时级"，后两个是"一天级"。

---

## 3. 练习一：Cliff Walking 上的动态规划

### 3.1 环境与数据结构（读 `envs/grid_world.py`）

- 网格 4 行 × 12 列，`state = row * ncol + col`，共 48 个状态；动作 `0=上, 1=下, 2=左, 3=右`。
- 起点 `(3,0)`，终点 `(3,11)`，悬崖是 `(3,1)…(3,10)`。
- **转移模型是全给的**：`env.P[s][a]` 是一个列表，每项 `(prob, next_state, reward, done)`。
  - 因为会"打滑"：`prob(按意图执行) = 1 - slip_chance`，`prob(其他每个动作) = slip_chance / 3`。
  - 撞墙：`next_state` 被裁剪回原格（等于原地不动）。
  - 掉悬崖：`reward = -100, done = True`；到终点：`reward = -1, done = True`；其它每步 `-1`。
  - **终止状态自身**：`P[s][a] = [(1.0, s, 0.0, True)]`（原地、奖励 0）。这点决定了终止状态的 V 恒为 0，不会污染数值。

### 3.2 核心公式

所有你要写的东西，本质只有**一个**函数：从当前 `v` 算 `Q(s,a)`。

```
Q(s,a) = Σ_{(p, s', r, done) ∈ P[s][a]}  p * ( r + γ · v(s') · (1 - done) )
```

- `policy_evaluation`：`v_new(s) = Σ_a π(a|s) · Q(s,a)`，一直迭代到 `max_s |v_new(s) - v(s)| < theta`
- `policy_improvement`：`π_new(s) = argmax_a Q(s,a)`，并列最优的动作**平分概率**（这部分已经给你写好了）
- `value_iteration`：`v_new(s) = max_a Q(s,a)`，收敛后再从 v 提取一次贪心策略
- `get_policy`：就是 `policy_improvement` 的 Q 计算部分（不算 π 加权）

### 3.3 ‼️ 伪代码 vs 代码骨架：到底哪里"不一致"

结论：**算法没有变**。你要实现的就是"模型已知的动态规划（planning）"——不是 Q-learning/神经网络那种采样学习。README 的文字、`assets/imgs/` 里的伪代码图、`ex1_mdp.py` 的骨架，要的是同一个 policy iteration + value iteration。

但"觉得不一致"是有理由的：下面 5 处写法差异会直接影响你怎么落笔，其中第 4 处是**图本身的错误**。

**逐行对照**

| README 伪代码（`assets/imgs/*.png`） | `ex1_mdp.py` 骨架 | 判断 |
|---|---|---|
| `Initialize π(s) and V(s) arbitrarily` | `v = zeros`，`pi = uniform` | ✅ 一致（"任意"初值） |
| `while Δ > θ do:`（策略评估循环） | `while True: … if max_diff < theta: break` | ⚠️ 伪代码是笔误，见 ② |
| `v ← V(s)`；`V(s) ← r(s,π(s)) + γΣ P(s'\|s,π(s))V(s')` | 先用 `new_v` 攒一轮，循环外才 `self.v = new_v` | ⚠️ **原地更新 vs 同步更新**，见 ① |
| `π_old ← π`；`π(s) ← argmax_a [r + γΣPV]` | `policy_improvement`：算 Q → 取 max → **并列最优平分概率** | ⚠️ 确定性策略 vs 随机策略矩阵，见 ③ |
| `if π_old = π: stop; return V, π` | `np.allclose(old_pi, new_pi)` + `break` | ✅ 一致 |
| VI：`V(s) ← max_a r(s,a) + γΣ P(s'\|s,a)V(s')` | `new_v[s] = max(qsa_list)`，`qsa` 是完整 Q | ❌ **图的排版错误**，见 ④ |
| VI：`Return π(s) = argmax_a{r(s,a) + γΣ…}` | 收敛后调 `self.get_policy()` | ✅ 一致 |

**① 同步更新（Jacobi）vs 原地更新（Gauss-Seidel）——最可能卡住你的地方**

伪代码的 `V(s) ← …` 是**在同一个 V 数组上覆盖写**，所以 for 循环走到后面的状态时，用的是前面刚更新过的值（Gauss-Seidel）；骨架是先算 `new_v`，一整轮结束后才 `self.v = new_v`（Jacobi）。

两者的**不动点完全一样**，最终 V^π / V* 相同，只是迭代次数不同。**但你必须按骨架写**，两个硬理由：

- `max_diff = max(max_diff, abs(new_v[s] - self.v[s]))` 这一行已经拿 `self.v[s]` 当"旧值"了。你一旦原地写 `self.v[s]`，旧值就没了，Δ 会算错、甚至恒为 0 → 立刻"收敛"到错的解。
- 原地写法要读"已更新的 V(s')"，但骨架里只有 `self.v`（旧值）和 `new_v`（新值数组，未更新的位置还是 0），拿 `new_v[s']` 当 V(s') 会读到假 0。
- 所以规则很简单：**读 `self.v`（旧），写 `new_v`（新）**，绝不混用。

**② 伪代码的 `while Δ > θ` 是笔误**

Δ 是在循环体里才被赋值的，所以第一次判断时 Δ 还不存在（或为 0）。字面翻译成 Python 会**一次循环都不进**，直接返回全 0 的 V。正确形式是 do-while：

```text
while True:
    max_diff = 0
    ...
    if max_diff < self.theta:
        break
```

骨架给的就是这个结构。

**③ 确定性策略 π(s) vs 随机策略矩阵 π[s][a]**

伪代码为了简洁写成确定性策略 `π(s)`，评估式是 `r(s,π(s)) + γΣP V`，没有对动作求和。代码里策略是**概率矩阵**（初值均匀随机，改进后并列最优平分），所以评估必须是 `Σ_a π(s,a)·Q(s,a)`。

- 对确定性策略，两者完全等价；
- 但 π 的一行**有多个非零**时（tie 平分就会出现），照伪代码"只取 π 最大的那个动作"会漏项、算错。
- 所以：**乘上 `self.pi[s][a]` 再求和**，不要"取 argmax 动作代进公式"。

**④ VI 图里 `max` 的位置是错的**

图里 `V(s) ← max_a r(s,a) + γΣ P V` 视觉上 max 只套住了 `r(s,a)`。照字面写就成了"取最大的即时奖励，再加 bootstrap"，这是错的。正确的是 max 套住整个括号：

```text
V(s) ← max_a [ r(s,a) + γ Σ_{s'} P(s'|s,a) V(s') ]
```

同一张图最后一行 `argmax_a{r(s,a)+γΣ…}` 是带花括号的，可以互证。代码里 `new_v[s] = max(qsa_list)`，`qsa` 是完整的 Q，天然正确。

**⑤ 随机性不在伪代码里，也不该出现在你的代码里**

README 花大篇幅讲 slip，伪代码里却一个 ε 都没有。这不是矛盾：**随机性已经全部烘焙进 `env.P[s][a]` 的多条 `(p, s', r, done)` 里了**。你的 DP 只要对 P 做期望求和，**不要写采样、不要加 epsilon、不要自己判断"这次有没有打滑"**——加了等于在模型里又随机一次，结果会错。

**实操提醒**：因为 tie 平分，策略图上同一格可能画出**多个箭头**（终止格画出 4 个箭头也正常——终止状态所有 Q 都是 0）。这不是 bug，是"并列最优平分概率"的直接结果，`scripts/ex1_plot.py` 会把所有 best action 的箭头拼起来显示。

### 3.4 逐函数要点（`exercises/ex1_mdp.py`）

**① `policy_evaluation`（第 39–76 行）**

骨架已经写好了 `for s / for a / new_v[s] = sum(qsa_list)`。你只补两处：

1. 内层 `for a` 里：算 `qsa = Σ p*(r + γ·v[s'])`，然后**乘以 `self.pi[s][a]` 再 append**。
   ⚠️ **最容易错的坑**：注释第 3 步写"Sum over all actions to obtain new_v[s]"，但下一行代码 `new_v[s] = sum(qsa_list)` **已经替你求和了**。所以 `qsa_list` 里放的必须是**已乘 π 的值**，否则会重复加权或漏掉加权。
2. 循环末尾：`if max_diff < self.theta: break`（`max_diff` 每一轮已经在算了）。

其他注意：
- 用 `self.v`（旧值）算 Q，用 `new_v` 存新值——代码已经是"整轮同步更新"（Jacobi 风格），别偷偷写 `self.v[s] = ...` 做原地更新，否则 `max_diff` 的语义就不对了。
- `done=True` 时不要再 bootstrapping（乘 `1-done`）。本环境里因为终止状态 V=0，**写不写都对**，但请写对，这是标准写法，视频里也能讲。
- 收敛判据用 `< theta` 还是 `<=` 都行；`theta=1e-3, gamma=0.9` 大约几百轮内收敛。

**② `policy_improvement`（第 78–108 行）**

只补内层 Q 计算（**不加 π 权重**），append 到 `qsa_list`。后面的 `max_q` / `num_best_actions` / `np.isclose` 平分逻辑已经给你了，读懂即可。

**③ `policy_iteration`（第 110–140 行）**

主体骨架是 `old_pi = deepcopy(self.pi)` 和末尾的 `if np.allclose(old_pi, new_pi): break`。你要补中间三步：

```text
self.policy_evaluation()          # 评估到收敛
new_pi = self.policy_improvement()  # 改进（它已把结果写回 self.pi 并 return）
# 然后让 if np.allclose(...) 生效
```

⚠️ 注意 `policy_improvement` 已经在**原地修改** `self.pi` 并把它 return 出来，所以 `new_pi` 直接接返回值即可，不要再去 `deepcopy` 一次也别重复调用。如果发现死循环，先检查是不是忘了 break 或 `max_diff` 没更新。

**④ `value_iteration`（第 169–204 行）**

结构同 `policy_evaluation`，但 `new_v[s] = max(qsa_list)` 已经写好了。你补：
1. 内层 Q 计算（同 ①，但**不加 π**）；
2. 收敛判据 `if max_diff < self.theta: break`。

循环后面 `self.get_policy()` 已经写好了。

**⑤ `get_policy`（第 206–230 行）**

和 ② 完全一样的 Q 计算。**这里建议你抽一个私有方法**（例如 `_compute_q(self, s, a)`），三处复用——不改任何公开签名，评分只看行为。代码更短，也不容易三处写错两处。

### 3.5 自测与验收

```powershell
cd ethz-robot-learning-2026\hw4_reinforcement_learning
.\.venv\Scripts\Activate.ps1
python scripts\run_policy_iteration.py
python scripts\run_value_iteration.py --slip_chance 0.2
```

- 产物在 `logs/mdp/`：状态值热力图 + 最优策略箭头图，文件名形如 `policy_iteration_values_slip_0.2.png` / `policy_iteration_policy_slip_0.png`（value iteration 同理）。
- ⚠️ 脚本最后会调 `show_plots()` → `plt.show()`，会弹窗并**阻塞**，关掉窗口脚本才结束。跑批时按顺序关窗即可。
- **不看图也能先判断对错**：非终止状态的 V 应该都是负数（每步 -1），且离终点越远越负；策略图上大部分箭头朝右。如果 V 全是 0 → 内层 Q 没算进去；如果 V 爆炸到 -1e9 → `done` 没处理或收敛判据写错导致死循环。
- 两个脚本只接受 `--slip_chance` 一个参数（`gamma=0.9`、`theta=1e-3` 写死在脚本里，见 `scripts/run_*.py`），所以别想着用命令行调 γ。
- 每个 `--slip_chance` 会生成一组图（值函数 + 策略），文件名里带 slip 值，不会互相覆盖。
- 交付图片：`slip_chance = 0.0 / 0.01 / 0.2` 各跑一次两个脚本，共 6×2 张图（值 + 策略）放进视频。

### 3.6 理论题要点（视频里说）

1. **PI vs VI 的更新过程差别**：PI 是"评估到底（解出当前策略的完整 V）再贪心改进"，每次迭代内部要跑一整个内循环；VI 把评估和改进合成一步 `max`，每轮只做一次局部更新。PI 每轮贵但轮数少；VI 每轮便宜但轮数多。两者都是 Bellman 算子的不动点迭代。
2. **γ→0 / γ→1**：γ→0 只看一步奖励，agent 近似短视，V 几乎等于即时奖励；γ→1 长期回报权重变大，V 更负、更重视"别掉悬崖"，同时压缩映射的压缩因子 γ 接近 1，**收敛显著变慢**（误差界 ~ θγ/(1-γ)）。
3. **slip 增大**：最优策略从"贴着悬崖抄近路"变成"先往上走、绕远但安全"。因为随机性让每个动作都有概率把你推进悬崖，悬崖边的期望代价急剧变差 → 行为更保守。

---

## 4. 练习二：CartPole 上的 DQN

### 4.1 背景

状态 4 维连续（`[车位置, 车速, 杆角度, 角速度]`），动作 2 个离散（左推 / 右推）。每步 +1 奖励，杆倒或车出界则回合结束。表格法失效 → 用神经网络 `Q_θ(s, a)` 做函数逼近。三个支柱：**Q-learning 目标、经验回放、目标网络**。

### 4.2 逐函数要点（`exercises/ex2_dqn.py`）

**① `ReplayBuffer.add`（第 29–41 行）** — 一行。`collections.deque(maxlen=capacity)` 已经建好，`append` 一个 5 元组，顺序必须和 `sample()` 里的 `zip(*transitions)` 解包顺序一致：`(state, action, reward, next_state, done)`。

**② `QNet.forward`（第 99–111 行）** — 一行。`fc1 → ReLU → fc2`，输入 `(B, 4)`，输出 `(B, 2)`。用已 import 的 `F.relu`。

**③ `DQN.take_action`（第 153–172 行）** — ε-greedy：

```text
以概率 epsilon  →  np.random.randint(self.action_dim)   # 探索
否则            →  state 转 tensor，unsqueeze(0)，.to(device)，q_net(...).argmax().item()  # 利用
```

- `state` 是 `np.ndarray` shape `(4,)`，必须 `unsqueeze(0)` 变成 `(1,4)` 才能进网络。
- `predict_action`（第 174–188 行）已经把"贪心"那半写好了，可以直接照抄它的 tensor 处理方式。
- 返回必须是 Python `int`（训练脚本会把它存进 buffer，`sample` 里 `zip` 后要能转 `torch.long`）。

**④ `DQN.update` 里的 TD target（第 190–241 行）** — 最关键的一处：

```
q_targets = r + γ · (1 - done) · max_a' Q_target(s', a')
```

具体到代码（`states/actions/rewards/next_states/dones` 的形状已经在上面准备好了，是 `(B,1)` 或 `(B,)`）：

```text
next_q = self.target_q_net(next_states).max(dim=1)[0].view(-1, 1)   # (B,1)
q_targets = rewards + self.gamma * next_q * (1 - dones)            # (B,1)
```

- 已经在 `with torch.no_grad():` 里，**不要**再对 target 求梯度。
- `q_values = self.q_net(states).gather(1, actions)` 已给，形状 `(B,1)`；`q_targets` 也必须是 `(B,1)` 才能和它算 MSE。少了 `.view(-1,1)` 会广播成 `(B,B)`——这是最经典的静默 bug，loss 不会报错但学不出来。
- `dones` 是 float 的 0/1，`(1 - dones)` 就是在终止时掐掉 bootstrap。

**⑤ `ex2_dqn_config.py`：超参**

只允许调 `lr`、`epsilon`、`target_update`、`hidden_dim`（其余固定：`gamma=0.99`, `num_episodes=500`, `buffer_size=10000`, `minimal_size=500`, `batch_size=64`, `seed=0`）。建议做法：

| 超参 | 建议搜索范围 | 影响 |
|---|---|---|
| `lr` | 5e-4 ~ 2e-3 | 太大 loss 震荡/崩，太小 500 回合学不完 |
| `epsilon` | 0.01 ~ 0.1 | 纯 ε-greedy 常数探索；太大回报上不去，太小容易卡在次优 |
| `target_update` | 10 ~ 100（单位：**梯度步**，见 `self.count`） | 太频繁 ≈ 没目标网络（不稳）；太慢 ≈ 目标过时（学得慢） |
| `hidden_dim` | 64 / 128 / 256 | 太小欠拟合，CartPole 用 64–128 通常够 |

⚠️ 默认给的 `epsilon=0.03, target_update=10, hidden_dim=128, lr=1e-3` 是**能跑通的起点**，但题目要求你"调优"并展示对比。做法：改一组跑一次，把 `logs/dqn/results/dqn_training_curve.png` 存成不同文件名，最后在视频/PDF 里放 2–3 条曲线对比。**别只放一条曲线就说是调优结果。**

### 4.3 自测与验收

```powershell
python scripts\train_dqn.py       # 约几分钟（CPU）
python scripts\eval_dqn.py        # 打印评估摘要
python scripts\eval_dqn.py --play # 打开窗口看策略（和 --record_video 互斥）
```

- 训练中每 10 回合打印一次最近 10 回合平均回报。**500 回合后平均回报应该稳定在 300~500**（满 500）；如果一直在 20 上下，先查 `.view(-1,1)`、`done`、以及 buffer 里存的顺序。
- `logs/dqn/models/dqn_cartpole.pth` + `logs/dqn/results/dqn_training_curve.png` 是交付图。
- `eval_dqn.py` 默认找 `logs/dqn/models/dqn_cartpole.pth`，默认跑 50 回合；可以用 `--model_path`、`--num_episodes` 覆盖。PPO/SAC 的 eval 脚本会自动选最新 run 里 iteration 最大的 checkpoint。
- 交付 PDF：训练曲线 + 评估摘要（截图/文本）。

### 4.4 理论题要点

1. **回放为什么重要**：连续样本高度相关，直接训会破坏 i.i.d. 假设、导致灾难性遗忘/震荡；回放池把旧数据打散重用，既降低相关性又提高样本效率（每条 transition 被多次采样）。
2. **目标网络的作用**：TD target 里含 `Q_target`，如果和在线网络是同一个网络，目标会随参数移动 → "追自己的尾巴"，容易振荡甚至发散。目标网络隔 C 步才同步，给出一个**短期固定**的回归目标，训练更稳。
3. **Double DQN**：标准 DQN 的 `max` 会用同一个网络既选动作又估值，`max` 对噪声正偏 → 系统性高估。Double DQN 用**在线网络选 argmax 动作**、用**目标网络给该动作估值**，把"选择"和"评估"解耦，降低过估计。

---

## 5. 练习三：SO100 上的 PPO（连续控制，on-policy）

### 5.1 背景

- 环境：`SO100RLEnv`，观测 19 维（6 关节角 + 3 末端位置 + 4 四元数 + 3 目标位置 + 3 位置误差），动作 6 维连续，范围 `[-1,1]`，由 `process_action` 线性映射到关节角范围。
- 策略：`GaussianActor` —— 网络输出均值 `mu`，`log_std` 是**可学习的独立参数**（与状态无关），`std = exp(log_std)`。
- **GAE 和回报已经由 `RolloutBuffer.compute_returns()` 算好了**（含 advantage 归一化）。你不需要实现 GAE。
- 训练脚本每轮：采 `n_steps=2048` 步 → `compute_returns` → `update()`（10 个 epoch × 2 个 mini-batch = 20 次梯度步）→ 评估 5 回合。
- UTD（更新/数据比）≈ `20 / 2048 ≈ 0.01`，非常 on-policy。

### 5.2 逐函数要点（`exercises/ex3_ppo.py`）

**① `select_action`（第 68–100 行）** — 一次返回 6 个值，别搞错顺序：

| 返回 | 怎么来 | 用途 |
|---|---|---|
| `action` | `self.actor.act(obs)` 的采样结果 | 存进 buffer 当 `act`（**原始未裁剪**动作） |
| `action_clipped` | `torch.clamp(action, -1, 1)` | 给环境 step |
| `value` | `self.critic(obs)` | 存进 buffer 当 `val` |
| `action_log_prob` | `self.actor.get_actions_log_prob(action)` | 存进 buffer 当 `logp`（旧策略 logp） |
| `action_mu` | `self.actor.action_mean` | 存 mu（算 KL 用） |
| `action_std` | `self.actor.action_std` | 存 std（算 KL 用） |

要点：
- 外面已经套了 `with torch.inference_mode():`，**不要**在函数里再造图；把 `value` 取成 `float`（`.item()`）或 0 维 tensor 都行。
- **`action` 存原始、`action_clipped` 进环境**。因为后面 `update` 里用 buffer 里的 `act` 重算 `logp`，必须和当初采样的那个动作分布一致；如果存了裁剪后的动作，比例 `ratio` 就错了。
- `mu`/`std` 形状 `(1, 6)`，`buffer.store` 会自动填进 `(6,)` 的行，没问题。

**② `compute_kl_mean`（第 109–136 行）** — 两个对角高斯之间的解析 KL：

```
KL(N(mu_old, std_old) || N(mu, std))
  = log(std / std_old) + (std_old² + (mu_old - mu)²) / (2·std²) - 0.5
```

逐维算 → 对动作维求和（`dim=-1`）→ 对 batch 取平均，返回标量。

- 注意方向：问的是"新旧策略之间"的 KL，按 hint 的写法即可；KL 本身非负，自测时把"新旧设成同一组 mu/std"，结果应 ≈ 0。
- 返回**0 维 tensor**（签名是 tensor），可以用 `.detach()` 切断图。

**③ `compute_surrogate_loss`（第 150–173 行）** — PPO 的核心：

```
ratio         = exp(logp_new - logp_old)
clipped_ratio = clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
objective     = min(ratio · adv, clipped_ratio · adv)
loss          = - mean(objective)          # 因为我们要最大化 objective
return coeff * loss
```

- **负号**最容易漏；`min` 在"优势为负"时起到"不允许把概率降得太狠"的保护作用。
- `logp_old` 来自 buffer（第 244 行 `old_logp_batch`），`logp_new` 是第 252 行用当前 actor 重算的。ratio 的意义是重要性采样权重。

**④ `compute_value_loss`（第 175–193 行）** — 带裁剪的 critic 损失：

```
unclipped = (val - ret)²
val_clipped = old_val + clamp(val - old_val, -c, c)
loss = max(unclipped, (val_clipped - ret)²).mean()
return value_loss_coeff * loss
```

- `old_val` 是 buffer 里的 `val`（采集时的 V）。裁剪的意义和 PPO 的 clip 一致：不让 critic 一步走太远。
- `max` 表示"取更悲观/更大的那个"，保证不被裁剪项占便宜。

**⑤ `compute_entropy_loss`（第 195–201 行）** — 一行，**带负号**：

```
-entropy_coeff * entropy_batch.mean()
```

因为 PPO 要**最大化**熵，而优化器是最小化 loss，所以是负的。`config` 里 `entropy_coeff=0.005`。

**⑥ `update` 的一步（第 225–298 行）** — 按顺序串起来：

```text
kl = self.compute_kl_mean(old_mu, old_std, mu, std)
self.learning_rate = self.adjust_learning_rate(kl, self.learning_rate)   # 已给你
for pg in optimizer.param_groups: pg["lr"] = self.learning_rate
surrogate_loss = ...
value_loss     = ...
entropy_loss   = ...
loss = surrogate_loss + value_loss + entropy_loss
# 后面 zero_grad / backward / clip_grad_norm_ / step 已经写好了
```

⚠️ **大坑**：`compute_value_loss` 和 `compute_surrogate_loss` 内部**已经乘过各自的 coeff**，`compute_entropy_loss` 也已经带了负号和 coeff。所以 `loss` 就是**三者直接相加**，千万别再统一乘一遍 coeff。
⚠️ `adjust_learning_rate` 比较用的是 `kl`，如果 `kl` 是 tensor，`if kl > ...` 也能工作，但建议在调用前 `.item()`，日志更干净。
⚠️ `mean_kl += kl` 时如果是带梯度的 tensor，整个 update 的图会被统计变量挂着；建议 `kl.detach()`。

### 5.3 自测与验收

```powershell
python scripts\train_ppo.py                       # 500 轮，CPU 上约几十分钟
tensorboard --logdir logs/ppo                     # 看曲线
python scripts\eval_ppo.py                        # 打印评估摘要
python scripts\eval_ppo.py --play                 # 可视化回放
```

- **先做"形状/数值"自测再跑长训练**：写个 10 行小脚本，构造 `PPOAgent(obs_dim=19, act_dim=6, hidden_sizes=[64,64])`，随机 `obs`：
  - `select_action` 的 6 个返回值形状对（`(1,6)`, `(1,6)`, `()`, `()`, `(1,6)`, `(1,6)`）；
  - 同一组 `mu/std` 传进两次 `compute_kl_mean`，结果 ≈ 0；
  - `ratio=1` 时 surrogate = `-adv.mean()`；
  - 三项 loss 都是 0 维 tensor 且 `requires_grad=True`。
- 训练曲线看这几个量：`train/mean_kl` 应保持在 `target_kl=0.01` 量级上下（持续暴涨说明 clip/lr 有问题）；`eval/return` 上升；`eval/ee_tracking_error` 下降；`train/global_action_std` 应逐渐变小（策略变确定）。
- 评估摘要参考（README 给的例子）：`Mean return ≈ 60`，`Mean tracking error ≈ 0.015`。交付 PDF 放 TensorBoard 曲线 + 评估摘要。

### 5.4 理论题要点

1. **为什么用 clip 而不是像 TRPO 那样约束 KL**：TRPO 要解带约束的优化（二阶信息/共轭梯度），实现复杂且每次更新都要重算 Fisher 矩阵；PPO-Clip 只是一阶目标，用 `min` + `clamp` 做一个"软信任域"，实现简单、可多次复用数据，实践中效果相当。**完全去掉 clip**：早期步长过大会让策略一步跳出好区域，性能断崖式下跌且很难恢复（ratio 会引爆梯度）。
2. **为什么不能拿旧 rollout 反复多做几步梯度**：PPO 是 on-policy，旧数据是旧策略采的，离策略越远，重要性采样 ratio 偏离 1 越多，梯度估计方差暴涨/有偏，等于偷偷变成 off-policy 又没做修正。多 epoch 复用是有限度的，所以有 `target_kl` 早停/降 lr。
3. **GAE 的 λ 控制什么**：偏差-方差权衡。`λ=0` 退化成单步 TD 误差（低方差、高偏差、依赖 critic）；`λ=1` 退化成蒙特卡洛回报（无偏、高方差、依赖完整轨迹）。`λ=0.95` 是常用折中。

---

## 6. 练习四：SO100 上的 SAC（连续控制，off-policy）

### 6.1 背景

SAC = 双 Q critic + 随机策略 actor + 最大熵 + 自动温度 α。目标不是最大化回报，而是最大化 `Σ (r + α·H[π(·|s)])`。

三个目标函数（`ex4_sac.py` 顶部的 docstring 已经写好了，直接对照）：

```
Critic target:  y = r + γ(1-done)[ min(Q1_t(s',a'), Q2_t(s',a')) - α·log π(a'|s') ]
Actor objective: J_π = E[ α·log π(a|s) - min(Q1(s,a), Q2(s,a)) ]
Temperature:    J_α = E[ -log_α · ( log π(a|s) + target_entropy ) ]
```

**已经给你的东西**（别重复实现）：`SquashedGaussianActor.act()` 内含 `rsample` + `tanh` + **log-prob 的 tanh 雅可比修正**；`DoubleQNet` 同时输出 q1/q2；`critic_target` 已冻结梯度；`ReplayBuffer` 已就绪；`train_sac.py` 已经在 `train_freq=500` 步后跑 `gradient_steps=200` 次 `update`。

### 6.2 逐函数要点（`exercises/ex4_sac.py`）

**① `sample_action`（第 123–138 行）** — 一行：`self.actor.act(obs)` 返回 `(action, log_prob)`，环境交互只要 `action`。外面已套 `torch.no_grad()`。

**② `compute_critic_loss`（第 146–183 行）** — 最核心，按 5 步来：

```text
# 已经在 with torch.no_grad(): 里
next_action, next_logp = self.actor.act(next_obs)          # 用"当前 actor"采下一个动作
q1_next, q2_next      = self.critic_target(next_obs, next_action)
q_next  = torch.min(q1_next, q2_next)                       # 双 Q 取 min，抑制过估计
q_next  = q_next - self.alpha * next_logp                   # 熵项（注意 logp 是求和后的标量）
target_q = rew + self.gamma * (1 - done) * q_next           # 形状要对：rew/done 是 (B,)

q1_pred, q2_pred = self.critic(obs, act)                    # 这里需要梯度！
critic_loss = 0.5 * (F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q))
```

坑：
- `rew` / `done` 在 `ReplayBatch` 里是 `(B,)` 而 q 是 `(B,)`，所以这里**不需要** `.view(-1,1)`（和 DQN 不同，别照抄）。
- `next_action` 必须是**当前 actor 采样**的，不是 buffer 里的 `act`（那是旧策略的动作）。
- `α` 在 target 里就是一个标量，直接用 `self.alpha` 即可（在 `no_grad` 里，无梯度问题）。
- `logp` 已经是对动作维**求和**过的 `(B,)`，不要再 `.sum()`。

**③ `compute_actor_loss`（第 185–211 行）**：

```text
q1_new, q2_new = self.critic(obs, act_new)
q_new   = torch.min(q1_new, q2_new)
actor_loss = (self.alpha.detach() * logp_new - q_new).mean()
```

坑：
- `act_new, logp_new` 由 `update` 里现采（第 286 行），**不能用 buffer 里的动作**，否则梯度传不到 actor。
- `self.alpha` 在这里要 `.detach()`：α 由自己的优化器更新，不该让 actor loss 给 `log_alpha` 累梯度（虽然不 step 它，但语义上要切断）。
- 注意**函数内部**只算 loss，不 step。

**④ `compute_alpha_loss`（第 213–235 行）**：

```text
alpha_loss = -(self.log_alpha * (logp_new + self.target_entropy).detach()).mean()
```

坑：
- 用 `log_alpha` 而不是 `alpha`（对 log 空间求导才稳定，且保证 α>0）。
- `(logp_new + target_entropy)` 要 `.detach()`，否则会顺着 logp 把梯度传回 actor。
- `target_entropy` 默认 `-act_dim = -6`（`__init__` 里已处理 `None`）。

**⑤ `soft_update_targets`（第 237–251 行）** — Polyak：

```text
target_param ← (1 - tau) · target_param + tau · param
```

`zip(self.critic_target.parameters(), self.critic.parameters())` 的顺序已经保证一一对应（两边结构相同）。用 `target_param.data.copy_(...)` 或 `with torch.no_grad(): ... .copy_()`。

**⑥ `update`（第 253–306 行）** — 顺序不能乱：

1. `critic_loss = self.compute_critic_loss(obs, act, rew, next_obs, done)` → zero_grad/backward/step（已给）
2. `act_new, logp_new = self.actor.act(obs)` ← **必须重新采样一次**（不是复用 buffer 的 `act`）
3. `actor_loss = self.compute_actor_loss(obs, act_new, logp_new)` → step actor（已给）
4. `alpha_loss = self.compute_alpha_loss(logp_new)` → step alpha（已给）
5. 调用 `self.soft_update_targets()`  ← 第 298 行那个 TODO

坑：
- 第 4 步复用第 2 步的 `logp_new` 是对的（同一次前向），α 那边会自己 detach。
- 别忘了第 5 步：漏掉 soft update 的话 target 永远是最初的随机网络，训练会"看起来在学但学不动"。
- `eval_mode()` 会把 `log_alpha.requires_grad` 设为 False，`train_mode()` 再打开——`train_sac.py` 每次循环都会调用 `train_mode()`，不用你管；但你自己写自测脚本时要注意别在 eval 模式里 update。

### 6.3 自测与验收

```powershell
python scripts\train_sac.py        # 这台机器是 CPU-only，约 5–6 小时，务必后台跑
tensorboard --logdir logs/sac
python scripts\eval_sac.py
python scripts\eval_sac.py --play
```

- **先跑冒烟测试再跑长训练**：临时把 `ex4_sac_config.py` 的 `total_iterations` 改成 3、`gradient_steps` 改成 5、`eval_freq` 改成 512，跑通不报错再改回来（题目说"其余参数不要改"，所以你**调完必须复原**）。
- 曲线关注：`train/alpha`（初值 0.2，应随熵变化而变，通常先升或先降后收敛）、`train/critic_loss`（量级应在个位数~几十徘徊，爆炸说明 target 写错）、`eval/return` 上升、`eval/ee_tracking_error` 下降。
- 评估摘要格式和 PPO 一样，交付 PDF 放曲线 + 摘要。

### 6.4 理论题要点

1. **熵奖励的好处**：鼓励探索、避免策略过早坍缩成确定性策略；在多峰（multimodal）奖励或需要持续探索的场景更鲁棒；理论上等价于在最大熵框架下做 RL，有更好的探索/稳定性。
2. **为什么 tanh 压缩需要修正 log-prob**：动作 `a = tanh(u)` 是变量的非线性变换，概率密度要乘雅可比 `|da/du| = (1 - tanh(u)²)`，即 `log π(a) = log μ(u) - Σ log(1 - tanh(u)²)`。不修正的话 log-prob 不是 `[-1,1]` 上动作的合法密度，熵项/策略梯度都是错的。**代码里 `SquashedGaussianActor` 已经修正过了，你不要再乘一遍**（再乘一次就变成双重修正）。
3. **α 的行为**：熵**高于**目标熵 → `J_α` 促使 α 下降（降低探索权重）；熵**低于**目标熵 → α 上升（加大熵项，逼它多探索）。所以 α 是"熵的自动控制器"。
4. **UTD（更新/数据比）**：PPO = `n_epochs × (n_steps / mini_batch_size) / n_steps = 10 × 2 / 2048 ≈ 0.01`；SAC = `gradient_steps / train_freq = 200 / 500 = 0.4`。**SAC 的 UTD 是 PPO 的约 40 倍**，样本复用率天差地别，这正是 off-policy 的样本效率优势（但也带来更大的过拟合/不稳定风险）。
5. **on vs off-policy 优缺点**：on-policy（PPO）——数据必须新鲜、样本效率低、但稳定、无分布偏移问题、实现/调参相对省心；off-policy（SAC）——可重复利用历史数据、样本效率高、可离线训练，但需要处理分布偏移（目标网络、双 Q、熵正则），对超参和实现细节更敏感。

---

## 7. 久没写代码专用：API 速查

### PyTorch（你会在 Ex2/3/4 反复用）

| 需求 | 写法 |
|---|---|
| numpy → tensor | `torch.tensor(x, dtype=torch.float32)` 或 `torch.as_tensor(...)` |
| 加 batch 维 | `x.unsqueeze(0)` → `(1, D)` |
| 丢 batch 维 | `x.squeeze(0)` / `.item()` 取标量 |
| 换设备 | `.to(self.device)` |
| 按动作索引取 Q | `q_values.gather(1, actions.long())` |
| 求 max/min | `x.max(dim=1)[0]`（返回 namedtuple，要 `[0]`）、`torch.min(a, b)` |
| 切断梯度 | `x.detach()`、`with torch.no_grad():`、`@torch.inference_mode()` |
| MSE | `torch.nn.functional.mse_loss(pred, target)`（默认已经对 batch 求平均） |
| clamp | `torch.clamp(x, -1.0, 1.0)` |
| exp/log | `torch.exp`、`torch.log(x + 1e-6)` 防 log(0) |
| 梯度裁剪 | `torch.nn.utils.clip_grad_norm_(params, max_norm)` |
| 零梯度 / 反向 / 步进 | `opt.zero_grad()` → `loss.backward()` → `opt.step()` |
| 改学习率 | `for pg in opt.param_groups: pg["lr"] = new_lr` |
| 网络定义 | `class Net(torch.nn.Module)` + `super().__init__()` + `def forward(self, x)` |

### NumPy（Ex1 用）

`np.zeros / np.zeros_like`、`np.isclose(a, b)`（浮点相等）、`np.allclose(A, B)`（比较策略是否变了）、`np.random.random()`、`np.random.randint(n)`、`np.argsort` / `np.argmax`。

---

## 8. 建议动手顺序与时间预算

| 阶段 | 内容 | 预估 |
|---|---|---|
| 0 | 读 `rl/networks.py`、`rl/buffers.py`、`envs/grid_world.py` | 30 min |
| 1 | Ex1 五个函数（本质是一个 Q 函数公式写 3 遍） + 跑图 | 1–1.5 h |
| 2 | Ex2 四个 TODO + 调参跑 2–3 组 | 2–3 h（含训练等待） |
| 3 | Ex3 六个 TODO + 冒烟自测 + 长训练 | 半天（训练几十分钟） |
| 4 | Ex4 六个 TODO + 冒烟自测 + 后台长训练 | 半天到一天（训练 5–6h，可挂机） |
| 5 | 录视频（每练习 1 分钟 + Ex4 2 分钟）+ 3 份 PDF | 2–3 h |

**并行化技巧**：Ex4 训练挂后台后，去写 Ex3 的文档/曲线截图；Ex1 的 6 张图早点跑完，视频素材就有了。

---

## 9. 跑训练的命令与产物（速查）

```powershell
cd ethz-robot-learning-2026\hw4_reinforcement_learning
.\.venv\Scripts\Activate.ps1     # 环境已经装好（torch 2.15.0+cpu）

# Ex1
python scripts\run_policy_iteration.py --slip_chance 0.2
python scripts\run_value_iteration.py --slip_chance 0.2
# → logs/mdp/

# Ex2
python scripts\train_dqn.py            # → logs/dqn/{models,results}
python scripts\eval_dqn.py [--play | --record_video]

# Ex3
python scripts\train_ppo.py            # → logs/ppo/<时间戳>_model/ (ckpt + tb events)
python scripts\eval_ppo.py [--play]

# Ex4
python scripts\train_sac.py            # → logs/sac/<时间戳>_model/
python scripts\eval_sac.py [--play]

# 看曲线
tensorboard --logdir logs
```

> 注意：这台机器的 `.venv` 里是 **CPU 版 torch**（`torch.cuda.is_available() == False`）。三个脚本都会自动 print `Using device: cpu`，不用改代码。SAC 的 5–6 小时就是 CPU 上的量级；要省时间可以把训练放后台/过夜。

---

## 10. 常见报错速查

| 现象 / 报错 | 原因 |
|---|---|
| `NotImplementedError` 出现在你没改的行 | 还有 TODO 没填；按 traceback 行号定位 |
| loss 不降、回报一直在 20 | `q_targets` 少了 `.view(-1,1)` 被广播；或 `done` 没掐 bootstrap |
| `RuntimeError: shape ... cannot be broadcast` | 存进 buffer 的 action 是 tensor 不是 int，或 batch 维没加/多加 |
| `ValueError: RolloutBuffer is full` | 你改了 `n_steps` 配置但 buffer 没重新对齐 |
| PPO 训着训着 loss 变 NaN | lr 太大 / clip 漏了负号 / `log(0)` 没加 eps / 熵项符号反了 |
| SAC 的 alpha 变成 0 或爆炸 | alpha loss 用错了系数或没 detach；`target_entropy` 符号反了 |
| `eval_*.py --play` 打不开窗口 | 和 `--record_video` 互斥；或 MuJoCo 渲染环境问题（Linux 用 `mjpython`） |
| Windows 上路径报错 | 脚本都用 `pathlib`，从 `hw4_reinforcement_learning` 目录里跑就行，别在别的 cwd 跑 |

---

## 11. 提交前 checklist

- [ ] `exercises/ex1_mdp.py` 7 处 TODO 全填，两个脚本都能跑出图
- [ ] `exercises/ex2_dqn.py` 4 处 TODO + `ex2_dqn_config.py` 4 个超参（并说明调优依据）
- [ ] `exercises/ex3_ppo.py` 6 处 TODO，`train_ppo.py` 能跑、曲线不 NaN
- [ ] `exercises/ex4_sac.py` 6 处 TODO，`train_sac.py` 能跑
- [ ] **没有改任何函数签名、没有重命名文件**
- [ ] 调试期间临时改的 config（如 `total_iterations`）**已复原**
- [ ] Ex1 的 6 张图（3 个 slip × 2 个算法）都进了视频
- [ ] Ex2/3/4 各一份 PDF（曲线 + 评估摘要）
- [ ] 视频 ≤ 5 分钟，结果**显示在屏幕上**，理论题都回答了，没有倍速
- [ ] 视频/音频都检查过
