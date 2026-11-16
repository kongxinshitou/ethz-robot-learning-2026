"""HW4 训练耗时基准（只读作业代码，不修改任何作业文件）。

在 ethz-robot-learning-2026/hw4_reinforcement_learning 目录下运行。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HW4 = Path(__file__).resolve().parent.parent / "ethz-robot-learning-2026" / "hw4_reinforcement_learning"
sys.path.insert(0, str(HW4))

from envs.cartpole_wrapper import CartPoleWrapper  # noqa: E402
from envs.so100_rl_env import SO100RLEnv  # noqa: E402
from rl.networks import (  # noqa: E402
    DoubleQNet, GaussianActor, SquashedGaussianActor, ValueNet, build_mlp,
)

torch.manual_seed(0)
DEV = torch.device("cpu")
OBS, ACT = 19, 6
HIDDEN = [256, 128, 128]


def timeit(fn, n: int) -> float:
    """返回 n 次调用的平均秒数（先 warmup 1 次）。"""
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main() -> None:
    print(f"threads={torch.get_num_threads()} cuda={torch.cuda.is_available()}")

    # ── A. 环境 step 成本 ───────────────────────────────────────────────
    print("\n=== A. 环境 step 成本（这是 RL 的主要成本）===")
    so_env = SO100RLEnv(xml_path=HW4 / "assets" / "mujoco" / "so100_pos_ctrl.xml", render_mode=None)
    obs, _ = so_env.reset()
    a0 = np.zeros(ACT, dtype=np.float32)
    t_so = timeit(lambda: so_env.step(a0), 100)
    print(f"  SO100RLEnv.step: {t_so*1000:.2f} ms/步（episode 长 {so_env.max_episode_length} 步 = 3 s 仿真）")
    print(f"  1,024,000 步（PPO/SAC 默认预算）≈ {t_so*1_024_000/60:.1f} 分钟纯环境时间")

    cp = CartPoleWrapper(seed=0)
    cp.reset()

    def cp_step() -> None:
        s, r, d, _ = cp.step(0)
        if d:
            cp.reset()

    t_cp = timeit(cp_step, 2000)
    print(f"  CartPole.step: {t_cp*1000:.3f} ms/步")

    # ── B. PPO ─────────────────────────────────────────────────────────
    print("\n=== B. PPO（配置：500 iterations x 2048 steps，batch 1024，10 epochs）===")
    actor = GaussianActor(OBS, ACT, HIDDEN)
    critic = ValueNet(OBS, HIDDEN)
    opt = torch.optim.AdamW(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)

    s1 = torch.randn(1024, OBS)
    act_b = torch.randn(1024, ACT)
    adv = torch.randn(1024)
    ret = torch.randn(1024)
    logp_old = torch.randn(1024) * 0.1

    def ppo_update_step() -> None:
        mu = actor.mu_net(s1)
        std = torch.exp(actor.log_std).expand_as(mu)
        logp = torch.distributions.Normal(mu, std).log_prob(act_b).sum(-1)
        ratio = torch.exp(logp - logp_old)
        surr = -(torch.minimum(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv)).mean()
        v = critic(s1)
        vloss = F.mse_loss(v, ret)
        loss = surr + 0.01 * vloss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    t_upd1 = timeit(ppo_update_step, 20)
    n_upd_per_iter = 10 * (2048 // 1024)  # n_epochs * minibatches
    t_update = t_upd1 * n_upd_per_iter

    obs1 = torch.randn(1, OBS)

    def ppo_select() -> None:
        with torch.no_grad():
            actor.act(obs1)
            critic(obs1)

    t_sel = timeit(ppo_select, 1000)

    def ppo_eval_ep() -> None:
        so_env.reset()
        for _ in range(so_env.max_episode_length):
            with torch.no_grad():
                a = actor.act_inference(obs1)
            so_env.step(a.squeeze(0).numpy())

    t_eval_ep = timeit(ppo_eval_ep, 3)

    rollout_s = 2048 * (t_so + t_sel)
    eval_s = 5 * t_eval_ep
    per_iter = rollout_s + t_update + eval_s
    print(f"  单次梯度更新 {t_upd1*1000:.2f} ms x {n_upd_per_iter} = {t_update:.2f} s/iteration")
    print(f"  单步动作选择 {t_sel*1000:.2f} ms；rollout 2048 步 = {rollout_s:.2f} s")
    print(f"  评估 5 条 = {eval_s:.2f} s")
    print(f"  -> 每个 iteration ≈ {per_iter:.2f} s；500 iterations ≈ {per_iter*500/60:.1f} 分钟")

    # ── C. SAC ─────────────────────────────────────────────────────────
    print("\n=== C. SAC（配置：2048 iterations x 500 steps，200 梯度步/iter，batch 1024）===")
    sac_actor = SquashedGaussianActor(OBS, ACT, HIDDEN)
    dq = DoubleQNet(OBS, ACT, HIDDEN)
    dq_target = DoubleQNet(OBS, ACT, HIDDEN)
    dq_target.load_state_dict(dq.state_dict())
    opt_sac = torch.optim.AdamW(list(sac_actor.parameters()) + list(dq.parameters()), lr=3e-4)

    obs_b = torch.randn(1024, OBS)
    nobs_b = torch.randn(1024, OBS)
    act_b2 = torch.rand(1024, ACT) * 2 - 1
    rew = torch.randn(1024)
    done = torch.zeros(1024)
    alpha = torch.tensor(0.2, requires_grad=True)
    log_alpha_opt = torch.optim.AdamW([alpha], lr=3e-4)

    def sac_grad_step() -> None:
        with torch.no_grad():
            a2, logp2 = sac_actor.act(nobs_b)
            q1t, q2t = dq_target(nobs_b, a2)
            target = rew + 0.99 * (1 - done) * (torch.minimum(q1t, q2t) - alpha * logp2)
        q1, q2 = dq(obs_b, act_b2)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        opt_sac.zero_grad(set_to_none=True)
        critic_loss.backward(retain_graph=False)
        opt_sac.step()

        a, logp = sac_actor.act(obs_b)
        q1a, _ = dq.q1(obs_b, a), None
        actor_loss = (alpha.detach() * logp - q1a).mean()
        opt_sac.zero_grad(set_to_none=True)
        actor_loss.backward()
        opt_sac.step()

    t_sac_step = timeit(sac_grad_step, 10)

    def sac_sample() -> None:
        with torch.no_grad():
            sac_actor.act(obs1)

    t_sac_act = timeit(sac_sample, 1000)

    steps = 2048 * 500
    n_grad = 2048 * 200
    t_env_total = steps * (t_so + t_sac_act)
    t_grad_total = n_grad * t_sac_step
    t_eval_sac = (steps // 2048) * 5 * t_eval_ep
    total_sac = t_env_total + t_grad_total + t_eval_sac
    print(f"  单次 SAC 梯度步（critic+actor）: {t_sac_step*1000:.2f} ms x {n_grad:,} = {t_grad_total/60:.1f} 分钟")
    print(f"  环境+采样: {steps:,} 步 x {(t_so+t_sac_act)*1000:.2f} ms = {t_env_total/60:.1f} 分钟")
    print(f"  评估: {t_eval_sac/60:.1f} 分钟")
    print(f"  -> SAC 总计 ≈ {total_sac/60:.1f} 分钟")

    # ── D. DQN ─────────────────────────────────────────────────────────
    print("\n=== D. DQN（CartPole，500 episodes，batch 64，hidden 128）===")
    qnet = build_mlp(4, [128, 128], 2)
    qnet_target = build_mlp(4, [128, 128], 2)
    qnet_target.load_state_dict(qnet.state_dict())
    opt_dqn = torch.optim.AdamW(qnet.parameters(), lr=1e-3)
    st = torch.randn(64, 4)
    ac = torch.randint(0, 2, (64, 1))
    rw = torch.randn(64)
    nst = torch.randn(64, 4)
    dn = torch.zeros(64)

    def dqn_update() -> None:
        q = qnet(st).gather(1, ac).squeeze(-1)
        with torch.no_grad():
            td = rw + 0.99 * (1 - dn) * qnet_target(nst).max(1).values
        loss = F.mse_loss(q, td)
        opt_dqn.zero_grad(set_to_none=True)
        loss.backward()
        opt_dqn.step()

    def dqn_act() -> None:
        with torch.no_grad():
            qnet(torch.randn(1, 4)).argmax(dim=1)

    t_dqn_upd = timeit(dqn_update, 50)
    t_dqn_act = timeit(dqn_act, 2000)
    per_step = t_cp + t_dqn_act + t_dqn_upd  # 每步都 update（minimal_size 之后）
    print(f"  update {t_dqn_upd*1000:.3f} ms + act {t_dqn_act*1000:.3f} ms + env {t_cp*1000:.3f} ms"
          f" = {per_step*1000:.2f} ms/步")
    for n in (50_000, 100_000, 200_000):
        print(f"  {n:,} 步 ≈ {per_step*n/60:.1f} 分钟")


if __name__ == "__main__":
    main()
