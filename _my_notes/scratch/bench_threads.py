"""SAC/PPO 更新步在不同线程数下的耗时（判断 10 线程是否是瓶颈）。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HW4 = Path(__file__).resolve().parent.parent / "ethz-robot-learning-2026" / "hw4_reinforcement_learning"
sys.path.insert(0, str(HW4))

from rl.networks import DoubleQNet, GaussianActor, SquashedGaussianActor, ValueNet  # noqa: E402

OBS, ACT, HIDDEN = 19, 6, [256, 128, 128]


def build():
    actor = SquashedGaussianActor(OBS, ACT, HIDDEN)
    dq = DoubleQNet(OBS, ACT, HIDDEN)
    dqt = DoubleQNet(OBS, ACT, HIDDEN)
    dqt.load_state_dict(dq.state_dict())
    opt = torch.optim.AdamW(list(actor.parameters()) + list(dq.parameters()), lr=3e-4)
    alpha = torch.tensor(0.2, requires_grad=True)
    obs = torch.randn(1024, OBS)
    nobs = torch.randn(1024, OBS)
    act = torch.rand(1024, ACT) * 2 - 1
    rew = torch.randn(1024)
    done = torch.zeros(1024)

    def step():
        with torch.no_grad():
            a2, logp2 = actor.act(nobs)
            q1t, q2t = dqt(nobs, a2)
            target = rew + 0.99 * (1 - done) * (torch.minimum(q1t, q2t) - alpha * logp2)
        q1, q2 = dq(obs, act)
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        a, logp = actor.act(obs)
        aloss = (alpha.detach() * logp - dq.q1(obs, a)).mean()
        opt.zero_grad(set_to_none=True)
        aloss.backward()
        opt.step()

    return step


def timeit(fn, n=8):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


print("SAC 梯度步（batch 1024）随线程数的变化:")
for nt in (1, 2, 4, 6, 8, 10):
    torch.set_num_threads(nt)
    step = build()
    print(f"  threads={nt:<2} -> {timeit(step)*1000:6.2f} ms/step")

torch.set_num_threads(10)
print("\nPPO 更新步（batch 1024）随线程数的变化:")
for nt in (1, 4, 10):
    torch.set_num_threads(nt)
    actor = GaussianActor(OBS, ACT, HIDDEN)
    critic = ValueNet(OBS, HIDDEN)
    opt = torch.optim.AdamW(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    s = torch.randn(1024, OBS)
    a = torch.randn(1024, ACT)
    adv = torch.randn(1024)
    ret = torch.randn(1024)
    old = torch.randn(1024) * 0.1

    def upd():
        mu = actor.mu_net(s)
        std = torch.exp(actor.log_std).expand_as(mu)
        logp = torch.distributions.Normal(mu, std).log_prob(a).sum(-1)
        ratio = torch.exp(logp - old)
        surr = -(torch.minimum(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv)).mean()
        v = critic(s)
        loss = surr + 0.01 * F.mse_loss(v, ret)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    print(f"  threads={nt:<2} -> {timeit(upd, 15)*1000:6.2f} ms/step")
