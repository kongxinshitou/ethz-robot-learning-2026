"""Self-check for HW2 Exercise 3 (PPO tracking environment).

Run from hw2_robot_control_mdps with the hw2 venv active:

    python check_ex3.py

It only *tests* your implementation in exercises/ex3.py - it does not solve it.
Checks:
  1) the five functions in isolation (shapes, ranges, formulas, frame conversion)
  2) the gym environment: reset / step / obs space / reward
  3) a short random-action rollout (no PPO needed)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# exercises/ex3.py does `import __init__`, which resolves to scripts/__init__.py
# when the scripts directory is on sys.path (same as when running scripts/train.py).
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import mujoco

from scripts import XML_PATH
from exercises.ex3 import (
    reset_robot,
    reset_target_position,
    process_action,
    compute_reward,
    get_obs,
)
from env.so100_tracking_env import SO100TrackEnv

NUM_JOINTS = 6


def section(title):
    print("\n" + "-" * 62)
    print(title)
    print("-" * 62)


def report(checks):
    ok = True
    for name, good in checks:
        print(f"  [{'PASS' if good else 'FAIL'}] {name}")
        ok &= bool(good)
    return ok


def check_reset_robot():
    section("1) reset_robot(default_qpos)")
    checks = []
    default = np.array([0.0, -1.57, 1.0, 1.0, 0.0, 0.02239])
    samples = np.stack([reset_robot(default) for _ in range(500)])
    print(f"  500 samples: shape={samples.shape}, "
          f"mean noise={np.round(samples.mean(axis=0) - default, 3)}")
    checks += [
        ("shape (6,)", samples.shape == (500, NUM_JOINTS)),
        ("noise stays within +-0.5", np.all(np.abs(samples - default) <= 0.5 + 1e-12)),
        ("noise actually applied", np.any(np.abs(samples - default) > 1e-6)),
        ("independent samples", np.std(samples, axis=0).min() > 0.1),
    ]
    return report(checks)


def check_reset_target():
    section("2) reset_target_position(base_pos)")
    checks = []
    base = np.array([0.7, -1.2, 0.3])
    samples = np.stack([reset_target_position(base) for _ in range(1000)])
    off = samples - base
    print(f"  1000 samples offset ranges: "
          f"x[{off[:,0].min():.3f},{off[:,0].max():.3f}] "
          f"y[{off[:,1].min():.3f},{off[:,1].max():.3f}] "
          f"z[{off[:,2].min():.3f},{off[:,2].max():.3f}]")
    checks += [
        ("shape (3,)", samples.shape == (1000, 3)),
        ("x offset in [0.2, 0.4]", off[:, 0].min() >= 0.2 - 1e-9 and off[:, 0].max() <= 0.4 + 1e-9),
        ("y offset in [-0.2, 0.2]", off[:, 1].min() >= -0.2 - 1e-9 and off[:, 1].max() <= 0.2 + 1e-9),
        ("z offset in [0.1, 0.4]", off[:, 2].min() >= 0.1 - 1e-9 and off[:, 2].max() <= 0.4 + 1e-9),
        ("base position is added", np.allclose(samples.mean(axis=0) - base,
                                               [0.3, 0.0, 0.25], atol=0.05)),
    ]
    return report(checks)


def check_process_action():
    section("3) process_action(action, jnt_range)")
    checks = []
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    jnt = model.jnt_range.copy()
    print("  jnt_range =", np.round(jnt, 4).tolist())
    u = process_action(np.zeros(NUM_JOINTS), jnt)
    mid = 0.5 * (jnt[:, 0] + jnt[:, 1])
    print(f"  action 0  -> {np.round(u, 4)}")
    print(f"  action +1 -> {np.round(process_action(np.ones(NUM_JOINTS), jnt), 4)}")
    print(f"  action -1 -> {np.round(process_action(-np.ones(NUM_JOINTS), jnt), 4)}")
    a = np.array([0.0, 0.5, -1.0, 1.0, -0.25, 0.0])
    print(f"  example {a} -> {np.round(process_action(a, jnt), 4)}")
    a2 = np.array([-2.0, 3.0, 0.3, -0.7, 1.4, -1.9])
    clipped = process_action(a2, jnt)
    checks += [
        ("shape (6,)", u.shape == (NUM_JOINTS,)),
        ("0 -> midpoint of each joint range", np.allclose(u, mid)),
        ("+1 -> upper limit", np.allclose(process_action(np.ones(NUM_JOINTS), jnt), jnt[:, 1])),
        ("-1 -> lower limit", np.allclose(process_action(-np.ones(NUM_JOINTS), jnt), jnt[:, 0])),
        ("output always inside jnt_range", np.all(clipped >= jnt[:, 0]) and np.all(clipped <= jnt[:, 1])),
        ("out-of-range actions are clipped", np.allclose(clipped, process_action(np.clip(a2, -1, 1), jnt))),
    ]
    return report(checks)


def check_reward():
    section("4) compute_reward(ee_tracking_error)")
    checks = []
    errors = np.array([0.0, 0.004, 0.005, 0.02, 0.05, 0.1, 0.3, 1.0])
    rewards = np.array([compute_reward(e) for e in errors])
    print("  " + "  ".join(f"e={e:.3f}->r={r:.4f}" for e, r in zip(errors, rewards)))
    dense = np.exp(-2.0 * errors)
    sparse = (errors < 0.005).astype(float)
    checks += [
        ("returns a scalar float", all(isinstance(compute_reward(e), float) for e in errors)),
        ("matches exp(-2e) + 1[e<0.005]", np.allclose(rewards, dense + sparse)),
        ("e=0 gives 2.0", abs(compute_reward(0.0) - 2.0) < 1e-12),
        ("e=0.005 gives dense only", abs(compute_reward(0.005) - np.exp(-0.01)) < 1e-12),
        ("monotonically decreasing", np.all(np.diff(rewards) <= 0)),
        ("bounded in (0, 2]", np.all(rewards > 0) and np.all(rewards <= 2.0 + 1e-12)),
    ]
    return report(checks)


def check_obs():
    section("5) get_obs(...)")
    checks = []
    qpos = np.array([0.0, -1.57, 1.0, 1.0, 0.0, 0.02239])
    ee_pos = np.array([0.3238, 0.0, 0.2587])
    ee_rot = np.eye(3)
    base_pos = np.zeros(3)
    base_rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # Rz(90 deg)
    target = np.array([0.25, -0.1, 0.2])
    obs = get_obs(qpos, ee_pos, ee_rot, base_pos, base_rot, target)
    print("  obs =", np.round(obs, 4))
    checks += [
        ("shape (16,)", obs.shape == (16,)),
        ("qpos comes first", np.allclose(obs[0:6], qpos)),
        ("EE position in base frame", np.allclose(obs[6:9], base_rot.T @ (ee_pos - base_pos))),
        ("EE quaternion is normalized", abs(np.linalg.norm(obs[9:13]) - 1.0) < 1e-9),
        ("target in base frame", np.allclose(obs[13:16], base_rot.T @ (target - base_pos))),
    ]

    # rigid-transform invariance: moving/rotating the whole world must not change obs
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3))
    R0, _ = np.linalg.qr(A)  # random rotation
    if np.linalg.det(R0) < 0:
        R0[:, 0] *= -1
    t0 = np.array([1.5, -0.7, 2.2])
    obs2 = get_obs(
        qpos,
        R0 @ ee_pos + t0,
        R0 @ ee_rot,
        R0 @ base_pos + t0,
        R0 @ base_rot,
        R0 @ target + t0,
    )
    print(f"  max |obs - obs_after_global_rigid_transform| = {np.abs(obs - obs2).max():.2e}")
    checks.append(("invariant to global translation/rotation", np.allclose(obs, obs2, atol=1e-9)))
    return report(checks)


def check_env():
    section("6) environment: reset / step / spaces")
    checks = []
    try:
        env = SO100TrackEnv(xml_path=XML_PATH, render_mode=None)
        print("  observation_space:", env.observation_space)
        print("  action_space:     ", env.action_space)
        obs, info = env.reset(seed=0)
        print(f"  reset: obs shape={obs.shape}, finite={np.isfinite(obs).all()}")
        err0 = float(np.linalg.norm(env.data.site("ee_site").xpos - env.data.mocap_pos[0]))
        print(f"  initial EE error = {err0:.4f} m")
        checks += [
            ("observation space is (16,)", env.observation_space.shape == (16,)),
            ("action space is (6,) in [-1,1]", env.action_space.shape == (6,)
             and np.allclose(env.action_space.low, -1) and np.allclose(env.action_space.high, 1)),
            ("reset returns (16,) observation", obs.shape == (16,) and np.isfinite(obs).all()),
        ]

        rng = np.random.default_rng(0)
        rewards = []
        for i in range(20):
            obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, size=6).astype(np.float32))
            rewards.append(r)
            assert obs.shape == (16,) and np.isfinite(r)
            assert 0.0 < r <= 2.0 + 1e-12
            assert "ee_tracking_error" in info
            if i < 19:
                assert not trunc
        print(f"  20 random steps: reward mean={np.mean(rewards):.4f}, "
              f"last err={info['ee_tracking_error']:.4f} m")
        checks.append(("20 random steps run without error", True))
        checks.append(("reward in (0, 2]", all(0 < r <= 2.0 + 1e-12 for r in rewards)))
        checks.append(("info contains ee_tracking_error", "ee_tracking_error" in info))

        # full episode length: 100 control steps (10 s) then truncated
        env2 = SO100TrackEnv(xml_path=XML_PATH, render_mode=None)
        env2.reset(seed=1)
        truncs, terms = [], []
        for _ in range(100):
            _, r, term, trunc, _ = env2.step(np.zeros(6, dtype=np.float32))
            truncs.append(bool(trunc))
            terms.append(bool(term))
        checks += [
            ("episode runs exactly 100 control steps", len(truncs) == 100),
            ("no early truncation before step 100", not any(truncs[:-1])),
            ("truncated at step 100", truncs[-1]),
            ("terminated is always False", not any(terms)),
        ]
    except NotImplementedError:
        print("  !! ex3 still has an unimplemented TODO")
        checks.append(("environment constructs", False))
    except Exception as exc:  # noqa: BLE001
        print(f"  !! {type(exc).__name__}: {exc}")
        checks.append(("environment constructs", False))
    return report(checks)


def main():
    ok = True
    ok &= check_reset_robot()
    ok &= check_reset_target()
    ok &= check_process_action()
    ok &= check_reward()
    ok &= check_obs()
    ok &= check_env()
    section("RESULT")
    print("  ALL CHECKS PASSED - ex3 implementation looks good, you can start training." if ok
          else "  Some checks failed - see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
