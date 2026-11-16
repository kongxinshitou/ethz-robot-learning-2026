"""Self-check for HW2 Exercise 2 (quintic spline waypoints + PID control).

Run from hw2_robot_control_mdps with the hw2 venv active:

    python check_ex2.py

It only *tests* your implementation in exercises/ex2.py - it does not solve it.
"""
import numpy as np
import mujoco

from scripts import TORQUE_CTRL_XML_PATH
from exercises.ex1 import build_keypoints, ik_track
from exercises.ex2 import generate_quintic_spline_waypoints, pid_control

NUM_WAYPOINTS = 5
DT = 0.002
KP, KI, KD = 150.0, 0.0, 0.01
# quintic time scaling evaluated at s = 0, .25, .5, .75, 1
EXPECTED_ALPHA = np.array([0.0, 0.103515625, 0.5, 0.896484375, 1.0])


def section(title):
    print("\n" + "-" * 62)
    print(title)
    print("-" * 62)


def report(checks):
    ok = True
    for name, good in checks:
        print(f"  [{'PASS' if good else 'FAIL'}] {name}")
        ok &= good
    return ok


def check_quintic():
    section("1) generate_quintic_spline_waypoints(start, end, num_points)")
    checks = []
    try:
        keypoints = build_keypoints()
        start, end = keypoints[0], keypoints[1]
        wp = np.asarray(generate_quintic_spline_waypoints(start, end, NUM_WAYPOINTS), dtype=float)
        print(f"  waypoint array shape = {wp.shape}  (expect ({NUM_WAYPOINTS}, 3))")
        if wp.shape == (NUM_WAYPOINTS, 3):
            print(f"  wp[0]  = {np.round(wp[0], 6)}   start = {np.round(start, 6)}")
            print(f"  wp[-1] = {np.round(wp[-1], 6)}   end   = {np.round(end, 6)}")
            delta = end - start
            alpha = (wp - start) @ delta / (delta @ delta)
            print(f"  progress along the segment = {np.round(alpha, 6)}")
            print(f"  expected quintic scaling   = {np.round(EXPECTED_ALPHA, 6)}")
            off_line = np.linalg.norm(np.cross(delta, wp - start), axis=1).max()
            print(f"  max distance from straight segment = {off_line:.2e} m")
            checks += [
                ("output shape is (num_points, 3)", wp.shape == (NUM_WAYPOINTS, 3)),
                ("first waypoint == start", np.allclose(wp[0], start)),
                ("last waypoint == end", np.allclose(wp[-1], end)),
                ("follows the quintic time scaling", np.allclose(alpha, EXPECTED_ALPHA, atol=1e-6)),
                ("stays on the straight segment", off_line < 1e-9),
            ]
        else:
            checks.append(("output shape is (num_points, 3)", False))

        # dense sampling: check the shape of the time-scaling curve itself
        n = 2001
        f = np.asarray(generate_quintic_spline_waypoints(np.zeros(1), np.ones(1), n),
                       dtype=float).ravel()
        print(f"\n  dense curve: length = {f.size}  (expect {n})")
        if f.size == n:
            s = np.linspace(0.0, 1.0, n)
            df = np.gradient(f, s)
            symmetry = np.abs(f + f[::-1] - 1.0).max()
            print(f"  f(0) = {f[0]:.3e}   f(1) = {f[-1]:.3e}   max |df| at ends = "
                  f"{max(abs(df[0]), abs(df[-1])):.2e}")
            checks += [
                ("dense sampling returns num_points points", True),
                ("starts at 0 and ends at 1", abs(f[0]) < 1e-12 and abs(f[-1] - 1.0) < 1e-12),
                ("zero velocity at both ends", max(abs(df[0]), abs(df[-1])) < 1e-3),
                ("monotone (no overshoot)", np.all(np.diff(f) >= -1e-12)),
                ("symmetric: s(1-t) == 1 - s(t)", symmetry < 1e-9),
            ]
        else:
            checks.append(("dense sampling returns num_points points", False))
    except NotImplementedError:
        print("  !! generate_quintic_spline_waypoints still raises NotImplementedError")
        checks.append(("implemented", False))
    except Exception as exc:  # noqa: BLE001 - report the failure, don't crash the check
        print(f"  !! {type(exc).__name__}: {exc}")
        checks.append(("callable with 3-D waypoints", False))
    return report(checks)


def check_pid():
    section("2) pid_control(tracking_error_history, timestep, Kp, Ki, Kd)")
    checks = []
    try:
        model = mujoco.MjModel.from_xml_path(str(TORQUE_CTRL_XML_PATH))
        data = mujoco.MjData(model)
        nu = model.nu
        rng = np.random.default_rng(0)
        hist = rng.uniform(-0.5, 0.5, size=(7, nu))

        u = np.asarray(pid_control(hist, DT), dtype=float)
        print(f"  {hist.shape} history -> output shape = {u.shape}  (expect ({nu},))")
        print(f"  one sample: {np.round(u, 4)}")
        checks.append(("returns one control vector per joint", u.shape == (nu,)))

        try:
            data.ctrl[:] = u
            checks.append(("result can be assigned to data.ctrl[:]", True))
        except Exception as exc:  # noqa: BLE001
            print(f"  !! data.ctrl[:] = result failed: {type(exc).__name__}: {exc}")
            checks.append(("result can be assigned to data.ctrl[:]", False))

        e = np.full(nu, 0.1)
        u1 = np.asarray(pid_control(e.reshape(1, -1), DT), dtype=float)
        u2 = np.asarray(pid_control(np.vstack([np.zeros(nu), e]), DT), dtype=float)
        print(f"\n  Kp*e                    = {np.round(u1, 4)}  (expect {KP * e[0]:.4f})")
        print(f"  step 0 -> e (P + D)     = {np.round(u2, 4)}  (expect {KP * e[0] + KD * e[0] / DT:.4f})")
        checks += [
            ("P term: one error sample, D = 0", np.allclose(u1, KP * e)),
            ("P + D terms on a step", np.allclose(u2, KP * e + KD * e / DT)),
        ]

        # derivative must be taken along time (rows), not across joints (columns)
        slopes = np.arange(1, nu + 1, dtype=float) * 0.01
        u3 = np.asarray(pid_control(np.vstack([np.zeros(nu), slopes]), DT,
                                    Kp=0.0, Ki=0.0, Kd=KD), dtype=float)
        print(f"  per-joint slopes {np.round(slopes, 3)} -> D term {np.round(u3, 4)}")
        checks.append(("D term is per joint, along time", np.allclose(u3, KD * slopes / DT)))

        # integral over the whole stored window (4 samples of e)
        u4 = np.asarray(pid_control(np.tile(e, (4, 1)), DT,
                                    Kp=0.0, Ki=100.0, Kd=0.0), dtype=float)
        print(f"  I term, 4 samples of 0.1 -> {np.round(u4, 5)}  (expect {100.0 * 4 * 0.1 * DT:.5f})")
        checks.append(("I term sums the stored history * timestep",
                       np.allclose(u4, 100.0 * 4 * 0.1 * DT)))

        # P must use the most recent row
        a, b = np.full(nu, -0.2), np.full(nu, 0.3)
        u5 = np.asarray(pid_control(np.vstack([a, b]), DT), dtype=float)
        print(f"  history [-0.2, 0.3] -> {np.round(u5, 4)}  "
              f"(expect {KP * 0.3 + KD * (0.3 + 0.2) / DT:.4f})")
        checks.append(("P uses the most recent error",
                       np.allclose(u5, KP * b + KD * (b - a) / DT)))
    except NotImplementedError:
        print("  !! pid_control still raises NotImplementedError")
        checks.append(("implemented", False))
    except Exception as exc:  # noqa: BLE001
        print(f"  !! {type(exc).__name__}: {exc}")
        checks.append(("callable on (T, 6) history", False))
    return report(checks)


def check_closed_loop():
    section("3) closed loop: quintic + IK + PID on the torque model")
    checks = []
    try:
        model = mujoco.MjModel.from_xml_path(str(TORQUE_CTRL_XML_PATH))
        data = mujoco.MjData(model)
        keypoints = build_keypoints()
        total = np.vstack([
            generate_quintic_spline_waypoints(keypoints[i], keypoints[(i + 1) % len(keypoints)],
                                             NUM_WAYPOINTS)
            for i in range(4)
        ])
        print(f"  {len(total)} waypoints, timestep = {model.opt.timestep} s")

        target = ik_track(model, data, "ee_site", total[0])
        data.qpos[:] = target
        mujoco.mj_forward(model, data)

        history = np.zeros((0, model.nu))
        max_steps = 1500
        errors, steps, wid = [], 0, 0
        while wid < len(total):
            target = ik_track(model, data, "ee_site", total[wid])
            for _ in range(max_steps):
                history = np.vstack([history, target - data.qpos.copy()])[-10:]
                data.ctrl[:] = pid_control(history, model.opt.timestep)
                mujoco.mj_step(model, data)
                steps += 1
                err = np.linalg.norm(data.site("ee_site").xpos - total[wid])
                if err < 3e-2:
                    errors.append(err)
                    history = np.zeros((0, model.nu))
                    wid += 1
                    break
            else:
                print(f"  !! stuck at waypoint {wid}: ee error = {err:.4f} m "
                      f"after {max_steps} steps ({max_steps * model.opt.timestep:.1f} s)")
                break
        reached = len(errors)
        print(f"  reached {reached}/{len(total)} waypoints in {steps} simulation steps")
        if errors:
            print(f"  error at switch: mean = {np.mean(errors):.4f} m, "
                  f"max = {np.max(errors):.4f} m  (threshold 0.03 m)")
        checks += [
            ("no exception in the closed loop", True),
            ("all waypoints reached within 3 cm", reached == len(total)),
        ]
    except NotImplementedError:
        print("  !! ex2 still has an unimplemented TODO")
        checks.append(("closed loop runs", False))
    except Exception as exc:  # noqa: BLE001
        print(f"  !! {type(exc).__name__}: {exc}")
        checks.append(("closed loop runs", False))
    return report(checks)


def main():
    ok = True
    ok &= check_quintic()
    ok &= check_pid()
    ok &= check_closed_loop()
    section("RESULT")
    print("  ALL CHECKS PASSED - exercise 2 looks good to record." if ok
          else "  Some checks failed - see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
