"""Inspect a multicube raw teleop zarr: episodes, goals, layouts, static frames, likely success."""
import sys
from pathlib import Path

import numpy as np
import zarr

p = Path(sys.argv[1])
r = zarr.open_group(str(p), mode="r")
print("store:", p)
print("attrs:", dict(r.attrs))
data = r["data"]
ep = np.asarray(r["meta"]["episode_ends"][:], dtype=np.int64)
print("arrays:", {k: data[k].shape for k in data})
print("episode_ends:", ep.shape, "steps:", int(ep[-1]) if len(ep) else 0)

need = ["state_joints", "state_ee", "state_gripper", "action_gripper",
        "state_goal", "goal_pos", "pos_cube_red", "pos_cube_green", "pos_cube_blue"]
missing = [k for k in need if k not in data]
if missing:
    print("MISSING:", missing)
    sys.exit(1)

arr = {k: np.asarray(data[k][:]) for k in need}
starts = np.r_[0, ep[:-1]]
colors = ("red", "green", "blue")

print("\nidx  color  len   static<1e-4  static<5e-4  lift  success  bin_xy  end_cube_z  goal_pos(x,y)")
n_by_color = {c: 0 for c in colors}
succ_by_color = {c: 0 for c in colors}
n_success = 0
static_all = []
len_all = []
goals = []
bins = []
starts_xyz = []
for i, (s, e) in enumerate(zip(starts, ep)):
    s, e = int(s), int(e)
    goal = arr["state_goal"][s:e]
    onehot = goal.argmax(axis=1)
    goal_idx = int(np.bincount(onehot, minlength=3).argmax())
    color = colors[goal_idx]
    n_by_color[color] += 1
    goals.append(color)
    gpos = arr["goal_pos"][s:e]
    bins.append(gpos[0, :2].copy())

    ee = arr["state_ee"][s:e, :3]
    d_ee = np.linalg.norm(np.diff(ee, axis=0), axis=1) if e - s > 1 else np.array([0.0])
    static_1 = float(np.mean(d_ee < 1e-4))
    static_5 = float(np.mean(d_ee < 5e-4))
    static_all.append(static_5)
    len_all.append(e - s)

    cubes = np.stack([arr[f"pos_cube_{c}"][s:e, :3] for c in colors], axis=0)  # (3,L,3)
    target = cubes[goal_idx]
    lift = bool(target[:, 2].max() > 0.06)
    end = target[-1]
    bin_xy = float(np.linalg.norm(end[:2] - gpos[-1, :2]))
    in_xy = bin_xy < 0.04
    in_z = 0.0 < end[2] < 0.04
    success = bool(in_xy and in_z)
    n_success += int(success)
    succ_by_color[color] += int(success)
    starts_xyz.append(np.stack([cubes[j, 0] for j in range(3)])[:, :2])

    print(f"{i:3d}  {color:5s} {e-s:4d}   {100*static_1:5.1f}%      {100*static_5:5.1f}%     "
          f"{'Y' if lift else '.':>3s}   {'Y' if success else '.':>4s}    {bin_xy*100:5.2f}cm  "
          f"{end[2]:9.3f}   ({gpos[0,0]:6.3f},{gpos[0,1]:6.3f})")

print("\n=== summary ===")
print("episodes:", len(ep))
print("per color:", n_by_color)
print("likely success per color:", succ_by_color, "total", n_success)
print("episode length: min/median/max =", int(np.min(len_all)), int(np.median(len_all)), int(np.max(len_all)))
print("static<5e-4: mean %.1f%%  per-episode min/median/max = %.1f/%.1f/%.1f%%" % (
    100*np.mean(static_all), 100*np.min(static_all), 100*np.median(static_all), 100*np.max(static_all)))
print("bin xy std (cm):", np.round(bins_std := np.std(np.array(bins), axis=0)*100, 2))
print("cube start xy std per cube (cm):")
for j, c in enumerate(colors):
    xyz = np.stack([s[j] for s in starts_xyz])
    print("  ", c, np.round(np.std(xyz, axis=0)*100, 2), "range x", np.round(np.ptp(xyz[:,0])*100,3), "range y", np.round(np.ptp(xyz[:,1])*100,3))

# goal consistency inside episodes
bad_goal = []
for i, (s, e) in enumerate(zip(starts, ep)):
    g = arr["state_goal"][int(s):int(e)]
    if len(g) and not np.allclose(g, g[0]):
        bad_goal.append(i)
print("episodes whose state_goal changes within episode:", bad_goal if bad_goal else "none")

# duplicate layouts?
layout_keys = [tuple(np.round(s.ravel(), 4)) for s in starts_xyz]
from collections import Counter
dup = [k for k, n in Counter(layout_keys).items() if n > 1]
print("duplicate cube-start layouts:", len(dup))
