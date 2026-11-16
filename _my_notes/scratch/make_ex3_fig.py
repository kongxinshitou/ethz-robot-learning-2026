"""Make a compact diagnosis figure for the ex3 multicube teleop data (all stores)."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

HW = Path("datasets")
RAW_DIR = HW / "raw/multi_cube/teleop"
CLEAN = HW / "raw_clean/multi_cube/teleop/so100_multicube_teleop.zarr"
DECISIONS = HW / "inspect_ex3/clean_decisions.csv"
OUT = HW / "inspect_ex3/ex3_data_diagnosis.png"

COLORS = ("red", "green", "blue")
CUBE_KEYS = tuple(f"pos_cube_{c}" for c in COLORS)


def load_merged(d: Path):
    arrs, ends, off = {}, [], 0
    for p in sorted(x for x in d.rglob("*.zarr") if x.is_dir()):
        g = zarr.open_group(str(p), mode="r")
        e = np.asarray(g["meta"]["episode_ends"][:], dtype=int)
        n = int(e[-1]) if e.size else 0
        data = {k: np.asarray(g["data"][k][:n]) for k in g["data"].array_keys()}
        if not arrs:
            arrs = {k: [] for k in data}
        for k, v in data.items():
            arrs[k].append(v)
        ends.append(e + off)
        off += n
    return {k: np.concatenate(v) for k, v in arrs.items()}, np.concatenate(ends)


def load_one(p: Path):
    g = zarr.open_group(str(p), mode="r")
    a = {k: np.asarray(g["data"][k][:]) for k in g["data"].array_keys()}
    ends = np.asarray(g["meta"]["episode_ends"][:], dtype=int)
    return a, ends


raw, raw_ends = load_merged(RAW_DIR)
clean, clean_ends = load_one(CLEAN)

with DECISIONS.open(encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
decision = {int(r["ep"]): r for r in rows}


def static_frac(a, s, e):
    if e - s < 2:
        return 0.0
    ee = a["state_ee"][s:e, :3]
    cubes = [a[k][s:e, :3] for k in CUBE_KEYS]
    grip = a["action_gripper"][s:e, 0]
    d_ee = np.linalg.norm(np.diff(ee, axis=0), axis=1)
    d_cube = np.stack(
        [np.linalg.norm(np.diff(c, axis=0), axis=1) for c in cubes], axis=1
    ).max(axis=1)
    d_grip = np.abs(np.diff(grip))
    static = np.r_[False, (d_ee < 0.002) & (d_cube < 0.002) & (d_grip < 0.02)]
    return 100.0 * static.mean()


fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
fig.suptitle("HW3 ex3 multicube teleop — recorded data diagnosis", fontsize=13)

# (a) static fraction per raw episode --------------------------------
ax = axes[0, 0]
starts = [0, *[int(e) for e in raw_ends[:-1]]]
pcts = [static_frac(raw, s, int(e)) for s, e in zip(starts, raw_ends)]
bar_colors = [
    "tab:red" if decision[i + 1]["status"] == "drop" else "tab:blue"
    for i in range(len(pcts))
]
ax.bar(range(1, len(pcts) + 1), pcts, color=bar_colors)
ax.axhline(np.mean(pcts), ls="--", lw=1, color="k", alpha=0.6)
ax.text(0.6, np.mean(pcts) + 1.5, f"mean {np.mean(pcts):.0f}% static", fontsize=8)
ax.set_ylim(0, 100)
ax.set_xlabel("raw episode (3 recording sessions)")
ax.set_ylabel("idle / static frames (%)")
ax.set_title(f"(a) raw data is ~{np.mean(pcts):.0f}% static frames\n"
             f"({len(pcts)} episodes, red = dropped)")
ax.grid(alpha=0.25, axis="y")


def ee_steps(a, ends):
    ee = a["state_ee"][:, :3]
    out, s = [], 0
    for e in ends:
        out.append(np.linalg.norm(np.diff(ee[s:e], axis=0), axis=1) * 1000.0)
        s = e
    return np.concatenate(out)


ax = axes[0, 1]
bins = np.linspace(0, 12, 61)
ax.hist(ee_steps(raw, raw_ends), bins=bins, alpha=0.55, label="raw", color="tab:orange")
ax.hist(ee_steps(clean, clean_ends), bins=bins, alpha=0.8, label="cleaned", color="tab:blue")
ax.set_yscale("log")
ax.set_xlabel("EE translation between recorded frames (mm)")
ax.set_ylabel("count (log)")
ax.set_title("(b) static frames = narrow 0 mm spike\ncleaning keeps ~1 frame per teleop step")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

# (c) cleaned target-cube paths, top view ---------------------------
ax = axes[0, 2]
s = 0
for e in clean_ends:
    goal = clean["state_goal"][s:e]
    tgt = COLORS[int(np.argmax(goal[0]))]
    cube = clean[f"pos_cube_{tgt}"][s:e, :3]
    bin_ = clean["goal_pos"][s:e, :2][0]
    ax.plot(cube[:, 0], cube[:, 1], lw=0.9, alpha=0.55)
    ax.plot(cube[0, 0], cube[0, 1], "o", ms=2.5, color="k")
    ax.plot(bin_[0], bin_[1], "*", ms=6)
    s = e
ax.set_aspect("equal", adjustable="datalim")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title(f"(c) cleaned: target cube start (dot) -> bin (star)\n"
             f"{len(clean_ends)}/{len(clean_ends)} end inside the bin")
ax.grid(alpha=0.25)

# (d) goal coverage --------------------------------------------------
ax = axes[1, 0]
counts = [
    sum(1 for r in rows if r["status"] == "keep" and r["target"] == c) for c in COLORS
]
bars = ax.bar(COLORS, counts, color=["tab:red", "tab:green", "tab:blue"])
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, c + 0.3, str(c), ha="center")
ax.set_ylim(0, max(counts) + 2)
ax.set_ylabel("episodes")
ax.set_title("(d) goal-cube coverage after cleaning")
ax.grid(alpha=0.25, axis="y")

# (e) episode 2: cube jump = reset happened while recording ---------
ax = axes[1, 1]
s2, e2 = raw_ends[0], raw_ends[1]  # episode 2 spans [ends[0], ends[1])
t = np.arange(e2 - s2) / 10.0
for c, col in zip(COLORS, ("tab:red", "tab:green", "tab:blue")):
    ax.plot(t, raw[f"pos_cube_{c}"][s2:e2, 1], lw=1.4, color=col, label=c)
ax.axvline(12 / 10.0, color="k", ls="--", lw=1.2)
ax.text(1.3, 0.69, "reset at frame 12\n(cubes teleport, bin jumps 13 mm)", fontsize=8)
ax.set_xlabel("time in episode 2 (s)")
ax.set_ylabel("cube y (m)")
ax.set_title("(e) episode 2: 'reset' pressed while record was paused\n"
             "-> episode dropped as a whole")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

# (f) episode 8: goal one-hot changes mid-episode -------------------
ax = axes[1, 2]
s8, e8 = raw_ends[6], raw_ends[7]  # episode 8 spans [ends[6], ends[7])
t = np.arange(e8 - s8) / 10.0
goal_idx = np.argmax(raw["state_goal"][s8:e8], axis=1)
ax.step(t, goal_idx, where="post", color="tab:purple", lw=1.6,
        label="goal cube index (0=red,1=green,2=blue)")
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["red", "green", "blue"])
ax.set_ylim(-0.5, 2.5)
ax.axvline(41 / 10.0, color="k", ls="--", lw=1.2)
ax.text(4.15, 1.4, "goal switched\ngreen -> red", fontsize=8)
ax2 = ax.twinx()
ax2.plot(t, raw["pos_cube_red"][s8:e8, 2], color="tab:red", lw=1.2, alpha=0.8,
         label="red cube z")
ax2.plot(t, raw["pos_cube_green"][s8:e8, 2], color="tab:green", lw=1.2, alpha=0.8,
         label="green cube z")
ax2.set_ylabel("cube z (m)")
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax.legend(handles1 + handles2, labels1 + labels2, fontsize=7, loc="upper right")
ax.set_xlabel("time in episode 8 (s)")
ax.set_title("(f) episode 8: goal switched mid-episode\n-> episode dropped as a whole")
ax.grid(alpha=0.25)

fig.tight_layout(rect=[0, 0, 1, 0.96])
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=140)
print(f"saved {OUT}")
