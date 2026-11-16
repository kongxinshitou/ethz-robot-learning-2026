import zarr, numpy as np
from pathlib import Path
base = Path(r"D:\Users\00796533\Documents\deepseek-harness\default-workspace\ethz-robot-learning-2026\hw3_imitation_learning")
raw = base/"datasets/raw/single_cube/teleop/2026-11-08_05-59-58/so100_transfer_cube_teleop.zarr"
g = zarr.open_group(str(raw), mode="r")
ee = np.asarray(g["data"]["state_ee"])[:, :3]
cube = np.asarray(g["data"]["state_cube"])[:, :3]
grip = np.asarray(g["data"]["action_gripper"]).ravel()
ends = np.asarray(g["meta"]["episode_ends"][:])
starts = np.concatenate(([0], ends[:-1]))

def mask(s, e, step_mm, grip_step=0.02):
    E, C, G = ee[s:e], cube[s:e], grip[s:e]
    n = len(E); step = step_mm/1000.0
    keep = np.zeros(n, bool); keep[0]=keep[-1]=True; last=0
    for i in range(1, n-1):
        if (np.linalg.norm(E[i]-E[last])>=step or np.linalg.norm(C[i]-C[last])>=step
                or abs(G[i]-G[last])>=grip_step):
            keep[i]=True; last=i
    return keep

for sm in (4, 6, 8, 10):
    lens, deltas, gaps = [], [], []
    for s, e in zip(starts, ends):
        k = mask(s, e, sm); idx = np.where(k)[0]
        lens.append(len(idx))
        d = np.linalg.norm(np.diff(ee[s:e][idx], axis=0), axis=1)
        deltas.append(np.median(d))
        z = cube[s:e][:,2]; gi = grip[s:e][idx]
        closed = np.where(gi < -0.05)[0]
        if closed.size:
            later = np.where((z[idx] > 0.06) & (np.arange(len(idx)) > closed[0]))[0]
            if later.size: gaps.append(later[0]-closed[0])
    print(f"step_mm={sm:>2}: 帧/条 {np.mean(lens):5.0f} | 相邻动作中位 {np.median(deltas)*1000:.1f} mm | close->lift {np.mean(gaps):4.0f} 帧 ({np.mean(gaps)*0.1:.1f}s)")
