import zarr, numpy as np
from pathlib import Path
base = Path(r"D:\Users\00796533\Documents\deepseek-harness\default-workspace\ethz-robot-learning-2026\hw3_imitation_learning")
print("raw dirs:")
for p in sorted((base/"datasets/raw/single_cube/teleop").iterdir()):
    print(" ", p.name)
g = zarr.open_group(str(base/"datasets/processed/single_cube/processed_ee_xyz_new.zarr"), mode="r")
ag = np.asarray(g["data"]["action_gripper"]).ravel()
ee = np.asarray(g["data"]["state_ee_xyz"])
ends = np.asarray(g["meta"]["episode_ends"][:])
print("\nnew data action_gripper: min %.3f max %.3f" % (ag.min(), ag.max()))
for q in (0, 1, 5, 25, 50, 75, 95, 99, 100):
    print("  p%02d = %.3f" % (q, np.percentile(ag, q)))
print("  frames with ag < -0.1: %.1f%%" % (100*np.mean(ag < -0.1)))
print("  frames with ag >  1.0: %.1f%%" % (100*np.mean(ag > 1.0)))
# per-episode gripper closed fraction (state)
sg = np.asarray(g["data"]["state_gripper"]).ravel()
print("  state_gripper: min %.3f max %.3f" % (sg.min(), sg.max()))
