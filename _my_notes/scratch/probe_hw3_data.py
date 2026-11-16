import zarr
import numpy as np

p = "datasets/processed/single_cube/processed_ee_xyz.zarr"
g = zarr.open_group(p, mode="r")
print("attrs:", dict(g.attrs))
print("data keys:", list(g["data"].keys()))
for k in g["data"]:
    a = g["data"][k]
    print(f"  {k}: shape={a.shape} dtype={a.dtype}")
ee = np.asarray(g["meta"]["episode_ends"][:])
print("episode_ends:", ee, "n_episodes:", ee.size)
