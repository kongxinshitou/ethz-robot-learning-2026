import json
import zarr
from pathlib import Path

root = Path(r"D:\Users\00796533\Documents\deepseek-harness\default-workspace\ethz-robot-learning-2026\hw3_imitation_learning")

print("=== processed zarr attrs ===")
g = zarr.open_group(str(root / "datasets/processed/single_cube/processed_ee_xyz.zarr"), mode="r")
print(json.dumps({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in dict(g.attrs).items()}, indent=2))
print("episode_ends:", g["meta"]["episode_ends"][:])

print("\n=== raw teleop sessions ===")
for p in sorted((root / "datasets/raw").rglob("*.zarr")):
    print("-", p.relative_to(root))
    try:
        z = zarr.open_group(str(p), mode="r")
        ends = z["meta"]["episode_ends"][:]
        print("   episodes:", len(ends), "steps:", int(ends[-1]) if len(ends) else 0)
        print("   data keys:", list(z["data"].keys()))
        print("   attrs:", {k: v for k, v in dict(z.attrs).items() if k in ("control_hz", "xml")})
    except Exception as e:  # noqa: BLE001
        print("   ERROR:", type(e).__name__, e)

print("\n=== raw_clean ===")
for p in sorted((root / "datasets/raw_clean").rglob("*.zarr")):
    print("-", p.relative_to(root))
    z = zarr.open_group(str(p), mode="r")
    ends = z["meta"]["episode_ends"][:]
    print("   episodes:", len(ends), "steps:", int(ends[-1]) if len(ends) else 0)
