"""Inspect recorded / processed HW3 zarr datasets.

Run with the HW3 venv python from the hw3_imitation_learning folder:

    .venv\\Scripts\\python.exe ..\\hw3_check_data.py

Prints, for every .zarr under datasets/raw and datasets/processed:
  - number of recorded steps and episodes
  - which data arrays exist (state_ee, state_cube, action_gripper, ...)
  - useful metadata (control_hz, dagger episode count, ...)
"""

from __future__ import annotations

import sys
from pathlib import Path

import zarr


def describe(path: Path, root: Path) -> None:
    try:
        g = zarr.open_group(str(path), mode="r")
    except Exception as exc:  # noqa: BLE001
        print(f"  [!] {path}: {exc}")
        return

    try:
        data = g["data"]
        keys = sorted(data.array_keys())
    except Exception:  # noqa: BLE001
        keys = []

    steps = 0
    for k in keys:
        try:
            steps = max(steps, int(data[k].shape[0]))
        except Exception:  # noqa: BLE001
            pass

    try:
        episodes = int(g["meta"]["episode_ends"].shape[0])
    except Exception:  # noqa: BLE001
        episodes = None

    print(f"  {path.relative_to(root)}")
    print(f"    steps={steps}  episodes={episodes if episodes is not None else '?'}")
    print(f"    arrays={keys}")

    attrs = dict(g.attrs)
    for k in (
        "control_hz",
        "action_key",
        "state_key",
        "num_dagger_episodes",
        "action_space",
    ):
        if k in attrs:
            print(f"    {k}={attrs[k]}")


def main() -> int:
    root = Path.cwd()
    if not (root / "scripts" / "record_teleop_demos.py").exists():
        print(
            "[ERROR] Run this from the hw3_imitation_learning folder.\n"
            f"        Current folder: {root}"
        )
        return 1

    found_any = False
    for sub in ("raw", "processed"):
        base = root / "datasets" / sub
        print(f"\n=== datasets/{sub} ===")
        if not base.exists():
            print("  (does not exist yet)")
            continue
        stores = sorted(p for p in base.rglob("*.zarr") if p.is_dir())
        if not stores:
            print("  (empty)")
            continue
        found_any = True
        for p in stores:
            describe(p, root)

    if not found_any:
        print(
            "\nNo datasets yet. Record some demos first:\n"
            "  python scripts/record_teleop_demos.py            (single cube, ex1)\n"
            "  python scripts/record_teleop_demos.py --multicube (multicube, ex3)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
