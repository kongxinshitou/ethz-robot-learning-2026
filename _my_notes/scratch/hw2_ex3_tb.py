"""Print the hw2 ex3 training curves from the TensorBoard event file."""
import sys
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator

LOG = Path(r"D:\Users\00796533\Documents\deepseek-harness\default-workspace\ethz-robot-learning-2026\hw2_robot_control_mdps\logs\so100_tracking")
ROOT = LOG / "so100_tracking_1"

ea = event_accumulator.EventAccumulator(str(ROOT), size_guidance={event_accumulator.SCALARS: 0})
ea.Reload()
tags = ea.Tags()["scalars"]
print("tags:", tags)

def show(tag, every=50, fmt="{:.4f}"):
    if tag not in tags:
        print(f"\n{tag}: (missing)")
        return
    ev = ea.Scalars(tag)
    print(f"\n{tag}  (n={len(ev)})")
    for e in ev:
        if e.step % every == 0 or e is ev[-1]:
            print(f"  update {e.step:4d}  {fmt.format(e.value)}")

for t in tags:
    show(t, every=50 if "epoch" not in t else 5)

# summary of the last 100 update steps of every scalar
print("\n=== last 5 values per tag ===")
for t in tags:
    ev = ea.Scalars(t)
    print(f"{t:35s} {[round(e.value, 4) for e in ev[-5:]]}")
