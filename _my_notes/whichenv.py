"""Print which virtual environment the current 'python' belongs to."""
import os
import sys
import importlib.metadata as md

venv = os.environ.get("VIRTUAL_ENV")
print("VIRTUAL_ENV :", venv if venv else "(not set - no venv activated)")
print("python      :", sys.executable)
print("prefix      :", sys.prefix)

hw = ""
for tag in ("hw1_pytorch_tutorial", "hw2_robot_control_mdps",
            "hw3_imitation_learning", "hw4_reinforcement_learning"):
    if tag in sys.executable:
        hw = tag
print("hw folder   :", hw if hw else "(unknown / system python)")

print("--- key packages ---")
for name in ("torch", "torchvision", "mujoco", "gymnasium", "stable-baselines3",
             "zarr", "opencv-python", "dm-control", "moviepy", "jupyterlab", "pygame"):
    try:
        print(f"  {name:18s} {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"  {name:18s} -")
