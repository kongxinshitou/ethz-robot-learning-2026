@echo off
rem Open the MuJoCo viewer with the SO-101 arm (Homework 2 smoke test).
cd /d "%~dp0ethz-robot-learning-2026\hw2_robot_control_mdps"
".venv\Scripts\python.exe" scripts\interactive.py
if errorlevel 1 pause
