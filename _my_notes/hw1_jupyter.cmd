@echo off
rem Start JupyterLab for Homework 1 (notebooks in hw1_pytorch_tutorial\src).
cd /d "%~dp0ethz-robot-learning-2026\hw1_pytorch_tutorial"
".venv\Scripts\python.exe" -m jupyter lab src
if errorlevel 1 pause
