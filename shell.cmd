@echo off
rem Open a shell with one homework's virtual environment activated.
rem Usage:  shell.cmd hw1  |  hw2  |  hw3  |  hw4
rem Works both from the workspace root and from inside ethz-robot-learning-2026.
setlocal

set "BASE=%~dp0"
if exist "%BASE%ethz-robot-learning-2026\hw1_pytorch_tutorial" (
  set "ROOT=%BASE%ethz-robot-learning-2026"
) else if exist "%BASE%hw1_pytorch_tutorial" (
  for %%I in ("%BASE%.") do set "ROOT=%%~dpIethz-robot-learning-2026"
) else (
  echo [ERROR] Cannot find the course folder.
  echo         Put this file in the workspace root or in ethz-robot-learning-2026.
  pause
  exit /b 1
)

set "HW=%~1"
if "%HW%"=="" set "HW=hw1"
if /i "%HW%"=="hw1" set "DIR=hw1_pytorch_tutorial"
if /i "%HW%"=="hw2" set "DIR=hw2_robot_control_mdps"
if /i "%HW%"=="hw3" set "DIR=hw3_imitation_learning"
if /i "%HW%"=="hw4" set "DIR=hw4_reinforcement_learning"
if not defined DIR set "DIR=%HW%"

if not exist "%ROOT%\%DIR%\.venv\Scripts\activate.bat" (
  echo [ERROR] No venv found for "%DIR%".
  echo         Available: hw1 hw2 hw3 hw4
  pause
  exit /b 1
)

call :deactivate_if_needed
cd /d "%ROOT%\%DIR%"
call ".venv\Scripts\activate.bat"
echo.
echo ==========================================================
echo  HW folder : %CD%
echo  Python    : %VIRTUAL_ENV%
echo  Type "deactivate" + Enter to leave the environment.
echo ==========================================================
echo.
cmd /k
exit /b 0

:deactivate_if_needed
if defined VIRTUAL_ENV (
  call "%VIRTUAL_ENV%\Scripts\deactivate.bat" 2>nul
)
exit /b 0
