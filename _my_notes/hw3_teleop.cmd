@echo off
rem HW3 helper: keyboard teleoperation, data check and evaluation.
rem Usage:  hw3_teleop.cmd          -> interactive menu
rem         hw3_teleop.cmd 1        -> configure keys
rem         hw3_teleop.cmd 2        -> record single-cube demos (ex1)
rem         hw3_teleop.cmd 3        -> record multicube demos (ex3)
rem         hw3_teleop.cmd 4        -> evaluate a checkpoint (headless)
rem         hw3_teleop.cmd 5        -> inspect recorded datasets
rem NOTE: teleoperation needs a real desktop window + keyboard focus.
setlocal
set "HERE=%~dp0"
cd /d "%HERE%ethz-robot-learning-2026\hw3_imitation_learning"
set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [ERROR] HW3 virtual environment not found:
  echo         %CD%\.venv
  pause
  exit /b 1
)

set "ACT=%~1"
if not "%ACT%"=="" goto run

:menu
echo.
echo  ===== HW3 Imitation Learning - teleop helper =====
echo     [1] Configure keys            (run once)
echo     [2] Record single-cube demos  (ex1)
echo     [3] Record multicube demos    (ex3)
echo     [4] Evaluate a checkpoint     (headless success rate)
echo     [5] Inspect recorded datasets
echo     [0] Exit
echo.
set "CH="
set /p CH=Choose [0-5]: 
if "%CH%"=="" goto menu
set "ACT=%CH%"

:run
if "%ACT%"=="1" goto keys
if "%ACT%"=="2" goto rec1
if "%ACT%"=="3" goto rec3
if "%ACT%"=="4" goto ev
if "%ACT%"=="5" goto check
if "%ACT%"=="0" exit /b 0
echo Unknown choice: %ACT%
exit /b 1

:keys
"%PY%" scripts\configure_keys.py
goto done

:rec1
"%PY%" scripts\record_teleop_demos.py
goto done

:rec3
"%PY%" scripts\record_teleop_demos.py --multicube
goto done

:ev
set "CKPT="
set /p CKPT=Checkpoint path (e.g. checkpoints\single_cube\ex1.pt): 
if "%CKPT%"=="" (
  echo No checkpoint given.
  goto done
)
"%PY%" scripts\eval.py --checkpoint "%CKPT%" --headless --num-episodes 50
goto done

:check
"%PY%" "%HERE%hw3_check_data.py"
goto done

:done
pause
exit /b 0
