@echo off
setlocal enabledelayedexpansion
rem SplatGen trainer - one-time setup on Windows with an NVIDIA GPU.
rem Downloads about 3 GB (PyTorch with CUDA). Needs an internet connection.
rem Change these two lines to match your GPU driver if needed:
set TORCH_INDEX=https://download.pytorch.org/whl/cu124
set GSPLAT_INDEX=https://docs.gsplat.studio/whl/pt24cu124

cd /d "%~dp0"
title SplatGen setup

rem -- find Python 3.10 - 3.12 (PyTorch 2.4 has no wheels for 3.13) --------
set PY=
for %%V in (3.12 3.11 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set PY=py -%%V
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1 && set PY=python
)
if not defined PY (
  echo Python 3.10 - 3.12 was not found. Trying to install Python 3.11 with winget...
  winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
  py -3.11 -c "import sys" >nul 2>&1 && set PY=py -3.11
)
if not defined PY (
  echo.
  echo Please install Python 3.11 from https://www.python.org/downloads/release/python-3119/
  echo ^(tick "Add python.exe to PATH"^), then run this file again.
  pause
  exit /b 1
)
echo Using !PY!

if not exist .venv (
  echo Creating virtual environment...
  !PY! -m venv .venv || goto :error
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo Installing PyTorch with CUDA (large download)...
python -m pip install torch==2.4.1 --index-url %TORCH_INDEX% || goto :error
echo Installing SplatGen...
python -m pip install -e . || goto :error
echo Installing gsplat (fast CUDA renderer)...
python -m pip install gsplat --index-url %GSPLAT_INDEX% || python -m pip install gsplat || echo gsplat could not be installed - training will use the slow PyTorch renderer.
python -m splatgen info
echo.
echo Done. Start SplatGen by double-clicking SplatGen.bat
pause
exit /b 0

:error
echo Setup failed. See the messages above.
pause
exit /b 1
