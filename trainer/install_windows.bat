@echo off
setlocal
rem SplatGen trainer - one-time setup on Windows with an NVIDIA GPU.
rem Needs Python 3.10-3.12 on PATH (python.org installer, "Add to PATH" ticked).
rem Change these two lines to match your GPU driver if needed:
set TORCH_INDEX=https://download.pytorch.org/whl/cu124
set GSPLAT_INDEX=https://docs.gsplat.studio/whl/pt24cu124

cd /d "%~dp0"
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv || goto :error
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo Installing PyTorch with CUDA...
python -m pip install torch==2.4.1 --index-url %TORCH_INDEX% || goto :error
echo Installing SplatGen...
python -m pip install -e . || goto :error
echo Installing gsplat (fast CUDA renderer)...
python -m pip install gsplat --index-url %GSPLAT_INDEX% || python -m pip install gsplat || echo gsplat could not be installed - training will use the slow PyTorch renderer.
python -m splatgen info
echo.
echo Done. Start SplatGen with SplatGen.bat
pause
exit /b 0

:error
echo Setup failed. See the messages above.
pause
exit /b 1
