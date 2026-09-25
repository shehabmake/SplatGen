@echo off
rem Starts SplatGen and opens it in your browser. Close this window to quit.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo SplatGen is not installed yet. Run install_windows.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m splatgen app %*
