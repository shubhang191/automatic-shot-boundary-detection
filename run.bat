@echo off
setlocal
set "PYTHONUTF8=1"
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat once before starting the application.
  exit /b 1
)
".venv\Scripts\python.exe" scripts\validate_environment.py
if errorlevel 1 (
  echo Environment check failed. Run setup.bat or read docs\troubleshooting.md.
  exit /b 1
)
echo Starting locally at http://127.0.0.1:7860 or the next free port.
".venv\Scripts\python.exe" app.py %*
exit /b %errorlevel%
