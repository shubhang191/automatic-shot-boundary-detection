@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first to create the Python environment.
  exit /b 1
)
".venv\Scripts\python.exe" scripts\build_verifier.py %*
exit /b %errorlevel%
