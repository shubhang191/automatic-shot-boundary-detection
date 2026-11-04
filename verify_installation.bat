@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto invalid
".venv\Scripts\python.exe" -B scripts\interpreter_probe.py --venv
if errorlevel 1 goto invalid
".venv\Scripts\python.exe" -B verify_installation.py
exit /b %errorlevel%
:invalid
echo FAIL: Missing or invalid local environment. Run setup.bat to diagnose and safely repair it.
exit /b 1
