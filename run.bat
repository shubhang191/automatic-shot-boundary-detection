@echo off
setlocal
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto setup_required
".venv\Scripts\python.exe" -B scripts\interpreter_probe.py --venv
if errorlevel 1 goto setup_required
findstr /x /c:"PASS" ".venv\sbd_setup_status.txt" >nul 2>nul
if errorlevel 1 goto setup_required
echo Starting locally at http://127.0.0.1:7860 or the next free port.
".venv\Scripts\python.exe" -B scripts\launch_app.py %*
exit /b %errorlevel%
:setup_required
echo Setup is missing, incomplete, or invalid. Run setup.bat. Read installation_logs and docs\troubleshooting.md if it fails.
exit /b 1
