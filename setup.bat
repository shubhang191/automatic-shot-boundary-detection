@echo off
setlocal
set "PYTHONUTF8=1"
cd /d "%~dp0"
echo Automatic Shot Boundary Detection - Windows setup
echo Requires 64-bit Python 3.13. First installation uses the internet.
if exist ".venv\Scripts\python.exe" goto install
where py >nul 2>nul
if errorlevel 1 goto python_fallback
py -3.13 -c "import sys,struct; assert sys.version_info[:2]==(3,13) and struct.calcsize('P')==8" >nul 2>nul
if errorlevel 1 goto python_fallback
py -3.13 -m venv .venv
if errorlevel 1 goto failed
goto install
:python_fallback
python -c "import sys,struct; assert sys.version_info[:2]==(3,13) and struct.calcsize('P')==8" >nul 2>nul
if errorlevel 1 (
  echo Install free 64-bit Python 3.13 from https://www.python.org/downloads/windows/ and rerun setup.bat.
  exit /b 1
)
python -m venv .venv
if errorlevel 1 goto failed
:install
if exist wheelhouse (
  ".venv\Scripts\python.exe" -m pip install --no-index --find-links wheelhouse -r requirements-lock.txt
) else (
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt -c requirements-lock.txt
)
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed
".venv\Scripts\python.exe" scripts\bootstrap_ffmpeg.py
if errorlevel 1 goto failed
if not exist "third_party\sbdv2\bin\sbdv.dll" (
  ".venv\Scripts\python.exe" scripts\build_verifier.py
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" scripts\validate_environment.py
if errorlevel 1 goto failed
echo Setup complete. Run run.bat to open the app.
exit /b 0
:failed
echo Setup failed. Read the error above and docs\troubleshooting.md. No research files were changed.
exit /b 1
