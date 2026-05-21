@echo off
REM Double-click to scan incoming/ and transcribe new audio files.
REM Pass extra args (e.g. --watch, --force) on the command line if you want.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m src.app %*
echo.
pause
