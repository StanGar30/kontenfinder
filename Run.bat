@echo off
setlocal
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
".venv\Scripts\python.exe" -u "main.py" > "log.txt" 2>&1
exit /b %errorlevel%
