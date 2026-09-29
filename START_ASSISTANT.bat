@echo off
title Launch AI Teaching Assistant
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    start http://localhost:5000
    ".venv\Scripts\python.exe" app.py
) else (
    start http://localhost:5000
    python app.py
)
pause
