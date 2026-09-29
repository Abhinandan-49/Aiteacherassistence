@echo off
title AI Teaching Assistant (Google Gemini RAG)
echo =====================================================================
echo           Starting AI Teaching Assistant Full-Stack Platform
echo =====================================================================
echo.
echo Local Web UI: http://localhost:5000
echo.

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
) else (
    python app.py
)

pause
