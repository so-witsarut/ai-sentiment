@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment not found. Run: py -m venv .venv
    exit /b 1
)
if not exist ".env" (
    echo [ERROR] .env not found. Create it from .env.example with real REST and OpenRouter credentials.
    exit /b 1
)
echo Starting REST-only sentiment worker. Press Ctrl+C to stop.
".venv\Scripts\python.exe" ai_sentiment.py
exit /b %errorlevel%
