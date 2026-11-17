@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment not found. Run: py -m venv .venv
    exit /b 1
)
if not exist ".env" (
    echo [ERROR] .env not found. Create it from .env.example with BE_API_TOKEN.
    exit /b 1
)

if "%~1"=="" goto interactive
".venv\Scripts\python.exe" sentiment_backfill.py %*
exit /b %errorlevel%

:interactive
echo === Sentiment Backfill (Ollama deepseek-r1:8b, RTX 5060 Ti 8GB) ===
set /p FROM_DATE=From date (YYYY-MM-DD): 
set /p TO_DATE=To date   (YYYY-MM-DD): 
set /p SAVE_MODE=Save results to DB? [y/N]: 
if /i "%SAVE_MODE%"=="y" (
    ".venv\Scripts\python.exe" sentiment_backfill.py --from-date "%FROM_DATE%" --to-date "%TO_DATE%" --save-db
) else (
    ".venv\Scripts\python.exe" sentiment_backfill.py --from-date "%FROM_DATE%" --to-date "%TO_DATE%" --dry-run
)
exit /b %errorlevel%
