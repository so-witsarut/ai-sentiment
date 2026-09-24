# AI Sentiment (REST-only)

Windows production worker for the Blue Eye REST API. It reads pending posts, classifies with Jev and routes uncertain or conflicting results to DeepSeek, then submits the existing sentiment payload through REST. No direct MySQL or MongoDB connection is used.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `BE_API_TOKEN`, `OPENROUTER_API_KEY`, and the other required values in `.env`. Do not commit `.env`. Set `SAVE_DB=true` only when ready to submit results. `SAVE_DB=false` prevents REST writes but still makes paid provider calls.

## Run

Start **one** worker from the repository directory:

```powershell
py ai_sentiment.py
```

Or run `run.bat`, which uses the local `.venv`. `--mode rest` is accepted but optional; `--mode db` and `--mode both` are not supported. Stop with Ctrl+C and wait for the process to exit before restarting.

The worker polls the REST queue every `RUN_INTERVAL_SECONDS` (default 5). `BATCH_SIZE`, `CONCURRENT_WORKERS`, `MAX_IN_FLIGHT`, and `DEEPSEEK_MAX_CONCURRENCY` bound work in each process. Running multiple processes multiplies those limits and risks duplicate work.

## Offline tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/probabilistic -p "test_*.py"
```

The test suite mocks provider and REST calls; it must not submit results or incur provider charges.
