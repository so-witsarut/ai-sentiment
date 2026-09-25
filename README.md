# AI Sentiment (REST-only)

Windows production worker for the Blue Eye REST API. It reads pending posts, classifies with Jev and routes uncertain or conflicting results to DeepSeek, then submits sentiment and optional intent through REST. No direct MySQL or MongoDB connection is used.

The worker evaluates sentiment and intent toward `project_name` when provided. Keywords select a bounded excerpt of the post, which also retains a standalone occurrence of the project name; a keyword mention alone does not establish that the post concerns the project. Short Latin names such as `PEA` are matched as words rather than inside words such as `appear`. Intent is one of `complaint`, `information`, `recommendation`, or `enquiry`. The worker omits `intent` when the project is unrelated or the model supplies no valid intent. Publisher names and the host/path of `feed_link` are passed as context; the worker does not fetch the linked page.

When `project_name` is absent, the worker classifies the overall sentiment and intent of the post. Jev then asks only the sentiment and intent questions; keywords still select the excerpt. The REST result can therefore be submitted without project metadata. Empty content remains neutral by rule.

Set `OPENROUTER_PROVIDERS=relace/fp4` to prioritize Relace for DeepSeek. List several provider endpoint slugs separated by commas to try them in order. The DeepSeek route asks for JSON in its prompt and validates the result, so providers that do not support `response_format=json_object` can also run. Set `OPENROUTER_ALLOW_FALLBACKS=false` to restrict routing to the listed providers. Restart the worker after changing `.env`.

`AI_COST_MODE=low` is the default. It accepts Jev results with moderate confidence when there is no conflict, and also accepts confidently unrelated results as neutral toward the project. This reduces DeepSeek calls but can miss subtle mentions or mixed opinions. `JEV_TEXT_MAX_CHARS=1800` limits Jev's excerpt around the project name and keyword; DeepSeek can still inspect up to `DEEPSEEK_TEXT_MAX_CHARS` (default 3000) when called. Set `AI_COST_MODE=standard` and `JEV_TEXT_MAX_CHARS=3000` to restore the previous routing and excerpt size. The usage log reports `low_cost_accepted` and `escalated` for each batch.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `BE_API_TOKEN`, `OPENROUTER_API_KEY`, and the other required values in `.env`. Do not commit `.env`. Set `SAVE_DB=true` only when ready to submit results. `SAVE_DB=false` prevents REST writes but still makes paid provider calls.

AI decisions and neutral defaults after provider failure are cached in `.cache/sentiment_results.sqlite3` for 30 days. Identical content, project, keywords, and source reuse one decision across rows, batches, and worker restarts. If the REST write fails, the worker reuses the saved result on the next attempt. Changing the worker code, model, or analysis inputs creates a new cache key. When Jev and DeepSeek cannot produce a usable decision, the worker submits neutral with no intent; this may miss a positive or negative post. Set `SENTIMENT_CACHE_ENABLED=false` to disable the cache, `SENTIMENT_CACHE_PATH` to move its database, or `SENTIMENT_CACHE_TTL_DAYS` to change its lifetime (1–365 days). In dry-run mode the first analysis still uses paid providers; later repeats can reuse cached results.

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
