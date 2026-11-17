# coding=utf-8
"""
Standalone historical sentiment backfill worker using local Ollama (deepseek-r1:8b),
aligned with api-blueeye's production `sentiment_worker.py` rules and schema.

Key guarantees:
- Uses Ollama exclusively (never calls OpenRouter or Gemini).
- Follows sentiment_worker.py's Author's Own Voice prompt and produces full verdicts:
  sentiment, sentiment_reason, sentiment_scores (w/ multi_target), intent,
  intent_reason, and incident.
- Never writes `rule:provider_failure` or any `rule:` placeholder on failure:
  failed posts are retried locally, logged to JSONL, and left pending.
- Live mode (--save-db) pulls via `/internal/sentiment/pending?claim=true` so it
  leases disjoint rows safely alongside `sentiment_worker.py`; dry-run uses
  read-only `/internal/sentiment/posts`.
"""

import argparse
import json
import os
import re
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    if stream and hasattr(stream, "reconfigure") and getattr(stream, "encoding", "") != "utf-8":
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

THAI_TIMEZONE = timezone(timedelta(hours=7))

SENTIMENT_VALUES = ("positive", "neutral", "negative")
INTENT_VALUES = ("recommendation", "complaint", "information", "enquiry")
MAX_CHARS = 800
MIN_REASON = 15
UNSURE_BELOW = 60

DEFAULT_BASE_URL = "https://api.blueeye.io/api/v1"
DEFAULT_MODEL = "deepseek-r1:8b"
DEFAULT_TIMEOUT = 120
DEFAULT_NUM_CTX = 4096
DEFAULT_KEEP_ALIVE = "30m"
DEFAULT_FAIL_LOG = os.path.join(os.path.dirname(__file__), "reports", "sentiment_backfill_failures.jsonl")

# Mirrors app/crud/sentiment_worker.py PROMPT (Author's Own Voice + Thai concrete reasons + incident)
# adapted for single-post Ollama inference with scores placed before sentiment for argmax consistency.
PROMPT = """\
Classify each post's sentiment and intent, and whether it reports an incident.

Every post is Thai or English social-media / news content. The verdict is shared by
every client that monitors the post, so judge the post itself: no client brand is
given and none may be assumed.

- sentiment: positive|neutral|negative. The author's tone toward what the post is
  mainly about (a product, service, brand, person or event). Several subjects in
  different tones: the dominant tone of the whole post.
- Judge the tone in the AUTHOR'S OWN voice, not whether the events are good or bad.
  Neutral: ads, promotions, sale listings, invitations, announcements, press releases
  and factual news reports, even with selling words, good results or bad news, and
  even when they report someone else's praise or criticism.
  Positive or negative: the author's own opinion or feeling, including loaded wording
  in the author's own voice, also in a news headline: shock, outrage, mockery, insult
  or admiration. Selling words in an ad do not count as admiration. Examples:
  "SET ปิดลดลง 4.10 จุด จากแรงกดดันหุ้นพลังงาน" -> neutral (reports a number)
  "ปล่อยบัตรคอนเสิร์ต โซน M วิวดี สนใจ IB ได้" -> neutral (a sale listing)
  "พลาดไม่ได้! ปาร์ตี้ริมทะเลสุดชิล ชวนเพื่อนมาด้วยกัน" -> neutral (an ad's selling words)
  "ธนาคาร X คว้า 2 รางวัล ยกระดับบริการ" -> neutral (PR about an award)
  "ประธาน กมธ. ซัดหน่วยงานแถลงข่าวไม่คืบหน้า" -> neutral (reports someone's criticism)
  "ช็อก! สินค้าหมดอายุวางขายเกลื่อนห้าง เห็นแล้วจุกอก" -> negative (the author's own shock)
  "นักการเมืองคราบขอทาน ผลาญเงินภาษีเราทั้งนั้น" -> negative (the author's mockery)
  "ยอดนักสู้ตัวจริง เด็ดขาด ไม่เกรงกลัวใคร" -> positive (the author's admiration)
  "รอคิวสองชั่วโมง พนักงานไม่สนใจเลย แย่มาก" -> negative (the author's complaint)
  "ร้านนี้อร่อยมาก ประทับใจ จะกลับมาอีกแน่นอน" -> positive (the author's praise)
- positive, neutral, negative: integer percentages summing to 100.
  CRITICAL: `sentiment` MUST match the strictly LARGEST score (for example: if neutral=80,
  negative=20, positive=0, then `sentiment` MUST be "neutral").
- intent: information|complaint|enquiry|recommendation. The author's purpose.
  recommendation = urges the reader to buy/try/join/support/attend. Announcing an
  award, result or news is information. complaint = reports a problem or
  dissatisfaction. enquiry = asks a question or asks for help.
- sentiment_reason and intent_reason: ONE short Thai sentence EACH, even for an English post,
  that cites concrete details of THIS input post (at least 15 Thai characters each).
  Write natural Thai only (no Chinese characters). Never copy example sentences and never
  write "แบรนด์ลูกค้า" or "แบรนด์ของเรา".
- multi_target: true if the post names two or more brands/products in different tones
  (praises one, criticises another), else false.
- incident: true if the post reports or alleges something harmful to a person or organisation:
  an accident, injury or death, a crime, arrest, lawsuit or investigation, corruption or fraud,
  a scandal, a food/product safety or health problem, a serious product or service failure,
  a call to boycott. True even when worded neutrally as news, and independent of sentiment.
  A plain complaint about everyday slow service, an ad, or fiction is false.

Return ONLY one valid JSON object with exact keys:
{"sentiment_reason":"<1 short Thai sentence >=15 chars>","positive":0,"neutral":100,"negative":0,"sentiment":"neutral","intent_reason":"<1 short Thai sentence >=15 chars>","intent":"information","multi_target":false,"incident":false}"""


class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama is unreachable or the circuit breaker trips."""


class _LimitReached(Exception):
    """Internal signal to stop cleanly once --limit unique posts are handled."""


@dataclass(frozen=True)
class Verdict:
    sentiment: str
    reason: str
    scores: Dict[str, Any]
    intent: str
    intent_reason: str
    incident: bool


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def validate_date_str(value: str) -> str:
    s = str(value or "").strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        raise ValueError(f"Invalid date format (expected YYYY-MM-DD): {value!r}")
    datetime.strptime(s, "%Y-%m-%d")
    return s


def parse_think_flag(value: str, model_name: str) -> Optional[bool]:
    raw = str(value or "").strip().lower()
    if raw in ("true", "1", "yes"):
        return True
    if raw in ("false", "0", "no"):
        return False
    if raw in ("", "auto") and "deepseek-r1" in str(model_name or "").lower():
        return False
    return None


def _clean_reason(value: Any) -> str:
    r = str(value or "").strip()
    r = re.sub(r"\s*语气[中正负]性\s*$", "", r).strip()
    if len(r) < MIN_REASON or not re.search(r"[ก-๙]", r):
        raise ValueError(f"reason too short or missing Thai text: {r!r}")
    thai_chars = len(re.findall(r"[ก-๙]", r))
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", r))
    if thai_chars < 10 or cjk_chars >= 4:
        raise ValueError(f"reason not mostly Thai: {r!r}")
    if "แบรนด์ลูกค้า" in r or "แบรนด์ของเรา" in r:
        raise ValueError(f"reason uses forbidden generic label: {r!r}")
    return r


def parse_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extract a single JSON dict, stripping any reasoning tags or code fences."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None
    clean = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    clean = re.sub(r"<thought>.*?</thought>", "", clean, flags=re.DOTALL)
    clean = re.sub(r"<(?:think|thought)>.*?(?=(?:```|\{|$))", "", clean, flags=re.DOTALL).strip()
    clean = re.sub(r"```(?:json)?\s*", "", clean).replace("```", "").strip()
    for candidate in (clean,):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if m:
        try:
            obj = json.loads(re.sub(r",\s*([\}\]])", r"\1", m.group(0)))
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
    return None


def validate_verdict(x: Dict[str, Any], model: str) -> Verdict:
    """Validate and normalize one model item using sentiment_worker.py rules."""
    if not isinstance(x, dict):
        raise ValueError("response is not a JSON object")
    vals = {k: int(x[k]) for k in SENTIMENT_VALUES}
    total = sum(vals.values())
    if any(v < 0 for v in vals.values()) or not (90 <= total <= 110):
        raise ValueError(f"invalid score sum {total}: {vals}")
    if total != 100:
        vals = {k: round(v * 100 / total) for k, v in vals.items()}
        top = max(vals, key=vals.get)
        vals[top] += 100 - sum(vals.values())
    sentiment = str(x["sentiment"]).strip().lower()
    if sentiment not in SENTIMENT_VALUES:
        raise ValueError(f"invalid sentiment: {sentiment!r}")
    if not all(vals[sentiment] > v for k, v in vals.items() if k != sentiment):
        raise ValueError(f"sentiment {sentiment!r} is not strictly largest in {vals}")
    intent = str(x["intent"]).strip().lower()
    if intent not in INTENT_VALUES:
        raise ValueError(f"invalid intent: {intent!r}")
    if not isinstance(x["incident"], bool) or not isinstance(x["multi_target"], bool):
        raise ValueError("incident and multi_target must be booleans")
    return Verdict(
        sentiment=sentiment,
        reason=_clean_reason(x["sentiment_reason"]),
        scores={"model": model, "multi_target": x["multi_target"], **vals},
        intent=intent,
        intent_reason=_clean_reason(x["intent_reason"]),
        incident=x["incident"],
    )


def is_unsure(verdict: Verdict) -> bool:
    scores = verdict.scores or {}
    return bool(scores.get("multi_target")) or max(int(scores[k]) for k in SENTIMENT_VALUES) < UNSURE_BELOW


def format_result_item(match_post_id: str, verdict: Verdict, intent_only: bool = False) -> Dict[str, Any]:
    """Build one BatchVerdictItem matching app/api/routes/sentiment.py."""
    if intent_only:
        return {
            "match_post_id": match_post_id,
            "intent": verdict.intent,
            "intent_reason": verdict.intent_reason,
        }
    return {
        "match_post_id": match_post_id,
        "sentiment": verdict.sentiment,
        "sentiment_reason": verdict.reason,
        "sentiment_scores": verdict.scores,
        "intent": verdict.intent,
        "intent_reason": verdict.intent_reason,
        "incident": verdict.incident,
    }


class OllamaWorkerClient:
    """Direct single-slot Ollama client tuned for RTX 5060 Ti 8GB (deepseek-r1:8b)."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
        num_ctx: int = DEFAULT_NUM_CTX,
        num_predict: Optional[int] = None,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        think: Optional[bool] = False,
        retries: int = 3,
    ):
        self.host = host.rstrip("/")
        self.generate_url = f"{self.host}/api/generate"
        self.chat_url = f"{self.host}/api/chat"
        self.tags_url = f"{self.host}/api/tags"
        self.ps_url = f"{self.host}/api/ps"
        self.model = model
        self.timeout = max(5, int(timeout))
        self.num_ctx = max(512, int(num_ctx))
        self.num_predict = max(64, int(num_predict)) if num_predict is not None else (512 if think is False else 2048)
        self.keep_alive = str(keep_alive).strip()
        self.think = think
        self.retries = max(1, int(retries))
        self.session = requests.Session()
        self._vram_reported = False

    def check_health(self, require_models: Optional[List[str]] = None) -> Tuple[bool, str]:
        try:
            resp = self.session.get(self.tags_url, timeout=10)
            if resp.status_code != 200:
                return False, f"Ollama /api/tags HTTP {resp.status_code}: {str(getattr(resp, 'text', ''))[:120]}"
            installed = {m.get("name", "") for m in (resp.json().get("models") or []) if isinstance(m, dict)}
            needed = require_models if require_models is not None else [self.model]
            for target in needed:
                aliases = {target}
                if ":" not in target:
                    aliases.add(f"{target}:latest")
                if not any(name in aliases for name in installed):
                    avail = ", ".join(sorted(installed)[:10]) or "(none)"
                    return False, f"Model '{target}' not found in Ollama (installed: {avail}). Run: ollama pull {target}"
            return True, f"Models ready: {', '.join(needed)}"
        except requests.RequestException as exc:
            return False, f"Cannot connect to Ollama at {self.host}: {exc}"

    def log_vram_status_once(self) -> None:
        if self._vram_reported:
            return
        try:
            resp = self.session.get(self.ps_url, timeout=5)
            if resp.status_code != 200:
                return
            for entry in resp.json().get("models") or []:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name") or entry.get("model") or "")
                if self.model.split(":", 1)[0] in name:
                    size = int(entry.get("size") or 0)
                    vram = int(entry.get("size_vram") or 0)
                    if size > 0:
                        gpu_pct = round(100.0 * vram / size, 1)
                        gb = size / (1024 ** 3)
                        vram_gb = vram / (1024 ** 3)
                        status = "100% GPU" if gpu_pct >= 99.0 else f"{gpu_pct}% GPU / {100.0 - gpu_pct:.1f}% CPU"
                        print(f"🎮 [Ollama VRAM] {name}: {vram_gb:.2f} / {gb:.2f} GB ({status})", flush=True)
                        if gpu_pct < 99.0:
                            print("   ⚠️ โมเดลบางส่วนล้นไปที่ CPU แนะนำลด --num-ctx (เช่น 2048) เพื่อให้อยู่ใน VRAM 8GB ทั้งหมด", flush=True)
                        self._vram_reported = True
                        return
        except Exception:
            pass

    def wait_until_healthy(self, max_wait: int = 20) -> bool:
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            ok, _ = self.check_health(require_models=[])
            if ok:
                return True
            time.sleep(2)
        return False

    def _post_with_think_fallback(self, url: str, payload: Dict[str, Any]) -> requests.Response:
        resp = self.session.post(url, json=payload, timeout=(10, self.timeout))
        if (resp.status_code == 400 and "think" in payload
                and "thinking" in str(getattr(resp, "text", "") or "").lower()):
            stripped = {k: v for k, v in payload.items() if k != "think"}
            resp = self.session.post(url, json=stripped, timeout=(10, self.timeout))
        return resp

    def _attempt_specs(self, model_name: str, content: str) -> List[Tuple[str, str, Dict[str, Any], Any]]:
        think_val = parse_think_flag(
            "auto" if self.think is None else ("true" if self.think else "false"),
            model_name,
        )
        user_text = f"Post:\n{content}"

        def build(temperature: float, seed: int) -> Dict[str, Any]:
            body: Dict[str, Any] = {
                "model": model_name,
                "system": PROMPT,
                "prompt": user_text,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": temperature,
                    "seed": seed,
                    "num_ctx": self.num_ctx,
                    "num_predict": self.num_predict,
                },
            }
            if think_val is not None:
                body["think"] = think_val
            if self.keep_alive:
                body["keep_alive"] = self.keep_alive
            return body

        # Stay on /api/generate with format="json" and vary (temperature, seed) on retries;
        # freeform /api/generate triggers <think> loops and /api/chat can return empty content.
        specs = [
            ("generate+json#1", self.generate_url, build(0.0, 42), lambda data: data.get("response", "")),
            ("generate+json#2", self.generate_url, build(0.2, 43), lambda data: data.get("response", "")),
            ("generate+json#3", self.generate_url, build(0.4, 44), lambda data: data.get("response", "")),
        ]
        return [specs[idx % len(specs)] for idx in range(self.retries)]

    def analyse_post(self, content: str, model_override: Optional[str] = None) -> Tuple[Optional[Verdict], str]:
        target_model = model_override or self.model
        last_error = "unknown_error"
        specs = self._attempt_specs(target_model, content)
        for attempt_idx, (mode, url, payload, extract_text) in enumerate(specs, start=1):
            try:
                resp = self._post_with_think_fallback(url, payload)
                if resp.status_code != 200:
                    last_error = f"{mode}: HTTP {resp.status_code} ({str(getattr(resp, 'text', ''))[:120]})"
                    print(f"  ⚠️ [Ollama {target_model} {mode} {attempt_idx}/{len(specs)}] {last_error}", flush=True)
                    if resp.status_code >= 500:
                        time.sleep(min(10, 2 * attempt_idx))
                    continue

                data = resp.json()
                self.log_vram_status_once()
                raw_text = extract_text(data)
                parsed = parse_json_object(raw_text)
                if parsed is None:
                    last_error = f"{mode}: unparseable_json ({str(raw_text)[:120]!r})"
                    print(f"  ⚠️ [Ollama {target_model} {mode} {attempt_idx}/{len(specs)}] {last_error}", flush=True)
                    continue

                verdict = validate_verdict(parsed, target_model)
                return verdict, ""
            except ValueError as exc:
                last_error = f"{mode}: validation_error ({exc})"
                print(f"  ⚠️ [Ollama {target_model} {mode} {attempt_idx}/{len(specs)}] {last_error}", flush=True)
            except (KeyError, TypeError) as exc:
                last_error = f"{mode}: schema_error ({type(exc).__name__}: {exc})"
                print(f"  ⚠️ [Ollama {target_model} {mode} {attempt_idx}/{len(specs)}] {last_error}", flush=True)
            except requests.exceptions.Timeout as exc:
                last_error = f"{mode}: timeout after {self.timeout}s ({type(exc).__name__})"
                print(f"  ⚠️ [Ollama {target_model} {mode} {attempt_idx}/{len(specs)}] {last_error}", flush=True)
                self.wait_until_healthy(max_wait=15)
            except requests.exceptions.RequestException as exc:
                last_error = f"{mode}: connection_error ({exc})"
                print(f"  ⚠️ [Ollama {target_model} {mode} {attempt_idx}/{len(specs)}] {last_error}", flush=True)
                time.sleep(min(10, 2 * attempt_idx))
                self.wait_until_healthy(max_wait=20)
        return None, last_error


class BlueEyeRestClient:
    """Client for /internal/sentiment/pending, /posts, and /results."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-Internal-Token": token, "Content-Type": "application/json"}
        self.session = requests.Session()
        self.last_fetch_error = False

    def fetch_pending_claimed(
        self,
        *,
        limit: int,
        date_from: str,
        date_to: str,
        missing: str,
        lease_seconds: int,
        project_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/internal/sentiment/pending"
        params: Dict[str, Any] = {
            "limit": max(1, min(200, int(limit))),
            "date_from": date_from,
            "date_to": date_to,
            "missing": missing,
            "claim": "true",
            "lease_seconds": max(30, min(3600, int(lease_seconds))),
        }
        if project_id:
            params["project_id"] = project_id
        for attempt in range(1, 4):
            try:
                r = self.session.get(url, params=params, headers=self.headers, timeout=60)
                if r.status_code == 200:
                    data = r.json()
                    return data if isinstance(data, list) else []
                print(f"[REST API] /pending HTTP {r.status_code} (attempt {attempt}/3): {r.text[:150]}", flush=True)
            except requests.RequestException as exc:
                print(f"[REST API] /pending error (attempt {attempt}/3): {exc}", flush=True)
            if attempt < 3:
                time.sleep(2)
        self.last_fetch_error = True
        return []

    def fetch_posts_page(
        self,
        *,
        date_from: str,
        date_to: str,
        page: int,
        page_size: int = 500,
        project_id: Optional[str] = None,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        url = f"{self.base_url}/internal/sentiment/posts"
        params: Dict[str, Any] = {
            "date_from": date_from,
            "date_to": date_to,
            "page": page,
            "page_size": max(1, min(500, int(page_size))),
        }
        if project_id:
            params["project_id"] = project_id
        for attempt in range(1, 4):
            try:
                r = self.session.get(url, params=params, headers=self.headers, timeout=60)
                if r.status_code == 200:
                    body = r.json()
                    return int(body.get("total", 0)), list(body.get("items") or [])
                print(f"[REST API] /posts HTTP {r.status_code} (attempt {attempt}/3): {r.text[:150]}", flush=True)
            except requests.RequestException as exc:
                print(f"[REST API] /posts error (attempt {attempt}/3): {exc}", flush=True)
            if attempt < 3:
                time.sleep(2)
        self.last_fetch_error = True
        return 0, []

    def write_results(self, results: List[Dict[str, Any]]) -> int:
        if not results:
            return 0
        url = f"{self.base_url}/internal/sentiment/results"
        total_written = 0
        for start in range(0, len(results), 500):
            chunk = results[start:start + 500]
            for attempt in range(1, 4):
                try:
                    r = self.session.post(url, headers=self.headers, json={"results": chunk}, timeout=60)
                    if r.status_code in (200, 201):
                        body = r.json()
                        updated = int(body.get("updated", len(chunk)))
                        not_found = body.get("not_found") or []
                        print(f"  ✅ [REST API] บันทึก {len(chunk)} แถว → updated={updated}"
                              + (f" (not_found={len(not_found)})" if not_found else ""), flush=True)
                        total_written += updated
                        break
                    print(f"  ❌ [REST API] /results HTTP {r.status_code} (attempt {attempt}/3): {r.text[:250]}", flush=True)
                    if r.status_code == 422:
                        break  # Schema rejection is deterministic; do not retry identical body
                except requests.RequestException as exc:
                    print(f"  ❌ [REST API] /results error (attempt {attempt}/3): {exc}", flush=True)
                if attempt < 3:
                    time.sleep(2)
        return total_written


class SentimentBackfillRunner:
    """Runs historical backfill matching sentiment_worker.py rules without rule:provider_failure."""

    def __init__(
        self,
        rest: BlueEyeRestClient,
        ollama: OllamaWorkerClient,
        *,
        batch_size: int = 10,
        fetch_limit: int = 80,
        lease_seconds: int = 600,
        save_db: bool = False,
        missing: str = "sentiment",
        project_id: Optional[str] = None,
        escalate_model: str = "",
        max_chars: int = MAX_CHARS,
        max_consecutive_failures: int = 5,
        fail_log_path: str = DEFAULT_FAIL_LOG,
        limit: Optional[int] = None,
    ):
        self.rest = rest
        self.ollama = ollama
        self.batch_size = max(1, int(batch_size))
        self.fetch_limit = max(1, min(200, int(fetch_limit)))
        self.lease_seconds = max(30, min(3600, int(lease_seconds)))
        self.save_db = save_db
        self.missing = missing
        self.project_id = project_id
        self.escalate_model = escalate_model.strip()
        self.max_chars = max(200, int(max_chars))
        self.max_consecutive_failures = max(1, int(max_consecutive_failures))
        self.fail_log_path = fail_log_path
        self.limit = max(1, int(limit)) if limit is not None else None

        self.total_posts_done = 0
        self.total_rows_written = 0
        self.total_failed_posts = 0
        self.total_escalated = 0
        self.consecutive_failures = 0
        self.failed_post_ids: Set[str] = set()
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def _record_failure(self, post_id: str, match_post_ids: List[str], error_reason: str) -> None:
        entry = {
            "timestamp": datetime.now(THAI_TIMEZONE).isoformat(timespec="seconds"),
            "post_id": post_id,
            "match_post_ids": match_post_ids,
            "model": self.ollama.model,
            "error": error_reason,
        }
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.fail_log_path)), exist_ok=True)
            with open(self.fail_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"  ⚠️ Cannot write failure log {self.fail_log_path}: {exc}", flush=True)

    def _flush_items(self, items: List[Dict[str, Any]]) -> int:
        if not items:
            return 0
        if self.save_db:
            return self.rest.write_results(items)
        print(f"  🔒 [DRY-RUN: ปิดการบันทึก] จำลองส่ง {len(items)} แถว (ไม่เขียนลง DB):", flush=True)
        for item in items:
            s = item.get("sentiment") or "(keep)"
            print(f"      📝 id={item['match_post_id']} sentiment={s} intent={item['intent']} "
                  f"incident={item.get('incident')} scores={item.get('sentiment_scores')}", flush=True)
        return len(items)

    def _process_rows(self, rows: List[Dict[str, Any]]) -> None:
        # Deduplicate by post_id just like sentiment_worker.py._process, skipping empty/already-failed posts
        ids_by_post: Dict[str, List[Tuple[str, bool]]] = {}
        posts: List[Dict[str, str]] = []
        for r in rows:
            pid = str(r.get("post_id") or "")
            mid = str(r.get("match_post_id") or "")
            if not pid or not mid or pid in self.failed_post_ids:
                continue
            content = (r.get("content") or "").strip()[:self.max_chars]
            if not content:
                continue
            if pid not in ids_by_post:
                if self.limit is not None and (self.total_posts_done + self.total_failed_posts + len(posts)) >= self.limit:
                    break
                posts.append({"post_id": pid, "content": content})
            intent_only = (self.missing == "intent" and r.get("sentiment") is not None)
            ids_by_post.setdefault(pid, []).append((mid, intent_only))

        if not posts:
            return

        for start in range(0, len(posts), self.batch_size):
            if self._stop_requested:
                break
            chunk = posts[start:start + self.batch_size]
            batch_started = time.perf_counter()
            pending_writes: List[Dict[str, Any]] = []
            print(f"\n🔄 [Backfill] Batch {start + 1}-{start + len(chunk)}/{len(posts)} posts "
                  f"(model={self.ollama.model})", flush=True)

            try:
                for idx, post in enumerate(chunk, start=1):
                    if self._stop_requested:
                        break
                    pid = post["post_id"]
                    t0 = time.perf_counter()
                    verdict, err = self.ollama.analyse_post(post["content"])

                    if self.escalate_model and (verdict is None or is_unsure(verdict)):
                        esc_verdict, esc_err = self.ollama.analyse_post(
                            post["content"], model_override=self.escalate_model)
                        if esc_verdict is not None:
                            verdict = esc_verdict
                            self.total_escalated += 1
                        elif verdict is None:
                            err = f"{err}; escalate {self.escalate_model}: {esc_err}"

                    dt = time.perf_counter() - t0
                    if verdict is None:
                        self.total_failed_posts += 1
                        self.consecutive_failures += 1
                        self.failed_post_ids.add(pid)
                        mids = [m for m, _ in ids_by_post[pid]]
                        self._record_failure(pid, mids, err)
                        print(f"  ⏭️ [{idx:02d}] post={pid[:18]:<18} | SKIPPED (ไม่บันทึก rule:provider_failure) | {err}",
                              flush=True)
                        if self.consecutive_failures >= self.max_consecutive_failures:
                            self.total_rows_written += self._flush_items(pending_writes)
                            pending_writes = []
                            raise OllamaUnavailableError(
                                f"หยุดทำงานชั่วคราว: Ollama ล้มเหลวติดต่อกัน {self.consecutive_failures} โพสต์ "
                                "(บันทึกโพสต์ที่สำเร็จก่อนหน้าแล้ว ส่วนที่พลาดยังคงรอวิเคราะห์อยู่)"
                            )
                        continue

                    self.consecutive_failures = 0
                    self.total_posts_done += 1
                    for mid, intent_only in ids_by_post[pid]:
                        pending_writes.append(format_result_item(mid, verdict, intent_only=intent_only))

                    icon = "🟢" if verdict.sentiment == "positive" else ("🔴" if verdict.sentiment == "negative" else "⚪")
                    print(
                        f"  [{idx:02d}] {icon} post={pid[:18]:<18} ({len(ids_by_post[pid])} row) | "
                        f"{verdict.sentiment.upper():<8} | intent={verdict.intent} | incident={verdict.incident} | "
                        f"{dt:.2f}s | Model={verdict.scores['model']}",
                        flush=True,
                    )
                    print(
                        f"       📊 POS {verdict.scores['positive']}% | NEU {verdict.scores['neutral']}% | "
                        f"NEG {verdict.scores['negative']}% | multi={verdict.scores['multi_target']}",
                        flush=True,
                    )
                    print(f"       💬 Sentiment: {verdict.reason}", flush=True)
                    print(f"       🎯 Intent   : {verdict.intent_reason}", flush=True)
            except KeyboardInterrupt:
                print("\n🛑 หยุดตามคำสั่ง (Ctrl+C) — กำลังบันทึกผลที่วิเคราะห์สำเร็จแล้วใน Batch นี้...", flush=True)
                self._stop_requested = True
                self.total_rows_written += self._flush_items(pending_writes)
                pending_writes = []

            if pending_writes:
                self.total_rows_written += self._flush_items(pending_writes)

            elapsed = time.perf_counter() - batch_started
            rate = len(chunk) / (elapsed / 60.0) if elapsed > 0 else 0.0
            print(f"  ⏱️ Batch done in {elapsed:.1f}s ({rate:.1f} posts/min) | "
                  f"total_posts={self.total_posts_done}, rows_written={self.total_rows_written}, skipped={self.total_failed_posts}",
                  flush=True)

            if self.limit is not None and (self.total_posts_done + self.total_failed_posts) >= self.limit:
                raise _LimitReached()

    def run(self, date_from: str, date_to: str) -> int:
        date_from = validate_date_str(date_from)
        date_to = validate_date_str(date_to)
        if date_from > date_to:
            raise ValueError(f"date_from ({date_from}) cannot be after date_to ({date_to})")

        started = time.perf_counter()
        aborted_reason = ""
        try:
            if self.save_db:
                while not self._stop_requested:
                    req_limit = self.fetch_limit
                    if self.limit is not None:
                        remaining = max(0, self.limit - (self.total_posts_done + self.total_failed_posts))
                        if remaining == 0:
                            break
                        req_limit = min(req_limit, max(remaining * 2, self.batch_size))
                    rows = self.rest.fetch_pending_claimed(
                        limit=req_limit,
                        date_from=date_from,
                        date_to=date_to,
                        missing=self.missing,
                        lease_seconds=self.lease_seconds,
                        project_id=self.project_id,
                    )
                    if not rows:
                        break
                    fresh = [r for r in rows if str(r.get("post_id") or "") not in self.failed_post_ids and (r.get("content") or "").strip()]
                    if not fresh:
                        break
                    self._process_rows(rows)
            else:
                page = 1
                seen_ids: Set[str] = set()
                while not self._stop_requested:
                    total_rows, rows = self.rest.fetch_posts_page(
                        date_from=date_from,
                        date_to=date_to,
                        page=page,
                        page_size=500,
                        project_id=self.project_id,
                    )
                    if page == 1:
                        print(f"[REST API] พบรายการรอวิเคราะห์ทั้งหมด {total_rows} แถวในช่วง {date_from}..{date_to}", flush=True)
                    if not rows:
                        break
                    filtered = []
                    for r in rows:
                        mid = str(r.get("match_post_id") or "")
                        if not mid or mid in seen_ids:
                            continue
                        seen_ids.add(mid)
                        if self.missing == "intent" and r.get("intent") is not None:
                            continue
                        filtered.append(r)
                    self._process_rows(filtered)
                    if page * 500 >= total_rows:
                        break
                    page += 1
        except _LimitReached:
            pass
        except OllamaUnavailableError as exc:
            aborted_reason = str(exc)
            print(f"\n❌ {aborted_reason}", flush=True)

        total_elapsed = time.perf_counter() - started
        rate = self.total_posts_done / (total_elapsed / 60.0) if total_elapsed > 0 else 0.0
        print("\n" + "=" * 75, flush=True)
        print("📊 สรุปผลการวิเคราะห์ย้อนหลัง (sentiment_backfill / Ollama Only)", flush=True)
        print(f"   • ช่วงวันที่         : {date_from} ถึง {date_to} (missing={self.missing})", flush=True)
        print(f"   • โมเดล             : {self.ollama.model} @ {self.ollama.host}"
              + (f" (escalate={self.escalate_model})" if self.escalate_model else ""), flush=True)
        print(f"   • โหมดบันทึก        : {'LIVE (/pending?claim=true -> DB)' if self.save_db else 'DRY-RUN (/posts -> ไม่เขียน DB)'}", flush=True)
        print(f"   • วิเคราะห์สำเร็จ     : {self.total_posts_done} โพสต์ ({rate:.1f} posts/min)", flush=True)
        print(f"   • บันทึก/จำลองแล้ว   : {self.total_rows_written} แถว", flush=True)
        if self.escalate_model:
            print(f"   • ส่งต่อ escalate   : {self.total_escalated} โพสต์", flush=True)
        print(f"   • ข้ามเพราะ error    : {self.total_failed_posts} โพสต์ (ไม่สร้าง rule:provider_failure)", flush=True)
        if self.total_failed_posts > 0:
            print(f"   • Log รายการที่ข้าม  : {self.fail_log_path}", flush=True)
        print(f"   • เวลาทั้งหมด        : {total_elapsed:.1f} วินาที", flush=True)
        print("=" * 75, flush=True)

        if aborted_reason:
            return 2
        if self.rest.last_fetch_error:
            return 1
        return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    env_save = os.environ.get("SAVE_DB", "false").strip().lower() in ("true", "1", "yes")
    parser = argparse.ArgumentParser(
        description="Ollama historical backfill worker matching api-blueeye sentiment_worker.py schema."
    )
    parser.add_argument("--from-date", dest="date_from", metavar="YYYY-MM-DD", help="Start date (BKK inclusive)")
    parser.add_argument("--to-date", dest="date_to", metavar="YYYY-MM-DD", help="End date (BKK inclusive)")
    parser.add_argument("--days", type=int, help="Backfill last N days up to yesterday (if --from-date omitted)")
    parser.add_argument("--project-id", default=None, help="Optional single project_id filter")
    parser.add_argument("--missing", choices=("sentiment", "intent"), default="sentiment",
                        help="Queue to backfill (default: sentiment)")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_BACKFILL_MODEL", "").strip() or DEFAULT_MODEL)
    parser.add_argument("--escalate-model", default="", help="Optional second Ollama model when score < 60 or multi_target")
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "").strip() or "http://localhost:11434")
    parser.add_argument("--timeout", type=int, default=_env_int("OLLAMA_DEEP_TIMEOUT", DEFAULT_TIMEOUT))
    parser.add_argument("--num-ctx", type=int, default=_env_int("OLLAMA_NUM_CTX", DEFAULT_NUM_CTX))
    parser.add_argument("--num-predict", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS)
    parser.add_argument("--keep-alive", default=os.environ.get("OLLAMA_KEEP_ALIVE", "").strip() or DEFAULT_KEEP_ALIVE)
    parser.add_argument("--think", default=os.environ.get("OLLAMA_THINK", "").strip() or "false",
                        help="Thinking toggle: false, true, or auto (default: false)")
    parser.add_argument("--retries", type=int, default=3, help="Attempts per post before skipping (default: 3)")
    parser.add_argument("--batch-size", type=int, default=_env_int("OLLAMA_BACKFILL_BATCH_SIZE", 10),
                        help="Posts analyzed per write flush (default: 10)")
    parser.add_argument("--fetch-limit", type=int, default=80,
                        help="Rows claimed per /internal/sentiment/pending poll in live mode (1-200, default: 80)")
    parser.add_argument("--lease-seconds", type=int, default=600,
                        help="Claim lease duration in live mode (30-3600, default: 600)")
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    parser.add_argument("--fail-log", default=DEFAULT_FAIL_LOG)
    parser.add_argument("--limit", type=int, default=None, help="Stop after N unique posts (useful for test runs)")

    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument("--save-db", dest="save_db", action="store_true", default=env_save,
                            help="Write results via /internal/sentiment/results")
    save_group.add_argument("--dry-run", dest="save_db", action="store_false",
                            help="Read-only dry-run without writing to DB")

    args = parser.parse_args(argv)

    if args.days is not None:
        if args.date_from or args.date_to:
            parser.error("Use either --days OR --from-date/--to-date, not both")
        if args.days < 1:
            parser.error("--days must be >= 1")
        today = datetime.now(THAI_TIMEZONE).date()
        end_date = today - timedelta(days=1)
        start_date = end_date - timedelta(days=args.days - 1)
        args.date_from = start_date.isoformat()
        args.date_to = end_date.isoformat()
    else:
        if not args.date_from or not args.date_to:
            parser.error("Specify --from-date YYYY-MM-DD and --to-date YYYY-MM-DD (or --days N)")
        try:
            args.date_from = validate_date_str(args.date_from)
            args.date_to = validate_date_str(args.date_to)
        except ValueError as exc:
            parser.error(str(exc))
        if args.date_from > args.date_to:
            parser.error("--from-date cannot be after --to-date")

    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    base_url = os.environ.get("BE_API_BASE_URL", "").strip() or DEFAULT_BASE_URL
    token = os.environ.get("BE_API_TOKEN", "").strip()
    if not token:
        print("❌ BE_API_TOKEN is not set in .env", flush=True)
        return 2

    think_flag = parse_think_flag(args.think, args.model)
    ollama = OllamaWorkerClient(
        host=args.host,
        model=args.model,
        timeout=args.timeout,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        keep_alive=args.keep_alive,
        think=think_flag,
        retries=args.retries,
    )
    rest = BlueEyeRestClient(base_url=base_url, token=token)

    print("🤖 SENTIMENT BACKFILL WORKER (sentiment_worker.py rules + Ollama Only)", flush=True)
    print("=" * 75, flush=True)
    print(f"📅 ช่วงวันที่    : {args.date_from} ถึง {args.date_to} (missing={args.missing}"
          + (f", project={args.project_id}" if args.project_id else "") + ")", flush=True)
    print(f"🧠 โมเดล        : {args.model} @ {args.host} (think={think_flag}, num_ctx={args.num_ctx})", flush=True)
    print(f"💾 โหมดบันทึก   : {'LIVE (claim=true -> /results)' if args.save_db else 'DRY-RUN (/posts -> ไม่เขียน DB)'}", flush=True)

    needed = [args.model] + ([args.escalate_model] if args.escalate_model.strip() else [])
    ok, msg = ollama.check_health(require_models=needed)
    if not ok:
        print(f"❌ {msg}", flush=True)
        return 2
    print(f"✅ {msg}", flush=True)

    runner = SentimentBackfillRunner(
        rest=rest,
        ollama=ollama,
        batch_size=args.batch_size,
        fetch_limit=args.fetch_limit,
        lease_seconds=args.lease_seconds,
        save_db=args.save_db,
        missing=args.missing,
        project_id=args.project_id,
        escalate_model=args.escalate_model,
        max_chars=args.max_chars,
        max_consecutive_failures=args.max_consecutive_failures,
        fail_log_path=args.fail_log,
        limit=args.limit,
    )

    def _handle_sigterm(signum, frame):
        runner.request_stop()

    try:
        signal.signal(signal.SIGTERM, _handle_sigterm)
    except (ValueError, AttributeError):
        pass

    return runner.run(args.date_from, args.date_to)


if __name__ == "__main__":
    sys.exit(main())
