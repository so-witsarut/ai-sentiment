# coding=utf-8
"""
Sentiment Analysis System (REST API)
Using Ollama (qwen3-8b-instruct) Fast Triage + Gemini Deep Analysis Cascade
"""

import os
import argparse
import re
import json
import time
import sys
import math
import random
import requests
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Tuple, Union

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

# REST keyword analysis uses metadata supplied by the API.
GLOBAL_PROJECT_RESOLVER = None

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BE_API_TOKEN = os.environ.get("BE_API_TOKEN", "")
BE_API_BASE_URL = os.environ.get("BE_API_BASE_URL", "https://api.blueeye.io/api/v1")

# Reconfigure stdout for UTF-8 output on Windows
try:
    if hasattr(sys.stdout, 'reconfigure') and sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def get_keyword_context(text, keyword, window=150, max_fallback_length=400):
    """
    Extract text context surrounding the Keyword
    """
    if not text:
        return ""
        
    if not keyword or keyword not in text:
        return text[:max_fallback_length] + ("..." if len(text) > max_fallback_length else "")

    start_idx = text.find(keyword)
    left_bound = max(0, start_idx - window)
    right_bound = min(len(text), start_idx + len(keyword) + window)
    
    sliced_text = text[left_bound:right_bound]
    
    if left_bound > 0:
        sliced_text = "..." + sliced_text
    if right_bound < len(text):
        sliced_text = sliced_text + "..."
        
    return sliced_text


def validate_date_str(date_str):
    """Validate YYYY-MM-DD date string (ป้องกัน SQL injection และรูปแบบวันที่ไม่ถูกต้อง)"""
    from datetime import date as dt_date
    if date_str is None:
        raise ValueError("Date string cannot be None")
    if isinstance(date_str, (datetime, dt_date)):
        return date_str.strftime("%Y-%m-%d")
    s = str(date_str).strip()
    if not s:
        raise ValueError("Date string cannot be empty")
    m = re.match(r"^(\d{4}-\d{2}-\d{2})(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$", s)
    if not m:
        raise ValueError(f"Invalid date format (expected YYYY-MM-DD or ISO timestamp): {date_str}")
    date_part = m.group(1)
    try:
        datetime.strptime(date_part, "%Y-%m-%d")
        return date_part
    except ValueError:
        raise ValueError(f"Invalid calendar date: {date_str}")


# =============================================================================
# Typed Probabilistic Decision & System 1 AI Engine (Zero String Tax)
# =============================================================================
def parse_bounded_int(env_name: str, default: int, min_val: int = 1, max_val: int = 256) -> int:
    val_str = os.environ.get(env_name)
    if val_str is None or val_str.strip() == "":
        return default
    try:
        val = int(val_str.strip())
    except (ValueError, TypeError):
        raise ValueError(f"Configuration error: {env_name} must be a valid integer, got {val_str!r}")
    if val < min_val or val > max_val:
        raise ValueError(f"Configuration error: {env_name} must be between {min_val} and {max_val}, got {val}")
    return val

ENABLE_JEV_HYBRID = os.environ.get("ENABLE_JEV_HYBRID", "true").lower() in ("true", "1", "yes")
ENABLE_PROBABILISTIC_MODE = os.environ.get("ENABLE_PROBABILISTIC_MODE", "true").lower() in ("true", "1", "yes")
BYPASS_LOCAL_TRIAGE = os.environ.get("BYPASS_LOCAL_TRIAGE", "true").lower() in ("true", "1", "yes")

# Jev acceptance threshold is code-owned constant exactly 0.65, inclusive
JEV_ACCEPTANCE_THRESHOLD = 0.65
HYBRID_PROMPT_V2 = os.environ.get("HYBRID_PROMPT_V2", "false").lower() in ("true", "1", "yes")

JEV_MODEL = os.environ.get("JEV_MODEL", "typesafe/jev-1.13")
_raw_jev_fallbacks = os.environ.get("JEV_FALLBACK_MODELS", "~typesafe/jev-latest,typesafe/jev-latest")
JEV_FALLBACK_MODELS = [m.strip() for m in _raw_jev_fallbacks.split(",") if m.strip()]
JEV_API_TIMEOUT = parse_bounded_int("JEV_API_TIMEOUT", 20, min_val=1, max_val=300)
JEV_MAX_RETRIES = parse_bounded_int("JEV_MAX_RETRIES", 3, min_val=0, max_val=10)

DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek/deepseek-v4-flash-0731")
DEEPSEEK_API_TIMEOUT = parse_bounded_int("DEEPSEEK_API_TIMEOUT", 45, min_val=1, max_val=300)
DEEPSEEK_MAX_RETRIES = parse_bounded_int("DEEPSEEK_MAX_RETRIES", 2, min_val=0, max_val=10)
DEEPSEEK_MAX_CONCURRENCY = parse_bounded_int("DEEPSEEK_MAX_CONCURRENCY", 4, min_val=1, max_val=256)
DEEPSEEK_MAX_TOKENS = parse_bounded_int("DEEPSEEK_MAX_TOKENS", 400, min_val=400, max_val=2048)

CONCURRENT_WORKERS = parse_bounded_int("CONCURRENT_WORKERS", 16, min_val=1, max_val=256)
BATCH_SIZE = parse_bounded_int("BATCH_SIZE", 100, min_val=1, max_val=1000)
MAX_IN_FLIGHT = parse_bounded_int("MAX_IN_FLIGHT", 100, min_val=1, max_val=256)

JEV_CASCADE_TIMEOUT = min(120.0, JEV_API_TIMEOUT * (1 + JEV_MAX_RETRIES) + 10.0)
DEEPSEEK_CASCADE_TIMEOUT = min(180.0, DEEPSEEK_API_TIMEOUT * (1 + DEEPSEEK_MAX_RETRIES) + 15.0)


def _retry_within_deadline(attempt: int, deadline: float, response=None) -> bool:
    """Wait for a retry without extending the provider cascade deadline."""
    retry_after = response.headers.get("Retry-After") if response is not None else None
    try:
        numeric_after = float(retry_after)
        delay = min(numeric_after, 10.0) if math.isfinite(numeric_after) and numeric_after >= 0 else None
    except (TypeError, ValueError):
        delay = None
    if delay is None:
        delay = min(0.5 * (2 ** attempt) + random.uniform(0.0, 0.25), 5.0)
    if delay >= deadline - time.monotonic():
        return False
    time.sleep(delay)
    return True


def _log_batch_timing(source, batch_start, batch_end, count, completed, fallback_count, save_db, started):
    elapsed = time.perf_counter() - started
    rate = count / elapsed if elapsed > 0 else 0.0
    action = "updated" if save_db else "dry_run"
    print(f"  ⏱️ [{source}] Batch {batch_start + 1}-{batch_end}: {count} posts in {elapsed:.1f}s "
          f"({rate:.2f} posts/s) | {action}={completed} | fallback_neutral={fallback_count}")


def _usage_int(value):
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _usage_float(value, allow_negative=False):
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number) or (number < 0 and not allow_negative):
        return 0.0
    return number


def _extract_provider_usage(response_data):
    """Normalize OpenRouter and Gemini token usage without trusting response content."""
    if not isinstance(response_data, dict):
        return {"input": 0, "output": 0, "cached": 0, "cost": 0.0, "cache_discount": 0.0}
    usage = response_data.get("usage") or response_data.get("usageMetadata") or {}
    if not isinstance(usage, dict):
        return {"input": 0, "output": 0, "cached": 0, "cost": 0.0, "cache_discount": 0.0}
    prompt_details = usage.get("prompt_tokens_details") or {}
    if not isinstance(prompt_details, dict):
        prompt_details = {}
    return {
        "input": _usage_int(usage.get("prompt_tokens", usage.get("input_tokens", usage.get("promptTokenCount", 0)))),
        "output": _usage_int(usage.get("completion_tokens", usage.get("output_tokens", usage.get("candidatesTokenCount", 0)))),
        "cached": _usage_int(prompt_details.get("cached_tokens", usage.get("cached_tokens", usage.get("cachedContentTokenCount", 0)))),
        "cost": _usage_float(usage.get("cost", response_data.get("cost", 0.0))),
        "cache_discount": _usage_float(
            usage.get("cache_discount", response_data.get("cache_discount", 0.0)), allow_negative=True
        )
    }


class _BatchUsageMetrics:
    """Thread-safe, per-analyzer-call provider usage accumulator."""

    def __init__(self):
        self._lock = threading.Lock()
        self._providers = {}

    def record(self, route, model, provider, retry, response_data=None):
        usage = _extract_provider_usage(response_data)
        key = (str(route), str(model or "unknown"), str(provider or "unknown"))
        with self._lock:
            row = self._providers.setdefault(key, {
                "route": key[0], "model": key[1], "provider": key[2],
                "requests": 0, "retries": 0, "input_tokens": 0,
                "output_tokens": 0, "cached_tokens": 0, "cost": 0.0,
                "cache_discount": 0.0
            })
            row["requests"] += 1
            row["retries"] += int(bool(retry))
            row["input_tokens"] += usage["input"]
            row["output_tokens"] += usage["output"]
            row["cached_tokens"] += usage["cached"]
            row["cost"] += usage["cost"]
            row["cache_discount"] += usage["cache_discount"]

    def summary(self, results):
        with self._lock:
            providers = [dict(row) for row in self._providers.values()]
        providers.sort(key=lambda row: (row["route"], row["model"], row["provider"]))
        routes = {"jev": 0, "deepseek": 0, "rule": 0}
        for result in results:
            route = result.get("route") if isinstance(result, dict) else None
            if route:
                routes[route] = routes.get(route, 0) + 1
        return {
            "routes": routes,
            "providers": providers,
            "input_tokens": sum(row["input_tokens"] for row in providers),
            "output_tokens": sum(row["output_tokens"] for row in providers),
            "cached_tokens": sum(row["cached_tokens"] for row in providers),
            "cost": sum(row["cost"] for row in providers),
            "cache_discount": sum(row["cache_discount"] for row in providers),
            "requests": sum(row["requests"] for row in providers),
            "retries": sum(row["retries"] for row in providers)
        }


def _log_cost_telemetry(telemetry):
    routes = telemetry.get("routes", {})
    print("  💰 [AI Usage] "
          f"routes=jev:{routes.get('jev', 0)},deepseek:{routes.get('deepseek', 0)},rule:{routes.get('rule', 0)} | "
          f"input={telemetry.get('input_tokens', 0)} output={telemetry.get('output_tokens', 0)} "
          f"cached={telemetry.get('cached_tokens', 0)} retries={telemetry.get('retries', 0)} "
          f"cost=${telemetry.get('cost', 0.0):.6f} cache_discount=${telemetry.get('cache_discount', 0.0):.6f}")
    for row in telemetry.get("providers", []):
        print("     ↳ "
              f"{row['route']} model={row['model']} provider={row['provider']} requests={row['requests']} "
              f"input={row['input_tokens']} output={row['output_tokens']} "
              f"cached={row['cached_tokens']} retries={row['retries']} cost=${row['cost']:.6f}")


PROBABILISTIC_SYSTEM_PROMPT = (
    "Classify Thai social text toward the specified target. Return JSON only; never explain.\n"
    "POSITIVE=praise/satisfaction/recommendation; NEUTRAL=facts/news/questions/PR or unrelated sentiment; "
    "NEGATIVE=complaint/criticism/damage; AMBIGUOUS_OR_IRONY=sarcasm/mixed or unclear tone.\n"
    "Use semantic evidence about the target. Four probabilities must be finite, 0..1, and sum to 1.00.\n"
    "entity_found is true only when the text directly refers to the target.\n"
    "{\n"
    '  "probabilities": {\n'
    '    "POSITIVE": <float 0.00-1.00>,\n'
    '    "NEUTRAL": <float 0.00-1.00>,\n'
    '    "NEGATIVE": <float 0.00-1.00>,\n'
    '    "AMBIGUOUS_OR_IRONY": <float 0.00-1.00>\n'
    "  },\n"
    '  "entity_found": <boolean>\n'
    "}"
)

# The hybrid path has a narrower contract than the legacy probabilistic mode.
HYBRID_SYSTEM_PROMPT = (
    "Assess Thai social text toward Target only. Keywords retrieved the post; a keyword match alone "
    "does not prove relevance. Set entity_found=true only if the text names Target or gives a clear, "
    "attributable link to it. If unrelated, set entity_found=false and NEUTRAL=1. "
    "Sentiment about a topic or another entity is not sentiment about Target. "
    "Return JSON only: {\"probabilities\":{\"POSITIVE\":number,\"NEUTRAL\":number,"
    "\"NEGATIVE\":number,\"AMBIGUOUS_OR_IRONY\":number},\"entity_found\":boolean}. "
    "Probabilities must be finite, between 0 and 1, and sum to 1. "
    "POSITIVE=praise; NEGATIVE=criticism; NEUTRAL=facts/no target sentiment; "
    "AMBIGUOUS_OR_IRONY=sarcasm or mixed/unclear sentiment."
)

KEYWORD_SYSTEM_PROMPT = (
    "Assess Thai social text toward the listed Target keywords only. "
    "A keyword mention alone does not establish sentiment. Match the intended place, brand, or topic, not an unrelated homonym. "
    "Emotion about another person, hardship, or event is neutral for Target unless it explicitly evaluates Target. "
    "A venue or institution named only as the location of an event does not inherit criticism of the event, film, cinema operator, or another person. "
    "Set entity_found=true when the text refers to any Target keyword in its intended sense, even in neutral factual text; "
    "otherwise set entity_found=false and NEUTRAL=1. "
    "Return JSON only: {\"probabilities\":{\"POSITIVE\":number,\"NEUTRAL\":number,"
    "\"NEGATIVE\":number,\"AMBIGUOUS_OR_IRONY\":number},\"entity_found\":boolean}. "
    "Probabilities must be finite, between 0 and 1, and sum to 1. "
    "POSITIVE=explicit independent praise, satisfaction, or support toward Target from a speaker, not the advertiser's own slogan; "
    "NEGATIVE=explicit complaint or criticism attributable to Target; hardship, crime news, or bad events merely mentioning Target are neutral; "
    "NEUTRAL=factual news, job ads, PR, ads, sales slogans, calls to buy, and the seller's own praise, even with words like 'love' or 'great'; "
    "for example 'Love X? Shop today' is an ad, not a consumer endorsement; "
    "AMBIGUOUS_OR_IRONY=sarcasm or mixed/unclear sentiment."
)


# Strict Schema Definition (Pydantic contract if available)
try:
    from pydantic import BaseModel, Field
    from enum import Enum

    class DecisionChoice(str, Enum):
        POSITIVE = "POSITIVE"
        NEUTRAL = "NEUTRAL"
        NEGATIVE = "NEGATIVE"
        AMBIGUOUS_OR_IRONY = "AMBIGUOUS_OR_IRONY"

    class ProbabilisticSentimentResult(BaseModel):
        entity_found: bool = Field(default=True, description="True if Target Entity is relevantly mentioned")
        probabilities: dict = Field(
            description="Probability distribution across 4 choices summing strictly to 1.0"
        )
except ImportError:
    pass


def parse_probabilistic_response(parsed_dict):
    """Parse normalized float probabilities (0.00-1.00) from model JSON response."""
    if not isinstance(parsed_dict, dict):
        return None

    probs_raw = parsed_dict.get("probabilities")
    if probs_raw is None:
        for k in parsed_dict:
            if str(k).strip().lower() == "probabilities":
                probs_raw = parsed_dict[k]
                break
    if probs_raw is None:
        # Check if any nested dict has positive/negative/neutral keys
        for k, v in parsed_dict.items():
            if isinstance(v, dict):
                v_lower = {str(vk).strip().lower() for vk in v.keys()}
                if "positive" in v_lower or "pos" in v_lower:
                    probs_raw = v
                    break

    if isinstance(probs_raw, str):
        try:
            probs_raw = json.loads(probs_raw)
        except Exception:
            pass
    if not isinstance(probs_raw, dict):
        probs_raw = parsed_dict

    # Lowercase-normalized dictionary for robust key lookup
    clean_probs = {str(k).strip().lower(): v for k, v in probs_raw.items()}

    def _get_val(*keys):
        for k in keys:
            k_lower = str(k).strip().lower()
            if k_lower in clean_probs and clean_probs[k_lower] is not None:
                val = clean_probs[k_lower]
                try:
                    if isinstance(val, str):
                        val = val.replace("%", "").strip()
                    float_val = float(val)
                    if math.isnan(float_val) or math.isinf(float_val):
                        return 0.0
                    return max(0.0, float_val)
                except (ValueError, TypeError):
                    continue
        return 0.0

    pos = _get_val("POSITIVE", "positive", "pos", "positive_percent")
    neg = _get_val("NEGATIVE", "negative", "neg", "negative_percent")
    neu = _get_val("NEUTRAL", "neutral", "neu", "neutral_percent")
    irony = _get_val("AMBIGUOUS_OR_IRONY", "ambiguous_or_irony", "irony", "ambiguous", "sarcasm")

    total = pos + neg + neu + irony
    if total > 1.5:
        pos = pos / total
        neg = neg / total
        neu = neu / total
        irony = irony / total
    elif total <= 0:
        neu = 1.0
        pos, neg, irony = 0.0, 0.0, 0.0
    else:
        pos = pos / total
        neg = neg / total
        neu = neu / total
        irony = irony / total

    raw_entity = parsed_dict.get("entity_found")
    if raw_entity is None:
        for k in parsed_dict:
            if str(k).strip().lower() in ("entity_found", "entityfound", "target_found", "entity_present", "target_present"):
                raw_entity = parsed_dict[k]
                break
    if raw_entity is None:
        raw_entity = True

    if isinstance(raw_entity, str):
        entity_found = raw_entity.strip().lower() in ("true", "1", "yes")
    else:
        entity_found = bool(raw_entity)

    return {
        "probabilities": {
            "POSITIVE": round(pos, 4),
            "NEUTRAL": round(neu, 4),
            "NEGATIVE": round(neg, 4),
            "AMBIGUOUS_OR_IRONY": round(irony, 4)
        },
        "entity_found": entity_found
    }


def resolve_policy(probabilities, entity_found=True):
    """
    Code Owns Policy: Convert continuous probability vector into discrete
    sentiment scores (-100, 0, 100) and integer distribution percentages (0-100%).
    Conservative brand monitoring allocates high irony/sarcasm toward negative.
    """
    if not entity_found or not probabilities:
        return {"sentiment": "neutral", "score": 0, "pos": 0, "neg": 0, "neu": 100, "irony_score": 0}

    clean_probs = {str(k).strip().lower(): v for k, v in probabilities.items()}

    def _get_p(*keys):
        for k in keys:
            k_lower = str(k).strip().lower()
            if k_lower in clean_probs and clean_probs[k_lower] is not None:
                try:
                    val = clean_probs[k_lower]
                    if isinstance(val, str):
                        val = val.replace("%", "").strip()
                    v = float(val)
                    if not math.isnan(v) and not math.isinf(v):
                        return max(0.0, v)
                except (ValueError, TypeError):
                    continue
        return 0.0

    pos = _get_p("POSITIVE", "positive", "pos")
    neg = _get_p("NEGATIVE", "negative", "neg")
    neu = _get_p("NEUTRAL", "neutral", "neu")
    irony = _get_p("AMBIGUOUS_OR_IRONY", "ambiguous_or_irony", "irony", "ambiguous", "sarcasm")

    # If inputs were percentages (> 1.5), scale down
    p_total = pos + neg + neu + irony
    if p_total > 1.5:
        pos /= p_total
        neg /= p_total
        neu /= p_total
        irony /= p_total

    # Conservative brand monitoring: allocate high irony/sarcasm to negative
    effective_neg = neg + (irony * 0.7)
    effective_pos = pos

    if effective_pos > effective_neg and effective_pos > neu:
        sentiment = "positive"
        score = 100
    elif effective_neg > effective_pos and effective_neg > neu:
        sentiment = "negative"
        score = -100
    elif effective_pos == effective_neg and effective_pos > neu:
        sentiment = "negative"
        score = -100
    else:
        sentiment = "neutral"
        score = 0

    total = effective_pos + effective_neg + neu
    if total <= 0:
        pos_pct, neg_pct, neu_pct = 0, 0, 100
    else:
        pos_pct = int(round(effective_pos / total * 100))
        neg_pct = int(round(effective_neg / total * 100))
        if pos_pct + neg_pct > 100:
            neg_pct = 100 - pos_pct
        neu_pct = max(0, 100 - pos_pct - neg_pct)

    irony_pct = max(0, min(100, int(round(irony * 100))))

    return {
        "sentiment": sentiment,
        "score": score,
        "pos": pos_pct,
        "neg": neg_pct,
        "neu": neu_pct,
        "irony_score": irony_pct
    }


def generate_synthetic_reason(target, pos, neg, neu, irony, entity_found=True):
    """
    Synthetic Reason Engine: Generates natural Thai explanation deterministically
    without LLM output token cost (Zero String Tax).
    """
    def _to_float(v):
        try:
            if isinstance(v, str):
                v = v.replace("%", "").strip()
            f = float(v)
            return 0.0 if (math.isnan(f) or math.isinf(f)) else max(0.0, f)
        except Exception:
            return 0.0

    pos = _to_float(pos)
    neg = _to_float(neg)
    neu = _to_float(neu)
    irony = _to_float(irony)

    t = pos + neg + neu + irony
    if t > 1.5:
        pos /= t
        neg /= t
        neu /= t
        irony /= t

    target_name = sanitize_target(target)

    if not entity_found:
        return f"ไม่พบการกล่าวถึงหรือความคิดเห็นต่อ {target_name} โดยตรง"

    if irony >= 0.30:
        if neg > pos:
            return f"พบการใช้ภาษาประชดประชันหรือเสียดสี โดยมีแนวโน้มเชิงลบต่อ {target_name}"
        elif pos > neg:
            return f"พบการใช้ภาษาเสียดสีหรือสำนวนกึ่งเล่นกึ่งจริง โดยรวมมีความชื่นชม {target_name}"
        else:
            return f"ข้อความมีความประชดประชันหรือกำกวม ไม่สามารถระบุเจตนาที่ชัดเจนต่อ {target_name} ได้"

    if pos >= 0.70:
        return f"ผู้ใช้งานแสดงความชื่นชม ประทับใจ หรือให้คะแนนเชิงบวกต่อ {target_name} อย่างชัดเจน"

    if pos >= 0.40 and pos > neg:
        if neg >= 0.20:
            return f"ผู้ใช้งานมีความรู้สึกเชิงบวกต่อ {target_name} เป็นหลัก แต่มีข้อติติงหรือคำถามบางส่วน"
        return f"เนื้อหามีโทนโน้มเอียงไปในทิศทางที่ดีหรือสนับสนุน {target_name}"

    if neg >= 0.70:
        return f"ผู้ใช้งานร้องเรียน ตำหนิ หรือแสดงความไม่พอใจต่อ {target_name} อย่างรุนแรง"

    if neg >= 0.40 and neg > pos:
        if pos >= 0.20:
            return f"ผู้ใช้งานแสดงความไม่พอใจหรือติติง {target_name} แม้จะมีบางจุดที่กล่าวถึงเชิงบวก"
        return f"พบข้อร้องเรียน ความกังวล หรือความคิดเห็นเชิงลบต่อ {target_name}"

    if neu >= 0.65:
        return f"รายงานข้อเท็จจริง ข่าวสาร ประชาสัมพันธ์ หรือสอบถามข้อมูลทั่วไปเกี่ยวกับ {target_name}"

    if pos >= 0.25 and neg >= 0.25:
        return f"มีความคิดเห็นหลากหลายทั้งเชิงบวกและเชิงลบผสมผสานกันต่อ {target_name}"

    return f"ข้อความทั่วไปหรือข่าวสารที่กล่าวถึง {target_name} ในบริบทเป็นกลาง"


def is_placeholder_target(target) -> bool:
    """Check if target resolution resulted in an empty or generic placeholder."""
    if target is None:
        return True
    s = str(target).strip().lower()
    return s in ("", "the target entity", "target entity", "unknown", "none", "null", "เป้าหมายที่ระบุ")


def sanitize_target(target: str, max_chars: int = 100) -> str:
    """Sanitize target entity name for synthetic reasons (strip HTML/control chars, cap length)."""
    if not target or is_placeholder_target(target):
        return "เป้าหมายที่ระบุ"
    cleaned = re.sub(r"<[^>]+>", " ", str(target))
    cleaned = re.sub(r"[\r\n\t\x00-\x1f\x7f-\x9f]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(' "\' \t\r\n')
    if not cleaned or is_placeholder_target(cleaned):
        return "เป้าหมายที่ระบุ"
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].strip() + "..."
        if max_chars > 3:
            cleaned = cleaned[: max_chars - 3].rstrip() + "..."
        else:
            cleaned = cleaned[:max_chars]
    return cleaned


def cap_text(text: str, max_chars: int = 8000, keyword: str = "", target: str = "") -> str:
    """Keep the target/keyword neighborhoods and post edges within a hard character cap."""
    text = text or ""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 5:
        return text[:max_chars]

    separator = " ... "
    folded = text.casefold()
    intervals = []
    for term, width in ((target, max_chars // 3), (keyword, max_chars // 4)):
        if not term:
            continue
        index = folded.find(str(term).casefold())
        if index < 0:
            continue
        width = max(len(term), width)
        start = max(0, index - (width - len(term)) // 2)
        end = min(len(text), start + width)
        intervals.append((max(0, end - width), end))

    if not intervals:
        head = (max_chars - len(separator)) // 2
        return text[:head] + separator + text[-(max_chars - len(separator) - head):]

    edge_width = max(1, max_chars // 5)
    intervals.extend(((0, edge_width), (len(text) - edge_width, len(text))))
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    excerpt = separator.join(text[start:end] for start, end in merged)
    return excerpt[:max_chars]


def _legacy_cap_text(text: str, max_chars: int = 8000, keyword: str = "") -> str:
    """Retain the current live prompt excerpt until prompt-v2 meets its canary gate."""
    if not text or len(text) <= max_chars:
        return text or ""
    if keyword and keyword in text:
        index = text.find(keyword)
        return f"{text[:1500]} ... {text[max(0, index - 1500):min(len(text), index + len(keyword) + 1500)]} ... {text[-1500:]}"
    half = (max_chars - 10) // 2
    return f"{text[:half]} ... {text[-half:]}"


def _sentiment_target(explicit_target, project_name, company_name, keywords, resolved_project, tracking_kind=""):
    """Choose an actual entity; a retrieval keyword by itself is not a target."""
    keyword_set = {str(value).strip().casefold() for value in keywords if str(value).strip()}
    explicit = str(explicit_target or "").strip()
    explicit = re.sub(r"\s*\((?:หัวข้อ/คีย์เวิร์ด|กลุ่มคู่แข่ง):.*\)\s*$", "", explicit).strip()
    if explicit and not is_placeholder_target(explicit) and explicit.casefold() not in keyword_set:
        return explicit
    matched_rival = str((resolved_project or {}).get("competitor_matched") or "").strip()
    if matched_rival:
        return matched_rival
    if tracking_kind == "competitor":
        return ""
    resolved_name = str((resolved_project or {}).get("project_name") or "").strip()
    rival_name = (resolved_name or str(project_name or "")).casefold()
    if (resolved_project or {}).get("is_rival") and ("rival" in rival_name or "คู่แข่ง" in rival_name):
        return ""
    if resolved_name and not is_placeholder_target(resolved_name):
        return resolved_name
    for candidate in (company_name, project_name):
        candidate = str(candidate or "").strip()
        if candidate and not is_placeholder_target(candidate):
            return candidate
    return ""


def _compact_context_lines(resolved_context, text_override=None):
    """Build stable context lines while omitting exact duplicate metadata."""
    if resolved_context.get("analysis_scope") == "keyword":
        text = text_override if text_override is not None else resolved_context.get("capped_text") or ""
        return [f"Target keywords={resolved_context.get('sentiment_target') or ''}", f"Text={text}"]
    target_value = (resolved_context.get("sentiment_target", resolved_context.get("actual_target"))
                    if HYBRID_PROMPT_V2 else resolved_context.get("actual_target"))
    target = str(target_value or "the Target Entity").strip()
    project = str(resolved_context.get("project_name") or "").strip()
    project_desc = str(resolved_context.get("project_desc") or "").strip()
    source = str(resolved_context.get("source_info") or "").strip()
    text = str(text_override if text_override is not None else resolved_context.get("capped_text") or "")
    seen = {target.casefold()}
    lines = [f"Target={target}"]
    if project and project.casefold() not in seen:
        lines.append(f"Project={project}")
        seen.add(project.casefold())
    if project_desc and project_desc.casefold() not in seen:
        lines.append(f"Context={project_desc}")
        seen.add(project_desc.casefold())
    keywords = []
    for value in resolved_context.get("keywords") or []:
        keyword = str(value).strip()
        folded = keyword.casefold()
        if keyword and folded not in seen:
            seen.add(folded)
            keywords.append(keyword)
    if keywords:
        lines.append(f"Keywords={', '.join(keywords)}")
    if source and (not HYBRID_PROMPT_V2 or
                   (not source.lower().startswith("source link=") and not re.search(r"https?://", source, re.I))):
        lines.append(f"Source={source}")
    lines.append(f"Text={text}")
    return lines


def build_jev_state_prompt(resolved_context):
    return "\n".join(_compact_context_lines(resolved_context))


def build_deepseek_user_prompt(resolved_context, jev_signal=None, include_jev_signal=None):
    if resolved_context.get("analysis_scope") == "keyword":
        max_chars = parse_bounded_int("DEEPSEEK_TEXT_MAX_CHARS", 3000, min_val=3000, max_val=8000)
        full_text = resolved_context.get("clean_text") or resolved_context.get("capped_text") or ""
        keywords = resolved_context.get("keywords") or []
        first_match = next((k for k in keywords if k.casefold() in full_text.casefold()), "")
        excerpt = cap_text(full_text, max_chars=max_chars, keyword=first_match) if len(full_text) > max_chars else full_text
        return "\n".join(_compact_context_lines(resolved_context, text_override=excerpt))
    if not HYBRID_PROMPT_V2:
        lines = _compact_context_lines(resolved_context)
        if isinstance(jev_signal, dict):
            lines.append(f"Jev={jev_signal.get('probabilities')}; confidence={jev_signal.get('confidence')}; "
                         f"conflicts={jev_signal.get('conflict_reasons') or []}")
        return "\n".join(lines)

    max_chars = parse_bounded_int("DEEPSEEK_TEXT_MAX_CHARS", 3000, min_val=3000, max_val=8000)
    full_text = resolved_context.get("clean_text") or resolved_context.get("capped_text") or ""
    if max_chars > 3000 and len(full_text) > 3000:
        keywords = resolved_context.get("keywords") or []
        target = resolved_context.get("sentiment_target", resolved_context.get("actual_target")) or ""
        text_override = cap_text(full_text, max_chars=max_chars,
                                 keyword=str(keywords[0]) if keywords else "", target=target)
    else:
        text_override = None
    lines = _compact_context_lines(resolved_context, text_override=text_override)
    if include_jev_signal is None:
        include_jev_signal = os.environ.get("DEEPSEEK_INCLUDE_JEV_SIGNAL", "true").lower() in ("true", "1", "yes")
    if include_jev_signal and isinstance(jev_signal, dict):
        probs = jev_signal.get("probabilities")
        conf = jev_signal.get("confidence")
        conflicts = jev_signal.get("conflict_reasons") or []
        if isinstance(probs, dict):
            compact_probs = ",".join(
                f"{short}:{float(probs[label]):.3f}"
                for label, short in (("POSITIVE", "POS"), ("NEUTRAL", "NEU"),
                                     ("NEGATIVE", "NEG"), ("AMBIGUOUS_OR_IRONY", "IRONY"))
                if isinstance(probs.get(label), (int, float)) and not isinstance(probs[label], bool)
            )
            lines.append(f"Jev={compact_probs}; confidence={conf}; conflicts={','.join(conflicts)}")
    return "\n".join(lines)


def validate_jev_response(data: Any) -> Optional[Dict[str, Any]]:
    """
    Strict validation for TypeSafe Jev OpenRouter Decisions API response.
    Returns normalized structure if valid, or None if invalid/corrupt.
    """
    if not isinstance(data, dict):
        return None

    answers = data.get("answers")
    if not isinstance(answers, dict):
        return None

    # 1. Sentiment question
    sentiment_ans = answers.get("sentiment")
    if not isinstance(sentiment_ans, dict):
        return None

    s_conf = sentiment_ans.get("confidence")
    if not isinstance(s_conf, (int, float)) or isinstance(s_conf, bool):
        return None
    try:
        s_conf = float(s_conf)
        if math.isnan(s_conf) or math.isinf(s_conf) or s_conf < 0.0 or s_conf > 1.0:
            return None
    except (ValueError, TypeError, OverflowError):
        return None

    s_probs = sentiment_ans.get("probabilities")
    if not isinstance(s_probs, dict):
        return None

    clean_s_probs = {str(k).strip().lower(): v for k, v in s_probs.items()}
    required_s_keys = ["positive", "neutral", "negative", "irony"]
    parsed_s_vals = {}
    for k in required_s_keys:
        if k not in clean_s_probs:
            return None
        v = clean_s_probs[k]
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None
        try:
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv) or fv < 0.0 or fv > 1.0:
                return None
            parsed_s_vals[k] = fv
        except (ValueError, TypeError, OverflowError):
            return None

    s_sum = sum(parsed_s_vals.values())
    if not (0.98 <= s_sum <= 1.02):
        return None
    normalized_s_probs = {k: v / s_sum for k, v in parsed_s_vals.items()}

    s_choice = sentiment_ans.get("choice")
    if s_choice is not None and not isinstance(s_choice, str):
        return None
    if s_choice:
        s_choice = s_choice.strip().lower()
        if s_choice not in required_s_keys:
            return None

    # 2. Entity relevance question
    entity_ans = answers.get("entity_relevance")
    if not isinstance(entity_ans, dict):
        return None

    e_conf = entity_ans.get("confidence")
    if not isinstance(e_conf, (int, float)) or isinstance(e_conf, bool):
        return None
    try:
        e_conf = float(e_conf)
        if math.isnan(e_conf) or math.isinf(e_conf) or e_conf < 0.0 or e_conf > 1.0:
            return None
    except (ValueError, TypeError, OverflowError):
        return None

    e_probs = entity_ans.get("probabilities")
    if not isinstance(e_probs, dict):
        return None

    clean_e_probs = {str(k).strip().lower(): v for k, v in e_probs.items()}
    required_e_keys = ["relevant", "unrelated", "uncertain"]
    parsed_e_vals = {}
    for k in required_e_keys:
        if k not in clean_e_probs:
            return None
        v = clean_e_probs[k]
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None
        try:
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv) or fv < 0.0 or fv > 1.0:
                return None
            parsed_e_vals[k] = fv
        except (ValueError, TypeError, OverflowError):
            return None

    e_sum = sum(parsed_e_vals.values())
    if not (0.98 <= e_sum <= 1.02):
        return None
    normalized_e_probs = {k: v / e_sum for k, v in parsed_e_vals.items()}

    e_choice = entity_ans.get("choice")
    if e_choice is not None and not isinstance(e_choice, str):
        return None
    if e_choice:
        e_choice = e_choice.strip().lower()
        if e_choice not in required_e_keys:
            return None

    return {
        "sentiment_probabilities": normalized_s_probs,
        "sentiment_confidence": s_conf,
        "sentiment_choice": s_choice,
        "entity_probabilities": normalized_e_probs,
        "entity_confidence": e_conf,
        "entity_choice": e_choice
    }


def validate_deepseek_response(parsed_data: Any) -> Optional[Dict[str, Any]]:
    """
    Strict validation for DeepSeek fallback response.
    Requires explicit entity_found boolean and all 4 finite probabilities.
    Supports valid fractions or percentage inputs (if sum ~ 1.0 or ~ 100).
    Rejects all-zero, missing-field, prose-only, and out-of-range responses.
    """
    if not isinstance(parsed_data, dict):
        return None

    raw_entity = parsed_data.get("entity_found")
    if not isinstance(raw_entity, bool):
        return None

    probs_dict = parsed_data.get("probabilities")
    if not isinstance(probs_dict, dict):
        return None
    required_keys = ("POSITIVE", "NEUTRAL", "NEGATIVE", "AMBIGUOUS_OR_IRONY")
    if any(key not in probs_dict for key in required_keys):
        return None
    values = []
    for key in required_keys:
        value = probs_dict[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        try:
            value = float(value)
        except (ValueError, TypeError, OverflowError):
            return None
        if not math.isfinite(value) or value < 0.0:
            return None
        values.append(value)

    total = sum(values)
    if 98.0 <= total <= 102.0:
        if any(value > 100.0 for value in values):
            return None
        values = [value / 100.0 for value in values]
        total /= 100.0

    if not (0.98 <= total <= 1.02):
        return None
    if any(value > 1.0 for value in values):
        return None

    return {
        "probabilities": {key: value / total for key, value in zip(required_keys, values)},
        "entity_found": raw_entity
    }


def route_sentiment(jev_data: Optional[Dict[str, Any]], actual_target: str) -> Dict[str, Any]:
    """
    Deterministic Confidence & Conflict Router:
    Computes:
      sentiment_peak = max(positive, neutral, negative, irony)
      sentiment_confidence = min(provider_sentiment_confidence, sentiment_peak)
      entity_confidence = provider entity confidence
      routing_confidence = min(sentiment_confidence, entity_confidence)

    Conflict detection returns named reasons.
    Valid Jev AND routing_confidence >= 0.65 AND no conflict -> accept Jev.
    Anything else -> DeepSeek.
    """
    if not jev_data or not isinstance(jev_data, dict):
        return {
            "accept_jev": False,
            "routing_confidence": 0.0,
            "conflict_reasons": ["jev_unavailable_or_invalid"],
            "entity_found": False,
            "probabilities": None
        }

    s_probs = jev_data["sentiment_probabilities"]
    pos = s_probs["positive"]
    neu = s_probs["neutral"]
    neg = s_probs["negative"]
    irony = s_probs["irony"]

    sentiment_peak = max(pos, neu, neg, irony)
    provider_sentiment_conf = jev_data["sentiment_confidence"]
    sentiment_confidence = min(provider_sentiment_conf, sentiment_peak)
    entity_confidence = jev_data["entity_confidence"]
    routing_confidence = min(sentiment_confidence, entity_confidence)

    # Sentiment choice and argmax
    s_choices = [("positive", pos), ("neutral", neu), ("negative", neg), ("irony", irony)]
    sentiment_argmax = max(s_choices, key=lambda x: x[1])[0]
    provider_s_choice = (jev_data.get("sentiment_choice") or "").strip().lower()

    # Entity choice and argmax
    e_probs = jev_data["entity_probabilities"]
    e_choices = [("relevant", e_probs["relevant"]), ("unrelated", e_probs["unrelated"]), ("uncertain", e_probs["uncertain"])]
    entity_argmax = max(e_choices, key=lambda x: x[1])[0]
    provider_e_choice = (jev_data.get("entity_choice") or "").strip().lower()

    entity_decision = provider_e_choice if provider_e_choice in ("relevant", "unrelated", "uncertain") else entity_argmax

    conflict_reasons = []

    # 1. Entity decision is uncertain
    if entity_decision == "uncertain":
        conflict_reasons.append("entity_uncertain")

    # 2. Entity is unrelated while positive + negative + irony >= 0.35
    if entity_decision == "unrelated" and ((pos + neg + irony) >= 0.35):
        conflict_reasons.append("unrelated_with_sentiment_mass")

    # 3. Both positive and negative are at least 0.25
    if pos >= 0.25 and neg >= 0.25:
        conflict_reasons.append("bipolar_positive_negative")

    # 4. Difference between largest and second-largest sentiment choice is less than 0.15
    sorted_probs = sorted([pos, neu, neg, irony], reverse=True)
    if (sorted_probs[0] - sorted_probs[1]) < 0.15:
        conflict_reasons.append("narrow_sentiment_margin")

    # 5. Provider selected choice disagrees with probability argmax
    if provider_s_choice and provider_s_choice != sentiment_argmax:
        conflict_reasons.append("provider_choice_disagrees_with_argmax")

    # 6. Target resolution ended with only generic placeholder
    if is_placeholder_target(actual_target):
        conflict_reasons.append("generic_placeholder_target")

    accept_jev = (routing_confidence >= JEV_ACCEPTANCE_THRESHOLD) and (len(conflict_reasons) == 0)

    # When a confidently unrelated Jev result passes the gate, entity_found=False
    if accept_jev:
        entity_found = (entity_decision != "unrelated")
    else:
        entity_found = (entity_decision == "relevant")

    return {
        "accept_jev": accept_jev,
        "routing_confidence": routing_confidence,
        "conflict_reasons": conflict_reasons,
        "entity_found": entity_found,
        "probabilities": {
            "POSITIVE": pos,
            "NEUTRAL": neu,
            "NEGATIVE": neg,
            "AMBIGUOUS_OR_IRONY": irony
        }
    }


# =============================================================================
# Ollama & Gemini Sentiment Analyzer Engine (2-Pass Pipeline)
# =============================================================================
class OllamaSentimentAnalyzer:
    CONCURRENT_WORKERS = CONCURRENT_WORKERS
    MAX_IN_FLIGHT = MAX_IN_FLIGHT
    DEEPSEEK_MAX_CONCURRENCY = DEEPSEEK_MAX_CONCURRENCY

    def __init__(self, model="qcwind/qwen3-8b-instruct-Q4-K-M:latest"):
        self.model = model
        self.host_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self.base_url = f"{self.host_url}/api/generate"
        self.chat_url = f"{self.host_url}/api/chat"
        self.triage_timeout = int(os.environ.get("TRIAGE_TIMEOUT_SEC", 60))
        self.CONCURRENT_WORKERS = CONCURRENT_WORKERS
        self.MAX_IN_FLIGHT = MAX_IN_FLIGHT
        self.DEEPSEEK_MAX_CONCURRENCY = DEEPSEEK_MAX_CONCURRENCY
        self.deepseek_semaphore = threading.BoundedSemaphore(self.DEEPSEEK_MAX_CONCURRENCY)
        self._thread_local = threading.local()

    def get_session(self) -> requests.Session:
        if not hasattr(self._thread_local, "session") or self._thread_local.session is None:
            sess = requests.Session()
            adapter = requests.adapters.HTTPAdapter(max_retries=0)
            sess.mount("http://", adapter)
            sess.mount("https://", adapter)
            self._thread_local.session = sess
        return self._thread_local.session

    @property
    def session(self):
        return self.get_session()

    @session.setter
    def session(self, s):
        self._thread_local.session = s

    def _record_provider_attempt(self, route, model, retry=False, response_data=None, provider_hint=""):
        metrics = getattr(self._thread_local, "usage_metrics", None)
        if metrics is None:
            return
        provider = provider_hint
        if isinstance(response_data, dict):
            provider = response_data.get("provider") or response_data.get("provider_name") or provider
            model = response_data.get("model") or model
        metrics.record(route, model, provider, retry, response_data)

    def _run_with_usage_metrics(self, metrics, func, *args):
        previous = getattr(self._thread_local, "usage_metrics", None)
        self._thread_local.usage_metrics = metrics
        try:
            return func(*args)
        finally:
            self._thread_local.usage_metrics = previous

    def _call_gemini_api(self, model_name, system_instruction, user_prompt, max_retries=3, max_tokens=None):
        api_key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return None
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        gen_config = {"temperature": 0.0}
        if max_tokens:
            gen_config["maxOutputTokens"] = max_tokens

        is_gemma = "gemma" in model_name.lower()
        if is_gemma:
            # Gemma models on Google AI Studio API do not support native system_instruction
            combined_prompt = f"{system_instruction}\n\n{user_prompt}"
            payload = {
                "contents": [{"parts": [{"text": combined_prompt}]}],
                "generationConfig": gen_config
            }
        else:
            gen_config["responseMimeType"] = "application/json"
            payload = {
                "system_instruction": {"parts": [{"text": system_instruction}]},
                "contents": [{"parts": [{"text": user_prompt}]}],
                "generationConfig": gen_config
            }
        api_timeout = int(os.environ.get("GEMINI_API_TIMEOUT", 30))
        
        for attempt in range(max_retries):
            try:
                response = self.session.post(url, json=payload, timeout=api_timeout)
                # Fallback for models that reject system_instruction or responseMimeType with 400 Bad Request
                if response.status_code == 400 and not is_gemma:
                    fallback_config = {"temperature": 0.0}
                    if max_tokens:
                        fallback_config["maxOutputTokens"] = max_tokens
                    fallback_payload = {
                        "contents": [{"parts": [{"text": f"{system_instruction}\n\n{user_prompt}"}]}],
                        "generationConfig": fallback_config
                    }
                    response = self.session.post(url, json=fallback_payload, timeout=api_timeout)

                if response.status_code == 200:
                    res_data = response.json()
                    candidates = res_data.get("candidates") or []
                    if not candidates:
                        print(f"  -> Gemini API Empty Candidates [{model_name}]: {res_data}")
                        continue
                    candidate = candidates[0] if isinstance(candidates[0], dict) else {}
                    content = candidate.get("content") or {}
                    parts = content.get("parts") or [] if isinstance(content, dict) else []
                    
                    # Extract text: filter out thought parts (Gemma 4/Gemini thinking) and join in order
                    non_thought_parts = [
                        p["text"] for p in parts
                        if isinstance(p, dict) and not p.get("thought") and "text" in p
                    ]
                    if non_thought_parts:
                        result_text = "".join(non_thought_parts)
                    elif parts:
                        result_text = "".join(
                            p.get("text", "") if isinstance(p, dict) else (p if isinstance(p, str) else "")
                            for p in parts
                        )
                    else:
                        result_text = ""

                    parsed_res = self._parse_json_result(result_text)
                    if parsed_res is None:
                        print(f"  -> Gemini API Parsing Error [{model_name}] (attempt {attempt + 1}/{max_retries}): {result_text[:200]}")
                        if attempt < max_retries - 1:
                            time.sleep(2)
                        continue
                    return parsed_res
                elif response.status_code == 404:
                    print(f"  -> Gemini API Model Not Found [{model_name}]: 404 (ข้ามโมเดลนี้ทันที)")
                    break
                elif response.status_code == 429:
                    retry_after = 3
                    try:
                        ra_hdr = response.headers.get("Retry-After")
                        if ra_hdr:
                            retry_after = max(int(float(ra_hdr)), 2)
                    except Exception:
                        pass
                    print(f"  -> Gemini API Rate Limited (429) [{model_name}] (attempt {attempt + 1}/{max_retries}): รอ {retry_after}s...")
                    if attempt < max_retries - 1:
                        time.sleep(retry_after)
                        continue
                else:
                    print(f"  -> Gemini API Error [{model_name}]: {response.status_code} - {response.text[:200]}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
            except Exception as e:
                print(f"  -> Gemini API Exception [{model_name}]: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        return None

    def _call_openrouter_api(self, model_name, system_instruction, user_prompt, max_retries=2, max_tokens=None):
        api_key = OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            return None

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://github.com/so-witsarut/ai-sentiment"),
            "X-Title": os.environ.get("OPENROUTER_TITLE", "AI Sentiment Pipeline"),
            "Content-Type": "application/json"
        }

        raw_providers = os.environ.get("OPENROUTER_PROVIDERS", "OpenInference,Relace")
        providers = [p.strip() for p in raw_providers.split(",") if p.strip()]
        allow_fallbacks = os.environ.get("OPENROUTER_ALLOW_FALLBACKS", "true").lower() in ("true", "1", "yes")

        provider_cfg = {"allow_fallbacks": allow_fallbacks}
        if "deepseek" in model_name.lower() and providers:
            provider_cfg["order"] = providers

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "provider": provider_cfg
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        api_timeout = int(os.environ.get("OPENROUTER_API_TIMEOUT", 45))

        for attempt in range(max_retries):
            try:
                response = self.session.post(url, headers=headers, json=payload, timeout=api_timeout)
                if response.status_code == 200:
                    res_data = response.json()
                    choices = res_data.get("choices") or []
                    if not choices:
                        print(f"  -> OpenRouter API Empty Choices [{model_name}]: {res_data}")
                        continue
                    message = choices[0].get("message") or {}
                    result_text = message.get("content") or ""

                    # Log cache hit information
                    usage = res_data.get("usage") or {}
                    prompt_details = usage.get("prompt_tokens_details") or {}
                    cached_tokens = prompt_details.get("cached_tokens", 0) or usage.get("cached_tokens", 0)
                    if cached_tokens:
                        total_prompt = usage.get("prompt_tokens", 0)
                        print(f"  ⚡ [OpenRouter Cache Hit] {cached_tokens}/{total_prompt} tokens cached ({model_name})")

                    parsed_res = self._parse_json_result(result_text)
                    if parsed_res is None:
                        print(f"  -> OpenRouter API Parsing Error [{model_name}] (attempt {attempt + 1}/{max_retries}): {result_text[:200]}")
                        if attempt < max_retries - 1:
                            time.sleep(2)
                        continue
                    return parsed_res
                elif response.status_code in (401, 403):
                    print(f"  -> OpenRouter Auth Error [{model_name}]: {response.status_code} - {response.text[:200]}")
                    break
                elif response.status_code == 404:
                    print(f"  -> OpenRouter Model Not Found [{model_name}]: 404")
                    break
                elif response.status_code == 429:
                    retry_after = 3
                    try:
                        ra_hdr = response.headers.get("Retry-After")
                        if ra_hdr:
                            retry_after = max(int(float(ra_hdr)), 2)
                    except Exception:
                        pass
                    print(f"  -> OpenRouter Rate Limited (429) [{model_name}] (attempt {attempt + 1}/{max_retries}): รอ {retry_after}s...")
                    if attempt < max_retries - 1:
                        time.sleep(retry_after)
                        continue
                else:
                    print(f"  -> OpenRouter API Error [{model_name}]: {response.status_code} - {response.text[:200]}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
            except Exception as e:
                print(f"  -> OpenRouter API Exception [{model_name}]: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        return None

    def _call_ollama_generic(self, model_name, system_instruction, user_prompt):
        deep_timeout = int(os.environ.get("OLLAMA_DEEP_TIMEOUT", 30))
        payload_generate = {
            "model": model_name,
            "system": system_instruction,
            "prompt": user_prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "seed": 42}
        }
        try:
            response = self.session.post(self.base_url, json=payload_generate, timeout=deep_timeout)
            if response.status_code == 200:
                result_text = response.json().get("response", "{}")
                parsed = self._parse_json_result(result_text)
                if parsed:
                    return parsed
            else:
                print(f"  -> Ollama Deep Model [{model_name}] error {response.status_code}: {response.text[:150]}")
        except Exception as e:
            print(f"  -> Ollama Deep Model [{model_name}] exception: {e}")
            
        payload_chat = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user",   "content": user_prompt}
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "seed": 42}
        }
        try:
            response = self.session.post(self.chat_url, json=payload_chat, timeout=deep_timeout)
            if response.status_code == 200:
                result_text = response.json().get("message", {}).get("content", "{}")
                return self._parse_json_result(result_text)
            else:
                print(f"  -> Ollama Chat Model [{model_name}] error {response.status_code}: {response.text[:150]}")
        except Exception as e:
            print(f"  -> Ollama Chat Model [{model_name}] exception: {e}")
        return None

    def _normalize_distribution(self, positive=0, negative=0, neutral=100):
        """Normalize percentage distribution so the total is exactly 100."""
        def num(v, default=0):
            try:
                if isinstance(v, str):
                    v = v.replace("%", "").strip()
                return max(0.0, min(100.0, float(v)))
            except Exception:
                return default
        positive, negative, neutral = num(positive), num(negative), num(neutral)
        total = positive + negative + neutral
        if total <= 0:
            return {"positive_percent": 0, "negative_percent": 0, "neutral_percent": 100}
        positive = int(round(positive * 100 / total))
        negative = int(round(negative * 100 / total))
        neutral = max(0, 100 - positive - negative)
        return {"positive_percent": positive, "negative_percent": negative, "neutral_percent": neutral}

    def _distribution_to_sentiment(self, positive=0, negative=0, neutral=100, positive_percent=None, negative_percent=None, neutral_percent=None, **kwargs):
        def _to_num(val, default):
            if val is None:
                return default
            try:
                if isinstance(val, str):
                    val = val.replace("%", "").strip()
                return float(val)
            except (ValueError, TypeError):
                return default

        pos = _to_num(positive_percent if positive_percent is not None else positive, 0)
        neg = _to_num(negative_percent if negative_percent is not None else negative, 0)
        neu = _to_num(neutral_percent if neutral_percent is not None else neutral, 100)

        if pos > neg and pos > neu:
            return 100
        if neg > pos and neg > neu:
            return -100
        # Tie between pos and neg, both above neutral → default to negative (conservative for brand monitoring)
        if pos == neg and pos > neu:
            return -100
        return 0

    def _parse_json_result(self, result_text):
        if not result_text:
            return None
        # Clean thought/think tags (both closed and unclosed)
        clean_text = re.sub(r'<think>.*?</think>', '', result_text, flags=re.DOTALL)
        clean_text = re.sub(r'<thought>.*?</thought>', '', clean_text, flags=re.DOTALL)
        clean_text = re.sub(r'<(?:think|thought)>.*?(?=(?:```|\[|\{|$))', '', clean_text, flags=re.DOTALL).strip()
        clean_text = re.sub(r'```(?:json)?\s*', '', clean_text)
        clean_text = clean_text.replace('```', '').strip()

        def _clean_reason(s):
            if not isinstance(s, str):
                return ""
            r = re.sub(r'\(?\s*(?:ตาม)?กฎข้อ\s*[\d\s,และ|-]+\)?', '', s, flags=re.I)
            r = re.sub(r'\(?\s*Rule\s*[\d\s,and|-]+\)?', '', r, flags=re.I).strip(' "\' \t\r\n')
            return r

        parsed = None
        # 1. Try direct parse first
        try:
            raw_parsed = json.loads(clean_text)
            if isinstance(raw_parsed, list) and raw_parsed:
                parsed = raw_parsed[0] if isinstance(raw_parsed[0], dict) else None
            elif isinstance(raw_parsed, dict):
                parsed = raw_parsed
        except json.JSONDecodeError:
            pass

        # 2. Try regex extraction of JSON object {...} or array of objects [{...}]
        if parsed is None:
            for pattern in (r'\{.*\}', r'\[\s*\{.*\}\s*\]'):
                m = re.search(pattern, clean_text, re.DOTALL)
                if m:
                    sub_text = re.sub(r',\s*([\}\]])', r'\1', m.group(0))
                    try:
                        raw_parsed = json.loads(sub_text)
                        if isinstance(raw_parsed, list) and raw_parsed:
                            if isinstance(raw_parsed[0], dict):
                                parsed = raw_parsed[0]
                                break
                        elif isinstance(raw_parsed, dict):
                            parsed = raw_parsed
                            break
                    except json.JSONDecodeError:
                        pass

        if parsed is not None:
            upper_keys = {str(k).upper() for k in parsed.keys()}
            if "PROBABILITIES" in upper_keys or ("POSITIVE" in upper_keys and "NEGATIVE" in upper_keys) or "AMBIGUOUS_OR_IRONY" in upper_keys:
                return parsed

        if parsed is None:
            targeted_m = re.search(r'\{[^{}]*(?:probabilities|POSITIVE|positive)[^{}]*\}', clean_text, re.DOTALL | re.IGNORECASE)
            if targeted_m:
                try:
                    target_parsed = json.loads(targeted_m.group(0))
                    if isinstance(target_parsed, dict):
                        upper_keys = {str(k).upper() for k in target_parsed.keys()}
                        if "PROBABILITIES" in upper_keys or ("POSITIVE" in upper_keys and "NEGATIVE" in upper_keys) or "AMBIGUOUS_OR_IRONY" in upper_keys:
                            if "entity_found" not in target_parsed:
                                entity_m = re.search(r'[`"\']?entity_found[`"\']?\s*[:=]\s*(true|false)', clean_text, re.I)
                                if entity_m:
                                    target_parsed["entity_found"] = entity_m.group(1).lower() == "true"
                            return target_parsed
                except Exception:
                    pass

        if parsed is None:
            # Recover percentage fields or probabilities even when model JSON is malformed.
            def find_num(names):
                for name in names:
                    m = re.search(r'[`"\']?' + re.escape(name) + r'[`"\']?\s*[:=]\s*(-?\d+(?:\.\d+)?)', clean_text, re.I)
                    if m:
                        return m.group(1)
                return None

            prob_irony = find_num(["AMBIGUOUS_OR_IRONY", "ambiguous_or_irony", "irony", "sarcasm"])
            prob_pos = find_num(["POSITIVE", "positive", "pos"])
            prob_neg = find_num(["NEGATIVE", "negative", "neg"])
            prob_neu = find_num(["NEUTRAL", "neutral", "neu"])

            is_probabilistic = (
                ENABLE_PROBABILISTIC_MODE
                or prob_irony is not None
                or "probabilities" in clean_text.lower()
            )

            if is_probabilistic and (prob_pos is not None or prob_neg is not None or prob_neu is not None or prob_irony is not None):
                entity_m = re.search(r'[`"\']?entity_found[`"\']?\s*[:=]\s*(true|false)', clean_text, re.I)
                entity_found = entity_m.group(1).lower() == "true" if entity_m else True
                return {
                    "probabilities": {
                        "POSITIVE": float(prob_pos or 0),
                        "NEUTRAL": float(prob_neu or 0),
                        "NEGATIVE": float(prob_neg or 0),
                        "AMBIGUOUS_OR_IRONY": float(prob_irony or 0),
                    },
                    "entity_found": entity_found
                }

            pos = find_num(["positive_percent", "positive"])
            neg = find_num(["negative_percent", "negative"])
            neu = find_num(["neutral_percent", "neutral"])
            sent = find_num(["ai_sentiment"])
            if pos is None and neg is None and neu is None and sent is None:
                return None
            if pos is None and neg is None and neu is None:
                sent_val = float(sent) if sent is not None else 0
                if sent_val > 0:
                    dist = {"positive_percent": 80, "negative_percent": 0, "neutral_percent": 20}
                elif sent_val < 0:
                    dist = {"positive_percent": 0, "negative_percent": 80, "neutral_percent": 20}
                else:
                    dist = {"positive_percent": 5, "negative_percent": 5, "neutral_percent": 90}
            else:
                dist = self._normalize_distribution(pos or 0, neg or 0, neu or 0)
            reason_m = re.search(r'[`"\']?reason[`"\']?\s*[:=]\s*[`"\']?(.*?)[`"\']?(?:,|\n|\}|$)', clean_text, re.I)
            entity_m = re.search(r'[`"\']?entity_found[`"\']?\s*[:=]\s*(true|false)', clean_text, re.I)
            reason = _clean_reason(reason_m.group(1)) if reason_m else _clean_reason(clean_text[:100].replace("\n", " "))
            entity_found = entity_m.group(1).lower() == "true" if entity_m else True
            return {"ai_sentiment": self._distribution_to_sentiment(**dist),
                    "reason": reason, "entity_found": entity_found, **dist}

        pos = parsed.get("positive_percent", parsed.get("positive"))
        neg = parsed.get("negative_percent", parsed.get("negative"))
        neu = parsed.get("neutral_percent", parsed.get("neutral"))
        if pos is None and neg is None and neu is None:
            legacy = parsed.get("ai_sentiment", 0)
            try: legacy = float(legacy)
            except Exception: legacy = 0
            if legacy > 0: dist = {"positive_percent": 80, "negative_percent": 0, "neutral_percent": 20}
            elif legacy < 0: dist = {"positive_percent": 0, "negative_percent": 80, "neutral_percent": 20}
            else: dist = {"positive_percent": 5, "negative_percent": 5, "neutral_percent": 90}
        else:
            dist = self._normalize_distribution(pos or 0, neg or 0, neu or 0)

        reason = _clean_reason(parsed.get("reason", ""))
        entity_found = parsed.get("entity_found", True)
        if isinstance(entity_found, str):
            entity_found = entity_found.lower() in ("true", "1")
        parsed.update(dist)
        parsed["ai_sentiment"] = self._distribution_to_sentiment(**dist)
        parsed["reason"] = reason
        parsed["entity_found"] = bool(entity_found)
        return parsed

    # -----------------------------------------------------------------
    # PASS 1: Fast Triage (Qwen 8B Local)
    # -----------------------------------------------------------------
    def _parse_triage_result(self, result_text):
        """Parse triage result from Qwen. Returns True (has sentiment) or False (no sentiment)."""
        if not result_text:
            return True

        clean_text = re.sub(r'<think>.*?</think>', '', result_text, flags=re.DOTALL)
        clean_text = re.sub(r'<thought>.*?</thought>', '', clean_text, flags=re.DOTALL)
        clean_text = re.sub(r'<(?:think|thought)>.*?(?=(?:```|\[|\{|$))', '', clean_text, flags=re.DOTALL).strip()
        clean_text = re.sub(r'```(?:json)?\s*', '', clean_text).replace('```', '').strip()

        json_match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        if json_match:
            clean_text = json_match.group(0)

        try:
            parsed = json.loads(clean_text)
            if isinstance(parsed, dict):
                triage_val = str(parsed.get("triage", "yes")).strip().lower()
                return triage_val in ("yes", "true", "1")
        except json.JSONDecodeError:
            pass

        triage_match = re.search(r'[`"\']?triage[`"\']?\s*[:=]\s*[`"\']?(yes|no|true|false)[`"\']?', clean_text, re.IGNORECASE)
        if triage_match:
            val = triage_match.group(1).lower()
            return val in ("yes", "true")
        # Fail-safe: if unparseable, return True (send to Pass 2)
        return True

    def _triage_post(self, post_id, content, actual_target=""):
        """PASS 1: conservative relevance + sentiment triage."""
        if not content or not str(content).strip():
            return False

        triage_system = (
            "You are a fast, conservative triage classifier for Thai social-media sentiment monitoring.\n"
            "Decide ONLY whether this post should be sent to a deeper sentiment analysis model.\n"
            "Answer YES if the text may contain an opinion, evaluation, emotion, experience, praise, criticism, complaint, "
            "satisfaction, dissatisfaction, sarcasm, comparison, recommendation, or subjective reaction that could be relevant "
            "to the Target Entity or its product/service/context.\n"
            "Answer NO only when the text is clearly factual, informational, promotional, administrative, a plain announcement, "
            "or otherwise contains no meaningful subjective opinion requiring deep analysis.\n"
            "Do NOT decide positive/negative/neutral. Do NOT require explicit Target mention here; Pass 2 verifies relevance.\n"
            "If uncertain, ambiguous, mixed, sarcastic, or unsure about relevance, choose YES.\n"
            'Return ONLY JSON: {"triage":"yes"} or {"triage":"no"}'
        )
        triage_prompt = f"Target Entity={actual_target or 'Unknown'}\nText={content}"

        payload = {
            "model": self.model, "system": triage_system, "prompt": triage_prompt,
            "stream": False, "format": "json", "think": False, "keep_alive": -1,
            "options": {"temperature": 0.0, "top_p": 0.1, "seed": 42, "num_predict": 16,
                        "num_ctx": 512, "num_batch": 256, "flash_attn": True}
        }
        try:
            response = self.session.post(self.base_url, json=payload, timeout=self.triage_timeout)
            if response.status_code == 200:
                return self._parse_triage_result(response.json().get("response", "{}"))
            print(f"  -> Triage HTTP Error [{post_id}]: {response.status_code}")
        except Exception as e:
            print(f"  -> Triage Error [{post_id}]: {e}")
        return True

    # -----------------------------------------------------------------
    # PASS 2: Deep Analysis (Gemma API)
    # -----------------------------------------------------------------
    def _deep_analyze_post(self, post_id, actual_target, source_info, expanded_content):
        """PASS 2: Deep Target-specific sentiment distribution."""
        # Static instruction for Prefix Caching (100% identical across all requests)
        deep_system = (
            "You are an expert Thai Social Media Brand Reputation Analyst.\n"
            "Analyze sentiment specifically TOWARD the specified Target Entity, not the overall mood of the post. "
            "A keyword match alone is not enough.\n\n"
            "TASK:\nReturn a NUANCED sentiment distribution toward the Target Entity only. "
            "The three percentages represent HOW MUCH of the text's sentiment leans toward each category. "
            "Real-world posts rarely have 100% pure sentiment — almost always there is some residual neutrality or mixed feeling.\n\n"
            "DISTRIBUTION GUIDELINES:\n"
            "- Strong positive with no caveats: 75-85 positive, 0-5 negative, 15-25 neutral\n"
            "- Mild/moderate positive: 40-65 positive, 0-10 negative, 30-55 neutral\n"
            "- Strong negative with no caveats: 0-5 positive, 75-85 negative, 15-25 neutral\n"
            "- Mild/moderate negative: 0-10 positive, 40-65 negative, 30-55 neutral\n"
            "- Mixed positive and negative: allocate both, e.g. 40 positive, 35 negative, 25 neutral\n"
            "- Mostly factual/news but slightly positive tone: 15-25 positive, 0-5 negative, 70-85 neutral\n"
            "- Mostly factual/news but slightly negative tone: 0-5 positive, 15-25 negative, 70-85 neutral\n"
            "- Pure factual/unrelated: 0-5 positive, 0-5 negative, 90-100 neutral\n"
            "- AVOID using exactly 100/0/0 or 0/0/100 unless the text is absolutely extreme or completely unrelated.\n\n"
            "DECISION RULES:\n"
            "1. ENTITY CHECK: If the Target is absent, coincidental/unrelated, or the opinion is clearly about another entity, "
            "set entity_found=false and use 0/0/100.\n"
            "2. OWNED/PR: Official Target content or pure PR/advertising → lean heavily neutral (e.g. 10/0/90) unless user opinion is embedded.\n"
            "3. POSITIVE: praise, recommendation, satisfaction, support, good experience, or favorable evaluation toward Target.\n"
            "4. NEGATIVE: criticism, complaint, anger, disappointment, bad experience, or unfavorable evaluation toward Target.\n"
            "5. NEUTRAL: factual news, announcements, questions without evaluation, promotions, sports/results, ambiguity, or sentiment aimed elsewhere.\n"
            "6. MIXED: If both positive and negative evaluation toward Target exist, allocate both shares proportionally. Do not force one polarity.\n"
            "7. PERCENTAGES: positive_percent + negative_percent + neutral_percent MUST equal exactly 100. "
            "Use multiples of 5: 0,5,10,15,...,100.\n"
            "8. Never assign Target sentiment from an emotion that is directed at another entity.\n\n"
            "For reason, explain concisely in natural Thai and mention the Target-related context. No rule numbers.\n"
            'Return ONLY valid JSON with exactly these keys:\n'
            'Examples:\n'
            '{"entity_found":true,"reason":"ผู้ใช้ชื่นชมบริการ แต่บ่นเรื่องราคาเล็กน้อย","positive_percent":60,"negative_percent":15,"neutral_percent":25}\n'
            '{"entity_found":true,"reason":"เป็นข่าวรายงานข้อเท็จจริง มีโทนเชิงบวกเล็กน้อย","positive_percent":15,"negative_percent":0,"neutral_percent":85}\n'
            '{"entity_found":true,"reason":"ผู้ใช้แสดงความไม่พอใจอย่างมาก","positive_percent":0,"negative_percent":80,"neutral_percent":20}'
        )
        # Dynamic content placed in user prompt at the end to maximize prefix caching
        deep_prompt = (
            f"Target Entity={actual_target}\n"
            f"Source Info={source_info}\n"
            f"Text={expanded_content}"
        )
        env_models = os.environ.get("VALIDATION_MODELS")
        if env_models:
            validation_models = [m.strip() for m in env_models.split(",") if m.strip()]
        else:
            validation_models = [
                "openrouter:google/gemma-4-26b-a4b-it:free",
                "openrouter:deepseek/deepseek-v4-flash-0731",
                "gemma4:31b-cloud",
                "api:gemma-4-26b-a4b-it",
                "api:gemma-4-31b-it",
                "api:gemini-3.1-flash-lite",
                "api:gemini-2.5-flash",
                "api:gemini-3.5-flash-lite",
            ]
        openrouter_retries = int(os.environ.get("OPENROUTER_MAX_RETRIES", 2))
        gemini_retries = int(os.environ.get("GEMINI_MAX_RETRIES", 1))
        for val_model in validation_models:
            if val_model.startswith("openrouter:"):
                actual_model = val_model.replace("openrouter:", "", 1)
                res = self._call_openrouter_api(actual_model, deep_system, deep_prompt, max_retries=openrouter_retries)
            elif val_model.startswith("api:"):
                actual_model = val_model.replace("api:", "", 1)
                res = self._call_gemini_api(actual_model, deep_system, deep_prompt, max_retries=gemini_retries)
            else:
                actual_model = val_model
                res = self._call_ollama_generic(val_model, deep_system, deep_prompt)

            if res and "ai_sentiment" in res:
                entity_found = res.get("entity_found", True)
                if isinstance(entity_found, str):
                    entity_found = entity_found.lower() in ("true", "1")
                if not entity_found:
                    res.update({"positive_percent":0,"negative_percent":0,"neutral_percent":100,"ai_sentiment":0})
                res["post_id"] = post_id
                res["model"] = actual_model
                return res
        return None

    def _probabilistic_analyze_post(self, post_id, actual_target, source_info, expanded_content, project_name="", project_desc=""):
        """Single-Pass Direct Probabilistic Inference (Zero String Tax + System 1 Engine)."""
        user_prompt_lines = [f"Target Entity={actual_target}"]
        if project_name:
            user_prompt_lines.append(f"Project={project_name}")
        if project_desc:
            user_prompt_lines.append(f"Project Context={project_desc}")
        user_prompt_lines.append(f"Source Info={source_info}")
        user_prompt_lines.append(f"Text={expanded_content}")
        user_prompt = "\n".join(user_prompt_lines)

        env_models = os.environ.get("PROBABILISTIC_MODELS") or os.environ.get("VALIDATION_MODELS")
        if env_models:
            validation_models = [m.strip() for m in env_models.split(",") if m.strip()]
        else:
            validation_models = [
                "openrouter:deepseek/deepseek-v4-flash-0731",
                "api:gemini-2.5-flash",
            ]

        openrouter_retries = int(os.environ.get("OPENROUTER_MAX_RETRIES", 2))
        gemini_retries = int(os.environ.get("GEMINI_MAX_RETRIES", 2))

        for val_model in validation_models:
            if val_model.startswith("openrouter:"):
                actual_model = val_model.replace("openrouter:", "", 1)
                res = self._call_openrouter_api(actual_model, PROBABILISTIC_SYSTEM_PROMPT, user_prompt, max_retries=openrouter_retries, max_tokens=500)
            elif val_model.startswith("api:"):
                actual_model = val_model.replace("api:", "", 1)
                res = self._call_gemini_api(actual_model, PROBABILISTIC_SYSTEM_PROMPT, user_prompt, max_retries=gemini_retries, max_tokens=500)
            else:
                actual_model = val_model
                res = self._call_ollama_generic(val_model, PROBABILISTIC_SYSTEM_PROMPT, user_prompt)

            if res:
                norm = parse_probabilistic_response(res)
                if norm:
                    probs = norm["probabilities"]
                    entity_found = norm["entity_found"]
                    policy = resolve_policy(probs, entity_found)
                    synthetic_reason = generate_synthetic_reason(
                        actual_target,
                        probs["POSITIVE"],
                        probs["NEGATIVE"],
                        probs["NEUTRAL"],
                        probs["AMBIGUOUS_OR_IRONY"],
                        entity_found
                    )
                    return {
                        "post_id": post_id,
                        "ai_sentiment": policy["score"],
                        "sentiment": policy["sentiment"],
                        "positive_percent": policy["pos"],
                        "negative_percent": policy["neg"],
                        "neutral_percent": policy["neu"],
                        "irony_score": policy.get("irony_score", 0),
                        "reason": synthetic_reason,
                        "entity_found": entity_found,
                        "model": actual_model,
                        "project_name": project_name,
                        "raw_probabilities": probs
                    }
        return None

    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Accept only a complete JSON object from the provider answer channel."""
        if not isinstance(text, str):
            return None
        try:
            parsed = json.loads(text.strip())
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def _call_typesafe_jev(self, state_prompt: str, actual_target: str, keyword_scope: bool = False) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Invoke TypeSafe Jev via OpenRouter Decisions API with two typed questions:
        1. sentiment (choice: positive, neutral, negative, irony)
        2. entity_relevance (choice: relevant, unrelated, uncertain)
        """
        api_key = OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            return None, ""

        url = os.environ.get("OPENROUTER_DECISIONS_URL", "https://openrouter.ai/api/alpha/decisions")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://blueeye.co.th"),
            "X-Title": os.environ.get("OPENROUTER_TITLE", "BlueEye Sentiment Analysis"),
            "Content-Type": "application/json"
        }

        payload = {
            "model": JEV_MODEL,
            "state": state_prompt,
            "questions": {
                "sentiment": {
                    "type": "choice",
                    "instructions": ("In one pass, check whether the text expresses an opinion or emotion, then identify what it evaluates. Classify sentiment toward Target keywords only. If no opinion evaluates Target, choose neutral even when the text is emotional about someone else or Target is only a venue. Factual news, job ads, PR and seller slogans are neutral; a reported quote explicitly praising or criticizing Target still counts."
                                     if keyword_scope else
                                     "Classify sentiment toward Target only; unrelated topic sentiment is neutral."
                                     if HYBRID_PROMPT_V2 else f"Classify sentiment toward '{actual_target}'."),
                    "criteria": {
                        "positive": ("Independent praise, satisfaction, or support explicitly directed at a Target keyword, including an attributed quote; favorable news or advertiser self-praise alone does not qualify."
                                     if keyword_scope else "Praise, satisfaction, recommendation, endorsement, or directly beneficial news."),
                        "neutral": ("No subjective evaluation of Target: factual news, job ads, PR, seller slogans, or emotion about another person, film, event, or operator at a named venue."
                                    if keyword_scope else "Facts, news, PR, or a general inquiry without sentiment."),
                        "negative": ("Complaint, blame, or criticism explicitly directed at a Target keyword, including an attributed quote; a venue named only as a location does not inherit criticism of others."
                                     if keyword_scope else "Criticism, complaint, defect, boycott, damage, or frustration."),
                        "irony": "Sarcasm, mockery, cynical humor, satire, or backhanded praise."
                    }
                },
                "entity_relevance": {
                    "type": "choice",
                    "instructions": ("Does the text refer to any Target keyword in its intended sense? Mere mention can be relevant but sentiment-neutral; homonyms are unrelated."
                                     if keyword_scope else
                                     "Is the text about Target? A keyword match alone is insufficient; require a named or attributable link in the text."
                                     if HYBRID_PROMPT_V2 else f"Does the text directly refer to '{actual_target}'?"),
                    "criteria": {
                        "relevant": ("At least one Target keyword is mentioned in its intended sense, even in neutral news or a list."
                                     if keyword_scope else "The opinion, experience, or fact is directly about the target."),
                        "unrelated": ("No Target keyword is referred to in its intended sense; a homonym or different entity does not count."
                                      if keyword_scope else "The mention is coincidental, about another entity, or unrelated."),
                        "uncertain": ("It is unclear whether the text refers to a Target keyword."
                                      if keyword_scope else "Relevance is ambiguous or impossible to determine.")
                    }
                }
            }
        }

        candidates = [JEV_MODEL]
        for m in JEV_FALLBACK_MODELS:
            if m and m not in candidates:
                candidates.append(m)

        session = self.get_session()
        deadline = time.monotonic() + JEV_CASCADE_TIMEOUT

        for cand_model in candidates:
            payload["model"] = cand_model
            for attempt in range(1 + JEV_MAX_RETRIES):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, ""
                try:
                    response = session.post(url, headers=headers, json=payload, timeout=min(JEV_API_TIMEOUT, remaining))
                    if response.status_code == 200:
                        try:
                            resp_json = response.json()
                        except Exception:
                            self._record_provider_attempt("jev", cand_model, attempt > 0, provider_hint="openrouter")
                            return None, cand_model
                        self._record_provider_attempt("jev", cand_model, attempt > 0, resp_json, "openrouter")
                        validated = validate_jev_response(resp_json)
                        if validated:
                            return validated, cand_model
                        else:
                            return None, cand_model
                    self._record_provider_attempt("jev", cand_model, attempt > 0, provider_hint="openrouter")
                    if response.status_code in (401, 403):
                        print(f"  ❌ [Jev Auth Error] HTTP {response.status_code} on {cand_model}")
                        return None, cand_model
                    elif response.status_code == 404:
                        print(f"  ⚠️ [Jev 404 Not Found] {cand_model} -> Advancing to next candidate")
                        break
                    elif response.status_code == 400:
                        print(f"  ⚠️ [Jev 400 Bad Request] {cand_model} -> Non-retryable")
                        break
                    elif response.status_code in (408, 409, 425, 429) or response.status_code >= 500:
                        if attempt < JEV_MAX_RETRIES and _retry_within_deadline(attempt, deadline, response):
                            continue
                        else:
                            break
                    else:
                        break
                except requests.exceptions.RequestException:
                    self._record_provider_attempt("jev", cand_model, attempt > 0, provider_hint="openrouter")
                    if attempt < JEV_MAX_RETRIES and _retry_within_deadline(attempt, deadline):
                        continue
                    else:
                        break
        return None, ""

    def _call_deepseek_fallback(self, resolved_context: Dict[str, Any], jev_signal: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        DeepSeek deep reasoning fallback for low-confidence or conflicted posts.
        Bounded by DEEPSEEK_MAX_CONCURRENCY semaphore.
        """
        api_key = OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            return None, ""

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://blueeye.co.th"),
            "X-Title": os.environ.get("OPENROUTER_TITLE", "BlueEye Sentiment Analysis"),
            "Content-Type": "application/json"
        }

        user_prompt = build_deepseek_user_prompt(resolved_context, jev_signal)

        raw_providers = os.environ.get("OPENROUTER_PROVIDERS", "OpenInference,Relace")
        providers = [p.strip() for p in raw_providers.split(",") if p.strip()]
        allow_fallbacks = os.environ.get("OPENROUTER_ALLOW_FALLBACKS", "true").lower() in ("true", "1", "yes")

        provider_cfg = {"allow_fallbacks": allow_fallbacks}
        if providers:
            provider_cfg["order"] = providers

        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": (KEYWORD_SYSTEM_PROMPT if resolved_context.get("analysis_scope") == "keyword"
                                               else HYBRID_SYSTEM_PROMPT if HYBRID_PROMPT_V2 else PROBABILISTIC_SYSTEM_PROMPT)},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": DEEPSEEK_MAX_TOKENS,
            "provider": provider_cfg
        }

        session = self.get_session()
        provider_hint = providers[0] if providers else "openrouter"
        deadline = time.monotonic() + DEEPSEEK_CASCADE_TIMEOUT
        for attempt in range(1 + DEEPSEEK_MAX_RETRIES):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, ""
            try:
                if not self.deepseek_semaphore.acquire(timeout=remaining):
                    return None, ""
                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None, ""
                    response = session.post(url, headers=headers, json=payload, timeout=min(DEEPSEEK_API_TIMEOUT, remaining))
                finally:
                    self.deepseek_semaphore.release()
            except requests.exceptions.RequestException:
                self._record_provider_attempt("deepseek", DEEPSEEK_MODEL, attempt > 0, provider_hint=provider_hint)
                if attempt < DEEPSEEK_MAX_RETRIES and _retry_within_deadline(attempt, deadline):
                    continue
                break

            if response.status_code == 200:
                try:
                    res_data = response.json()
                except (ValueError, TypeError):
                    self._record_provider_attempt("deepseek", DEEPSEEK_MODEL, attempt > 0, provider_hint=provider_hint)
                    return None, ""
                self._record_provider_attempt("deepseek", DEEPSEEK_MODEL, attempt > 0, res_data, provider_hint)
                if not isinstance(res_data, dict):
                    return None, ""
                choices = res_data.get("choices")
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    return None, ""
                msg = choices[0].get("message")
                if not isinstance(msg, dict):
                    return None, ""
                parsed = self._extract_json_from_text(msg.get("content"))
                validated = validate_deepseek_response(parsed)
                return (validated, DEEPSEEK_MODEL) if validated else (None, "")
            self._record_provider_attempt("deepseek", DEEPSEEK_MODEL, attempt > 0, provider_hint=provider_hint)
            if response.status_code in (408, 409, 425, 429) or response.status_code >= 500:
                if attempt < DEEPSEEK_MAX_RETRIES and _retry_within_deadline(attempt, deadline, response):
                    continue
                break
            print(f"  ❌ [DeepSeek Error] HTTP {response.status_code} - Non-retryable")
            return None, ""

        return self._call_gemini_strict_fallback(resolved_context, deadline)

    def _call_gemini_strict_fallback(self, resolved_context: Dict[str, Any], deadline: Optional[float] = None) -> Tuple[Optional[Dict[str, Any]], str]:
        """Optional provider fallback to Gemini when DeepSeek is exhausted."""
        gemini_key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        if not gemini_key:
            return None, ""
        env_models = os.environ.get("VALIDATION_MODELS") or ""
        gemini_model = next((m.strip()[4:] for m in env_models.split(",") if m.strip().startswith("api:") and m.strip()[4:]), "")
        if not gemini_model:
            return None, ""

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={gemini_key}"
        user_prompt = build_deepseek_user_prompt(resolved_context)
        payload = {
            "system_instruction": {"parts": [{"text": (KEYWORD_SYSTEM_PROMPT if resolved_context.get("analysis_scope") == "keyword"
                                                      else HYBRID_SYSTEM_PROMPT if HYBRID_PROMPT_V2 else PROBABILISTIC_SYSTEM_PROMPT)}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json", "maxOutputTokens": 400}
        }
        session = self.get_session()
        try:
            remaining = (deadline - time.monotonic()) if deadline is not None else 30.0
            if remaining <= 0:
                return None, ""
            if not self.deepseek_semaphore.acquire(timeout=remaining):
                return None, ""
            try:
                remaining = (deadline - time.monotonic()) if deadline is not None else 30.0
                if remaining <= 0:
                    return None, ""
                resp = session.post(url, json=payload, timeout=min(parse_bounded_int("GEMINI_API_TIMEOUT", 30, min_val=1, max_val=300), remaining))
            finally:
                self.deepseek_semaphore.release()
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except (ValueError, TypeError):
                    self._record_provider_attempt("deepseek", gemini_model, provider_hint="google-ai")
                    return None, ""
                self._record_provider_attempt("deepseek", gemini_model, response_data=data, provider_hint="google-ai")
                candidates = data.get("candidates") if isinstance(data, dict) else None
                if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
                    parts = candidates[0].get("content", {}).get("parts", [])
                    txt = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
                    parsed = self._extract_json_from_text(txt)
                    validated = validate_deepseek_response(parsed)
                    if validated:
                        return validated, gemini_model
            else:
                self._record_provider_attempt("deepseek", gemini_model, provider_hint="google-ai")
        except Exception:
            self._record_provider_attempt("deepseek", gemini_model, provider_hint="google-ai")
            pass
        return None, ""

    def _hybrid_analyze_post(self, resolved_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Hybrid Pipeline:
        Entity Resolution -> TypeSafe Jev (System 1) -> confidence/conflict router
        -> accept Jev only when confidence >= 0.65 and no conflict
        -> otherwise DeepSeek deep reasoning -> Synthetic Reason Engine
        """
        post_id = resolved_context["post_id"]
        actual_target = resolved_context["actual_target"]
        project_name = resolved_context["project_name"]
        sentiment_target = (resolved_context.get("sentiment_target", actual_target)
                            if HYBRID_PROMPT_V2 else actual_target)
        keyword_scope = resolved_context.get("analysis_scope") == "keyword"

        is_placeholder = is_placeholder_target(sentiment_target)
        if HYBRID_PROMPT_V2 and is_placeholder:
            return self._neutral_unresolved_target_result(post_id, project_name)

        jev_raw = None
        jev_model_used = ""

        if not is_placeholder:
            state_prompt = build_jev_state_prompt(resolved_context)
            if keyword_scope:
                jev_raw, jev_model_used = self._call_typesafe_jev(state_prompt, sentiment_target, keyword_scope=True)
            else:
                jev_raw, jev_model_used = self._call_typesafe_jev(state_prompt, sentiment_target)

        route_info = route_sentiment(jev_raw, sentiment_target)

        # 1. Accept Jev if confidence >= 0.65 and no conflicts
        if route_info["accept_jev"]:
            probs = route_info["probabilities"]
            entity_found = route_info["entity_found"]
            policy = resolve_policy(probs, entity_found=entity_found)
            sanitized_tgt = sanitize_target(sentiment_target)
            reason = generate_synthetic_reason(
                sanitized_tgt, probs["POSITIVE"], probs["NEGATIVE"], probs["NEUTRAL"],
                probs["AMBIGUOUS_OR_IRONY"], entity_found=entity_found)
            return {
                "post_id": post_id,
                "ai_sentiment": policy["score"],
                "sentiment": policy["sentiment"],
                "positive_percent": policy["pos"],
                "negative_percent": policy["neg"],
                "neutral_percent": policy["neu"],
                "irony_score": policy["irony_score"],
                "confidence": round(route_info["routing_confidence"], 4),
                "reason": reason,
                "entity_found": entity_found,
                "model": jev_model_used or JEV_MODEL,
                "route": "jev",
                "conflict_reasons": route_info["conflict_reasons"],
                "raw_probabilities": probs,
                "project_name": project_name
            }

        # 2. Otherwise route to DeepSeek deep-reasoning fallback
        jev_signal = None
        if jev_raw is not None:
            jev_signal = {
                "probabilities": route_info.get("probabilities"),
                "confidence": route_info.get("routing_confidence"),
                "conflict_reasons": route_info.get("conflict_reasons", [])
            }

        if getattr(self._thread_local, "defer_deepseek", False):
            return {
                "_deferred_deepseek": True,
                "resolved_context": resolved_context,
                "route_info": route_info,
                "jev_signal": jev_signal
            }
        return self._complete_deepseek_route(resolved_context, route_info, jev_signal)

    def _complete_deepseek_route(self, resolved_context, route_info, jev_signal):
        post_id = resolved_context["post_id"]
        actual_target = resolved_context["actual_target"]
        project_name = resolved_context["project_name"]
        sentiment_target = (resolved_context.get("sentiment_target", actual_target)
                            if HYBRID_PROMPT_V2 else actual_target)

        ds_result, ds_model_used = self._call_deepseek_fallback(resolved_context, jev_signal)
        if ds_result is None:
            return self._neutral_error_result(post_id, project_name)

        ds_probs = ds_result["probabilities"]
        ds_entity_found = ds_result["entity_found"]

        if not HYBRID_PROMPT_V2 and is_placeholder_target(sentiment_target) and not ds_entity_found:
            return self._neutral_error_result(post_id, project_name)

        policy = resolve_policy(ds_probs, entity_found=ds_entity_found)
        sanitized_tgt = sanitize_target(sentiment_target)
        reason = generate_synthetic_reason(
            sanitized_tgt, ds_probs["POSITIVE"], ds_probs["NEGATIVE"], ds_probs["NEUTRAL"],
            ds_probs["AMBIGUOUS_OR_IRONY"], entity_found=ds_entity_found)

        ds_confidence = round(max(ds_probs.values()), 4)

        return {
            "post_id": post_id,
            "ai_sentiment": policy["score"],
            "sentiment": policy["sentiment"],
            "positive_percent": policy["pos"],
            "negative_percent": policy["neg"],
            "neutral_percent": policy["neu"],
            "irony_score": policy["irony_score"],
            "confidence": ds_confidence,
            "reason": reason,
            "entity_found": ds_entity_found,
            "model": ds_model_used or DEEPSEEK_MODEL,
            "route": "deepseek",
            "conflict_reasons": route_info.get("conflict_reasons", []),
            "raw_probabilities": ds_probs,
            "project_name": project_name
        }

    @staticmethod
    def _neutral_error_result(post_id, project_name=""):
        """Explicit neutral default when inference cannot produce a usable result."""
        return {
            "post_id": str(post_id),
            "ai_sentiment": 0,
            "sentiment": "neutral",
            "positive_percent": 0,
            "negative_percent": 0,
            "neutral_percent": 100,
            "irony_score": 0,
            "confidence": 0.0,
            "reason": "ระบบ AI วิเคราะห์ไม่สำเร็จ จึงกำหนดผลเป็นกลางตามค่าเริ่มต้น",
            "entity_found": False,
            "model": "rule:provider_failure",
            "route": "rule",
            "conflict_reasons": [],
            "project_name": project_name
        }

    @staticmethod
    def _neutral_unresolved_target_result(post_id, project_name=""):
        result = OllamaSentimentAnalyzer._neutral_error_result(post_id, project_name)
        result["model"] = "rule:unresolved_target"
        result["reason"] = "ไม่สามารถระบุโปรเจกต์หรือคู่แข่งที่ต้องวิเคราะห์ได้"
        return result

    # -----------------------------------------------------------------
    # Main Pipeline: Hybrid (Jev + DeepSeek) or Legacy
    # -----------------------------------------------------------------
    def _analyze_single_post(self, post, company_name=""):
        raw_id = post.get("match_post_id") or post.get("id") or post.get("post_id") or post.get("msg_id", "")
        post_id = str(raw_id)
        raw_id = None
        for key in ("match_post_id", "id", "post_id", "msg_id"):
            val = post.get(key)
            if val is not None and str(val).strip() != "":
                raw_id = val
                break
        if raw_id is None:
            return None
        post_id = str(raw_id).strip()
        if not post_id:
            return None

        full_text = post.get("full_text") or post.get("content") or post.get("text") or post.get("message") or post.get("feedcontent") or ""
        clean_raw_text = re.sub(r"<[^>]+>", " ", str(full_text)) if full_text else ""
        clean_raw_text = re.sub(r"&[a-zA-Z0-9#]+;", " ", clean_raw_text)
        clean_raw_text = re.sub(r"\s+", " ", clean_raw_text).strip()

        # Check empty content upfront -> deterministic neutral rule
        if not clean_raw_text:
            return {
                "post_id": post_id,
                "ai_sentiment": 0,
                "sentiment": "neutral",
                "positive_percent": 0,
                "negative_percent": 0,
                "neutral_percent": 100,
                "irony_score": 0,
                "confidence": 0.0,
                "reason": "ไม่มีข้อความให้วิเคราะห์",
                "entity_found": False,
                "model": "rule:empty_content",
                "route": "rule",
                "conflict_reasons": []
            }

        if post.get("_analysis_scope") == "keyword":
            raw_keywords = post.get("keywords") or []
            if isinstance(raw_keywords, str):
                raw_keywords = raw_keywords.split(",")
            keywords = [str(value).strip() for value in raw_keywords if str(value).strip()]
            if not keywords:
                return self._neutral_unresolved_target_result(post_id)
            folded_text = clean_raw_text.casefold()
            matched_keywords = [value for value in keywords if value.casefold() in folded_text]
            target_keywords = matched_keywords or keywords
            keyword_target = ", ".join(target_keywords)
            return self._hybrid_analyze_post({
                "post_id": post_id,
                "analysis_scope": "keyword",
                "actual_target": keyword_target,
                "sentiment_target": keyword_target,
                "project_name": "",
                "project_desc": "",
                "keywords": target_keywords,
                "source_info": "",
                "clean_text": clean_raw_text,
                "capped_text": cap_text(clean_raw_text, max_chars=3000, keyword=target_keywords[0])
            })

        raw_kw = post.get("keywords") or post.get("keyword_name") or post.get("keyword")
        if isinstance(raw_kw, str):
            keywords = [k.strip() for k in raw_kw.split(",") if k.strip()]
        elif isinstance(raw_kw, (list, tuple)):
            keywords = [str(k).strip() for k in raw_kw if str(k).strip()]
        else:
            keywords = []
        kw_name = post.get("keyword_name", "")
        if not keywords and kw_name:
            keywords = [k.strip() for k in str(kw_name).split(",") if k.strip()]

        project_id = post.get("project_id")
        project_name = post.get("project_name", "")
        project_desc = post.get("project_desc", "")
        actual_target = post.get("actual_target")
        effective_company_name = ((post.get("company_name") or company_name)
                                  if HYBRID_PROMPT_V2 else company_name)
        resolved_proj = None

        if GLOBAL_PROJECT_RESOLVER:
            resolved_proj = GLOBAL_PROJECT_RESOLVER.resolve_target(
                project_id=project_id,
                keywords=keywords,
                company_name=effective_company_name,
                content_text=clean_raw_text
            )
            if not actual_target:
                actual_target = resolved_proj["actual_target"]
            if not project_name:
                project_name = resolved_proj.get("project_name", "")
            if not project_desc:
                project_desc = resolved_proj.get("project_desc", "")
        else:
            if not actual_target:
                if keywords:
                    actual_target = ", ".join(keywords)
                elif effective_company_name:
                    actual_target = effective_company_name
                else:
                    actual_target = "the Target Entity"

        feed_link = post.get("feed_link", "")
        post_user = post.get("post_user", "")
        if not feed_link and post_user:
            source_info = f"User={post_user}"
        elif feed_link:
            source_info = f"Source Link={feed_link}"
        else:
            source_info = "Source=Social Media Post"

        sentiment_target = (
            _sentiment_target(post.get("actual_target"), project_name, effective_company_name,
                              keywords, resolved_proj, post.get("tracking_kind", ""))
            if HYBRID_PROMPT_V2 else actual_target
        )
        first_keyword = keywords[0] if keywords else ""
        capped_text = (cap_text(clean_raw_text, max_chars=3000, keyword=first_keyword, target=sentiment_target)
                       if HYBRID_PROMPT_V2 else _legacy_cap_text(clean_raw_text, max_chars=8000,
                                                                   keyword=first_keyword))

        resolved_context = {
            "post_id": post_id,
            "actual_target": actual_target,
            "sentiment_target": sentiment_target,
            "project_name": project_name,
            "project_desc": project_desc,
            "keywords": keywords,
            "source_info": source_info,
            "clean_text": clean_raw_text,
            "capped_text": capped_text
        }

        # --- HYBRID PIPELINE (DEFAULT) ---
        if ENABLE_JEV_HYBRID:
            return self._hybrid_analyze_post(resolved_context)

        # --- LEGACY MODES (IF ENABLE_JEV_HYBRID=False) ---
        expanded_content = get_keyword_context(clean_raw_text, str(first_keyword), window=300) if first_keyword else clean_raw_text
        content = get_keyword_context(clean_raw_text, str(first_keyword), window=150) if first_keyword else clean_raw_text

        if ENABLE_PROBABILISTIC_MODE:
            if BYPASS_LOCAL_TRIAGE:
                return self._probabilistic_analyze_post(post_id, actual_target, source_info, expanded_content, project_name=project_name, project_desc=project_desc)
            else:
                has_sentiment = self._triage_post(post_id, content, actual_target)
                if not has_sentiment:
                    return {
                        "post_id": post_id,
                        "ai_sentiment": 0,
                        "sentiment": "neutral",
                        "positive_percent": 0,
                        "negative_percent": 0,
                        "neutral_percent": 100,
                        "irony_score": 0,
                        "confidence": 0.0,
                        "reason": "ไม่พบเนื้อหาแสดงความรู้สึก",
                        "model": self.model,
                        "route": "rule",
                        "conflict_reasons": []
                    }
                return self._probabilistic_analyze_post(post_id, actual_target, source_info, expanded_content, project_name=project_name, project_desc=project_desc)

        has_sentiment = self._triage_post(post_id, content, actual_target)
        if not has_sentiment:
            return {
                "post_id": post_id,
                "ai_sentiment": 0,
                "sentiment": "neutral",
                "positive_percent": 0,
                "negative_percent": 0,
                "neutral_percent": 100,
                "irony_score": 0,
                "confidence": 0.0,
                "reason": "ไม่พบเนื้อหาแสดงความรู้สึก",
                "model": self.model,
                "route": "rule",
                "conflict_reasons": []
            }
        return self._deep_analyze_post(post_id, actual_target, source_info, expanded_content)

    def _analyze_batch_fast_post(self, post, company_name):
        self._thread_local.defer_deepseek = True
        try:
            return self._analyze_single_post(post, company_name)
        finally:
            self._thread_local.defer_deepseek = False

    def analyze_post_sentiments(self, json_posts, company_name=""):
        if isinstance(json_posts, str):
            posts = json.loads(json_posts)
        else:
            posts = json_posts
        results = []
        usage_metrics = _BatchUsageMetrics()

        seen_ids = set()
        unique_posts = []
        duplicate_count = 0
        for post in posts:
            raw_id = post.get("match_post_id") or post.get("id") or post.get("post_id") or post.get("msg_id", "")
            pid = str(raw_id)
            raw_id = None
            for key in ("match_post_id", "id", "post_id", "msg_id"):
                val = post.get(key)
                if val is not None and str(val).strip() != "":
                    raw_id = val
                    break
            if raw_id is None:
                continue
            pid = str(raw_id).strip()
            if not pid:
                continue
            if pid in seen_ids:
                duplicate_count += 1
                continue
            seen_ids.add(pid)
            unique_posts.append(post)

        if duplicate_count > 0:
            print(f"  ⚠️ [Deduplication] พบโพสต์ซ้ำ {duplicate_count} รายการใน Batch (ประมวลผล {len(unique_posts)} โพสต์)")

        if not unique_posts:
            telemetry = usage_metrics.summary(results)
            return {"data": [], "token_usage": {"input": 0, "output": 0, "total": 0}, "telemetry": telemetry}

        with ThreadPoolExecutor(max_workers=self.CONCURRENT_WORKERS) as executor, \
                ThreadPoolExecutor(max_workers=self.DEEPSEEK_MAX_CONCURRENCY) as deep_executor:
            post_iter = iter(unique_posts)
            active_futures = {}
            fast_worker = self._analyze_batch_fast_post if ENABLE_JEV_HYBRID else self._analyze_single_post

            def fill_fast_window():
                while len(active_futures) < self.MAX_IN_FLIGHT:
                    try:
                        p = next(post_iter)
                    except StopIteration:
                        break
                    fut = executor.submit(self._run_with_usage_metrics, usage_metrics, fast_worker, p, company_name)
                    active_futures[fut] = p

            fill_fast_window()

            while active_futures:
                done, _ = concurrent.futures.wait(
                    active_futures.keys(),
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                for fut in done:
                    p = active_futures.pop(fut)
                    try:
                        res = fut.result()
                        if isinstance(res, dict) and res.get("_deferred_deepseek"):
                            slow_future = deep_executor.submit(
                                self._run_with_usage_metrics, usage_metrics, self._complete_deepseek_route,
                                res["resolved_context"], res["route_info"], res["jev_signal"]
                            )
                            active_futures[slow_future] = p
                        elif res is not None:
                            results.append(res)
                    except Exception as e:
                        failed_id = str(p.get("match_post_id") or p.get("post_id", ""))[:15]
                        print(f"  ❌ [Worker Error] Post {failed_id:<15} | {e}")
                        raw_id = next((p[key] for key in ("match_post_id", "id", "post_id", "msg_id")
                                       if p.get(key) is not None and str(p[key]).strip()), None)
                        if raw_id is not None:
                            results.append(self._neutral_error_result(str(raw_id).strip(), p.get("project_name", "")))

                fill_fast_window()

        telemetry = usage_metrics.summary(results)
        _log_cost_telemetry(telemetry)
        token_usage = {
            "input": telemetry["input_tokens"],
            "output": telemetry["output_tokens"],
            "total": telemetry["input_tokens"] + telemetry["output_tokens"]
        }
        return {"data": results, "token_usage": token_usage, "telemetry": telemetry}


# =============================================================================
# FLOW 1: Sentiment REST API Manager
# =============================================================================
class SentimentAPI:
    def __init__(self, analyzer=None):
        self.ollama = analyzer or OllamaSentimentAnalyzer()
        self.headers = {
            'X-Internal-Token': BE_API_TOKEN,
            'Content-Type': 'application/json'
        }
        self.last_pending_count = 0
        self.last_fetch_error = False

    def fetch_pending(self, date_from, date_to, retries=3, delay=2):
        self.last_fetch_error = False
        url = f"{BE_API_BASE_URL}/internal/sentiment/pending?date_from={date_from}&date_to={date_to}"
        print(f"\n🌐 [Flow 1: REST API] กำลังดึงข้อมูลผ่าน REST API ({url})...")
        for attempt in range(1, retries + 1):
            try:
                response = requests.get(url, headers=self.headers, timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list):
                        self.last_pending_count = len(data)
                        return data
                    elif isinstance(data, dict) and "data" in data:
                        posts = data["data"] if isinstance(data["data"], list) else []
                        self.last_pending_count = len(posts)
                        return posts
                    elif isinstance(data, dict) and "results" in data:
                        posts = data["results"] if isinstance(data["results"], list) else []
                        self.last_pending_count = len(posts)
                        return posts
                    elif isinstance(data, dict) and "posts" in data:
                        posts = data["posts"] if isinstance(data["posts"], list) else []
                        self.last_pending_count = len(posts)
                        return posts
                    else:
                        print("⚠️ API คืนค่ามาในรูปแบบที่ไม่คาดคิด (ไม่มีฟิลด์ list/data/results/posts)")
                        self.last_pending_count = 0
                        return []
                else:
                    print(f"❌ API Fetch Error (attempt {attempt}/{retries}) {response.status_code}: {response.text}")
                    if attempt < retries:
                        time.sleep(delay)
            except Exception as e:
                print(f"❌ Exception in fetch_pending (attempt {attempt}/{retries}): {e}")
                if attempt < retries:
                    time.sleep(delay)
        self.last_fetch_error = True
        self.last_pending_count = 0
        return []

    def bulk_update(self, results, retries=3, delay=2):
        if not results:
            print("  ⚠️ [REST API] ไม่มีผลลัพธ์ที่วิเคราะห์สำเร็จใน Batch นี้ (ข้ามการส่งข้อมูล)")
            return 0
            
        url = f"{BE_API_BASE_URL}/internal/sentiment/results"
        payload = {"results": results}
        
        for attempt in range(1, retries + 1):
            try:
                response = requests.post(url, headers=self.headers, json=payload, timeout=60)
                if response.status_code in [200, 201]:
                    try:
                        resp_body = response.json()
                        not_found = resp_body.get("not_found", [])
                        if not not_found and isinstance(resp_body.get("data"), dict):
                            not_found = resp_body["data"].get("not_found", [])

                        raw_updated = resp_body.get("updated")
                        if raw_updated is None and isinstance(resp_body.get("data"), dict):
                            raw_updated = resp_body["data"].get("updated")

                        if raw_updated is not None:
                            try:
                                actual_updated = int(raw_updated)
                            except (ValueError, TypeError):
                                actual_updated = len(results) - len(not_found)
                        else:
                            actual_updated = len(results) - len(not_found)

                        # If API returned 200 OK and not_found is empty, but actual_updated is 0
                        # (e.g. the API found results already identical),
                        # the batch was still processed successfully.
                        if actual_updated == 0 and not not_found and len(results) > 0:
                            actual_updated = len(results)

                        print(f"  ✅ [REST API] บันทึกข้อมูลสำเร็จ (ส่ง {len(results)} โพสต์ → API อัปเดต {actual_updated} รายการ)")
                        if not_found:
                            print(f"  ⚠️ [REST API] ไม่พบ match_post_id เหล่านี้ในระบบ: {not_found}")
                        return max(0, actual_updated)
                    except Exception:
                        print(f"  ✅ [REST API] บันทึกข้อมูลสำเร็จ ({len(results)} โพสต์)")
                        return len(results)
                else:
                    print(f"  ❌ API Update Error (attempt {attempt}/{retries}) {response.status_code}: {response.text}")
                    if attempt < retries:
                        time.sleep(delay)
            except Exception as e:
                print(f"  ❌ Exception in bulk_update (attempt {attempt}/{retries}): {e}")
                if attempt < retries:
                    time.sleep(delay)
        return 0

    def run(self, date_from, date_to, save_db=None):
        if save_db is None:
            save_db = os.environ.get("SAVE_DB", "false").lower() in ("true", "1", "yes")
        date_from = validate_date_str(date_from)
        date_to = validate_date_str(date_to)
        if date_from > date_to:
            raise ValueError(f"date_from ({date_from}) cannot be after date_to ({date_to})")
        pending_posts = self.fetch_pending(date_from, date_to)
        
        if not pending_posts:
            print("⏩ [REST API] ไม่มีข้อมูลใหม่ให้วิเคราะห์ (0 โพสต์)")
            return 0

        total = len(pending_posts)
        print(f"📦 [REST API] พบข้อความที่ต้องวิเคราะห์ทั้งหมด: {total} โพสต์")

        BATCH_SIZE = parse_bounded_int("BATCH_SIZE", 100, min_val=1, max_val=1000)
        total_updated = 0
        for batch_start in range(0, total, BATCH_SIZE):
            batch = pending_posts[batch_start:batch_start + BATCH_SIZE]
            batch_end = min(batch_start + BATCH_SIZE, total)
            batch_started = time.perf_counter()
            updated_before = total_updated
            print(f"\n🔄 [REST API] กำลังประมวลผล Batch {batch_start + 1}-{batch_end} จากทั้งหมด {total} โพสต์...")

            posts_for_ai = []
            for post in batch:
                content = post.get("content") or post.get("full_text") or post.get("text") or post.get("message") or post.get("feedcontent") or ""
                raw_content = "" if content is None or str(content).strip().lower() == "none" else str(content)
                text = re.sub(r"<[^>]+>", "", raw_content)
                text = re.sub(r"\s+", " ", text).strip()
                
                raw_kw = post.get("keywords") or post.get("keyword_name") or post.get("keyword")
                if isinstance(raw_kw, str):
                    keywords = [k.strip() for k in raw_kw.split(",") if k.strip()]
                elif isinstance(raw_kw, (list, tuple)):
                    keywords = [str(k).strip() for k in raw_kw if str(k).strip()]
                else:
                    keywords = []
                keyword = keywords[0] if keywords else ""
                clean_short_content = get_keyword_context(text, keyword, window=150)
                
                modified_post = post.copy()
                modified_post["content"] = clean_short_content
                modified_post["full_text"] = text
                modified_post["keywords"] = keywords
                modified_post["_analysis_scope"] = "keyword"
                posts_for_ai.append(modified_post)

            ollama_response = self.ollama.analyze_post_sentiments(posts_for_ai)
            ollama_results = ollama_response.get("data", [])
            
            ollama_map = {}
            if isinstance(ollama_results, list):
                for res in ollama_results:
                    if "post_id" in res and "ai_sentiment" in res:
                        ollama_map[str(res["post_id"])] = {
                            "val": res["ai_sentiment"],
                            "ai_sentiment": res["ai_sentiment"],
                            "sentiment": res.get("sentiment"),
                            "positive_percent": res.get("positive_percent", 0),
                            "negative_percent": res.get("negative_percent", 0),
                            "neutral_percent": res.get("neutral_percent", 100),
                            "irony_score": res.get("irony_score", 0),
                            "reason": res.get("reason", ""),
                            "model": res.get("model", "unknown")
                        }

            api_results = []
            unresolved_count = 0
            for idx, post_for_ai in enumerate(posts_for_ai, 1):
                raw_id = post_for_ai.get("match_post_id") or post_for_ai.get("id") or post_for_ai.get("post_id") or post_for_ai.get("msg_id", "")
                match_post_id = str(raw_id)
                if not match_post_id:
                    continue
                ai_content = post_for_ai.get("content", "").replace("\n", " ")
                
                if len(ai_content) > 120:
                    ai_content = ai_content[:120] + "..."

                if match_post_id in ollama_map:
                    if ollama_map[match_post_id]["model"] == "rule:unresolved_target":
                        unresolved_count += 1
                        continue  # Unknown project is not a valid neutral classification.
                    raw_val = ollama_map[match_post_id]["val"]
                    ai_reason = ollama_map[match_post_id]["reason"]
                    pos_score = ollama_map[match_post_id]["positive_percent"]
                    neg_score = ollama_map[match_post_id]["negative_percent"]
                    neu_score = ollama_map[match_post_id]["neutral_percent"]
                    irony_score = ollama_map[match_post_id].get("irony_score", 0)
                    model_used = ollama_map[match_post_id]["model"]

                    # Primary sentiment string from analyzer, fallback to distribution derivation
                    sentiment_str = ollama_map[match_post_id].get("sentiment")
                    if not sentiment_str:
                        if pos_score > neg_score and pos_score > neu_score:
                            sentiment_str = "positive"
                        elif neg_score > pos_score and neg_score > neu_score:
                            sentiment_str = "negative"
                        elif pos_score == neg_score and pos_score > neu_score:
                            sentiment_str = "negative"
                        else:
                            sentiment_str = "neutral"

                    icon = "🟢" if sentiment_str == "positive" else ("🔴" if sentiment_str == "negative" else "⚪")
                        
                    int_id = int(match_post_id) if match_post_id.isdigit() else match_post_id

                    api_results.append({
                        "match_post_id": raw_id,
                        "id": int_id,
                        "post_id": post_for_ai.get("post_id", raw_id),
                        "sentiment": sentiment_str,
                        "sentiment_score": raw_val,
                        "sentiment_status": "1",
                        "sentiment_reason": ai_reason,
                        "ai_reason": ai_reason,
                        "sentiment_scores": {
                            "positive": pos_score,
                            "neutral": neu_score,
                            "negative": neg_score,
                            "model": model_used
                        }
                    })
                    
                    keywords = post_for_ai.get("keywords", [])
                    keyword_str = ", ".join(keywords) if keywords else "None"
                    feed_link = post_for_ai.get("feed_link", "None")
                    p_name = post_for_ai.get("project_name", "")
                    p_desc = post_for_ai.get("project_desc", "")

                    print(f"  [{idx:02d}] {icon} 🆔 {match_post_id[:15]:<15} | {sentiment_str.upper():<8}")
                    if p_name:
                        desc_info = f" ({p_desc})" if p_desc else ""
                        print(f"       🏢 Project: {p_name}{desc_info}")
                    print(f"       🔑 Keyword: {keyword_str}")
                    print(f"       🔗 Source: {feed_link}")
                    print(f"       📄 Content: {ai_content}")
                    print(f"       📊 Distribution: POS {pos_score}% | NEG {neg_score}% | NEU {neu_score}% | Irony={irony_score}% | Legacy={raw_val} | Model={model_used}")
                    if ai_reason:
                        print(f"       💡 Reason: {ai_reason}")
                    print(f"  {'-'*90}")
                        
            if unresolved_count:
                print(f"  ⚠️ [REST API] Skipped {unresolved_count} posts with unresolved targets; they remain pending for project metadata.")
            if save_db:
                updated_count = self.bulk_update(api_results)
                total_updated += updated_count
            else:
                print(f"  🔒 [DRY-RUN: ปิดการบันทึก] ข้ามการบันทึกลง REST API ({len(api_results)} โพสต์) — จำลอง Payload ที่จะ POST:")
                for item in api_results:
                    print(f"      📝 [DRY-RUN POST] `/internal/sentiment/results` -> match_post_id='{item['match_post_id']}', id={item.get('id')}, sentiment='{item['sentiment']}', sentiment_scores={item['sentiment_scores']}")
                total_updated += len(api_results)
            fallback_count = sum(1 for item in ollama_map.values()
                                 if item.get("model") in ("rule:provider_failure", "rule:unresolved_target"))
            _log_batch_timing("REST API", batch_start, batch_end, len(batch), total_updated - updated_before,
                              fallback_count, save_db, batch_started)
        return total_updated


# =============================================================================
# Main Program Loop (Continuous Execution)
# =============================================================================
# Backward-compatible import name now points to the REST manager.
sentiment = SentimentAPI


def parse_run_mode(argv=None):
    parser = argparse.ArgumentParser(description="Run sentiment analysis through the REST API")
    parser.add_argument("--mode", choices=("rest",), default="rest")
    return parser.parse_args(argv).mode


def create_apps_for_mode(mode, analyzer):
    if mode != "rest":
        raise ValueError(f"Invalid run mode: {mode}")
    return SentimentAPI(analyzer=analyzer)


def run_main_loop(app_api, save_db, sleep_seconds):
    while True:
        start_time = time.time()
        
        yesterday = str(datetime.now() - timedelta(days=1))[:10]
        now       = str(datetime.now())[:10]

        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 เริ่มการดึงข้อมูลและวิเคราะห์รอบใหม่...")
        print(f"📅 ช่วงเวลาที่วิเคราะห์: {yesterday} ถึง {now}")
        print("-" * 75)

        total_posts = 0
        try:
            total_posts = app_api.run(yesterday, now, save_db=save_db)
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในระบบ REST API: {e}")

        end_time = time.time()
        total_time = end_time - start_time

        if not total_posts or total_posts == 0:
            print(f"\n⏳ ไม่มีข้อมูลใหม่ให้วิเคราะห์ (0 โพสต์) พัก {sleep_seconds} วินาทีก่อนเริ่มรอบถัดไป... (กด Ctrl+C เพื่อหยุดโปรแกรม)")
            try:
                time.sleep(sleep_seconds)
            except KeyboardInterrupt:
                print("\n🛑 หยุดการทำงานตามคำสั่งผู้ใช้ (Ctrl+C)")
                return
        else:
            posts_per_min = total_posts / (total_time / 60) if total_time > 0 else 0
            print(f"\n🎉 สิ้นสุดการทำงานในรอบนี้! วิเคราะห์ไปทั้งหมด {total_posts} โพสต์ (ใช้เวลา {total_time:.2f} วินาที | ⚡ {posts_per_min:.1f} posts/min)")
            print(f"⏳ รอ {sleep_seconds} วินาทีก่อนเริ่มรอบถัดไป... (กด Ctrl+C เพื่อหยุดโปรแกรม)")
            try:
                time.sleep(sleep_seconds)
            except KeyboardInterrupt:
                print("\n🛑 หยุดการทำงานตามคำสั่งผู้ใช้ (Ctrl+C)")
                return


if __name__ == "__main__":
    parse_run_mode()
    SAVE_DB = os.environ.get("SAVE_DB", "false").lower() in ("true", "1", "yes")
    SLEEP_SECONDS = max(0, int(os.environ.get("RUN_INTERVAL_SECONDS", "5")))

    shared_analyzer = OllamaSentimentAnalyzer(model=os.environ.get("OLLAMA_MODEL", "qcwind/qwen3-8b-instruct-Q4-K-M:latest"))
    app_api = create_apps_for_mode("rest", shared_analyzer)

    if SAVE_DB:
        print(" 🤖 SENTIMENT ANALYSIS SYSTEM (REST API - LIVE / SAVE TO DB)")
        print("=" * 75)
        print("💾 LIVE MODE: ระบบจะบันทึกผลลัพธ์ผ่าน REST API จริง")
    else:
        print(" 🤖 SENTIMENT ANALYSIS SYSTEM (REST API - MOCK MODE / NO SAVE)")
        print("=" * 75)
        print("🧪 MOCK MODE: การบันทึกจริงถูกปิดอยู่ ระบบจะแสดงผลก่อนบันทึกเท่านั้น")

    run_main_loop(app_api, SAVE_DB, SLEEP_SECONDS)
