# coding=utf-8
"""
Sentiment Analysis System (Hybrid: REST API + Direct MySQL/MongoDB)
Using Ollama (qwen3-8b-instruct) and Gemini Validation Cascades
"""

import os
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

# PyMySQL & MongoDB imports for Direct DB processing
import pymysql
from pymongo import MongoClient

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

# Import local DatabaseConnection helper if available
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import connection
    CONN = connection.DatabaseConnection()
except Exception as e:
    CONN = None
    print(f"⚠️ Warning: Could not initialize connection module for Direct DB: {e}")

try:
    from project_resolver import GLOBAL_PROJECT_RESOLVER, ProjectResolver
except ImportError:
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

JEV_MODEL = os.environ.get("JEV_MODEL", "typesafe/jev-1.13")
_raw_jev_fallbacks = os.environ.get("JEV_FALLBACK_MODELS", "~typesafe/jev-latest,typesafe/jev-latest")
JEV_FALLBACK_MODELS = [m.strip() for m in _raw_jev_fallbacks.split(",") if m.strip()]
JEV_API_TIMEOUT = parse_bounded_int("JEV_API_TIMEOUT", 20, min_val=1, max_val=300)
JEV_MAX_RETRIES = parse_bounded_int("JEV_MAX_RETRIES", 3, min_val=0, max_val=10)

DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek/deepseek-v4-flash-0731")
DEEPSEEK_API_TIMEOUT = parse_bounded_int("DEEPSEEK_API_TIMEOUT", 45, min_val=1, max_val=300)
DEEPSEEK_MAX_RETRIES = parse_bounded_int("DEEPSEEK_MAX_RETRIES", 2, min_val=0, max_val=10)
DEEPSEEK_MAX_CONCURRENCY = parse_bounded_int("DEEPSEEK_MAX_CONCURRENCY", 4, min_val=1, max_val=256)

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

PROBABILISTIC_SYSTEM_PROMPT = (
    "[ROLE & INTERFACE CONTRACT]\n"
    "You are a deterministic probability engine.\n"
    "Your sole task is to evaluate unstructured Thai social text against a predefined discrete decision space toward the specified Target Entity.\n"
    "Do not converse, do not justify, do not explain, and do not output conversational text.\n"
    "Output MUST strictly follow the requested probability distribution schema.\n\n"
    "[DECISION SPACE]\n"
    "- POSITIVE: Explicit praise, satisfaction, recommendation, or favorable evaluation toward Target Entity.\n"
    "- NEUTRAL: Factual statements, news, questions without sentiment, official announcements, or unrelated sentiment.\n"
    "- NEGATIVE: Complaints, dissatisfaction, criticism, bad experience, anger, or damage toward Target Entity.\n"
    "- AMBIGUOUS_OR_IRONY: Sarcasm, irony, mixed conflicting emotions, ambiguous tone, or insufficient context.\n\n"
    "[CALIBRATION RULES]\n"
    "1. Probabilities across all 4 choices must sum up to exactly 1.00.\n"
    "2. If sarcasm, irony, or mixed conflicting sentiment is present, allocate weight to AMBIGUOUS_OR_IRONY.\n"
    "3. Base your judgment strictly on semantic weight toward the Target Entity. Do not execute routing policy.\n\n"
    "[OUTPUT FORMAT]\n"
    "Return ONLY a valid JSON object matching this schema:\n"
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


def cap_text(text: str, max_chars: int = 8000, keyword: str = "") -> str:
    """Deterministically cap text to bounded Unicode characters keeping target neighborhood and edges."""
    if not text or len(text) <= max_chars:
        return text or ""
    if keyword and keyword in text:
        kw_idx = text.find(keyword)
        prefix = text[:1500]
        kw_start = max(0, kw_idx - 1500)
        kw_end = min(len(text), kw_idx + len(keyword) + 1500)
        kw_chunk = text[kw_start:kw_end]
        suffix = text[-1500:]
        return f"{prefix} ... {kw_chunk} ... {suffix}"
    half = (max_chars - 10) // 2
    return f"{text[:half]} ... {text[-half:]}"


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
            elif response.status_code == 404:
                print(f"  -> Ollama Model Not Found [{model_name}]: 404 (ข้ามไปยังโมเดลถัดไป)")
                return None
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
    # PASS 2: Deep Analysis (Gemma / Gemini API)
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
                # "gemma4:31b-cloud",
                "api:gemma-4-26b-a4b-it",
                "api:gemma-4-31b-it",
                "api:gemini-3.1-flash-lite",
                "api:gemini-2.5-flash",
                "api:gemini-3.5-flash-lite",
            ]
        openrouter_retries = int(os.environ.get("OPENROUTER_MAX_RETRIES", 2))
        gemini_retries = int(os.environ.get("GEMINI_MAX_RETRIES", 2))
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

    def _call_typesafe_jev(self, state_prompt: str, actual_target: str) -> Tuple[Optional[Dict[str, Any]], str]:
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
                    "instructions": (
                        f"Evaluate the public sentiment towards the target entity '{actual_target}'. "
                        "Select the best fitting category based on the text."
                    ),
                    "criteria": {
                        "positive": f"Praise, satisfaction, recommendation, endorsement, or good news directly benefiting {actual_target}.",
                        "neutral": f"Objective news report, factual information, PR announcement, or general inquiry regarding {actual_target}.",
                        "negative": f"Criticism, complaint, defect, boycott, dissatisfaction, damage, or frustration directed at {actual_target}.",
                        "irony": f"Sarcasm, mockery, cynical humor, satirical tone, or backhanded praise directed at {actual_target}."
                    }
                },
                "entity_relevance": {
                    "type": "choice",
                    "instructions": (
                        f"Determine if the post content specifically refers to or evaluates '{actual_target}'."
                    ),
                    "criteria": {
                        "relevant": f"The opinion, experience, or factual mention is directly about {actual_target}.",
                        "unrelated": f"The mention is coincidental, about another entity, or unrelated to {actual_target}.",
                        "uncertain": f"It is ambiguous, unclear, or impossible to determine if it refers to {actual_target}."
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
                            return None, cand_model
                        validated = validate_jev_response(resp_json)
                        if validated:
                            return validated, cand_model
                        else:
                            return None, cand_model
                    elif response.status_code in (401, 403):
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

        actual_target = resolved_context.get("actual_target", "the Target Entity")
        project_name = resolved_context.get("project_name", "")
        project_desc = resolved_context.get("project_desc", "")
        source_info = resolved_context.get("source_info", "")
        capped_text = resolved_context.get("capped_text", "")

        user_prompt_lines = [f"Target Entity={actual_target}"]
        if project_name:
            user_prompt_lines.append(f"Project={project_name}")
        if project_desc:
            user_prompt_lines.append(f"Project Context={project_desc}")
        user_prompt_lines.append(f"Source Info={source_info}")
        user_prompt_lines.append(f"Text={capped_text}")

        if jev_signal and isinstance(jev_signal, dict):
            probs = jev_signal.get("probabilities")
            conf = jev_signal.get("confidence")
            conflicts = jev_signal.get("conflict_reasons")
            user_prompt_lines.append(f"Prior Jev Assessment: probabilities={probs}, confidence={conf}, conflicts={conflicts}")

        user_prompt = "\n".join(user_prompt_lines)

        raw_providers = os.environ.get("OPENROUTER_PROVIDERS", "OpenInference,Relace")
        providers = [p.strip() for p in raw_providers.split(",") if p.strip()]
        allow_fallbacks = os.environ.get("OPENROUTER_ALLOW_FALLBACKS", "true").lower() in ("true", "1", "yes")

        provider_cfg = {"allow_fallbacks": allow_fallbacks}
        if providers:
            provider_cfg["order"] = providers

        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": PROBABILISTIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 400,
            "provider": provider_cfg
        }

        session = self.get_session()
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
                if attempt < DEEPSEEK_MAX_RETRIES and _retry_within_deadline(attempt, deadline):
                    continue
                break

            if response.status_code == 200:
                try:
                    res_data = response.json()
                except (ValueError, TypeError):
                    return None, ""
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
        user_prompt = f"Target Entity={resolved_context.get('actual_target', '')}\nText={resolved_context.get('capped_text', '')}"
        payload = {
            "system_instruction": {"parts": [{"text": PROBABILISTIC_SYSTEM_PROMPT}]},
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
                data = resp.json()
                candidates = data.get("candidates") if isinstance(data, dict) else None
                if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
                    parts = candidates[0].get("content", {}).get("parts", [])
                    txt = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
                    parsed = self._extract_json_from_text(txt)
                    validated = validate_deepseek_response(parsed)
                    if validated:
                        return validated, gemini_model
        except Exception:
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
        keywords = resolved_context["keywords"]
        source_info = resolved_context["source_info"]
        capped_text = resolved_context["capped_text"]

        is_placeholder = is_placeholder_target(actual_target)
        jev_raw = None
        jev_model_used = ""

        if not is_placeholder:
            kw_str = ", ".join(keywords) if keywords else actual_target
            state_prompt = (
                f"TARGET ENTITY: {actual_target}\n"
                f"PROJECT: {project_name}\n"
                f"KEYWORDS: {kw_str}\n"
                f"SOURCE INFO: {source_info}\n"
                f"POST CONTENT: {capped_text}"
            )
            jev_raw, jev_model_used = self._call_typesafe_jev(state_prompt, actual_target)

        route_info = route_sentiment(jev_raw, actual_target)

        # 1. Accept Jev if confidence >= 0.65 and no conflicts
        if route_info["accept_jev"]:
            probs = route_info["probabilities"]
            entity_found = route_info["entity_found"]
            policy = resolve_policy(probs, entity_found=entity_found)
            sanitized_tgt = sanitize_target(actual_target)
            reason = generate_synthetic_reason(
                sanitized_tgt,
                probs["POSITIVE"],
                probs["NEGATIVE"],
                probs["NEUTRAL"],
                probs["AMBIGUOUS_OR_IRONY"],
                entity_found=entity_found
            )
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
        is_placeholder = is_placeholder_target(actual_target)

        ds_result, ds_model_used = self._call_deepseek_fallback(resolved_context, jev_signal)
        if ds_result is None:
            return None

        ds_probs = ds_result["probabilities"]
        ds_entity_found = ds_result["entity_found"]

        if is_placeholder and not ds_entity_found:
            print(f"  ⚠️ [Target Resolution Warning] Post {post_id[:15]}: placeholder target without entity relevance -> deferring")
            return None

        policy = resolve_policy(ds_probs, entity_found=ds_entity_found)
        sanitized_tgt = sanitize_target(actual_target)
        reason = generate_synthetic_reason(
            sanitized_tgt,
            ds_probs["POSITIVE"],
            ds_probs["NEGATIVE"],
            ds_probs["NEUTRAL"],
            ds_probs["AMBIGUOUS_OR_IRONY"],
            entity_found=ds_entity_found
        )

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

        if GLOBAL_PROJECT_RESOLVER:
            resolved_proj = GLOBAL_PROJECT_RESOLVER.resolve_target(
                project_id=project_id,
                keywords=keywords,
                company_name=company_name,
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
                elif company_name:
                    actual_target = company_name
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

        first_keyword = keywords[0] if keywords else ""
        capped_text = cap_text(clean_raw_text, max_chars=8000, keyword=first_keyword)

        resolved_context = {
            "post_id": post_id,
            "actual_target": actual_target,
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
            return {"data": [], "token_usage": {"input": 0, "output": 0, "total": 0}}

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
                    fut = executor.submit(fast_worker, p, company_name)
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
                                self._complete_deepseek_route,
                                res["resolved_context"], res["route_info"], res["jev_signal"]
                            )
                            active_futures[slow_future] = p
                        elif res is not None:
                            results.append(res)
                    except Exception as e:
                        failed_id = str(p.get("match_post_id") or p.get("post_id", ""))[:15]
                        print(f"  ❌ [Worker Error] Post {failed_id:<15} | {e}")

                fill_fast_window()

        return {"data": results, "token_usage": {"input": 0, "output": 0, "total": 0}}


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
                        # (e.g. MySQL 0 rows changed because values already identical),
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
                keyword = keywords[0] if keywords else str(post.get("project_id", "") or "")

                # Resolve Project Info
                project_id = post.get("project_id")
                project_name = ""
                project_desc = ""
                if GLOBAL_PROJECT_RESOLVER and project_id:
                    p_info = GLOBAL_PROJECT_RESOLVER.get_project(project_id)
                    if p_info:
                        project_name = p_info.get("name", "")
                        project_desc = p_info.get("description", "")

                keyword = keywords[0] if keywords else (project_name or str(post.get("project_id", "") or ""))
                clean_short_content = get_keyword_context(text, keyword, window=150)
                
                modified_post = post.copy()
                modified_post["content"] = clean_short_content
                modified_post["full_text"] = text
                modified_post["keywords"] = keywords
                modified_post["project_id"] = project_id
                modified_post["project_name"] = project_name
                modified_post["project_desc"] = project_desc
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
            for idx, post_for_ai in enumerate(posts_for_ai, 1):
                raw_id = post_for_ai.get("match_post_id") or post_for_ai.get("id") or post_for_ai.get("post_id") or post_for_ai.get("msg_id", "")
                match_post_id = str(raw_id)
                if not match_post_id:
                    continue
                ai_content = post_for_ai.get("content", "").replace("\n", " ")
                
                if len(ai_content) > 120:
                    ai_content = ai_content[:120] + "..."

                if match_post_id in ollama_map:
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
                        
            if save_db:
                updated_count = self.bulk_update(api_results)
                total_updated += updated_count
            else:
                print(f"  🔒 [DRY-RUN: ปิดการบันทึก] ข้ามการบันทึกลง REST API ({len(api_results)} โพสต์) — จำลอง Payload ที่จะ POST:")
                for item in api_results:
                    print(f"      📝 [DRY-RUN POST] `/internal/sentiment/results` -> match_post_id='{item['match_post_id']}', id={item.get('id')}, sentiment='{item['sentiment']}', sentiment_scores={item['sentiment_scores']}")
                total_updated += len(api_results)
        return total_updated


# =============================================================================
# FLOW 2: Direct Database Manager (MySQL + MongoDB)
# =============================================================================
class SentimentDB:
    def __init__(self, config=None, analyzer=None):
        # หมายเหตุ: credentials (user/password) จัดการโดย connection.py ผ่าน .env เท่านั้น — ห้าม hardcode ในโค้ด
        self.config = config or {
            "mysql_host_1":   os.environ.get("MYSQL_HOST_1",   "10.130.84.170"),
            "mysql_host_2":   os.environ.get("MYSQL_HOST_2",   "10.130.69.57"),
            "mysql_db":       os.environ.get("MYSQL_DB",       "blue_eye"),
            "mongo_db":       os.environ.get("MONGO_DB",       "blue_eye"),
        }
        self.ollama = analyzer or OllamaSentimentAnalyzer()
        self.review_offset = 0

    def get_content(self, list_id_with_info, collection):
        list_content = []
        if not list_id_with_info or CONN is None:
            return list_content

        company_map   = {msg_id: comp         for (msg_id, comp, proj, post_user, kw_name) in list_id_with_info}
        project_map   = {msg_id: proj         for (msg_id, comp, proj, post_user, kw_name) in list_id_with_info}
        post_user_map = {msg_id: post_user    for (msg_id, comp, proj, post_user, kw_name) in list_id_with_info}
        keyword_map   = {msg_id: kw_name      for (msg_id, comp, proj, post_user, kw_name) in list_id_with_info}
        list_id = [msg_id for (msg_id, comp, proj, post_user, kw_name) in list_id_with_info]

        for attempt in range(1, 4):
            try:
                attempt_content = []
                DB_CONNECTION = CONN.get_mongo_client()
                if DB_CONNECTION is None:
                    raise Exception("Mongo Client connection returned None")
                DB = DB_CONNECTION[self.config.get("mongo_db", "blue_eye")]
                DB_COLLECTION = DB[collection]

                result = DB_COLLECTION.find({"_id": {"$in": list_id}})
                columnName = "feedcontent" if collection == "Feed" else "commentcontent"

                for e in result:
                    feedcontent = e.get(columnName) or ""
                    msg_id = e["_id"]
                    comp_name = company_map.get(msg_id, "")
                    proj_name = project_map.get(msg_id, "")
                    post_user = post_user_map.get(msg_id, "")
                    kw_name   = keyword_map.get(msg_id, "")
                    if not post_user:
                        post_user = str(msg_id).split("_")[0]
                    attempt_content.append((msg_id, feedcontent, comp_name, proj_name, post_user, kw_name))
                list_content = attempt_content
                break
            except Exception as e:
                print(f"❌ Error fetching Mongo content (Attempt {attempt}/3): {e}")
                if hasattr(CONN, 'reset_mongo'):
                    CONN.reset_mongo()
                if attempt < 3:
                    time.sleep(3)

        return list_content

    def mark_missing_content(self, missing_ids, host, server=1, table_prefix="own_match"):
        """บันทึกสถานะให้โพสต์ที่ไม่มีใน MongoDB เพื่อไม่ให้ค้างอยู่ในคิว sentiment_status = '0'"""
        if not missing_ids or CONN is None:
            return
        tunnel, DB_CONNECTION = None, None
        try:
            tunnel, DB_CONNECTION = CONN.get_mysql_connection(server=server, host=host, database=self.config["mysql_db"])
            cursor = None
            try:
                cursor = DB_CONNECTION.cursor()
                for msg_id in missing_ids:
                    for tbl in [table_prefix, f"{table_prefix}_daily", f"{table_prefix}_3months"]:
                        try:
                            cursor.execute(
                                f'UPDATE `{tbl}` SET `{table_prefix}_sentiment` = %s, `sentiment_status` = %s, `ai_reason` = %s WHERE msg_id = %s',
                                (0.00, "1", "Content not found in MongoDB", str(msg_id))
                            )
                        except Exception:
                            pass
                DB_CONNECTION.commit()
                print(f"  ⚠️ เคลียร์โพสต์ที่ไม่มีใน MongoDB ({len(missing_ids)} โพสต์) -> ปรับ status='1' เพื่อไม่ให้ค้างคิว")
            finally:
                if cursor is not None:
                    try:
                        cursor.close()
                    except Exception:
                        pass
        except Exception as e:
            print(f"  ❌ Error marking missing content: {e}")
        finally:
            if DB_CONNECTION:
                try:
                    DB_CONNECTION.close()
                except Exception:
                    pass
            if tunnel:
                try:
                    tunnel.stop()
                except Exception:
                    pass

    def get_db_connection(self, server=1, host=None, database=None):
        """Helper to obtain MySQL connection and SSH tunnel for testability and runtime."""
        if CONN is None:
            return None, None
        return CONN.get_mysql_connection(
            server=server,
            host=host or self.config.get(f"mysql_host_{server}"),
            database=database or self.config.get("mysql_db", "blue_eye")
        )

    def analysis(self, list_content, host=None, server=1, table_prefix="own_match", save_db=None, current_host=None):
        if not list_content:
            return 0

        actual_host = host or current_host or self.config.get(f"mysql_host_{server}", "localhost")

        if save_db is None:
            save_db = os.environ.get("SAVE_DB", "false").lower() in ("true", "1", "yes")

        tunnel, DB_CONNECTION = None, None
        if save_db:
            try:
                conn_res = self.get_db_connection(server=server, host=actual_host, database=self.config.get("mysql_db", "blue_eye"))
                if conn_res:
                    if isinstance(conn_res, tuple) and len(conn_res) == 2:
                        first, second = conn_res
                        if hasattr(first, "cursor"):
                            DB_CONNECTION, tunnel = first, second
                        else:
                            tunnel, DB_CONNECTION = first, second
                    elif hasattr(conn_res, "cursor"):
                        DB_CONNECTION = conn_res
            except Exception as e:
                print(f"❌ Error connecting to MySQL Server {server} ({actual_host}): {e}")
                return 0
            if DB_CONNECTION is None:
                print(f"⚠️ [Direct DB] ไม่สามารถเชื่อมต่อ DB ได้เนื่องจากการเชื่อมต่อล้มเหลว")
                return 0

        total_persisted = 0
        try:
            BATCH_SIZE = parse_bounded_int("BATCH_SIZE", 100, min_val=1, max_val=1000)
            total = len(list_content)
            print(f"\n📦 [Direct DB Server {server}] พบข้อความที่ต้องวิเคราะห์ ({table_prefix}): {total} โพสต์")

            for batch_start in range(0, total, BATCH_SIZE):
                batch = list_content[batch_start:batch_start + BATCH_SIZE]
                batch_end = min(batch_start + BATCH_SIZE, total)
                print(f"\n🔄 [Direct DB Server {server}] กำลังประมวลผล Batch {batch_start + 1}-{batch_end} จากทั้งหมด {total} โพสต์...")

                is_competitor = (table_prefix == "competitor_match")
                posts_for_ai = []
                batch_company_name = ""
                batch_project_name = ""
                empty_text_ids = []
                for item in batch:
                    if isinstance(item, (list, tuple)):
                        if len(item) >= 6:
                            _id, content, company_name, project_name, post_user, kw_name = item[:6]
                        elif len(item) == 5:
                            _id, content, company_name, project_name, post_user = item
                            kw_name = ""
                        elif len(item) >= 2:
                            _id, content = item[0], item[1]
                            company_name, project_name, post_user, kw_name = "", "", "", ""
                        else:
                            continue
                    elif isinstance(item, dict):
                        _id = item.get("match_post_id") or item.get("post_id") or item.get("id") or item.get("msg_id")
                        content = item.get("content") or item.get("full_text") or ""
                        company_name = item.get("company_name", "")
                        project_name = item.get("project_name", "")
                        post_user = item.get("post_user", "")
                        kw_name = item.get("keyword_name") or item.get("keywords", "")
                        if isinstance(kw_name, list):
                            kw_name = ", ".join(kw_name)
                    else:
                        continue

                    raw_content = "" if content is None or str(content).strip().lower() == "none" else str(content)
                    text = re.sub(r"<[^>]+>", "", raw_content)
                    text = re.sub(r"\s+", " ", text).strip()

                    if not batch_company_name and company_name:
                        batch_company_name = company_name
                    if not batch_project_name and project_name:
                        batch_project_name = project_name

                    if text:
                        first_keyword = kw_name.split(",")[0].strip() if kw_name else ""
                        clean_short_content = get_keyword_context(text, first_keyword, window=150)

                        if is_competitor:
                            target_hint = first_keyword if first_keyword else project_name
                        else:
                            target_hint = company_name

                        post_item = {
                            "post_id": str(_id),
                            "post_user": post_user,
                            "company_name": company_name,
                            "keyword_name": kw_name,
                            "keywords": [k.strip() for k in kw_name.split(",") if k.strip()] if kw_name else [],
                            "project_name": project_name,
                            "content": clean_short_content,
                            "full_text": text
                        }
                        if target_hint:
                            post_item["actual_target"] = target_hint
                        posts_for_ai.append(post_item)
                    else:
                        empty_text_ids.append(str(_id))

                ollama_map = {}
                for eid in empty_text_ids:
                    ollama_map[eid] = {
                        "val": 0,
                        "ai_sentiment": 0,
                        "sentiment": "neutral",
                        "positive_percent": 0,
                        "negative_percent": 0,
                        "neutral_percent": 100,
                        "irony_score": 0,
                        "confidence": 0,
                        "reason": "ไม่มีข้อความให้วิเคราะห์",
                        "model": "rule:empty_content"
                    }

                if posts_for_ai:
                    target_label = f"{'COMPETITOR' if is_competitor else 'OWN'} | Company: {batch_company_name} | Proj: {batch_project_name}"
                    print(f"  🚀 ส่ง {len(posts_for_ai)} โพสต์ไปยัง AI Engine ({target_label})")

                    ollama_response = self.ollama.analyze_post_sentiments(posts_for_ai, batch_company_name)
                    ollama_results = ollama_response.get("data", [])

                    if isinstance(ollama_results, list):
                        for res in ollama_results:
                            if "post_id" in res and "ai_sentiment" in res:
                                ollama_map[str(res["post_id"])] = {
                                    "val": res["ai_sentiment"],
                                    "ai_sentiment": res["ai_sentiment"],
                                    "sentiment": res.get("sentiment", "neutral"),
                                    "positive_percent": res.get("positive_percent", 0),
                                    "negative_percent": res.get("negative_percent", 0),
                                    "neutral_percent": res.get("neutral_percent", 100),
                                    "irony_score": res.get("irony_score", 0),
                                    "confidence": res.get("confidence", 0),
                                    "reason": res.get("reason", ""),
                                    "model": res.get("model", "unknown")
                                }

                if not ollama_map:
                    continue

                print(f"\n  📊 สรุปผลลัพธ์จาก AI Engine (สำเร็จ {len(ollama_map)}/{len(batch)} โพสต์)")
                print(f"  {'-'*90}")

                for idx, item in enumerate(batch, 1):
                    if isinstance(item, (list, tuple)):
                        str_id = str(item[0])
                        item_content = item[1] if len(item) > 1 else ""
                        item_comp = item[2] if len(item) > 2 else ""
                        item_user = item[4] if len(item) > 4 else ""
                    elif isinstance(item, dict):
                        str_id = str(item.get("match_post_id") or item.get("post_id") or item.get("id") or item.get("msg_id"))
                        item_content = item.get("content") or ""
                        item_comp = item.get("company_name", "")
                        item_user = item.get("post_user", "")
                    else:
                        continue

                    if str_id in ollama_map:
                        ollama_val = float(ollama_map[str_id]["ai_sentiment"])
                        ai_reason = ollama_map[str_id].get("reason", "")
                        if ollama_val > 0:
                            icon = "🟢 Positive"
                        elif ollama_val < 0:
                            icon = "🔴 Negative"
                        else:
                            icon = "⚪ Neutral "
                    else:
                        ollama_val = None
                        ai_reason = ""
                        icon = "⚠️ N/A     "

                    actual_target = next((p.get("actual_target") for p in posts_for_ai if p["post_id"] == str_id), item_comp)
                    ai_content = next((p["content"] for p in posts_for_ai if p["post_id"] == str_id), str(item_content))

                    original_preview = str(item_content).replace("\n", " ")
                    if len(original_preview) > 120:
                        original_preview = original_preview[:120] + "..."

                    print(f"  [{idx:02d}] 🆔 {str_id[:15]:<15} | {icon:<11} | User: {str(item_user)[:12]:<12} | Target: {actual_target[:15]:<15}")
                    if str_id in ollama_map:
                        irony_pct = ollama_map[str_id].get('irony_score', 0)
                        model_used = ollama_map[str_id].get('model', 'unknown')
                        print(f"       📊 Distribution: POS {ollama_map[str_id].get('positive_percent', 0)}% | NEG {ollama_map[str_id].get('negative_percent', 0)}% | NEU {ollama_map[str_id].get('neutral_percent', 100)}% | Irony={irony_pct}% | Legacy={ollama_val} | Model={model_used}")
                    if ai_reason:
                        print(f"       💡 Reason: {ai_reason}")
                    print(f"       📄 Content: {original_preview}")
                    print(f"  {'-'*90}")

                update_params = []
                for item in batch:
                    if isinstance(item, (list, tuple)):
                        str_id = str(item[0])
                    elif isinstance(item, dict):
                        str_id = str(item.get("match_post_id") or item.get("post_id") or item.get("id") or item.get("msg_id"))
                    else:
                        continue
                    if str_id in ollama_map:
                        sentiment_val = float(ollama_map[str_id]["ai_sentiment"])
                        ai_reason_val = ollama_map[str_id].get("reason", "") or ""
                        update_params.append((sentiment_val, "1", ai_reason_val, str_id))

                if not update_params:
                    continue

                if save_db:
                    try:
                        DB_CONNECTION.ping(reconnect=True)
                    except Exception:
                        try:
                            if DB_CONNECTION:
                                try: DB_CONNECTION.close()
                                except Exception: pass
                            if tunnel:
                                try: tunnel.stop()
                                except Exception: pass
                            reconn_res = self.get_db_connection(server=server, host=actual_host, database=self.config.get("mysql_db", "blue_eye"))
                            if reconn_res:
                                if isinstance(reconn_res, tuple) and len(reconn_res) == 2:
                                    first, second = reconn_res
                                    if hasattr(first, "cursor"):
                                        DB_CONNECTION, tunnel = first, second
                                    else:
                                        tunnel, DB_CONNECTION = first, second
                                elif hasattr(reconn_res, "cursor"):
                                    DB_CONNECTION = reconn_res
                        except Exception as reconn_err:
                            print(f"  ❌ Reconnecting to MySQL failed: {reconn_err}")
                            DB_CONNECTION = None
                    
                    if DB_CONNECTION is not None:
                        cursor = None
                        try:
                            cursor = DB_CONNECTION.cursor()
                            for tbl in [table_prefix, f"{table_prefix}_daily", f"{table_prefix}_3months"]:
                                cursor.executemany(
                                    f'UPDATE `{tbl}` SET `{table_prefix}_sentiment` = %s, `sentiment_status` = %s, `ai_reason` = %s WHERE msg_id = %s',
                                    update_params
                                )
                            DB_CONNECTION.commit()
                            total_persisted += len(update_params)
                            print(f"  💾 บันทึกลง MySQL เรียบร้อย ({len(update_params)} โพสต์)")
                        except Exception as sql_err:
                            if DB_CONNECTION:
                                try:
                                    DB_CONNECTION.rollback()
                                except Exception:
                                    pass
                            print(f"  ❌ Batch update failed, rolled back ({len(update_params)} posts remain retryable): {sql_err}")
                        finally:
                            if cursor is not None:
                                try:
                                    cursor.close()
                                except Exception:
                                    pass
                    else:
                        print(f"  ❌ ไม่สามารถบันทึกลง MySQL ได้เนื่องจากการเชื่อมต่อ DB ล้มเหลว")
                else:
                    total_persisted += len(update_params)
                    print(f"  🚫 [MOCKUP DB] ข้ามการบันทึกลง MySQL ({len(update_params)} โพสต์) — จำลองค่าที่จะ UPDATE:")
                    for param in update_params:
                        print(f"      📝 [MySQL UPDATE] Tables: [`{table_prefix}`, `{table_prefix}_daily`, `{table_prefix}_3months`]")
                        print(f"         └─ SET `{table_prefix}_sentiment` = {param[0]}, `sentiment_status` = '{param[1]}', `ai_reason` = '{param[2]}' WHERE msg_id = '{param[3]}'")

            return total_persisted
        except Exception as e:
            print(f"❌ Error during DB analysis execution: {e}")
            return total_persisted
        finally:
            if 'DB_CONNECTION' in locals() and DB_CONNECTION:
                try:
                    DB_CONNECTION.close()
                except Exception:
                    pass
            if 'tunnel' in locals() and tunnel:
                try:
                    tunnel.stop()
                except Exception:
                    pass

    def run(self, date_from, date_to, save_db=None):
        if save_db is None:
            save_db = os.environ.get("SAVE_DB", "false").lower() in ("true", "1", "yes")
        date_from = validate_date_str(date_from)
        date_to = validate_date_str(date_to)

        # Calculate next_date = date_to + 1 day for sargable range query
        dt_to = datetime.strptime(date_to, "%Y-%m-%d")
        next_date = (dt_to + timedelta(days=1)).strftime("%Y-%m-%d")

        if CONN is None:
            print("⚠️ [Direct DB] ไม่สามารถเชื่อมต่อ DB ได้เนื่องจากเชื่อมต่อ connection module ล้มเหลว")
            return 0

        total_processed_posts = 0

        # Pagination logic:
        # - ถ้า save_db=True (บันทึกจริง): offset ต้องเป็น 0 เสมอ เพราะแถวที่ทำเสร็จจะเปลี่ยนเป็น status='1' หลุดจากคิวไปเอง
        # - ถ้า save_db=False (Mock Mode): เลื่อน offset ตาม review_offset เพื่อเปิดดูหน้าถัดไปเรื่อยๆ โดยไม่ซ้ำชุดเดิม
        if save_db:
            offset = 0
        else:
            page_size = 100
            offset = self.review_offset * page_size
            print(f"📄 [Direct DB Mock] Review cycle={self.review_offset} (OFFSET {offset})")

        targets = [
            {
                "name": "OWN MATCH",
                "table_prefix": "own_match",
                "sql_feed": (
                    f"SELECT omd.msg_id, IFNULL(c.company_name, '') as company_name, "
                    f"IFNULL(ck.company_keyword_name, '') as project_name, IFNULL(omd.post_user, '') as post_user, "
                    f"IFNULL(GROUP_CONCAT(DISTINCT k.keyword_name SEPARATOR ', '), '') as keyword_name "
                    f"FROM own_match_daily omd "
                    f"LEFT JOIN company_keyword ck ON omd.company_keyword_id = ck.company_keyword_id "
                    f"LEFT JOIN client c ON omd.client_id = c.client_id "
                    f"LEFT JOIN own_key_match okm ON okm.own_match_id = omd.own_match_id "
                    f"LEFT JOIN keyword k ON okm.keyword_id = k.keyword_id "
                    f"WHERE omd.msg_time >= '{date_from} 00:00:00' AND omd.msg_time < '{next_date} 00:00:00' "
                    f"AND omd.sentiment_status = '0' AND omd.match_type = 'Feed' "
                    f"GROUP BY omd.msg_id, company_name, project_name, post_user "
                    f"ORDER BY MIN(omd.msg_time) ASC "
                    f"LIMIT 100 OFFSET {offset}"
                ),
                "sql_comment": (
                    f"SELECT omd.msg_id, IFNULL(c.company_name, '') as company_name, "
                    f"IFNULL(ck.company_keyword_name, '') as project_name, IFNULL(omd.post_user, '') as post_user, "
                    f"IFNULL(GROUP_CONCAT(DISTINCT k.keyword_name SEPARATOR ', '), '') as keyword_name "
                    f"FROM own_match_daily omd "
                    f"LEFT JOIN company_keyword ck ON omd.company_keyword_id = ck.company_keyword_id "
                    f"LEFT JOIN client c ON omd.client_id = c.client_id "
                    f"LEFT JOIN own_key_match okm ON okm.own_match_id = omd.own_match_id "
                    f"LEFT JOIN keyword k ON okm.keyword_id = k.keyword_id "
                    f"WHERE omd.msg_time >= '{date_from} 00:00:00' AND omd.msg_time < '{next_date} 00:00:00' "
                    f"AND omd.sentiment_status = '0' AND omd.match_type = 'Comment' "
                    f"GROUP BY omd.msg_id, company_name, project_name, post_user "
                    f"ORDER BY MIN(omd.msg_time) ASC "
                    f"LIMIT 100 OFFSET {offset}"
                ),
            },
            {
                "name": "COMPETITOR MATCH",
                "table_prefix": "competitor_match",
                "sql_feed": (
                    f"SELECT cmd.msg_id, IFNULL(c.company_name, '') as company_name, "
                    f"IFNULL(ck.company_keyword_name, '') as project_name, IFNULL(cmd.post_user, '') as post_user, "
                    f"IFNULL(GROUP_CONCAT(DISTINCT k.keyword_name SEPARATOR ', '), '') as keyword_name "
                    f"FROM competitor_match_daily cmd "
                    f"LEFT JOIN company_keyword ck ON cmd.company_keyword_id = ck.company_keyword_id "
                    f"LEFT JOIN client c ON cmd.client_id = c.client_id "
                    f"LEFT JOIN competitor_key_match ckm ON ckm.competitor_match_id = cmd.competitor_match_id "
                    f"LEFT JOIN keyword k ON ckm.keyword_id = k.keyword_id "
                    f"WHERE cmd.msg_time >= '{date_from} 00:00:00' AND cmd.msg_time < '{next_date} 00:00:00' "
                    f"AND cmd.sentiment_status = '0' AND cmd.match_type = 'Feed' "
                    f"GROUP BY cmd.msg_id, company_name, project_name, post_user "
                    f"ORDER BY MIN(cmd.msg_time) ASC "
                    f"LIMIT 100 OFFSET {offset}"
                ),
                "sql_comment": (
                    f"SELECT cmd.msg_id, IFNULL(c.company_name, '') as company_name, "
                    f"IFNULL(ck.company_keyword_name, '') as project_name, IFNULL(cmd.post_user, '') as post_user, "
                    f"IFNULL(GROUP_CONCAT(DISTINCT k.keyword_name SEPARATOR ', '), '') as keyword_name "
                    f"FROM competitor_match_daily cmd "
                    f"LEFT JOIN company_keyword ck ON cmd.company_keyword_id = ck.company_keyword_id "
                    f"LEFT JOIN client c ON cmd.client_id = c.client_id "
                    f"LEFT JOIN competitor_key_match ckm ON ckm.competitor_match_id = cmd.competitor_match_id "
                    f"LEFT JOIN keyword k ON ckm.keyword_id = k.keyword_id "
                    f"WHERE cmd.msg_time >= '{date_from} 00:00:00' AND cmd.msg_time < '{next_date} 00:00:00' "
                    f"AND cmd.sentiment_status = '0' AND cmd.match_type = 'Comment' "
                    f"GROUP BY cmd.msg_id, company_name, project_name, post_user "
                    f"ORDER BY MIN(cmd.msg_time) ASC "
                    f"LIMIT 100 OFFSET {offset}"
                ),
            },
        ]

        for server_id in [1, 2]:
            current_host = self.config.get(f"mysql_host_{server_id}")
            if not current_host:
                continue
                
            print(f"\n🖥️  [Direct DB] เริ่มทำงานกับ MYSQL SERVER {server_id} ({current_host})")

            for target in targets:
                print(f"🎯 กำลังดึงข้อมูล: {target['name']} (Server {server_id})...")
                list_content = []
                
                try:
                    _item_feed = CONN.getfromdb(
                        query=target["sql_feed"], 
                        DB='mysqldb', 
                        database=self.config["mysql_db"], 
                        server=server_id, 
                        host=current_host
                    )
                    list_id_feed = [(x[0], x[1], x[2], x[3], x[4]) for x in (_item_feed or [])]
                    print(f"  👉 พบข้อมูลจาก Feed: {len(list_id_feed)} โพสต์")
                    feed_content = self.get_content(list_id_feed, "Feed")
                    list_content = feed_content

                    # ป้องกันโพสต์ค้างคิว: ตรวจจับและเคลียร์โพสต์ที่ไม่มีเนื้อหาใน MongoDB
                    if save_db and list_id_feed:
                        found_feed_ids = {item[0] for item in feed_content}
                        missing_feed_ids = [x[0] for x in list_id_feed if x[0] not in found_feed_ids]
                        if missing_feed_ids:
                            self.mark_missing_content(missing_feed_ids, current_host, server=server_id, table_prefix=target["table_prefix"])
                except Exception as e:
                    print(f"  ❌ Error querying Feed SQL: {e}")

                try:
                    _item_comment = CONN.getfromdb(
                        query=target["sql_comment"], 
                        DB='mysqldb', 
                        database=self.config["mysql_db"], 
                        server=server_id, 
                        host=current_host
                    )
                    list_id_comment = [(x[0], x[1], x[2], x[3], x[4]) for x in (_item_comment or [])]
                    print(f"  👉 พบข้อมูลจาก Comment: {len(list_id_comment)} โพสต์")
                    comment_content = self.get_content(list_id_comment, "Comment")
                    list_content += comment_content

                    # ป้องกันโพสต์ค้างคิว: ตรวจจับและเคลียร์โพสต์ที่ไม่มีเนื้อหาใน MongoDB
                    if save_db and list_id_comment:
                        found_comment_ids = {item[0] for item in comment_content}
                        missing_comment_ids = [x[0] for x in list_id_comment if x[0] not in found_comment_ids]
                        if missing_comment_ids:
                            self.mark_missing_content(missing_comment_ids, current_host, server=server_id, table_prefix=target["table_prefix"])
                except Exception as e:
                    print(f"  ❌ Error querying Comment SQL: {e}")

                if list_content:
                    try:
                        persisted = self.analysis(list_content, current_host, server=server_id, table_prefix=target["table_prefix"], save_db=save_db)
                        total_processed_posts += (persisted or 0)
                    except Exception as e:
                        print(f"  ❌ Error analyzing content for {target['name']} (Server {server_id}): {e}")
                else:
                    print(f"  ⏩ ไม่มีข้อมูลใหม่สำหรับ {target['name']} (Server {server_id})")

        # เลื่อนหน้า review สำหรับรอบถัดไป เฉพาะโหมด Mock (save_db=False)
        if not save_db:
            if total_processed_posts == 0 and self.review_offset > 0:
                self.review_offset = 0
            else:
                self.review_offset += 1
        else:
            self.review_offset = 0
        return total_processed_posts


# Backward compatibility alias
sentiment = SentimentDB


# =============================================================================
# Main Program Loop (Continuous Execution)
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 75)
    print(" 🤖 SENTIMENT ANALYSIS SYSTEM (REST API ONLY - RETROACTIVE / HISTORICAL MODE)")
    print("=" * 75)

    shared_analyzer = OllamaSentimentAnalyzer(model=os.environ.get("OLLAMA_MODEL", "qcwind/qwen3-8b-instruct-Q4-K-M:latest"))
    app_api = SentimentAPI(analyzer=shared_analyzer)
    
    SLEEP_MINUTES = int(os.environ.get("RUN_INTERVAL_MINUTES", 10))
    # ปิดการบันทึกข้อมูล (Dry-run mode) ตามคำสั่ง — หากต้องการเปิดบันทึกจริงในภายหลังให้ตั้ง SAVE_DB=true
    SAVE_DB = os.environ.get("SAVE_DB", "false").lower() in ("true", "1", "yes")

    if not SAVE_DB:
        print("🔒 [DRY-RUN MODE] ปิดการบันทึกข้อมูลลง REST API / Database (save_db=False)")
    else:
        print("💾 โหมดบันทึกจริง: เปิดการบันทึกผลลัพธ์ลง REST API (save_db=True)")

    # รองรับการระบุ วันที่เริ่มต้น (DATE_FROM) และ วันที่สิ้นสุด (DATE_TO) เพื่อวิเคราะห์ย้อนหลัง
    # เช่น python ai_sentimentREST_API.py 2026-08-01 2026-08-25
    # หรือระบุใน .env / env variables: DATE_FROM=2026-08-01 DATE_TO=2026-08-25
    custom_date_from = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DATE_FROM", "")
    custom_date_to   = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("DATE_TO", "")

    is_retroactive_mode = bool(custom_date_from and custom_date_to)

    if is_retroactive_mode:
        print(f"📜 [RETROACTIVE MODE] เริ่มทำการวิเคราะห์ย้อนหลังสำหรับช่วง: {custom_date_from} ถึง {custom_date_to}")
        print("🌐 โหมดการทำงาน: REST API ONLY (ดึงและจำลองผลลัพธ์ผ่าน REST API)")
        print("-" * 75)

        grand_total = 0
        round_num = 0
        consecutive_zero_updates = 0

        try:
            while True:
                round_num += 1
                round_start = time.time()
                print(f"\n🔁 [รอบที่ {round_num}] กำลังดึงข้อมูล pending ผ่าน REST API สำหรับช่วง {custom_date_from} ถึง {custom_date_to}...")
                
                updated_posts = app_api.run(custom_date_from, custom_date_to, save_db=SAVE_DB)
                round_time = time.time() - round_start

                if not SAVE_DB:
                    grand_total += updated_posts
                    print(f"\n🔒 [DRY-RUN] จำลองการวิเคราะห์เสร็จสิ้น 1 รอบ ({updated_posts} โพสต์) — หยุดการทำงานอัตโนมัติเนื่องจากปิดการบันทึก (save_db=False)")
                    break

                if getattr(app_api, "last_fetch_error", False):
                    consecutive_zero_updates += 1
                    print(f"\n❌ [รอบที่ {round_num}] เกิดข้อผิดพลาดในการดึงข้อมูลจาก REST API (ล้มเหลวต่อเนื่อง {consecutive_zero_updates}/3)")
                    if consecutive_zero_updates >= 3:
                        print(f"\n🛑 หยุดการทำงาน: ไม่สามารถเชื่อมต่อ REST API หรือดึงข้อมูลได้ติดต่อกัน {consecutive_zero_updates} รอบ กรุณาตรวจสอบ Network / BE_API_TOKEN / Backend Status")
                        break
                    print(f"⏳ รอ 5 วินาทีก่อนลองรอบถัดไป...")
                    time.sleep(5)
                    continue

                if getattr(app_api, "last_pending_count", 0) == 0:
                    print(f"\n✅ ไม่มีข้อมูลค้างเหลือในคิวแล้ว (queue ว่าง 0 โพสต์)!")
                    break

                if not updated_posts or updated_posts == 0:
                    consecutive_zero_updates += 1
                    print(f"  ⚠️ มีโพสต์ค้าง {app_api.last_pending_count} โพสต์ แต่ไม่สามารถอัปเดตได้ในรอบนี้ (รอบที่ {round_num}, ล้มเหลวต่อเนื่อง {consecutive_zero_updates}/3)")
                    if consecutive_zero_updates >= 3:
                        print(f"\n🛑 หยุดการทำงาน: ไม่สามารถอัปเดตโพสต์ที่ค้างอยู่ได้ติดต่อกัน {consecutive_zero_updates} รอบ กรุณาตรวจสอบสถานะ API / Model")
                        break
                    print(f"⏳ รอ 5 วินาทีก่อนลองรอบถัดไป...")
                    time.sleep(5)
                    continue

                consecutive_zero_updates = 0
                grand_total += updated_posts
                print(f"\n📊 [รอบที่ {round_num}] อัปเดตผ่าน REST API สำเร็จ {updated_posts} โพสต์ (ใช้เวลา {round_time:.1f} วินาที) | รวมสะสม: {grand_total} โพสต์")
                print(f"⏳ พัก 3 วินาทีก่อนดึงรอบถัดไป...")
                time.sleep(3)

        except KeyboardInterrupt:
            print(f"\n🛑 หยุดการทำงานตามคำสั่งผู้ใช้ (Ctrl+C)")

        print(f"\n{'=' * 75}")
        print(f"🎉 สรุปผล RETROACTIVE MODE: วิเคราะห์ย้อนหลังเสร็จสิ้น (REST API ONLY)")
        print(f"   📅 ช่วงวันที่: {custom_date_from} ถึง {custom_date_to}")
        print(f"   📦 จำนวนโพสต์ที่อัปเดตผ่าน REST API ทั้งหมด: {grand_total} โพสต์")
        print(f"   🔁 จำนวนรอบที่รัน: {round_num} รอบ")
        print(f"{'=' * 75}")
        sys.exit(0)

    # โหมดทำงานต่อเนื่อง (Loop Mode - REST API ONLY)
    while True:
        start_time = time.time()
        
        yesterday = str(datetime.now() - timedelta(days=1))[:10]
        now       = str(datetime.now())[:10]

        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 เริ่มการดึงข้อมูลและวิเคราะห์รอบใหม่ (REST API ONLY)...")
        print(f"📅 ช่วงเวลาที่วิเคราะห์: {yesterday} ถึง {now}")
        print("-" * 75)

        total_posts = 0
        try:
            total_posts = app_api.run(yesterday, now, save_db=SAVE_DB)
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในระบบ REST API: {e}")

        end_time = time.time()
        total_time = end_time - start_time

        if not total_posts or total_posts == 0:
            print(f"\n⏳ ไม่มีข้อมูลใหม่ให้วิเคราะห์ (0 โพสต์) พัก 1 นาทีก่อนเริ่มรอบถัดไป... (กด Ctrl+C เพื่อหยุดโปรแกรม)")
            try:
                time.sleep(60)
            except KeyboardInterrupt:
                print("\n🛑 หยุดการทำงานตามคำสั่งผู้ใช้ (Ctrl+C)")
                sys.exit(0)
        else:
            posts_per_min = total_posts / (total_time / 60) if total_time > 0 else 0
            print(f"\n🎉 สิ้นสุดการทำงานในรอบนี้! วิเคราะห์และอัปเดตผ่าน REST API ไปทั้งหมด {total_posts} โพสต์ (ใช้เวลา {total_time:.2f} วินาที | ⚡ {posts_per_min:.1f} posts/min)")
            print(f"⏳ รอ {SLEEP_MINUTES} นาทีก่อนเริ่มรอบถัดไป... (กด Ctrl+C เพื่อหยุดโปรแกรม)")
            try:
                time.sleep(SLEEP_MINUTES * 60)
            except KeyboardInterrupt:
                print("\n🛑 หยุดการทำงานตามคำสั่งผู้ใช้ (Ctrl+C)")
                sys.exit(0)
