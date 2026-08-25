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
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

# PyMySQL & MongoDB imports for Direct DB processing
import pymysql
from pymongo import MongoClient

try:
    from dotenv import load_dotenv
    load_dotenv()
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

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
BE_API_TOKEN = os.environ.get("BE_API_TOKEN", "10b6150ab6b7a8ef90904a32ef875f2b62789753109733d0194165d9ed3e854c")
BE_API_BASE_URL = os.environ.get("BE_API_BASE_URL", "https://api.blueeye.io/api/v1")

# Reconfigure stdout for UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


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
    """Validate YYYY-MM-DD date string (ป้องกันรูปแบบวันที่ไม่ถูกต้อง)"""
    try:
        datetime.strptime(str(date_str), "%Y-%m-%d")
        return str(date_str)
    except ValueError:
        raise ValueError(f"Invalid date format (expected YYYY-MM-DD): {date_str}")


# =============================================================================
# Ollama & Gemini Sentiment Analyzer Engine
# =============================================================================
class OllamaSentimentAnalyzer:
    CONCURRENT_WORKERS = 3

    def __init__(self, model="qcwind/qwen3-8b-instruct-Q4-K-M:latest"):
        self.model = model
        self.host_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self.base_url = f"{self.host_url}/api/generate"
        self.chat_url = f"{self.host_url}/api/chat"
        self.triage_timeout = int(os.environ.get("TRIAGE_TIMEOUT_SEC", 60))
        self.session = requests.Session()

    def _call_gemini_api(self, model_name, system_instruction, user_prompt, max_retries=3):
        if not GEMINI_API_KEY:
            return None
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.0}
        }
        
        for attempt in range(max_retries):
            try:
                response = self.session.post(url, json=payload, timeout=45)
                if response.status_code == 200:
                    res_data = response.json()
                    result_text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                    parsed_res = self._parse_json_result(result_text)
                    if parsed_res is None:
                        print(f"  -> Gemini API Parsing Error [{model_name}] (attempt {attempt + 1}/{max_retries}): {result_text[:200]}")
                        if attempt < max_retries - 1:
                            time.sleep(2)
                        continue
                    return parsed_res
                else:
                    print(f"  -> Gemini API Error [{model_name}]: {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
            except Exception as e:
                print(f"  -> Gemini API Exception [{model_name}]: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        return None

    def _call_ollama_generic(self, model_name, system_instruction, user_prompt):
        payload_generate = {
            "model": model_name,
            "system": system_instruction,
            "prompt": user_prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "seed": 42}
        }
        try:
            response = self.session.post(self.base_url, json=payload_generate, timeout=120)
            if response.status_code == 200:
                result_text = response.json().get("response", "{}")
                return self._parse_json_result(result_text)
        except Exception:
            pass
            
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
            response = self.session.post(self.chat_url, json=payload_chat, timeout=120)
            if response.status_code == 200:
                result_text = response.json().get("message", {}).get("content", "{}")
                return self._parse_json_result(result_text)
        except Exception:
            pass
        return None

    def _normalize_distribution(self, positive=0, negative=0, neutral=100):
        """Normalize percentage distribution so the total is exactly 100."""
        def num(v, default=0):
            try:
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
        return 0

    def _parse_json_result(self, result_text):
        if not result_text:
            return None
        clean_text = re.sub(r'<think>.*?</think>', '', result_text, flags=re.DOTALL).strip()
        clean_text = clean_text.replace('```json', '').replace('```', '').strip()
        json_match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        if json_match:
            clean_text = json_match.group(0)
        cleaned_json = re.sub(r',\s*([\}\]])', r'\1', clean_text)

        try:
            parsed = json.loads(cleaned_json)
        except json.JSONDecodeError:
            def find_num(names):
                for name in names:
                    m = re.search(r'[`"\']?' + re.escape(name) + r'[`"\']?\s*[:=]\s*(-?\d+(?:\.\d+)?)', clean_text, re.I)
                    if m:
                        return m.group(1)
                return None
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
            reason = reason_m.group(1).strip() if reason_m else clean_text[:100].replace("\n", " ")
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

        reason = parsed.get("reason", "")
        if isinstance(reason, str):
            reason = re.sub(r'\(?\s*(?:ตาม)?กฎข้อ\s*[\d\s,และ|-]+\)?', '', reason, flags=re.I)
            reason = re.sub(r'\(?\s*Rule\s*[\d\s,and|-]+\)?', '', reason, flags=re.I).strip()
        else:
            reason = ""
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

        clean_text = re.sub(r'<think>.*?</think>', '', result_text, flags=re.DOTALL).strip()
        clean_text = clean_text.replace('```json', '').replace('```', '').strip()

        json_match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        if json_match:
            clean_text = json_match.group(0)

        try:
            parsed = json.loads(clean_text)
            triage_val = str(parsed.get("triage", "yes")).strip().lower()
            return triage_val in ("yes", "true", "1")
        except json.JSONDecodeError:
            triage_match = re.search(r'[`"\']?triage[`"\']?\s*[:=]\s*[`"\']?(yes|no|true|false)[`"\']?', clean_text, re.IGNORECASE)
            if triage_match:
                val = triage_match.group(1).lower()
                return val in ("yes", "true")
            return True

    def _triage_post(self, post_id, content, actual_target=""):
        """PASS 1: conservative relevance + sentiment triage."""
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
        deep_system = (
            f"You are an expert Thai Social Media Brand Reputation Analyst. "
            f"Your Target Entity is '{actual_target}'. "
            "Analyze sentiment specifically TOWARD that Target Entity, not the overall mood of the post. "
            "A keyword match alone is not enough."
        )
        deep_prompt = (
            f"Target Entity={actual_target}\nSource Info={source_info}\nText={expanded_content}\n\n"
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
        validation_models = [
            "api:gemma-4-31b-it",
            "api:gemma-4-26b-a4b-it",
            "api:gemini-3.5-flash-lite",
            "api:gemini-3.1-flash-lite",
            "api:gemini-2.5-flash",
        ]
        for val_model in validation_models:
            actual_api_model = val_model.replace("api:", "", 1)
            res = self._call_gemini_api(actual_api_model, deep_system, deep_prompt, max_retries=1)
            if res and "ai_sentiment" in res:
                entity_found = res.get("entity_found", True)
                if isinstance(entity_found, str):
                    entity_found = entity_found.lower() in ("true", "1")
                if not entity_found:
                    res.update({"positive_percent":0,"negative_percent":0,"neutral_percent":100,"ai_sentiment":0})
                res["post_id"] = post_id
                return res
        return None

    # -----------------------------------------------------------------
    # Main Pipeline: 2-Pass (Triage → Deep Analysis)
    # -----------------------------------------------------------------
    def _analyze_single_post(self, post, company_name=""):
        post_id = str(post.get("match_post_id") or post.get("post_id", ""))
        keywords = post.get("keywords", [])
        kw_name = post.get("keyword_name", "")
        if not keywords and kw_name:
            keywords = [k.strip() for k in kw_name.split(",") if k.strip()]

        actual_target = post.get("actual_target")
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

        if "full_text" in post and first_keyword:
            content = get_keyword_context(post["full_text"], str(first_keyword), window=150)
            expanded_content = get_keyword_context(post["full_text"], str(first_keyword), window=300)
        else:
            content = post.get("content", "")
            expanded_content = content

        # --- PASS 1: Fast Triage (Qwen 8B Local) ---
        has_sentiment = self._triage_post(post_id, content, actual_target)

        if not has_sentiment:
            print(f"  ⏩ [Pass 1 Triage] Post {post_id[:15]:<15} | triage=no -> Neutral (0)")
            return {
                "post_id": post_id,
                "ai_sentiment": 0,
                "positive_percent": 0,
                "negative_percent": 0,
                "neutral_percent": 100,
                "confidence": 0,
                "reason": "ไม่พบเนื้อหาแสดงความรู้สึก"
            }

        # --- PASS 2: Deep Analysis (Gemma / Gemini API) ---
        print(f"  🔍 [Pass 2 Deep AI] Post {post_id[:15]:<15} | triage=yes -> กำลังวิเคราะห์เชิงลึก...")
        deep_result = self._deep_analyze_post(post_id, actual_target, source_info, expanded_content)

        if deep_result:
            return deep_result

        print(f"  ⚠️ [Pass 2 Failed] Post {post_id[:15]:<15} | ทุก API ล้มเหลว -> Defer ให้รอบถัดไป")
        return None

    def analyze_post_sentiments(self, json_posts, company_name=""):
        if isinstance(json_posts, str):
            posts = json.loads(json_posts)
        else:
            posts = json_posts
        results = []

        with ThreadPoolExecutor(max_workers=self.CONCURRENT_WORKERS) as executor:
            future_to_post = {
                executor.submit(self._analyze_single_post, post, company_name): post
                for post in posts
            }

            for future in as_completed(future_to_post):
                try:
                    result = future.result()
                except Exception as e:
                    failed_post = future_to_post[future]
                    failed_id = str(failed_post.get("match_post_id") or failed_post.get("post_id", ""))[:15]
                    print(f"  ❌ [Worker Error] Post {failed_id:<15} | {e}")
                    continue
                if result is not None:
                    results.append(result)

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

    def fetch_pending(self, date_from, date_to):
        url = f"{BE_API_BASE_URL}/internal/sentiment/pending?date_from={date_from}&date_to={date_to}"
        print(f"\n🌐 [Flow 1: REST API] กำลังดึงข้อมูลผ่าน REST API...")
        try:
            response = requests.get(url, headers=self.headers, timeout=60)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "data" in data:
                    return data["data"]
                elif isinstance(data, dict) and "results" in data:
                    return data["results"]
                else:
                    print("⚠️ API คืนค่ามาในรูปแบบที่ไม่คาดคิด (ไม่มีฟิลด์ list/data)")
                    return []
            else:
                print(f"❌ API Fetch Error {response.status_code}: {response.text}")
                return []
        except Exception as e:
            print(f"❌ Exception in fetch_pending: {e}")
            return []

    def bulk_update(self, results):
        if not results:
            return
            
        url = f"{BE_API_BASE_URL}/internal/sentiment/results"
        payload = {"results": results}
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            if response.status_code in [200, 201]:
                try:
                    resp_body = response.json()
                    actual_updated = resp_body.get("updated", "?")
                    not_found = resp_body.get("not_found", [])
                    print(f"  ✅ [REST API] บันทึกข้อมูลสำเร็จ (ส่ง {len(results)} โพสต์ → API อัปเดตจริง {actual_updated} รายการ)")
                    if not_found:
                        print(f"  ⚠️ [REST API] ไม่พบ match_post_id เหล่านี้ในระบบ: {not_found}")
                except Exception:
                    print(f"  ✅ [REST API] บันทึกข้อมูลสำเร็จ ({len(results)} โพสต์)")
            else:
                print(f"  ❌ API Update Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"  ❌ Exception in bulk_update: {e}")

    def run(self, date_from, date_to, save_db=True):
        date_from = validate_date_str(date_from)
        date_to = validate_date_str(date_to)
        pending_posts = self.fetch_pending(date_from, date_to)
        
        if not pending_posts:
            print("⏩ [REST API] ไม่มีข้อมูลใหม่ให้วิเคราะห์ (0 โพสต์)")
            return 0

        total = len(pending_posts)
        print(f"📦 [REST API] พบข้อความที่ต้องวิเคราะห์ทั้งหมด: {total} โพสต์")

        BATCH_SIZE = 5
        for batch_start in range(0, total, BATCH_SIZE):
            batch = pending_posts[batch_start:batch_start + BATCH_SIZE]
            batch_end = min(batch_start + BATCH_SIZE, total)
            print(f"\n🔄 [REST API] กำลังประมวลผล Batch {batch_start + 1}-{batch_end} จากทั้งหมด {total} โพสต์...")

            posts_for_ai = []
            for post in batch:
                content = post.get("content", "")
                text = re.sub(r"<[^>]+>", "", str(content))
                text = re.sub(r"\s+", " ", text).strip()
                
                keywords = post.get("keywords", [])
                keyword = str(keywords[0]) if keywords else str(post.get("project_id", ""))
                clean_short_content = get_keyword_context(text, keyword, window=150)
                
                modified_post = post.copy()
                modified_post["content"] = clean_short_content
                modified_post["full_text"] = text
                posts_for_ai.append(modified_post)

            json_str = json.dumps(posts_for_ai, ensure_ascii=False)
            
            ollama_response = self.ollama.analyze_post_sentiments(json_str)
            ollama_results = ollama_response.get("data", [])
            
            ollama_map = {}
            if isinstance(ollama_results, list):
                for res in ollama_results:
                    if "post_id" in res and "ai_sentiment" in res:
                        ollama_map[str(res["post_id"])] = {
                            "val": res["ai_sentiment"],
                            "reason": res.get("reason", "")
                        }

            api_results = []
            for idx, post_for_ai in enumerate(posts_for_ai, 1):
                match_post_id = str(post_for_ai.get("match_post_id", ""))
                ai_content = post_for_ai.get("content", "").replace("\n", " ")
                
                if len(ai_content) > 120:
                    ai_content = ai_content[:120] + "..."

                if match_post_id in ollama_map:
                    raw_val = ollama_map[match_post_id]["val"]
                    ai_reason = ollama_map[match_post_id]["reason"]

                    if raw_val > 0:
                        sentiment_str = "positive"
                        icon = "🟢"
                    elif raw_val < 0:
                        sentiment_str = "negative"
                        icon = "🔴"
                    else:
                        sentiment_str = "neutral"
                        icon = "⚪"
                        
                    api_results.append({
                        "match_post_id": match_post_id,
                        "sentiment": sentiment_str,
                        "sentiment_reason": ai_reason
                    })
                    
                    keywords = post_for_ai.get("keywords", [])
                    keyword_str = ", ".join(keywords) if keywords else "None"
                    feed_link = post_for_ai.get("feed_link", "None")

                    print(f"  [{idx:02d}] {icon} 🆔 {match_post_id[:15]:<15} | {sentiment_str.upper():<8}")
                    print(f"       🔑 Keyword: {keyword_str}")
                    print(f"       🔗 Source: {feed_link}")
                    print(f"       📄 Content: {ai_content}")
                    if ai_reason:
                        print(f"       💡 Reason: {ai_reason}")
                    print(f"  {'-'*90}")
                        
            if save_db:
                self.bulk_update(api_results)
            else:
                print(f"  🚫 [MOCKUP API] ข้ามการบันทึกลง API ({len(api_results)} โพสต์)")
        return total


# =============================================================================
# FLOW 2: Direct Database Manager (MySQL + MongoDB)
# =============================================================================
class SentimentDB:
    def __init__(self, config=None, analyzer=None):
        self.config = config or {
            "mysql_host_1":   os.environ.get("MYSQL_HOST_1",   "10.130.84.170"),
            "mysql_host_2":   os.environ.get("MYSQL_HOST_2",   "10.130.69.57"),
            "mysql_port":     int(os.environ.get("MYSQL_PORT", 3306)),
            "mysql_user":     os.environ.get("MYSQL_USER",     "blueeyeremote"),
            "mysql_password": os.environ.get("MYSQL_PASSWORD", "BEremotemysql3075"),
            "mysql_db":       os.environ.get("MYSQL_DB",       "blue_eye"),
            "mongo_host":     os.environ.get("MONGO_HOST",     "10.130.72.139"),
            "mongo_port":     int(os.environ.get("MONGO_PORT", 34596)),
            "mongo_user":     os.environ.get("MONGO_USER",     "blueeyeharvest"),
            "mongo_password": os.environ.get("MONGO_PASSWORD", "BEharvest3075"),
            "mongo_db":       os.environ.get("MONGO_DB",       "blue_eye"),
        }
        self.ollama = analyzer or OllamaSentimentAnalyzer()

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
                DB_CONNECTION = CONN.get_mongo_client()
                if DB_CONNECTION is None:
                    raise Exception("Mongo Client connection returned None")
                DB = DB_CONNECTION[self.config.get("mongo_db", "blue_eye")]
                DB_COLLECTION = DB[collection]

                result = DB_COLLECTION.find({"_id": {"$in": list_id}})
                columnName = "feedcontent" if collection == "Feed" else "commentcontent"

                for e in result:
                    feedcontent = e.get(columnName, "")
                    msg_id = e["_id"]
                    comp_name = company_map.get(msg_id, "")
                    proj_name = project_map.get(msg_id, "")
                    post_user = post_user_map.get(msg_id, "")
                    kw_name   = keyword_map.get(msg_id, "")
                    if not post_user:
                        post_user = str(msg_id).split("_")[0]
                    list_content.append((msg_id, feedcontent, comp_name, proj_name, post_user, kw_name))
                break
            except Exception as e:
                print(f"❌ Error fetching Mongo content (Attempt {attempt}/3): {e}")
                if hasattr(CONN, 'reset_mongo'):
                    CONN.reset_mongo()
                if attempt < 3:
                    time.sleep(3)

        return list_content

    def analysis(self, list_content, host, server=1, table_prefix="own_match", save_db=True):
        if not list_content or CONN is None:
            return

        try:
            tunnel, DB_CONNECTION = CONN.get_mysql_connection(server=server, host=host, database=self.config["mysql_db"])
        except Exception as e:
            print(f"❌ Error connecting to MySQL Server {server} ({host}): {e}")
            return

        try:
            BATCH_SIZE = 5
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
                for (_id, content, company_name, project_name, post_user, kw_name) in batch:
                    text = re.sub(r"<[^>]+>", "", str(content))
                    text = re.sub(r"\s+", " ", text).strip()

                    if not batch_company_name and company_name:
                        batch_company_name = company_name
                    if not batch_project_name and project_name:
                        batch_project_name = project_name

                    if text:
                        first_keyword = kw_name.split(",")[0].strip() if kw_name else ""
                        clean_short_content = get_keyword_context(text, first_keyword, window=150)

                        if is_competitor:
                            actual_target = first_keyword if first_keyword else project_name
                        else:
                            actual_target = company_name

                        posts_for_ai.append({
                            "post_id": str(_id),
                            "post_user": post_user,
                            "company_name": company_name,
                            "keyword_name": kw_name,
                            "actual_target": actual_target,
                            "content": clean_short_content,
                            "full_text": text
                        })

                if not posts_for_ai:
                    continue

                json_str = json.dumps(posts_for_ai, ensure_ascii=False)
                target_label = f"{'COMPETITOR' if is_competitor else 'OWN'} | Company: {batch_company_name} | Proj: {batch_project_name}"
                print(f"  🚀 ส่ง {len(posts_for_ai)} โพสต์ไปยัง Ollama ({target_label})")

                ollama_response = self.ollama.analyze_post_sentiments(json_str, batch_company_name)
                ollama_results = ollama_response.get("data", [])

                ollama_map = {}
                if isinstance(ollama_results, list):
                    for res in ollama_results:
                        if "post_id" in res and "ai_sentiment" in res:
                            ollama_map[str(res["post_id"])] = {
                                "ai_sentiment": res["ai_sentiment"],
                                "confidence": res.get("confidence", 0),
                                "reason": res.get("reason", "")
                            }

                print(f"\n  📊 สรุปผลลัพธ์จาก Ollama (สำเร็จ {len(ollama_map)}/{len(batch)} โพสต์)")
                print(f"  {'-'*90}")

                for idx, (_id, content, company_name, project_name, post_user, kw_name) in enumerate(batch, 1):
                    str_id = str(_id)

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

                    actual_target = next((p["actual_target"] for p in posts_for_ai if p["post_id"] == str_id), company_name)
                    ai_content = next((p["content"] for p in posts_for_ai if p["post_id"] == str_id), str(content))

                    original_preview = str(content).replace("\n", " ")
                    if len(original_preview) > 120:
                        original_preview = original_preview[:120] + "..."

                    print(f"  [{idx:02d}] 🆔 {str_id[:15]:<15} | {icon:<11} | User: {str(post_user)[:12]:<12} | Target: {actual_target[:15]:<15}")
                    if ai_reason:
                        print(f"       💡 Reason: {ai_reason}")
                    print(f"       📄 Content: {original_preview}")
                    print(f"  {'-'*90}")

                if save_db:
                    try:
                        DB_CONNECTION.ping(reconnect=True)
                    except Exception as e:
                        print(f"  ⚠️ Warning: MySQL Ping/Reconnect failed: {e}")
                    
                    cursor = DB_CONNECTION.cursor()
                    
                    for (_id, content, company_name, project_name, post_user, kw_name) in batch:
                        str_id = str(_id)
                        if str_id not in ollama_map:
                            continue
                        sentiment_val = float(ollama_map[str_id]["ai_sentiment"])
                        ai_reason_val = ollama_map[str_id].get("reason", "") or ""
                    
                        for tbl in [table_prefix, f"{table_prefix}_daily", f"{table_prefix}_3months"]:
                            cursor.execute(
                                f'UPDATE `{tbl}` SET `{table_prefix}_sentiment` = %s, `sentiment_status` = %s, `ai_reason` = %s WHERE msg_id = %s',
                                (sentiment_val, "1", ai_reason_val, str(_id))
                            )
                    
                    DB_CONNECTION.commit()
                    cursor.close()
                    print(f"  💾 บันทึกลง MySQL เรียบร้อย ({len([x for x in batch if str(x[0]) in ollama_map])} โพสต์)")
                else:
                    print(f"  🚫 [MOCKUP DB] ข้ามการบันทึกลง MySQL ({len([x for x in batch if str(x[0]) in ollama_map])} โพสต์)")

        except Exception as e:
            print(f"❌ Error during DB analysis execution: {e}")
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

    def run(self, date_from, date_to, save_db=True):
        if CONN is None:
            print("⚠️ [Direct DB] ไม่สามารถเชื่อมต่อ DB ได้เนื่องจากเชื่อมต่อ connection module ล้มเหลว")
            return 0

        total_processed_posts = 0

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
                    f"WHERE date(omd.msg_time) BETWEEN '{date_from}' AND '{date_to}' "
                    f"AND omd.sentiment_status = '0' AND omd.match_type = 'Feed' "
                    f"GROUP BY omd.msg_id, company_name, project_name, post_user "
                    f"ORDER BY omd.msg_time ASC "
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
                    f"WHERE date(omd.msg_time) BETWEEN '{date_from}' AND '{date_to}' "
                    f"AND omd.sentiment_status = '0' AND omd.match_type = 'Comment' "
                    f"GROUP BY omd.msg_id, company_name, project_name, post_user "
                    f"ORDER BY omd.msg_time ASC "
                )
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
                    f"WHERE date(cmd.msg_time) BETWEEN '{date_from}' AND '{date_to}' "
                    f"AND cmd.sentiment_status = '0' AND cmd.match_type = 'Feed' "
                    f"GROUP BY cmd.msg_id, company_name, project_name, post_user "
                    f"ORDER BY cmd.msg_time ASC "
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
                    f"WHERE date(cmd.msg_time) BETWEEN '{date_from}' AND '{date_to}' "
                    f"AND cmd.sentiment_status = '0' AND cmd.match_type = 'Comment' "
                    f"GROUP BY cmd.msg_id, company_name, project_name, post_user "
                    f"ORDER BY cmd.msg_time ASC "
                )
            }
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
                    list_content = self.get_content(list_id_feed, "Feed")
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
                    list_content += self.get_content(list_id_comment, "Comment")
                except Exception as e:
                    print(f"  ❌ Error querying Comment SQL: {e}")

                if list_content:
                    total_processed_posts += len(list_content)
                    try:
                        self.analysis(list_content, current_host, server=server_id, table_prefix=target["table_prefix"], save_db=save_db)
                    except Exception as e:
                        print(f"  ❌ Error analyzing content for {target['name']} (Server {server_id}): {e}")
                else:
                    print(f"  ⏩ ไม่มีข้อมูลใหม่สำหรับ {target['name']} (Server {server_id})")

        return total_processed_posts


# =============================================================================
# Main Program Loop (Continuous Execution)
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 75)
    print(" 🤖 SENTIMENT ANALYSIS SYSTEM (REST API ONLY - RETROACTIVE / HISTORICAL MODE)")
    print("=" * 75)

    shared_analyzer = OllamaSentimentAnalyzer(model="qcwind/qwen3-8b-instruct-Q4-K-M:latest")
    app_api = SentimentAPI(analyzer=shared_analyzer)
    
    SLEEP_MINUTES = int(os.environ.get("RUN_INTERVAL_MINUTES", 10))

    # รองรับการระบุ วันที่เริ่มต้น (DATE_FROM) และ วันที่สิ้นสุด (DATE_TO) เพื่อวิเคราะห์ย้อนหลัง
    # เช่น python ai_sentimentREST_API.py 2026-08-01 2026-08-25
    # หรือระบุใน .env / env variables: DATE_FROM=2026-08-01 DATE_TO=2026-08-25
    custom_date_from = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DATE_FROM", "")
    custom_date_to   = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("DATE_TO", "")

    is_retroactive_mode = bool(custom_date_from and custom_date_to)

    if is_retroactive_mode:
        print(f"📜 [RETROACTIVE MODE] เริ่มทำการวิเคราะห์ย้อนหลังสำหรับช่วง: {custom_date_from} ถึง {custom_date_to}")
        print("-" * 75)

        grand_total = 0
        round_num = 0

        try:
            while True:
                round_num += 1
                round_start = time.time()
                print(f"\n🔁 [รอบที่ {round_num}] กำลังดึงข้อมูล pending สำหรับช่วง {custom_date_from} ถึง {custom_date_to}...")
                
                total_posts = app_api.run(custom_date_from, custom_date_to, save_db=True)
                round_time = time.time() - round_start

                if not total_posts or total_posts == 0:
                    print(f"\n✅ ไม่มีข้อมูลค้างเหลือในคิวแล้ว!")
                    break
                
                grand_total += total_posts
                print(f"\n📊 [รอบที่ {round_num}] วิเคราะห์ได้ {total_posts} โพสต์ (ใช้เวลา {round_time:.1f} วินาที) | รวมสะสม: {grand_total} โพสต์")
                print(f"⏳ พัก 3 วินาทีก่อนดึงรอบถัดไป...")
                time.sleep(3)

        except KeyboardInterrupt:
            print(f"\n🛑 หยุดการทำงานตามคำสั่งผู้ใช้ (Ctrl+C)")

        print(f"\n{'=' * 75}")
        print(f"🎉 สรุปผล RETROACTIVE MODE: วิเคราะห์ย้อนหลังเสร็จสิ้น")
        print(f"   📅 ช่วงวันที่: {custom_date_from} ถึง {custom_date_to}")
        print(f"   📦 จำนวนโพสต์ทั้งหมด: {grand_total} โพสต์")
        print(f"   🔁 จำนวนรอบที่รัน: {round_num} รอบ")
        print(f"{'=' * 75}")
        sys.exit(0)

    # โหมดทำงานต่อเนื่อง (Loop Mode)
    while True:
        start_time = time.time()
        
        yesterday = str(datetime.now() - timedelta(days=1))[:10]
        now       = str(datetime.now())[:10]

        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 เริ่มการดึงข้อมูลและวิเคราะห์รอบใหม่...")
        print(f"📅 ช่วงเวลาที่วิเคราะห์: {yesterday} ถึง {now}")
        print("-" * 75)

        total_posts = 0
        try:
            total_posts = app_api.run(yesterday, now, save_db=True)
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
            print(f"\n🎉 สิ้นสุดการทำงานในรอบนี้! วิเคราะห์ไปทั้งหมด {total_posts} โพสต์ (ใช้เวลา {total_time:.2f} วินาที)")
            print(f"⏳ รอ {SLEEP_MINUTES} นาทีก่อนเริ่มรอบถัดไป... (กด Ctrl+C เพื่อหยุดโปรแกรม)")
            try:
                time.sleep(SLEEP_MINUTES * 60)
            except KeyboardInterrupt:
                print("\n🛑 หยุดการทำงานตามคำสั่งผู้ใช้ (Ctrl+C)")
                sys.exit(0)
