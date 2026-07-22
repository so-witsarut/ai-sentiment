# coding=utf-8
"""
Sentiment Analysis with Ollama (qwen3-8b-instruct)
Using Blueeye REST API
"""

import os
import re
import json
import time
import sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
BE_API_TOKEN = os.environ.get("BE_API_TOKEN", "10b6150ab6b7a8ef90904a32ef875f2b62789753109733d0194165d9ed3e854c")
BE_API_BASE_URL = os.environ.get("BE_API_BASE_URL", "https://api.blueeye.io/api/v1")

# ตั้งค่า stdout ให้รองรับการปริ้นภาษาไทยบน Windows (แก้ปัญหา UnicodeEncodeError)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def get_keyword_context(text, keyword, window=150, max_fallback_length=400):
    """
    ตัดข้อความให้เหลือแค่บริบทแวดล้อมของ Keyword
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

# =============================================================================
# Ollama Sentiment Analyzer (Local Ollama)
# =============================================================================
class OllamaSentimentAnalyzer:
    CONCURRENT_WORKERS = 3

    def __init__(self, model="qcwind/qwen3-8b-instruct-Q4-K-M:latest"):
        self.model = model
        self.base_url = "http://localhost:11434/api/generate"

        self.system_instruction = (
            "You are an expert Thai Social Media Brand Reputation Analyst. "
            "Given a social media post, determine the PUBLIC SENTIMENT toward a specific Target brand. "
            "Score: 100=positive brand impact, -100=negative brand impact, 0=neutral/irrelevant. "
            'Output ONLY minified JSON: {"reason":"<ภาษาไทย>","ai_sentiment":<-100|0|100>}'
        )

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
                response = requests.post(url, json=payload, timeout=45)
                if response.status_code == 200:
                    res_data = response.json()
                    result_text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                    parsed_res = self._parse_json_result(result_text)
                    if parsed_res is None:
                        print(f"  -> Gemini API Parsing Error [{model_name}]: {result_text}")
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
            response = requests.post("http://localhost:11434/api/generate", json=payload_generate, timeout=120)
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
            response = requests.post("http://localhost:11434/api/chat", json=payload_chat, timeout=120)
            if response.status_code == 200:
                result_text = response.json().get("message", {}).get("content", "{}")
                return self._parse_json_result(result_text)
        except Exception:
            pass
        return None

    def _parse_json_result(self, result_text):
        clean_text = re.sub(r'<think>.*?</think>', '', result_text, flags=re.DOTALL).strip()
        clean_text = clean_text.replace('```json', '').replace('```', '').strip()
        
        if not clean_text.startswith('{'):
            json_match = re.search(r'\{[^{}]*"ai_sentiment"[^{}]*\}', clean_text)
            if json_match:
                clean_text = json_match.group(0)
        
        try:
            parsed = json.loads(clean_text)
            ai_sentiment = parsed.get("ai_sentiment", 0)
            try:
                ai_sentiment = int(ai_sentiment)
            except:
                ai_sentiment = 0
            if ai_sentiment > 0: ai_sentiment = 100
            elif ai_sentiment < 0: ai_sentiment = -100
            parsed["ai_sentiment"] = ai_sentiment
            return parsed
        except json.JSONDecodeError:
            sentiment_match = re.search(r'[`"\']?ai_sentiment[`"\']?\s*[:=]\s*(-?100|0|100)', clean_text, re.IGNORECASE)
            reason_match = re.search(r'[`"\']?reason[`"\']?\s*[:=]\s*[`"\']?(.*?)[`"\']?(?:,|\n|\}|$)', clean_text, re.IGNORECASE)
            entity_match = re.search(r'[`"\']?entity_found[`"\']?\s*[:=]\s*(true|false)', clean_text, re.IGNORECASE)
            
            if sentiment_match:
                ai_sentiment = int(sentiment_match.group(1))
                reason = reason_match.group(1).strip() if reason_match else clean_text[:100].replace('\n', ' ')
                entity_found = True
                if entity_match:
                    entity_found = entity_match.group(1).lower() == 'true'
                return {"ai_sentiment": ai_sentiment, "reason": reason, "entity_found": entity_found}
        return None

    def _analyze_single_post(self, post):
        post_id = str(post.get("match_post_id", ""))
        keywords = post.get("keywords", [])
        target_entity = ", ".join(keywords) if keywords else "the Target Entity"
        feed_link = post.get("feed_link", "")

        # Pass 2: ขยาย Context Window เป็น 300 สำหรับ Gemini
        if "full_text" in post and keywords:
            content = get_keyword_context(post["full_text"], str(keywords[0]), window=300)
        else:
            content = post.get("content", "")

        # user_prompt = (
        #     f"Target Entity={actual_target}\n"
        #     f"Keyword={matched_keyword}\n"
        #     f"User={post_user}\n"
        #     f"Text={content}\n\n"
        #
        #     "INSTRUCTIONS (stop at first match):\n"
        #     "1. OWNED MEDIA: 'User' is Target's official page -> 0.\n"
        #     "2. NEUTRAL: Ads, promotions, sports results, or general news "
        #     "with no direct brand reputation impact -> 0.\n"
        #     "3. UNRELATED: Text is about a different entity, "
        #     "or Keyword is a location/idiom/generic word -> 0.\n"
        #     "4. NEGATIVE: Explicitly criticizes, boycotts, "
        #     "or reports scandal DIRECTLY against Target -> -100.\n"
        #     "5. POSITIVE: Explicitly praises or recommends Target DIRECTLY -> 100.\n"
        #     "6. DEFAULT: Otherwise -> 0.\n\n"
        #
        #     'JSON ONLY: {"reason":"ไทย 1 ประโยค","ai_sentiment":<int>,"is_ambiguous":<bool>}'
        # )

        user_prompt = (
            f"Target Entity={target_entity}\n"
            f"Source Link={feed_link}\n"
            f"Text={content}\n\n"
            "INSTRUCTIONS:\n"
            f"Analyze the sentiment of the text specifically towards the Target Entity ({target_entity}).\n"
            "1. UNRELATED REFERENCE: If the Target Entity name is used as a generic location, idiom, animal breed, or is completely unrelated to the intended subject -> 0.\n"
            "2. OWNED MEDIA: If Source Link or Text indicates it's posted by the Target Entity's official page or PR, return 0.\n"
            "3. NEGATIVE: Explicitly criticizes, complains, expresses anger, or reports a bad experience against the Target Entity -> -100.\n"
            "4. POSITIVE: Explicitly praises, recommends, expresses happiness, or shows support for the Target Entity -> 100.\n"
            "5. NEUTRAL: General news, ads, promotions, sports results, statements of facts, or ambiguous tone -> 0.\n\n"
            "CRITICAL: For the 'reason' field, you MUST explain WHICH rule you applied and WHY (e.g., 'เนื้อหาเป็นการรายงานผลกีฬา', 'เป็นโพสต์จาก Official Page'). Do NOT include phrases like 'จึงจัดเป็น 0' or 'จึงได้คะแนน 100'.\n"
            'JSON ONLY: {"reason":"เหตุผลที่อ้างอิงกฎข้างต้น (ภาษาไทยสั้นๆ)","ai_sentiment":<int>,"is_ambiguous":<bool>}'
        )

        def _call_ollama(is_deep_think=False):
            current_payload = {
                "model": self.model,
                "system": self.system_instruction,
                "prompt": user_prompt,
                "stream": False,
                "format": "json",
                "think": is_deep_think,
                "keep_alive": -1,
                "options": {
                    "temperature": 0.0,
                    "top_p":       0.1,
                    "seed":        42,
                    "num_predict": 384 if is_deep_think else 128,
                    "num_ctx":     768 if is_deep_think else 512,
                    "num_batch":   256,
                    "flash_attn":  True,
                }
            }

            try:
                response = requests.post(self.base_url, json=current_payload, timeout=300 if is_deep_think else 120)

                if response.status_code == 200:
                    result_text = response.json().get("response", "{}")

                    clean_text = re.sub(r'<think>.*?</think>', '', result_text, flags=re.DOTALL).strip()
                    clean_text = clean_text.replace('```json', '').replace('```', '').strip()

                    if not clean_text.startswith('{'):
                        json_match = re.search(r'\{[^{}]*"ai_sentiment"[^{}]*\}', clean_text)
                        if json_match:
                            clean_text = json_match.group(0)

                    try:
                        parsed = json.loads(clean_text)
                        ai_sentiment = parsed.get("ai_sentiment", 0)

                        if isinstance(ai_sentiment, str):
                            s = ai_sentiment.upper()
                            if s == "POSITIVE":  ai_sentiment = 100
                            elif s == "NEGATIVE": ai_sentiment = -100
                            else:                ai_sentiment = 0
                        else:
                            if ai_sentiment > 0:
                                ai_sentiment = 100
                            elif ai_sentiment < 0:
                                ai_sentiment = -100
                            else:
                                ai_sentiment = 0

                        return {
                            "post_id":      post_id,
                            "ai_sentiment": int(ai_sentiment),
                            "confidence":   0,
                            "reason":       parsed.get("reason", "")
                        }
                    except json.JSONDecodeError as e:
                        print(f"  -> JSONDecodeError [{post_id}]: {e} | Raw Clean Text: {clean_text[:150]}")
                else:
                    print(f"  -> HTTP Error {response.status_code}: {response.text}")
            except Exception as e:
                print(f"  -> Request Error [{post_id}]: {e}")

            return None

        first_pass_result = _call_ollama(is_deep_think=False)

        if first_pass_result and first_pass_result["ai_sentiment"] != 0:
            first_sentiment = first_pass_result["ai_sentiment"]
            first_reason = first_pass_result.get("reason", "")
            print(f"\n  🔍 [Pass 2 Triggered] Post {post_id} ได้ค่า {first_sentiment} -> กำลังยืนยันความถูกต้อง...")
            
            validation_system_instruction = (
                f"You are a strict Brand Mention Validator for '{target_entity}'. "
                f"Your #1 priority is to check if '{target_entity}' is EXPLICITLY mentioned in the text. "
                "If it is NOT mentioned, you MUST return ai_sentiment=0 regardless of the text's tone. "
                "Do NOT assume or infer that the text is about the Target Entity."
            )

            # validation_user_prompt = (
            #     f"Target Entity={actual_target}\n"
            #     f"Keyword={matched_keyword}\n"
            #     f"User={post_user}\n"
            #     f"Text={content}\n\n"
            #     f"First-Pass Sentiment={first_sentiment}\n"
            #     f"First-Pass Reason={first_reason}\n\n"
            #     "TASK (Follow these steps IN ORDER):\n"
            #     "Step 1: ENTITY CHECK - Scan the Text and determine if the Target Entity or any of its known brands/products are explicitly mentioned. Set entity_found=true or entity_found=false.\n"
            #     "Step 2: IF entity_found=false -> STOP. Set ai_sentiment=0. The text is UNRELATED. Do NOT proceed further.\n"
            #     "Step 3: IF entity_found=true -> Check these rules:\n"
            #     "   - OWNED MEDIA: Official page posts, PR, or promotional content by the Target Entity -> 0\n"
            #     "   - NEUTRAL: General ads, job hirings, sports news without direct positive/negative impact -> 0\n"
            #     "   - UNRELATED REFERENCE: Name used as location, idiom, person name, or animal breed -> 0\n"
            #     "   - NEGATIVE: Explicitly criticizes/boycotts/complains DIRECTLY against the Target Entity -> -100\n"
            #     "   - POSITIVE: Explicitly praises/recommends the Target Entity DIRECTLY -> 100\n\n"
            #     'JSON ONLY: {"entity_found":true/false,"reason":"[\u0e22\u0e37\u0e19\u0e22\u0e31\u0e19/\u0e41\u0e01\u0e49\u0e44\u0e02] \u0e2a\u0e23\u0e38\u0e1b\u0e2a\u0e31\u0e49\u0e19\u0e46 \u0e20\u0e32\u0e29\u0e32\u0e44\u0e17\u0e22","ai_sentiment":<-100|0|100>}'
            # )

            validation_user_prompt = (
                f"Target Entity={target_entity}\n"
                f"Source Link={feed_link}\n"
                f"Text={content}\n\n"
                f"First-Pass Sentiment={first_sentiment}\n"
                f"First-Pass Reason={first_reason}\n\n"
                "TASK:\n"
                f"Validate the sentiment specifically towards '{target_entity}'.\n"
                "   - UNRELATED REFERENCE: If the name is used as a generic location, idiom, or animal breed -> 0\n"
                "   - OWNED MEDIA: If Source Link indicates the post is from the Target Entity's official page, return 0.\n"
                "   - NEGATIVE: Explicitly criticizes, complains, expresses anger, or reports a bad experience -> -100\n"
                "   - POSITIVE: Explicitly praises, recommends, expresses happiness, or shows support -> 100\n"
                "   - NEUTRAL: General news, ads, promotions, sports results, statements of facts, or ambiguous tone -> 0\n\n"
                "CRITICAL: You MUST output ONLY a valid JSON object. Do NOT include markdown formatting, bullet points, or thinking processes.\n"
                "CRITICAL: For the 'reason' field, explain WHICH rule you applied and WHY (e.g., 'เนื้อหาเป็นการรายงานผลกีฬา'). Do NOT include phrases like 'จึงจัดเป็น 0' or 'จึงได้ 100'.\n"
                '{"entity_found": <boolean>, "reason": "<อธิบายเหตุผลอ้างอิงตามกฎ เป็นภาษาไทยสั้นๆ>", "ai_sentiment": <-100|0|100>}'
            )

            validation_models = [
                "api:gemma-4-31b-it", 
                "gemma4:31b-cloud", 
                "api:gemma-4-26b-a4b-it", 
                "api:gemini-3.1-flash-lite", 
                "api:gemini-3.5-flash-lite"
            ]
            second_pass_result = None
            
            for val_model in validation_models:
                if val_model.startswith("api:"):
                    actual_api_model = val_model.replace("api:", "", 1)
                    res = self._call_gemini_api(actual_api_model, validation_system_instruction, validation_user_prompt, max_retries=1)
                else:
                    res = self._call_ollama_generic(val_model, validation_system_instruction, validation_user_prompt)

                if res and "ai_sentiment" in res:
                    second_pass_result = res
                    used_model = val_model
                    break
            
            if second_pass_result:
                entity_found = second_pass_result.get("entity_found", True)
                if isinstance(entity_found, str):
                    entity_found = entity_found.lower() == "true"
                    
                if not entity_found:
                    second_pass_result["ai_sentiment"] = 0
                
                second_pass_result["post_id"] = post_id
                second_pass_result["reason"] = f"[Deep Checked: {used_model}] {second_pass_result.get('reason', '')} (รอบแรก: {first_reason})"
                return second_pass_result

        return first_pass_result

    def analyze_post_sentiments(self, json_posts):
        posts = json.loads(json_posts)
        results = []

        with ThreadPoolExecutor(max_workers=self.CONCURRENT_WORKERS) as executor:
            future_to_post = {
                executor.submit(self._analyze_single_post, post): post
                for post in posts
            }

            for future in as_completed(future_to_post):
                result = future.result()
                if result is not None:
                    results.append(result)

        return {"data": results, "token_usage": {"input": 0, "output": 0, "total": 0}}


# =============================================================================
# Sentiment API Manager
# =============================================================================
class SentimentAPI:
    def __init__(self):
        self.ollama = OllamaSentimentAnalyzer(model="qcwind/qwen3-8b-instruct-Q4-K-M:latest")
        self.headers = {
            'X-Internal-Token': BE_API_TOKEN,
            'Content-Type': 'application/json'
        }

    def fetch_pending(self, date_from, date_to):
        url = f"{BE_API_BASE_URL}/internal/sentiment/pending?date_from={date_from}&date_to={date_to}"
        print(f"\n🎯 กำลังดึงข้อมูลผ่าน API...")
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
                print(f"  ✅ บันทึกข้อมูลสำเร็จ ({len(results)} โพสต์)")
            else:
                print(f"  ❌ API Update Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"  ❌ Exception in bulk_update: {e}")

    def run(self, date_from, date_to, save_db=False):
        pending_posts = self.fetch_pending(date_from, date_to)
        
        if not pending_posts:
            print("⏩ ไม่มีข้อมูลใหม่ให้วิเคราะห์ (0 โพสต์)")
            return

        total = len(pending_posts)
        print(f"📦 พบข้อความที่ต้องวิเคราะห์ทั้งหมด: {total} โพสต์")

        BATCH_SIZE = 5
        for batch_start in range(0, total, BATCH_SIZE):
            batch = pending_posts[batch_start:batch_start + BATCH_SIZE]
            batch_end = min(batch_start + BATCH_SIZE, total)
            print(f"\n🔄 กำลังประมวลผล Batch {batch_start + 1}-{batch_end} จากทั้งหมด {total} โพสต์...")

            # Prepare data for AI
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
                modified_post["full_text"] = text  # เก็บเต็มไว้ให้ Pass 2
                posts_for_ai.append(modified_post)

            json_str = json.dumps(posts_for_ai, ensure_ascii=False)
            
            # Send to local Ollama
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

            # Prepare payload for POST API
            api_results = []
            for idx, post_for_ai in enumerate(posts_for_ai, 1):
                match_post_id = str(post_for_ai.get("match_post_id", ""))
                ai_content = post_for_ai.get("content", "").replace("\n", " ")
                
                # ตัดข้อความให้สั้นลงสำหรับการปรินต์แสดงผล (ถ้ายาวไป)
                if len(ai_content) > 120:
                    ai_content = ai_content[:120] + "..."

                if match_post_id in ollama_map:
                    raw_val = ollama_map[match_post_id]["val"]
                    ai_reason = ollama_map[match_post_id]["reason"]

                    # Map int to string
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
                        
            # Update batch back to API
            if save_db:
                self.bulk_update(api_results)
            else:
                print(api_results)
                print(f"  🚫 [MOCKUP] ข้ามการบันทึกลง API ({len(api_results)} โพสต์)")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" 🤖 Ollama SENTIMENT ANALYSIS SYSTEM (API MODE - RUN FOREVER)")
    print("=" * 70)

    app = SentimentAPI()
    
    # อ่านค่าจาก .env หรือตั้งค่าเริ่มต้นที่ 10 นาที
    SLEEP_MINUTES = int(os.environ.get("RUN_INTERVAL_MINUTES", 10))

    while True:
        start_time = time.time()
        
        # อัปเดตวันที่ให้เป็นปัจจุบันเสมอในแต่ละรอบ
        yesterday = str(datetime.now() - timedelta(days=1))[:10]
        now       = str(datetime.now())[:10]

        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 เริ่มการดึงข้อมูลรอบใหม่...")
        print(f"📅 ช่วงเวลา: {yesterday} ถึง {now}")
        print("-" * 70)

        try:
            app.run(yesterday, now, save_db=False)
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดระหว่างรัน: {e}")

        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n🎉 สิ้นสุดการทำงานรอบนี้! ใช้เวลาไปทั้งสิ้น: {total_time:.2f} วินาที")
        
        print(f"⏳ รอ {SLEEP_MINUTES} นาทีก่อนเริ่มรอบถัดไป... (กด Ctrl+C เพื่อหยุดโปรแกรม)")
        try:
            time.sleep(SLEEP_MINUTES * 60)
        except KeyboardInterrupt:
            print("\n🛑 หยุดการทำงานตามคำสั่งผู้ใช้ (Ctrl+C)")
            sys.exit(0)
