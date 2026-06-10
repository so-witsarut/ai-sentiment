# coding=utf-8
"""
Sentiment Analysis with Ollama (qwen3-8b-instruct)
Flow หลักอิงจาก sentiment_analysis_ai_be_srv1_01.py
- ดึง msg_id ที่ sentiment_status = '0' จาก MySQL
- ดึง content จาก MongoDB
- ส่งให้ Ollama วิเคราะห์ sentiment
- อัพเดทผลลัพธ์กลับลง MySQL (own_match, own_match_daily, own_match_3months)
"""

import os
import re
import json
import time
import sys
import requests
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import pymysql
from pymongo import MongoClient
from datetime import datetime, timedelta

# pyrefly: ignore [missing-import]
from sshtunnel import SSHTunnelForwarder

sys.path.append(r"ai-sentiment")
# pyrefly: ignore [missing-import]
import connection

CONN=connection.DatabaseConnection()

# ตั้งค่า stdout ให้รองรับการปริ้นภาษาไทยบน Windows (แก้ปัญหา UnicodeEncodeError)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def get_keyword_context(text, keyword, window=150, max_fallback_length=400):
    """
    ตัดข้อความให้เหลือแค่บริบทแวดล้อมของ Keyword
    """
    if not text:
        return ""
        
    # ถ้าไม่มี Keyword ให้ตัดตามความยาวสูงสุดเลย
    if not keyword or keyword not in text:
        return text[:max_fallback_length] + ("..." if len(text) > max_fallback_length else "")

    # หาตำแหน่งของ Keyword ในข้อความ
    start_idx = text.find(keyword)
    
    # คำนวณจุดตัดหน้า-หลัง
    left_bound = max(0, start_idx - window)
    right_bound = min(len(text), start_idx + len(keyword) + window)
    
    # หั่นข้อความและเติม ... ให้รู้ว่าถูกตัดมา
    sliced_text = text[left_bound:right_bound]
    
    if left_bound > 0:
        sliced_text = "..." + sliced_text
    if right_bound < len(text):
        sliced_text = sliced_text + "..."
        
    return sliced_text

# =============================================================================
# Ollama Sentiment Analyzer (Local Ollama)
# =============================================================================
# 🖥️ GPU Profile: NVIDIA GTX 1660 (6GB GDDR5, 1408 CUDA cores, 120W TDP)
#   ⚠️ GTX 1660 มี CUDA cores น้อยกว่า RTX 3050 Ti (1408 vs 2560)
#   → Strategy: ลด workload ต่อ request ให้เบาที่สุด แต่คง parallelism ไว้
#   → Workers 3 ตัว = จุดสมดุลระหว่าง parallelism กับ compute pressure
#   - Model: qwen3-8b Q4_K_M (~4.5GB) → เหลือ VRAM ~1.5GB
#   - KV cache per seq (1024 ctx) ≈ 100-200MB
#   - 3 workers × 200MB = 600MB < 1.5GB ✅
# =============================================================================
class OllamaSentimentAnalyzer:
    # ✅ GTX 1660: 3 workers = สมดุลระหว่าง parallelism กับ CUDA cores
    # ⚠️ อย่าลดต่ำกว่า 3 → จะช้ามาก (ทดสอบ 2 workers = 168 วินาที!)
    CONCURRENT_WORKERS = 3

    def __init__(self, model="qcwind/qwen3-8b-instruct-Q4-K-M:latest"):
        
        self.model = model
        self.base_url = "http://localhost:11434/api/generate"

        # === System Prompt (สร้างครั้งเดียว ใช้ซ้ำทุก request) ===
        # self.system_instruction = (
        #     "You are a strict Thai Sentiment Analyzer. "
        #     # "/no_think "                              # ✅ magic token ของ qwen3 (ปิด thinking mode)
        #     'Output ONLY a minified JSON object: {"reason": "<เหตุผลสั้นๆ ภาษาไทย>", "ai_sentiment": <int>}. '
        #     "Do not explain or add markdown formatting."
        # )
        # self.system_instruction = (
        #     'Return only JSON: {"reason":"<thai>","ai_sentiment":<int>}'
        # )
        self.system_instruction = (
            "You are an expert Thai Social Media Brand Reputation Analyst. "
            "Given a social media post, determine the PUBLIC SENTIMENT toward a specific Target brand. "
            "Score: 100=positive brand impact, -100=negative brand impact, 0=neutral/irrelevant. "
            'Output ONLY minified JSON: {"reason":"<ภาษาไทย>","ai_sentiment":<-100|0|100>}'
        )

    def _analyze_single_post(self, post, company_name):
        """
        วิเคราะห์ sentiment ของโพสต์เดียว (ใช้ใน ThreadPoolExecutor)
        Returns: dict with post_id, ai_sentiment, confidence หรือ None ถ้า error
        """
        post_id   = post["post_id"]
        content   = post["content"]
        post_user = post["post_user"]
        matched_keyword = post.get("keyword_name", company_name)

        # ใช้ actual_target ที่คำนวณไว้แล้วจาก analysis()
        # ถ้าเป็น COMPETITOR → actual_target = keyword_name (ชื่อแบรนด์จริง)
        # ถ้าเป็น OWN → actual_target = company_name
        actual_target = post.get("actual_target", company_name)

        user_prompt = (
            f"Target Entity={actual_target}\n"
            f"Keyword={matched_keyword}\n"
            f"User={post_user}\n"
            f"Text={content}\n\n"

            "INSTRUCTIONS (stop at first match):\n"
            "1. OWNED MEDIA: 'User' is Target's official page -> 0.\n"
            "2. NEUTRAL: Ads, promotions, sports results, or general news "
            "with no direct brand reputation impact -> 0.\n"
            "3. UNRELATED: Text is about a different entity, "
            "or Keyword is a location/idiom/generic word -> 0.\n"
            "4. NEGATIVE: Explicitly criticizes, boycotts, "
            "or reports scandal DIRECTLY against Target -> -100.\n"
            "5. POSITIVE: Explicitly praises or recommends Target DIRECTLY -> 100.\n"
            "6. DEFAULT: Otherwise -> 0.\n\n"

            'JSON ONLY: {"reason":"ไทย 1 ประโยค","ai_sentiment":<int>,"is_ambiguous":<bool>}'
        )

        # user_prompt = (
        #     f"Target={actual_target}\n"
        #     f"Keyword={matched_keyword}\n"
        #     f"User={post_user}\n"
        #     f"Text={content}\n\n"

        #     "INSTRUCTIONS:\n"
        #     "1. OWNED MEDIA: If 'User' is Target's official page -> 0.\n"
        #     "2. FALSE POSITIVE: If Keyword refers to UNRELATED location (e.g. จังหวัดสิงห์บุรี) or idiom (e.g. เสือเป็นสิงห์) -> 0. (Note: 'สิงห์ปาร์ค', 'สิงห์ เชียงราย' ARE valid Target entities).\n"
        #     "3. NEUTRAL NEWS: Ads, sports, or general news (like crimes/drug busts) NOT explicitly damaging the Target -> 0.\n"
        #     "4. NEGATIVE: Explicitly criticizes, boycotts (e.g. 'บอกลา'), or reports a real scandal DIRECTLY against Target -> -100.\n"
        #     "5. POSITIVE: Explicitly praises or supports Target -> 100.\n"
        #     "6. DEFAULT: Otherwise -> 0.\n\n"

        #     'JSON ONLY:\n{"reason":"เหตุผล 1 ประโยคภาษาไทย","ai_sentiment":<int>}'
        # )


        def _call_ollama(is_deep_think=False):
            current_payload = {
                "model": self.model,
                "system": self.system_instruction,
                "prompt": user_prompt,
                "stream": False,
                "format": "json",
                "think": is_deep_think,
                "keep_alive": -1,                    # ✅ โมเดลค้างใน GPU ไม่ต้อง reload
                "options": {
                    "temperature": 0.0,
                    "top_p":       0.1,
                    "seed":        42,
                    "num_predict": 384 if is_deep_think else 128,
                    "num_ctx":     768 if is_deep_think else 512,   # ลดจาก 1536/1024
                    "num_batch":   256,
                    "flash_attn":  True,  # ⚡ เปิดใช้ Flash Attention เพื่อประหยัด VRAM ของ 1660
                    "num_gpu":     99,  # ⚠️ คอมเมนต์ออก เพื่อป้องกันปัญหา memory layout (OOM) 
                }
            }

            try:
                # ให้เวลารอนานขึ้นถ้ารอบ deep think เพราะ GTX 1660 ประมวลผล concurrent นาน
                response = requests.post(self.base_url, json=current_payload, timeout=300 if is_deep_think else 120)

                if response.status_code == 200:
                    result_text = response.json().get("response", "{}")

                    # --- ทำความสะอาดผลลัพธ์ ---
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
                            # Normalize: บวก→100, ลบ→-100, ศูนย์→0
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

        # ==========================================
        # 🚀 TWO-PASS FILTERING (Cascade Architecture)
        # ==========================================
        # Pass 1: วิเคราะห์แบบเร็ว (Fast Pass)
        first_pass_result = _call_ollama(is_deep_think=False)

        # ถ้าผลลัพธ์รอบแรกออกมาเป็น Positive (100) หรือ Negative (-100)
        # ให้ส่งทำ Pass 2 เพื่อยืนยันความถูกต้องด้วย Deep Reasoning
        if first_pass_result and first_pass_result["ai_sentiment"] != 0:
            print(f"\n  🔍 [Pass 2 Triggered] Post {post_id} ได้ค่า {first_pass_result['ai_sentiment']} -> กำลังวิเคราะห์เชิงลึก (Think=True)...")
            
            second_pass_result = _call_ollama(is_deep_think=True)
            
            if second_pass_result:
                # เติม tag ไว้ใน reason ให้รู้ว่าผ่านการ Deep Checked แล้ว
                second_pass_result["reason"] = "[Deep Checked] " + second_pass_result["reason"]
                return second_pass_result

        # ถ้าเป็น Neutral (0) หรือ Error ให้ส่งกลับเลย
        return first_pass_result

    def analyze_post_sentiments(self, json_posts, company_name=""):
        """
        ✅ ส่งหลายโพสต์พร้อมกัน (concurrent) เพื่อใช้ Ollama continuous batching
        GTX 1660: ใช้ 3 workers — Ollama จะ batch requests บน GPU ให้เอง
        Returns: {"data": [...], "token_usage": {...}}
        """
        posts = json.loads(json_posts)
        results = []

        # ✅ ส่ง requests แบบ concurrent ด้วย ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.CONCURRENT_WORKERS) as executor:
            future_to_post = {
                executor.submit(self._analyze_single_post, post, company_name): post
                for post in posts
            }

            for future in as_completed(future_to_post):
                result = future.result()
                if result is not None:
                    results.append(result)

        return {"data": results, "token_usage": {"input": 0, "output": 0, "total": 0}}


# =============================================================================
# Sentiment Analysis
# =============================================================================
class sentiment:
    def __init__(self, config):
        
        self.config = config
        self.ollama = OllamaSentimentAnalyzer(model="qcwind/qwen3-8b-instruct-Q4-K-M:latest")

    def get_content(self, list_id_with_info, collection):
        list_content = []

        company_map   = {msg_id: comp         for (msg_id, comp, proj, post_user, kw_name) in list_id_with_info}
        project_map   = {msg_id: proj         for (msg_id, comp, proj, post_user, kw_name) in list_id_with_info}
        post_user_map = {msg_id: post_user    for (msg_id, comp, proj, post_user, kw_name) in list_id_with_info}
        keyword_map   = {msg_id: kw_name      for (msg_id, comp, proj, post_user, kw_name) in list_id_with_info}
        list_id = [msg_id for (msg_id, comp, proj, post_user, kw_name) in list_id_with_info]

        if not list_id:
            return list_content

        DB_CONNECTION = CONN.get_mongo_client()
        DB = DB_CONNECTION[self.config.get("mongo_db", "blue_eye")]
        DB_COLLECTION = DB[collection]

        result = DB_COLLECTION.find({"_id": {"$in": list_id}})

        if collection == "Feed":
            columnName = "feedcontent"
        else:
            columnName = "commentcontent"

        for e in result:
            feedcontent = e.get(columnName, "")
            msg_id = e["_id"]
            comp_name = company_map.get(msg_id, "")
            proj_name = project_map.get(msg_id, "")
            post_user = post_user_map.get(msg_id, "")
            kw_name   = keyword_map.get(msg_id, "")
            # fallback: ถ้า post_user ว่าง ให้ใช้ prefix ก่อน '_' จาก msg_id
            if not post_user:
                post_user = str(msg_id).split("_")[0]
            list_content.append((msg_id, feedcontent, comp_name, proj_name, post_user, kw_name))

        # ไม่ปิด connection — เป็น shared connection ที่ CONN จัดการให้

        return list_content

    def analysis(self, list_content, host, server=1, table_prefix="own_match", save_db=True):
        """
        วิเคราะห์ sentiment แล้วอัพเดทลง MySQL

        Args:
            list_content (list): [(msg_id, content, company_name, project_name, post_user, kw_name), ...]
            host (str): MySQL host
            server (int): MySQL Server ID (1 or 2)
            table_prefix (str): 'own_match' หรือ 'competitor_match'
            save_db (bool): บันทึกลงฐานข้อมูลหรือไม่ (ถ้า False จะแค่จำลองการทำงาน)
        """
        tunnel, DB_CONNECTION = CONN.get_mysql_connection(server=server, host=host, database=self.config["mysql_db"])

        BATCH_SIZE = 5
        total = len(list_content)
        print(f"\n📦 พบข้อความที่ต้องวิเคราะห์ทั้งหมด: {total} โพสต์")

        for batch_start in range(0, total, BATCH_SIZE):
            batch = list_content[batch_start:batch_start + BATCH_SIZE]
            batch_end = min(batch_start + BATCH_SIZE, total)
            print(f"\n🔄 กำลังประมวลผล Batch {batch_start + 1}-{batch_end} จากทั้งหมด {total} โพสต์...")

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
                    # กรณีที่มีหลายคำคั่นด้วยลูกน้ำ (เช่น ทราย, สุนิษฐ์, พาย) อาจจะเลือกคำแรกมาใช้สแกน
                    first_keyword = kw_name.split(",")[0].strip() if kw_name else ""
                    clean_short_content = get_keyword_context(text, first_keyword, window=150)

                    # กำหนด actual_target ตามประเภท:
                    # - COMPETITOR: ใช้ keyword เป็น Target
                    # - OWN: ใช้ company_name เป็น Target (ชื่อแบรนด์จริง)
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
                        "content": clean_short_content
                    })

            if not posts_for_ai:
                print("  -> All content in batch is empty, skipping.")
                continue

            json_str = json.dumps(posts_for_ai, ensure_ascii=False)

            target_label = f"{'COMPETITOR' if is_competitor else 'OWN'} | Company: {batch_company_name} | Proj: {batch_project_name}"
            print(f"  🚀 ส่ง {len(posts_for_ai)} โพสต์ไปยัง Ollama ({target_label})")

            ollama_response = self.ollama.analyze_post_sentiments(json_str, batch_company_name)

            if "error" in ollama_response:
                print(f"  ❌ ข้อผิดพลาดจาก Ollama: {ollama_response.get('message', ollama_response['error'])}")

            ollama_results = ollama_response.get("data", [])
            ollama_token_usage = ollama_response.get("token_usage", {})

            ollama_map = {}
            if isinstance(ollama_results, list):
                for res in ollama_results:
                    if "post_id" in res and "ai_sentiment" in res:
                        ollama_map[res["post_id"]] = {
                            "ai_sentiment": res["ai_sentiment"],
                            "confidence": res.get("confidence", 0),
                            "reason": res.get("reason", "")
                        }

            print(f"\n  📊 สรุปผลลัพธ์จาก Ollama (สำเร็จ {len(ollama_map)}/{len(batch)} โพสต์)")
            print(f"  {'-'*100}")

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

                # หา actual_target และ content ที่ถูกตัดแล้ว (ส่งให้ AI) ที่ตรงกับโพสต์นี้
                actual_target = next((p["actual_target"] for p in posts_for_ai if p["post_id"] == str_id), company_name)
                ai_content = next((p["content"] for p in posts_for_ai if p["post_id"] == str_id), str(content))

                # เตรียมข้อความสำหรับแสดงผล
                original_preview = str(content).replace("\n", " ")
                if len(original_preview) > 150:
                    original_preview = original_preview[:150] + "..."
                
                ai_sliced = ai_content.replace("\n", " ")

                print(f"  [{idx:02d}] 🆔 {str_id[:15]:<15} | {icon:<11} | User: {str(post_user)[:12]:<12} | Client: {company_name[:10]:<10} | Proj: {project_name[:10]:<10} | Target: {actual_target[:15]:<15} | KW: {kw_name}")
                if ai_reason:
                    print(f"       💡 Reason: {ai_reason}")
                print(f"       📄 Full (Preview): {original_preview}")
                print(f"       ✂️ Sliced (to AI): {ai_sliced}")
                print(f"  {'-'*100}")

            # ===== อัพเดท DB =====
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
                print(f"  💾 บันทึกลง DB เรียบร้อย ({len([x for x in batch if str(x[0]) in ollama_map])} โพสต์)")
            else:
                print(f"  🚫 [MOCKUP] ข้ามการบันทึกลง DB ({len([x for x in batch if str(x[0]) in ollama_map])} โพสต์)")

            # ไม่ต้องมีการดีเลย์ระหว่าง batch สำหรับ Local Ollama

        DB_CONNECTION.close()
        if 'tunnel' in locals() and tunnel:
            tunnel.stop()
        print("\n✅ วิเคราะห์และบันทึกข้อมูลทั้งหมดเสร็จสิ้นแล้ว")

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    start_time = time.time()


    config = {
        # MySQL 
        "mysql_host_1":   os.environ.get("MYSQL_HOST_1",   "10.130.84.170"),
        "mysql_host_2":   os.environ.get("MYSQL_HOST_2",   "10.130.69.57"),
        "mysql_port":     int(os.environ.get("MYSQL_PORT", 3306)),
        "mysql_user":     os.environ.get("MYSQL_USER",     "blueeyeremote"),
        "mysql_password": os.environ.get("MYSQL_PASSWORD", "BEremotemysql3075"),
        "mysql_db":       os.environ.get("MYSQL_DB",       "blue_eye"),
        "server":       os.environ.get("server",       1),

        # SSH Tunnel
        "ssh_host":       os.environ.get("SSH_HOST"),
        "ssh_port":       int(os.environ.get("SSH_PORT", 22)),
        "ssh_user":       os.environ.get("SSH_USER"),
        "ssh_password":   os.environ.get("SSH_PASSWORD"),

        # MongoDB
        "mongo_host":     os.environ.get("MONGO_HOST",     "10.130.72.139"),
        "mongo_port":     int(os.environ.get("MONGO_PORT", 34596)),
        "mongo_user":     os.environ.get("MONGO_USER",     "blueeyeharvest"),
        "mongo_password": os.environ.get("MONGO_PASSWORD", "BEharvest3075"),
        "mongo_db":       os.environ.get("MONGO_DB",       "blue_eye"),
    }

    yesterday = str(datetime.now() - timedelta(days=1))[:10]
    now       = str(datetime.now())[:10]

    targets = [
        {
            "name": "OWN MATCH",
            "table_prefix": "own_match",
            "sql_feed": (
                f"SELECT omd.msg_id, "
                f"IFNULL(ck.company_keyword_name, '') as project_name, "
                f"IFNULL(omd.post_user, '') as post_user, "
                f"IFNULL(GROUP_CONCAT(DISTINCT k.keyword_name SEPARATOR ', '), '') as keyword_name "
                f"FROM own_match_daily omd "
                f"LEFT JOIN company_keyword ck ON omd.company_keyword_id = ck.company_keyword_id "
                f"LEFT JOIN own_key_match okm ON okm.own_match_id = omd.own_match_id "
                f"LEFT JOIN keyword k ON okm.keyword_id = k.keyword_id "
                f"WHERE date(omd.msg_time) BETWEEN '{yesterday}' AND '{now}' "
                f"AND omd.sentiment_status = '0' AND omd.match_type = 'Feed' "
                f"GROUP BY omd.msg_id, project_name, post_user "
                f"ORDER BY omd.msg_time ASC "
            ),
            "sql_comment": (
                f"SELECT omd.msg_id, "
                f"IFNULL(ck.company_keyword_name, '') as project_name, "
                f"IFNULL(omd.post_user, '') as post_user, "
                f"IFNULL(GROUP_CONCAT(DISTINCT k.keyword_name SEPARATOR ', '), '') as keyword_name "
                f"FROM own_match_daily omd "
                f"LEFT JOIN company_keyword ck ON omd.company_keyword_id = ck.company_keyword_id "
                f"LEFT JOIN own_key_match okm ON okm.own_match_id = omd.own_match_id "
                f"LEFT JOIN keyword k ON okm.keyword_id = k.keyword_id "
                f"WHERE date(omd.msg_time) BETWEEN '{yesterday}' AND '{now}' "
                f"AND omd.sentiment_status = '0' AND omd.match_type = 'Comment' "
                f"GROUP BY omd.msg_id, project_name, post_user "
                f"ORDER BY omd.msg_time ASC "
            )
        },
        {
            "name": "COMPETITOR MATCH",
            "table_prefix": "competitor_match",
            "sql_feed": (
                f"SELECT cmd.msg_id, "
                f"IFNULL(ck.company_keyword_name, '') as project_name, "
                f"IFNULL(cmd.post_user, '') as post_user, "
                f"IFNULL(GROUP_CONCAT(DISTINCT k.keyword_name SEPARATOR ', '), '') as keyword_name "
                f"FROM competitor_match_daily cmd "
                f"LEFT JOIN company_keyword ck ON cmd.company_keyword_id = ck.company_keyword_id "
                f"LEFT JOIN competitor_key_match ckm ON ckm.competitor_match_id = cmd.competitor_match_id "
                f"LEFT JOIN keyword k ON ckm.keyword_id = k.keyword_id "
                f"WHERE date(cmd.msg_time) BETWEEN '{yesterday}' AND '{now}' "
                f"AND cmd.sentiment_status = '0' AND cmd.match_type = 'Feed' "
                f"GROUP BY cmd.msg_id, project_name, post_user "
                f"ORDER BY cmd.msg_time ASC "
            ),
            "sql_comment": (
                f"SELECT cmd.msg_id, "
                f"IFNULL(ck.company_keyword_name, '') as project_name, "
                f"IFNULL(cmd.post_user, '') as post_user, "
                f"IFNULL(GROUP_CONCAT(DISTINCT k.keyword_name SEPARATOR ', '), '') as keyword_name "
                f"FROM competitor_match_daily cmd "
                f"LEFT JOIN company_keyword ck ON cmd.company_keyword_id = ck.company_keyword_id "
                f"LEFT JOIN competitor_key_match ckm ON ckm.competitor_match_id = cmd.competitor_match_id "
                f"LEFT JOIN keyword k ON ckm.keyword_id = k.keyword_id "
                f"WHERE date(cmd.msg_time) BETWEEN '{yesterday}' AND '{now}' "
                f"AND cmd.sentiment_status = '0' AND cmd.match_type = 'Comment' "
                f"GROUP BY cmd.msg_id, project_name, post_user "
                f"ORDER BY cmd.msg_time ASC "
            )
        }
    ]

    print("\n" + "=" * 70)
    print(" 🤖 Ollama (qwen3-8b-instruct) SENTIMENT ANALYSIS SYSTEM")
    print(f" 📅 ช่วงเวลา: {yesterday} ถึง {now}")
    print("=" * 70)

    sa = sentiment(config)

    for server_id in [1 , 2]: #, 2]: ทดสอบเฉพาะ server 1 ก่อน
        current_host = config[f"mysql_host_{server_id}"]
        
        print(f"\n{'=' * 70}")
        print(f" 🖥️  เริ่มทำงานกับ MYSQL SERVER {server_id} ({current_host})")
        print(f"{'=' * 70}")

        for target in targets:
            print(f"\n🎯 กำลังดึงข้อมูล: {target['name']} ...")
            print(target["sql_feed"])
            # --- ดึง Feed msg_ids + project_name ---
            _item = CONN.getfromdb(
                query=target["sql_feed"], 
                DB='mysqldb', 
                database=config["mysql_db"], 
                server=server_id, 
                host=current_host
            )

            list_id_with_info = [(x[0], x[1], x[2], x[3], x[4]) for x in _item]
            print(f"  👉 พบข้อมูลจาก Feed: {len(list_id_with_info)} โพสต์")

            list_content = sa.get_content(list_id_with_info, "Feed")

            # --- ดึง Comment msg_ids + company_name ---
            _item = CONN.getfromdb(
                query=target["sql_comment"], 
                DB='mysqldb', 
                database=config["mysql_db"], 
                server=server_id, 
                host=current_host
            )

            list_id_with_info = [(x[0], x[1], x[2], x[3], x[4]) for x in _item]
            print(f"  👉 พบข้อมูลจาก Comment: {len(list_id_with_info)} โพสต์")

            list_content += sa.get_content(list_id_with_info, "Comment")

            print(f"  📌 รวมทั้งหมดที่ต้องประมวลผล: {len(list_content)} โพสต์")

            # --- วิเคราะห์ด้วย Ollama แล้วบันทึกลง DB ---
            if list_content:
                sa.analysis(list_content, current_host, server=server_id, table_prefix=target["table_prefix"])
            else:
                print(f"  ⏩ ไม่มีข้อมูลใหม่สำหรับ {target['name']} ข้ามไปทำส่วนถัดไป...")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n🎉 สิ้นสุดการทำงานทั้งหมด! ใช้เวลาไปทั้งสิ้น: {total_time:.2f} วินาที")

