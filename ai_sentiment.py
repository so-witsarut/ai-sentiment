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

def get_keyword_context(text, keyword, window=150, max_fallback_length=1000):
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
        self.system_instruction = (
            "Thai Brand Sentiment Classifier. "
            "Given a post and Target Entity, reply ONLY: {\"ai_sentiment\": 0|100|-100}. "
            "No explanation."
        )

    def _analyze_single_post(self, post, project_name):
        """
        วิเคราะห์ sentiment ของโพสต์เดียว (ใช้ใน ThreadPoolExecutor)
        Returns: dict with post_id, ai_sentiment, confidence หรือ None ถ้า error
        """
        post_id   = post["post_id"]
        content   = post["content"]
        post_user = post["post_user"]
        matched_keyword = post.get("keyword_name", project_name)

        user_prompt = (
            f"Target: '{project_name}' | Keyword: '{matched_keyword}' | User: '{post_user}'\n"
            f"Post: '{content[:200]}'\n\n"   # ✅ ตัดเหลือ 200 ตัวอักษร ลด token ให้เบาสุด
            f"Rules:\n"
            f"1. Own official page → 0\n"
            f"2. Keyword is place/surname/idiom unrelated to brand → 0\n"
            f"3. Ad/PR/sports news (no brand scandal) → 0\n"
            f"4. Criticize/boycott/mock/scandal → -100\n"
            f"5. Praise/support/defend → 100\n"
            f"6. Default → 0\n"
            f"{{\"ai_sentiment\": ?}}"
        )

        payload = {
            "model": self.model,
            "system": self.system_instruction,
            "prompt": user_prompt,
            "stream": False,
            "format": "json",
            "think": False,
            "keep_alive": -1,                    # ✅ โมเดลค้างใน GPU ไม่ต้อง reload
            "options": {
                "temperature": 0.0,
                "top_p": 0.1,
                "seed": 42,
                "num_predict": 15,               # ✅ JSON แค่ {"ai_sentiment": -100} ไม่ต้องเผื่อ
                "num_ctx": 1024,                 # ✅ context window เล็ก ประหยัด VRAM
                "num_batch": 256,                # ✅ สมดุลระหว่างความเร็วกับ VRAM
                "num_gpu": 99,                   # ✅ บังคับทุก layer ขึ้น GPU
            }
        }

        try:
            response = requests.post(self.base_url, json=payload, timeout=120)

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

                    return {
                        "post_id":      post_id,
                        "ai_sentiment": int(ai_sentiment),
                        "confidence":   0
                    }
                except json.JSONDecodeError as e:
                    print(f"  -> JSONDecodeError [{post_id}]: {e} | Raw Clean Text: {clean_text[:150]}")
            else:
                print(f"  -> HTTP Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"  -> Request Error [{post_id}]: {e}")

        return None

    def analyze_post_sentiments(self, json_posts, project_name=""):
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
                executor.submit(self._analyze_single_post, post, project_name): post
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

    def get_content(self, list_id_with_project, collection):
        list_content = []

        project_map   = {msg_id: proj         for (msg_id, proj, post_user, kw_name) in list_id_with_project}
        post_user_map = {msg_id: post_user    for (msg_id, proj, post_user, kw_name) in list_id_with_project}
        keyword_map   = {msg_id: kw_name      for (msg_id, proj, post_user, kw_name) in list_id_with_project}
        list_id = [msg_id for (msg_id, proj, post_user, kw_name) in list_id_with_project]

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
            proj_name = project_map.get(msg_id, "")
            post_user = post_user_map.get(msg_id, "")
            kw_name   = keyword_map.get(msg_id, "")
            # fallback: ถ้า post_user ว่าง ให้ใช้ prefix ก่อน '_' จาก msg_id
            if not post_user:
                post_user = str(msg_id).split("_")[0]
            list_content.append((msg_id, feedcontent, proj_name, post_user, kw_name))

        # ไม่ปิด connection — เป็น shared connection ที่ CONN จัดการให้

        return list_content

    # def get_keyword(self):
    #     sql = """
    #         SELECT keyword_name, client_id, status
    #         FROM keyword
    #         WHERE client_id = 67 AND status = 'active'
    #         ORDER BY created_date DESC
    #         LIMIT 1 
    #     """
    #     keywords_item = CONN.getfromdb(query=sql,
    #                                 DB='mysqldb',
    #                                 database='blue_eye',
    #                                 server=1,
    #                                 host='10.130.84.170'
    #     )

    #     for item in keywords_item:
    #         if item[0] not in self.list_of_keywords:
    #             self.list_of_keywords.append((item[0], item[1], item[2]))

    def analysis(self, list_content, host, server=1, table_prefix="own_match"):
        """
        วิเคราะห์ sentiment แล้วอัพเดทลง MySQL

        Args:
            list_content (list): [(msg_id, content, project_name, post_user, kw_name), ...]
            host (str): MySQL host
            server (int): MySQL Server ID (1 or 2)
            table_prefix (str): 'own_match' หรือ 'competitor_match'
        """
        tunnel, DB_CONNECTION = CONN.get_mysql_connection(server=server, host=host, database=self.config["mysql_db"])

        BATCH_SIZE = 20
        total = len(list_content)
        print(f"\nTotal content to analyze: {total}")

        for batch_start in range(0, total, BATCH_SIZE):
            batch = list_content[batch_start:batch_start + BATCH_SIZE]
            batch_end = min(batch_start + BATCH_SIZE, total)
            print(f"\n--- Batch {batch_start + 1}-{batch_end} / {total} ---")

            posts_for_ai = []
            batch_project_name = ""
            for (_id, content, project_name, post_user, kw_name) in batch:
                text = re.sub(r"<[^>]+>", "", str(content))
                text = re.sub(r"\s+", " ", text).strip()

                if not batch_project_name and project_name:
                    batch_project_name = project_name

                if text:
                    # กรณีที่มีหลายคำคั่นด้วยลูกน้ำ (เช่น ทราย, สุนิษฐ์, พาย) อาจจะเลือกคำแรกมาใช้สแกน
                    first_keyword = kw_name.split(",")[0].strip() if kw_name else ""
                    clean_short_content = get_keyword_context(text, first_keyword, window=200)  # ✅ GTX 1660: ลดจาก 300 → 200 ลด token
                    print("clean_short_content >>>>>>> ", clean_short_content)

                    posts_for_ai.append({
                        "post_id": str(_id),
                        "post_user": post_user,
                        "project_name": project_name,
                        "keyword_name": kw_name,
                        "content": clean_short_content
                    })

            if not posts_for_ai:
                print("  -> All content in batch is empty, skipping.")
                continue

            json_str = json.dumps(posts_for_ai, ensure_ascii=False)

            print(f"  -> Sending {len(posts_for_ai)} posts to Ollama (qwen3-8b-instruct)...")
            print(f"  -> Project: {batch_project_name}")

            ollama_response = self.ollama.analyze_post_sentiments(json_str, batch_project_name)

            if "error" in ollama_response:
                print(f"  -> Ollama Error: {ollama_response.get('message', ollama_response['error'])}")

            ollama_results = ollama_response.get("data", [])
            ollama_token_usage = ollama_response.get("token_usage", {})
            print(f"  -> Ollama Tokens: input={ollama_token_usage.get('input', 0)}, "
                  f"output={ollama_token_usage.get('output', 0)}, "
                  f"total={ollama_token_usage.get('total', 0)}")

            ollama_map = {}
            if isinstance(ollama_results, list):
                for res in ollama_results:
                    if "post_id" in res and "ai_sentiment" in res:
                        ollama_map[res["post_id"]] = {
                            "ai_sentiment": res["ai_sentiment"],
                            "confidence": res.get("confidence", 0)
                        }

            print(f"\n  {'No.':<5} {'Post ID':<32} {'Ollama':<12} {'Conf.':<7} {'Project':<20} Keyword")
            print(f"  {'='*120}")

            for idx, (_id, content, project_name, post_user, kw_name) in enumerate(batch, 1):
                str_id = str(_id)

                ollama_conf_str = "N/A"
                if str_id in ollama_map:
                    ollama_val = float(ollama_map[str_id]["ai_sentiment"])
                    ollama_conf = ollama_map[str_id]["confidence"]
                    ollama_conf_str = f"{ollama_conf}%" if ollama_conf is not None else "N/A"
                    if ollama_val > 0:
                        ollama_label = "Positive"
                    elif ollama_val < 0:
                        ollama_label = "Negative"
                    else:
                        ollama_label = "Neutral"
                else:
                    ollama_val = None
                    ollama_label = "N/A"

                display_content = str(content)[:300].replace("\n", " ")
                print(f"  {idx:<5} {str_id[:30]:<32} {ollama_label:<12} {ollama_conf_str:<7} {project_name:<20} {kw_name}")
                print(f"        Content: {display_content}")
                if ollama_val is not None:
                    print(f"        Ollama={ollama_val} (Conf: {ollama_conf_str})")
                print()

            print(f"  {'='*120}")
            print(f"  Batch: {len(batch)} posts | Ollama results: {len(ollama_map)}")

            # # ===== อัพเดท DB =====
            # for (_id, content, project_name, post_user, kw_name) in batch:
            #     str_id = str(_id)
            #     if str_id not in ollama_map:
            #         continue
            #     sentiment_val = float(ollama_map[str_id]["ai_sentiment"])
            #     cursor = DB_CONNECTION.cursor()
            #     SQL = f'UPDATE {table_prefix} SET {table_prefix}_sentiment = {{0}}, sentiment_status = {{1}} WHERE msg_id = "{{2}}"'.format(sentiment_val, "1", str(_id))
            #     cursor.execute(SQL)
            #     SQL = f'UPDATE {table_prefix}_daily SET {table_prefix}_sentiment = {{0}}, sentiment_status = {{1}} WHERE msg_id = "{{2}}"'.format(sentiment_val, "1", str(_id))
            #     cursor.execute(SQL)
            #     SQL = f'UPDATE {table_prefix}_3months SET {table_prefix}_sentiment = {{0}}, sentiment_status = {{1}} WHERE msg_id = "{{2}}"'.format(sentiment_val, "1", str(_id))
            #     cursor.execute(SQL)
            # DB_CONNECTION.commit()

            # ไม่ต้องมีการดีเลย์ระหว่าง batch สำหรับ Local Ollama

        DB_CONNECTION.close()
        if 'tunnel' in locals() and tunnel:
            tunnel.stop()
        print("\n--- Process Complete (บันทึกลง DB แล้ว) ---")

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
                f"'' as keyword_name "
                f"FROM competitor_match_daily cmd "
                f"LEFT JOIN company_keyword ck ON cmd.company_keyword_id = ck.company_keyword_id "
                f"WHERE date(cmd.msg_time) BETWEEN '{yesterday}' AND '{now}' "
                f"AND cmd.sentiment_status = '0' AND cmd.match_type = 'Feed' "
                f"GROUP BY cmd.msg_id, project_name, post_user "
                f"ORDER BY cmd.msg_time ASC "
            ),
            "sql_comment": (
                f"SELECT cmd.msg_id, "
                f"IFNULL(ck.company_keyword_name, '') as project_name, "
                f"IFNULL(cmd.post_user, '') as post_user, "
                f"'' as keyword_name "
                f"FROM competitor_match_daily cmd "
                f"LEFT JOIN company_keyword ck ON cmd.company_keyword_id = ck.company_keyword_id "
                f"WHERE date(cmd.msg_time) BETWEEN '{yesterday}' AND '{now}' "
                f"AND cmd.sentiment_status = '0' AND cmd.match_type = 'Comment' "
                f"GROUP BY cmd.msg_id, project_name, post_user "
                f"ORDER BY cmd.msg_time ASC "
            )
        }
    ]

    print("=" * 65)
    print("     Ollama (qcwind/qwen3-8b-instruct-Q4-K-M:latest) SENTIMENT ANALYSIS")
    print(f"     Date Range: {yesterday} ~ {now}")
    print("=" * 65)

    sa = sentiment(config)

    for server_id in [1, 2]:
        current_host = config[f"mysql_host_{server_id}"]
        
        print(f"\n{'=' * 65}")
        print(f"     STARTING MYSQL SERVER {server_id} ({current_host})")
        print(f"{'=' * 65}")

        for target in targets:
            print(f"\n{'=' * 65}")
            print(f"     PROCESSING: {target['name']} (Server {server_id})")
            print(f"{'=' * 65}")

            # --- ดึง Feed msg_ids + project_name ---
            _item = CONN.getfromdb(
                query=target["sql_feed"], 
                DB='mysqldb', 
                database=config["mysql_db"], 
                server=server_id, 
                host=current_host
            )

            list_id_with_project = [(x[0], x[1], x[2], x[3]) for x in _item]
            print(f"\nFeed posts to analyze: {len(list_id_with_project)}")

            print(f"\n--- SQL_FEED Query ({target['name']}) ---")
            print(f"SQL: {target['sql_feed']}")
            print(f"\n{'No.':<5} {'msg_id':<45} {'project_name'}")
            print(f"{'-'*80}")
            for i, (msg_id, proj, post_user, kw_name) in enumerate(list_id_with_project, 1):
                print(f"{i:<5} {str(msg_id)[:43]:<45} {proj}")
            print(f"{'-'*80}")

            list_content = sa.get_content(list_id_with_project, "Feed")

            # --- ดึง Comment msg_ids + project_name ---
            _item = CONN.getfromdb(
                query=target["sql_comment"], 
                DB='mysqldb', 
                database=config["mysql_db"], 
                server=server_id, 
                host=current_host
            )

            list_id_with_project = [(x[0], x[1], x[2], x[3]) for x in _item]
            print(f"Comment posts to analyze: {len(list_id_with_project)}")

            list_content += sa.get_content(list_id_with_project, "Comment")

            print(f"Total content to process for {target['name']} (Server {server_id}): {len(list_content)}")

            # --- วิเคราะห์ด้วย Ollama แล้วบันทึกลง DB ---
            if list_content:
                sa.analysis(list_content, current_host, server=server_id, table_prefix=target["table_prefix"])
            else:
                print(f"\nNo content to analyze for {target['name']} on Server {server_id}.")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n--- Total Execution Time: {total_time:.2f} seconds ---")

