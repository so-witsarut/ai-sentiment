# coding=utf-8
r"""
run_forever.py — รัน Sentiment Analysis วนซ้ำตลอดเวลาบน Windows
─────────────────────────────────────────────────────────────────
วิธีรัน:
    .venv\Scripts\python.exe run_forever.py

ตั้งค่าใน .env:
    RUN_INTERVAL_MINUTES=10     ← รอกี่นาทีก่อนรอบถัดไป (default: 10)
    RUN_MODE=mockup             ← mockup = ข้อมูลจำลอง | db = DB จริง
"""

import os
import sys
import time
import logging
import traceback
import json
import re
import pymysql
from pymongo import MongoClient
from datetime import datetime, timedelta

# ─── โหลด .env ────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── แก้ปัญหาภาษาไทยบน Windows ───────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ─── สร้างโฟลเดอร์ logs ───────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"sentiment_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ─── Import คลาสหลักจาก ai_sentiment.py ──────────────────────
try:
    from ai_sentiment import sentiment, OllamaSentimentAnalyzer
except ImportError as e:
    print(f"[ERROR] import ai_sentiment.py ไม่ได้: {e}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# ตั้งค่า
# ═══════════════════════════════════════════════════════════════
INTERVAL  = 10
RUN_MODE  = "mockup"
DB_CONFIG = {}

def load_config():
    global INTERVAL, RUN_MODE, DB_CONFIG
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass

    # Reload connection เพื่ออัพเดท module-level globals จาก environment variables ใน .env
    try:
        import importlib
        import connection
        importlib.reload(connection)
    except Exception as e:
        log.warning(f"ไม่สามารถ reload connection module ได้: {e}")

    INTERVAL  = int(os.environ.get("RUN_INTERVAL_MINUTES", 10))   # นาที
    RUN_MODE  = os.environ.get("RUN_MODE", "mockup").lower()       # mockup | db

    DB_CONFIG = {
        "mysql_host_1":   os.environ.get("MYSQL_HOST_1", "10.130.84.170"),
        "mysql_host_2":   os.environ.get("MYSQL_HOST_2", "10.130.69.57"),
        "mysql_port":     int(os.environ.get("MYSQL_PORT", 3306)),
        "mysql_user":     os.environ.get("MYSQL_USER"),
        "mysql_password": os.environ.get("MYSQL_PASSWORD"),
        "mysql_db":       os.environ.get("MYSQL_DB", "blue_eye"),
        "mongo_host":     os.environ.get("MONGO_HOST"),
        "mongo_port":     int(os.environ.get("MONGO_PORT", 34596)),
        "mongo_user":     os.environ.get("MONGO_USER"),
        "mongo_password": os.environ.get("MONGO_PASSWORD"),
        "mongo_db":       os.environ.get("MONGO_DB", "blue_eye"),
        "ssh_host":       os.environ.get("SSH_HOST"),
        "ssh_port":       int(os.environ.get("SSH_PORT", 22)),
        "ssh_user":       os.environ.get("SSH_USER"),
        "ssh_password":   os.environ.get("SSH_PASSWORD"),
    }

# โหลดการตั้งค่าครั้งแรก
load_config()

# ─── ข้อมูลจำลอง (ใช้เมื่อ RUN_MODE=mockup) ─────────────────
# รูปแบบ: (msg_id, content, company_name, project_name, post_user, keyword_name)
MOCKUP_DATA = [
    ('DYjCAykTYVB',
     'อนาคต ทางเชื่อมใต้ดินห้าง Central Park กับ สถานีรถไฟใต้ดิน MRT สถานีสีลม…อีก 3 เดือน skywalk จะเปิดมาครบ 1 ปีแล้ว ทางใต้ดินนี้ยังก่อสร้างอยู่ งานใต้ดินใช้เวลานานกว่าบนดินมาก แต่คืบหน้าเป็นรูปเป็นร่างไปมากเช่นกัน หวังว่าภายในปีนี้น่าจะสร้างเสร็จ อยากให้อลังการแบบทางเชื่อมตามสถานีใต้ดินที่โตเกียวจัง😄',
     'Central', 'Central', 'ฟุตบาทไทยสไตล์', 'Central Park'),

    ('100080317547658_1024757183544857',
     'บอกลาเบียร์สิงห์ 🤣 ปลอม คุณหลอกดาว คุณไม่รักครอบครัว กุจะหันไปกินช้าง',
     'Boonrawd', 'Boonrawd', 'Siri Preyachy', 'สิงห์'),

    ('28143837526',
     '"สิงห์" เสี่ยงเสียความผูกพันต่อผู้บริโภค จากมหากาพย์ดราม่าครอบครัว',
     'Boonrawd', 'Boonrawd', 'Positioningmag', 'สิงห์'),

    ('100064782188264_1454988ddddddddd',
     '"บุญรอดบริวเวอรี่" สั่งปลดพายสก๊อตพ้นทุกตำแหน่งแล้ว หลังดราม่าครอบครัว #ทรายสก๊อต',
     'Boonrawd', 'Boonrawd', 'Dailynews', 'บุญรอดบริวเวอรี่'),

    ('DYjYKgfFEH5',
     'งานแถลง HISENSE Football Youth Cup 2026 สนามสิงห์เชียงราย สเตเดียม',
     'Boonrawd', 'Boonrawd', 'chiangrai_united', 'สิงห์'),

    ('DYjgfFeeEH4',
     'สวยแบบจักรวาลร้องว๊าววววควรค่าแก่การเป็นสะใภ้รังษีสิงห์พิพัฒน์หุ่นคือไม่อ้วน ไม่ผอม ดูรวย ดูแพงลักษณะคนมีวาสนาของแท่',
     'Boonrawd', 'Boonrawd', 'ก็ชอบแบบนี้ ก็ชอบแบบนี้', 'สิงห์'),

    ('DYjgfFeeEH3',
     '#บันเทิง #ดราม่า #ทรายสก็อต #มายด์',
     'Boonrawd', 'Boonrawd', 'Sparktrends', 'ทราย'),

    ('DYjgfFeeEH8',
     'ไม่ใช่ทายาท ไม่ใช่ลูก เป็นแค่ลูกค้าของ สก๊อต และ สิงห์เท่านั้น เราชอบ เรื่องอื่นเราไม่ยุ่ง มุ่งแต่เรื่องของเราก็ปวดหัวพอแล้วจ้ะ',
     'Boonrawd', 'Boonrawd', 'ยิ่งรัก ยิ่งลุ่มหลง', 'สิงห์'),

     ('DYjgfFeeEH8',
     'ชอบฟังเพลงชาร์คเต้ฮะสมัย ม.ปลาย เปิดวนอ่ะ เพลงเบอร์สอง',
     'Boonrawd', 'Boonrawd', 'Min Thitapus', 'ชาร์คเต้'),
]

# ═══════════════════════════════════════════════════════════════
# ดึงข้อมูลจาก DB จริง
# ═══════════════════════════════════════════════════════════════
def process_targets(sa_obj):
    yesterday = str(datetime.now() - timedelta(days=1))[:10]
    now_str   = str(datetime.now())[:10]

    targets = [
        {
            "name": "OWN MATCH",
            "table_prefix": "own_match",
            "sql_feed": (
                f"SELECT omd.msg_id, "
                f"IFNULL(c.company_name, '') as company_name, "
                f"IFNULL(ck.company_keyword_name, '') as project_name, "
                f"IFNULL(omd.post_user, '') as post_user, "
                f"IFNULL(GROUP_CONCAT(DISTINCT k.keyword_name SEPARATOR ', '), '') as keyword_name "
                f"FROM own_match_daily omd "
                f"LEFT JOIN company_keyword ck ON omd.company_keyword_id = ck.company_keyword_id "
                f"LEFT JOIN client c ON omd.client_id = c.client_id "
                f"LEFT JOIN own_key_match okm ON okm.own_match_id = omd.own_match_id "
                f"LEFT JOIN keyword k ON okm.keyword_id = k.keyword_id "
                f"WHERE date(omd.msg_time) BETWEEN '{yesterday}' AND '{now_str}' "
                f"AND omd.sentiment_status = '0' AND omd.match_type = 'Feed' "
                f"GROUP BY omd.msg_id, company_name, project_name, post_user "
                f"ORDER BY omd.msg_time ASC"
            ),
            "sql_comment": (
                f"SELECT omd.msg_id, "
                f"IFNULL(c.company_name, '') as company_name, "
                f"IFNULL(ck.company_keyword_name, '') as project_name, "
                f"IFNULL(omd.post_user, '') as post_user, "
                f"IFNULL(GROUP_CONCAT(DISTINCT k.keyword_name SEPARATOR ', '), '') as keyword_name "
                f"FROM own_match_daily omd "
                f"LEFT JOIN company_keyword ck ON omd.company_keyword_id = ck.company_keyword_id "
                f"LEFT JOIN client c ON omd.client_id = c.client_id "
                f"LEFT JOIN own_key_match okm ON okm.own_match_id = omd.own_match_id "
                f"LEFT JOIN keyword k ON okm.keyword_id = k.keyword_id "
                f"WHERE date(omd.msg_time) BETWEEN '{yesterday}' AND '{now_str}' "
                f"AND omd.sentiment_status = '0' AND omd.match_type = 'Comment' "
                f"GROUP BY omd.msg_id, company_name, project_name, post_user "
                f"ORDER BY omd.msg_time ASC"
            )
        },
        {
            "name": "COMPETITOR MATCH",
            "table_prefix": "competitor_match",
            "sql_feed": (
                f"SELECT cmd.msg_id, "
                f"IFNULL(c.company_name, '') as company_name, "
                f"IFNULL(ck.company_keyword_name, '') as project_name, "
                f"IFNULL(cmd.post_user, '') as post_user, "
                f"IFNULL(GROUP_CONCAT(DISTINCT k.keyword_name SEPARATOR ', '), '') as keyword_name "
                f"FROM competitor_match_daily cmd "
                f"LEFT JOIN company_keyword ck ON cmd.company_keyword_id = ck.company_keyword_id "
                f"LEFT JOIN client c ON cmd.client_id = c.client_id "
                f"LEFT JOIN competitor_key_match ckm ON ckm.competitor_match_id = cmd.competitor_match_id "
                f"LEFT JOIN keyword k ON ckm.keyword_id = k.keyword_id "
                f"WHERE date(cmd.msg_time) BETWEEN '{yesterday}' AND '{now_str}' "
                f"AND cmd.sentiment_status = '0' AND cmd.match_type = 'Feed' "
                f"GROUP BY cmd.msg_id, company_name, project_name, post_user "
                f"ORDER BY cmd.msg_time ASC"
            ),
            "sql_comment": (
                f"SELECT cmd.msg_id, "
                f"IFNULL(c.company_name, '') as company_name, "
                f"IFNULL(ck.company_keyword_name, '') as project_name, "
                f"IFNULL(cmd.post_user, '') as post_user, "
                f"IFNULL(GROUP_CONCAT(DISTINCT k.keyword_name SEPARATOR ', '), '') as keyword_name "
                f"FROM competitor_match_daily cmd "
                f"LEFT JOIN company_keyword ck ON cmd.company_keyword_id = ck.company_keyword_id "
                f"LEFT JOIN client c ON cmd.client_id = c.client_id "
                f"LEFT JOIN competitor_key_match ckm ON ckm.competitor_match_id = cmd.competitor_match_id "
                f"LEFT JOIN keyword k ON ckm.keyword_id = k.keyword_id "
                f"WHERE date(cmd.msg_time) BETWEEN '{yesterday}' AND '{now_str}' "
                f"AND cmd.sentiment_status = '0' AND cmd.match_type = 'Comment' "
                f"GROUP BY cmd.msg_id, company_name, project_name, post_user "
                f"ORDER BY cmd.msg_time ASC"
            )
        }
    ]

    for server_id in [1,2]:
        current_host = DB_CONFIG.get(f"mysql_host_{server_id}")
        
        log.info(f"\n{'=' * 40}")
        log.info(f"   PROCESSING MYSQL SERVER {server_id} ({current_host})")
        log.info(f"{'=' * 40}")

        for target in targets:
            # log.info(target["sql_feed"])
            import sys
            if r"ai-sentiment" not in sys.path:
                sys.path.append(r"ai-sentiment")
            import connection
            CONN = connection.DatabaseConnection()

            feeds = CONN.getfromdb(
                query=target["sql_feed"], 
                DB='mysqldb', 
                database=DB_CONFIG["mysql_db"], 
                server=server_id, 
                host=current_host
            )
            comments = CONN.getfromdb(
                query=target["sql_comment"], 
                DB='mysqldb', 
                database=DB_CONFIG["mysql_db"], 
                server=server_id, 
                host=current_host
            )

            log.info(f"DB [{target['name']}] (Server {server_id}) → Feed: {len(feeds)} | Comment: {len(comments)} รายการ")

            content = []
            if feeds:    content += sa_obj.get_content(list(feeds), "Feed")
            if comments: content += sa_obj.get_content(list(comments), "Comment")
            
            if content:
                sa_obj.analysis(content, current_host, server=server_id, table_prefix=target["table_prefix"])
            else:
                log.info(f"ไม่มีข้อมูลที่ต้องวิเคราะห์สำหรับ {target['name']} บน Server {server_id}")

# ═══════════════════════════════════════════════════════════════
# รัน 1 รอบ
# ═══════════════════════════════════════════════════════════════
def run_once(round_num):
    # โหลด config ใหม่ในแต่ละรอบ เพื่อให้สามารถเปลี่ยนสลับ RUN_MODE จาก .env ได้แบบ dynamic
    load_config()

    log.info(f"{'─'*60}")
    log.info(f"  รอบที่ {round_num}  [{RUN_MODE.upper()}]  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"{'─'*60}")

    sa = sentiment(DB_CONFIG)

    if RUN_MODE == "mockup":
        list_content = MOCKUP_DATA
        log.info(f"[MOCKUP] ใช้ข้อมูลจำลอง {len(list_content)} รายการ")
        if list_content:
            sa.analysis(list_content, DB_CONFIG.get("mysql_host_1") or "localhost", table_prefix="own_match", save_db=False)
    else:
        process_targets(sa)

    log.info(f"รอบที่ {round_num} เสร็จสิ้น ✓")

# ═══════════════════════════════════════════════════════════════
# Main — loop ตลอดเวลา
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         AI Sentiment — รันตลอดเวลา (Windows)           ║")
    print(f"║  Mode: {RUN_MODE:<10}  |  ช่วงเวลา: ทุก {INTERVAL} นาที            ║")
    print(f"║  Log: logs/sentiment_{datetime.now().strftime('%Y%m%d')}.log              ║")
    print("║  กด  Ctrl+C  เพื่อหยุด                                 ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    round_num = 0
    while True:
        round_num += 1
        try:
            run_once(round_num)
        except KeyboardInterrupt:
            print("\n[หยุด] ผู้ใช้กด Ctrl+C")
            break
        except Exception:
            log.error(f"[ERROR] รอบที่ {round_num} เกิดข้อผิดพลาด:")
            log.error(traceback.format_exc())
            log.info("จะลองใหม่ในรอบถัดไป...")

        # ─── นับถอยหลังแบบแสดงนาทีที่เหลือ ───────────────────
        log.info(f"\nรอ {INTERVAL} นาที ก่อนรอบถัดไป... (Ctrl+C เพื่อหยุด)\n")
        try:
            for remaining in range(INTERVAL * 60, 0, -60):
                mins = remaining // 60
                print(f"  ⏳ อีก {mins} นาที...", end="\r", flush=True)
                time.sleep(60)
            print(" " * 30, end="\r")  # ลบบรรทัดนับถอยหลัง
        except KeyboardInterrupt:
            print("\n[หยุด] ผู้ใช้กด Ctrl+C")
            break
