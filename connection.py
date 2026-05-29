# charset-utf8
import os
from sshtunnel import SSHTunnelForwarder

from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime
import pymysql, pymongo

import requests, boto3
import threading

# ─── โหลด .env ────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- 1. ตั้งค่า DigitalOcean Spaces ---
REGION_NAME = os.environ.get('DO_SPACES_REGION', 'sgp1')
ENDPOINT_URL = f'https://{REGION_NAME}.digitaloceanspaces.com'
ACCESS_KEY = os.environ.get('DO_SPACES_ACCESS_KEY', '')
SECRET_KEY = os.environ.get('DO_SPACES_SECRET_KEY', '')
SPACE_NAME = os.environ.get('DO_SPACES_BUCKET', 'be-post-scrape-img')

# สร้าง Client เพื่อเชื่อมต่อ
client = boto3.client('s3',
                      region_name=REGION_NAME,
                      endpoint_url=ENDPOINT_URL,
                      aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

now_bkk = datetime.now()
cutoff = now_bkk.replace(hour=0, minute=0, second=1, microsecond=0)\
                .strftime("%Y-%m-%d %H:%M:%S")

# ─── SSH Tunnel (สำหรับ MongoDB) ──────────────────────────────
SSH_HOST = os.environ.get("SSH_HOST", "")
SSH_PORT = int(os.environ.get("SSH_PORT", 22))
SSH_USER = os.environ.get("SSH_USER", "")
SSH_PASSWORD = os.environ.get("SSH_PASSWORD", "")

# ─── MongoDB ──────────────────────────────────────────────────
BE_MONGO_PRIVATE_IP = os.environ.get("MONGO_HOST", "10.130.72.139")
BE_MONGO_PORT = int(os.environ.get("MONGO_PORT", 34596))
BE_MONGO_USERNAME = os.environ.get("MONGO_USER", "")
BE_MONGO_PASSWORD = os.environ.get("MONGO_PASSWORD", "")

# ─── SSH Tunnel (สำหรับ MySQL — รองรับหลาย server) ────────────
SSH_MYSQL_HOST = {
    1: os.environ.get("SSH_MYSQL_HOST_1", ""),
    2: os.environ.get("SSH_MYSQL_HOST_2", ""),
}
SSH_MYSQL_USER = {
    1: os.environ.get("SSH_MYSQL_USER_1", ""),
    2: os.environ.get("SSH_MYSQL_USER_2", ""),
}
SSH_MYSQL_PASSWORD = {
    1: os.environ.get("SSH_MYSQL_PASSWORD_1", ""),
    2: os.environ.get("SSH_MYSQL_PASSWORD_2", ""),
}

SSH_MYSQL_PORT = 22

# ─── MySQL ────────────────────────────────────────────────────
BE_MYSQL_PORT = int(os.environ.get("MYSQL_PORT", 3306))
BE_MYSQL_USER = os.environ.get("MYSQL_USER", "")
BE_MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")

class DatabaseConnection():        
    def __init__(self):
        self._mongo_tunnel = None
        self._mongo_client = None
        self._mongo_lock = threading.Lock()

    # ─── MongoDB (ผ่าน SSH Tunnel) ─────────────────────────────
    def _get_mongo_client(self):
        with self._mongo_lock:
            if self._mongo_client is not None and self._mongo_tunnel is not None:
                if self._mongo_tunnel.is_active:
                    return self._mongo_client
                else:
                    try:
                        self._mongo_tunnel.stop()
                    except:
                        pass
                    self._mongo_client.close()
                    self._mongo_client = None
                    self._mongo_tunnel = None
            
            self._mongo_tunnel = SSHTunnelForwarder(
                (SSH_HOST, SSH_PORT),
                ssh_username=SSH_USER,
                ssh_password=SSH_PASSWORD,
                remote_bind_address=(BE_MONGO_PRIVATE_IP, BE_MONGO_PORT),
            )
            self._mongo_tunnel.start()
            self._mongo_client = MongoClient(
                host="127.0.0.1",
                port=self._mongo_tunnel.local_bind_port,
                username=BE_MONGO_USERNAME,
                password=BE_MONGO_PASSWORD,
                maxPoolSize=200,
            )
            return self._mongo_client

    def get_mongo_client(self):
        """Public method — คืน MongoClient ที่เชื่อมผ่าน SSH Tunnel แล้ว"""
        return self._get_mongo_client()

    def checktoday(self, timepost):
        check = 0
        time = str(timepost)
        now = str(datetime.now()).split(' ')[0]
        feedtimepost = time.split(' ')[0]
        if now == feedtimepost:
            check = 1
        return check

    # ─── MySQL SELECT (ผ่าน SSH Tunnel) ────────────────────────
    def getfromdb(self, **kwargs):
        query = kwargs.get("query")
        db_type = kwargs.get("DB")
        database = kwargs.get("database")

        if db_type == "mysqldb":
            if database == "blue_eye":
                server = kwargs.get("server")
                remote_host = kwargs.get("host")

                with SSHTunnelForwarder(
                    (SSH_MYSQL_HOST[server], SSH_MYSQL_PORT),
                    ssh_username=SSH_MYSQL_USER[server],
                    ssh_password=SSH_MYSQL_PASSWORD[server],
                    remote_bind_address=(remote_host, BE_MYSQL_PORT),
                ) as tunnel:
                    conn = pymysql.connect(
                        host="127.0.0.1",             # ต่อเข้า tunnel ที่ฝั่งเรา
                        port=tunnel.local_bind_port,  # port ที่ tunnel bind ไว้
                        user=BE_MYSQL_USER,
                        passwd=BE_MYSQL_PASSWORD,
                        db=database,
                        charset="utf8",
                    )
                    try:
                        with conn.cursor() as cursor:
                            cursor.execute(query)
                            rows = cursor.fetchall()
                        # ถ้าเป็น SELECT อย่างเดียวไม่ต้อง commit ก็ได้
                        return rows
                    finally:
                        conn.close()

            else:
                MYSQL_DB_CONNECTION = pymysql.connect(
                    host = os.environ.get("DO_MYSQL_HOST", ""),
                    port = int(os.environ.get("DO_MYSQL_PORT", 25060)),
                    user = os.environ.get("DO_MYSQL_USER", ""),
                    passwd = os.environ.get("DO_MYSQL_PASSWORD", ""),
                    db = os.environ.get("DO_MYSQL_DB", "blueeye"),
                    charset = "utf8"
                )
                CURSOR = MYSQL_DB_CONNECTION.cursor()
                CURSOR.execute(query)
                result = CURSOR.fetchall()

                MYSQL_DB_CONNECTION.close()
                return result
            
            # ถ้าอยากรองรับ DB แบบอื่น ค่อยมาเติมด้านล่าง
            # raise ValueError("Unsupported DB or database in getfromdb")

        elif db_type == "mongodb":
            if database == "blue_eye":
                if kwargs.get("__type") == "update_engagement":
                    DB_CONNECTION = self._get_mongo_client()
                    if True:
                        DB = DB_CONNECTION[database]
                    
                        # Collections Definitions
                        COLL_dailyfeed = DB["DairyFeed"] 
                        COLL_3monthsfeed = DB["threeMonthsFeed"] 
                        COLL_feed = DB["Feed"] 
                        COLL_dailycomment = DB["DairyComment"] 
                        COLL_3monthscomment = DB["threeMonthsComment"] 
                        COLL_comment = DB["Comment"] 

                        base_filter = {"sourceid": 1}
                        flt = {**base_filter}

                        proj = {"_id": 1, "feedlink": 1}
                        result = list(COLL_dailyfeed.find(flt, proj))

                elif kwargs.get("__type") == "thumbnail":
                    DB_CONNECTION = self._get_mongo_client()
                    if True:
                        DB = DB_CONNECTION[database]

                        coll_dailyfeed = DB["DairyFeed"] # collection dairyFeed
                        coll_3monthsfeed = DB["threeMonthsFeed"] # collection 3monthsFeed
                        coll_feed = DB["Feed"] # collection Feed

                        colname = kwargs.get("colname")
                        colval = kwargs.get("colval")

                        # ถ้า colval เป็น list/tuple → ใช้ $in
                        if isinstance(colval, (list, tuple, set)):
                            query = {colname: {"$in": list(colval)}}
                        else:
                            query = {colname: colval}

                        # ดึงเฉพาะ field ที่ต้องใช้ก็ได้ จะเร็วขึ้นนิดหน่อย
                        cursor = coll_3monthsfeed.find(query, {"_id": 1, "feedlink": 1, "thumbnails": 1})
                        return list(cursor)
                else:
                    DB_CONNECTION = self._get_mongo_client()
                    DB = DB_CONNECTION[database]
                    COLL_dailyfeed = DB["DairyFeed"] # collection dairyfeed
                    COLL_feed = DB["Feed"] # collection feed
                    count_daily = COLL_dailyfeed.count_documents({ "_id": {"$regex": query } })
                    count_feed = COLL_feed.count_documents({ "_id": {"$regex": query } })
                    result = count_daily + count_feed

            # DB_CONNECTION.close() # Shared connection should not be closed
        return result

    # ─── MySQL Connection สำหรับ UPDATE/INSERT (ต้อง commit เอง) ─
    def get_mysql_connection(self, server=1, host="10.130.84.170", database="blue_eye"):
        """คืนค่า (tunnel, conn) เพื่อเอาไปใช้กับ UPDATE/INSERT ที่ต้องมีการ commit() และจัดการ connection เอง"""
        tunnel = SSHTunnelForwarder(
            (SSH_MYSQL_HOST[server], SSH_MYSQL_PORT),
            ssh_username=SSH_MYSQL_USER[server],
            ssh_password=SSH_MYSQL_PASSWORD[server],
            remote_bind_address=(host, BE_MYSQL_PORT),
        )
        tunnel.start()
        conn = pymysql.connect(
            host="127.0.0.1",
            port=tunnel.local_bind_port,
            user=BE_MYSQL_USER,
            passwd=BE_MYSQL_PASSWORD,
            db=database,
            charset="utf8",
            connect_timeout=10,
        )
        return tunnel, conn