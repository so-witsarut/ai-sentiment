# charset-utf8
import os
from sshtunnel import SSHTunnelForwarder

from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime
import pymysql, pymongo
import pymongo.common
import pymongo.topology_description

# Patch PyMongo 4.x to allow connecting to MongoDB 4.0 / wire version 7 servers
pymongo.MIN_SUPPORTED_WIRE_VERSION = 0
pymongo.common.MIN_SUPPORTED_WIRE_VERSION = 0
pymongo.topology_description.common.MIN_SUPPORTED_WIRE_VERSION = 0

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

import time

class DatabaseConnection():        
    def __init__(self):
        self._mongo_tunnel = None
        self._mongo_client = None
        self._mongo_lock = threading.Lock()

    def _cleanup_mongo_unsafe(self):
        if self._mongo_client is not None:
            try:
                self._mongo_client.close()
            except Exception:
                pass
            self._mongo_client = None
        if self._mongo_tunnel is not None:
            try:
                self._mongo_tunnel.stop()
            except Exception:
                pass
            self._mongo_tunnel = None

    def reset_mongo(self):
        """ล้างการเชื่อมต่อ Mongo + SSH Tunnel เพื่อเตรียมต่อใหม่"""
        with self._mongo_lock:
            self._cleanup_mongo_unsafe()

    # ─── MongoDB (ผ่าน SSH Tunnel) ─────────────────────────────
    def _get_mongo_client(self, retries=3, delay=3):
        with self._mongo_lock:
            for attempt in range(1, retries + 1):
                try:
                    if self._mongo_client is not None and self._mongo_tunnel is not None:
                        if self._mongo_tunnel.is_active:
                            try:
                                self._mongo_client.admin.command('ping')
                                return self._mongo_client
                            except Exception:
                                pass
                        self._cleanup_mongo_unsafe()

                    self._mongo_tunnel = SSHTunnelForwarder(
                        (SSH_HOST, SSH_PORT),
                        ssh_username=SSH_USER,
                        ssh_password=SSH_PASSWORD,
                        remote_bind_address=(BE_MONGO_PRIVATE_IP, BE_MONGO_PORT),
                        set_keepalive=30
                    )
                    self._mongo_tunnel.start()
                    self._mongo_client = MongoClient(
                        host="127.0.0.1",
                        port=self._mongo_tunnel.local_bind_port,
                        username=BE_MONGO_USERNAME,
                        password=BE_MONGO_PASSWORD,
                        maxPoolSize=200,
                        readPreference='secondaryPreferred',
                        serverSelectionTimeoutMS=5000,
                        directConnection=True
                    )
                    self._mongo_client.admin.command('ping')
                    return self._mongo_client
                except Exception as e:
                    print(f"⚠️ [Mongo SSH Tunnel] พยายามเชื่อมต่อครั้งที่ {attempt}/{retries} ล้มเหลว: {e}")
                    self._cleanup_mongo_unsafe()
                    if attempt < retries:
                        time.sleep(delay)
            print("❌ ไม่สามารถเชื่อมต่อ Mongo SSH Tunnel ได้หลังจากลองใหม่ครบแล้ว")
            return None

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

                for attempt in range(1, 4):
                    try:
                        with SSHTunnelForwarder(
                            (SSH_MYSQL_HOST[server], SSH_MYSQL_PORT),
                            ssh_username=SSH_MYSQL_USER[server],
                            ssh_password=SSH_MYSQL_PASSWORD[server],
                            remote_bind_address=(remote_host, BE_MYSQL_PORT),
                            set_keepalive=30
                        ) as tunnel:
                            conn = pymysql.connect(
                                host="127.0.0.1",             # ต่อเข้า tunnel ที่ฝั่งเรา
                                port=tunnel.local_bind_port,  # port ที่ tunnel bind ไว้
                                user=BE_MYSQL_USER,
                                passwd=BE_MYSQL_PASSWORD,
                                db=database,
                                charset="utf8",
                                connect_timeout=10,
                                read_timeout=30,
                            )
                            try:
                                with conn.cursor() as cursor:
                                    cursor.execute(query)
                                    rows = cursor.fetchall()
                                return rows
                            finally:
                                conn.close()
                    except Exception as e:
                        print(f"⚠️ [MySQL SELECT Server {server}] พยายามเชื่อมต่อครั้งที่ {attempt}/3 ล้มเหลว: {e}")
                        if attempt < 3:
                            time.sleep(3)
                return ()

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
    def get_mysql_connection(self, server=1, host="10.130.84.170", database="blue_eye", retries=3, delay=3):
        """คืนค่า (tunnel, conn) เพื่อเอาไปใช้กับ UPDATE/INSERT พร้อมระบบ Retry"""
        for attempt in range(1, retries + 1):
            try:
                tunnel = SSHTunnelForwarder(
                    (SSH_MYSQL_HOST[server], SSH_MYSQL_PORT),
                    ssh_username=SSH_MYSQL_USER[server],
                    ssh_password=SSH_MYSQL_PASSWORD[server],
                    remote_bind_address=(host, BE_MYSQL_PORT),
                    set_keepalive=30
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
                    read_timeout=30,
                )
                return tunnel, conn
            except Exception as e:
                print(f"⚠️ [MySQL Connection Server {server}] พยายามเชื่อมต่อครั้งที่ {attempt}/{retries} ล้มเหลว: {e}")
                if 'tunnel' in locals() and tunnel:
                    try:
                        tunnel.stop()
                    except Exception:
                        pass
                if attempt < retries:
                    time.sleep(delay)
        raise Exception(f"ไม่สามารถเชื่อมต่อ MySQL SSH Tunnel Server {server} ได้หลังจากลองใหม่ครบ {retries} ครั้ง")