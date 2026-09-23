# coding=utf-8
import os
import sys
import json
import time
import re
from typing import Dict, Any, Optional, List

# Reconfigure stdout for UTF-8 on Windows
try:
    if hasattr(sys.stdout, 'reconfigure') and sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

DEFAULT_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "projects_cache.json")


class ProjectResolver:
    """
    Project Resolver & Entity Context Manager
    Loads project records (id, name, description, is_active) from:
      1. DigitalOcean Managed MySQL (if DO_MYSQL_HOST / credentials exist)
      2. Local JSON Cache (projects_cache.json) as safe fallback
    """

    def __init__(self, cache_path: str = DEFAULT_CACHE_PATH, ttl_seconds: int = 3600):
        self.cache_path = cache_path
        self.ttl_seconds = ttl_seconds
        self._last_sync_time = 0
        self._projects: Dict[str, Dict[str, Any]] = {}
        self.load_cache()
        self.try_sync_from_db()

    def load_cache(self) -> None:
        """Load projects from local JSON cache file."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._projects = data
                        print(f"📦 [ProjectResolver] โหลดโปรเจกต์จากแคชสำเร็จ ({len(self._projects)} รายการ)")
            except Exception as e:
                print(f"⚠️ [ProjectResolver] ไม่สามารถอ่านไฟล์แคช {self.cache_path}: {e}")

    def save_cache(self) -> None:
        """Persist memory cache to JSON file atomically."""
        temp_path = f"{self.cache_path}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._projects, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.cache_path)
        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            print(f"⚠️ [ProjectResolver] บันทึกไฟล์แคชไม่สำเร็จ: {e}")

    def try_sync_from_db(self, force: bool = False) -> bool:
        """
        Attempt to sync projects from DO MySQL if configured.
        Requires DO_MYSQL_HOST, DO_MYSQL_USER, DO_MYSQL_PASSWORD.
        """
        now = time.time()
        if not force and (now - self._last_sync_time < self.ttl_seconds) and self._projects:
            return True

        do_host = os.environ.get("DO_MYSQL_HOST")
        do_user = os.environ.get("DO_MYSQL_USER")
        do_pass = os.environ.get("DO_MYSQL_PASSWORD")
        do_port = int(os.environ.get("DO_MYSQL_PORT", 25060))
        do_db = os.environ.get("DO_MYSQL_DB", "blueeye")

        if not (do_host and do_user and do_pass):
            # Not configured for direct DB sync, will rely on local JSON cache
            return False

        conn = None
        try:
            import pymysql
            conn = pymysql.connect(
                host=do_host,
                port=do_port,
                user=do_user,
                passwd=do_pass,
                db=do_db,
                charset="utf8mb4",
                connect_timeout=10,
                read_timeout=15,
                ssl={"ssl_mode": "REQUIRED"} if do_port == 25060 else None
            )
            with conn.cursor() as cur:
                cur.execute("SELECT id, name, description, is_active FROM projects")
                rows = cur.fetchall()
                new_map = {}
                for row in rows:
                    p_id = str(row[0]).strip()
                    p_name = str(row[1]).strip() if row[1] else ""
                    p_desc = str(row[2]).strip() if row[2] else None
                    is_active = bool(row[3]) if row[3] is not None else True
                    new_map[p_id] = {
                        "name": p_name,
                        "description": p_desc,
                        "is_active": is_active
                    }
                if new_map:
                    self._projects = new_map
                    self._last_sync_time = now
                    self.save_cache()
                    print(f"✅ [ProjectResolver] ซิงค์โปรเจกต์จากฐานข้อมูลสำเร็จ ({len(new_map)} รายการ)")
                    return True
        except Exception as e:
            print(f"⚠️ [ProjectResolver] ซิงค์จาก DB ไม่สำเร็จ (ใช้แคชเดิมแทน): {e}")
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
        return False

    def get_project(self, project_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Look up project by UUID string."""
        if not project_id:
            return None
        return self._projects.get(str(project_id).strip())

    def resolve_target(
        self,
        project_id: Optional[str],
        keywords: Optional[List[str]] = None,
        company_name: str = "",
        content_text: str = ""
    ) -> Dict[str, Any]:
        """
        Resolve complete Target Entity and Project Context.
        Returns a dict:
          - actual_target: Formatted target for LLM analysis & synthetic reasoning
          - project_name: Name of the project (e.g. "Boonrawd", "Central", "PEA")
          - project_desc: Project description or competitor list (e.g. "ThaiBev , Chang , CP")
          - is_rival: True if this is a competitor-tracking project
          - competitor_matched: Name of matched competitor brand if applicable
        """
        keywords = keywords or []
        kw_str = ", ".join(keywords) if keywords else ""
        
        project = self.get_project(project_id)
        if not project:
            # Fallback when project is not in registry
            if keywords:
                actual_target = kw_str
            elif company_name:
                actual_target = company_name
            else:
                actual_target = "the Target Entity"
            return {
                "actual_target": actual_target,
                "project_name": "",
                "project_desc": "",
                "is_rival": False,
                "competitor_matched": ""
            }

        proj_name = project.get("name") or ""
        proj_desc = project.get("description") or ""

        # Check if project is tracking competitors/rivals
        is_rival = (
            "rival" in proj_name.lower()
            or "คู่แข่ง" in proj_name.lower()
            or bool(proj_desc and any(sep in proj_desc for sep in [",", "/"]))
        )

        competitor_matched = ""
        if is_rival and proj_desc:
            # Split candidate competitor entities
            candidates = [c.strip() for c in re.split(r"[,/]+", proj_desc) if c.strip()]
            
            # Check if any candidate is in keywords or mentioned in text
            search_corpus = f"{kw_str} {content_text}".lower()
            for cand in candidates:
                if cand.lower() in search_corpus:
                    competitor_matched = cand
                    break

        if is_rival:
            if competitor_matched:
                target = f"{competitor_matched} (กลุ่มคู่แข่ง: {proj_name})"
            elif keywords:
                target = f"{kw_str} (กลุ่มคู่แข่ง: {proj_name})"
            else:
                target = f"{proj_name} ({proj_desc})" if proj_desc else proj_name
        else:
            # Own brand project
            if keywords:
                # If keyword is identical to brand name, don't repeat
                if kw_str.strip().lower() == proj_name.strip().lower():
                    target = proj_name
                else:
                    target = f"{proj_name} (หัวข้อ/คีย์เวิร์ด: {kw_str})"
            else:
                target = proj_name

        return {
            "actual_target": target,
            "project_name": proj_name,
            "project_desc": proj_desc,
            "is_rival": is_rival,
            "competitor_matched": competitor_matched
        }


# Global singleton instance
GLOBAL_PROJECT_RESOLVER = ProjectResolver()

