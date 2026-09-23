# coding=utf-8
import os
import sys
import unittest

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add repository root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from project_resolver import ProjectResolver

class TestProjectResolver(unittest.TestCase):
    def setUp(self):
        self.resolver = ProjectResolver()

    def test_cache_loaded(self):
        self.assertGreater(len(self.resolver._projects), 5)
        pea = self.resolver.get_project("01a00f1b-67b7-7071-960b-8d7d4ee7a892")
        self.assertIsNotNone(pea)
        self.assertEqual(pea["name"], "PEA")

    def test_own_brand_with_keyword(self):
        res = self.resolver.resolve_target(
            project_id="01a00f1b-67b7-7071-960b-8d7d4ee7a892",
            keywords=["ค่าไฟ"]
        )
        self.assertEqual(res["project_name"], "PEA")
        self.assertFalse(res["is_rival"])
        self.assertEqual(res["actual_target"], "PEA (หัวข้อ/คีย์เวิร์ด: ค่าไฟ)")

    def test_own_brand_keyword_identical(self):
        res = self.resolver.resolve_target(
            project_id="019ebaa3-40cd-753c-8754-db275e0c5112",
            keywords=["EGAT"]
        )
        self.assertEqual(res["project_name"], "EGAT")
        self.assertEqual(res["actual_target"], "EGAT")

    def test_rival_brand_with_matched_competitor(self):
        res = self.resolver.resolve_target(
            project_id="019e6d78-0ea0-7386-b861-a4e80a68a2b6",
            keywords=["เบียร์ช้าง"],
            content_text="ดื่ม Chang รสชาตินุ่มลิ้นมาก"
        )
        self.assertEqual(res["project_name"], "Boonrawd's Rivals")
        self.assertTrue(res["is_rival"])
        self.assertEqual(res["competitor_matched"], "Chang")
        self.assertIn("Chang", res["actual_target"])
        self.assertIn("Boonrawd's Rivals", res["actual_target"])

    def test_rival_brand_with_general_keyword(self):
        res = self.resolver.resolve_target(
            project_id="019e818a-5583-74b1-b042-6e2794c489ad",
            keywords=["พารากอน"],
            content_text="ไปเดินสยามพารากอนมาวันนี้"
        )
        self.assertEqual(res["project_name"], "Central's Rivals")
        self.assertTrue(res["is_rival"])
        self.assertIn("พารากอน", res["actual_target"])

    def test_unknown_project_fallback(self):
        res = self.resolver.resolve_target(
            project_id="non-existent-uuid",
            keywords=["SCB"],
            company_name="SCB Bank"
        )
        self.assertEqual(res["actual_target"], "SCB")
        self.assertEqual(res["project_name"], "")

    def test_none_project_fallback(self):
        res = self.resolver.resolve_target(
            project_id=None,
            keywords=[],
            company_name="Kasikorn"
        )
        self.assertEqual(res["actual_target"], "Kasikorn")

    def test_try_sync_from_db_connection_closed(self):
        """Verify DB connection is always closed even on error or success"""
        from unittest.mock import MagicMock, patch
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            ("uuid-1", "Project 1", "Desc 1", 1)
        ]

        with patch.dict(os.environ, {
            "DO_MYSQL_HOST": "localhost",
            "DO_MYSQL_USER": "testuser",
            "DO_MYSQL_PASSWORD": "secretpassword"
        }), patch("pymysql.connect", return_value=mock_conn), patch.object(self.resolver, "save_cache"):
            success = self.resolver.try_sync_from_db(force=True)
            self.assertTrue(success)
            self.assertTrue(mock_conn.close.called)
            self.assertIn("uuid-1", self.resolver._projects)

        # Failure path also closes connection
        mock_conn.reset_mock()
        mock_cursor.fetchall.side_effect = RuntimeError("Query error")
        with patch.dict(os.environ, {
            "DO_MYSQL_HOST": "localhost",
            "DO_MYSQL_USER": "testuser",
            "DO_MYSQL_PASSWORD": "secretpassword"
        }), patch("pymysql.connect", return_value=mock_conn):
            success = self.resolver.try_sync_from_db(force=True)
            self.assertFalse(success)
            self.assertTrue(mock_conn.close.called)

    def test_save_cache_atomic(self):
        """Verify save_cache uses atomic replace and cleans up temp files"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            test_cache = os.path.join(tmpdir, "projects.json")
            resolver = ProjectResolver(cache_path=test_cache)
            resolver._projects = {"test-id": {"name": "Test", "description": "Desc", "is_active": True}}
            resolver.save_cache()
            self.assertTrue(os.path.exists(test_cache))
            self.assertFalse(os.path.exists(f"{test_cache}.tmp"))


if __name__ == "__main__":
    unittest.main()

