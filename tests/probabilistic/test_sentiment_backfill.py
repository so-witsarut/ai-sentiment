"""Offline unit tests for sentiment_backfill.py (sentiment_worker.py rules + Ollama)."""

import io
import json
import os
import socket
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import requests

os.environ["PYTHON_DOTENV_DISABLED"] = "1"

with patch.object(socket.socket, "connect", side_effect=AssertionError("network access in offline test")):
    import sentiment_backfill as backfill


GOOD_SR = "ผู้เขียนรายงานดัชนีตลาดหุ้นปรับลดลงตามข่าว"
GOOD_IR = "ผู้เขียนรายงานข้อเท็จจริงเกี่ยวกับตัวเลขตลาดหุ้น"


def valid_item(**over):
    obj = {
        "sentiment_reason": GOOD_SR,
        "positive": 5,
        "neutral": 85,
        "negative": 10,
        "sentiment": "neutral",
        "intent_reason": GOOD_IR,
        "intent": "information",
        "multi_target": False,
        "incident": False,
    }
    obj.update(over)
    return obj


class SentimentBackfillTests(unittest.TestCase):
    def test_validate_verdict_matches_sentiment_worker_rules(self):
        v = backfill.validate_verdict(valid_item(positive=2, neutral=80, negative=16, incident=True), "deepseek-r1:8b")
        self.assertEqual(v.sentiment, "neutral")
        self.assertEqual(v.scores, {"model": "deepseek-r1:8b", "multi_target": False,
                                    "positive": 2, "neutral": 82, "negative": 16})
        self.assertTrue(v.incident)
        self.assertEqual(v.intent, "information")

        bad_cases = [
            valid_item(sentiment="negative"),                     # not strictly largest
            valid_item(positive=50, neutral=50, negative=0, sentiment="neutral"),  # tie forbidden
            valid_item(positive=20, neutral=20, negative=20),     # sum out of 90..110
            valid_item(sentiment_reason="สั้นไป"),                # < 15 chars
            valid_item(sentiment_reason="English reason only long enough"),  # no Thai
            valid_item(sentiment_reason="รายงานเรื่องราวทั้งหมด 中文内容很多"),  # mostly/heavy Chinese leak
            valid_item(intent_reason="ผู้เขียนพูดถึงแบรนด์ลูกค้าในโพสต์นี้"),   # forbidden phrase
            valid_item(incident="yes"),                           # non-bool incident
        ]
        for bad in bad_cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                backfill.validate_verdict(bad, "deepseek-r1:8b")

        # Single stray character in an otherwise valid Thai reason is tolerated like sentiment_worker.py
        tolerated = backfill.validate_verdict(
            valid_item(sentiment_reason="บทกวีนี้เน้นสอนใจและคำคม ไม่มีการวิพ论หรือแสดงความรู้สึกต่อแบรนด์"),
            "deepseek-r1:8b",
        )
        self.assertEqual(tolerated.sentiment, "neutral")

    def test_client_retries_generate_json_with_varied_seeds(self):
        client = backfill.OllamaWorkerClient(model="deepseek-r1:8b", retries=3, think=False)
        calls = []

        def fake_post(url, json=None, timeout=None):
            calls.append((url, json, timeout))
            if len(calls) == 1:
                raise requests.exceptions.ReadTimeout("simulated timeout")
            if len(calls) == 2:
                return SimpleNamespace(status_code=200, text="", json=lambda: {"response": "{invalid"})
            body = json_module.dumps(valid_item(), ensure_ascii=False)
            return SimpleNamespace(status_code=200, text="", json=lambda: {"response": body})

        json_module = json
        with patch.object(client.session, "post", side_effect=fake_post), \
             patch.object(client, "wait_until_healthy", return_value=True), \
             patch.object(client, "log_vram_status_once"), \
             redirect_stdout(io.StringIO()):
            verdict, err = client.analyse_post("SET ปิดลดลง 4.10 จุด")

        self.assertIsNotNone(verdict)
        self.assertEqual(err, "")
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(u.endswith("/api/generate") for u, _, _ in calls))
        self.assertEqual([c[1]["options"]["seed"] for c in calls], [42, 43, 44])
        self.assertEqual([c[1]["options"]["temperature"] for c in calls], [0.0, 0.2, 0.4])
        self.assertIs(calls[0][1]["think"], False)
        self.assertEqual(calls[0][2], (10, 120))

    def test_live_mode_claims_dedupes_post_id_and_never_writes_rule_provider_failure(self):
        rest = backfill.BlueEyeRestClient(base_url="https://api.example/api/v1", token="tok")
        ollama = backfill.OllamaWorkerClient(model="deepseek-r1:8b", retries=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            fail_log = os.path.join(tmpdir, "failures.jsonl")
            runner = backfill.SentimentBackfillRunner(
                rest=rest,
                ollama=ollama,
                batch_size=10,
                fetch_limit=50,
                save_db=True,
                fail_log_path=fail_log,
            )

            claimed_batches = [
                [
                    {"match_post_id": "M1", "post_id": "P1", "content": "ข่าวตลาดหุ้นรายงานตัวเลขเปิดตลาด"},
                    {"match_post_id": "M2", "post_id": "P1", "content": "ข่าวตลาดหุ้นรายงานตัวเลขเปิดตลาด"},
                    {"match_post_id": "M3", "post_id": "P2", "content": "โพสต์ที่โมเดลตอบไม่ได้"},
                ],
                [],
            ]
            verdict_p1 = backfill.validate_verdict(valid_item(incident=True), "deepseek-r1:8b")

            with patch.object(rest, "fetch_pending_claimed", side_effect=claimed_batches) as fetch_mock, \
                 patch.object(rest, "write_results", return_value=2) as write_mock, \
                 patch.object(ollama, "analyse_post", side_effect=[(verdict_p1, ""), (None, "timeout")]) as analyse_mock, \
                 redirect_stdout(io.StringIO()):
                rc = runner.run("2026-11-01", "2026-11-16")

            self.assertEqual(rc, 0)
            self.assertEqual(fetch_mock.call_count, 2)
            # P1 analysed once even though it had M1 and M2
            self.assertEqual(analyse_mock.call_count, 2)
            self.assertEqual(runner.total_posts_done, 1)
            self.assertEqual(runner.total_rows_written, 2)
            self.assertEqual(runner.total_failed_posts, 1)

            sent = write_mock.call_args.args[0]
            self.assertEqual([item["match_post_id"] for item in sent], ["M1", "M2"])
            for item in sent:
                self.assertEqual(item["sentiment"], "neutral")
                self.assertEqual(item["sentiment_reason"], GOOD_SR)
                self.assertEqual(item["intent"], "information")
                self.assertEqual(item["intent_reason"], GOOD_IR)
                self.assertTrue(item["incident"])
                self.assertEqual(item["sentiment_scores"]["model"], "deepseek-r1:8b")

            with open(fail_log, "r", encoding="utf-8") as fh:
                logged = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(logged), 1)
            self.assertEqual(logged[0]["post_id"], "P2")
            self.assertEqual(logged[0]["match_post_ids"], ["M3"])

    def test_missing_intent_omits_sentiment_half_when_sentiment_exists(self):
        rest = backfill.BlueEyeRestClient(base_url="https://api.example/api/v1", token="tok")
        ollama = backfill.OllamaWorkerClient(model="deepseek-r1:8b")
        runner = backfill.SentimentBackfillRunner(
            rest=rest, ollama=ollama, save_db=True, missing="intent"
        )
        rows = [
            {"match_post_id": "M1", "post_id": "P1", "content": "สอบถามค่าบริการรายเดือนหน่อยครับ", "sentiment": "neutral"},
        ]
        verdict = backfill.validate_verdict(valid_item(intent="enquiry"), "deepseek-r1:8b")

        with patch.object(rest, "fetch_pending_claimed", side_effect=[rows, []]), \
             patch.object(rest, "write_results", return_value=1) as write_mock, \
             patch.object(ollama, "analyse_post", return_value=(verdict, "")), \
             redirect_stdout(io.StringIO()):
            runner.run("2026-11-01", "2026-11-16")

        sent = write_mock.call_args.args[0]
        self.assertEqual(sent, [{"match_post_id": "M1", "intent": "enquiry", "intent_reason": GOOD_IR}])

    def test_dry_run_reads_posts_endpoint_and_escalate_keeps_unsure_on_failure(self):
        rest = backfill.BlueEyeRestClient(base_url="https://api.example/api/v1", token="tok")
        ollama = backfill.OllamaWorkerClient(model="deepseek-r1:8b")
        runner = backfill.SentimentBackfillRunner(
            rest=rest, ollama=ollama, save_db=False, escalate_model="qwen3.8:latest", limit=1
        )
        unsure = backfill.validate_verdict(
            valid_item(positive=45, neutral=50, negative=5, sentiment="neutral"), "deepseek-r1:8b"
        )
        page_items = [
            {"match_post_id": "M1", "post_id": "P1", "content": "โพสต์ทดสอบ"},
            {"match_post_id": "M2", "post_id": "P2", "content": "เกินลิมิต"},
        ]

        with patch.object(rest, "fetch_posts_page", return_value=(2, page_items)) as posts_mock, \
             patch.object(rest, "write_results") as write_mock, \
             patch.object(ollama, "analyse_post", side_effect=[(unsure, ""), (None, "escalate down")]) as analyse_mock, \
             redirect_stdout(io.StringIO()):
            rc = runner.run("2026-11-01", "2026-11-16")

        self.assertEqual(rc, 0)
        posts_mock.assert_called_once()
        write_mock.assert_not_called()
        self.assertEqual(analyse_mock.call_count, 2)
        self.assertEqual(runner.total_posts_done, 1)
        self.assertEqual(runner.total_escalated, 0)

    def test_incomplete_mode_walks_cursors_and_posts_replace_payload(self):
        rest = backfill.BlueEyeRestClient(base_url="https://api.example/api/v1", token="tok")
        ollama = backfill.OllamaWorkerClient(model="deepseek-r1:8b", retries=1)
        runner = backfill.SentimentBackfillRunner(
            rest=rest,
            ollama=ollama,
            save_db=True,
            missing="incomplete",
            organization_id="org-1",
        )
        verdict = backfill.validate_verdict(
            valid_item(
                positive=5, neutral=15, negative=80, sentiment="negative",
                intent="complaint", incident=False,
            ),
            "deepseek-r1:8b",
        )
        pages = [
            ([], "c1"),  # empty page with non-null cursor must keep going
            (
                [
                    {"post_id": "1234567890_987654", "content": "สินค้าเสียหลังใช้ 2 วัน"},
                    {"post_id": "1234567890_987654", "content": "สินค้าเสียหลังใช้ 2 วัน"},  # dup across projects
                ],
                None,
            ),
        ]

        with patch.object(rest, "fetch_incomplete_page", side_effect=pages) as inc_mock, \
             patch.object(rest, "replace_results", return_value=3) as replace_mock, \
             patch.object(rest, "write_results") as write_mock, \
             patch.object(ollama, "analyse_post", return_value=(verdict, "")) as analyse_mock, \
             redirect_stdout(io.StringIO()):
            rc = runner.run("2026-11-01", "2026-11-16")

        self.assertEqual(rc, 0)
        self.assertEqual(inc_mock.call_count, 2)
        self.assertEqual(inc_mock.call_args_list[0].kwargs["cursor"], None)
        self.assertEqual(inc_mock.call_args_list[1].kwargs["cursor"], "c1")
        self.assertEqual(inc_mock.call_args_list[0].kwargs["organization_id"], "org-1")
        self.assertEqual(analyse_mock.call_count, 1)
        write_mock.assert_not_called()
        replace_mock.assert_called_once()

        payload = replace_mock.call_args.args[0]
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["post_id"], "1234567890_987654")
        self.assertNotIn("match_post_id", payload[0])
        self.assertEqual(payload[0]["sentiment"], "negative")
        self.assertEqual(payload[0]["intent"], "complaint")
        self.assertIs(payload[0]["incident"], False)
        self.assertEqual(payload[0]["sentiment_scores"], {
            "model": "deepseek-r1:8b",
            "multi_target": False,
            "positive": 5,
            "neutral": 15,
            "negative": 80,
        })
        self.assertEqual(runner.total_posts_done, 1)
        self.assertEqual(runner.total_rows_written, 3)

    def test_incomplete_dry_run_reads_incomplete_without_posting_replace(self):
        rest = backfill.BlueEyeRestClient(base_url="https://api.example/api/v1", token="tok")
        ollama = backfill.OllamaWorkerClient(model="deepseek-r1:8b")
        runner = backfill.SentimentBackfillRunner(
            rest=rest, ollama=ollama, save_db=False, missing="incomplete", limit=1
        )
        verdict = backfill.validate_verdict(valid_item(), "deepseek-r1:8b")

        with patch.object(rest, "fetch_incomplete_page",
                          return_value=([{"post_id": "P_INC_1", "content": "ทดสอบข้อความ"}], "c2")) as inc_mock, \
             patch.object(rest, "fetch_posts_page") as posts_mock, \
             patch.object(rest, "replace_results") as replace_mock, \
             patch.object(ollama, "analyse_post", return_value=(verdict, "")), \
             redirect_stdout(io.StringIO()):
            rc = runner.run("2026-11-01", "2026-11-16")

        self.assertEqual(rc, 0)
        inc_mock.assert_called_once()
        posts_mock.assert_not_called()
        replace_mock.assert_not_called()
        self.assertEqual(runner.total_posts_done, 1)

    def test_parse_args_incomplete_and_organization_validation(self):
        args = backfill.parse_args([
            "--from-date", "2026-11-01",
            "--to-date", "2026-11-16",
            "--missing", "incomplete",
            "--organization-id", "org-123",
        ])
        self.assertEqual(args.missing, "incomplete")
        self.assertEqual(args.organization_id, "org-123")

        for bad_argv in (
            ["--from-date", "2026-11-01", "--to-date", "2026-11-16", "--missing", "incomplete",
             "--project-id", "p1", "--organization-id", "o1"],
            ["--from-date", "2026-11-01", "--to-date", "2026-11-16", "--missing", "sentiment",
             "--organization-id", "o1"],
        ):
            with self.subTest(bad_argv=bad_argv), self.assertRaises(SystemExit), \
                 redirect_stdout(io.StringIO()):
                backfill.parse_args(bad_argv)


if __name__ == "__main__":
    unittest.main()
