"""The REST worker reuses only valid decisions for identical analysis inputs."""

import io
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import ai_sentiment as module


def decision(post, model="jev"):
    return {"post_id": str(post["match_post_id"]), "ai_sentiment": 0,
            "sentiment": "neutral", "neutral_percent": 100,
            "entity_found": False, "model": model}


class TestResultCache(unittest.TestCase):
    def setUp(self):
        cache_dir = Path(module.__file__).parent / ".cache"
        cache_dir.mkdir(exist_ok=True)
        self.path = str(cache_dir / f"test_results_{uuid.uuid4().hex}.sqlite3")
        self.addCleanup(lambda: Path(self.path).unlink(missing_ok=True))
        self.posts = [
            {"match_post_id": "a", "project_name": "BLCP", "content": "BLCP ค่าไฟ",
             "keywords": ["ค่าไฟ"], "feed_link": "https://example.org/post"},
            {"match_post_id": "b", "project_name": "BLCP", "content": "BLCP ค่าไฟ",
             "keywords": ["ค่าไฟ"], "feed_link": "https://example.org/post"},
        ]

    def test_same_batch_and_next_process_reuse_without_rest_writes(self):
        analyzer = Mock()
        analyzer.analyze_post_sentiments.side_effect = lambda posts: {"data": [decision(p) for p in posts]}
        api = module.SentimentAPI(analyzer=analyzer, result_cache=module.AnalysisResultCache(self.path))
        with patch.object(api, "fetch_pending", return_value=self.posts), \
             patch.object(api, "bulk_update") as update, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(api.run("2026-09-25", "2026-09-25", save_db=False), 2)
        update.assert_not_called()
        self.assertEqual(analyzer.analyze_post_sentiments.call_count, 1)
        self.assertEqual(len(analyzer.analyze_post_sentiments.call_args.args[0]), 1)
        api.result_cache.db.close()

        second_analyzer = Mock()
        second_api = module.SentimentAPI(
            analyzer=second_analyzer, result_cache=module.AnalysisResultCache(self.path))
        with patch.object(second_api, "fetch_pending", return_value=self.posts), \
             patch.object(second_api, "bulk_update", return_value=2) as update, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(second_api.run("2026-09-25", "2026-09-25", save_db=True), 2)
        second_analyzer.analyze_post_sentiments.assert_not_called()
        self.assertEqual([r["match_post_id"] for r in update.call_args.args[0]], ["a", "b"])
        second_api.result_cache.db.close()

    def test_target_and_source_changes_require_new_decisions(self):
        analyzer = Mock()
        analyzer.analyze_post_sentiments.side_effect = lambda posts: {"data": [decision(p) for p in posts]}
        api = module.SentimentAPI(analyzer=analyzer, result_cache=module.AnalysisResultCache(self.path))
        posts = [self.posts[0],
                 {**self.posts[0], "match_post_id": "ignored-target-fields",
                  "actual_target": "Wrong target", "company_name": "Wrong company"},
                 {**self.posts[0], "match_post_id": "different-target", "project_name": "EGAT"},
                 {**self.posts[0], "match_post_id": "different-source", "feed_link": "https://example.org/other"}]
        with patch.object(api, "fetch_pending", return_value=posts), redirect_stdout(io.StringIO()):
            self.assertEqual(api.run("2026-09-25", "2026-09-25", save_db=False), 4)
        self.assertEqual(len(analyzer.analyze_post_sentiments.call_args.args[0]), 3)
        api.result_cache.db.close()

    def test_provider_failure_neutral_is_submitted_and_reused_after_rest_failure(self):
        analyzer = Mock()
        analyzer.analyze_post_sentiments.side_effect = lambda posts: {
            "data": [decision(p, model="rule:provider_failure") for p in posts]}
        api = module.SentimentAPI(analyzer=analyzer, result_cache=module.AnalysisResultCache(self.path))
        with patch.object(api, "fetch_pending", return_value=self.posts), \
             patch.object(api, "bulk_update", return_value=0) as update, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(api.run("2026-09-25", "2026-09-25", save_db=True), 0)
        self.assertEqual(analyzer.analyze_post_sentiments.call_count, 1)
        sent = update.call_args.args[0]
        self.assertEqual([row["match_post_id"] for row in sent], ["a", "b"])
        self.assertTrue(all(row["sentiment"] == "neutral" for row in sent))
        self.assertTrue(all(row["sentiment_scores"]["neutral"] == 100 for row in sent))
        self.assertTrue(all("intent" not in row for row in sent))
        api.result_cache.db.close()

        second_analyzer = Mock()
        second_api = module.SentimentAPI(
            analyzer=second_analyzer, result_cache=module.AnalysisResultCache(self.path))
        with patch.object(second_api, "fetch_pending", return_value=self.posts), \
             patch.object(second_api, "bulk_update", return_value=2) as update, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(second_api.run("2026-09-25", "2026-09-25", save_db=True), 2)
        second_analyzer.analyze_post_sentiments.assert_not_called()
        self.assertEqual([row["match_post_id"] for row in update.call_args.args[0]], ["a", "b"])
        second_api.result_cache.db.close()

    def test_failed_rest_write_reuses_decision_on_retry(self):
        analyzer = Mock()
        analyzer.analyze_post_sentiments.side_effect = lambda posts: {"data": [decision(p) for p in posts]}
        api = module.SentimentAPI(analyzer=analyzer, result_cache=module.AnalysisResultCache(self.path))
        with patch.object(api, "fetch_pending", return_value=[self.posts[0]]), \
             patch.object(api, "bulk_update", side_effect=[0, 1]) as update, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(api.run("2026-09-25", "2026-09-25", save_db=True), 0)
            self.assertEqual(api.run("2026-09-25", "2026-09-25", save_db=True), 1)
        self.assertEqual(analyzer.analyze_post_sentiments.call_count, 1)
        self.assertEqual(update.call_count, 2)
        api.result_cache.db.close()


if __name__ == "__main__":
    unittest.main()
