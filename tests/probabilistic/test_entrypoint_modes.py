"""Offline checks for the REST-only candidate entrypoint."""

import importlib.util
import io
import os
import socket
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import Mock, call, patch

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["OPENROUTER_API_KEY"] = "offline-test-key"

with patch.object(socket.socket, "connect", side_effect=AssertionError("network access in offline test")):
    import ai_sentiment as sentiment_module


class TestEntrypointModes(unittest.TestCase):
    def test_rest_does_not_persist_unresolved_target_as_neutral(self):
        posts = [
            {"match_post_id": "unknown", "project_id": "missing", "content": "keyword", "keywords": ["keyword"]},
            {"match_post_id": "known", "project_id": "known", "content": "brand", "keywords": ["brand"]},
        ]
        analyzer = Mock()
        analyzer.analyze_post_sentiments.return_value = {"data": [
            {"post_id": "unknown", "ai_sentiment": 0, "sentiment": "neutral",
             "neutral_percent": 100, "model": "rule:unresolved_target"},
            {"post_id": "known", "ai_sentiment": 100, "sentiment": "positive",
             "positive_percent": 90, "neutral_percent": 10, "model": "jev"},
        ]}
        api = sentiment_module.SentimentAPI(analyzer=analyzer)
        with patch.object(api, "fetch_pending", return_value=posts), \
                patch.object(api, "bulk_update", return_value=1) as update, \
                patch.object(sentiment_module, "GLOBAL_PROJECT_RESOLVER", None), \
                redirect_stdout(io.StringIO()):
            updated = api.run("2026-09-23", "2026-09-24", save_db=True)
        self.assertEqual(updated, 1)
        sent = update.call_args.args[0]
        self.assertEqual([item["match_post_id"] for item in sent], ["known"])

    def test_mode_argument_defaults_to_rest_and_rejects_db_modes(self):
        self.assertEqual(sentiment_module.parse_run_mode([]), "rest")
        self.assertEqual(sentiment_module.parse_run_mode(["--mode", "rest"]), "rest")
        for mode in ("db", "both", "unknown"):
            with self.subTest(mode=mode), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                sentiment_module.parse_run_mode(["--mode", mode])

    def test_only_rest_client_is_created(self):
        analyzer = object()
        with patch.object(sentiment_module, "SentimentAPI") as api_class:
            api = sentiment_module.create_apps_for_mode("rest", analyzer)
        api_class.assert_called_once_with(analyzer=analyzer)
        self.assertIs(api, api_class.return_value)
        for mode in ("db", "both"):
            with self.assertRaises(ValueError):
                sentiment_module.create_apps_for_mode(mode, analyzer)
        self.assertIs(sentiment_module.sentiment, sentiment_module.SentimentAPI)

    def test_import_does_not_require_database_modules_or_connect(self):
        blocked = {name: None for name in ("connection", "project_resolver", "pymysql", "pymongo")}
        spec = importlib.util.spec_from_file_location("rest_only_import_probe", sentiment_module.__file__)
        candidate = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", blocked), \
             patch.object(socket.socket, "connect", side_effect=AssertionError("unexpected network access")):
            spec.loader.exec_module(candidate)
        self.assertFalse(hasattr(candidate, "SentimentDB"))

    def test_rest_only_repeats_after_empty_queue_and_error(self):
        api = Mock()
        api.run.side_effect = [0, RuntimeError("temporary REST error")]
        with patch.object(sentiment_module.time, "sleep", side_effect=[None, KeyboardInterrupt]) as sleep:
            with redirect_stdout(io.StringIO()):
                sentiment_module.run_main_loop(api, True, 5)
        self.assertEqual(api.run.call_count, 2)
        self.assertTrue(all(c.kwargs == {"save_db": True} for c in api.run.call_args_list))
        self.assertEqual(sleep.call_args_list, [call(5), call(5)])

if __name__ == "__main__":
    unittest.main()
