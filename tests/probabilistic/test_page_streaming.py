"""Offline checks for streaming and mutable pending pagination."""
import io
import os
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
import ai_sentiment as module


class PageStreamingTests(unittest.TestCase):
    def simulate(self, pending_only=False, save=True, write_limit=None, fail_page=None):
        rows = [{"id": str(i), "sentiment_status": "0"} for i in range(1100)]
        api = module.SentimentAPI(analyzer=Mock())
        events = []
        analyzed = []

        def get(url, params, **kwargs):
            page = params["page"]
            events.append(("fetch", page))
            if page == fail_page:
                return SimpleNamespace(status_code=503)
            current = [r for r in rows if not pending_only or r["sentiment_status"] == "0"]
            start = (page - 1) * 500
            body = {"posts": [r.copy() for r in current[start:start + 500]],
                    "pagination": {"total": len(current), "page_size": 500}}
            return SimpleNamespace(status_code=200, json=lambda: body)

        def process(posts, save_db):
            ids = [r["id"] for r in posts]
            events.append(("analyze", len(ids)))
            analyzed.extend(ids)
            written = ids if write_limit is None else ids[:write_limit]
            if save_db:
                for identifier in written:
                    rows[int(identifier)]["sentiment_status"] = "1"
            return len(written)

        with patch.object(module.requests, "get", side_effect=get), \
                patch.object(module.time, "sleep"), \
                patch.object(api, "_process_pending_posts", side_effect=process), \
                redirect_stdout(io.StringIO()):
            count = api.run("2026-09-01", "2026-09-30", save_db=save)
        return api, events, analyzed, count

    def test_starts_analysis_before_next_fetch_and_handles_stable_pages(self):
        api, events, ids, count = self.simulate()
        self.assertEqual(events[:2], [("fetch", 1), ("analyze", 500)])
        self.assertEqual(len(ids), 1100)
        self.assertEqual(len(set(ids)), 1100)
        self.assertEqual(count, 1100)
        self.assertFalse(api.last_fetch_error)

    def test_pending_queue_shrinks_without_skipping_rows(self):
        api, events, ids, count = self.simulate(pending_only=True)
        self.assertEqual([v for kind, v in events if kind == "fetch"], [1, 1, 1, 1])
        self.assertEqual(len(set(ids)), 1100)
        self.assertEqual(count, 1100)

    def test_dry_run_advances_without_refetching(self):
        _, events, ids, count = self.simulate(save=False)
        self.assertEqual([v for kind, v in events if kind == "fetch"], [1, 2, 3])
        self.assertEqual(len(set(ids)), 1100)

    def test_partial_writes_do_not_repeat_analysis_or_skip_rows(self):
        _, _, ids, _ = self.simulate(pending_only=True, write_limit=1)
        self.assertEqual(len(ids), 1100)
        self.assertEqual(len(set(ids)), 1100)

    def test_failed_writes_advance_without_looping(self):
        _, events, ids, count = self.simulate(pending_only=True, write_limit=0)
        self.assertEqual([v for kind, v in events if kind == "fetch"], [1, 2, 3])
        self.assertEqual(len(set(ids)), 1100)
        self.assertEqual(count, 0)

    def test_later_fetch_failure_preserves_completed_work(self):
        api, _, ids, count = self.simulate(fail_page=2)
        self.assertTrue(api.last_fetch_error)
        self.assertEqual(len(ids), 500)
        self.assertEqual(count, 500)


if __name__ == "__main__":
    unittest.main()
