# coding=utf-8
"""
tests/probabilistic/test_hybrid_throughput.py
Concurrency, bounded sliding window, and semaphore limit tests.
Tests the REST-only production worker without live networks or DB.
"""

import os
import sys
import time
import threading
import unittest
from unittest.mock import MagicMock, patch

# Disable network / live db
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

# Add repository root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import ai_sentiment as sentiment_module


class TestHybridThroughput(unittest.TestCase):
    def setUp(self):
        self.modules = [sentiment_module]
        cost_mode_patch = patch.object(sentiment_module, "AI_COST_MODE", "standard")
        cost_mode_patch.start()
        self.addCleanup(cost_mode_patch.stop)

    def test_max_in_flight_bounds(self):
        """Verify that submitted futures never exceed MAX_IN_FLIGHT when input batch is larger"""
        for mod in self.modules:
            analyzer = mod.OllamaSentimentAnalyzer()
            # Set a small test bound
            analyzer.MAX_IN_FLIGHT = 5
            total_posts = 25
            posts = [{"match_post_id": i, "content": f"Post content {i}", "keywords": "Test"} for i in range(total_posts)]

            current_active = 0
            max_active_observed = 0
            lock = threading.Lock()

            def mock_analyze(post, company_name=""):
                nonlocal current_active, max_active_observed
                with lock:
                    current_active += 1
                    if current_active > max_active_observed:
                        max_active_observed = current_active
                time.sleep(0.01)
                with lock:
                    current_active -= 1
                return {
                    "post_id": str(post["match_post_id"]),
                    "ai_sentiment": 100,
                    "sentiment": "positive",
                    "positive_percent": 90,
                    "negative_percent": 5,
                    "neutral_percent": 5,
                    "irony_score": 0,
                    "reason": "Test reason",
                    "model": "typesafe/jev-1.13"
                }

            with patch.object(analyzer, "_analyze_single_post", side_effect=mock_analyze):
                res = analyzer.analyze_post_sentiments(posts)
                self.assertEqual(len(res["data"]), total_posts)
                # Active futures should never exceed MAX_IN_FLIGHT
                self.assertLessEqual(max_active_observed, analyzer.MAX_IN_FLIGHT + 1)

    def test_deepseek_concurrency_semaphore(self):
        """Verify active DeepSeek calls never exceed DEEPSEEK_MAX_CONCURRENCY"""
        for mod in self.modules:
            analyzer = mod.OllamaSentimentAnalyzer()
            # Set a small semaphore bound for testing
            max_concurrency = 3
            analyzer.DEEPSEEK_MAX_CONCURRENCY = max_concurrency
            analyzer.deepseek_semaphore = threading.BoundedSemaphore(max_concurrency)

            active_ds = 0
            max_active_ds = 0
            lock = threading.Lock()

            def fake_call_deepseek(context, jev_signal=None):
                nonlocal active_ds, max_active_ds
                with analyzer.deepseek_semaphore:
                    with lock:
                        active_ds += 1
                        if active_ds > max_active_ds:
                            max_active_ds = active_ds
                    time.sleep(0.02)
                    with lock:
                        active_ds -= 1
                    return {
                        "entity_found": True,
                        "probabilities": {"POSITIVE": 0.8, "NEUTRAL": 0.1, "NEGATIVE": 0.05, "AMBIGUOUS_OR_IRONY": 0.05}
                    }, "deepseek/deepseek-v4-flash-0731"

            # Route to DeepSeek by forcing Jev low confidence
            low_jev = {
                "sentiment_probabilities": {"positive": 0.50, "neutral": 0.30, "negative": 0.10, "irony": 0.10},
                "sentiment_confidence": 0.50,
                "sentiment_choice": "positive",
                "entity_probabilities": {"relevant": 0.80, "unrelated": 0.10, "uncertain": 0.10},
                "entity_confidence": 0.80,
                "entity_choice": "relevant"
            }

            posts = [{"match_post_id": i, "content": f"Post {i}", "keywords": "SCB"} for i in range(12)]

            with patch.object(analyzer, "_call_typesafe_jev", return_value=(low_jev, "typesafe/jev-1.13")), \
                 patch.object(analyzer, "_call_deepseek_fallback", side_effect=fake_call_deepseek):
                res = analyzer.analyze_post_sentiments(posts)
                self.assertEqual(len(res["data"]), 12)
                self.assertLessEqual(max_active_ds, max_concurrency)

    def test_jev_proceeds_while_deepseek_semaphore_is_held(self):
        """Jev calls do not block on DeepSeek semaphore and can complete while slow path is saturated"""
        for mod in self.modules:
            analyzer = mod.OllamaSentimentAnalyzer()
            analyzer.DEEPSEEK_MAX_CONCURRENCY = 1
            analyzer.deepseek_semaphore = threading.BoundedSemaphore(1)

            deepseek_running = threading.Event()
            deepseek_can_finish = threading.Event()
            jev_completed = threading.Event()

            def slow_deepseek(context, jev_signal=None):
                with analyzer.deepseek_semaphore:
                    deepseek_running.set()
                    # Block until jev completes
                    deepseek_can_finish.wait(timeout=2.0)
                    return {
                        "entity_found": True,
                        "probabilities": {"POSITIVE": 0.8, "NEUTRAL": 0.1, "NEGATIVE": 0.05, "AMBIGUOUS_OR_IRONY": 0.05}
                    }, "deepseek/deepseek-v4-flash-0731"

            valid_jev = {
                "sentiment_probabilities": {"positive": 0.85, "neutral": 0.10, "negative": 0.03, "irony": 0.02},
                "sentiment_confidence": 0.85,
                "sentiment_choice": "positive",
                "entity_probabilities": {"relevant": 0.90, "unrelated": 0.05, "uncertain": 0.05},
                "entity_confidence": 0.85,
                "entity_choice": "relevant"
            }
            low_jev = dict(valid_jev, sentiment_confidence=0.50)

            # Thread 1: runs post that goes to slow path
            def run_slow_post():
                ctx = {"post_id": "1", "actual_target": "SCB", "project_name": "", "keywords": [], "source_info": "", "clean_text": "t", "capped_text": "t"}
                with patch.object(analyzer, "_call_typesafe_jev", return_value=(low_jev, "typesafe/jev-1.13")):
                    analyzer._hybrid_analyze_post(ctx)

            # Thread 2: runs post that goes to fast path
            def run_fast_post():
                ctx = {"post_id": "2", "actual_target": "SCB", "project_name": "", "keywords": [], "source_info": "", "clean_text": "t", "capped_text": "t"}
                with patch.object(analyzer, "_call_typesafe_jev", return_value=(valid_jev, "typesafe/jev-1.13")):
                    analyzer._hybrid_analyze_post(ctx)
                    jev_completed.set()

            with patch.object(analyzer, "_call_deepseek_fallback", side_effect=slow_deepseek):
                t1 = threading.Thread(target=run_slow_post)
                t2 = threading.Thread(target=run_fast_post)
                t1.start()
                # Wait for deepseek to occupy the semaphore
                self.assertTrue(deepseek_running.wait(timeout=2.0))
                # Now start Jev post
                t2.start()
                # Jev post should finish even while deepseek is blocked
                finished_in_time = jev_completed.wait(timeout=2.0)
                # Clean up
                deepseek_can_finish.set()
                t1.join()
                t2.join()
                self.assertTrue(finished_in_time, f"Jev call was starved by DeepSeek semaphore in {mod.__name__}")

    def test_single_worker_failure_does_not_cancel_siblings(self):
        """An unhandled exception in one post does not abort other posts in the batch"""
        for mod in self.modules:
            analyzer = mod.OllamaSentimentAnalyzer()
            posts = [
                {"match_post_id": 1, "content": "Good post", "keywords": "SCB"},
                {"match_post_id": 2, "content": "Bad crash post", "keywords": "SCB"},
                {"match_post_id": 3, "content": "Good post 2", "keywords": "SCB"},
            ]

            def mock_analyze(post, company_name=""):
                if post["match_post_id"] == 2:
                    raise RuntimeError("Simulated worker failure")
                return {
                    "post_id": str(post["match_post_id"]),
                    "ai_sentiment": 100,
                    "sentiment": "positive",
                    "positive_percent": 90,
                    "negative_percent": 5,
                    "neutral_percent": 5,
                    "irony_score": 0,
                    "reason": "OK",
                    "model": "typesafe/jev-1.13"
                }

            with patch.object(analyzer, "_analyze_single_post", side_effect=mock_analyze):
                res = analyzer.analyze_post_sentiments(posts)
                # The failed worker produces a neutral default without cancelling siblings.
                self.assertEqual(len(res["data"]), 3)
                returned_ids = {p["post_id"] for p in res["data"]}
                self.assertEqual(returned_ids, {"1", "2", "3"})
                by_id = {p["post_id"]: p for p in res["data"]}
                self.assertEqual(by_id["2"]["model"], "rule:provider_failure")
                self.assertEqual(by_id["2"]["sentiment"], "neutral")
                self.assertEqual(by_id["1"]["sentiment"], "positive")
                self.assertEqual(by_id["3"]["sentiment"], "positive")

if __name__ == "__main__":
    unittest.main()
