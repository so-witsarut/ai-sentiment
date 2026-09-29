"""Offline behavior checks for per-post provider scheduling."""

import json
import io
import os
import threading
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import ai_sentiment as module


VALID_ANSWER = {
    "probabilities": {"POSITIVE": 0.1, "NEGATIVE": 0.0,
                      "NEUTRAL": 0.9, "AMBIGUOUS_OR_IRONY": 0.0},
    "entity_found": True,
    "intent": "information",
}


def context(profile):
    return {"post_id": "1", "actual_target": "SEC", "sentiment_target": "SEC",
            "project_name": "SEC", "project_desc": "", "keywords": [],
            "source_info": "Source=offline", "clean_text": "SEC update",
            "capped_text": "SEC update", "_provider_profile": profile}


class ScheduleTests(unittest.TestCase):
    def test_thailand_time_boundaries_ignore_pc_timezone(self):
        with patch.object(module, "PROVIDER_SCHEDULE_ENABLED", True), \
             patch.object(module, "OPENROUTER_SCHEDULE_START", 8 * 60), \
             patch.object(module, "OPENROUTER_SCHEDULE_END", 22 * 60):
            expected = {(7, 59): "overnight", (8, 0): "daytime",
                        (21, 59): "daytime", (22, 0): "overnight"}
            for (hour, minute), profile in expected.items():
                bangkok = datetime(2026, 9, 29, hour, minute, tzinfo=module.THAI_TIMEZONE)
                self.assertEqual(module.current_provider_profile(bangkok), profile)
                self.assertEqual(module.current_provider_profile(
                    bangkok.astimezone(timezone(timedelta(hours=-4)))), profile)

    def test_model_order_preserves_order_inside_each_provider(self):
        models = ["local:a", "api:gemma-4-26b-a4b-it", "openrouter:other/x",
                  "api:gemma-4-31b-it", "local:b", "api:gemini-2.5-flash"]
        self.assertEqual(module.ordered_validation_models("overnight", models=models),
                         [models[1], models[3], models[5], models[0], models[4], models[2]])
        self.assertEqual(module.ordered_validation_models("daytime", models=models),
                         [models[2], models[1], models[3], models[5], models[0], models[4]])

    def test_legacy_mode_uses_scheduled_order_without_local_triage(self):
        analyzer = module.OllamaSentimentAnalyzer()
        post = {"match_post_id": "1", "content": "SEC update", "project_name": "SEC",
                "_provider_profile": "daytime"}
        with patch.object(module, "ENABLE_JEV_HYBRID", False), \
             patch.object(module, "ENABLE_PROBABILISTIC_MODE", True), \
             patch.object(module, "BYPASS_LOCAL_TRIAGE", False), \
             patch.object(analyzer, "_triage_post") as triage, \
             patch.object(analyzer, "_probabilistic_analyze_post", return_value={"model": "ok"}) as analyze:
            result = analyzer._analyze_single_post(post)
        self.assertEqual(result, {"model": "ok"})
        self.assertEqual(analyze.call_args.kwargs["provider_profile"], "daytime")
        triage.assert_not_called()
        self.assertEqual(module.scheduled_legacy_models("overnight", ["local:x", "api:y"]),
                         ["api:y", "local:x", f"openrouter:{module.DEEPSEEK_MODEL}"])

    def test_daytime_openrouter_then_gemini_on_bad_response(self):
        analyzer = module.OllamaSentimentAnalyzer()
        calls = []

        def post(url, **kwargs):
            calls.append("openrouter")
            return SimpleNamespace(status_code=200, json=lambda: {"choices": []})

        def gemini(model, *args, **kwargs):
            calls.append(model)
            return VALID_ANSWER

        with patch.dict(os.environ, {"ENABLE_OPENROUTER": "true", "ENABLE_GEMINI": "true",
                                     "ENABLE_OLLAMA": "true", "VALIDATION_MODELS": "local:cloud,api:g1,api:g2"}), \
             patch.object(module, "OPENROUTER_API_KEY", "offline-key"), \
             patch.object(analyzer, "get_session", return_value=SimpleNamespace(post=post)), \
             patch.object(analyzer, "_call_gemini_api", side_effect=gemini), \
             patch.object(analyzer, "_call_ollama_generic") as ollama:
            result, model = analyzer._call_deepseek_fallback(context("daytime"))
        self.assertEqual(model, "g1")
        self.assertIsNotNone(result)
        self.assertEqual(calls, ["openrouter", "g1"])
        ollama.assert_not_called()

    def test_overnight_gemini_ollama_then_openrouter(self):
        analyzer = module.OllamaSentimentAnalyzer()
        calls = []

        def post(url, **kwargs):
            calls.append("openrouter")
            body = {"choices": [{"message": {"content": json.dumps(VALID_ANSWER)}}]}
            return SimpleNamespace(status_code=200, json=lambda: body)

        def gemini(model, *args, **kwargs):
            calls.append(model)
            return None

        def ollama(model, *args, **kwargs):
            calls.append(model)
            return None

        with patch.dict(os.environ, {"ENABLE_OPENROUTER": "true", "ENABLE_GEMINI": "true",
                                     "ENABLE_OLLAMA": "true", "VALIDATION_MODELS": "local:cloud,api:g1,api:g2"}), \
             patch.object(module, "OPENROUTER_API_KEY", "offline-key"), \
             patch.object(analyzer, "get_session", return_value=SimpleNamespace(post=post)), \
             patch.object(analyzer, "_call_gemini_api", side_effect=gemini), \
             patch.object(analyzer, "_call_ollama_generic", side_effect=ollama):
            result, model = analyzer._call_deepseek_fallback(context("overnight"))
        self.assertEqual(model, module.DEEPSEEK_MODEL)
        self.assertIsNotNone(result)
        self.assertEqual(calls, ["g1", "g2", "local:cloud", "openrouter"])

    def test_overnight_openrouter_validation_model_runs_after_deepseek_failure(self):
        analyzer = module.OllamaSentimentAnalyzer()
        calls = []

        def post(url, **kwargs):
            model = kwargs["json"]["model"]
            calls.append(model)
            if model == module.DEEPSEEK_MODEL:
                return SimpleNamespace(status_code=500, text="offline failure", headers={})
            body = {"choices": [{"message": {"content": json.dumps(VALID_ANSWER)}}]}
            return SimpleNamespace(status_code=200, json=lambda: body, headers={})

        with patch.dict(os.environ, {"ENABLE_OPENROUTER": "true", "ENABLE_GEMINI": "true",
                                     "ENABLE_OLLAMA": "false",
                                     "VALIDATION_MODELS": "api:g1,openrouter:other/x"}), \
             patch.object(module, "OPENROUTER_API_KEY", "offline-key"), \
             patch.object(module, "DEEPSEEK_MAX_RETRIES", 0), \
             patch.object(analyzer, "get_session", return_value=SimpleNamespace(post=post)), \
             patch.object(analyzer, "_call_gemini_api", return_value=None) as gemini, \
             redirect_stdout(io.StringIO()):
            result, model = analyzer._call_deepseek_fallback(context("overnight"))
        self.assertIsNotNone(result)
        self.assertEqual(model, "other/x")
        gemini.assert_called_once()
        self.assertEqual(calls, [module.DEEPSEEK_MODEL, "other/x"])

    def test_disabled_providers_are_never_called_even_when_scheduled(self):
        analyzer = module.OllamaSentimentAnalyzer()
        with patch.dict(os.environ, {"ENABLE_OPENROUTER": "false", "ENABLE_GEMINI": "true",
                                     "ENABLE_OLLAMA": "false", "VALIDATION_MODELS": "local:cloud,api:g1,openrouter:other/x"}), \
             patch.object(analyzer, "_call_gemini_api", return_value=VALID_ANSWER) as gemini, \
             patch.object(analyzer, "_call_ollama_generic") as ollama, \
             patch.object(analyzer, "get_session") as session:
            result, model = analyzer._call_deepseek_fallback(context("overnight"))
        self.assertIsNotNone(result)
        self.assertEqual(model, "g1")
        gemini.assert_called_once()
        ollama.assert_not_called()
        session.assert_not_called()

    def test_overnight_skips_jev_before_validation(self):
        analyzer = module.OllamaSentimentAnalyzer()
        with patch.object(analyzer, "_call_typesafe_jev") as jev, \
             patch.object(analyzer, "_call_deepseek_fallback",
                          return_value=(VALID_ANSWER, "g1")) as fallback:
            result = analyzer._hybrid_analyze_post(context("overnight"))
        jev.assert_not_called()
        fallback.assert_called_once()
        self.assertEqual(result["model"], "g1")

    def test_validation_api_usage_is_attributed_to_provider_and_profile(self):
        analyzer = module.OllamaSentimentAnalyzer()
        metrics = module._BatchUsageMetrics()
        analyzer._thread_local.usage_metrics = metrics
        analyzer._thread_local.provider_profile = "overnight"
        answer = json.dumps(VALID_ANSWER)

        def post(url, **kwargs):
            if "generativelanguage.googleapis.com" in url:
                body = {"candidates": [{"content": {"parts": [{"text": answer}]}}],
                        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 6},
                        "cost": 0.003}
            else:
                body = {"choices": [{"message": {"content": answer}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.007}}
            return SimpleNamespace(status_code=200, json=lambda: body, headers={})

        with patch.dict(os.environ, {"ENABLE_OPENROUTER": "true", "ENABLE_GEMINI": "true"}), \
             patch.object(module, "OPENROUTER_API_KEY", "offline-key"), \
             patch.object(module, "GEMINI_API_KEY", "offline-key"), \
             patch.object(analyzer, "get_session", return_value=SimpleNamespace(post=post)):
            self.assertIsNotNone(analyzer._call_openrouter_api("other/x", "system", "prompt", max_retries=1))
            self.assertIsNotNone(analyzer._call_gemini_api("g1", "system", "prompt", max_retries=1))
        summary = metrics.summary([])
        self.assertEqual(summary["requests"], 2)
        self.assertAlmostEqual(summary["cost"], 0.01)
        self.assertEqual({row["provider"] for row in summary["providers"]},
                         {"openrouter", "google-ai"})
        self.assertEqual({row["profile"] for row in summary["providers"]}, {"overnight"})

    def test_deferred_thread_keeps_the_post_profile(self):
        analyzer = module.OllamaSentimentAnalyzer()
        seen = []
        lock = threading.Lock()

        def fast(ctx):
            return {"_deferred_deepseek": True, "resolved_context": ctx,
                    "route_info": {}, "jev_signal": None}

        def slow(ctx, *args):
            with lock:
                seen.append((ctx["post_id"], ctx["_provider_profile"],
                             analyzer._thread_local.provider_profile))
            analyzer._record_provider_attempt("deepseek", "mock/model", provider_hint="gemini")
            return {"post_id": ctx["post_id"], "ai_sentiment": 0,
                    "sentiment": "neutral", "model": "mock/model", "route": "deepseek"}

        posts = [{"match_post_id": "1", "content": "SEC update", "project_name": "SEC"},
                 {"match_post_id": "2", "content": "SEC update", "project_name": "SEC"}]
        with patch.object(module, "current_provider_profile", side_effect=["daytime", "overnight"]), \
             patch.object(analyzer, "_hybrid_analyze_post", side_effect=fast), \
             patch.object(analyzer, "_complete_deepseek_route", side_effect=slow):
            batch = analyzer.analyze_post_sentiments(posts)
        self.assertEqual(sorted(seen), [("1", "daytime", "daytime"),
                                        ("2", "overnight", "overnight")])
        self.assertEqual({row["profile"] for row in batch["telemetry"]["providers"]},
                         {"daytime", "overnight"})


if __name__ == "__main__":
    unittest.main()
