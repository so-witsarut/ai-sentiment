"""Offline regression checks for routing, unresolved sentiment and keyword excerpts."""

import io
import json
import os
import socket
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
import ai_sentiment as module


def decision(pos=0.02, neu=0.05, neg=0.91, ambiguity=0.02, entity=True):
    return {"probabilities": {"POSITIVE": pos, "NEUTRAL": neu,
                              "NEGATIVE": neg, "AMBIGUOUS_OR_IRONY": ambiguity},
            "entity_found": entity, "intent": "complaint"}


def jev_body(pos=0.02, neu=0.05, neg=0.91, ambiguity=0.02, entity="relevant"):
    probabilities = {"positive": pos, "neutral": neu, "negative": neg, "irony": ambiguity}
    return {"answers": {
        "sentiment": {"choice": max(probabilities, key=probabilities.get),
                      "confidence": 0.95, "probabilities": probabilities},
        "entity_relevance": {"choice": entity, "confidence": 0.95,
                             "probabilities": {"relevant": 0.95 if entity == "relevant" else 0.03,
                                               "unrelated": 0.95 if entity == "unrelated" else 0.03,
                                               "uncertain": 0.02}},
        "intent": {"choice": "complaint"},
    }}


def response(status=200, body=None):
    return SimpleNamespace(status_code=status, headers={}, text="offline response", json=lambda: body)


def context(profile="daytime"):
    return {"post_id": "p1", "_provider_profile": profile, "analysis_scope": "keyword",
            "actual_target": "BLCP", "sentiment_target": "BLCP", "project_name": "BLCP",
            "keywords": ["ค่าไฟ"], "clean_text": "BLCP ค่าไฟแพงมาก",
            "capped_text": "BLCP ค่าไฟแพงมาก"}


class SentimentImprovementsTests(unittest.TestCase):
    def setUp(self):
        for patcher in (
            patch.dict(os.environ, {"ENABLE_OPENROUTER": "true", "ENABLE_GEMINI": "true",
                                    "ENABLE_OLLAMA": "true", "VALIDATION_MODELS": "local:first,api:g1,api:g2"}),
            patch.object(module, "OPENROUTER_API_KEY", "offline-key"),
            patch.object(module, "HYBRID_PROMPT_V2", True),
            patch.object(module, "ENABLE_JEV_HYBRID", True),
            patch.object(module, "DEEPSEEK_MAX_RETRIES", 0),
            patch.object(module, "_retry_within_deadline", return_value=True),
            patch.object(socket.socket, "connect", side_effect=AssertionError("offline tests cannot use network")),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.analyzer = module.OllamaSentimentAnalyzer()

    def test_daytime_failed_jev_twice_then_deepseek_gemini_ollama(self):
        calls = []

        def post(url, **kwargs):
            calls.append(kwargs["json"]["model"])
            return response(503) if "decisions" in url else response(body={"choices": []})

        def gemini(model, *args, **kwargs):
            calls.append(model)
            return None

        def ollama(model, *args, **kwargs):
            calls.append(model)
            return decision()

        # An OpenRouter validation entry must not move ahead of Gemini/Ollama.
        with patch.dict(os.environ, {"VALIDATION_MODELS": "local:first,openrouter:unused,api:g1,api:g2"}), \
             patch.object(self.analyzer, "get_session", return_value=SimpleNamespace(post=post)), \
             patch.object(self.analyzer, "_call_gemini_api", side_effect=gemini), \
             patch.object(self.analyzer, "_call_ollama_generic", side_effect=ollama), \
             redirect_stdout(io.StringIO()):
            result = self.analyzer._hybrid_analyze_post(context())
        self.assertEqual(calls, [module.JEV_MODEL, module.JEV_MODEL, module.DEEPSEEK_MODEL,
                                 "g1", "g2", "local:first"])
        self.assertEqual(result["model"], "local:first")

    def test_daytime_jev_timeout_budget_ignores_nested_retry_settings(self):
        session = Mock()
        session.post.side_effect = requests.exceptions.Timeout("offline timeout")
        with patch.object(module, "JEV_MAX_RETRIES", 10), \
             patch.object(module, "JEV_FALLBACK_MODELS", ["jev:other", "jev:last"]), \
             patch.object(self.analyzer, "get_session", return_value=session), \
             patch.object(self.analyzer, "_call_deepseek_fallback", return_value=(decision(), "deepseek")) as fallback:
            result = self.analyzer._hybrid_analyze_post(context())
        self.assertEqual(session.post.call_count, 2)
        fallback.assert_called_once()
        self.assertEqual(result["sentiment"], "negative")

    def test_daytime_invalid_jev_retries_once_but_valid_low_confidence_does_not(self):
        for body, expected_calls in (({}, 2), (jev_body(pos=0.50, neu=0.25, neg=0.20, ambiguity=0.05), 1)):
            with self.subTest(body=body):
                session = Mock()
                session.post.return_value = response(body=body)
                with patch.object(module, "AI_COST_MODE", "standard"), \
                     patch.object(self.analyzer, "get_session", return_value=session), \
                     patch.object(self.analyzer, "_call_deepseek_fallback", return_value=(decision(), "deepseek")):
                    self.analyzer._hybrid_analyze_post(context())
                self.assertEqual(session.post.call_count, expected_calls)

    def test_daytime_successful_second_jev_stops_before_fallback(self):
        session = Mock()
        session.post.side_effect = [response(429), response(body=jev_body())]
        with patch.object(self.analyzer, "get_session", return_value=session), \
             patch.object(self.analyzer, "_call_deepseek_fallback") as fallback:
            result = self.analyzer._hybrid_analyze_post(context())
        self.assertEqual(session.post.call_count, 2)
        self.assertEqual(result["route"], "jev")
        fallback.assert_not_called()

    def test_overnight_all_gemini_then_ollama_before_openrouter(self):
        calls = []

        def gemini(model, *args, **kwargs):
            calls.append(model)
            return None

        def ollama(model, *args, **kwargs):
            calls.append(model)
            return None

        def post(url, **kwargs):
            calls.append(kwargs["json"]["model"])
            return response(body={"choices": [{"message": {"content": json.dumps(decision())}}]})

        with patch.object(self.analyzer, "_call_typesafe_jev") as jev, \
             patch.object(self.analyzer, "_call_gemini_api", side_effect=gemini), \
             patch.object(self.analyzer, "_call_ollama_generic", side_effect=ollama), \
             patch.object(self.analyzer, "get_session", return_value=SimpleNamespace(post=post)):
            result = self.analyzer._hybrid_analyze_post(context("overnight"))
        jev.assert_not_called()
        self.assertEqual(calls, ["g1", "g2", "local:first", module.DEEPSEEK_MODEL])
        self.assertEqual(result["model"], module.DEEPSEEK_MODEL)

    def test_overnight_gemini_success_never_calls_ollama_or_openrouter(self):
        with patch.object(self.analyzer, "_call_gemini_api", return_value=decision()) as gemini, \
             patch.object(self.analyzer, "_call_ollama_generic") as ollama, \
             patch.object(self.analyzer, "_call_typesafe_jev") as jev, \
             patch.object(self.analyzer, "get_session") as session:
            result = self.analyzer._hybrid_analyze_post(context("overnight"))
        gemini.assert_called_once()
        ollama.assert_not_called()
        jev.assert_not_called()
        session.assert_not_called()
        self.assertEqual(result["model"], "g1")

    def test_all_providers_fail_in_either_schedule_and_rest_still_saves_neutral(self):
        for profile, expected_openrouter_calls in (("daytime", 3), ("overnight", 1)):
            with self.subTest(profile=profile):
                api = module.SentimentAPI(analyzer=self.analyzer)
                session = Mock()
                session.post.return_value = response(body={"choices": []})
                post = {"match_post_id": "p1", "project_name": "BLCP",
                        "content": "BLCP ค่าไฟแพงมาก", "keywords": ["ค่าไฟ"]}
                output = io.StringIO()
                with patch.object(module, "current_provider_profile", return_value=profile), \
                     patch.object(api, "fetch_pending", return_value=[post]), \
                     patch.object(api, "bulk_update", return_value=1) as update, \
                     patch.object(self.analyzer, "get_session", return_value=session), \
                     patch.object(self.analyzer, "_call_gemini_api", return_value=None) as gemini, \
                     patch.object(self.analyzer, "_call_ollama_generic", return_value=None) as ollama, \
                     redirect_stdout(output):
                    self.assertEqual(api.run("2026-10-02", "2026-10-02", save_db=True), 1)
                self.assertEqual(session.post.call_count, expected_openrouter_calls)
                self.assertEqual(gemini.call_count, 2)
                ollama.assert_called_once()
                payload = update.call_args.args[0][0]
                self.assertEqual(payload["sentiment"], "neutral")
                self.assertEqual(payload["intent"], "information")
                self.assertEqual(payload["sentiment_status"], "1")
                self.assertEqual(payload["sentiment_scores"]["neutral"], 100)
                self.assertEqual(payload["sentiment_scores"]["model"], "rule:provider_failure")
                self.assertIn(f"profile={profile} requests: OpenRouter={expected_openrouter_calls}", output.getvalue())
                self.assertIn("provider_failure=1", output.getvalue())

    def test_ambiguity_escalates_even_in_low_mode_and_final_result_is_neutral(self):
        ambiguous = decision(pos=0.05, neu=0.10, neg=0.10, ambiguity=0.75)
        raw = module.validate_jev_response(jev_body(pos=0.05, neu=0.10, neg=0.10, ambiguity=0.75))
        for scope in ("keyword", "overall"):
            with self.subTest(scope=scope), \
                 patch.object(module, "AI_COST_MODE", "low"), \
                 patch.object(self.analyzer, "_call_typesafe_jev", return_value=(raw, "jev")), \
                 patch.object(self.analyzer, "_call_deepseek_fallback", return_value=(ambiguous, "deepseek")) as fallback:
                result = self.analyzer._hybrid_analyze_post({**context(), "analysis_scope": scope})
            fallback.assert_called_once()
            self.assertEqual(result["sentiment"], "neutral")
            self.assertTrue(result["is_ambiguous"])
            self.assertEqual(result["negative_percent"], 10)
            self.assertEqual(result["neutral_percent"], 85)
            self.assertIn("กำกวม", result["reason"])
            self.assertNotIn("เชิงลบ", result["reason"])

    def test_clear_negative_meaning_and_mixed_tie_policy(self):
        negative = module.resolve_policy(decision()["probabilities"])
        self.assertEqual(negative["sentiment"], "negative")
        self.assertFalse(negative["is_ambiguous"])
        mixed = module.resolve_policy(decision(pos=0.45, neg=0.45, neu=0.10, ambiguity=0)["probabilities"])
        self.assertEqual(mixed["sentiment"], "neutral")
        self.assertEqual((mixed["pos"], mixed["neg"], mixed["neu"]), (45, 45, 10))

    def test_jev_intent_is_whole_post_even_for_unrelated_target(self):
        session = Mock()
        session.post.return_value = response(body=jev_body(pos=0.01, neu=0.96, neg=0.02,
                                                         ambiguity=0.01, entity="unrelated"))
        with patch.object(self.analyzer, "get_session", return_value=session):
            result = self.analyzer._hybrid_analyze_post(context())
        question = session.post.call_args.kwargs["json"]["questions"]["intent"]
        self.assertIn("whole post", question["instructions"])
        self.assertTrue(all("Target" not in criterion for criterion in question["criteria"].values()))
        self.assertEqual(result["sentiment"], "neutral")
        self.assertEqual(result["intent"], "complaint")
        for prompt in (module.PROBABILISTIC_SYSTEM_PROMPT, module.HYBRID_SYSTEM_PROMPT,
                       module.KEYWORD_SYSTEM_PROMPT, module.OVERALL_SYSTEM_PROMPT):
            self.assertIn(module.INTENT_INSTRUCTIONS, prompt)
            self.assertNotIn("Omit intent", prompt)

    def test_multiple_keywords_keep_distant_windows_and_post_edges(self):
        text = "ต้น" * 500 + " ค่าไฟแพงมาก " + "กลาง" * 800 + " โรงไฟฟ้าบริการแย่ " + "ท้าย" * 500
        for values in (["ไม่พบ", "ค่าไฟ", "โรงไฟฟ้า"], "ไม่พบ, ค่าไฟ, โรงไฟฟ้า, ค่าไฟ"):
            excerpt = module.cap_text(text, max_chars=1200, keywords=values)
            self.assertLessEqual(len(excerpt), 1200)
            self.assertIn("ค่าไฟแพงมาก", excerpt)
            self.assertIn("โรงไฟฟ้าบริการแย่", excerpt)
            self.assertTrue(excerpt.startswith("ต้น" * 50))
            self.assertTrue(excerpt.endswith("ท้าย" * 50))

    def test_repeated_target_later_criticism_and_latin_boundaries(self):
        text = "a" * 800 + " ABC " + "b" * 2500 + " ABC บริการแย่มาก " + "c" * 2500
        excerpt = module.cap_text(text, max_chars=1800, target="ABC")
        self.assertIn("บริการแย่มาก", excerpt)
        self.assertLessEqual(len(excerpt), 1800)
        misleading = "a" * 1500 + " BUSINESS " + "b" * 1500
        self.assertEqual(module.cap_text(misleading, max_chars=400, keywords=["US"]),
                         module.cap_text(misleading, max_chars=400))

    def test_rest_prefers_full_text_and_keeps_preview_separate(self):
        original = "ต้น" * 500 + " ค่าไฟแพง " + "กลาง" * 800 + " โรงไฟฟ้าเสีย " + "ท้าย" * 500
        for values in ([" ค่าไฟ ", "โรงไฟฟ้า", "ค่าไฟ", None], "ค่าไฟ, โรงไฟฟ้า, ค่าไฟ"):
            with self.subTest(values=values):
                api = module.SentimentAPI(analyzer=self.analyzer)
                post = {"match_post_id": "p1", "project_name": "BLCP", "content": "preview",
                        "full_text": original, "keywords": values, "_provider_profile": "daytime"}
                captured = []

                def hybrid(resolved):
                    captured.append(resolved)
                    return module.OllamaSentimentAnalyzer._neutral_error_result("p1", "BLCP")

                with patch.object(api, "fetch_pending", return_value=[post]), \
                     patch.object(self.analyzer, "_hybrid_analyze_post", side_effect=hybrid), \
                     patch.object(api, "bulk_update", return_value=1) as update, \
                     redirect_stdout(io.StringIO()):
                    self.assertEqual(api.run("2026-10-02", "2026-10-02", save_db=True), 1)
                resolved = captured[0]
                self.assertEqual(resolved["clean_text"], original)
                self.assertEqual(resolved["keywords"], ["ค่าไฟ", "โรงไฟฟ้า"])
                self.assertIn("ค่าไฟแพง", resolved["capped_text"])
                self.assertIn("โรงไฟฟ้าเสีย", resolved["capped_text"])
                with patch.dict(os.environ, {"DEEPSEEK_TEXT_MAX_CHARS": "1200"}):
                    fallback_prompt = module.build_deepseek_user_prompt(resolved)
                self.assertIn("ค่าไฟแพง", fallback_prompt)
                self.assertIn("โรงไฟฟ้าเสีย", fallback_prompt)
                payload = update.call_args.args[0][0]
                self.assertEqual(payload["sentiment_status"], "1")
                self.assertEqual(payload["sentiment_scores"]["model"], "rule:provider_failure")
                self.assertNotIn("is_ambiguous", payload)

    def test_overlap_short_text_and_dense_keywords_respect_cap(self):
        short = "ค่าไฟโรงไฟฟ้า ค่าไฟ"
        self.assertEqual(module.cap_text(short, max_chars=1200, keywords=["ค่าไฟ", "โรงไฟฟ้า"]), short)
        text = " ".join(f"TERM{i}" for i in range(300))
        excerpt = module.cap_text(text, max_chars=600, target="TERM299",
                                  keywords=[f"TERM{i}" for i in range(300)])
        self.assertLessEqual(len(excerpt), 600)
        self.assertIn("TERM299", excerpt)
        self.assertEqual(module.post_keywords({"keywords": [], "keyword_name": " A, a, B "}), ["A", "B"])

    def test_profile_telemetry_counts_ollama_http_attempts_and_failure_defaults(self):
        session = Mock()
        session.post.side_effect = [response(503), response(503)]
        metrics = module._BatchUsageMetrics()
        with patch.object(self.analyzer, "get_session", return_value=session), redirect_stdout(io.StringIO()):
            self.analyzer._run_with_usage_metrics(
                metrics, lambda ctx: self.analyzer._call_ollama_generic("local:first", "system", "text"),
                context("overnight"))
        metrics.record("jev", "jev", "endpoint/fp4", False, profile="daytime", provider_family="openrouter")
        metrics.record("deepseek", "gemini", "google-ai", False, profile="overnight")
        failure = {**self.analyzer._neutral_error_result("p1"), "_provider_profile": "overnight"}
        profiles = metrics.summary([failure])["profiles"]
        self.assertEqual(profiles["overnight"], {
            "requests": {"openrouter": 0, "gemini": 1, "ollama": 2}, "provider_failures": 1})
        self.assertEqual(profiles["daytime"]["requests"]["openrouter"], 1)


if __name__ == "__main__":
    unittest.main()
