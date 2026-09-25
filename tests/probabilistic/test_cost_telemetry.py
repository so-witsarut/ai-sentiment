"""Offline tests for provider usage telemetry and compact hybrid prompts."""

import io
import json
import os
import socket
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["OPENROUTER_API_KEY"] = "offline-test-key"

with patch.object(socket.socket, "connect", side_effect=AssertionError("network access in offline test")):
    import ai_sentiment as module


def response(status, body=None):
    result = MagicMock()
    result.status_code = status
    result.headers = {}
    result.json.return_value = body
    return result


def valid_jev_body(confidence=0.50):
    return {
        "model": "typesafe/jev-1.13",
        "provider": "typesafe",
        "usage": {"input_tokens": 120, "output_tokens": 0, "cost": 0.00001},
        "answers": {
            "sentiment": {
                "choice": "positive",
                "confidence": confidence,
                "probabilities": {"positive": 0.70, "neutral": 0.15, "negative": 0.10, "irony": 0.05},
            },
            "entity_relevance": {
                "choice": "relevant",
                "confidence": 0.90,
                "probabilities": {"relevant": 0.90, "unrelated": 0.05, "uncertain": 0.05},
            },
        },
    }


def valid_deepseek_body():
    content = {
        "entity_found": True,
        "probabilities": {
            "POSITIVE": 0.80,
            "NEUTRAL": 0.10,
            "NEGATIVE": 0.05,
            "AMBIGUOUS_OR_IRONY": 0.05,
        },
    }
    return {
        "model": "deepseek/deepseek-v4-flash-0731",
        "provider": "sail-research/fp4",
        "choices": [{"message": {"content": json.dumps(content)}}],
        "usage": {
            "prompt_tokens": 240,
            "completion_tokens": 24,
            "prompt_tokens_details": {"cached_tokens": 80},
            "cost": 0.00010,
            "cache_discount": 0.00002,
        },
    }


class TestCostTelemetry(unittest.TestCase):
    def setUp(self):
        cost_mode_patch = patch.object(module, "AI_COST_MODE", "standard")
        cost_mode_patch.start()
        self.addCleanup(cost_mode_patch.stop)
        self.context_post = {
            "match_post_id": "cost-1",
            "content": "SCB ให้บริการดีมาก",
            "actual_target": "SCB",
            "project_name": "SCB",
            "project_desc": "ธนาคารไทย",
            "keywords": ["SCB", "บริการธนาคาร"],
            "post_user": "tester",
        }

    def test_usage_normalizes_openrouter_and_gemini_shapes(self):
        self.assertEqual(
            module._extract_provider_usage(valid_deepseek_body()),
            {"input": 240, "output": 24, "cached": 80, "cost": 0.00010, "cache_discount": 0.00002},
        )
        self.assertEqual(
            module._extract_provider_usage({
                "usageMetadata": {
                    "promptTokenCount": 40,
                    "candidatesTokenCount": 7,
                    "cachedContentTokenCount": 10,
                }
            }),
            {"input": 40, "output": 7, "cached": 10, "cost": 0.0, "cache_discount": 0.0},
        )

    def test_batch_reports_routes_provider_tokens_cache_and_no_duplicate_calls(self):
        analyzer = module.OllamaSentimentAnalyzer()
        mock_session = MagicMock()
        mock_session.post.side_effect = [
            response(200, valid_jev_body(confidence=0.50)),
            response(200, valid_deepseek_body()),
        ]
        with patch.object(module, "OPENROUTER_API_KEY", "offline-test-key"), \
                patch.object(analyzer, "get_session", return_value=mock_session), \
                redirect_stdout(io.StringIO()):
            result = analyzer.analyze_post_sentiments([self.context_post])

        self.assertEqual(mock_session.post.call_count, 2)
        self.assertEqual(result["token_usage"], {"input": 360, "output": 24, "total": 384})
        telemetry = result["telemetry"]
        self.assertEqual(telemetry["routes"], {"jev": 0, "deepseek": 1, "rule": 0})
        self.assertEqual(telemetry["requests"], 2)
        self.assertEqual(telemetry["retries"], 0)
        self.assertEqual(telemetry["cached_tokens"], 80)
        self.assertAlmostEqual(telemetry["cost"], 0.00011)
        self.assertAlmostEqual(telemetry["cache_discount"], 0.00002)
        self.assertEqual({row["route"] for row in telemetry["providers"]}, {"jev", "deepseek"})
        deepseek_row = next(row for row in telemetry["providers"] if row["route"] == "deepseek")
        self.assertEqual(deepseek_row["provider"], "sail-research/fp4")
        self.assertEqual(deepseek_row["model"], "deepseek/deepseek-v4-flash-0731")

    def test_rest_fallback_passes_jev_hint_and_counts_escalation(self):
        analyzer = module.OllamaSentimentAnalyzer()
        post = {**self.context_post, "_analysis_scope": "keyword"}
        mock_session = MagicMock()
        mock_session.post.side_effect = [
            response(200, valid_jev_body(confidence=0.50)),
            response(200, valid_deepseek_body()),
        ]
        with patch.object(module, "OPENROUTER_API_KEY", "offline-test-key"), \
             patch.object(analyzer, "get_session", return_value=mock_session), \
             patch.dict(os.environ, {"DEEPSEEK_INCLUDE_JEV_SIGNAL": "true"}), \
             redirect_stdout(io.StringIO()):
            result = analyzer.analyze_post_sentiments([post])

        prompt = mock_session.post.call_args_list[1].kwargs["json"]["messages"][1]["content"]
        self.assertIn("Jev preliminary", prompt)
        self.assertIn("sentiment=positive(0.50)", prompt)
        self.assertIn("relevance=relevant(0.90)", prompt)
        self.assertIn("route_confidence=0.50", prompt)
        self.assertEqual(result["data"][0]["route"], "deepseek")
        self.assertEqual(result["telemetry"]["escalated_posts"], 1)
        self.assertEqual(result["telemetry"]["escalation_rate_percent"], 100.0)

    def test_failed_fallback_still_counts_as_escalated(self):
        analyzer = module.OllamaSentimentAnalyzer()
        jev = module.validate_jev_response(valid_jev_body(confidence=0.50))
        with patch.object(analyzer, "_call_typesafe_jev", return_value=(jev, "jev")), \
             patch.object(analyzer, "_call_deepseek_fallback", return_value=(None, "")), \
             redirect_stdout(io.StringIO()):
            result = analyzer.analyze_post_sentiments([{**self.context_post, "_analysis_scope": "keyword"}])
        self.assertEqual(result["data"][0]["model"], "rule:provider_failure")
        self.assertEqual(result["telemetry"]["escalated_posts"], 1)
        self.assertEqual(result["telemetry"]["escalation_rate_percent"], 100.0)

    def test_retry_count_counts_only_calls_after_first_attempt(self):
        analyzer = module.OllamaSentimentAnalyzer()
        mock_session = MagicMock()
        mock_session.post.side_effect = [
            response(429),
            response(200, valid_jev_body(confidence=0.90)),
        ]
        with patch.object(module, "OPENROUTER_API_KEY", "offline-test-key"), \
                patch.object(module, "_retry_within_deadline", return_value=True), \
                patch.object(analyzer, "get_session", return_value=mock_session), \
                redirect_stdout(io.StringIO()):
            result = analyzer.analyze_post_sentiments([self.context_post])

        self.assertEqual(mock_session.post.call_count, 2)
        self.assertEqual(result["telemetry"]["requests"], 2)
        self.assertEqual(result["telemetry"]["retries"], 1)
        self.assertEqual(result["telemetry"]["routes"]["jev"], 1)

    @patch.object(module, "HYBRID_PROMPT_V2", True)
    def test_compact_prompts_keep_context_and_remove_exact_duplicates(self):
        context = {
            "actual_target": "SCB",
            "project_name": "SCB",
            "project_desc": "ธนาคารไทย",
            "keywords": ["SCB", "บริการธนาคาร", "บริการธนาคาร"],
            "source_info": "User=tester",
            "capped_text": "SCB ให้บริการดีมาก",
        }
        jev_prompt = module.build_jev_state_prompt(context)
        self.assertEqual(jev_prompt.count("Target=SCB"), 1)
        self.assertNotIn("Project=SCB", jev_prompt)
        self.assertEqual(jev_prompt.count("บริการธนาคาร"), 1)
        self.assertIn("Context=ธนาคารไทย", jev_prompt)
        self.assertIn("Source=User=tester", jev_prompt)
        self.assertIn("Text=SCB ให้บริการดีมาก", jev_prompt)

        deep_prompt = module.build_deepseek_user_prompt(context, {
            "probabilities": {"POSITIVE": 0.5},
            "confidence": 0.6,
            "conflict_reasons": ["low_confidence"],
        })
        self.assertIn("Jev=", deep_prompt)
        self.assertIn("confidence=0.6", deep_prompt)
        self.assertIn("conflicts=low_confidence", deep_prompt)


if __name__ == "__main__":
    unittest.main()
