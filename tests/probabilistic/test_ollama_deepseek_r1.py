import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
import ai_sentiment as module


class OllamaDeepSeekR1Tests(unittest.TestCase):
    def test_ollama_think_setting_defaults_and_overrides(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OLLAMA_THINK", None)
            self.assertFalse(module._ollama_think_setting("deepseek-r1:8b"))
            self.assertFalse(module._ollama_think_setting("DeepSeek-R1:14b"))
            self.assertIsNone(module._ollama_think_setting("gemma4:31b-cloud"))

        with patch.dict(os.environ, {"OLLAMA_THINK": "true"}):
            self.assertTrue(module._ollama_think_setting("deepseek-r1:8b"))

        with patch.dict(os.environ, {"OLLAMA_THINK": "false"}):
            self.assertFalse(module._ollama_think_setting("gemma4:31b-cloud"))

    def test_deepseek_r1_ollama_call_disables_thinking_and_sets_keep_alive(self):
        analyzer = module.OllamaSentimentAnalyzer(model="deepseek-r1:8b")
        captured = []
        valid = {
            "probabilities": {"POSITIVE": 0.0, "NEUTRAL": 0.0, "NEGATIVE": 1.0, "AMBIGUOUS_OR_IRONY": 0.0},
            "entity_found": True,
            "intent": "complaint",
        }

        def fake_post(url, **kwargs):
            captured.append((url, kwargs["json"], kwargs["timeout"]))
            return SimpleNamespace(
                status_code=200,
                text="",
                headers={},
                json=lambda: {"model": "deepseek-r1:8b", "response": json.dumps(valid)},
            )

        with patch.dict(os.environ, {
            "ENABLE_OLLAMA": "true",
            "OLLAMA_DEEP_TIMEOUT": "90",
            "OLLAMA_KEEP_ALIVE": "15m",
        }), patch.object(analyzer, "get_session", return_value=SimpleNamespace(post=fake_post)):
            res = analyzer._call_ollama_generic("deepseek-r1:8b", "system", "user")

        self.assertEqual(res, valid)
        self.assertEqual(len(captured), 1)
        url, payload, timeout = captured[0]
        self.assertTrue(url.endswith("/api/generate"))
        self.assertIs(payload["think"], False)
        self.assertEqual(payload["keep_alive"], "15m")
        self.assertEqual(timeout, 90)

    def test_ollama_think_400_retries_without_think_key(self):
        analyzer = module.OllamaSentimentAnalyzer()
        payloads = []
        valid = {
            "probabilities": {"POSITIVE": 0.1, "NEUTRAL": 0.9, "NEGATIVE": 0.0, "AMBIGUOUS_OR_IRONY": 0.0},
            "entity_found": True,
            "intent": "information",
        }

        def fake_post(url, **kwargs):
            payloads.append(dict(kwargs["json"]))
            if "think" in kwargs["json"]:
                return SimpleNamespace(
                    status_code=400,
                    text='{"error":"registry.ollama.ai/library/model does not support thinking"}',
                    headers={},
                    json=lambda: {},
                )
            return SimpleNamespace(
                status_code=200,
                text="",
                headers={},
                json=lambda: {"response": json.dumps(valid)},
            )

        with patch.dict(os.environ, {"ENABLE_OLLAMA": "true", "OLLAMA_THINK": "false"}), \
             patch.object(analyzer, "get_session", return_value=SimpleNamespace(post=fake_post)):
            res = analyzer._call_ollama_generic("custom-model:latest", "system", "user")

        self.assertEqual(res, valid)
        self.assertEqual(len(payloads), 2)
        self.assertIn("think", payloads[0])
        self.assertNotIn("think", payloads[1])

    def test_backfill_pipeline_routes_exclusively_to_deepseek_r1_ollama(self):
        analyzer = module.OllamaSentimentAnalyzer(model="deepseek-r1:8b")
        valid = {
            "probabilities": {"POSITIVE": 0.0, "NEUTRAL": 0.1, "NEGATIVE": 0.9, "AMBIGUOUS_OR_IRONY": 0.0},
            "entity_found": True,
            "intent": "complaint",
        }

        def fake_post(url, **kwargs):
            self.assertIn("11434", url)
            return SimpleNamespace(
                status_code=200,
                text="",
                headers={},
                json=lambda: {"model": "deepseek-r1:8b", "response": json.dumps(valid)},
            )

        post = {
            "match_post_id": "hist-1",
            "project_name": "BLCP",
            "keywords": ["ค่าไฟ"],
            "full_text": "ค่าไฟขึ้นอีกแล้ว เดือดร้อนมาก",
            "_analysis_scope": "overall",
        }
        with patch.dict(os.environ, {
            "ENABLE_OPENROUTER": "false",
            "ENABLE_GEMINI": "false",
            "ENABLE_OLLAMA": "true",
            "VALIDATION_MODELS": "deepseek-r1:8b",
            "FORCE_OVERALL_SENTIMENT": "true",
        }), patch.object(analyzer, "get_session", return_value=SimpleNamespace(post=fake_post)):
            out = analyzer.analyze_post_sentiments([post])
            provider = module.inference_provider(out["data"][0]["model"])

        self.assertEqual(len(out["data"]), 1)
        row = out["data"][0]
        self.assertEqual(row["model"], "deepseek-r1:8b")
        self.assertEqual(row["sentiment"], "negative")
        self.assertEqual(row["intent"], "complaint")
        self.assertEqual(provider, "ollama")


if __name__ == "__main__":
    unittest.main()
