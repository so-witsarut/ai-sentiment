# coding=utf-8
"""
tests/probabilistic/test_hybrid_routing.py
Comprehensive unit tests verifying the 17 hybrid routing behaviors specified in .ai/HANDOFF.md.
Tests the REST-only production worker with no network access.
"""

import os
import sys
import math
import json
import unittest
from unittest.mock import MagicMock, patch

# Disable network / live db
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["OPENROUTER_API_KEY"] = "mock-openrouter-key-for-unit-tests"

# Add repository root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import ai_sentiment as sentiment_module


class TestHybridRouting(unittest.TestCase):
    def setUp(self):
        self.modules = [sentiment_module]

    # -------------------------------------------------------------------------
    # 1. Gate boundary 0.6500 accepted vs 0.6499 rejected
    # -------------------------------------------------------------------------
    def test_gate_boundary_0650_accepted(self):
        """Valid Jev at confidence exactly 0.6500, no conflict -> accept_jev is True"""
        for mod in self.modules:
            jev_data = {
                "sentiment_probabilities": {"positive": 0.65, "neutral": 0.20, "negative": 0.10, "irony": 0.05},
                "sentiment_confidence": 0.65,
                "sentiment_choice": "positive",
                "entity_probabilities": {"relevant": 0.70, "unrelated": 0.20, "uncertain": 0.10},
                "entity_confidence": 0.65,
                "entity_choice": "relevant"
            }
            route = mod.route_sentiment(jev_data, actual_target="SCB")
            self.assertTrue(route["accept_jev"], f"Failed for {mod.__name__}")
            self.assertAlmostEqual(route["routing_confidence"], 0.65, places=4)
            self.assertEqual(route["conflict_reasons"], [])

    def test_gate_boundary_06499_rejected(self):
        """Valid Jev at confidence 0.6499, no conflict -> accept_jev is False, routed to DeepSeek"""
        for mod in self.modules:
            jev_data = {
                "sentiment_probabilities": {"positive": 0.65, "neutral": 0.20, "negative": 0.10, "irony": 0.05},
                "sentiment_confidence": 0.6499,
                "sentiment_choice": "positive",
                "entity_probabilities": {"relevant": 0.70, "unrelated": 0.20, "uncertain": 0.10},
                "entity_confidence": 0.6499,
                "entity_choice": "relevant"
            }
            route = mod.route_sentiment(jev_data, actual_target="SCB")
            self.assertFalse(route["accept_jev"], f"Failed for {mod.__name__}")
            self.assertAlmostEqual(route["routing_confidence"], 0.6499, places=4)

    # -------------------------------------------------------------------------
    # 2. Named Conflict Conditions
    # -------------------------------------------------------------------------
    def test_conflict_entity_uncertain(self):
        """Entity uncertainty triggers 'entity_uncertain' conflict and rejects Jev"""
        for mod in self.modules:
            jev_data = {
                "sentiment_probabilities": {"positive": 0.80, "neutral": 0.10, "negative": 0.05, "irony": 0.05},
                "sentiment_confidence": 0.85,
                "sentiment_choice": "positive",
                "entity_probabilities": {"relevant": 0.30, "unrelated": 0.30, "uncertain": 0.40},
                "entity_confidence": 0.70,
                "entity_choice": "uncertain"
            }
            route = mod.route_sentiment(jev_data, actual_target="SCB")
            self.assertFalse(route["accept_jev"])
            self.assertIn("entity_uncertain", route["conflict_reasons"])

    def test_conflict_unrelated_with_sentiment_mass(self):
        """Unrelated entity with (pos + neg + irony) >= 0.35 triggers conflict"""
        for mod in self.modules:
            jev_data = {
                "sentiment_probabilities": {"positive": 0.40, "neutral": 0.50, "negative": 0.05, "irony": 0.05},
                "sentiment_confidence": 0.80,
                "sentiment_choice": "neutral",
                "entity_probabilities": {"relevant": 0.10, "unrelated": 0.85, "uncertain": 0.05},
                "entity_confidence": 0.85,
                "entity_choice": "unrelated"
            }
            route = mod.route_sentiment(jev_data, actual_target="SCB")
            self.assertFalse(route["accept_jev"])
            self.assertIn("unrelated_with_sentiment_mass", route["conflict_reasons"])

    def test_conflict_bipolar_positive_negative(self):
        """Both positive and negative >= 0.25 triggers 'bipolar_positive_negative' conflict"""
        for mod in self.modules:
            jev_data = {
                "sentiment_probabilities": {"positive": 0.40, "neutral": 0.10, "negative": 0.40, "irony": 0.10},
                "sentiment_confidence": 0.80,
                "sentiment_choice": "positive",
                "entity_probabilities": {"relevant": 0.85, "unrelated": 0.10, "uncertain": 0.05},
                "entity_confidence": 0.85,
                "entity_choice": "relevant"
            }
            route = mod.route_sentiment(jev_data, actual_target="SCB")
            self.assertFalse(route["accept_jev"])
            self.assertIn("bipolar_positive_negative", route["conflict_reasons"])

    def test_conflict_narrow_sentiment_margin(self):
        """Top two probabilities with difference < 0.15 triggers 'narrow_sentiment_margin'"""
        for mod in self.modules:
            jev_data = {
                "sentiment_probabilities": {"positive": 0.45, "neutral": 0.38, "negative": 0.12, "irony": 0.05},
                "sentiment_confidence": 0.80,
                "sentiment_choice": "positive",
                "entity_probabilities": {"relevant": 0.85, "unrelated": 0.10, "uncertain": 0.05},
                "entity_confidence": 0.85,
                "entity_choice": "relevant"
            }
            route = mod.route_sentiment(jev_data, actual_target="SCB")
            self.assertFalse(route["accept_jev"])
            self.assertIn("narrow_sentiment_margin", route["conflict_reasons"])

    def test_conflict_provider_choice_disagrees_with_argmax(self):
        """Provider choice differing from mathematical argmax triggers conflict"""
        for mod in self.modules:
            jev_data = {
                "sentiment_probabilities": {"positive": 0.70, "neutral": 0.15, "negative": 0.10, "irony": 0.05},
                "sentiment_confidence": 0.80,
                "sentiment_choice": "negative",  # disagrees with argmax (positive)
                "entity_probabilities": {"relevant": 0.85, "unrelated": 0.10, "uncertain": 0.05},
                "entity_confidence": 0.85,
                "entity_choice": "relevant"
            }
            route = mod.route_sentiment(jev_data, actual_target="SCB")
            self.assertFalse(route["accept_jev"])
            self.assertIn("provider_choice_disagrees_with_argmax", route["conflict_reasons"])

    def test_conflict_generic_placeholder_target(self):
        """Generic placeholder target forces 'generic_placeholder_target' conflict"""
        for mod in self.modules:
            jev_data = {
                "sentiment_probabilities": {"positive": 0.80, "neutral": 0.10, "negative": 0.05, "irony": 0.05},
                "sentiment_confidence": 0.85,
                "sentiment_choice": "positive",
                "entity_probabilities": {"relevant": 0.85, "unrelated": 0.10, "uncertain": 0.05},
                "entity_confidence": 0.85,
                "entity_choice": "relevant"
            }
            for placeholder in ["the Target Entity", "target entity", "unknown", "None", "null", ""]:
                route = mod.route_sentiment(jev_data, actual_target=placeholder)
                self.assertFalse(route["accept_jev"], f"Placeholder '{placeholder}' was accepted in {mod.__name__}")
                self.assertIn("generic_placeholder_target", route["conflict_reasons"])

    # -------------------------------------------------------------------------
    # 3. High-Confidence Ironic Jev Accepted
    # -------------------------------------------------------------------------
    def test_high_confidence_ironic_jev_accepted(self):
        """High confidence irony without conflicts is accepted and conservative policy applies"""
        for mod in self.modules:
            jev_data = {
                "sentiment_probabilities": {"positive": 0.05, "neutral": 0.10, "negative": 0.10, "irony": 0.75},
                "sentiment_confidence": 0.80,
                "sentiment_choice": "irony",
                "entity_probabilities": {"relevant": 0.90, "unrelated": 0.05, "uncertain": 0.05},
                "entity_confidence": 0.85,
                "entity_choice": "relevant"
            }
            route = mod.route_sentiment(jev_data, actual_target="SCB")
            self.assertTrue(route["accept_jev"])
            self.assertEqual(route["conflict_reasons"], [])

            # Policy converts heavy irony into conservative negative sentiment
            probs = {"POSITIVE": 0.05, "NEUTRAL": 0.10, "NEGATIVE": 0.10, "AMBIGUOUS_OR_IRONY": 0.75}
            policy = mod.resolve_policy(probs, entity_found=True)
            self.assertEqual(policy["sentiment"], "negative")
            self.assertEqual(policy["score"], -100)
            self.assertEqual(policy["irony_score"], 75)
            self.assertEqual(policy["pos"] + policy["neg"] + policy["neu"], 100)

    # -------------------------------------------------------------------------
    # 4. Confident Unrelated Entity Accepted as Neutral
    # -------------------------------------------------------------------------
    def test_confident_unrelated_accepted_as_neutral(self):
        """Confident unrelated entity with neutral mass passes gate and resolves to entity_found=False"""
        for mod in self.modules:
            jev_data = {
                "sentiment_probabilities": {"positive": 0.05, "neutral": 0.90, "negative": 0.03, "irony": 0.02},
                "sentiment_confidence": 0.85,
                "sentiment_choice": "neutral",
                "entity_probabilities": {"relevant": 0.05, "unrelated": 0.90, "uncertain": 0.05},
                "entity_confidence": 0.85,
                "entity_choice": "unrelated"
            }
            route = mod.route_sentiment(jev_data, actual_target="SCB")
            self.assertTrue(route["accept_jev"])
            self.assertFalse(route["entity_found"])
            self.assertEqual(route["conflict_reasons"], [])

            # Policy with entity_found=False yields neutral 0/0/100
            probs = {"POSITIVE": 0.05, "NEUTRAL": 0.90, "NEGATIVE": 0.03, "AMBIGUOUS_OR_IRONY": 0.02}
            policy = mod.resolve_policy(probs, entity_found=False)
            self.assertEqual(policy["sentiment"], "neutral")
            self.assertEqual(policy["score"], 0)
            self.assertEqual(policy["pos"], 0)
            self.assertEqual(policy["neg"], 0)
            self.assertEqual(policy["neu"], 100)

    # -------------------------------------------------------------------------
    # 5. TypeSafe Jev Validation & Rejection
    # -------------------------------------------------------------------------
    def test_validate_jev_response_corrupt_inputs(self):
        """Corrupt Jev payloads (NaN, infinite, negative, out-of-sum, missing keys) return None"""
        for mod in self.modules:
            # 1. Non-dict
            self.assertIsNone(mod.validate_jev_response("not a dict"))
            self.assertIsNone(mod.validate_jev_response(None))

            # 2. Missing answers
            self.assertIsNone(mod.validate_jev_response({}))

            # 3. NaN in probabilities
            nan_payload = {
                "answers": {
                    "sentiment": {"confidence": 0.8, "probabilities": {"positive": float("nan"), "neutral": 0.5, "negative": 0.1, "irony": 0.0}},
                    "entity_relevance": {"confidence": 0.8, "probabilities": {"relevant": 0.8, "unrelated": 0.1, "uncertain": 0.1}}
                }
            }
            self.assertIsNone(mod.validate_jev_response(nan_payload))

            # 4. Negative probability
            neg_payload = {
                "answers": {
                    "sentiment": {"confidence": 0.8, "probabilities": {"positive": -0.1, "neutral": 0.8, "negative": 0.2, "irony": 0.1}},
                    "entity_relevance": {"confidence": 0.8, "probabilities": {"relevant": 0.8, "unrelated": 0.1, "uncertain": 0.1}}
                }
            }
            self.assertIsNone(mod.validate_jev_response(neg_payload))

            # 5. Sum outside [0.98, 1.02]
            bad_sum_payload = {
                "answers": {
                    "sentiment": {"confidence": 0.8, "probabilities": {"positive": 0.3, "neutral": 0.3, "negative": 0.1, "irony": 0.0}},
                    "entity_relevance": {"confidence": 0.8, "probabilities": {"relevant": 0.8, "unrelated": 0.1, "uncertain": 0.1}}
                }
            }
            self.assertIsNone(mod.validate_jev_response(bad_sum_payload))

            # 6. Missing confidence
            no_conf_payload = {
                "answers": {
                    "sentiment": {"probabilities": {"positive": 0.7, "neutral": 0.1, "negative": 0.1, "irony": 0.1}},
                    "entity_relevance": {"confidence": 0.8, "probabilities": {"relevant": 0.8, "unrelated": 0.1, "uncertain": 0.1}}
                }
            }
            self.assertIsNone(mod.validate_jev_response(no_conf_payload))

    # -------------------------------------------------------------------------
    # 6. DeepSeek Strict Parsing & Normalization
    # -------------------------------------------------------------------------
    def test_validate_deepseek_response_valid_and_invalid(self):
        """Strict validation of DeepSeek fallback responses"""
        for mod in self.modules:
            # Valid float fractions
            valid_floats = {
                "entity_found": True,
                "probabilities": {"POSITIVE": 0.80, "NEUTRAL": 0.10, "NEGATIVE": 0.05, "AMBIGUOUS_OR_IRONY": 0.05}
            }
            res = mod.validate_deepseek_response(valid_floats)
            self.assertIsNotNone(res)
            self.assertTrue(res["entity_found"])
            self.assertAlmostEqual(res["probabilities"]["POSITIVE"], 0.80, places=2)

            # Valid percentages
            valid_pct = {
                "entity_found": True,
                "probabilities": {"POSITIVE": 70, "NEUTRAL": 20, "NEGATIVE": 10, "AMBIGUOUS_OR_IRONY": 0}
            }
            res_pct = mod.validate_deepseek_response(valid_pct)
            self.assertIsNotNone(res_pct)
            self.assertTrue(res_pct["entity_found"])
            self.assertAlmostEqual(res_pct["probabilities"]["POSITIVE"], 0.70, places=2)

            # Missing entity_found
            no_entity = {
                "probabilities": {"POSITIVE": 0.80, "NEUTRAL": 0.10, "NEGATIVE": 0.05, "AMBIGUOUS_OR_IRONY": 0.05}
            }
            self.assertIsNone(mod.validate_deepseek_response(no_entity))

            # All zeros
            all_zeros = {
                "entity_found": True,
                "probabilities": {"POSITIVE": 0.0, "NEUTRAL": 0.0, "NEGATIVE": 0.0, "AMBIGUOUS_OR_IRONY": 0.0}
            }
            self.assertIsNone(mod.validate_deepseek_response(all_zeros))

            # Out of bounds sum
            bad_sum = {
                "entity_found": True,
                "probabilities": {"POSITIVE": 0.5, "NEUTRAL": 0.2, "NEGATIVE": 0.1, "AMBIGUOUS_OR_IRONY": 0.0}
            }
            self.assertIsNone(mod.validate_deepseek_response(bad_sum))

    # -------------------------------------------------------------------------
    # 7. DeepSeek Failure Never Falls Back to Rejected Jev
    # -------------------------------------------------------------------------
    def test_deepseek_failure_never_falls_back_to_rejected_jev(self):
        """DeepSeek failure produces the explicit neutral default, never rejected Jev."""
        for mod in self.modules:
            analyzer = mod.OllamaSentimentAnalyzer()
            low_conf_jev = {
                "sentiment_probabilities": {"positive": 0.50, "neutral": 0.30, "negative": 0.10, "irony": 0.10},
                "sentiment_confidence": 0.50,
                "sentiment_choice": "positive",
                "entity_probabilities": {"relevant": 0.80, "unrelated": 0.10, "uncertain": 0.10},
                "entity_confidence": 0.80,
                "entity_choice": "relevant"
            }

            with patch.object(analyzer, "_call_typesafe_jev", return_value=(low_conf_jev, "typesafe/jev-1.13")), \
                 patch.object(analyzer, "_call_deepseek_fallback", return_value=(None, "")):
                context = {
                    "post_id": "999",
                    "actual_target": "SCB",
                    "project_name": "SCB Project",
                    "keywords": ["SCB"],
                    "source_info": "User=tester",
                    "clean_text": "พอใช้ได้",
                    "capped_text": "พอใช้ได้"
                }
                res = analyzer._hybrid_analyze_post(context)
                self.assertEqual(res["model"], "rule:provider_failure")
                self.assertEqual(res["sentiment"], "neutral")
                self.assertEqual(res["ai_sentiment"], 0)
                self.assertEqual(res["neutral_percent"], 100)

    # -------------------------------------------------------------------------
    # 8. Provider Reason Ignored & Deterministic Synthetic Reason
    # -------------------------------------------------------------------------
    def test_provider_reason_ignored(self):
        """Provider-returned reason text is ignored; synthetic reason engine is used"""
        for mod in self.modules:
            analyzer = mod.OllamaSentimentAnalyzer()
            valid_jev = {
                "sentiment_probabilities": {"positive": 0.85, "neutral": 0.10, "negative": 0.03, "irony": 0.02},
                "sentiment_confidence": 0.85,
                "sentiment_choice": "positive",
                "entity_probabilities": {"relevant": 0.90, "unrelated": 0.05, "uncertain": 0.05},
                "entity_confidence": 0.85,
                "entity_choice": "relevant",
                "reason": "This is provider reason that MUST BE IGNORED"
            }

            with patch.object(analyzer, "_call_typesafe_jev", return_value=(valid_jev, "typesafe/jev-1.13")):
                context = {
                    "post_id": "100",
                    "actual_target": "SCB",
                    "project_name": "SCB Project",
                    "keywords": ["SCB"],
                    "source_info": "User=tester",
                    "clean_text": "SCB ดีมาก",
                    "capped_text": "SCB ดีมาก"
                }
                res = analyzer._hybrid_analyze_post(context)
                self.assertIsNotNone(res)
                self.assertNotIn("This is provider reason", res["reason"])
                self.assertIn("ชื่นชม", res["reason"])
                self.assertEqual(res["route"], "jev")

    # -------------------------------------------------------------------------
    # 9. Single Paid Call Per Attempt
    # -------------------------------------------------------------------------
    def test_single_call_fast_path_and_slow_path(self):
        """Fast path calls Jev once, DeepSeek 0 times. Slow path calls Jev once, DeepSeek once."""
        for mod in self.modules:
            analyzer = mod.OllamaSentimentAnalyzer()
            valid_jev = {
                "sentiment_probabilities": {"positive": 0.85, "neutral": 0.10, "negative": 0.03, "irony": 0.02},
                "sentiment_confidence": 0.85,
                "sentiment_choice": "positive",
                "entity_probabilities": {"relevant": 0.90, "unrelated": 0.05, "uncertain": 0.05},
                "entity_confidence": 0.85,
                "entity_choice": "relevant"
            }
            valid_ds = {
                "entity_found": True,
                "probabilities": {"POSITIVE": 0.80, "NEUTRAL": 0.10, "NEGATIVE": 0.05, "AMBIGUOUS_OR_IRONY": 0.05}
            }

            context = {
                "post_id": "101",
                "actual_target": "SCB",
                "project_name": "SCB",
                "keywords": ["SCB"],
                "source_info": "User=test",
                "clean_text": "test",
                "capped_text": "test"
            }

            # 1. Fast path
            with patch.object(analyzer, "_call_typesafe_jev", return_value=(valid_jev, "typesafe/jev-1.13")) as mock_jev, \
                 patch.object(analyzer, "_call_deepseek_fallback", return_value=(valid_ds, "deepseek/deepseek-v4-flash-0731")) as mock_ds:
                res = analyzer._hybrid_analyze_post(context)
                self.assertIsNotNone(res)
                self.assertEqual(mock_jev.call_count, 1)
                self.assertEqual(mock_ds.call_count, 0)

            # 2. Slow path
            low_jev = dict(valid_jev, sentiment_confidence=0.50)
            with patch.object(analyzer, "_call_typesafe_jev", return_value=(low_jev, "typesafe/jev-1.13")) as mock_jev, \
                 patch.object(analyzer, "_call_deepseek_fallback", return_value=(valid_ds, "deepseek/deepseek-v4-flash-0731")) as mock_ds:
                res = analyzer._hybrid_analyze_post(context)
                self.assertIsNotNone(res)
                self.assertEqual(mock_jev.call_count, 1)
                self.assertEqual(mock_ds.call_count, 1)

    # -------------------------------------------------------------------------
    # 10. REST Payload Schema & No Internal Leakage
    # -------------------------------------------------------------------------
    def test_rest_payload_schema_no_leakage(self):
        """REST payload matches contract exactly and does not leak internal routing/confidence fields"""
        for mod in self.modules:
            mock_analyzer = MagicMock()
            mock_analyzer.analyze_post_sentiments.return_value = {
                "data": [
                    {
                        "post_id": "2001",
                        "ai_sentiment": 100,
                        "sentiment": "positive",
                        "positive_percent": 85,
                        "negative_percent": 5,
                        "neutral_percent": 10,
                        "irony_score": 0,
                        "confidence": 0.85,
                        "reason": "ผู้ใช้งานแสดงความชื่นชมต่อ TrueMoney",
                        "model": "typesafe/jev-1.13",
                        "route": "jev",
                        "conflict_reasons": []
                    }
                ]
            }

            api = mod.SentimentAPI(analyzer=mock_analyzer)
            posts = [{"match_post_id": 2001, "id": 2001, "post_id": "P-2001", "content": "TrueMoney ดี", "keywords": "TrueMoney"}]

            with patch.object(api, "fetch_pending", return_value=posts), \
                 patch.object(api, "bulk_update", return_value=1) as mock_bulk:
                api.run("2026-08-01", "2026-08-05", save_db=True)
                self.assertTrue(mock_bulk.called)
                payload_items = mock_bulk.call_args[0][0]
                item = payload_items[0]

                # Permitted fields
                expected_keys = {
                    "match_post_id", "id", "post_id", "sentiment", "sentiment_score",
                    "sentiment_status", "sentiment_reason", "ai_reason", "sentiment_scores"
                }
                self.assertEqual(set(item.keys()), expected_keys)
                self.assertEqual(item["match_post_id"], 2001)
                self.assertEqual(item["sentiment_scores"]["positive"], 85)
                self.assertEqual(item["sentiment_scores"]["negative"], 5)
                self.assertEqual(item["sentiment_scores"]["neutral"], 10)
                self.assertEqual(item["sentiment_scores"]["model"], "typesafe/jev-1.13")

                # Strictly forbidden internal leakage
                self.assertNotIn("route", item)
                self.assertNotIn("conflict_reasons", item)
                self.assertNotIn("confidence", item)
                self.assertNotIn("irony_score", item)
                self.assertNotIn("raw_probabilities", item)

    # -------------------------------------------------------------------------
    # 11. Target Sanitization and Length Cap
    # -------------------------------------------------------------------------
    def test_sanitize_target(self):
        """sanitize_target removes HTML tags, control chars, and caps at 100 chars"""
        for mod in self.modules:
            dirty = "<script>alert('xss')</script><b>TrueMoney\n\tBank</b>" + "A" * 200
            clean = mod.sanitize_target(dirty)
            self.assertNotIn("<script>", clean)
            self.assertNotIn("<b>", clean)
            self.assertNotIn("\n", clean)
            self.assertLessEqual(len(clean), 100)

    # -------------------------------------------------------------------------
    # 12. Text Capping at 8,000 Chars
    # -------------------------------------------------------------------------
    def test_cap_text(self):
        """cap_text preserves prefix, keyword window, and suffix within 8,000 chars"""
        for mod in self.modules:
            short_text = "Short text"
            self.assertEqual(mod.cap_text(short_text, max_chars=8000), short_text)

            huge_text = "Start " + ("X" * 10000) + " Middle KEYWORD " + ("Y" * 10000) + " End"
            capped = mod.cap_text(huge_text, max_chars=8000, keyword="KEYWORD")
            self.assertLessEqual(len(capped), 8050)  # small buffer for ellipsis
            self.assertIn("KEYWORD", capped)
            self.assertTrue(capped.startswith("Start"))
            self.assertTrue(capped.endswith("End"))

    # -------------------------------------------------------------------------
    # 13. Duplicate Post Deduplication
    # -------------------------------------------------------------------------
    def test_deduplicate_duplicate_post_ids(self):
        """analyze_post_sentiments deduplicates duplicate IDs in batch"""
        for mod in self.modules:
            analyzer = mod.OllamaSentimentAnalyzer()
            posts = [
                {"match_post_id": 101, "content": "Post 1", "keywords": "SCB"},
                {"match_post_id": 101, "content": "Post 1 duplicate", "keywords": "SCB"},
                {"match_post_id": 102, "content": "Post 2", "keywords": "SCB"},
            ]
            dummy_res = {
                "post_id": "101",
                "ai_sentiment": 100,
                "sentiment": "positive",
                "positive_percent": 90,
                "negative_percent": 5,
                "neutral_percent": 5,
                "irony_score": 0,
                "reason": "ดีมาก",
                "model": "typesafe/jev-1.13"
            }
            with patch.object(analyzer, "_analyze_single_post", return_value=dummy_res) as mock_single:
                res = analyzer.analyze_post_sentiments(posts)
                # Exactly 2 unique posts (101 and 102) should be processed
                self.assertEqual(mock_single.call_count, 2)
                self.assertEqual(len(res["data"]), 2)

    # -------------------------------------------------------------------------
    # 14. Jev 404 Candidate Cascade
    # -------------------------------------------------------------------------
    def test_jev_candidate_cascade_on_404(self):
        """When primary Jev model returns 404, it immediately tries the fallback candidate model"""
        for mod in self.modules:
            analyzer = mod.OllamaSentimentAnalyzer()
            valid_jev_json = {
                "answers": {
                    "sentiment": {
                        "confidence": 0.85,
                        "choice": "positive",
                        "probabilities": {"positive": 0.85, "neutral": 0.10, "negative": 0.03, "irony": 0.02}
                    },
                    "entity_relevance": {
                        "confidence": 0.85,
                        "choice": "relevant",
                        "probabilities": {"relevant": 0.90, "unrelated": 0.05, "uncertain": 0.05}
                    }
                }
            }

            def fake_post(url, headers=None, json=None, timeout=None):
                model = json.get("model")
                mock_resp = MagicMock()
                if model == mod.JEV_MODEL:
                    mock_resp.status_code = 404
                    mock_resp.json.return_value = {"error": "Not Found"}
                else:
                    mock_resp.status_code = 200
                    mock_resp.json.return_value = valid_jev_json
                return mock_resp

            with patch.object(analyzer.get_session(), "post", side_effect=fake_post):
                with patch.object(mod, "JEV_FALLBACK_MODELS", ["typesafe/jev-fallback"]):
                    res, used_model = analyzer._call_typesafe_jev("test state", "SCB")
                    self.assertIsNotNone(res)
                    self.assertEqual(used_model, "typesafe/jev-fallback")

    # -------------------------------------------------------------------------
    # 15. Jev 401/403 Authentication Abort
    # -------------------------------------------------------------------------
    def test_jev_auth_error_abort(self):
        """When Jev returns 401 or 403, cascade aborts immediately without trying fallback candidates"""
        for mod in self.modules:
            analyzer = mod.OllamaSentimentAnalyzer()
            call_count = 0

            def fake_post(url, headers=None, json=None, timeout=None):
                nonlocal call_count
                call_count += 1
                mock_resp = MagicMock()
                mock_resp.status_code = 401
                mock_resp.json.return_value = {"error": "Unauthorized"}
                return mock_resp

            with patch.object(analyzer.get_session(), "post", side_effect=fake_post):
                with patch.object(mod, "JEV_FALLBACK_MODELS", ["typesafe/jev-fallback"]):
                    res, used_model = analyzer._call_typesafe_jev("test state", "SCB")
                    self.assertIsNone(res)
                    # Must have called only once and aborted immediately
                    self.assertEqual(call_count, 1)

    # -------------------------------------------------------------------------
    # 16. Jev 429 Retry-After Handling
    # -------------------------------------------------------------------------
    def test_jev_429_retry_after_handling(self):
        """When Jev returns 429 with Retry-After header, it retries and succeeds"""
        for mod in self.modules:
            analyzer = mod.OllamaSentimentAnalyzer()
            calls = 0
            valid_jev_json = {
                "answers": {
                    "sentiment": {
                        "confidence": 0.85,
                        "choice": "positive",
                        "probabilities": {"positive": 0.85, "neutral": 0.10, "negative": 0.03, "irony": 0.02}
                    },
                    "entity_relevance": {
                        "confidence": 0.85,
                        "choice": "relevant",
                        "probabilities": {"relevant": 0.90, "unrelated": 0.05, "uncertain": 0.05}
                    }
                }
            }

            def fake_post(url, headers=None, json=None, timeout=None):
                nonlocal calls
                calls += 1
                mock_resp = MagicMock()
                if calls == 1:
                    mock_resp.status_code = 429
                    mock_resp.headers = {"Retry-After": "1"}
                    mock_resp.json.return_value = {"error": "Rate limit exceeded"}
                else:
                    mock_resp.status_code = 200
                    mock_resp.json.return_value = valid_jev_json
                return mock_resp

            with patch.object(analyzer.get_session(), "post", side_effect=fake_post), \
                 patch("time.sleep") as mock_sleep:
                res, used_model = analyzer._call_typesafe_jev("test state", "SCB")
                self.assertIsNotNone(res)
                self.assertEqual(calls, 2)
                mock_sleep.assert_called_with(1.0)

    # -------------------------------------------------------------------------
    # 17. Target Resolution Precedence
    # -------------------------------------------------------------------------
    def test_target_resolution_precedence(self):
        """Target precedence: explicit actual_target > resolver target > keywords > company > placeholder"""
        for mod in self.modules:
            analyzer = mod.OllamaSentimentAnalyzer()

            captured_targets = []
            def fake_hybrid(ctx):
                captured_targets.append(ctx["actual_target"])
                return {
                    "post_id": ctx["post_id"],
                    "ai_sentiment": 100,
                    "sentiment": "positive",
                    "positive_percent": 85,
                    "negative_percent": 5,
                    "neutral_percent": 10,
                    "irony_score": 0,
                    "reason": "OK",
                    "model": "jev"
                }

            with patch.object(analyzer, "_hybrid_analyze_post", side_effect=fake_hybrid):
                # 1. Explicit actual_target provided
                post1 = {"match_post_id": 1, "content": "Post 1", "actual_target": "ExplicitTarget", "company_name": "Comp", "keywords": "Kw"}
                analyzer._analyze_single_post(post1, company_name="Comp")
                self.assertEqual(captured_targets[-1], "ExplicitTarget")

                # 2. Keywords fallback when no resolver & no explicit target
                with patch.object(mod, "GLOBAL_PROJECT_RESOLVER", None):
                    post2 = {"match_post_id": 2, "content": "Post 2", "company_name": "Comp", "keywords": "Kw1, Kw2"}
                    analyzer._analyze_single_post(post2, company_name="Comp")
                    self.assertEqual(captured_targets[-1], "Kw1, Kw2")

                    # 3. Company fallback
                    post3 = {"match_post_id": 3, "content": "Post 3", "company_name": "CompOnly"}
                    analyzer._analyze_single_post(post3, company_name="CompOnly")
                    self.assertEqual(captured_targets[-1], "CompOnly")

                    # 4. Placeholder fallback
                    post4 = {"match_post_id": 4, "content": "Post 4"}
                    analyzer._analyze_single_post(post4)
                    self.assertEqual(captured_targets[-1], "the Target Entity")

    # -------------------------------------------------------------------------
    # 18. REST-only architecture
    # -------------------------------------------------------------------------
    def test_architectural_role_parity(self):
        """The production worker exposes the hybrid router and REST client, not Direct DB."""
        self.assertEqual(sentiment_module.JEV_ACCEPTANCE_THRESHOLD, 0.65)
        self.assertTrue(sentiment_module.ENABLE_JEV_HYBRID)
        self.assertTrue(hasattr(sentiment_module, "OllamaSentimentAnalyzer"))
        self.assertTrue(hasattr(sentiment_module, "SentimentAPI"))
        self.assertFalse(hasattr(sentiment_module, "SentimentDB"))


if __name__ == "__main__":
    unittest.main()
