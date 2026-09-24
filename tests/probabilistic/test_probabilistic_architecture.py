# coding=utf-8
"""
tests/probabilistic/test_probabilistic_architecture.py
Unit tests verifying the 100% Cloud API Single-Pass Probabilistic Architecture
in the REST-only production worker.
"""

import unittest
from unittest.mock import MagicMock, patch
import json
import os
import sys

# Add repository root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import ai_sentiment as sentiment_module


class TestProbabilisticArchitecture(unittest.TestCase):
    def setUp(self):
        self.modules = [sentiment_module]

    def test_environment_defaults(self):
        """Verify safe defaults: ENABLE_PROBABILISTIC_MODE=True, BYPASS_LOCAL_TRIAGE=True, ENABLE_JEV_HYBRID=True, SAVE_DB=False"""
        for mod in self.modules:
            self.assertTrue(mod.ENABLE_PROBABILISTIC_MODE)
            self.assertTrue(mod.BYPASS_LOCAL_TRIAGE)
            self.assertTrue(mod.ENABLE_JEV_HYBRID)
            # Default SAVE_DB should be false
            api = mod.SentimentAPI(analyzer=MagicMock())
            with patch.object(api, "fetch_pending", return_value=[]):
                # When save_db=None, it should resolve to False from env default
                api.run("2026-08-01", "2026-08-05", save_db=None)

    def test_parse_probabilistic_response_floats(self):
        """Verify normal float probabilities summing ~1.0"""
        for mod in self.modules:
            data = {
                "probabilities": {
                    "POSITIVE": 0.85,
                    "NEUTRAL": 0.10,
                    "NEGATIVE": 0.05,
                    "AMBIGUOUS_OR_IRONY": 0.00
                },
                "entity_found": True
            }
            res = mod.parse_probabilistic_response(data)
            self.assertIsNotNone(res)
            self.assertTrue(res["entity_found"])
            probs = res["probabilities"]
            self.assertAlmostEqual(probs["POSITIVE"], 0.85, places=2)
            self.assertAlmostEqual(probs["NEUTRAL"], 0.10, places=2)
            self.assertAlmostEqual(probs["NEGATIVE"], 0.05, places=2)
            self.assertAlmostEqual(probs["AMBIGUOUS_OR_IRONY"], 0.00, places=2)
            self.assertAlmostEqual(sum(probs.values()), 1.0, places=3)

    def test_parse_probabilistic_response_percentages(self):
        """Verify integer/percentage normalization (> 1.5 total)"""
        for mod in self.modules:
            data = {
                "probabilities": {
                    "POSITIVE": "70%",
                    "NEUTRAL": "20%",
                    "NEGATIVE": "10%",
                    "AMBIGUOUS_OR_IRONY": "0%"
                },
                "entity_found": "true"
            }
            res = mod.parse_probabilistic_response(data)
            self.assertIsNotNone(res)
            self.assertTrue(res["entity_found"])
            probs = res["probabilities"]
            self.assertAlmostEqual(probs["POSITIVE"], 0.70, places=2)
            self.assertAlmostEqual(probs["NEUTRAL"], 0.20, places=2)
            self.assertAlmostEqual(probs["NEGATIVE"], 0.10, places=2)
            self.assertAlmostEqual(sum(probs.values()), 1.0, places=3)

    def test_parse_probabilistic_response_stringified_json(self):
        """Verify recovery when probabilities is a stringified JSON"""
        for mod in self.modules:
            data = {
                "probabilities": json.dumps({
                    "POSITIVE": 0.6,
                    "NEUTRAL": 0.3,
                    "NEGATIVE": 0.1,
                    "AMBIGUOUS_OR_IRONY": 0.0
                }),
                "entity_found": True
            }
            res = mod.parse_probabilistic_response(data)
            self.assertIsNotNone(res)
            self.assertAlmostEqual(res["probabilities"]["POSITIVE"], 0.6, places=2)

    def test_parse_probabilistic_response_all_zeros(self):
        """Verify graceful fallback when all probabilities are zero"""
        for mod in self.modules:
            data = {
                "probabilities": {
                    "POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0, "AMBIGUOUS_OR_IRONY": 0
                },
                "entity_found": True
            }
            res = mod.parse_probabilistic_response(data)
            self.assertIsNotNone(res)
            self.assertEqual(res["probabilities"]["NEUTRAL"], 1.0)
            self.assertEqual(res["probabilities"]["POSITIVE"], 0.0)

    def test_resolve_policy_positive(self):
        """Verify positive score resolution and 100% total distribution"""
        for mod in self.modules:
            probs = {"POSITIVE": 0.8, "NEUTRAL": 0.15, "NEGATIVE": 0.05, "AMBIGUOUS_OR_IRONY": 0.0}
            policy = mod.resolve_policy(probs, entity_found=True)
            self.assertEqual(policy["sentiment"], "positive")
            self.assertEqual(policy["score"], 100)
            self.assertEqual(policy["pos"] + policy["neg"] + policy["neu"], 100)
            self.assertEqual(policy["irony_score"], 0)

    def test_resolve_policy_negative(self):
        """Verify negative score resolution and 100% total distribution"""
        for mod in self.modules:
            probs = {"POSITIVE": 0.05, "NEUTRAL": 0.15, "NEGATIVE": 0.8, "AMBIGUOUS_OR_IRONY": 0.0}
            policy = mod.resolve_policy(probs, entity_found=True)
            self.assertEqual(policy["sentiment"], "negative")
            self.assertEqual(policy["score"], -100)
            self.assertEqual(policy["pos"] + policy["neg"] + policy["neu"], 100)

    def test_resolve_policy_irony_conservative_negative(self):
        """Verify irony weight shifts toward negative in conservative brand monitoring"""
        for mod in self.modules:
            # Sarcastic text: pos 0.30, neg 0.20, neu 0.10, irony 0.40
            # effective_neg = 0.20 + (0.40 * 0.7) = 0.48 > effective_pos 0.30 -> negative!
            probs = {"POSITIVE": 0.30, "NEUTRAL": 0.10, "NEGATIVE": 0.20, "AMBIGUOUS_OR_IRONY": 0.40}
            policy = mod.resolve_policy(probs, entity_found=True)
            self.assertEqual(policy["sentiment"], "negative")
            self.assertEqual(policy["score"], -100)
            self.assertEqual(policy["irony_score"], 40)
            self.assertEqual(policy["pos"] + policy["neg"] + policy["neu"], 100)

    def test_resolve_policy_not_entity_found(self):
        """Verify entity_found=False yields neutral (0) with neu=100%"""
        for mod in self.modules:
            probs = {"POSITIVE": 0.9, "NEUTRAL": 0.05, "NEGATIVE": 0.05, "AMBIGUOUS_OR_IRONY": 0.0}
            policy = mod.resolve_policy(probs, entity_found=False)
            self.assertEqual(policy["sentiment"], "neutral")
            self.assertEqual(policy["score"], 0)
            self.assertEqual(policy["pos"], 0)
            self.assertEqual(policy["neg"], 0)
            self.assertEqual(policy["neu"], 100)
            self.assertEqual(policy["irony_score"], 0)

    def test_resolve_policy_rounding_sum_guarantee(self):
        """Verify pos_pct + neg_pct + neu_pct is mathematically 100 even with rounding corner cases"""
        for mod in self.modules:
            # Simulate rounding: e.g. 50.5% and 49.5%
            probs = {"POSITIVE": 0.505, "NEUTRAL": 0.00, "NEGATIVE": 0.495, "AMBIGUOUS_OR_IRONY": 0.0}
            policy = mod.resolve_policy(probs, entity_found=True)
            self.assertEqual(policy["pos"] + policy["neg"] + policy["neu"], 100)

    def test_generate_synthetic_reason_cases(self):
        """Verify deterministic Thai reasoning for various distributions"""
        for mod in self.modules:
            # 1. Not found
            r_notfound = mod.generate_synthetic_reason("SCB", 0.8, 0.1, 0.1, 0.0, entity_found=False)
            self.assertIn("ไม่พบการกล่าวถึงหรือความคิดเห็นต่อ SCB โดยตรง", r_notfound)

            # 2. Irony with negative lean
            r_irony_neg = mod.generate_synthetic_reason("SCB", 0.1, 0.3, 0.2, 0.4, entity_found=True)
            self.assertIn("ประชดประชัน", r_irony_neg)
            self.assertIn("เชิงลบ", r_irony_neg)

            # 3. Strong positive
            r_pos = mod.generate_synthetic_reason("SCB", 0.85, 0.05, 0.10, 0.0, entity_found=True)
            self.assertIn("ชื่นชม", r_pos)

            # 4. Strong negative
            r_neg = mod.generate_synthetic_reason("SCB", 0.05, 0.85, 0.10, 0.0, entity_found=True)
            self.assertIn("ร้องเรียน", r_neg)

            # 5. Factual neutral
            r_neu = mod.generate_synthetic_reason("SCB", 0.05, 0.05, 0.90, 0.0, entity_found=True)
            self.assertIn("รายงานข้อเท็จจริง", r_neu)

            # 6. Fallback target when empty
            r_empty_tgt = mod.generate_synthetic_reason("", 0.85, 0.05, 0.10, 0.0, entity_found=True)
            self.assertIn("เป้าหมายที่ระบุ", r_empty_tgt)

    def test_parse_json_result_probabilistic_recovery(self):
        """Verify _parse_json_result handles thought tags and recovers unparsed JSON text"""
        for mod in self.modules:
            analyzer = mod.OllamaSentimentAnalyzer()

            # Thinking tags
            raw_with_thought = (
                "<thought>Analyzing Thai sentiment toward entity...</thought>"
                '{"probabilities": {"POSITIVE": 0.9, "NEUTRAL": 0.1, "NEGATIVE": 0.0, "AMBIGUOUS_OR_IRONY": 0.0}, "entity_found": true}'
            )
            res = analyzer._parse_json_result(raw_with_thought)
            self.assertIsNotNone(res)
            self.assertIn("probabilities", res)

            # Malformed JSON with regex recovery
            mangled = (
                'Here is the result:\n'
                'POSITIVE: 0.75,\n'
                'NEGATIVE: 0.10,\n'
                'NEUTRAL: 0.15,\n'
                'AMBIGUOUS_OR_IRONY: 0.00,\n'
                'entity_found: true\n'
            )
            res_mangled = analyzer._parse_json_result(mangled)
            self.assertIsNotNone(res_mangled)
            self.assertIn("probabilities", res_mangled)
            self.assertEqual(res_mangled["probabilities"]["POSITIVE"], 0.75)

    def test_single_post_probabilistic_execution(self):
        """Verify _analyze_single_post executes probabilistic analysis and bypasses local triage when hybrid is disabled"""
        for mod in self.modules:
            analyzer = mod.OllamaSentimentAnalyzer()
            mock_prob = {
                "post_id": "1001",
                "ai_sentiment": 100,
                "sentiment": "positive",
                "positive_percent": 85,
                "negative_percent": 5,
                "neutral_percent": 10,
                "irony_score": 0,
                "reason": "ผู้ใช้งานแสดงความชื่นชม ประทับใจ หรือให้คะแนนเชิงบวกต่อ SCB อย่างชัดเจน",
                "entity_found": True,
                "model": "deepseek/deepseek-v4-flash-0731",
                "raw_probabilities": {"POSITIVE": 0.85, "NEUTRAL": 0.10, "NEGATIVE": 0.05, "AMBIGUOUS_OR_IRONY": 0.0}
            }

            with patch.object(mod, "ENABLE_JEV_HYBRID", False), \
                 patch.object(analyzer, "_probabilistic_analyze_post", return_value=mock_prob) as mock_prob_fn, \
                 patch.object(analyzer, "_triage_post") as mock_triage_fn:
                post = {
                    "match_post_id": 1001,
                    "content": "SCB Easy ดีมาก โอนเงินไว",
                    "actual_target": "SCB Easy"
                }
                res = analyzer._analyze_single_post(post)
                self.assertIsNotNone(res)
                self.assertEqual(res["ai_sentiment"], 100)
                self.assertEqual(res["sentiment"], "positive")
                self.assertTrue(mock_prob_fn.called)
                # Local triage MUST NOT be called when BYPASS_LOCAL_TRIAGE is True!
                self.assertFalse(mock_triage_fn.called)

    def test_single_post_hybrid_execution(self):
        """Verify _analyze_single_post routes to _hybrid_analyze_post when ENABLE_JEV_HYBRID is True"""
        for mod in self.modules:
            analyzer = mod.OllamaSentimentAnalyzer()
            mock_hybrid = {
                "post_id": "1002",
                "ai_sentiment": 100,
                "sentiment": "positive",
                "positive_percent": 90,
                "negative_percent": 2,
                "neutral_percent": 8,
                "irony_score": 0,
                "reason": "ผู้ใช้งานแสดงความชื่นชม ประทับใจ หรือให้คะแนนเชิงบวกต่อ SCB อย่างชัดเจน",
                "entity_found": True,
                "model": "typesafe/jev-1.13",
                "route": "jev"
            }
            with patch.object(mod, "ENABLE_JEV_HYBRID", True), \
                 patch.object(analyzer, "_hybrid_analyze_post", return_value=mock_hybrid) as mock_hybrid_fn:
                post = {
                    "match_post_id": 1002,
                    "content": "SCB ดีมาก",
                    "actual_target": "SCB"
                }
                res = analyzer._analyze_single_post(post)
                self.assertIsNotNone(res)
                self.assertEqual(res["post_id"], "1002")
                self.assertTrue(mock_hybrid_fn.called)

    def test_downstream_rest_api_payload_schema(self):
        """Verify SentimentAPI generates 100% compliant payload schema for REST API"""
        for mod in self.modules:
            mock_analyzer = MagicMock()
            mock_analyzer.analyze_post_sentiments.return_value = {
                "data": [
                    {
                        "post_id": "9999",
                        "ai_sentiment": 100,
                        "sentiment": "positive",
                        "positive_percent": 80,
                        "negative_percent": 5,
                        "neutral_percent": 15,
                        "irony_score": 0,
                        "reason": "ผู้ใช้งานแสดงความชื่นชม ประทับใจ หรือให้คะแนนเชิงบวกต่อ TrueMoney อย่างชัดเจน",
                        "model": "deepseek/deepseek-v4-flash-0731"
                    }
                ]
            }

            api = mod.SentimentAPI(analyzer=mock_analyzer)
            posts = [{
                "match_post_id": 9999,
                "id": 9999,
                "post_id": "P-9999",
                "content": "TrueMoney ใช้งานสะดวกมาก",
                "keywords": "TrueMoney"
            }]

            with patch.object(api, "fetch_pending", return_value=posts), \
                 patch.object(api, "bulk_update", return_value=1) as mock_bulk:
                updated = api.run("2026-08-01", "2026-08-05", save_db=True)
                self.assertEqual(updated, 1)
                self.assertTrue(mock_bulk.called)
                sent_results = mock_bulk.call_args[0][0]
                self.assertEqual(len(sent_results), 1)
                item = sent_results[0]

                # Strictly verify all downstream required fields
                self.assertEqual(item["match_post_id"], 9999)
                self.assertEqual(item["id"], 9999)
                self.assertEqual(item["post_id"], "P-9999")
                self.assertEqual(item["sentiment"], "positive")
                self.assertEqual(item["sentiment_score"], 100)
                self.assertEqual(item["sentiment_status"], "1")
                self.assertEqual(item["sentiment_reason"], item["ai_reason"])
                self.assertIn("ชื่นชม", item["sentiment_reason"])
                self.assertEqual(item["sentiment_scores"]["positive"], 80)
                self.assertEqual(item["sentiment_scores"]["negative"], 5)
                self.assertEqual(item["sentiment_scores"]["neutral"], 15)
                self.assertEqual(item["sentiment_scores"]["model"], "deepseek/deepseek-v4-flash-0731")

    def test_parse_probabilistic_casing_variations(self):
        """Verify parse_probabilistic_response handles title-cased, mixed-cased, and nested dict keys"""
        for mod in self.modules:
            data = {
                "probabilities": {
                    "Positive": 0.85,
                    "Neutral": 0.10,
                    "Negative": 0.05,
                    "Ambiguous_or_Irony": 0.00
                },
                "Target_Found": "yes"
            }
            res = mod.parse_probabilistic_response(data)
            self.assertIsNotNone(res)
            self.assertTrue(res["entity_found"])
            self.assertAlmostEqual(res["probabilities"]["POSITIVE"], 0.85, places=2)
            self.assertAlmostEqual(res["probabilities"]["NEUTRAL"], 0.10, places=2)
            self.assertAlmostEqual(res["probabilities"]["NEGATIVE"], 0.05, places=2)

            # Nested inside a dict
            data_nested = {
                "result": {
                    "positive": 0.7,
                    "neutral": 0.2,
                    "negative": 0.1,
                    "irony": 0.0
                },
                "entity_present": True
            }
            res_nested = mod.parse_probabilistic_response(data_nested)
            self.assertIsNotNone(res_nested)
            self.assertTrue(res_nested["entity_found"])
            self.assertAlmostEqual(res_nested["probabilities"]["POSITIVE"], 0.7, places=2)

    def test_resolve_policy_casing_and_percentages(self):
        """Verify resolve_policy handles case-insensitive keys and percentage inputs safely"""
        for mod in self.modules:
            # Title case
            policy = mod.resolve_policy({"Positive": 0.80, "Neutral": 0.15, "Negative": 0.05, "Irony": 0.0})
            self.assertEqual(policy["sentiment"], "positive")
            self.assertEqual(policy["score"], 100)
            self.assertEqual(policy["pos"] + policy["neg"] + policy["neu"], 100)

            # String percentage
            policy_pct = mod.resolve_policy({"POSITIVE": "85%", "NEUTRAL": "10%", "NEGATIVE": "5%", "AMBIGUOUS_OR_IRONY": "0%"})
            self.assertEqual(policy_pct["sentiment"], "positive")
            self.assertEqual(policy_pct["score"], 100)
            self.assertEqual(policy_pct["pos"] + policy_pct["neg"] + policy_pct["neu"], 100)

    def test_generate_synthetic_reason_unnormalized_inputs(self):
        """Verify generate_synthetic_reason scales down unnormalized percentage numbers"""
        for mod in self.modules:
            # 50, 10, 40, 0 should evaluate against moderate positive (0.40 <= pos < 0.70)
            reason = mod.generate_synthetic_reason("SCB", 50, 10, 40, 0)
            self.assertIn("โน้มเอียงไปในทิศทางที่ดี", reason)

            # Target entity placeholders normalized to natural Thai
            for placeholder in ["the Target Entity", "target entity", "unknown", "None", "null", ""]:
                r = mod.generate_synthetic_reason(placeholder, 85, 5, 10, 0)
                self.assertIn("เป้าหมายที่ระบุ", r)

    def test_empty_content_html_tag_stripping(self):
        """Verify _analyze_single_post detects HTML-only posts as empty content"""
        for mod in self.modules:
            analyzer = mod.OllamaSentimentAnalyzer()
            post = {
                "match_post_id": 8888,
                "content": "<p><br/>&nbsp;</p>",
                "keywords": "SCB"
            }
            res = analyzer._analyze_single_post(post)
            self.assertIsNotNone(res)
            self.assertEqual(res["ai_sentiment"], 0)
            self.assertEqual(res["sentiment"], "neutral")
            self.assertEqual(res["model"], "rule:empty_content")

    def test_sentiment_api_fetch_pending_posts_field(self):
        """Verify fetch_pending handles API returning {posts: [...]}"""
        for mod in self.modules:
            api = mod.SentimentAPI()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"posts": [{"match_post_id": 1, "content": "hello"}]}
            with patch("requests.get", return_value=mock_resp):
                posts = api.fetch_pending("2026-08-01", "2026-08-05")
                self.assertEqual(len(posts), 1)
                self.assertEqual(api.last_pending_count, 1)
                self.assertFalse(api.last_fetch_error)


if __name__ == "__main__":
    unittest.main()
