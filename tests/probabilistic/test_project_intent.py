"""Offline checks for project targeting, intent, and REST output."""

import os
import json
import unittest
from unittest.mock import MagicMock, patch

os.environ["PYTHON_DOTENV_DISABLED"] = "1"

import ai_sentiment as module


def jev_response(intent="information", related=True):
    return {
        "answers": {
            "sentiment": {
                "choice": "neutral", "confidence": 0.9,
                "probabilities": {"positive": 0.02, "neutral": 0.94,
                                  "negative": 0.02, "irony": 0.02},
            },
            "entity_relevance": {
                "choice": "relevant" if related else "unrelated", "confidence": 0.9,
                "probabilities": {"relevant": 0.94, "unrelated": 0.03, "uncertain": 0.03}
                if related else {"relevant": 0.03, "unrelated": 0.94, "uncertain": 0.03},
            },
            "intent": {"choice": intent, "confidence": 0.9},
        }
    }


class ProjectIntentTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = module.OllamaSentimentAnalyzer()

    def test_keyword_without_real_target_analyzes_overall_post(self):
        post = {"match_post_id": "missing-target", "keywords": ["ค่าไฟ"],
                "content": "ค่าไฟแพงขึ้นอีกแล้ว", "_analysis_scope": "keyword",
                "actual_target": "Wrong target", "company_name": "Wrong company"}
        raw = jev_response("complaint")
        raw["answers"]["sentiment"] = {
            "choice": "negative", "confidence": 0.9,
            "probabilities": {"positive": 0.02, "neutral": 0.04,
                              "negative": 0.92, "irony": 0.02},
        }
        del raw["answers"]["entity_relevance"]
        accepted = module.validate_jev_response(raw, overall_scope=True)
        with patch.object(self.analyzer, "_call_typesafe_jev", return_value=(accepted, "jev")) as jev, \
             patch.object(self.analyzer, "_call_deepseek_fallback") as deepseek:
            result = self.analyzer._analyze_single_post(post)
        self.assertEqual(result["sentiment"], "negative")
        self.assertEqual(result["intent"], "complaint")
        self.assertEqual(result["route"], "jev")
        self.assertTrue(result["entity_found"])
        self.assertIn("Scope=overall post sentiment and intent", jev.call_args.args[0])
        self.assertNotIn("Target=", jev.call_args.args[0])
        self.assertNotIn("Wrong target", jev.call_args.args[0])
        self.assertNotIn("Wrong company", jev.call_args.args[0])
        self.assertTrue(jev.call_args.kwargs["overall_scope"])
        deepseek.assert_not_called()

    def test_overall_post_deepseek_keeps_sentiment_without_named_entity(self):
        post = {"match_post_id": "overall-fallback", "keywords": ["ค่าไฟ"],
                "content": "ค่าไฟแพงมาก", "_analysis_scope": "keyword"}
        fallback = {"probabilities": {"POSITIVE": 0.02, "NEUTRAL": 0.05,
                                      "NEGATIVE": 0.91, "AMBIGUOUS_OR_IRONY": 0.02},
                    "entity_found": False, "intent": "complaint"}
        with patch.object(self.analyzer, "_call_typesafe_jev", return_value=(None, "")), \
             patch.object(self.analyzer, "_call_deepseek_fallback", return_value=(fallback, "deepseek")):
            result = self.analyzer._analyze_single_post(post)
        self.assertEqual((result["sentiment"], result["intent"], result["route"]),
                         ("negative", "complaint", "deepseek"))
        self.assertTrue(result["entity_found"])

    def test_overall_jev_asks_only_sentiment_and_intent(self):
        raw = jev_response("information")
        del raw["answers"]["entity_relevance"]
        response = MagicMock(status_code=200)
        response.json.return_value = raw
        session = MagicMock()
        session.post.return_value = response
        with patch.object(module, "OPENROUTER_API_KEY", "offline-test-key"), \
             patch.object(self.analyzer, "get_session", return_value=session):
            result, _ = self.analyzer._call_typesafe_jev(
                "Scope=overall post sentiment and intent\nText=hello", module.OVERALL_POST_LABEL,
                overall_scope=True)
        self.assertIsNotNone(result)
        self.assertEqual(result["entity_choice"], "relevant")
        self.assertEqual(set(session.post.call_args.kwargs["json"]["questions"]),
                         {"sentiment", "intent"})

    def test_overall_deepseek_prompt_has_no_project_target(self):
        context = {"post_id": "overall-deepseek", "actual_target": module.OVERALL_POST_LABEL,
                   "sentiment_target": module.OVERALL_POST_LABEL, "project_name": "",
                   "analysis_scope": "overall", "keywords": ["ค่าไฟ"],
                   "source_info": "Publisher=reader", "clean_text": "ค่าไฟแพงมาก",
                   "capped_text": "ค่าไฟแพงมาก"}
        response = MagicMock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": json.dumps({
            "probabilities": {"POSITIVE": 0.02, "NEUTRAL": 0.05,
                              "NEGATIVE": 0.91, "AMBIGUOUS_OR_IRONY": 0.02},
            "entity_found": True, "intent": "complaint"
        })}}]}
        session = MagicMock()
        session.post.return_value = response
        with patch.object(module, "OPENROUTER_API_KEY", "offline-test-key"), \
             patch.object(self.analyzer, "get_session", return_value=session):
            result, _ = self.analyzer._call_deepseek_fallback(context)
        self.assertEqual(result["intent"], "complaint")
        messages = session.post.call_args.kwargs["json"]["messages"]
        self.assertEqual(messages[0]["content"], module.OVERALL_SYSTEM_PROMPT)
        self.assertIn("Scope=overall post sentiment and intent", messages[1]["content"])
        self.assertNotIn("Target=", messages[1]["content"])

    def test_rest_submits_overall_sentiment_when_project_is_missing(self):
        raw = jev_response("complaint")
        raw["answers"]["sentiment"] = {
            "choice": "negative", "confidence": 0.9,
            "probabilities": {"positive": 0.02, "neutral": 0.04,
                              "negative": 0.92, "irony": 0.02},
        }
        del raw["answers"]["entity_relevance"]
        accepted = module.validate_jev_response(raw, overall_scope=True)
        with patch.dict(os.environ, {"SENTIMENT_CACHE_ENABLED": "false"}):
            api = module.SentimentAPI(analyzer=self.analyzer)
        post = {"match_post_id": "overall-rest", "keywords": ["ค่าไฟ"],
                "content": "ค่าไฟแพงมาก"}
        with patch.object(api, "fetch_pending", return_value=[post]), \
             patch.object(self.analyzer, "_call_typesafe_jev", return_value=(accepted, "jev")), \
             patch.object(api, "bulk_update", return_value=1) as update:
            self.assertEqual(api.run("2026-09-25", "2026-09-25", save_db=True), 1)
        sent = update.call_args.args[0][0]
        self.assertEqual(sent["sentiment"], "negative")
        self.assertEqual(sent["intent"], "complaint")
        self.assertGreaterEqual(sent["sentiment_scores"]["negative"], 90)

    def test_project_is_target_and_keyword_only_selects_excerpt(self):
        long_text = "ต้นข่าว " + "ก" * 3500 + "BLCP" + "ก" * 1000 + " ค่าไฟ " + "ข" * 4000 + " ท้ายข่าว"
        post = {
            "match_post_id": "p1", "project_name": "BLCP", "keywords": ["ค่าไฟ"],
            "content": long_text, "_analysis_scope": "keyword",
            "actual_target": "Wrong target", "company_name": "Wrong company",
            "post_user": "BLCP Official", "feed_link": "https://example.org/posts/BLCP?tracking=1",
        }
        accepted = module.validate_jev_response(jev_response())
        with patch.object(self.analyzer, "_call_typesafe_jev", return_value=(accepted, "jev")) as jev:
            result = self.analyzer._analyze_single_post(post)
        state = jev.call_args.args[0]
        self.assertEqual(jev.call_args.args[1], "BLCP")
        self.assertIn("Target=BLCP\n", state)
        self.assertNotIn("Wrong target", state)
        self.assertNotIn("Wrong company", state)
        self.assertNotIn("Target keywords=", state)
        self.assertIn("Publisher=BLCP Official", state)
        self.assertIn("Source URL=example.org/posts/BLCP", state)
        self.assertNotIn("tracking=1", state)
        excerpt = state.split("Text=", 1)[1]
        self.assertIn("ค่าไฟ", excerpt)
        self.assertIn("BLCP", excerpt)
        self.assertIn("ต้นข่าว", excerpt)
        self.assertIn("ท้ายข่าว", excerpt)
        self.assertLessEqual(len(excerpt), module.JEV_TEXT_MAX_CHARS)
        self.assertEqual(result["intent"], "information")

    def test_unrelated_keyword_is_neutral_and_keeps_intent(self):
        unrelated = module.validate_jev_response(jev_response("complaint", related=False))
        post = {"match_post_id": "p2", "project_name": "BLCP", "keywords": ["ค่าไฟ"],
                "content": "ค่าไฟแพงมากในเดือนนี้", "_analysis_scope": "keyword"}
        with patch.object(self.analyzer, "_call_typesafe_jev", return_value=(unrelated, "jev")):
            result = self.analyzer._analyze_single_post(post)
        self.assertEqual(result["sentiment"], "neutral")
        self.assertFalse(result["entity_found"])
        self.assertEqual(result["intent"], "complaint")

    def test_short_latin_target_does_not_match_inside_other_words(self):
        self.assertEqual(module._find_term_index("The actor will appear on stage", "PEA"), -1)
        self.assertEqual(module._find_term_index("PEA announced a new service", "PEA"), 0)
        text = "start " + "x" * 2000 + " appeal " + "x" * 3000 + " visitors " + "y" * 4000
        excerpt = module.cap_text(text, max_chars=3000, keyword="visitors", target="PEA")
        self.assertNotIn("appeal", excerpt)

    def test_four_intents_and_invalid_provider_values(self):
        for intent in ("complaint", "information", "recommendation", "enquiry"):
            with self.subTest(intent=intent):
                self.assertEqual(module.validate_jev_response(jev_response(intent))["intent"], intent)
                response = {"probabilities": {"POSITIVE": 0.05, "NEUTRAL": 0.85,
                                              "NEGATIVE": 0.05, "AMBIGUOUS_OR_IRONY": 0.05},
                            "entity_found": True, "intent": intent}
                self.assertEqual(module.validate_deepseek_response(response)["intent"], intent)
        self.assertIsNone(module.validate_jev_response(jev_response("other"))["intent"])
        self.assertIsNone(module.validate_deepseek_response({
            "probabilities": {"POSITIVE": 0, "NEUTRAL": 1,
                              "NEGATIVE": 0, "AMBIGUOUS_OR_IRONY": 0},
            "entity_found": True, "intent": ["complaint"]})["intent"])

    def test_prompts_distinguish_official_news_and_independent_opinion(self):
        self.assertIn("self-praise are neutral", module.KEYWORD_SYSTEM_PROMPT)
        self.assertIn("independent person's opinion", module.KEYWORD_SYSTEM_PROMPT)
        self.assertIn("Publisher and URL are context", module.KEYWORD_SYSTEM_PROMPT)
        self.assertIn("pure praise", module.KEYWORD_SYSTEM_PROMPT)

    def test_deepseek_intent_wins_and_jev_intent_fills_missing_value(self):
        context = {"post_id": "p3", "actual_target": "BLCP", "sentiment_target": "BLCP",
                   "project_name": "BLCP"}
        deepseek_result = {
            "probabilities": {"POSITIVE": 0.05, "NEUTRAL": 0.85,
                              "NEGATIVE": 0.05, "AMBIGUOUS_OR_IRONY": 0.05},
            "entity_found": True, "intent": "enquiry",
        }
        with patch.object(self.analyzer, "_call_deepseek_fallback",
                          return_value=(deepseek_result, "deepseek")):
            result = self.analyzer._complete_deepseek_route(
                context, {"conflict_reasons": []}, {"intent": "complaint"})
        self.assertEqual(result["intent"], "enquiry")
        deepseek_result["intent"] = None
        with patch.object(self.analyzer, "_call_deepseek_fallback",
                          return_value=(deepseek_result, "deepseek")):
            result = self.analyzer._complete_deepseek_route(
                context, {"conflict_reasons": []}, {"intent": "complaint"})
        self.assertEqual(result["intent"], "complaint")

    def test_rest_payload_includes_only_valid_related_intent(self):
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_post_sentiments.return_value = {"data": [
            {"post_id": "1", "ai_sentiment": 0, "sentiment": "neutral", "intent": "enquiry",
             "entity_found": True, "model": "jev"},
            {"post_id": "2", "ai_sentiment": 0, "sentiment": "neutral", "intent": "complaint",
             "entity_found": False, "model": "jev"},
            {"post_id": "3", "ai_sentiment": 0, "sentiment": "neutral", "intent": "unknown",
             "entity_found": True, "model": "jev"},
        ]}
        api = module.SentimentAPI(analyzer=mock_analyzer)
        posts = [{"match_post_id": index, "project_name": "BLCP", "keywords": ["ค่าไฟ"],
                  "content": "BLCP ค่าไฟ"} for index in (1, 2, 3)]
        with patch.object(api, "fetch_pending", return_value=posts), \
             patch.object(api, "bulk_update", return_value=3) as bulk:
            api.run("2026-09-25", "2026-09-25", save_db=True)
        sent = {str(row["match_post_id"]): row for row in bulk.call_args.args[0]}
        self.assertEqual(sent["1"]["intent"], "enquiry")
        self.assertEqual(sent["2"]["intent"], "complaint")
        self.assertNotIn("intent", sent["3"])


if __name__ == "__main__":
    unittest.main()
