"""Offline checks for entity-targeted, bounded hybrid prompts."""

import os
import socket
import unittest
from unittest.mock import MagicMock, patch

os.environ["PYTHON_DOTENV_DISABLED"] = "1"

with patch.object(socket.socket, "connect", side_effect=AssertionError("offline test attempted network")):
    import ai_sentiment as module


def jev_result(sentiment, entity):
    return {
        "sentiment_probabilities": sentiment,
        "sentiment_confidence": 0.90,
        "sentiment_choice": max(sentiment, key=sentiment.get),
        "entity_probabilities": entity,
        "entity_confidence": 0.90,
        "entity_choice": max(entity, key=entity.get),
    }


class PromptTargetingTests(unittest.TestCase):
    def setUp(self):
        v2_patch = patch.object(module, "HYBRID_PROMPT_V2", True)
        v2_patch.start()
        self.addCleanup(v2_patch.stop)
        self.analyzer = module.OllamaSentimentAnalyzer()
        self.resolver = MagicMock()
        self.resolver.resolve_target.return_value = {
            "actual_target": "BLCP (หัวข้อ/คีย์เวิร์ด: ค่าไฟ)",
            "project_name": "BLCP",
            "project_desc": "โรงไฟฟ้า BLCP",
            "is_rival": False,
            "competitor_matched": "",
        }

    def test_target_resolution_keeps_keyword_out_of_target(self):
        self.assertEqual(module._sentiment_target(None, "BLCP", "", ["ค่าไฟ"],
                                                   self.resolver.resolve_target.return_value), "BLCP")
        self.assertEqual(module._sentiment_target("BLCP (หัวข้อ/คีย์เวิร์ด: ค่าไฟ)", "", "",
                                                   ["ค่าไฟ"], None), "BLCP")
        self.assertEqual(module._sentiment_target("ค่าไฟ", "", "", ["ค่าไฟ"], None), "")
        self.assertEqual(module._sentiment_target("ค่าไฟ", "Rivals", "", ["ค่าไฟ"],
                                                   {"is_rival": True, "competitor_matched": "Chang"}), "Chang")
        self.assertEqual(module._sentiment_target("ค่าไฟ", "Rivals", "", ["ค่าไฟ"],
                                                   {"is_rival": True, "competitor_matched": ""}), "")
        self.assertEqual(module._sentiment_target("ค่าไฟ", "BLCP", "", ["ค่าไฟ"],
                                                   {"is_rival": True, "project_name": "BLCP",
                                                    "competitor_matched": ""}), "BLCP")
        self.assertEqual(module._sentiment_target("สิงห์", "Boonrawd", "Demo", ["สิงห์"],
                                                   None, "competitor"), "")

    def test_keyword_jev_prompt_checks_opinion_target_in_one_request(self):
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "answers": {
                "sentiment": {"choice": "neutral", "confidence": 0.9,
                              "probabilities": {"positive": 0.02, "neutral": 0.94,
                                                "negative": 0.02, "irony": 0.02}},
                "entity_relevance": {"choice": "relevant", "confidence": 0.9,
                                     "probabilities": {"relevant": 0.94, "unrelated": 0.03,
                                                       "uncertain": 0.03}},
            }
        }
        session = MagicMock()
        session.post.return_value = response
        post = {"match_post_id": "venue-mention", "project_name": "Central Mall",
                "keywords": ["Central Mall"],
                "content": "The cinema has too few film showings at Central Mall",
                "_analysis_scope": "keyword"}
        with patch.object(module, "OPENROUTER_API_KEY", "offline-key"), \
             patch.object(self.analyzer, "get_session", return_value=session), \
             patch.object(self.analyzer, "_call_deepseek_fallback") as deepseek:
            result = self.analyzer._analyze_single_post(post)

        self.assertEqual(session.post.call_count, 1)
        deepseek.assert_not_called()
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(set(payload["questions"]), {"sentiment", "entity_relevance", "intent"})
        self.assertIn("Target=Central Mall", payload["state"])
        instructions = payload["questions"]["sentiment"]["instructions"].lower()
        self.assertIn("target project only", instructions)
        self.assertIn("independent opinion", instructions)
        self.assertIn("named venue", payload["questions"]["sentiment"]["criteria"]["neutral"])
        self.assertEqual((result["sentiment"], result["route"]), ("neutral", "jev"))

    def test_keyword_only_post_is_neutral_for_blcp(self):
        unrelated = jev_result(
            {"positive": 0.02, "neutral": 0.92, "negative": 0.03, "irony": 0.03},
            {"relevant": 0.03, "unrelated": 0.92, "uncertain": 0.05},
        )
        for index, content in enumerate(("ค่าไฟแพงมาก แต่ไม่มีข้อมูลเกี่ยวกับโรงไฟฟ้าใด",
                                         "ค่าไฟแพงเพราะบริษัท XYZ ให้บริการแย่"), 1):
            post = {"match_post_id": f"blcp-{index}", "project_id": "known", "keywords": ["ค่าไฟ"],
                    "content": content, "feed_link": "https://example.org/post"}
            with patch.object(module, "GLOBAL_PROJECT_RESOLVER", self.resolver), \
                 patch.object(self.analyzer, "_call_typesafe_jev", return_value=(unrelated, "jev")) as call_jev, \
                 patch.object(self.analyzer, "_call_deepseek_fallback") as call_deepseek:
                result = self.analyzer._analyze_single_post(post)
            self.assertEqual(call_jev.call_args.args[1], "BLCP")
            self.assertIn("Target=BLCP", call_jev.call_args.args[0])
            self.assertIn("Keywords=ค่าไฟ", call_jev.call_args.args[0])
            self.assertNotIn("https://example.org", call_jev.call_args.args[0])
            call_deepseek.assert_not_called()
            self.assertEqual((result["sentiment"], result["entity_found"], result["ai_sentiment"]),
                             ("neutral", False, 0))

    def test_direct_project_evidence_can_be_positive(self):
        post = {"match_post_id": "blcp-2", "project_id": "known", "keywords": ["ค่าไฟ"],
                "content": "BLCP ช่วยจัดการค่าไฟได้ดีมาก"}
        positive = jev_result(
            {"positive": 0.90, "neutral": 0.05, "negative": 0.03, "irony": 0.02},
            {"relevant": 0.90, "unrelated": 0.05, "uncertain": 0.05},
        )
        with patch.object(module, "GLOBAL_PROJECT_RESOLVER", self.resolver), \
             patch.object(self.analyzer, "_call_typesafe_jev", return_value=(positive, "jev")):
            result = self.analyzer._analyze_single_post(post)
        self.assertEqual((result["sentiment"], result["entity_found"], result["route"]),
                         ("positive", True, "jev"))
        self.assertIn("BLCP", result["reason"])

    def test_missing_entity_uses_overall_post_sentiment(self):
        self.resolver.resolve_target.return_value = {
            "actual_target": "ค่าไฟ", "project_name": "", "project_desc": "",
            "is_rival": False, "competitor_matched": "",
        }
        overall = jev_result(
            {"positive": 0.02, "neutral": 0.04, "negative": 0.92, "irony": 0.02},
            {"relevant": 1.0, "unrelated": 0.0, "uncertain": 0.0})
        with patch.object(module, "GLOBAL_PROJECT_RESOLVER", self.resolver), \
             patch.object(self.analyzer, "_call_typesafe_jev", return_value=(overall, "jev")) as call_jev, \
             patch.object(self.analyzer, "_call_deepseek_fallback") as call_deepseek:
            result = self.analyzer._analyze_single_post(
                {"match_post_id": "unknown-1", "project_id": "missing", "keywords": ["ค่าไฟ"],
                 "content": "ค่าไฟแพง"}
            )
        self.assertTrue(call_jev.call_args.kwargs["overall_scope"])
        call_deepseek.assert_not_called()
        self.assertEqual((result["model"], result["sentiment"], result["entity_found"]),
                         ("jev", "negative", True))

    def test_competitor_keyword_does_not_become_client_target(self):
        post = {"post_id": "rival-1", "content": "สิงห์ออกสินค้าใหม่", "tracking_kind": "competitor",
                "actual_target": "สิงห์", "company_name": "Demo", "project_name": "Boonrawd",
                "keywords": ["สิงห์"]}
        overall = jev_result(
            {"positive": 0.02, "neutral": 0.94, "negative": 0.02, "irony": 0.02},
            {"relevant": 1.0, "unrelated": 0.0, "uncertain": 0.0})
        with patch.object(module, "GLOBAL_PROJECT_RESOLVER", None), \
             patch.object(self.analyzer, "_call_typesafe_jev", return_value=(overall, "jev")) as call_jev, \
             patch.object(self.analyzer, "_call_deepseek_fallback") as call_deepseek:
            result = self.analyzer._analyze_single_post(post)
        self.assertTrue(call_jev.call_args.kwargs["overall_scope"])
        call_deepseek.assert_not_called()
        self.assertEqual((result["model"], result["sentiment"]), ("jev", "neutral"))

    def test_jev_payload_and_deepseek_prompt_use_same_target_contract(self):
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "answers": {
                "sentiment": {"confidence": 0.9, "choice": "neutral", "probabilities":
                              {"positive": 0.02, "neutral": 0.92, "negative": 0.03, "irony": 0.03}},
                "entity_relevance": {"confidence": 0.9, "choice": "unrelated", "probabilities":
                                     {"relevant": 0.03, "unrelated": 0.92, "uncertain": 0.05}},
            }
        }
        session = MagicMock()
        session.post.return_value = response
        with patch.object(module, "OPENROUTER_API_KEY", "offline-key"), \
             patch.object(self.analyzer, "get_session", return_value=session):
            self.analyzer._call_typesafe_jev("Target=BLCP\nKeywords=ค่าไฟ\nText=ค่าไฟแพง", "BLCP")
        jev_payload = session.post.call_args.kwargs["json"]
        self.assertEqual(jev_payload["state"].count("Target=BLCP"), 1)
        self.assertIn("keyword match alone", jev_payload["questions"]["entity_relevance"]["instructions"].lower())
        self.assertNotIn("ค่าไฟ", jev_payload["questions"]["sentiment"]["instructions"])

        context = {"sentiment_target": "BLCP", "actual_target": "BLCP (หัวข้อ/คีย์เวิร์ด: ค่าไฟ)",
                   "project_name": "BLCP", "project_desc": "โรงไฟฟ้า BLCP", "keywords": ["ค่าไฟ"],
                   "source_info": "Source Link=https://example.org/post", "capped_text": "ค่าไฟแพง"}
        prompt = module.build_deepseek_user_prompt(context, None)
        self.assertIn("Target=BLCP\n", prompt)
        self.assertNotIn("Target=BLCP (", prompt)
        self.assertNotIn("Project=BLCP", prompt)
        self.assertNotIn("https://example.org", prompt)
        self.assertIn("keyword match alone", module.HYBRID_SYSTEM_PROMPT.lower())
        self.assertIn("attributable link", module.HYBRID_SYSTEM_PROMPT.lower())

    def test_indirect_project_link_uses_deepseek_with_one_call(self):
        context = {"post_id": "blcp-indirect", "actual_target": "BLCP (หัวข้อ/คีย์เวิร์ด: ค่าไฟ)",
                   "sentiment_target": "BLCP", "project_name": "BLCP", "project_desc": "โรงไฟฟ้าบีแอลซีพี",
                   "keywords": ["ค่าไฟ"], "source_info": "Source Link=https://example.org/post",
                   "capped_text": "โรงไฟฟ้าบีแอลซีพีถูกวิจารณ์เรื่องค่าไฟ", "clean_text": "โรงไฟฟ้าบีแอลซีพีถูกวิจารณ์เรื่องค่าไฟ"}
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "choices": [{"message": {"content": '{"probabilities":{"POSITIVE":0.02,"NEUTRAL":0.03,'
                                                '"NEGATIVE":0.90,"AMBIGUOUS_OR_IRONY":0.05},'
                                                '"entity_found":true}'}}]
        }
        session = MagicMock()
        session.post.return_value = response
        with patch.object(module, "OPENROUTER_API_KEY", "offline-key"), \
             patch.object(self.analyzer, "get_session", return_value=session), \
             patch.dict(os.environ, {"DEEPSEEK_TEXT_MAX_CHARS": "3000"}):
            result, model = self.analyzer._call_deepseek_fallback(context, None)
        self.assertEqual(session.post.call_count, 1)
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(payload["max_tokens"], module.DEEPSEEK_MAX_TOKENS)
        self.assertEqual(payload["messages"][0]["content"], module.HYBRID_SYSTEM_PROMPT)
        self.assertIn("Target=BLCP\n", payload["messages"][1]["content"])
        self.assertNotIn("https://example.org", payload["messages"][1]["content"])
        self.assertEqual(result["entity_found"], True)
        self.assertEqual(result["probabilities"]["NEGATIVE"], 0.90)
        self.assertEqual(model, module.DEEPSEEK_MODEL)

    def test_long_text_hard_cap_and_deepseek_variants(self):
        text = "A" * 2500 + " BLCP ดำเนินงาน " + "B" * 2500 + "ค่าไฟ" + "C" * 2500
        clipped = module.cap_text(text, max_chars=3000, keyword="ค่าไฟ", target="BLCP")
        self.assertLessEqual(len(clipped), 3000)
        self.assertIn("BLCP ดำเนินงาน", clipped)
        self.assertIn("ค่าไฟ", clipped)
        self.assertTrue(clipped.startswith("A" * 100))
        self.assertTrue(clipped.endswith("C" * 100))

        context = {"sentiment_target": "BLCP", "actual_target": "BLCP", "project_name": "BLCP",
                   "keywords": ["ค่าไฟ"], "clean_text": text, "capped_text": clipped}
        signal = {"probabilities": {"POSITIVE": 0.1, "NEUTRAL": 0.2, "NEGATIVE": 0.6,
                                    "AMBIGUOUS_OR_IRONY": 0.1}, "confidence": 0.6,
                  "conflict_reasons": ["narrow_sentiment_margin"]}
        with patch.dict(os.environ, {"DEEPSEEK_TEXT_MAX_CHARS": "3000"}):
            short_prompt = module.build_deepseek_user_prompt(context, signal)
        self.assertIn("Jev=POS:0.100,NEU:0.200,NEG:0.600,IRONY:0.100", short_prompt)
        with patch.dict(os.environ, {"DEEPSEEK_TEXT_MAX_CHARS": "8000",
                                      "DEEPSEEK_INCLUDE_JEV_SIGNAL": "false"}):
            longer_prompt = module.build_deepseek_user_prompt(context, signal)
        self.assertNotIn("Jev=", longer_prompt)
        self.assertGreater(len(longer_prompt), len(short_prompt))

    def test_default_prompt_stays_on_existing_route_until_canary(self):
        context = {"actual_target": "BLCP (หัวข้อ/คีย์เวิร์ด: ค่าไฟ)", "project_name": "BLCP",
                   "keywords": ["ค่าไฟ"], "source_info": "Source Link=https://example.org/post",
                   "capped_text": "ค่าไฟแพง"}
        with patch.object(module, "HYBRID_PROMPT_V2", False):
            prompt = module.build_deepseek_user_prompt(context)
        self.assertIn("Target=BLCP (หัวข้อ/คีย์เวิร์ด: ค่าไฟ)", prompt)
        self.assertIn("https://example.org/post", prompt)


if __name__ == "__main__":
    unittest.main()
