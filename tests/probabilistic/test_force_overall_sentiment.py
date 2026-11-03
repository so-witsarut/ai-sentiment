import os
import unittest
from unittest.mock import patch, Mock
import ai_sentiment as module


class ForceOverallSentimentTests(unittest.TestCase):
    def test_force_overall_bypasses_project_name(self):
        analyzer = module.OllamaSentimentAnalyzer()
        post = {
            "match_post_id": "test-1",
            "project_name": "BLCP",
            "keywords": ["ค่าไฟ", "พลังงาน"],
            "full_text": "ด่วน ปรับขึ้นค่าไฟฟ้ารอบใหม่ ตัวเลขพุ่งสูงกว่าเดิม",
            "_analysis_scope": "keyword"
        }

        with patch.dict(os.environ, {"FORCE_OVERALL_SENTIMENT": "true"}), \
             patch.object(analyzer, "_hybrid_analyze_post") as mock_hybrid:
            analyzer._analyze_single_post(post)
            self.assertTrue(mock_hybrid.called)
            resolved = mock_hybrid.call_args[0][0]
            self.assertEqual(resolved["analysis_scope"], "overall")
            self.assertEqual(resolved["actual_target"], module.OVERALL_POST_LABEL)
            self.assertEqual(resolved["sentiment_target"], module.OVERALL_POST_LABEL)
            # The project_name is still preserved for database reference
            self.assertEqual(resolved["project_name"], "BLCP")

    def test_legacy_mode_respects_project_name_when_disabled(self):
        analyzer = module.OllamaSentimentAnalyzer()
        post = {
            "match_post_id": "test-2",
            "project_name": "BLCP",
            "keywords": ["ค่าไฟ"],
            "full_text": "BLCP ดีมาก",
            "_analysis_scope": "keyword"
        }

        with patch.dict(os.environ, {"FORCE_OVERALL_SENTIMENT": "false"}), \
             patch.object(analyzer, "_hybrid_analyze_post") as mock_hybrid:
            analyzer._analyze_single_post(post)
            self.assertTrue(mock_hybrid.called)
            resolved = mock_hybrid.call_args[0][0]
            self.assertEqual(resolved["analysis_scope"], "keyword")
            self.assertEqual(resolved["actual_target"], "BLCP")


if __name__ == "__main__":
    unittest.main()
