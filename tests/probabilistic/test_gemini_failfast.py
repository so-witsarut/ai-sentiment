import os
import unittest
from unittest.mock import Mock, patch
import ai_sentiment as module


class GeminiFailFastTests(unittest.TestCase):
    def setUp(self):
        self.limiter = module.GeminiRateLimiter()
        self.limiter.max_wait = 10

    def test_reserve_raises_failfast_when_cooldown_exceeds_max_wait(self):
        # Set cooldown of 3600 seconds
        self.limiter.cooldown("test-model", 3600.0)
        with self.assertRaises(module.GeminiRateLimited) as ctx:
            self.limiter.reserve("test-model", 100)
        self.assertIn("in cooldown", str(ctx.exception))
        self.assertIn("exceeds 10s limit", str(ctx.exception))

    def test_reserve_succeeds_when_cooldown_within_max_wait(self):
        # Cooldown within limit (e.g. 2s) should return (None, delay) rather than raise
        self.limiter.cooldown("test-model", 2.0)
        ticket, delay = self.limiter.reserve("test-model", 100)
        self.assertIsNone(ticket)
        self.assertGreater(delay, 0)
        self.assertLessEqual(delay, 2.5)

    def test_is_blocked_helper(self):
        self.assertEqual(self.limiter.is_blocked("unblocked-model"), 0.0)
        self.limiter.cooldown("blocked-model", 3600.0)
        self.assertGreater(self.limiter.is_blocked("blocked-model"), 10.0)

    def test_call_gemini_api_skips_immediately_when_blocked(self):
        analyzer = module.OllamaSentimentAnalyzer()
        with patch.object(module.GEMINI_RATE_LIMITER, "is_blocked", return_value=5600.0), \
             patch.object(analyzer, "_post_provider") as mock_post:
            result = analyzer._call_gemini_api("test-model", "sys", "user")
            self.assertIsNone(result)
            mock_post.assert_not_called()

    def test_call_gemini_api_skips_retries_on_large_429_cooldown(self):
        analyzer = module.OllamaSentimentAnalyzer()
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "5600"}
        mock_response.json.return_value = {}

        with patch.object(analyzer, "_post_provider", return_value=mock_response) as mock_post, \
             patch.object(module, "GEMINI_API_KEY", "dummy-key"):
            result = analyzer._call_gemini_api("test-model", "sys", "user", max_retries=3)
            self.assertIsNone(result)
            # Must NOT retry 3 times since retry_after > max_wait
            self.assertEqual(mock_post.call_count, 1)


if __name__ == "__main__":
    unittest.main()

