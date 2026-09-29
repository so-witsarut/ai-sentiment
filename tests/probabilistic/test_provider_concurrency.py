"""Offline checks for provider-specific HTTP concurrency limits."""

import os
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

os.environ["PYTHON_DOTENV_DISABLED"] = "1"

import ai_sentiment as sentiment_module

VALID_ANSWER = ('{"probabilities":{"POSITIVE":0.5,"NEUTRAL":0.5,'
                '"NEGATIVE":0,"AMBIGUOUS_OR_IRONY":0},"entity_found":true}')


class CountingSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = {"gemini": 0, "ollama": 0, "openrouter": 0}
        self.peak = self.active.copy()
        self.calls = []
        self.entered = {name: threading.Event() for name in self.active}
        self.block = {}
        self.reject_gemini_json_mode = False

    def post(self, url, **kwargs):
        if "generativelanguage.googleapis.com" in url:
            provider = "gemini"
        elif "openrouter.ai" in url:
            provider = "openrouter"
        else:
            provider = "ollama"
        with self.lock:
            self.active[provider] += 1
            self.peak[provider] = max(self.peak[provider], self.active[provider])
            self.calls.append((provider, url))
        self.entered[provider].set()
        try:
            gate = self.block.get(provider)
            if gate is not None and not gate.wait(timeout=3):
                raise TimeoutError(f"{provider} call was not released")
            time.sleep(0.01)
            if provider == "gemini":
                if self.reject_gemini_json_mode and kwargs["json"]["generationConfig"].get("responseMimeType"):
                    body, status = {}, 400
                else:
                    body = {"candidates": [{"content": {"parts": [{"text": VALID_ANSWER}]}}]}
                    status = 200
            elif provider == "openrouter":
                body = {"choices": [{"message": {"content": VALID_ANSWER}}]}
                status = 200
            elif url.endswith("/api/chat"):
                body = {"message": {"content": VALID_ANSWER}}
                status = 200
            elif kwargs["json"].get("think") is False:
                body = {"response": '{"triage": "yes"}'}
                status = 200
            else:
                body = {}
                status = 500  # Exercise the Ollama generate -> chat fallback.
            return SimpleNamespace(status_code=status, json=lambda: body, text="mock error", headers={})
        finally:
            with self.lock:
                self.active[provider] -= 1


class ProviderConcurrencyTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {
            "ENABLE_GEMINI": "true", "ENABLE_OLLAMA": "true", "ENABLE_OPENROUTER": "true",
            "GEMINI_MAX_CONCURRENCY": "1", "OLLAMA_MAX_CONCURRENCY": "1",
        })
        env.start()
        self.addCleanup(env.stop)
        self.analyzer = sentiment_module.OllamaSentimentAnalyzer()
        self.session = CountingSession()
        session_patch = patch.object(self.analyzer, "get_session", return_value=self.session)
        session_patch.start()
        self.addCleanup(session_patch.stop)
        for name in ("GEMINI_API_KEY", "OPENROUTER_API_KEY"):
            key_patch = patch.object(sentiment_module, name, "offline-key")
            key_patch.start()
            self.addCleanup(key_patch.stop)

    def test_gemini_requests_from_multiple_threads_use_one_slot(self):
        barrier = threading.Barrier(8)

        def call():
            barrier.wait(timeout=3)
            return self.analyzer._call_gemini_api("gemma-4-26b-a4b-it", "JSON only", "example",
                                                  max_retries=1)

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: call(), range(8)))
        self.assertTrue(all(result["entity_found"] for result in results))
        self.assertEqual(self.session.peak["gemini"], 1)

    def test_ollama_triage_generate_and_chat_share_one_slot(self):
        barrier = threading.Barrier(8)

        def call(index):
            barrier.wait(timeout=3)
            if index % 2:
                return self.analyzer._triage_post(str(index), "example", "SEC")
            return self.analyzer._call_ollama_generic("gemma4:31b-cloud", "JSON only", "example")

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(call, range(8)))
        self.assertTrue(all(result is True or result["entity_found"] for result in results))
        self.assertEqual(self.session.peak["ollama"], 1)
        self.assertEqual(sum(url.endswith("/api/chat") for _, url in self.session.calls), 4)

    def test_gemini_400_fallback_uses_the_same_slot(self):
        self.session.reject_gemini_json_mode = True
        barrier = threading.Barrier(4)

        def call():
            barrier.wait(timeout=3)
            return self.analyzer._call_gemini_api("gemini-2.5-flash", "JSON only", "example",
                                                  max_retries=1)

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: call(), range(4)))
        self.assertTrue(all(result["entity_found"] for result in results))
        self.assertEqual(self.session.peak["gemini"], 1)
        self.assertEqual(len(self.session.calls), 8)

    def test_providers_have_independent_slots_and_openrouter_is_unchanged(self):
        release_gemini = threading.Event()
        self.session.block["gemini"] = release_gemini
        with ThreadPoolExecutor(max_workers=3) as executor:
            gemini = executor.submit(self.analyzer._call_gemini_api,
                                     "gemma-4-26b-a4b-it", "JSON only", "example", 1)
            self.assertTrue(self.session.entered["gemini"].wait(timeout=2))
            try:
                ollama = executor.submit(self.analyzer._triage_post, "1", "example", "SEC")
                openrouter = executor.submit(self.analyzer._call_openrouter_api,
                                             "mock/model", "JSON only", "example", 1)
                self.assertTrue(self.session.entered["ollama"].wait(timeout=2))
                self.assertTrue(self.session.entered["openrouter"].wait(timeout=2))
                self.assertTrue(ollama.result(timeout=2))
                self.assertTrue(openrouter.result(timeout=2)["entity_found"])
            finally:
                release_gemini.set()
            self.assertTrue(gemini.result(timeout=2)["entity_found"])

    def test_disabled_providers_send_no_requests(self):
        with patch.dict(os.environ, {"ENABLE_GEMINI": "false", "ENABLE_OLLAMA": "false",
                                     "ENABLE_OPENROUTER": "false"}):
            self.assertIsNone(self.analyzer._call_gemini_api("gemma-4-26b-a4b-it", "", ""))
            self.assertIsNone(self.analyzer._call_ollama_generic("gemma4:31b-cloud", "", ""))
            self.assertTrue(self.analyzer._triage_post("1", "example"))
            self.assertIsNone(self.analyzer._call_openrouter_api("mock/model", "", ""))
        self.assertEqual(self.session.calls, [])


if __name__ == "__main__":
    unittest.main()
