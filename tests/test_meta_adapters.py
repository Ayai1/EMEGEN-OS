"""Offline integration tests for cache-backed Meta provider adapters."""

from __future__ import annotations

from datetime import date
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import httpx

from engramOS.db import initialize_database
from engramOS.extraction import ExtractionPrompt, ExtractionTurn
from engramOS.meta_adapters import (
    MetaFactExtractor,
    MetaModelConfig,
    MetaQueryClassifier,
    MetaReaderProvider,
    smoke_reader_prompt,
)
from engramOS.meta_provider import MetaModelAPIError, MetaModelChatProvider
from engramOS.query_classifier import QUERY_CLASSIFICATION_SCHEMA_VERSION
from engramOS.reader import READER_RESPONSE_SCHEMA_VERSION


class MetaAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.database = str(Path(self.folder.name) / "adapter.sqlite3")
        initialize_database(self.database)
        self.key_patch = patch.dict(os.environ, {"MODEL_API_KEY": "offline-test-key"})
        self.key_patch.start()
        self.addCleanup(self.key_patch.stop)
        self.addCleanup(self.folder.cleanup)
        self.config = MetaModelConfig(model="muse-spark-1.3", credits_per_usd=1.0)

    @staticmethod
    def response(text: str, model: str = "muse-spark-1.3") -> httpx.Response:
        return httpx.Response(200, json={
            "model": model,
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        })

    def make_api(self, handler) -> tuple[MetaModelChatProvider, httpx.Client]:
        client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        return self.config.create_api(client=client), client

    def test_classifier_uses_cache_and_logs_complete_call_accounting(self) -> None:
        calls: list[httpx.Request] = []
        text = json.dumps({
            "schema_version": QUERY_CLASSIFICATION_SCHEMA_VERSION,
            "status": "classified",
            "view": "current",
            "as_of": None,
            "history_range": None,
            "reason_code": "provider_classification",
        })

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return self.response(text)

        api, _ = self.make_api(handler)
        adapter = MetaQueryClassifier(
            self.database, api=api, config=self.config,
            phase_tag="dev_baselines", run_id="classifier-test",
        )

        first = adapter.classify("What city is current?", question_date=date(2026, 11, 3), timeout=2)
        second = adapter.classify("What city is current?", question_date=date(2026, 11, 3), timeout=2)

        self.assertEqual(first, text)
        self.assertEqual(second, text)
        self.assertEqual(len(calls), 1)
        self.assertTrue(adapter.last_call.cache_hit)  # type: ignore[union-attr]
        body = json.loads(calls[0].content)
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertFalse(body["store"])
        with closing(sqlite3.connect(self.database)) as conn:
            row = conn.execute(
                "SELECT task_tag, phase_tag, run_id, in_tokens, out_tokens, cost_credits, prompt_hash "
                "FROM llm_cache"
            ).fetchone()
            ledger = conn.execute(
                "SELECT status, actual_credits FROM llm_call_ledger"
            ).fetchone()
        self.assertEqual(row[:5], ("classifier", "dev_baselines", "classifier-test", 10, 2))
        self.assertGreater(row[5], 0)
        self.assertEqual(len(row[6]), 64)
        self.assertEqual(ledger[0], "succeeded")
        self.assertEqual(ledger[1], row[5])

    def test_reader_and_fact_extractor_keep_their_prompt_contracts(self) -> None:
        seen: list[dict[str, object]] = []
        reader_text = json.dumps({
            "schema_version": READER_RESPONSE_SCHEMA_VERSION,
            "answer": "Pune",
            "citations": ["turn:smoke-1"],
            "abstained": False,
        })

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            seen.append(body)
            if body["model"] == "muse-spark-1.3":
                if "response_format" not in body:
                    return self.response("[]")
                return self.response(reader_text)
            self.fail("unexpected model")

        api, _ = self.make_api(handler)
        reader = MetaReaderProvider(
            self.database, api=api, config=self.config,
            phase_tag="dev_baselines", run_id="reader-test",
        )
        result = reader.generate(smoke_reader_prompt(), timeout=2)
        self.assertEqual(json.loads(result)["answer"], "Pune")
        self.assertEqual(seen[0]["response_format"]["type"], "json_schema")
        self.assertEqual(
            seen[0]["response_format"]["json_schema"]["schema"]["properties"]["schema_version"]["const"],
            READER_RESPONSE_SCHEMA_VERSION,
        )

        extractor = MetaFactExtractor(
            self.database, api=api, config=self.config,
            phase_tag="dev_extraction", run_id="extractor-test",
        )
        turn = ExtractionTurn(
            "t1", "h1", "s1", "2026-11-03", 1_793_664_000.0,
            0, "user", "I prefer tea.", "user_message", 1_793_664_000.0,
        )
        extraction = extractor.extract(turn, ExtractionPrompt("extract facts", "I prefer tea."))
        self.assertEqual(extraction, "[]")
        self.assertNotIn("response_format", seen[1])
        with closing(sqlite3.connect(self.database)) as conn:
            rows = conn.execute(
                "SELECT task_tag, phase_tag, run_id FROM llm_cache ORDER BY task_tag"
            ).fetchall()
        self.assertEqual(rows, [
            ("extractor", "dev_extraction", "extractor-test"),
            ("reader", "dev_baselines", "reader-test"),
        ])

    def test_retry_is_bounded_and_the_failed_attempt_is_charged(self) -> None:
        attempts = 0
        waits: list[float] = []
        text = json.dumps({
            "schema_version": QUERY_CLASSIFICATION_SCHEMA_VERSION,
            "status": "classified",
            "view": "current",
            "as_of": None,
            "history_range": None,
            "reason_code": "provider_classification",
        })

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(503, text="internal diagnostic")
            return self.response(text)

        api, _ = self.make_api(handler)
        adapter = MetaQueryClassifier(
            self.database, api=api, config=self.config,
            phase_tag="dev_baselines", run_id="retry-test",
            max_attempts=2, sleep=waits.append, random_value=lambda: 0.0,
        )
        self.assertEqual(
            adapter.classify("What city is current?", question_date=date(2026, 11, 3), timeout=2),
            text,
        )
        self.assertEqual(attempts, 2)
        self.assertEqual(waits, [0.25])
        with closing(sqlite3.connect(self.database)) as conn:
            rows = sorted(conn.execute(
                "SELECT status, reserved_credits, actual_credits FROM llm_call_ledger ORDER BY started_at, call_id"
            ).fetchall())
        self.assertEqual([row[0] for row in rows], ["failed", "succeeded"])
        self.assertEqual(rows[0][1], rows[0][2])

    def test_non_transient_http_error_is_not_retried(self) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(400, text="private provider detail")

        api, _ = self.make_api(handler)
        adapter = MetaQueryClassifier(
            self.database, api=api, config=self.config,
            run_id="no-retry-test", max_attempts=3,
        )
        with self.assertRaises(MetaModelAPIError) as caught:
            adapter.classify("What city is current?", question_date=date(2026, 11, 3), timeout=2)
        self.assertFalse(caught.exception.retryable)
        self.assertEqual(attempts, 1)
        self.assertNotIn("private provider detail", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
