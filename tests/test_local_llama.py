"""Offline tests for the local llama.cpp transport and production parsers."""

from __future__ import annotations

from dataclasses import replace
from contextlib import closing
from datetime import date
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engramOS.local_llama import (  # noqa: E402
    LocalModelSettings,
    LocalOpenAIClient,
    LocalQueryClassifierProvider,
    LocalReaderProvider,
    canned_classifier_response,
)
from engramOS.db import initialize_database  # noqa: E402
from engramOS.llm_cache import BudgetPolicy  # noqa: E402
from engramOS.pipeline import answer_turn_baseline_query  # noqa: E402
from engramOS.query_classifier import classify_query  # noqa: E402
from engramOS.reader import READER_RESPONSE_SCHEMA_VERSION  # noqa: E402
from engramOS.retrieval import RetrievedTurn  # noqa: E402


class FakeJsonRequest:
    def __init__(self, *, content: str, finish_reason: str = "stop") -> None:
        self.content = content
        self.finish_reason = finish_reason
        self.chat_payloads: list[dict[str, object]] = []
        self.urls: list[str] = []

    def __call__(self, url: str, payload: object, timeout: float) -> object:
        self.urls.append(url)
        if url.endswith("/tokenize"):
            content = str(payload["content"])  # type: ignore[index]
            return {"tokens": content.split()}
        self.chat_payloads.append(dict(payload))  # type: ignore[arg-type]
        return {
            "model": "Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
            "choices": [{
                "message": {"role": "assistant", "content": self.content},
                "finish_reason": self.finish_reason,
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8},
        }


def settings(**changes: object) -> LocalModelSettings:
    return replace(LocalModelSettings(
        base_url="http://127.0.0.1:8080",
        model="Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
        model_sha256="82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38",
        gpu_layers=32,
    ), **changes)


def turn() -> RetrievedTurn:
    return RetrievedTurn(
        "local-pune", "local", "local-session", 0, "user", "I live in Pune.",
        "user_turn", 1790000000.0, 1790000000.0, None, None, 1.0, ("test",),
    )


def reader_response(*, answer: str = "Pune.", citations: list[str] | None = None) -> str:
    return json.dumps({
        "schema_version": READER_RESPONSE_SCHEMA_VERSION,
        "answer": answer,
        "citations": ["turn:local-pune"] if citations is None else citations,
        "abstained": False,
    })


class LocalLlamaTests(unittest.TestCase):
    def test_reader_and_classifier_use_production_parsers_without_network(self) -> None:
        query = "Which city is listed in my personal notes?"
        fake_classifier = FakeJsonRequest(content=canned_classifier_response())
        classifier = LocalQueryClassifierProvider(
            LocalOpenAIClient(settings(), json_request=fake_classifier)
        )
        classification = classify_query(query, question_date=date(2026, 11, 3), classifier=classifier)
        self.assertTrue(classification.ready_for_retrieval)
        self.assertEqual(len(fake_classifier.chat_payloads), 1)
        self.assertIn("response_format", fake_classifier.chat_payloads[0])

        fake_reader = FakeJsonRequest(content=reader_response())
        reader = LocalReaderProvider(LocalOpenAIClient(settings(), json_request=fake_reader))
        outcome = answer_turn_baseline_query(
            query, "local", [turn()], classification, reader,
            ordering="relevance", timeout=5.0,
        )
        self.assertEqual(outcome.reason_code, "answered")
        self.assertEqual(outcome.answer, "Pune.")
        self.assertEqual([str(item) for item in outcome.citations], ["turn:local-pune"])
        self.assertEqual(reader.last_completion.visible_tokens, len(reader_response().split()))
        self.assertEqual(len(fake_reader.chat_payloads), 1)
        self.assertNotIn("Authorization", str(fake_reader.chat_payloads[0]))

    def test_reader_invalid_json_is_rejected_by_production_parser(self) -> None:
        reader = LocalReaderProvider(
            LocalOpenAIClient(settings(), json_request=FakeJsonRequest(content="not json"))
        )
        classification = classify_query("What is current?", question_date=date(2026, 11, 3))
        outcome = answer_turn_baseline_query(
            "What city is listed?", "local", [turn()], classification, reader,
            ordering="relevance", timeout=5.0,
        )
        self.assertEqual(outcome.reason_code, "invalid_reader_output")

    def test_reader_finish_reason_length_is_a_counted_failure(self) -> None:
        reader = LocalReaderProvider(
            LocalOpenAIClient(
                settings(), json_request=FakeJsonRequest(content=reader_response(), finish_reason="length")
            )
        )
        classification = classify_query("What is current?", question_date=date(2026, 11, 3))
        outcome = answer_turn_baseline_query(
            "What city is listed?", "local", [turn()], classification, reader,
            ordering="relevance", timeout=5.0,
        )
        self.assertEqual(outcome.reason_code, "answered")
        self.assertEqual(reader.truncated_calls, 1)
        self.assertEqual(reader.last_completion.finish_reason, "length")

    def test_local_server_failure_propagates_without_fallback(self) -> None:
        def down(url: str, payload: object, timeout: float) -> object:
            raise ConnectionError("server unavailable")

        client = LocalOpenAIClient(settings(), json_request=down)
        with self.assertRaises(ConnectionError):
            client.complete({"messages": []}, timeout=1.0)

    def test_local_timeout_propagates_without_fallback(self) -> None:
        def timed_out(url: str, payload: object, timeout: float) -> object:
            raise TimeoutError("local server request timed out")

        client = LocalOpenAIClient(settings(), json_request=timed_out)
        with self.assertRaises(TimeoutError):
            client.complete({"messages": []}, timeout=1.0)

    def test_cached_local_reader_logs_zero_cost_and_reuses_response(self) -> None:
        with TemporaryDirectory() as folder:
            database = Path(folder) / "local-cache.sqlite3"
            initialize_database(database)
            fake = FakeJsonRequest(content=reader_response())
            client = LocalOpenAIClient(
                settings(),
                json_request=fake,
                cache_database=str(database),
                phase_tag="local_test",
                run_id="local-cache-test",
            )
            reader = LocalReaderProvider(client)
            query = "Which city is listed in my personal notes?"
            classification = classify_query("What is current?", question_date=date(2026, 11, 3))
            first = answer_turn_baseline_query(
                query, "local", [turn()], classification, reader,
                ordering="relevance", timeout=5.0,
            )
            self.assertEqual(first.reason_code, "answered")
            self.assertFalse(reader.last_completion.cache_hit)
            second = answer_turn_baseline_query(
                query, "local", [turn()], classification, reader,
                ordering="relevance", timeout=5.0,
            )
            self.assertEqual(second.reason_code, "answered")
            self.assertTrue(reader.last_completion.cache_hit)
            self.assertEqual(len(fake.chat_payloads), 1)
            with closing(sqlite3.connect(database)) as conn:
                rows, credits, ledger_rows = conn.execute(
                    """SELECT (SELECT COUNT(*) FROM llm_cache WHERE run_id=?),
                              (SELECT COALESCE(SUM(cost_credits), 0) FROM llm_cache WHERE run_id=?),
                              (SELECT COUNT(*) FROM llm_call_ledger WHERE run_id=?)""",
                    ("local-cache-test", "local-cache-test", "local-cache-test"),
                ).fetchone()
            self.assertEqual((rows, credits, ledger_rows), (1, 0.0, 1))

    def test_cache_identity_changes_with_complete_request_and_server_settings(self) -> None:
        payload = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 10}
        base = settings()
        original = base.request_hash(payload)
        self.assertNotEqual(original, base.request_hash({**payload, "max_tokens": 11}))
        self.assertNotEqual(original, replace(base, context_size=4096).request_hash(payload))
        self.assertNotEqual(original, replace(base, reasoning_mode="off").request_hash(payload))
        self.assertNotEqual(original, replace(base, gpu_layers=31).request_hash(payload))

    def test_300_second_timeout_is_accepted_and_not_part_of_cache_identity(self) -> None:
        with TemporaryDirectory() as folder:
            database = Path(folder) / "timeout-identity.sqlite3"
            initialize_database(database)
            payload = {
                "model": settings().model,
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "user"},
                ],
                "temperature": 0.0,
                "max_tokens": 2048,
            }
            budget = BudgetPolicy(reservation_lease_seconds=600.0)
            first_fake = FakeJsonRequest(content=reader_response())
            first = LocalOpenAIClient(
                settings(), json_request=first_fake, cache_database=str(database),
                budget_policy=budget,
            ).complete(payload, timeout=180.0, task_tag="reader")
            second_fake = FakeJsonRequest(content=reader_response(answer="different"))
            second = LocalOpenAIClient(
                settings(), json_request=second_fake, cache_database=str(database),
                budget_policy=budget,
            ).complete(payload, timeout=300.0, task_tag="reader")
            self.assertEqual(first.request_hash, second.request_hash)
            self.assertTrue(second.cache_hit)
            self.assertEqual(second_fake.chat_payloads, [])
            self.assertNotIn("timeout", settings().server_identity())

    def test_classifier_max_tokens_change_256_to_1024_misses_cache(self) -> None:
        with TemporaryDirectory() as folder:
            database = Path(folder) / "classifier-budget-cache.sqlite3"
            initialize_database(database)
            old_fake = FakeJsonRequest(content="", finish_reason="length")
            old_classifier = LocalQueryClassifierProvider(LocalOpenAIClient(
                settings(classifier_max_tokens=256),
                json_request=old_fake,
                cache_database=str(database),
                phase_tag="local_test",
                run_id="classifier-budget-cache-test",
            ))
            query = "Tell me the database associated with my notes."
            old_classifier.classify(query, question_date=date(2026, 11, 3), timeout=5.0)
            self.assertEqual(old_classifier.last_completion.finish_reason, "length")
            self.assertEqual(old_fake.chat_payloads[0]["max_tokens"], 256)

            new_fake = FakeJsonRequest(content=canned_classifier_response())
            new_classifier = LocalQueryClassifierProvider(LocalOpenAIClient(
                settings(classifier_max_tokens=1024),
                json_request=new_fake,
                cache_database=str(database),
                phase_tag="local_test",
                run_id="classifier-budget-cache-test",
            ))
            response = new_classifier.classify(
                query, question_date=date(2026, 11, 3), timeout=5.0
            )
            self.assertEqual(json.loads(response)["schema_version"], "engram.query-classification.v1")
            self.assertFalse(new_classifier.last_completion.cache_hit)
            self.assertEqual(new_fake.chat_payloads[0]["max_tokens"], 1024)
            self.assertNotEqual(
                old_classifier.last_completion.request_hash,
                new_classifier.last_completion.request_hash,
            )
            with closing(sqlite3.connect(database)) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0], 2)


if __name__ == "__main__":
    unittest.main()
