"""SQLite tests for complete-request caching and credit reservations."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engramOS.db import connection, initialize_database
from engramOS.llm_cache import (
    BudgetExceededError,
    BudgetPolicy,
    GenerationRequest,
    GenerationResult,
    cached_generate,
)


class FakeProvider:
    def __init__(self, result: GenerationResult | None = None) -> None:
        self.calls = 0
        self.result = result or GenerationResult("ok", 12, 3, 0.2, 4)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def generate(self, request: GenerationRequest, *, timeout: float) -> GenerationResult:
        self.calls += 1
        self.entered.set()
        if not self.release.is_set():
            self.release.wait(2)
        return self.result


class LLMCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = str(Path(self.temp_dir.name) / "llm.sqlite3")
        initialize_database(self.database)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def request(self, **kwargs: object) -> GenerationRequest:
        values: dict[str, object] = {
            "provider_id": "fake",
            "model": "fake-model",
            "system_prompt": "system",
            "user_prompt": "question",
            "max_output_tokens": 100,
            "task_tag": "reader",
            "phase_tag": "dev",
            "run_id": "run-1",
        }
        values.update(kwargs)
        return GenerationRequest(**values)  # type: ignore[arg-type]

    def test_complete_generation_settings_define_cache_identity(self) -> None:
        provider = FakeProvider()
        first = cached_generate(
            self.database, self.request(), provider, reserved_credits=1
        )
        hit = cached_generate(
            self.database, self.request(), provider, reserved_credits=1
        )
        changed = cached_generate(
            self.database,
            self.request(max_output_tokens=101),
            provider,
            reserved_credits=1,
        )

        self.assertFalse(first.cache_hit)
        self.assertTrue(hit.cache_hit)
        self.assertFalse(changed.cache_hit)
        self.assertEqual(provider.calls, 2)

    def test_reservation_fails_before_provider_when_cap_would_be_exceeded(self) -> None:
        provider = FakeProvider()
        with self.assertRaises(BudgetExceededError):
            cached_generate(
                self.database, self.request(), provider, reserved_credits=1,
                policy=BudgetPolicy(total_cap=0.5),
            )
        self.assertEqual(provider.calls, 0)

    def test_phase_reserve_protects_locked_test_and_buffer(self) -> None:
        provider = FakeProvider()
        with self.assertRaises(BudgetExceededError):
            cached_generate(
                self.database,
                self.request(phase_tag="dev_baselines"),
                provider,
                reserved_credits=20,
            )
        self.assertEqual(provider.calls, 0)

    def test_failed_call_is_charged_at_full_reservation(self) -> None:
        class FailingProvider:
            def generate(self, request: GenerationRequest, *, timeout: float) -> GenerationResult:
                raise TimeoutError("no provider usage available")

        with self.assertRaises(TimeoutError):
            cached_generate(
                self.database, self.request(), FailingProvider(), reserved_credits=0.7
            )
        with connection(self.database) as conn:
            row = conn.execute(
                "SELECT status, actual_credits FROM llm_call_ledger"
            ).fetchone()
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["actual_credits"], 0.7)
        with connection(self.database) as conn:
            summary = conn.execute("SELECT * FROM budget_summary").fetchone()
        self.assertEqual(summary["phase_tag"], "dev")
        self.assertEqual(summary["credits_spent"], 0.7)

    def test_success_summary_has_phase_and_run_attribution(self) -> None:
        cached_generate(
            self.database, self.request(), FakeProvider(), reserved_credits=1
        )
        with connection(self.database) as conn:
            summary = conn.execute("SELECT * FROM budget_summary").fetchone()
        self.assertEqual(summary["phase_tag"], "dev")
        self.assertEqual(summary["run_id"], "run-1")
        self.assertEqual(summary["credits_spent"], 0.2)
        self.assertEqual(summary["credits_reserved"], 0.0)

    def test_bypass_cache_uses_unique_nonce_and_records_both_calls(self) -> None:
        provider = FakeProvider()
        first = cached_generate(
            self.database, self.request(), provider, reserved_credits=1
        )
        second = cached_generate(
            self.database, self.request(), provider, reserved_credits=1,
            bypass_cache=True,
        )
        self.assertFalse(first.cache_hit)
        self.assertFalse(second.cache_hit)
        self.assertNotEqual(first.request_key, second.request_key)
        self.assertEqual(provider.calls, 2)

    def test_concurrent_identical_miss_is_coalesced(self) -> None:
        provider = FakeProvider()
        provider.release.clear()
        first_outcome: list[object] = []

        def first_call() -> None:
            first_outcome.append(cached_generate(
                self.database, self.request(), provider, reserved_credits=1,
                wait_timeout=3,
            ))

        worker = threading.Thread(target=first_call)
        worker.start()
        self.assertTrue(provider.entered.wait(1))
        second_result: list[object] = []
        second = threading.Thread(target=lambda: second_result.append(cached_generate(
            self.database, self.request(), provider, reserved_credits=1,
            wait_timeout=3,
        )))
        second.start()
        time.sleep(0.1)
        provider.release.set()
        worker.join(3)
        second.join(3)

        self.assertFalse(worker.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(provider.calls, 1)
        self.assertEqual(len(first_outcome), 1)
        self.assertEqual(len(second_result), 1)
        self.assertTrue(second_result[0].cache_hit)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
