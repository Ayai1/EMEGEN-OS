"""Tests for evaluation provider-failure flags and grading separation."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.run_eval import (  # noqa: E402
    _grade,
    _provider_failure_metadata,
    _provider_failure_stop_reason,
    _provider_failure_types,
    _wrong_answer_count,
)


class EvaluationProviderFailureTests(unittest.TestCase):
    def test_classifier_length_is_provider_failure_independent_of_grade(self) -> None:
        failure_types = _provider_failure_types(
            classifier_called=True,
            classifier_error=RuntimeError("truncated"),
            classifier_finish_reasons=["length"],
            reader_finish_reasons=[],
            classification=None,
            reader_reason_code="classification_failed",
            baseline_exception=None,
        )
        metadata = _provider_failure_metadata(failure_types)
        self.assertEqual(failure_types, ["classifier_finish_reason_length"])
        self.assertEqual(metadata, {
            "provider_failure": True,
            "provider_failure_types": ["classifier_finish_reason_length"],
        })
        self.assertFalse(_grade("I don't know.", True, "answer_contains", "supported"))

    def test_classifier_parse_failure_is_reported_after_bounded_attempts(self) -> None:
        failure_types = _provider_failure_types(
            classifier_called=True,
            classifier_error=None,
            classifier_finish_reasons=["stop", "stop"],
            reader_finish_reasons=[],
            classification=None,
            reader_reason_code="classification_failed",
            baseline_exception=ValueError("provider output rejected"),
        )
        self.assertEqual(failure_types, ["classifier_parse_failure"])

    def test_reader_parse_failure_and_timeout_are_distinct_provider_failures(self) -> None:
        reader_parse = _provider_failure_types(
            classifier_called=False,
            classifier_error=None,
            classifier_finish_reasons=[],
            reader_finish_reasons=["stop", "stop"],
            classification=None,
            reader_reason_code="invalid_reader_output",
            baseline_exception=None,
        )
        self.assertEqual(reader_parse, ["reader_parse_failure"])
        timeout = _provider_failure_types(
            classifier_called=True,
            classifier_error=TimeoutError("local request timed out"),
            classifier_finish_reasons=[],
            reader_finish_reasons=[],
            classification=None,
            reader_reason_code="classification_failed",
            baseline_exception=None,
        )
        self.assertEqual(timeout, ["classifier_timeout"])

    def test_wrong_answer_alone_does_not_set_provider_failure_flag(self) -> None:
        self.assertFalse(_grade("wrong answer", False, "answer_contains", "right answer"))
        self.assertEqual(
            _provider_failure_metadata([]),
            {"provider_failure": False, "provider_failure_types": []},
        )

    def test_provider_failure_is_not_double_counted_as_wrong_answer(self) -> None:
        rows = [
            {"correct": False, "provider_failure": True},
            {"correct": False, "provider_failure": False},
            {"correct": True, "provider_failure": False},
        ]
        self.assertEqual(_wrong_answer_count(rows), 1)

    def test_one_timeout_or_length_failure_continues_with_healthy_server(self) -> None:
        for failure in ("reader_timeout", "reader_finish_reason_length"):
            with self.subTest(failure=failure):
                self.assertIsNone(_provider_failure_stop_reason(
                    provider_failure_types=[failure], consecutive_failures=1,
                    stop_on_provider_failure=False, local_mode=True, server_healthy=True,
                ))

    def test_server_death_or_three_consecutive_failures_stops(self) -> None:
        self.assertEqual(_provider_failure_stop_reason(
            provider_failure_types=["reader_timeout"], consecutive_failures=1,
            stop_on_provider_failure=False, local_mode=True, server_healthy=False,
        ), "server_unavailable")
        self.assertEqual(_provider_failure_stop_reason(
            provider_failure_types=["reader_parse_failure"], consecutive_failures=3,
            stop_on_provider_failure=False, local_mode=True, server_healthy=True,
        ), "three_consecutive_provider_failures")


if __name__ == "__main__":
    unittest.main()
