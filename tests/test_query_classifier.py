"""Deterministic tests for query classification and untrusted output validation."""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engramOS.query_classifier import (  # noqa: E402
    QUERY_CLASSIFICATION_SCHEMA_VERSION,
    QueryClassificationError,
    QueryClassificationOutputError,
    TemporalRange,
    classify_query,
)
from engramOS.temporal import TemporalBoundary, as_of_timestamp  # noqa: E402


def output(
    *,
    status: str = "classified",
    view: str | None = "as_of",
    as_of: dict[str, object] | None = None,
    history_range: dict[str, object] | None = None,
    reason_code: str = "provider_classification",
) -> str:
    return json.dumps(
        {
            "schema_version": QUERY_CLASSIFICATION_SCHEMA_VERSION,
            "status": status,
            "view": view,
            "as_of": as_of,
            "history_range": history_range,
            "reason_code": reason_code,
        }
    )


class FakeClassifier:
    def __init__(self, response: object = "") -> None:
        self.response = response
        self.calls: list[tuple[str, date, float]] = []

    def classify(self, query: str, *, question_date: date, timeout: float) -> str:
        self.calls.append((query, question_date, timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response  # type: ignore[return-value]


class QueryClassificationTests(unittest.TestCase):
    question_date = date(2023, 3, 10)

    def test_current_state_fast_path_is_deterministic_and_skips_provider(self) -> None:
        provider = FakeClassifier("not json")
        result = classify_query(
            "What is my role?", question_date=self.question_date, classifier=provider
        )
        self.assertEqual((result.status, result.view, result.reason_code),
                         ("classified", "current", "present_state"))
        self.assertTrue(result.ready_for_retrieval)
        self.assertEqual(provider.calls, [])

    def test_current_question_with_history_as_noun_is_not_keyword_routed(self) -> None:
        result = classify_query(
            "What is my history teacher's name?", question_date=self.question_date
        )
        self.assertEqual((result.status, result.view), ("classified", "current"))

    def test_natural_explicit_date_routes_to_as_of_at_midnight_utc(self) -> None:
        result = classify_query(
            "What was my role on 1 March 2023?", question_date="2023-03-10"
        )
        self.assertEqual(result.view, "as_of")
        self.assertEqual(result.as_of, TemporalBoundary(as_of_timestamp("2023-03-01")))
        self.assertTrue(result.as_of.inclusive if result.as_of else False)

    def test_before_date_is_exclusive_as_of_and_not_history(self) -> None:
        result = classify_query(
            "What was my role before March 1, 2023?", question_date=self.question_date
        )
        self.assertEqual((result.status, result.view, result.reason_code),
                         ("classified", "as_of", "before_boundary"))
        self.assertEqual(
            result.as_of,
            TemporalBoundary(as_of_timestamp("2023-03-01"), inclusive=False),
        )

    def test_change_request_before_date_is_history_with_exclusive_end(self) -> None:
        result = classify_query(
            "Show me how my role changed before 2023-03-01",
            question_date=self.question_date,
        )
        self.assertEqual((result.status, result.view), ("classified", "history"))
        self.assertIsNone(result.as_of)
        self.assertEqual(
            result.history_range,
            TemporalRange(end=TemporalBoundary(as_of_timestamp("2023-03-01"), False)),
        )

    def test_history_change_query_without_dates_routes_to_history(self) -> None:
        result = classify_query(
            "How did my role change over time?", question_date=self.question_date
        )
        self.assertEqual((result.status, result.view, result.history_range),
                         ("classified", "history", None))

    def test_single_date_change_timeline_without_period_boundary_clarifies(self) -> None:
        result = classify_query(
            "How did my role change on 2023-03-01?",
            question_date=self.question_date,
        )
        self.assertEqual((result.status, result.view), ("clarification_required", None))

    def test_clear_history_month_is_calendar_range_not_arbitrary_point(self) -> None:
        result = classify_query(
            "How did my role change during March 2023?",
            question_date=self.question_date,
        )
        self.assertEqual(result.view, "history")
        self.assertEqual(
            result.history_range,
            TemporalRange(
                start=TemporalBoundary(as_of_timestamp("2023-03-01"), True),
                end=TemporalBoundary(as_of_timestamp("2023-04-01"), False),
            ),
        )

    def test_ambiguous_month_only_point_query_requires_clarification(self) -> None:
        result = classify_query(
            "What was my role in March 2023?", question_date=self.question_date
        )
        self.assertEqual((result.status, result.view, result.reason_code),
                         ("clarification_required", None, "ambiguous_month"))
        self.assertFalse(result.ready_for_retrieval)

    def test_before_month_uses_month_start_as_strict_boundary(self) -> None:
        result = classify_query(
            "What was my role before March 2023?", question_date=self.question_date
        )
        self.assertEqual((result.status, result.view, result.reason_code),
                         ("classified", "as_of", "before_boundary"))
        self.assertEqual(
            result.as_of,
            TemporalBoundary(as_of_timestamp("2023-03-01"), inclusive=False),
        )

    def test_history_before_month_ends_exclusively_at_month_start(self) -> None:
        result = classify_query(
            "How did my role change before March 2023?",
            question_date=self.question_date,
        )
        self.assertEqual(result.view, "history")
        self.assertEqual(
            result.history_range,
            TemporalRange(end=TemporalBoundary(as_of_timestamp("2023-03-01"), False)),
        )

    def test_history_from_until_uses_explicit_endpoint_semantics(self) -> None:
        result = classify_query(
            "Show me how my role changed from 2023-03-01 until 2023-03-10",
            question_date=self.question_date,
        )
        self.assertEqual(result.view, "history")
        self.assertEqual(
            result.history_range,
            TemporalRange(
                start=TemporalBoundary(as_of_timestamp("2023-03-01"), True),
                end=TemporalBoundary(as_of_timestamp("2023-03-10"), False),
            ),
        )

    def test_history_between_dates_clarifies_ambiguous_endpoints(self) -> None:
        result = classify_query(
            "Show how my role changed between 2023-03-01 and 2023-03-10",
            question_date=self.question_date,
        )
        self.assertEqual((result.status, result.view), ("clarification_required", None))

    def test_relative_dates_use_explicit_question_date(self) -> None:
        yesterday = classify_query(
            "What was my role yesterday?", question_date="2023-03-10"
        )
        days_ago = classify_query(
            "Where did I live 3 days ago?", question_date=date(2023, 3, 10)
        )
        self.assertEqual(yesterday.as_of, TemporalBoundary(as_of_timestamp("2023-03-09")))
        self.assertEqual(days_ago.as_of, TemporalBoundary(as_of_timestamp("2023-03-07")))

    def test_offset_timestamp_is_normalized_to_utc(self) -> None:
        result = classify_query(
            "What was my role at 2023-03-01T01:00:00+02:00?",
            question_date=self.question_date,
        )
        self.assertEqual(
            result.as_of,
            TemporalBoundary(as_of_timestamp("2023-02-28T23:00:00Z")),
        )

    def test_future_as_of_is_unsupported_instead_of_current(self) -> None:
        result = classify_query(
            "What will my role be on 2023-03-11?", question_date=self.question_date
        )
        self.assertEqual(
            (result.status, result.view, result.reason_code),
            ("unsupported", None, "future_as_of_unsupported"),
        )

    def test_future_relative_date_is_unsupported(self) -> None:
        result = classify_query(
            "What will my role be tomorrow?", question_date=self.question_date
        )
        self.assertEqual(
            (result.status, result.view, result.reason_code),
            ("unsupported", None, "future_as_of_unsupported"),
        )

    def test_current_query_with_temporal_date_requests_clarification(self) -> None:
        result = classify_query(
            "What is my current role on March 1, 2023?",
            question_date=self.question_date,
        )
        self.assertEqual((result.status, result.view), ("clarification_required", None))

    def test_knowledge_time_query_is_unsupported(self) -> None:
        result = classify_query(
            "What did I know at the time on 2020-03-01?",
            question_date=self.question_date,
        )
        self.assertEqual(
            (result.status, result.view, result.reason_code),
            ("unsupported", None, "knowledge_time_unsupported"),
        )

    def test_ambiguous_previous_word_does_not_force_history(self) -> None:
        result = classify_query(
            "What was my role previously?", question_date=self.question_date
        )
        self.assertNotEqual(result.view, "history")
        self.assertFalse(result.ready_for_retrieval)

    def test_change_verb_alone_does_not_force_history(self) -> None:
        result = classify_query(
            "How do I change my role?", question_date=self.question_date
        )
        self.assertEqual(result.status, "clarification_required")
        self.assertNotEqual(result.view, "history")

    def test_unknown_query_without_provider_requires_clarification(self) -> None:
        result = classify_query(
            "Summarize the project state", question_date=self.question_date
        )
        self.assertEqual((result.status, result.view), ("clarification_required", None))

    def test_provider_input_includes_explicit_question_date_and_timeout(self) -> None:
        provider = FakeClassifier(
            output(as_of={"value": "2023-03-01", "inclusive": True})
        )
        result = classify_query(
            "Summarize project state relative to 2023-03-01",
            question_date="2023-03-10",
            classifier=provider,
            timeout=2.5,
        )
        self.assertEqual(result.view, "as_of")
        self.assertEqual(provider.calls, [
            ("Summarize project state relative to 2023-03-01", date(2023, 3, 10), 2.5)
        ])

    def test_provider_receives_the_original_query_text(self) -> None:
        query = "  Summarize status relative to 2023-03-01  "
        provider = FakeClassifier(
            output(as_of={"value": "2023-03-01", "inclusive": True})
        )
        classify_query(query, question_date=self.question_date, classifier=provider)
        self.assertEqual(provider.calls[0][0], query)

    def test_provider_date_only_uses_midnight_utc(self) -> None:
        provider = FakeClassifier(
            output(as_of={"value": "2023-03-01", "inclusive": True})
        )
        result = classify_query(
            "Summarize status relative to 2023-03-01",
            question_date=self.question_date,
            classifier=provider,
        )
        self.assertEqual(result.as_of, TemporalBoundary(as_of_timestamp("2023-03-01")))

    def test_provider_cannot_route_explicit_date_to_current(self) -> None:
        provider = FakeClassifier(
            output(view="current", as_of=None, reason_code="provider_classification")
        )
        with self.assertRaises(QueryClassificationOutputError):
            classify_query(
                "Summarize status relative to 2023-03-01",
                question_date=self.question_date,
                classifier=provider,
            )

    def test_provider_must_preserve_exclusive_before_semantics(self) -> None:
        provider = FakeClassifier(
            output(as_of={"value": "2023-03-01", "inclusive": True})
        )
        with self.assertRaisesRegex(QueryClassificationOutputError, "exclusive"):
            classify_query(
                "Summarize status before 2023-03-01",
                question_date=self.question_date,
                classifier=provider,
            )

    def test_provider_must_not_invent_an_as_of_date(self) -> None:
        provider = FakeClassifier(
            output(as_of={"value": "2023-03-02", "inclusive": True})
        )
        with self.assertRaisesRegex(QueryClassificationOutputError, "does not match"):
            classify_query(
                "Summarize status relative to 2023-03-01",
                question_date=self.question_date,
                classifier=provider,
            )

    def test_provider_cannot_invent_an_as_of_date_absent_from_query(self) -> None:
        provider = FakeClassifier(
            output(as_of={"value": "2020-03-01", "inclusive": True})
        )
        with self.assertRaisesRegex(QueryClassificationOutputError, "not expressed"):
            classify_query(
                "What was my role?", question_date=self.question_date,
                classifier=provider,
            )

    def test_provider_cannot_drop_requested_multi_date_history_range(self) -> None:
        provider = FakeClassifier(
            output(view="history", as_of=None, reason_code="provider_classification")
        )
        with self.assertRaisesRegex(QueryClassificationOutputError, "omitted"):
            classify_query(
                "Summarize status between 2023-03-01 and 2023-03-10",
                question_date=self.question_date,
                classifier=provider,
            )

    def test_malformed_provider_outputs_fail_closed(self) -> None:
        malformed = (
            "not json",
            '{"schema_version":"engram.query-classification.v1",'
            '"schema_version":"engram.query-classification.v1"}',
            output().replace('"history_range": null,', '"extra": null,'),
            output().replace('"view": "as_of"', '"view": "invented"'),
            output(as_of={"value": "2023-02-30", "inclusive": True}),
            output(as_of={"value": "2023-03-01T10:00:00", "inclusive": True}),
            output(as_of={"value": "2023-03-01", "inclusive": 1}),
            output(view=None, as_of=None, status="clarification_required"),
            output().replace('"status": "classified"', '"status": []'),
            output(reason_code="present_state"),
        )
        for raw in malformed:
            with self.subTest(raw=raw), self.assertRaises(QueryClassificationOutputError):
                classify_query(
                    "Summarize status relative to 2023-03-01",
                    question_date=self.question_date,
                    classifier=FakeClassifier(raw),
                )

    def test_contradictory_history_range_fails_closed(self) -> None:
        provider = FakeClassifier(
            output(
                view="history",
                as_of=None,
                history_range={
                    "start": {"value": "2023-03-02", "inclusive": True},
                    "end": {"value": "2023-03-01", "inclusive": False},
                },
                reason_code="provider_classification",
            )
        )
        with self.assertRaisesRegex(QueryClassificationOutputError, "contradictory"):
            classify_query(
                "Summarize status between 2023-03-01 and 2023-03-02",
                question_date=self.question_date,
                classifier=provider,
            )

    def test_provider_failure_raises_without_default_route(self) -> None:
        provider = FakeClassifier(TimeoutError("provider timeout"))
        with self.assertRaises(QueryClassificationError):
            classify_query(
                "Describe my previous position", question_date=self.question_date,
                classifier=provider,
            )

    def test_deterministic_route_does_not_call_failed_provider(self) -> None:
        provider = FakeClassifier(TimeoutError("must not be called"))
        result = classify_query(
            "What is my role?", question_date=self.question_date, classifier=provider
        )
        self.assertEqual(result.view, "current")
        self.assertEqual(provider.calls, [])

    def test_invalid_question_date_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            classify_query("What is my role?", question_date="2023-02-30")
        with self.assertRaises(ValueError):
            classify_query("What is my role?", question_date=datetime(2023, 3, 10))

    def test_invalid_query_date_requests_clarification(self) -> None:
        result = classify_query(
            "What was my role on 2023-02-30?", question_date=self.question_date
        )
        self.assertEqual((result.status, result.view), ("clarification_required", None))


if __name__ == "__main__":
    unittest.main()
