"""Stage 1 deterministic classifier audit for the authored benchmark."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from engramOS.query_classifier import (  # noqa: E402
    QUERY_CLASSIFICATION_SCHEMA_VERSION,
    classify_query,
)
from eval.loader import DEFAULT_SUITE_PATH, load_suite  # noqa: E402


class CountingClassifier:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def classify(self, query: str, *, question_date: date, timeout: float) -> str:
        self.calls.append(query)
        return json.dumps({
            "schema_version": QUERY_CLASSIFICATION_SCHEMA_VERSION,
            "status": "classified",
            "view": "current",
            "reason_code": "provider_classification",
            "as_of": None,
            "history_range": None,
        })


class Stage1EvaluationAuditTests(unittest.TestCase):
    def test_representative_present_state_query_makes_zero_provider_calls(self) -> None:
        provider = CountingClassifier()
        result = classify_query(
            "What is my current city?",
            question_date=date(2026, 11, 3),
            classifier=provider,
        )
        self.assertEqual((result.status, result.view, result.reason_code),
                         ("classified", "current", "present_state"))
        self.assertEqual(provider.calls, [])

    def test_only_two_authored_queries_reach_the_classifier_provider_fallback(self) -> None:
        suite = load_suite(DEFAULT_SUITE_PATH)
        provider = CountingClassifier()
        question_date = date.fromisoformat(suite["question_date"])
        for question in suite["questions"]:
            classify_query(
                question["query"],
                question_date=question_date,
                classifier=provider,
            )
        self.assertEqual(
            provider.calls,
            [
                "Name the astronomy project I keep in my notes.",
                "What market feature was reported?",
            ],
        )


if __name__ == "__main__":
    unittest.main()
