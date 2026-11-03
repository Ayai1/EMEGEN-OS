"""Evaluation-only baseline Reader orchestration regression coverage."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import engramOS.pipeline as pipeline  # noqa: E402
from engramOS.pipeline import answer_query, answer_turn_baseline_query  # noqa: E402
from engramOS.query_classifier import QueryClassification  # noqa: E402
from engramOS.reader import ReaderPrompt  # noqa: E402
from engramOS.retrieval import RetrievalDiagnostics, RetrievalResult, RetrievedTurn  # noqa: E402


class CaptureReader:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def generate(self, prompt: ReaderPrompt, *, timeout: float) -> str:
        payload = json.loads(prompt.user_payload)
        self.payloads.append(payload)
        evidence_id = payload["evidence"][0]["evidence_id"]
        return json.dumps({
            "schema_version": "engram.reader-response.v1",
            "answer": "Evidence answer",
            "citations": [evidence_id],
            "abstained": False,
        })


def sample_turn() -> RetrievedTurn:
    return RetrievedTurn(
        "h:s:0", "h", "s", 0, "user", "The city is Oslo", "user_turn",
        100.0, 200.0, 0.5, -1.0, 0.03, ("turn_dense",),
    )


def classification() -> QueryClassification:
    return QueryClassification("classified", "current", date(2026, 11, 3), "present_state")


class BaselinePipelineTests(unittest.TestCase):
    def test_baseline_context_records_each_ordering_label(self) -> None:
        reader = CaptureReader()
        for label in ("relevance", "reverse_chronological"):
            with self.subTest(ordering=label):
                result = answer_turn_baseline_query(
                    "What is my city now?", "h", [sample_turn()], classification(), reader,
                    ordering=label,
                )
                self.assertFalse(result.abstained)
                self.assertEqual(reader.payloads[-1]["reader_context"]["ordering"], label)

    def test_answer_query_never_calls_evaluation_baseline_helper(self) -> None:
        turn = sample_turn()
        retrieval = RetrievalResult(
            "h", "current", "memory", (turn,), (),
            RetrievalDiagnostics({}, "fake-1024", 1024, "cpu", 60, 10, 10, 1, 0),
        )
        reader = CaptureReader()
        materialized = SimpleNamespace(candidates=(turn,), authorization_timestamp=200.0)
        with (
            patch.object(pipeline, "classify_query", return_value=classification()),
            patch.object(pipeline, "retrieve_memory", return_value=retrieval),
            patch.object(pipeline, "rehydrate_reader_candidates", return_value=materialized),
            patch.object(pipeline, "answer_turn_baseline_query") as baseline_helper,
        ):
            result = answer_query(
                "What is my city now?", "h", "What is my city now?",
                SimpleNamespace(), reader, question_date=date(2026, 11, 3),
            )
        self.assertEqual(result.status, "answered")
        baseline_helper.assert_not_called()


if __name__ == "__main__":
    unittest.main()
