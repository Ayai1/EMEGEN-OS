"""End-to-end Stage 4 query pipeline tests using local deterministic providers."""

from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engramOS.db import connection, initialize_database  # noqa: E402
from engramOS.embeddings import BGE_M3_DIMENSION, index_history  # noqa: E402
from engramOS.pipeline import answer_query  # noqa: E402
from engramOS.reader import ReaderPrompt  # noqa: E402
from engramOS.reranking import CandidateScore  # noqa: E402
from engramOS.retrieval import rehydrate_reader_candidates, retrieve_memory  # noqa: E402
from engramOS.temporal import FactInput, TemporalBoundary, insert_fact  # noqa: E402
from engramOS.query_classifier import TemporalRange  # noqa: E402
from tests.helpers import record_store_event  # noqa: E402


class FakeEmbeddingProvider:
    model_id = "fake-stage4-1024"
    dimension = BGE_M3_DIMENSION
    device = "cpu"

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        vector[int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little") % self.dimension] = 1
        return vector

    def encode_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def encode_documents(self, texts: list[str] | tuple[str, ...]) -> np.ndarray:
        return np.vstack([self._vector(text) for text in texts]) if texts else np.empty((0, self.dimension))


class FakeReader:
    def __init__(self) -> None:
        self.prompts: list[ReaderPrompt] = []

    def generate(self, prompt: ReaderPrompt, *, timeout: float) -> str:
        self.prompts.append(prompt)
        payload = json.loads(prompt.user_payload)
        citation = payload["evidence"][0]["evidence_id"]
        return json.dumps({
            "schema_version": "engram.reader-response.v1",
            "answer": "Supported answer.", "citations": [citation], "abstained": False,
        })


class Stage4PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = str(Path(self.temp_dir.name) / "stage4.sqlite3")
        initialize_database(self.database)
        self.embedding = FakeEmbeddingProvider()
        self.serial = 0

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def add_turn(self, text: str, event_time: float, turn_id: str) -> str:
        self.serial += 1
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO turns (turn_id, history_id, session_id, turn_idx,
                   role, raw_content, source_type, event_time, recorded_at, hash)
                   VALUES (?, 'h1', ?, ?, 'user', ?, 'user_turn', ?, ?, ?)""",
                (turn_id, f"s{self.serial}", self.serial, text, event_time,
                 event_time + 1000, f"hash-{turn_id}"),
            )
            record_store_event(conn, turn_id, created_at=event_time + 1000)
        return turn_id

    def add_fact(
        self, turn_id: str, fact_id: str, valid_from: float, recorded_at: float,
        value: str = "Tea", attribute: str = "drink",
    ) -> None:
        insert_fact(self.database, FactInput(
            fact_id, "h1", turn_id, 0, "user", attribute, value, "single",
            valid_from, recorded_at, key_text=f"user.drink: {value}",
        ))

    def test_unsupported_classification_stops_before_retrieval_and_reader(self) -> None:
        reader = FakeReader()
        outcome = answer_query(
            self.database, "h1", "What did I know on 2099-01-01?",
            self.embedding, reader, question_date=date(2026, 11, 2),
        )
        self.assertEqual(outcome.status, "unsupported")
        self.assertIsNone(outcome.retrieval)
        self.assertEqual(reader.prompts, [])

    def test_classifier_failure_stops_before_retrieval_reranking_and_reader(self) -> None:
        class FailedClassifier:
            def classify(inner_self, query, *, question_date, timeout):
                raise RuntimeError("private provider detail")

        class UnexpectedReranker:
            def rerank(inner_self, query, candidates, *, timeout):
                raise AssertionError("reranker must not run")

        reader = FakeReader()
        outcome = answer_query(
            self.database, "h1", "What happened?", self.embedding, reader,
            question_date=date(2026, 11, 2), classifier=FailedClassifier(),
            reranker=UnexpectedReranker(),
        )
        self.assertEqual(outcome.status, "abstained")
        self.assertEqual(outcome.reason_code, "classification_failed")
        self.assertIsNone(outcome.retrieval)
        self.assertIsNone(outcome.reranking)
        self.assertEqual(reader.prompts, [])

    def test_as_of_context_omits_later_source_text_and_marks_retrospective_fact(self) -> None:
        valid_time = 1_767_225_600.0  # 2026-01-01 UTC
        later_time = valid_time + 86_400
        source = self.add_turn("Later source says I prefer tea", later_time, "later-source")
        self.add_fact(source, "tea-fact", valid_time, later_time + 20)
        index_history(self.database, "h1", self.embedding)
        reader = FakeReader()
        outcome = answer_query(
            self.database, "h1", "What drink did I prefer on 2026-01-01?",
            self.embedding, reader, question_date=date(2026, 11, 2),
        )
        self.assertEqual(outcome.status, "answered")
        payload = json.loads(reader.prompts[0].user_payload)
        fact = payload["evidence"][0]
        self.assertEqual(fact["evidence_id"], "fact:tea-fact")
        self.assertTrue(fact["metadata"]["retrospective"])
        self.assertTrue(fact["provenance"])
        self.assertEqual(fact["provenance"][0]["text"], "")
        self.assertNotIn("Later source says", reader.prompts[0].user_payload)
        self.assertIn("reader_context", payload)

    def test_as_of_context_omits_later_confirmation_text(self) -> None:
        valid_time = 1_767_225_600.0  # 2026-01-01 UTC
        source = self.add_turn("Original source before cutoff", valid_time, "source")
        self.add_fact(source, "tea-fact", valid_time, valid_time + 1000)
        confirmation_time = valid_time + 86_400
        confirmation = self.add_turn(
            "Later confirmation text", confirmation_time, "confirmation"
        )
        insert_fact(self.database, FactInput(
            "tea-confirmation", "h1", confirmation, 0, "user", "drink", "Tea", "single",
            confirmation_time, confirmation_time + 1000, key_text="user.drink: Tea",
        ))
        index_history(self.database, "h1", self.embedding)
        reader = FakeReader()

        outcome = answer_query(
            self.database, "h1", "What drink did I prefer on 2026-01-01?",
            self.embedding, reader, question_date=date(2026, 11, 2),
        )
        self.assertEqual(outcome.status, "answered")
        payload = json.loads(reader.prompts[0].user_payload)
        fact = next(item for item in payload["evidence"] if item["evidence_id"] == "fact:tea-fact")
        confirmation_provenance = next(
            source for source in fact["provenance"]
            if source["relation"] == "confirmation"
        )
        self.assertEqual(confirmation_provenance["text"], "")
        self.assertNotIn("Later confirmation text", reader.prompts[0].user_payload)

    def test_as_of_trigger_does_not_expose_post_cutoff_cue(self) -> None:
        valid_time = 1_767_225_600.0  # 2026-01-01 UTC
        parent_turn = self.add_turn("My storage port is 5432", valid_time - 2000, "parent")
        self.add_fact(parent_turn, "port-fact", valid_time, valid_time - 1000,
                      value="5432", attribute="port")
        trigger_turn = self.add_turn(
            "When connecting to storage", valid_time + 86_400, "later-trigger"
        )
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO facts (
                    fact_id, history_id, turn_id, fact_seq, key_type, key_text,
                    parent_fact_id, valid_from, recorded_at
                ) VALUES ('storage-trigger', 'h1', ?, 0, 'trigger', ?,
                          'port-fact', ?, ?)""",
                (trigger_turn, "When connecting to storage", valid_time,
                 valid_time + 86_400 + 1000),
            )
        index_history(self.database, "h1", self.embedding)
        reader = FakeReader()

        outcome = answer_query(
            self.database, "h1", "What is my storage port as of 2026-01-01?",
            self.embedding, reader, question_date=date(2026, 11, 2),
        )

        self.assertEqual(outcome.status, "answered")
        payload = json.loads(reader.prompts[0].user_payload)
        trigger = next(
            item for item in payload["evidence"]
            if item["evidence_id"] == "fact:storage-trigger"
        )
        self.assertNotIn("retrieval_cue", trigger["metadata"])
        self.assertNotIn("When connecting to storage", reader.prompts[0].user_payload)

    def test_fact_recording_time_alone_marks_as_of_fact_retrospective(self) -> None:
        valid_time = 1_767_225_600.0  # 2026-01-01 UTC
        source = self.add_turn("My preferred drink is tea", valid_time - 2000, "source")
        self.add_fact(source, "tea-fact", valid_time, valid_time + 1000)
        index_history(self.database, "h1", self.embedding)
        reader = FakeReader()

        outcome = answer_query(
            self.database, "h1", "What drink did I prefer on 2026-01-01?",
            self.embedding, reader, question_date=date(2026, 11, 2),
        )

        self.assertEqual(outcome.status, "answered")
        payload = json.loads(reader.prompts[0].user_payload)
        fact = next(item for item in payload["evidence"] if item["evidence_id"] == "fact:tea-fact")
        self.assertTrue(fact["metadata"]["retrospective"])
        self.assertLessEqual(fact["provenance"][0]["event_time"], valid_time)
        self.assertLessEqual(fact["provenance"][0]["recorded_at"], valid_time)

    def test_as_of_turn_boundary_applies_before_dense_top_k(self) -> None:
        self.add_turn("shared topic before", 100, "before")
        self.add_turn("shared topic exactly", 200, "at")
        self.add_turn("shared topic after", 300, "after")
        index_history(self.database, "h1", self.embedding)
        result = retrieve_memory(
            self.database, "h1", "shared topic", self.embedding,
            view="as_of", as_of=TemporalBoundary(200, inclusive=True), top_k=1,
        )
        self.assertEqual([turn.turn_id for turn in result.turns], ["at"])

    def test_exclusive_as_of_turn_boundary_excludes_equal_event_time(self) -> None:
        self.add_turn("shared topic before", 100, "before")
        self.add_turn("shared topic exactly", 200, "at")
        self.add_turn("shared topic after", 300, "after")
        index_history(self.database, "h1", self.embedding)
        result = retrieve_memory(
            self.database, "h1", "shared topic", self.embedding,
            view="as_of", as_of=TemporalBoundary(200, inclusive=False), top_k=10,
        )
        self.assertEqual({turn.turn_id for turn in result.turns}, {"before"})

    def test_quarantine_between_retrieval_and_reader_removes_all_candidates(self) -> None:
        source = self.add_turn("I prefer green tea today", 100, "source")
        self.add_fact(source, "tea-fact", 100, 1100)
        index_history(self.database, "h1", self.embedding)
        reader = FakeReader()

        class QuarantiningReranker:
            def rerank(inner_self, query, candidates, *, timeout):
                with connection(self.database) as conn:
                    conn.execute(
                        """INSERT INTO gate_events (turn_id, signal_bridge, signal_policy,
                           signal_source, signals_fired, decision, created_at)
                           VALUES (?,1,1,0,2,'quarantine',999999)""",
                        (source,),
                    )
                return [CandidateScore(item.candidate_id, 1.0) for item in candidates]

        outcome = answer_query(
            self.database, "h1", "What is my current drink?", self.embedding, reader,
            question_date=date(2026, 11, 2), reranker=QuarantiningReranker(),
        )
        self.assertEqual(outcome.status, "abstained")
        self.assertEqual(outcome.reader.reason_code, "no_evidence")
        self.assertEqual(reader.prompts, [])

    def test_history_range_selects_valid_from_changes_and_labels_baseline(self) -> None:
        before = self.add_turn("I liked coffee", 100, "coffee")
        within = self.add_turn("I now like tea", 200, "tea")
        self.add_fact(before, "coffee-fact", 100, 1100, "Coffee")
        self.add_fact(within, "tea-fact", 200, 1200)
        index_history(self.database, "h1", self.embedding)
        result = retrieve_memory(
            self.database, "h1", "drink changes", self.embedding,
            view="history", history_range=TemporalRange(
                start=TemporalBoundary(150, inclusive=True),
                end=TemporalBoundary(250, inclusive=True),
            ),
        )
        self.assertEqual({fact.fact_id for fact in result.facts}, {"coffee-fact", "tea-fact"})
        baseline = next(fact for fact in result.facts if fact.fact_id == "coffee-fact")
        self.assertEqual(baseline.annotation, "[baseline before requested history range]")
        fresh = rehydrate_reader_candidates(
            self.database, "h1", result,
            tuple(("fact", item.fact_id) for item in result.facts),
            history_range=TemporalRange(
                start=TemporalBoundary(150, inclusive=True),
                end=TemporalBoundary(250, inclusive=True),
            ),
        )
        fresh_baseline = next(
            item for item in fresh.candidates if getattr(item, "fact_id", None) == "coffee-fact"
        )
        self.assertEqual(fresh_baseline.annotation, "[baseline before requested history range]")

    def test_history_range_respects_inclusive_and_exclusive_valid_from_endpoints(self) -> None:
        specs = (
            (100.0, "before", "baseline", "baseline-slot"),
            (150.0, "at-start", "Start", "start-slot"),
            (200.0, "inside", "Inside", "inside-slot"),
            (250.0, "at-end", "End", "end-slot"),
        )
        for timestamp, fact_id, value, attribute in specs:
            turn_id = self.add_turn(f"record {value}", timestamp, f"turn-{fact_id}")
            self.add_fact(
                turn_id, fact_id, timestamp, timestamp + 1000,
                value=value, attribute=attribute,
            )
        index_history(self.database, "h1", self.embedding)

        inclusive = retrieve_memory(
            self.database, "h1", "history", self.embedding, view="history",
            history_range=TemporalRange(
                start=TemporalBoundary(150, inclusive=True),
                end=TemporalBoundary(250, inclusive=True),
            ),
        )
        self.assertEqual(
            {fact.fact_id for fact in inclusive.facts},
            {"before", "at-start", "inside", "at-end"},
        )
        self.assertEqual(
            next(fact for fact in inclusive.facts if fact.fact_id == "before").annotation,
            "[baseline before requested history range]",
        )

        exclusive = retrieve_memory(
            self.database, "h1", "history", self.embedding, view="history",
            history_range=TemporalRange(
                start=TemporalBoundary(150, inclusive=False),
                end=TemporalBoundary(250, inclusive=False),
            ),
        )
        self.assertEqual(
            {fact.fact_id for fact in exclusive.facts}, {"before", "inside"}
        )
        self.assertEqual(
            next(fact for fact in exclusive.facts if fact.fact_id == "before").annotation,
            "[baseline before requested history range]",
        )

    def test_validated_combined_reranker_order_reaches_reader_unchanged(self) -> None:
        first = self.add_turn("I prefer coffee", 100, "first")
        second = self.add_turn("I now prefer tea", 200, "second")
        self.add_fact(first, "coffee-fact", 100, 1100, "Coffee")
        self.add_fact(second, "tea-fact", 200, 1200, "Tea")
        index_history(self.database, "h1", self.embedding)
        reader = FakeReader()

        class ReverseReranker:
            def rerank(inner_self, query, candidates, *, timeout):
                return [
                    CandidateScore(candidate.candidate_id, float(rank))
                    for rank, candidate in enumerate(reversed(candidates), start=1)
                ]

        outcome = answer_query(
            self.database, "h1", "What is my drink?", self.embedding, reader,
            question_date=date(2026, 11, 2), reranker=ReverseReranker(),
        )
        self.assertEqual(outcome.status, "answered")
        self.assertEqual(outcome.reranking.status, "reranked")
        payload = json.loads(reader.prompts[0].user_payload)
        expected = [
            str(candidate_id) for candidate_id in outcome.reranking.ordered_candidate_ids
        ]
        self.assertEqual(
            [item["evidence_id"] for item in payload["evidence"]], expected
        )


if __name__ == "__main__":
    unittest.main()
