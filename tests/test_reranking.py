"""Security and compatibility tests for candidate-only reranking."""

from __future__ import annotations

import hashlib
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engramOS.db import connection, initialize_database  # noqa: E402
from engramOS.embeddings import BGE_M3_DIMENSION, index_history  # noqa: E402
from engramOS.retrieval import (  # noqa: E402
    EvidenceTurn,
    RetrievalDiagnostics,
    RetrievalResult,
    RetrievedFact,
    RetrievedTurn,
    retrieve_memory,
)
from engramOS.reranking import (  # noqa: E402
    CandidateId,
    CandidateScore,
    RerankCandidate,
    RerankingOutcome,
    rerank_retrieval_result,
)
import engramOS.reranking as reranking_module  # noqa: E402
from tests.helpers import record_store_event  # noqa: E402


class FakeReranker:
    def __init__(self, output: object = None, error: Exception | None = None) -> None:
        self.output = output
        self.error = error
        self.calls = 0
        self.last_query: str | None = None
        self.last_candidates: tuple[RerankCandidate, ...] | None = None
        self.last_timeout: float | None = None

    def rerank(
        self,
        query: str,
        candidates: tuple[RerankCandidate, ...],
        *,
        timeout: float,
    ) -> Sequence[CandidateScore]:
        self.calls += 1
        self.last_query = query
        self.last_candidates = candidates
        self.last_timeout = timeout
        if self.error is not None:
            raise self.error
        if callable(self.output):
            return self.output(query, candidates, timeout)
        return self.output


def make_turn(turn_id: str, raw_content: str, *, event_time: float = 10.0) -> RetrievedTurn:
    return RetrievedTurn(
        turn_id, "h1", f"session-{turn_id}", 0, "user", raw_content,
        "user_turn", event_time, event_time + 1, None, None, 0.5, ("turn_lexical",),
    )


def make_fact(fact_id: str, key_text: str, *, valid_from: float = 1.0) -> RetrievedFact:
    source = EvidenceTurn(
        f"source-{fact_id}", "h1", f"session-source-{fact_id}", 0,
        "user", f"source text for {fact_id}", "user_turn", valid_from,
        valid_from + 1,
    )
    fact = {
        "fact_id": fact_id,
        "history_id": "h1",
        "key_text": key_text,
        "entity": "user",
        "attribute": "preference",
        "cardinality": "single",
        "kind": "fact",
        "key_type": "canonical",
        "valid_from": valid_from,
        "valid_until": None,
        # This deliberately is not passed to the reranker.
        "trust": 0.9,
    }
    return RetrievedFact(
        fact, None, (source,), source, None, None, None, None, 0.5,
        ("fact_lexical",),
    )


def make_result(
    *,
    turns: tuple[RetrievedTurn, ...] = (),
    facts: tuple[RetrievedFact, ...] = (),
    view: str = "current",
) -> RetrievalResult:
    diagnostics = RetrievalDiagnostics(
        {"turn_lexical": len(turns), "turn_dense": 0, "fact_lexical": len(facts), "fact_dense": 0},
        "fake-model", 3, "cpu", 60, 50, 50, len(turns), len(facts),
    )
    return RetrievalResult("h1", view, "memory", turns, facts, diagnostics)  # type: ignore[arg-type]


def scored(
    candidates: tuple[RerankCandidate, ...],
    scores: dict[CandidateId, object] | None = None,
) -> tuple[CandidateScore, ...]:
    return tuple(
        CandidateScore(candidate.candidate_id,
                       scores[candidate.candidate_id] if scores is not None else float(candidate.original_rank))
        for candidate in candidates
    )


class RerankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.turn_a = make_turn("turn-a", "original turn text A")
        self.turn_b = make_turn("turn-b", "original turn text B", event_time=20.0)
        self.fact_a = make_fact("fact-a", "user.preference: tea")
        self.fact_b = make_fact("fact-b", "user.preference: coffee", valid_from=2.0)
        self.result = make_result(
            turns=(self.turn_a, self.turn_b), facts=(self.fact_a, self.fact_b)
        )

    def test_accepts_valid_exact_permutation_and_reorders_original_candidates(self) -> None:
        def output(_query: str, candidates: tuple[RerankCandidate, ...], _timeout: float):
            scores = {
                CandidateId("turn", "turn-a"): 1,
                CandidateId("turn", "turn-b"): 3,
                CandidateId("fact", "fact-a"): 2,
                CandidateId("fact", "fact-b"): 4,
            }
            return tuple(
                CandidateScore(item.candidate_id, scores[item.candidate_id])
                for item in reversed(candidates)
            )

        fake = FakeReranker(output)
        outcome = rerank_retrieval_result("preferred drink", self.result, fake, timeout=2.5)

        self.assertEqual(outcome.status, "reranked")
        self.assertEqual(outcome.reason, "validated_permutation")
        self.assertEqual([turn.turn_id for turn in outcome.result.turns], ["turn-b", "turn-a"])
        self.assertEqual([fact.fact_id for fact in outcome.result.facts], ["fact-b", "fact-a"])
        self.assertIs(outcome.result.turns[0], self.turn_b)
        self.assertIs(outcome.result.facts[0], self.fact_b)
        self.assertEqual(fake.last_query, "preferred drink")
        self.assertEqual(fake.last_timeout, 2.5)

    def test_typed_ids_distinguish_same_raw_turn_and_fact_id(self) -> None:
        result = make_result(
            turns=(make_turn("same", "turn text"),),
            facts=(make_fact("same", "fact text"),),
        )
        fake = FakeReranker(
            lambda _q, candidates, _t: (
                CandidateScore(CandidateId("fact", "same"), 2),
                CandidateScore(CandidateId("turn", "same"), 1),
            )
        )

        outcome = rerank_retrieval_result("query", result, fake)

        self.assertEqual(
            set(outcome.ordered_candidate_ids),
            {CandidateId("turn", "same"), CandidateId("fact", "same")},
        )
        self.assertEqual(outcome.status, "reranked")

    def test_fabricated_identifier_rejects_entire_result(self) -> None:
        fake = FakeReranker(
            lambda _q, candidates, _t: (
                CandidateScore(candidates[0].candidate_id, 1),
                CandidateScore(CandidateId("turn", "fabricated"), 100),
                CandidateScore(candidates[2].candidate_id, 2),
                CandidateScore(candidates[3].candidate_id, 3),
            )
        )
        self.assert_exact_fallback(fake, "unknown_candidate_id")

    def test_missing_identifier_rejects_entire_result(self) -> None:
        fake = FakeReranker(
            lambda _q, candidates, _t: scored(candidates[:3])
        )
        self.assert_exact_fallback(fake, "candidate_set_mismatch")

    def test_duplicate_returned_identifier_rejects_entire_result(self) -> None:
        def output(_q: str, candidates: tuple[RerankCandidate, ...], _t: float):
            return tuple(
                CandidateScore(candidates[0].candidate_id, 1)
                for _ in candidates
            )

        self.assert_exact_fallback(FakeReranker(output), "duplicate_candidate_id")

    def test_duplicate_original_ids_skip_provider_and_preserve_result(self) -> None:
        repeated = make_turn("duplicate", "same candidate")
        result = make_result(turns=(repeated, repeated))
        fake = FakeReranker(())

        outcome = rerank_retrieval_result("query", result, fake)

        self.assertEqual(outcome.status, "skipped")
        self.assertEqual(outcome.reason, "duplicate_input_candidate_ids")
        self.assertIs(outcome.result, result)
        self.assertEqual(fake.calls, 0)

    def test_incorrect_candidate_identifier_types_reject_output(self) -> None:
        invalid_ids = ("turn:turn-a", 42, object())
        for invalid_id in invalid_ids:
            with self.subTest(invalid_id_type=type(invalid_id).__name__):
                fake = FakeReranker(
                    lambda _q, candidates, _t, invalid_id=invalid_id: (
                        CandidateScore(invalid_id, 1),
                        *scored(candidates[1:]),
                    )
                )
                self.assert_exact_fallback(fake, "invalid_candidate_id_type")

    def test_malformed_candidate_id_object_rejects_output_safely(self) -> None:
        malformed_id = object.__new__(CandidateId)
        object.__setattr__(malformed_id, "kind", "turn")
        object.__setattr__(malformed_id, "value", [])
        fake = FakeReranker(
            lambda _q, candidates, _t: (
                CandidateScore(malformed_id, 10),  # type: ignore[arg-type]
                *scored(candidates[1:]),
            )
        )
        self.assert_exact_fallback(fake, "invalid_candidate_id_type")

    def test_non_finite_boolean_and_invalid_scores_reject_entire_result(self) -> None:
        invalid_scores = (math.nan, math.inf, -math.inf, True, "high", None)
        for invalid_score in invalid_scores:
            with self.subTest(score=repr(invalid_score)):
                def output(_q: str, candidates: tuple[RerankCandidate, ...], _t: float):
                    rest = scored(candidates[1:])
                    return (CandidateScore(candidates[0].candidate_id, invalid_score), *rest)

                fake = FakeReranker(output)
                outcome = rerank_retrieval_result("query", self.result, fake)
                self.assertIs(outcome.result, self.result)
                self.assertEqual(outcome.status, "fallback")
                self.assertIn(outcome.reason, {"non_finite_score", "invalid_score_type"})

    def test_ties_use_original_rank_then_typed_identifier(self) -> None:
        turns = (make_turn("z-turn", "turn z"), make_turn("a-turn", "turn a"))
        facts = (make_fact("z-fact", "fact z"), make_fact("a-fact", "fact a"))
        result = make_result(turns=turns, facts=facts)
        fake = FakeReranker(
            lambda _q, candidates, _t: tuple(
                CandidateScore(candidate.candidate_id, 5)
                for candidate in reversed(candidates)
            )
        )

        outcome = rerank_retrieval_result("query", result, fake)

        self.assertEqual(
            outcome.ordered_candidate_ids,
            (
                CandidateId("fact", "z-fact"),
                CandidateId("turn", "z-turn"),
                CandidateId("fact", "a-fact"),
                CandidateId("turn", "a-turn"),
            ),
        )

    def test_provider_exception_falls_back_to_exact_original_result(self) -> None:
        self.assert_exact_fallback(FakeReranker(error=RuntimeError("private details")), "provider_exception")

    def test_provider_system_exit_is_contained_in_worker(self) -> None:
        class ExitingReranker:
            def rerank(
                self,
                _query: str,
                _candidates: tuple[RerankCandidate, ...],
                *,
                timeout: float,
            ):
                raise SystemExit("provider attempted to stop host")

        outcome = rerank_retrieval_result(
            "query", self.result, ExitingReranker()  # type: ignore[arg-type]
        )

        self.assertEqual(outcome.status, "fallback")
        self.assertEqual(outcome.reason, "provider_exception")
        self.assertIs(outcome.result, self.result)

    def test_timeout_falls_back_and_ignores_late_result(self) -> None:
        finished = threading.Event()

        def slow_output(_q: str, candidates: tuple[RerankCandidate, ...], _t: float):
            time.sleep(0.25)
            finished.set()
            return scored(candidates)

        fake = FakeReranker(slow_output)
        start = time.monotonic()
        outcome = rerank_retrieval_result("query", self.result, fake, timeout=0.001)
        elapsed = time.monotonic() - start

        self.assertEqual(outcome.status, "fallback")
        self.assertEqual(outcome.reason, "provider_timeout")
        self.assertIs(outcome.result, self.result)
        self.assertLess(elapsed, 0.15)
        self.assertTrue(finished.wait(1.0))

    def test_timed_out_workers_are_daemon_bounded_and_capacity_fails_closed(self) -> None:
        limit = reranking_module._MAX_CONCURRENT_RERANKERS
        condition = threading.Condition()
        release_workers = threading.Event()
        active_workers = 0
        worker_daemon_flags: list[bool] = []
        provider_calls = 0

        class BlockingReranker:
            def rerank(
                self,
                _query: str,
                candidates: tuple[RerankCandidate, ...],
                *,
                timeout: float,
            ) -> Sequence[CandidateScore]:
                nonlocal active_workers, provider_calls
                with condition:
                    active_workers += 1
                    provider_calls += 1
                    worker_daemon_flags.append(threading.current_thread().daemon)
                    condition.notify_all()
                release_workers.wait(5.0)
                with condition:
                    active_workers -= 1
                    condition.notify_all()
                return scored(candidates)

        reranker = BlockingReranker()
        try:
            with ThreadPoolExecutor(max_workers=limit) as callers:
                requests = [
                    callers.submit(
                        rerank_retrieval_result,
                        "query",
                        self.result,
                        reranker,
                        timeout=0.5,
                    )
                    for _ in range(limit)
                ]
                with condition:
                    all_workers_started = condition.wait_for(
                        lambda: active_workers == limit,
                        timeout=2.0,
                    )
                self.assertTrue(all_workers_started)
                outcomes = [request.result(timeout=2.0) for request in requests]
                self.assertTrue(all(item.reason == "provider_timeout" for item in outcomes))

                saturated = rerank_retrieval_result(
                    "query", self.result, reranker, timeout=0.1
                )
                self.assertEqual(saturated.status, "fallback")
                self.assertEqual(saturated.reason, "provider_capacity_exhausted")
                self.assertIs(saturated.result, self.result)
                self.assertEqual(provider_calls, limit)
                self.assertEqual(worker_daemon_flags, [True] * limit)
        finally:
            release_workers.set()
            with condition:
                workers_released = condition.wait_for(
                    lambda: active_workers == 0,
                    timeout=2.0,
                )
            self.assertTrue(workers_released)

    def test_malformed_provider_outputs_fall_back_exactly(self) -> None:
        malformed_outputs = (None, "not a result", {"candidate": "id"}, [object()])
        for output in malformed_outputs:
            with self.subTest(output_type=type(output).__name__):
                self.assert_exact_fallback(FakeReranker(output), "malformed_output")

    def test_candidate_text_and_metadata_are_immutable_snapshots(self) -> None:
        def output(_q: str, candidates: tuple[RerankCandidate, ...], _t: float):
            with self.assertRaises((AttributeError, TypeError)):
                candidates[0].text = "rewritten"  # type: ignore[misc]
            with self.assertRaises(TypeError):
                candidates[0].metadata["history_id"] = "other-history"  # type: ignore[index]
            return scored(candidates)

        fake = FakeReranker(output)
        outcome = rerank_retrieval_result("query", self.result, fake)

        self.assertEqual(outcome.status, "reranked")
        self.assertEqual(self.turn_a.raw_content, "original turn text A")
        self.assertEqual(self.fact_a.fact["key_text"], "user.preference: tea")

    def test_provenance_and_trust_are_not_mutable_or_exposed_as_ranking_metadata(self) -> None:
        def output(_q: str, candidates: tuple[RerankCandidate, ...], _t: float):
            fact_candidate = next(item for item in candidates if item.candidate_id.kind == "fact")
            self.assertNotIn("trust", fact_candidate.metadata)
            self.assertEqual(fact_candidate.metadata["source_turn_id"], "source-fact-a")
            return scored(candidates)

        original_source = self.fact_a.source_turn
        original_fact_mapping = self.fact_a.fact
        outcome = rerank_retrieval_result("query", self.result, FakeReranker(output))

        self.assertEqual(outcome.status, "reranked")
        fact_a = next(fact for fact in outcome.result.facts if fact.fact_id == "fact-a")
        self.assertIs(fact_a, self.fact_a)
        self.assertIs(fact_a.source_turn, original_source)
        self.assertIs(fact_a.fact, original_fact_mapping)
        self.assertEqual(fact_a.fact["trust"], 0.9)

    def test_history_view_skips_reranking_and_preserves_complete_timeline(self) -> None:
        history = make_result(
            turns=(self.turn_a,),
            facts=(
                make_fact("old", "old fact", valid_from=1),
                make_fact("middle", "middle fact", valid_from=2),
                make_fact("new", "new fact", valid_from=3),
            ),
            view="history",
        )
        fake = FakeReranker(())

        outcome = rerank_retrieval_result("history query", history, fake)

        self.assertEqual(outcome.status, "skipped")
        self.assertEqual(outcome.reason, "history_timeline_preserved")
        self.assertIs(outcome.result, history)
        self.assertEqual([fact.fact_id for fact in outcome.result.facts], ["old", "middle", "new"])
        self.assertEqual(fake.calls, 0)

    def test_empty_candidates_skip_provider(self) -> None:
        empty = make_result()
        fake = FakeReranker(())

        outcome = rerank_retrieval_result("query", empty, fake)

        self.assertEqual(outcome.status, "skipped")
        self.assertEqual(outcome.reason, "no_candidates")
        self.assertIs(outcome.result, empty)
        self.assertEqual(fake.calls, 0)

    def test_non_positive_or_non_finite_timeout_is_rejected_before_provider(self) -> None:
        fake = FakeReranker(())
        for timeout in (0, -1.0, math.inf, math.nan, True, "1"):
            with self.subTest(timeout=repr(timeout)):
                with self.assertRaises((TypeError, ValueError)):
                    rerank_retrieval_result("query", self.result, fake, timeout=timeout)  # type: ignore[arg-type]
        self.assertEqual(fake.calls, 0)

    def assert_exact_fallback(self, fake: FakeReranker, reason: str) -> None:
        original_turns = self.result.turns
        original_facts = self.result.facts
        outcome = rerank_retrieval_result("private query", self.result, fake)
        self.assertEqual(outcome.status, "fallback")
        self.assertEqual(outcome.reason, reason)
        self.assertIs(outcome.result, self.result)
        self.assertIs(outcome.result.turns, original_turns)
        self.assertIs(outcome.result.facts, original_facts)
        self.assertEqual(
            [turn.turn_id for turn in outcome.result.turns], ["turn-a", "turn-b"]
        )
        self.assertEqual(
            [fact.fact_id for fact in outcome.result.facts], ["fact-a", "fact-b"]
        )


class RetrievalCompatibilityTests(unittest.TestCase):
    def test_phase4a_result_can_be_reranked_without_database_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / "reranking.sqlite3")
            initialize_database(database)
            provider = FakeEmbeddingProvider()
            with connection(database) as conn:
                for turn_idx, turn_id, content in (
                    (0, "phase4a-a", "project plan includes retrieval"),
                    (1, "phase4a-b", "project plan includes evidence"),
                ):
                    event_time = float(100 + turn_idx)
                    conn.execute(
                        """INSERT INTO turns (turn_id, history_id, session_id, turn_idx,
                           role, raw_content, source_type, event_time, recorded_at, hash)
                           VALUES (?, 'h1', ?, ?, 'user', ?, 'user_turn', ?, ?, ?)""",
                        (turn_id, turn_id, turn_idx, content, event_time,
                         event_time + 10, f"hash-{turn_id}"),
                    )
                    record_store_event(conn, turn_id, created_at=event_time + 10)
            index_history(database, "h1", provider)
            phase4a = retrieve_memory(database, "h1", "project plan", provider, top_k=2)
            self.assertIsInstance(phase4a, RetrievalResult)
            self.assertEqual(len(phase4a.turns), 2)

            with connection(database) as conn:
                before = (
                    conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0],
                    conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0],
                    conn.execute("SELECT COUNT(*) FROM gate_events").fetchone()[0],
                )

            fake = FakeReranker(
                lambda _q, candidates, _t: tuple(
                    CandidateScore(item.candidate_id, item.original_rank)
                    for item in reversed(candidates)
                )
            )
            outcome = rerank_retrieval_result("project plan", phase4a, fake)

            self.assertEqual(outcome.status, "reranked")
            self.assertEqual(
                {turn.turn_id for turn in outcome.result.turns},
                {turn.turn_id for turn in phase4a.turns},
            )
            self.assertEqual(
                {turn.raw_content for turn in outcome.result.turns},
                {turn.raw_content for turn in phase4a.turns},
            )
            with connection(database) as conn:
                after = (
                    conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0],
                    conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0],
                    conn.execute("SELECT COUNT(*) FROM gate_events").fetchone()[0],
                )
            self.assertEqual(after, before)


class FakeEmbeddingProvider:
    model_id = "fake-reranking-model"
    dimension = BGE_M3_DIMENSION
    device = "cpu"

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        index = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little") % self.dimension
        vector[index] = 1.0
        return vector

    def encode_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def encode_documents(self, texts: list[str] | tuple[str, ...]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.vstack([self._vector(text) for text in texts])


if __name__ == "__main__":
    unittest.main()
