"""SQLite integration tests for hybrid, temporal, and quarantine-aware retrieval."""

from __future__ import annotations

import hashlib
import inspect
import sys
from pathlib import Path
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engramOS.db import connection, initialize_database  # noqa: E402
from engramOS.embeddings import (  # noqa: E402
    BGE_M3_DIMENSION,
    DenseIndexStaleError,
    check_index_freshness,
    index_history,
)
from engramOS.eligibility import (  # noqa: E402
    fact_eligibility_predicate,
    fact_is_eligible,
    observation_eligibility_predicate,
    turn_eligibility_predicate,
    turn_is_eligible,
)
import engramOS.eligibility as eligibility_module  # noqa: E402
from engramOS.ingestion import TurnInput, ingest_turns_only  # noqa: E402
from engramOS.retrieval import (  # noqa: E402
    DenseRetrievalUnavailable,
    hybrid_rag,
    recency,
    reciprocal_rank_fusion,
    retrieve_memory,
    retrieve_temporal_facts,
    safe_fts_query,
    search_facts_dense,
    search_facts_fts,
    search_turns_dense,
    search_turns_fts,
)
from engramOS.temporal import FactInput, insert_fact  # noqa: E402
import engramOS.retrieval as retrieval_module  # noqa: E402
from tests.helpers import record_store_event  # noqa: E402


class FakeEmbeddingProvider:
    model_id = "fake-bge-m3-1024"
    dimension = BGE_M3_DIMENSION
    device = "cpu"

    def __init__(self, mapping: dict[str, np.ndarray] | None = None) -> None:
        self.mapping = mapping or {}
        self.queries: list[str] = []

    def _vector(self, text: str) -> np.ndarray:
        if text in self.mapping:
            return self.mapping[text]
        vector = np.zeros(self.dimension, dtype=np.float32)
        idx = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little") % self.dimension
        vector[idx] = 1.0
        return vector

    def encode_query(self, text: str) -> np.ndarray:
        self.queries.append(text)
        return self._vector(text)

    def encode_documents(self, texts: list[str] | tuple[str, ...]) -> np.ndarray:
        return np.vstack([self._vector(text) for text in texts]) if texts else np.empty((0, self.dimension))


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = str(Path(self.temp_dir.name) / "retrieval.sqlite3")
        initialize_database(self.database)
        self.provider = FakeEmbeddingProvider()
        self.serial = 0

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def add_turn(
        self,
        text: str,
        *,
        history: str = "h1",
        event_time: float | None = None,
        turn_id: str | None = None,
        session_id: str | None = None,
        turn_idx: int = 0,
        gated: bool = True,
    ) -> str:
        self.serial += 1
        turn_id = turn_id or f"{history}:session-{self.serial}:{turn_idx}"
        session_id = session_id or f"session-{self.serial}"
        event_time = float(event_time if event_time is not None else self.serial * 100)
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO turns (turn_id, history_id, session_id, turn_idx, role,
                   raw_content, source_type, event_time, recorded_at, hash)
                   VALUES (?, ?, ?, ?, 'user', ?, 'user_turn', ?, ?, ?)""",
                (turn_id, history, session_id, turn_idx, text, event_time,
                 event_time + 1000, f"hash-{turn_id}"),
            )
            if gated:
                record_store_event(conn, turn_id, created_at=event_time + 1000)
        return turn_id

    def add_fact(
        self,
        turn_id: str,
        fact_id: str,
        value: str,
        valid_from: float,
        *,
        history: str = "h1",
        key_text: str | None = None,
        fact_seq: int = 0,
    ) -> str:
        return insert_fact(
            self.database,
            FactInput(
                fact_id, history, turn_id, fact_seq, "user", "city", value,
                "single", valid_from, valid_from + 1000,
                key_text=key_text or f"user.city: {value}",
            ),
        )

    def quarantine(self, turn_id: str, *, history_only: bool = False) -> None:
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO gate_events (turn_id, signal_bridge, signal_policy,
                   signal_source, signals_fired, decision, created_at)
                   VALUES (?,1,1,0,2,'quarantine',999999)""",
                (turn_id,),
            )

    def add_trigger(self, turn_id: str, parent_fact_id: str, cue: str, valid_from: float) -> str:
        trigger_id = f"trigger-{parent_fact_id}"
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO facts (fact_id, history_id, turn_id, fact_seq, key_type,
                   key_text, parent_fact_id, valid_from, recorded_at)
                   VALUES (?, 'h1', ?, 99, 'trigger', ?, ?, ?, ?)""",
                (trigger_id, turn_id, cue, parent_fact_id, valid_from, valid_from + 1),
            )
        return trigger_id

    def test_safe_fts_query_quotes_terms_and_handles_empty_input(self) -> None:
        self.add_turn("We discussed alpha in the plan")
        self.add_turn("A different note mentions beta")
        self.assertEqual(safe_fts_query("alpha OR * NEAR("")"), '"alpha" OR "OR" OR "NEAR"')
        self.assertEqual(search_turns_fts(self.database, "h1", "... !!!"), [])
        matches = search_turns_fts(self.database, "h1", 'alpha" OR *')
        self.assertEqual([item.record["raw_content"] for item in matches], ["We discussed alpha in the plan"])
        self.assertEqual([item.rank for item in matches], [1])

    def test_fts_multiple_tokens_bm25_order_and_history_isolation(self) -> None:
        self.add_turn("alpha beta", turn_id="a", session_id="a")
        self.add_turn("alpha beta alpha beta", turn_id="b", session_id="b")
        self.add_turn("alpha beta in another history", history="h2", turn_id="x", session_id="x")
        hits = search_turns_fts(self.database, "h1", "alpha beta")
        self.assertEqual({hit.ref_id for hit in hits}, {"a", "b"})
        self.assertEqual([hit.rank for hit in hits], [1, 2])
        self.assertEqual(hits[0].ref_id, "b")
        self.assertTrue(hits[0].bm25_score < hits[1].bm25_score)

    def test_missing_terms_and_injection_do_not_break_fts(self) -> None:
        self.add_turn("ordinary searchable sentence")
        self.assertEqual(search_turns_fts(self.database, "h1", "absentword"), [])
        self.assertEqual(search_turns_fts(self.database, "h1", '" OR * ) NEAR(foo'), [])

    def test_fts_and_dense_both_exclude_quarantined_turns(self) -> None:
        clean = self.add_turn("Paris launch planning", turn_id="clean", session_id="clean")
        poisoned = self.add_turn("Paris launch planning", turn_id="poison", session_id="poison")
        index_history(self.database, "h1", self.provider)
        self.quarantine(poisoned)
        self.assertEqual([hit.ref_id for hit in search_turns_fts(self.database, "h1", "Paris")], [clean])
        self.assertEqual(
            [hit.ref_id for hit in search_turns_dense(self.database, "h1", "Paris launch planning", self.provider)],
            [clean],
        )

    def test_fact_fts_and_dense_exclude_quarantined_source(self) -> None:
        clean_turn = self.add_turn("Clean city memory", turn_id="clean-turn", session_id="clean")
        poisoned_turn = self.add_turn("Poison city memory", turn_id="poison-turn", session_id="poison")
        self.add_fact(clean_turn, "f-clean", "Paris", 100)
        self.add_fact(poisoned_turn, "f-poison", "Rome", 200)
        index_history(self.database, "h1", self.provider)
        self.quarantine(poisoned_turn)
        self.assertEqual(
            [hit.ref_id for hit in search_facts_fts(self.database, "h1", "city")], ["f-clean"]
        )
        self.assertEqual(
            [hit.ref_id for hit in search_facts_dense(self.database, "h1", "user.city: Paris", self.provider)],
            ["f-clean"],
        )

    def test_current_as_of_date_only_and_history_fact_views(self) -> None:
        start = 1_715_299_200.0  # 2024-05-10 00:00:00 UTC
        early_turn = self.add_turn("Paris became the city", event_time=start + 86_500, turn_id="early", session_id="early")
        later_turn = self.add_turn("Rome replaced Paris", event_time=start + 172_800, turn_id="later", session_id="later")
        self.add_fact(early_turn, "paris-fact", "Paris", start - 60, key_text="user.city: Paris")
        self.add_fact(later_turn, "rome-fact", "Rome", start + 86_400, key_text="user.city: Rome")
        index_history(self.database, "h1", self.provider)

        current = retrieve_memory(self.database, "h1", "city", self.provider, view="current")
        self.assertEqual([result.fact["value"] for result in current.facts], ["Rome"])
        date_view = retrieve_memory(
            self.database, "h1", "city", self.provider, view="as_of", as_of="2024-05-10"
        )
        self.assertEqual([result.fact["value"] for result in date_view.facts], ["Paris"])
        self.assertGreater(date_view.facts[0].source_turn.event_time, float(date_view.facts[0].fact["valid_from"]))
        history_view = retrieve_memory(
            self.database, "h1", "city", self.provider, view="history", fact_top_k=1
        )
        self.assertEqual({result.fact["value"] for result in history_view.facts}, {"Paris", "Rome"})
        temporal_history = retrieve_temporal_facts(self.database, "h1", view="history")
        self.assertEqual([item.fact["value"] for item in temporal_history], ["Paris", "Rome"])
        self.assertLess(
            float(temporal_history[0].fact["valid_from"]),
            float(temporal_history[1].fact["valid_from"]),
        )
        paris = next(result for result in history_view.facts if result.fact["value"] == "Paris")
        self.assertEqual(paris.annotation, f"[valid: {start - 60} -> {start + 86_400.0}]")

    def test_as_of_temporal_retrieval_excludes_future_multi_observation(self) -> None:
        earlier_turn = self.add_turn(
            "I enjoy reading", event_time=100, turn_id="multi-earlier", session_id="multi-earlier"
        )
        future_turn = self.add_turn(
            "I enjoy hiking", event_time=300, turn_id="multi-future", session_id="multi-future"
        )
        insert_fact(
            self.database,
            FactInput(
                "multi-reading", "h1", earlier_turn, 0, "user", "hobby", "Reading",
                "multi", 100, 1100,
            ),
        )
        insert_fact(
            self.database,
            FactInput(
                "multi-hiking", "h1", future_turn, 0, "user", "hobby", "Hiking",
                "multi", 300, 1300,
            ),
        )

        result = retrieve_temporal_facts(
            self.database, "h1", view="as_of", as_of=200, entity="user", attribute="hobby"
        )

        self.assertEqual(
            [(item.fact_id, item.fact["value"]) for item in result],
            [("multi-reading", "Reading")],
        )
        self.assertEqual(len(result), 1)

    def test_ranked_as_of_retrieval_excludes_future_multi_observation(self) -> None:
        earlier_turn = self.add_turn(
            "I enjoy reading every evening",
            event_time=100,
            turn_id="ranked-multi-earlier",
            session_id="ranked-multi-earlier",
        )
        future_turn = self.add_turn(
            "I enjoy hiking on weekends",
            event_time=300,
            turn_id="ranked-multi-future",
            session_id="ranked-multi-future",
        )
        insert_fact(
            self.database,
            FactInput(
                "multi-reading", "h1", earlier_turn, 0, "user", "hobby", "Reading",
                "multi", 100, 1100, key_text="user.hobby: Reading",
            ),
        )
        insert_fact(
            self.database,
            FactInput(
                "multi-hiking", "h1", future_turn, 0, "user", "hobby", "Hiking",
                "multi", 300, 1300, key_text="user.hobby: Hiking",
            ),
        )
        index_history(self.database, "h1", self.provider)

        result = retrieve_memory(
            self.database, "h1", "hiking", self.provider,
            view="as_of", as_of=200,
        )

        self.assertEqual([fact.fact_id for fact in result.facts], ["multi-reading"])
        self.assertEqual(result.diagnostics.channel_candidate_counts["fact_lexical"], 0)
        self.assertEqual(result.diagnostics.channel_candidate_counts["fact_dense"], 1)
        reading = result.facts[0]
        self.assertEqual(reading.source_turn.turn_id, earlier_turn)
        self.assertEqual([turn.turn_id for turn in reading.supporting_turns], [earlier_turn])
        for fact in result.facts:
            self.assertNotEqual(fact.fact_id, "multi-hiking")
            self.assertNotEqual(fact.source_turn.turn_id, future_turn)
            if fact.confirmation_turn is not None:
                self.assertNotEqual(fact.confirmation_turn.turn_id, future_turn)
            self.assertNotIn(future_turn, {turn.turn_id for turn in fact.supporting_turns})

    def test_gated_turn_retrieval_requires_indexed_eligible_source_set(self) -> None:
        clean = self.add_turn("alpha clean memory", turn_id="clean-gated", session_id="clean-gated")
        quarantined = self.add_turn(
            "alpha quarantined memory", turn_id="poison-gated", session_id="poison-gated"
        )
        self.quarantine(quarantined)
        index_history(self.database, "h1", self.provider, gated=True)
        self.assertEqual(
            {turn.turn_id for turn in hybrid_rag(self.database, "h1", "alpha", self.provider).turns},
            {clean},
        )
        with self.assertRaises(DenseIndexStaleError):
            hybrid_rag(self.database, "h1", "alpha", self.provider, gated=False)

    def test_ungated_turn_indexes_are_filtered_again_by_gated_retrieval(self) -> None:
        clean = self.add_turn("alpha clean memory", turn_id="clean-ungated", session_id="clean-ungated")
        quarantined = self.add_turn(
            "alpha quarantined memory", turn_id="poison-ungated", session_id="poison-ungated"
        )
        self.quarantine(quarantined)
        index_history(self.database, "h1", self.provider, gated=False)
        self.assertEqual(
            check_index_freshness(
                self.database, "h1", self.provider, ref_types=("turn",), gated=False
            ).required_count,
            2,
        )
        self.assertEqual(
            check_index_freshness(
                self.database, "h1", self.provider, ref_types=("turn",), gated=True
            ).required_count,
            1,
        )

        self.assertEqual(
            {turn.turn_id for turn in hybrid_rag(
                self.database, "h1", "alpha", self.provider, gated=False
            ).turns},
            {clean, quarantined},
        )
        self.assertEqual(
            {turn.turn_id for turn in recency(
                self.database, "h1", "alpha", self.provider, gated=False
            ).turns},
            {clean, quarantined},
        )
        self.assertEqual(
            {turn.ref_id for turn in search_turns_fts(
                self.database, "h1", "alpha", gated=False
            )},
            {clean, quarantined},
        )
        self.assertEqual(
            {turn.ref_id for turn in search_turns_dense(
                self.database, "h1", "alpha", self.provider, gated=False
            )},
            {clean, quarantined},
        )
        default_hybrid = hybrid_rag(self.database, "h1", "alpha", self.provider)
        default_recency = recency(self.database, "h1", "alpha", self.provider)
        self.assertEqual({turn.turn_id for turn in default_hybrid.turns}, {clean})
        self.assertEqual({turn.turn_id for turn in default_recency.turns}, {clean})
        self.assertTrue(default_hybrid.diagnostics.gated)
        self.assertTrue(default_recency.diagnostics.gated)
        self.assertFalse(
            hybrid_rag(self.database, "h1", "alpha", self.provider, gated=False)
            .diagnostics.gated
        )
        self.assertFalse(
            recency(self.database, "h1", "alpha", self.provider, gated=False)
            .diagnostics.gated
        )
        self.assertEqual(
            {turn.ref_id for turn in search_turns_fts(self.database, "h1", "alpha")},
            {clean},
        )
        self.assertEqual(
            {turn.ref_id for turn in search_turns_dense(
                self.database, "h1", "alpha", self.provider
            )},
            {clean},
        )

    def test_ungated_turn_indexing_does_not_expose_quarantined_facts_or_evidence(self) -> None:
        clean = self.add_turn("clean memory", turn_id="fact-clean", session_id="fact-clean")
        quarantined = self.add_turn(
            "poison memory", turn_id="fact-poison", session_id="fact-poison"
        )
        self.add_fact(quarantined, "poison-fact", "Hiking", 300, key_text="user.hobby: Hiking")
        self.quarantine(quarantined)
        index_history(self.database, "h1", self.provider, gated=False)
        with connection(self.database) as conn:
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE ref_type='fact' AND ref_id='poison-fact'"
                ).fetchone()[0],
                0,
            )
        result = retrieve_memory(self.database, "h1", "Hiking", self.provider)
        self.assertTrue(result.diagnostics.gated)
        self.assertEqual([fact.fact_id for fact in result.facts], [])
        self.assertEqual(
            [turn.turn_id for fact in result.facts for turn in fact.supporting_turns], []
        )
        self.assertEqual([turn.turn_id for turn in result.turns], [clean])
        self.assertNotIn(quarantined, {turn.turn_id for turn in result.turns})

    def test_turn_only_gated_signatures_and_fact_index_guard(self) -> None:
        self.assertEqual(turn_eligibility_predicate("turn_alias", gated=False), "1 = 1")
        with self.assertRaises(ValueError):
            turn_eligibility_predicate("bad;alias", gated=False)
        for function in (
            retrieve_memory,
            retrieve_temporal_facts,
            search_facts_fts,
            search_facts_dense,
            fact_eligibility_predicate,
            observation_eligibility_predicate,
        ):
            self.assertNotIn("gated", inspect.signature(function).parameters)
        with connection(self.database) as conn:
            with self.assertRaises(ValueError):
                retrieval_module.InMemoryVectorIndex.load_from_connection(
                    conn, "h1", "fact", self.provider.model_id, gated=False
                )
        with self.assertRaises(ValueError):
            retrieval_module.InMemoryVectorIndex.load(
                self.database, "h1", "fact", self.provider.model_id, gated=False
            )

    def test_turns_only_baseline_indexes_without_any_gate_events(self) -> None:
        turns = (
            TurnInput(
                "baseline-history", "session-a", 0, "user", "alpha baseline memory",
                100.0, source_type="user_turn", session_date="1970-01-01",
            ),
            TurnInput(
                "baseline-history", "session-b", 0, "user", "beta baseline memory",
                200.0, source_type="user_turn", session_date="1970-01-01",
            ),
        )
        persisted = ingest_turns_only(self.database, turns, recorded_at=500.0)
        expected = {item.turn_id for item in persisted}
        with connection(self.database) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM gate_events").fetchone()[0], 0)
        index_history(self.database, "baseline-history", self.provider, gated=False)
        result = hybrid_rag(
            self.database, "baseline-history", "alpha", self.provider, gated=False
        )
        self.assertEqual({turn.turn_id for turn in result.turns}, expected)
        self.assertFalse(result.diagnostics.gated)

    def test_confirmation_evidence_is_deduplicated_and_provenance_aware(self) -> None:
        original = self.add_turn("I live in Paris", event_time=500, turn_id="original", session_id="original")
        confirmation = self.add_turn("I still live in Paris", event_time=900, turn_id="confirm", session_id="confirm")
        self.add_fact(original, "paris-first", "Paris", 100)
        self.add_fact(confirmation, "paris-confirm", " paris ", 200)
        index_history(self.database, "h1", self.provider)
        result = retrieve_memory(self.database, "h1", "city Paris", self.provider, view="history")
        self.assertEqual(len(result.facts), 1)
        fact = result.facts[0]
        self.assertEqual(fact.source_turn.turn_id, original)
        self.assertEqual(fact.confirmation_turn.turn_id, confirmation)
        self.assertEqual([turn.turn_id for turn in fact.supporting_turns], [original, confirmation])

    def test_quarantined_confirmation_turn_is_not_exposed_as_evidence(self) -> None:
        original = self.add_turn("I live in Paris", event_time=100, turn_id="origin", session_id="origin")
        confirmation = self.add_turn("Still in Paris", event_time=300, turn_id="confirmation", session_id="confirmation")
        self.add_fact(original, "origin-fact", "Paris", 100)
        self.add_fact(confirmation, "confirm-fact", "Paris", 200)
        index_history(self.database, "h1", self.provider)
        self.quarantine(confirmation)
        result = retrieve_memory(self.database, "h1", "city Paris", self.provider, view="history")
        self.assertTrue(result.facts)
        for fact in result.facts:
            self.assertNotIn(confirmation, {turn.turn_id for turn in fact.supporting_turns})

    def test_quarantined_parent_fact_excludes_trigger_from_fts_and_dense(self) -> None:
        source = self.add_turn("Use port 5432 for the database", turn_id="source", session_id="source")
        parent_id = self.add_fact(source, "policy-fact", "port 5432", 100, key_text="database.port: 5432")
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO facts (fact_id, history_id, turn_id, fact_seq, key_type,
                   key_text, parent_fact_id, valid_from, recorded_at)
                   VALUES ('cue1','h1',?,1,'trigger','When connecting to storage',?,100,200)""",
                (source, parent_id),
            )
        index_history(self.database, "h1", self.provider)
        with connection(self.database) as conn:
            conn.execute("UPDATE facts SET quarantined=1 WHERE fact_id=?", (parent_id,))
        self.assertEqual(search_facts_fts(self.database, "h1", "connecting"), [])
        self.assertEqual(
            search_facts_dense(self.database, "h1", "When connecting to storage", self.provider), []
        )

    def test_cross_history_and_sticky_quarantine_after_store(self) -> None:
        clean = self.add_turn("shared terms", history="h1", turn_id="clean", session_id="clean")
        other = self.add_turn("shared terms", history="h2", turn_id="other", session_id="other")
        index_history(self.database, "h1", self.provider)
        index_history(self.database, "h2", self.provider)
        self.quarantine(clean)
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO gate_events (turn_id, signal_bridge, signal_policy,
                   signal_source, signals_fired, decision, created_at)
                   VALUES (?,0,0,0,0,'store',1000000)""",
                (clean,),
            )
        self.assertEqual(search_turns_fts(self.database, "h1", "shared"), [])
        self.assertEqual([hit.ref_id for hit in search_turns_fts(self.database, "h2", "shared")], [other])
        self.assertEqual(search_turns_dense(self.database, "h1", "shared terms", self.provider), [])
        self.assertEqual([hit.ref_id for hit in search_turns_dense(self.database, "h2", "shared terms", self.provider)], [other])

    def test_hybrid_requires_dense_and_uses_lexical_and_dense_channels(self) -> None:
        with self.assertRaises(DenseRetrievalUnavailable):
            hybrid_rag(self.database, "h1", "alpha", self.provider)
        for idx in range(12):
            self.add_turn(
                f"alpha project reference {idx}", turn_id=f"t{idx:02d}",
                session_id=f"session-{idx:02d}", event_time=idx + 1,
            )
        index_history(self.database, "h1", self.provider)
        result = hybrid_rag(self.database, "h1", "alpha project", self.provider, top_k=10)
        self.assertEqual(len(result.turns), 10)
        self.assertGreater(result.diagnostics.channel_candidate_counts["turn_lexical"], 0)
        self.assertGreater(result.diagnostics.channel_candidate_counts["turn_dense"], 0)
        self.assertTrue(all("turn_lexical" in turn.channels and "turn_dense" in turn.channels for turn in result.turns[:5]))

    def test_stale_dense_index_fails_closed_and_reindexing_repairs_it(self) -> None:
        self.add_turn("initial retrieval source", turn_id="initial", session_id="initial")
        index_history(self.database, "h1", self.provider)
        self.add_turn("new retrieval source", turn_id="new", session_id="new", event_time=200)
        insert_fact(
            self.database,
            FactInput("new-retrieval-fact", "h1", "initial", 0, "user", "city", "Rome", "single", 150, 250),
        )
        with self.assertRaisesRegex(DenseIndexStaleError, "run index_history"):
            retrieve_memory(self.database, "h1", "retrieval source", self.provider)
        index_history(self.database, "h1", self.provider)
        result = retrieve_memory(self.database, "h1", "retrieval source", self.provider)
        self.assertIn("new", {turn.turn_id for turn in result.turns})

    def test_dense_retrieval_cannot_mix_embedding_model_identities(self) -> None:
        self.add_turn("model identity source", turn_id="model-source", session_id="model-source")
        index_history(self.database, "h1", self.provider)

        class OtherRevision(FakeEmbeddingProvider):
            model_id = "fake-bge-m3-other-revision"

        with self.assertRaisesRegex(DenseIndexStaleError, "model_identity_mismatch"):
            search_turns_dense(
                self.database, "h1", "model identity source", OtherRevision()
            )

    def test_eligible_trigger_fts_dense_evidence_and_temporal_views(self) -> None:
        origin = self.add_turn(
            "I prefer aisle seats", turn_id="trigger-origin", session_id="trigger-origin",
            event_time=100,
        )
        confirmation = self.add_turn(
            "I still prefer an aisle seat", turn_id="trigger-confirm", session_id="trigger-confirm",
            event_time=200,
        )
        parent_id = insert_fact(
            self.database,
            FactInput(
                "seat-preference", "h1", origin, 0, "user", "seat", "aisle",
                "single", 100, 150, key_text="user.seat: aisle",
            ),
        )
        insert_fact(
            self.database,
            FactInput("seat-confirmation", "h1", confirmation, 0, "user", "seat", "aisle", "single", 200, 300),
        )
        cue = "Find a place for my next trip"
        trigger_id = self.add_trigger(origin, parent_id, cue, 100)
        self.provider.mapping[cue] = np.eye(1, BGE_M3_DIMENSION, 7, dtype=np.float32)[0]
        index_history(self.database, "h1", self.provider)

        lexical = search_facts_fts(self.database, "h1", "next trip", view="current")
        trigger_lexical = next(item for item in lexical if item.ref_id == trigger_id)
        self.assertEqual(trigger_lexical.record["parent_fact"]["fact_id"], parent_id)
        dense = search_facts_dense(self.database, "h1", cue, self.provider, view="current")
        trigger_dense = next(item for item in dense if item.ref_id == trigger_id)
        self.assertEqual(trigger_dense.record["parent_fact"]["fact_id"], parent_id)

        expanded = retrieve_memory(self.database, "h1", "next trip", self.provider, view="history")
        trigger_result = next(item for item in expanded.facts if item.fact_id == trigger_id)
        self.assertEqual(trigger_result.parent_fact["fact_id"], parent_id)
        self.assertEqual(trigger_result.source_turn.turn_id, origin)
        self.assertEqual(trigger_result.confirmation_turn.turn_id, confirmation)
        self.assertEqual(
            {turn.turn_id for turn in trigger_result.supporting_turns}, {origin, confirmation}
        )

        # The trigger remains available for an as-of date covered by its parent.
        old_view = search_facts_fts(self.database, "h1", "next trip", view="as_of", as_of=250)
        self.assertIn(trigger_id, {item.ref_id for item in old_view})
        dense_old_view = search_facts_dense(
            self.database, "h1", cue, self.provider, view="as_of", as_of=250
        )
        self.assertIn(trigger_id, {item.ref_id for item in dense_old_view})
        changed = self.add_turn(
            "I prefer window seats now", turn_id="trigger-change", session_id="trigger-change",
            event_time=400,
        )
        insert_fact(
            self.database,
            FactInput("seat-window", "h1", changed, 0, "user", "seat", "window", "single", 300, 500),
        )
        index_history(self.database, "h1", self.provider)
        self.assertNotIn(
            trigger_id,
            {item.ref_id for item in search_facts_fts(self.database, "h1", "next trip", view="current")},
        )
        self.assertNotIn(
            trigger_id,
            {item.ref_id for item in search_facts_fts(self.database, "h1", "next trip", view="as_of", as_of=350)},
        )
        self.assertNotIn(
            trigger_id,
            {item.ref_id for item in search_facts_dense(
                self.database, "h1", cue, self.provider, view="as_of", as_of=350
            )},
        )

    def test_trigger_remains_excluded_after_parent_quarantine_and_later_store(self) -> None:
        origin = self.add_turn("source preference", turn_id="quarantine-origin", session_id="quarantine-origin")
        parent_id = insert_fact(
            self.database,
            FactInput(
                "parent-trigger-fact", "h1", origin, 0, "user", "seat", "aisle",
                "single", 100, 200, key_text="seat preference aisle",
            ),
        )
        trigger_id = self.add_trigger(origin, parent_id, "choose a flight seat", 100)
        index_history(self.database, "h1", self.provider)
        self.quarantine(origin)
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO gate_events (turn_id, signal_bridge, signal_policy,
                   signal_source, signals_fired, decision, created_at)
                   VALUES (?,0,0,0,0,'store',1000)""",
                (origin,),
            )
        self.assertNotIn(
            trigger_id,
            {item.ref_id for item in search_facts_fts(self.database, "h1", "flight seat", view="history")},
        )
        self.assertNotIn(
            trigger_id,
            {item.ref_id for item in search_facts_dense(
                self.database, "h1", "choose a flight seat", self.provider, view="history"
            )},
        )

    def test_retrieve_memory_uses_one_consistent_read_snapshot_during_quarantine(self) -> None:
        origin = self.add_turn(
            "snapshot memory phrase", turn_id="snapshot-origin", session_id="snapshot-origin",
            event_time=100,
        )
        self.add_turn(
            "unrelated clean memory", turn_id="snapshot-clean", session_id="snapshot-clean",
            event_time=200,
        )
        parent_id = self.add_fact(origin, "snapshot-fact", "Paris", 100, key_text="user.city: Paris")
        index_history(self.database, "h1", self.provider)

        reached_snapshot_read = threading.Event()
        continue_reader = threading.Event()
        original_search = retrieval_module._search_turns_fts

        def pause_after_first_snapshot_query(conn, history_id, query, *, limit):
            candidates = original_search(conn, history_id, query, limit=limit)
            reached_snapshot_read.set()
            if not continue_reader.wait(timeout=5):
                raise TimeoutError("test writer did not release snapshot reader")
            return candidates

        with mock.patch.object(retrieval_module, "_search_turns_fts", pause_after_first_snapshot_query):
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    retrieve_memory, self.database, "h1", "snapshot memory phrase", self.provider,
                    view="history",
                )
                self.assertTrue(reached_snapshot_read.wait(timeout=5))
                try:
                    # WAL lets this committed writer proceed while the reader keeps its old snapshot.
                    self.quarantine(origin)
                finally:
                    continue_reader.set()
                before_quarantine = future.result(timeout=5)
        self.assertIn(origin, {turn.turn_id for turn in before_quarantine.turns})
        fact_result = next(item for item in before_quarantine.facts if item.fact_id == parent_id)
        self.assertEqual(fact_result.source_turn.turn_id, origin)

        after_quarantine = retrieve_memory(
            self.database, "h1", "snapshot memory phrase", self.provider, view="history"
        )
        self.assertNotIn(origin, {turn.turn_id for turn in after_quarantine.turns})
        self.assertNotIn(parent_id, {item.fact_id for item in after_quarantine.facts})

    def test_temporal_fact_retrieval_keeps_projection_and_evidence_in_one_snapshot(self) -> None:
        origin = self.add_turn(
            "temporal snapshot source", turn_id="temporal-snapshot-origin",
            session_id="temporal-snapshot-origin", event_time=100,
        )
        fact_id = self.add_fact(
            origin, "temporal-snapshot-fact", "Paris", 100, key_text="user.city: Paris"
        )
        indexed = index_history(self.database, "h1", self.provider)
        self.assertGreater(indexed.indexed, 0)

        reached_evidence_read = threading.Event()
        continue_reader = threading.Event()
        original_expand = retrieval_module._expand_fact_evidence

        def pause_after_evidence_read(conn, history_id, fused, allowed, lexical, dense):
            evidence = original_expand(conn, history_id, fused, allowed, lexical, dense)
            reached_evidence_read.set()
            if not continue_reader.wait(timeout=5):
                raise TimeoutError("test writer did not release temporal reader")
            return evidence

        with mock.patch.object(retrieval_module, "_expand_fact_evidence", pause_after_evidence_read):
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(retrieve_temporal_facts, self.database, "h1", view="current")
                self.assertTrue(reached_evidence_read.wait(timeout=5))
                try:
                    self.quarantine(origin)
                finally:
                    continue_reader.set()
                pre_quarantine = future.result(timeout=5)
        result = next(item for item in pre_quarantine if item.fact_id == fact_id)
        self.assertEqual(result.source_turn.turn_id, origin)

        post_quarantine = retrieve_temporal_facts(self.database, "h1", view="current")
        self.assertNotIn(fact_id, {item.fact_id for item in post_quarantine})

    def test_recency_sorts_the_same_top_k_hybrid_candidates(self) -> None:
        for idx, when in enumerate((30, 10, 20)):
            self.add_turn(
                "alpha shared planning text", turn_id=f"t{idx}",
                session_id=f"session-{2-idx}", event_time=when,
            )
        index_history(self.database, "h1", self.provider)
        hybrid = hybrid_rag(self.database, "h1", "alpha shared", self.provider, top_k=2)
        recent = recency(self.database, "h1", "alpha shared", self.provider, top_k=2)
        self.assertEqual({t.turn_id for t in hybrid.turns}, {t.turn_id for t in recent.turns})
        self.assertEqual([t.event_time for t in recent.turns], sorted([t.event_time for t in recent.turns], reverse=True))

    def test_rrf_formula_shared_candidates_ties_and_duplicate_elimination(self) -> None:
        fused = reciprocal_rank_fusion(
            {"lex": ["a", "b", "b"], "dense": ["b", "c"]}, k=60
        )
        by_id = {item.ref_id: item for item in fused}
        self.assertAlmostEqual(by_id["a"].rrf_score, 1 / 61)
        self.assertAlmostEqual(by_id["b"].rrf_score, 1 / 62 + 1 / 61)
        self.assertAlmostEqual(by_id["c"].rrf_score, 1 / 62)
        self.assertEqual([item.ref_id for item in fused], ["b", "a", "c"])
        self.assertEqual(by_id["b"].channels, ("lex", "dense"))

    def test_store_required_turn_eligibility_matrix_across_turn_surfaces(self) -> None:
        histories = (
            ("none", (), False),
            ("store", ("store",), True),
            ("quarantine", ("quarantine",), False),
            ("store-quarantine", ("store", "quarantine"), False),
            ("quarantine-store", ("quarantine", "store"), False),
            ("store-store", ("store", "store"), True),
        )
        ids: dict[str, str] = {}
        for label, decisions, _expected in histories:
            turn_id = self.add_turn(
                "eligibility matrix marker", turn_id=f"matrix-{label}",
                session_id=f"matrix-{label}", gated=False,
            )
            ids[label] = turn_id
            with connection(self.database) as conn:
                for event_idx, decision in enumerate(decisions):
                    conn.execute(
                        """INSERT INTO gate_events (turn_id, signal_bridge, signal_policy,
                           signal_source, signals_fired, decision, created_at)
                           VALUES (?, 0, 0, 0, 0, ?, ?)""",
                        (turn_id, decision, 5000 + event_idx),
                    )

        index_history(self.database, "h1", self.provider, gated=False)
        gated_expected = {
            ids[label] for label, _decisions, expected in histories if expected
        }
        with connection(self.database) as conn:
            self.assertEqual(
                {turn_id for turn_id in ids.values() if turn_is_eligible(conn, turn_id)},
                gated_expected,
            )

        gated_surfaces = (
            {item.ref_id for item in search_turns_fts(
                self.database, "h1", "eligibility marker"
            )},
            {item.ref_id for item in search_turns_dense(
                self.database, "h1", "eligibility marker", self.provider
            )},
            {item.turn_id for item in hybrid_rag(
                self.database, "h1", "eligibility marker", self.provider
            ).turns},
            {item.turn_id for item in recency(
                self.database, "h1", "eligibility marker", self.provider
            ).turns},
        )
        for label, _decisions, expected in histories:
            with self.subTest(history=label, eligible=expected):
                for surface in gated_surfaces:
                    self.assertEqual(ids[label] in surface, expected)

        ungated_surfaces = (
            {item.ref_id for item in search_turns_fts(
                self.database, "h1", "eligibility marker", gated=False
            )},
            {item.ref_id for item in search_turns_dense(
                self.database, "h1", "eligibility marker", self.provider, gated=False
            )},
            {item.turn_id for item in hybrid_rag(
                self.database, "h1", "eligibility marker", self.provider, gated=False
            ).turns},
            {item.turn_id for item in recency(
                self.database, "h1", "eligibility marker", self.provider, gated=False
            ).turns},
        )
        for surface in ungated_surfaces:
            self.assertEqual(surface, set(ids.values()))

    def test_pending_turn_is_ignored_by_gated_indexing_and_freshness(self) -> None:
        pending = self.add_turn(
            "pending freshness marker", turn_id="pending-freshness",
            session_id="pending-freshness", gated=False,
        )
        gated_summary = index_history(self.database, "h1", self.provider, gated=True)
        gated_report = check_index_freshness(
            self.database, "h1", self.provider, ref_types=("turn",), gated=True
        )
        self.assertEqual(gated_summary.indexed, 0)
        self.assertEqual(gated_report.required_count, 0)
        self.assertNotIn(pending, {issue.ref_id for issue in gated_report.issues})

        ungated_report = check_index_freshness(
            self.database, "h1", self.provider, ref_types=("turn",), gated=False
        )
        self.assertEqual(ungated_report.required_count, 1)
        self.assertEqual({issue.ref_id for issue in ungated_report.issues}, {pending})
        index_history(self.database, "h1", self.provider, gated=False)
        self.assertTrue(check_index_freshness(
            self.database, "h1", self.provider, ref_types=("turn",), gated=False
        ).is_fresh)
        self.assertEqual(
            check_index_freshness(
                self.database, "h1", self.provider, ref_types=("turn",), gated=True
            ).required_count,
            0,
        )

    def test_gate_event_helpers_distinguish_missing_decision_from_quarantine(self) -> None:
        helper = getattr(eligibility_module, "turn_has_gate_decision", None)
        quarantine_helper = getattr(eligibility_module, "turn_has_quarantine_event", None)
        self.assertTrue(callable(helper))
        self.assertTrue(callable(quarantine_helper))
        pending = self.add_turn(
            "pending gate helper", turn_id="gate-helper-pending",
            session_id="gate-helper-pending", gated=False,
        )
        stored = self.add_turn(
            "stored gate helper", turn_id="gate-helper-stored",
            session_id="gate-helper-stored", gated=False,
        )
        quarantined = self.add_turn(
            "quarantined gate helper", turn_id="gate-helper-quarantined",
            session_id="gate-helper-quarantined", gated=False,
        )
        stored_then_quarantined = self.add_turn(
            "store then quarantine helper", turn_id="gate-helper-store-quarantine",
            session_id="gate-helper-store-quarantine", gated=False,
        )
        with connection(self.database) as conn:
            record_store_event(conn, stored, created_at=1)
            for turn_id, decision in (
                (quarantined, "quarantine"),
                (stored_then_quarantined, "store"),
            ):
                conn.execute(
                    """INSERT INTO gate_events (turn_id, signal_bridge, signal_policy,
                       signal_source, signals_fired, decision, created_at)
                       VALUES (?, 0, 0, 0, 0, ?, 2)""",
                    (turn_id, decision),
                )
            conn.execute(
                """INSERT INTO gate_events (turn_id, signal_bridge, signal_policy,
                   signal_source, signals_fired, decision, created_at)
                   VALUES (?, 0, 0, 0, 0, 'quarantine', 3)""",
                (stored_then_quarantined,),
            )
            self.assertFalse(helper(conn, pending))
            self.assertFalse(quarantine_helper(conn, pending))
            self.assertTrue(helper(conn, stored))
            self.assertFalse(quarantine_helper(conn, stored))
            self.assertTrue(helper(conn, quarantined))
            self.assertTrue(quarantine_helper(conn, quarantined))
            self.assertTrue(helper(conn, stored_then_quarantined))
            self.assertTrue(quarantine_helper(conn, stored_then_quarantined))

    def test_pending_fact_and_parent_source_are_excluded_from_fact_surfaces(self) -> None:
        pending = self.add_turn(
            "pending fact marker", turn_id="pending-fact-turn",
            session_id="pending-fact-turn", gated=False,
        )
        trigger_source = self.add_turn(
            "stored trigger source", turn_id="stored-trigger-source",
            session_id="stored-trigger-source",
        )
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO facts (fact_id, history_id, turn_id, fact_seq, key_type,
                   entity, attribute, value, value_norm, cardinality, kind, key_text,
                   valid_from, last_confirmed_at, last_confirmed_turn_id,
                   confirmation_count, recorded_at, trust, quarantined)
                   VALUES ('pending-fact', 'h1', ?, 0, 'slot', 'user', 'city',
                   'Paris', 'paris', 'single', 'state', 'pending fact marker',
                   100, 100, ?, 1, 110, 0.7, 0)""",
                (pending, pending),
            )
            conn.execute(
                """INSERT INTO fact_observations (observation_id, history_id, turn_id,
                   fact_seq, entity, attribute, value, value_norm, cardinality, kind,
                   key_text, valid_at, effective_at, canonical_fact_id,
                   last_confirmed_at, last_confirmed_turn_id, confirmation_count,
                   recorded_at, trust, quarantined)
                   VALUES ('pending-fact', 'h1', ?, 0, 'user', 'city', 'Paris',
                   'paris', 'single', 'state', 'pending fact marker', 100, 100,
                   'pending-fact', 100, ?, 1, 110, 0.7, 0)""",
                (pending, pending),
            )
            conn.execute(
                """INSERT INTO facts (fact_id, history_id, turn_id, fact_seq, key_type,
                   key_text, parent_fact_id, valid_from, recorded_at)
                   VALUES ('pending-parent-trigger', 'h1', ?, 1, 'trigger',
                   'pending parent marker', 'pending-fact', 101, 111)""",
                (trigger_source,),
            )

        with connection(self.database) as conn:
            self.assertFalse(fact_is_eligible(conn, "pending-fact"))
            self.assertFalse(fact_is_eligible(conn, "pending-parent-trigger"))
        self.assertEqual(search_facts_fts(
            self.database, "h1", "pending marker", view="history"
        ), [])
        index_history(self.database, "h1", self.provider)
        self.assertEqual(search_facts_dense(
            self.database, "h1", "pending marker", self.provider, view="history"
        ), [])
        for view, as_of in (("current", None), ("as_of", 1000), ("history", None)):
            with self.subTest(view=view):
                results = retrieve_temporal_facts(
                    self.database, "h1", view=view, as_of=as_of
                )
                self.assertNotIn("pending-fact", {item.fact_id for item in results})
                self.assertNotIn("pending-parent-trigger", {item.fact_id for item in results})
        memory = retrieve_memory(self.database, "h1", "pending marker", self.provider)
        self.assertNotIn(pending, {item.turn_id for item in memory.turns})
        self.assertNotIn("pending-fact", {item.fact_id for item in memory.facts})

        with connection(self.database) as conn:
            record_store_event(conn, pending, created_at=120)
            self.assertTrue(fact_is_eligible(conn, "pending-fact"))
            self.assertTrue(fact_is_eligible(conn, "pending-parent-trigger"))
        index_history(self.database, "h1", self.provider)
        self.assertIn("pending-fact", {
            item.ref_id for item in search_facts_fts(
                self.database, "h1", "pending marker", view="history"
            )
        })

    def test_consolidated_pending_and_quarantine_exclusion_across_retrieval(self) -> None:
        cases = (
            ("clean", ("store",), 0),
            ("pending", (), 0),
            ("quarantined", ("quarantine",), 1),
            ("store-quarantined", ("store", "quarantine"), 1),
        )
        turn_ids: dict[str, str] = {}
        fact_ids: dict[str, str] = {}
        for index, (label, decisions, quarantined) in enumerate(cases):
            turn_id = self.add_turn(
                f"matrix marker turn {label}", turn_id=f"consolidated-{label}",
                session_id=f"consolidated-{label}", gated=False,
            )
            turn_ids[label] = turn_id
            fact_id = f"consolidated-fact-{label}"
            fact_ids[label] = fact_id
            with connection(self.database) as conn:
                for event_idx, decision in enumerate(decisions):
                    conn.execute(
                        """INSERT INTO gate_events (turn_id, signal_bridge, signal_policy,
                           signal_source, signals_fired, decision, created_at)
                           VALUES (?, 0, 0, 0, 0, ?, ?)""",
                        (turn_id, decision, 7000 + index * 10 + event_idx),
                    )
                event_time = float(100 + index)
                recorded_at = float(1000 + index)
                key_text = f"matrix marker fact {label}"
                conn.execute(
                    """INSERT INTO facts (fact_id, history_id, turn_id, fact_seq, key_type,
                       entity, attribute, value, value_norm, cardinality, kind, key_text,
                       valid_from, last_confirmed_at, last_confirmed_turn_id,
                       confirmation_count, recorded_at, trust, quarantined)
                       VALUES (?, 'h1', ?, 0, 'slot', 'user', ?, ?, ?, 'single',
                       'state', ?, ?, ?, ?, 1, ?, 0.7, ?)""",
                    (fact_id, turn_id, f"attribute_{label}", label, label.lower(),
                     key_text, event_time, recorded_at, turn_id, recorded_at, quarantined),
                )
                conn.execute(
                    """INSERT INTO fact_observations (observation_id, history_id, turn_id,
                       fact_seq, entity, attribute, value, value_norm, cardinality, kind,
                       key_text, valid_at, effective_at, canonical_fact_id,
                       last_confirmed_at, last_confirmed_turn_id, confirmation_count,
                       recorded_at, trust, quarantined)
                       VALUES (?, 'h1', ?, 0, 'user', ?, ?, ?, 'single', 'state',
                       ?, ?, ?, ?, ?, ?, 1, ?, 0.7, ?)""",
                    (fact_id, turn_id, f"attribute_{label}", label, label.lower(),
                     key_text, event_time, event_time, fact_id, recorded_at, turn_id,
                     recorded_at, quarantined),
                )

        index_history(self.database, "h1", self.provider, gated=False)
        clean_turn = {turn_ids["clean"]}
        clean_fact = {fact_ids["clean"]}
        turn_surfaces = (
            {item.ref_id for item in search_turns_fts(
                self.database, "h1", "matrix marker"
            )},
            {item.ref_id for item in search_turns_dense(
                self.database, "h1", "matrix marker", self.provider
            )},
            {item.turn_id for item in hybrid_rag(
                self.database, "h1", "matrix marker", self.provider
            ).turns},
            {item.turn_id for item in recency(
                self.database, "h1", "matrix marker", self.provider
            ).turns},
        )
        for result in turn_surfaces:
            self.assertEqual(result, clean_turn)
        fact_surfaces = (
            {item.ref_id for item in search_facts_fts(
                self.database, "h1", "matrix marker", view="history"
            )},
            {item.ref_id for item in search_facts_dense(
                self.database, "h1", "matrix marker", self.provider, view="history"
            )},
        )
        for result in fact_surfaces:
            self.assertEqual(result, clean_fact)
        for view, as_of in (("current", None), ("as_of", 500), ("history", None)):
            with self.subTest(view=view):
                rows = retrieve_temporal_facts(
                    self.database, "h1", view=view, as_of=as_of
                )
                self.assertEqual({item.fact_id for item in rows}, clean_fact)
        memory = retrieve_memory(self.database, "h1", "matrix marker", self.provider)
        self.assertEqual({item.turn_id for item in memory.turns}, clean_turn)
        self.assertEqual({item.fact_id for item in memory.facts}, clean_fact)
        self.assertEqual(
            {turn.turn_id for fact in memory.facts for turn in fact.supporting_turns},
            clean_turn,
        )

        all_turn_ids = set(turn_ids.values())
        ungated_turn_surfaces = (
            {item.ref_id for item in search_turns_fts(
                self.database, "h1", "matrix marker", gated=False
            )},
            {item.ref_id for item in search_turns_dense(
                self.database, "h1", "matrix marker", self.provider, gated=False
            )},
            {item.turn_id for item in hybrid_rag(
                self.database, "h1", "matrix marker", self.provider, gated=False
            ).turns},
            {item.turn_id for item in recency(
                self.database, "h1", "matrix marker", self.provider, gated=False
            ).turns},
        )
        for result in ungated_turn_surfaces:
            self.assertEqual(result, all_turn_ids)

    def test_pending_confirmation_turn_is_not_exposed_as_memory_evidence(self) -> None:
        source_turn = self.add_turn(
            "evidence marker source", turn_id="evidence-source",
            session_id="evidence-source",
        )
        pending_confirmation = self.add_turn(
            "evidence marker pending confirmation", turn_id="evidence-pending",
            session_id="evidence-pending", gated=False,
        )
        fact_id = self.add_fact(
            source_turn, "evidence-fact", "Paris", 100,
            key_text="evidence marker city Paris",
        )
        with connection(self.database) as conn:
            conn.execute(
                """UPDATE fact_observations SET last_confirmed_at=200,
                   last_confirmed_turn_id=? WHERE observation_id=?""",
                (pending_confirmation, fact_id),
            )
            conn.execute(
                """UPDATE facts SET last_confirmed_at=200,
                   last_confirmed_turn_id=? WHERE fact_id=?""",
                (pending_confirmation, fact_id),
            )
            self.assertTrue(fact_is_eligible(conn, fact_id))
        index_history(self.database, "h1", self.provider)
        result = retrieve_memory(
            self.database, "h1", "evidence marker", self.provider
        )
        self.assertIn(source_turn, {turn.turn_id for turn in result.turns})
        self.assertNotIn(pending_confirmation, {turn.turn_id for turn in result.turns})
        self.assertNotIn(fact_id, {fact.fact_id for fact in result.facts})
        self.assertNotIn(
            pending_confirmation,
            {turn.turn_id for fact in result.facts for turn in fact.supporting_turns},
        )

    def test_as_of_requires_timestamp_and_history_is_explicit(self) -> None:
        self.add_turn("some memory")
        index_history(self.database, "h1", self.provider)
        with self.assertRaises(ValueError):
            retrieve_memory(self.database, "h1", "memory", self.provider, view="as_of")
        with self.assertRaises(ValueError):
            retrieve_memory(self.database, "h1", "memory", self.provider, view="current", as_of=10)


if __name__ == "__main__":
    unittest.main()
