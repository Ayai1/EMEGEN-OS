"""Embedding validation, indexing, and persistence integration tests."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
import types
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engramOS.db import connection, initialize_database  # noqa: E402
from engramOS.embeddings import (  # noqa: E402
    BGE_M3_DIMENSION,
    BGE_M3_MODEL_ID,
    BgeM3EmbeddingProvider,
    DenseIndexStaleError,
    EmbeddingModelUnavailable,
    EmbeddingValidationError,
    content_digest,
    check_index_freshness,
    deserialize_vector,
    index_history,
    normalize_vector,
    serialize_vector,
    _find_local_model_snapshot,
)
from engramOS.temporal import FactInput, insert_fact  # noqa: E402
from tests.helpers import record_store_event  # noqa: E402


class FakeEmbeddingProvider:
    model_id = "test-embedder-v1"
    dimension = BGE_M3_DIMENSION
    device = "cpu"

    def __init__(self, overrides: dict[str, np.ndarray] | None = None) -> None:
        self.overrides = overrides or {}
        self.document_calls: list[tuple[str, ...]] = []
        self.query_calls: list[str] = []

    def vector(self, text: str) -> np.ndarray:
        if text in self.overrides:
            return self.overrides[text].astype(np.float32)
        vector = np.zeros(self.dimension, dtype=np.float32)
        idx = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little") % self.dimension
        vector[idx] = 1.0
        return vector

    def encode_query(self, text: str) -> np.ndarray:
        self.query_calls.append(text)
        return self.vector(text)

    def encode_documents(self, texts: list[str] | tuple[str, ...]) -> np.ndarray:
        self.document_calls.append(tuple(texts))
        return np.vstack([self.vector(text) for text in texts]) if texts else np.empty((0, self.dimension))


class EmbeddingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = str(Path(self.temp_dir.name) / "embeddings.sqlite3")
        initialize_database(self.database)
        self.provider = FakeEmbeddingProvider()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def add_turn(
        self,
        turn_id: str,
        text: str,
        *,
        history_id: str = "h1",
        time: float = 10,
        gated: bool = True,
    ) -> str:
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO turns (turn_id, history_id, session_id, turn_idx, role,
                   raw_content, source_type, event_time, recorded_at, hash)
                   VALUES (?, ?, ?, 0, 'user', ?, 'user_turn', ?, ?, ?)""",
                (turn_id, history_id, f"session-{turn_id}", text, time, time + 100, f"hash-{turn_id}"),
            )
            if gated:
                record_store_event(conn, turn_id, created_at=time + 100)
        return turn_id

    def test_float16_serialization_round_trip_and_unit_normalization(self) -> None:
        vector = np.zeros(BGE_M3_DIMENSION, dtype=np.float32)
        vector[7] = 3.0
        vector[99] = 4.0
        blob = serialize_vector(vector)
        self.assertEqual(len(blob), BGE_M3_DIMENSION * 2)
        restored = deserialize_vector(blob, model_id="m", stored_model_id="m")
        self.assertEqual(restored.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(restored)), 1.0, places=5)
        self.assertAlmostEqual(float(restored[99]), 0.8, places=3)

    def test_invalid_dimensions_nonfinite_zero_and_blob_are_rejected(self) -> None:
        with self.assertRaises(EmbeddingValidationError):
            normalize_vector([1, 2], dimension=BGE_M3_DIMENSION)
        bad = np.zeros(BGE_M3_DIMENSION, dtype=np.float32)
        bad[0] = np.nan
        with self.assertRaises(EmbeddingValidationError):
            serialize_vector(bad)
        bad[0] = np.inf
        with self.assertRaises(EmbeddingValidationError):
            serialize_vector(bad)
        with self.assertRaises(EmbeddingValidationError):
            serialize_vector(np.zeros(BGE_M3_DIMENSION, dtype=np.float32))
        with self.assertRaises(EmbeddingValidationError):
            deserialize_vector(b"too short")

    def test_model_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(EmbeddingValidationError, "does not match"):
            deserialize_vector(
                serialize_vector(np.ones(BGE_M3_DIMENSION)),
                model_id="model-b", stored_model_id="model-a",
            )

    def test_indexing_is_idempotent_and_stale_content_is_replaced(self) -> None:
        turn_id = self.add_turn("t1", "first content")
        first = index_history(self.database, "h1", self.provider)
        self.assertEqual(first.indexed, 1)
        self.assertEqual(first.unchanged, 0)
        call_count = sum(len(call) for call in self.provider.document_calls)
        replay = index_history(self.database, "h1", self.provider)
        self.assertEqual(replay.indexed, 0)
        self.assertEqual(replay.unchanged, 1)
        self.assertEqual(call_count, sum(len(call) for call in self.provider.document_calls))
        with connection(self.database) as conn:
            old = conn.execute(
                "SELECT vec, content_hash FROM embeddings WHERE ref_type='turn' AND ref_id=?",
                (turn_id,),
            ).fetchone()
        # Source turns are immutable by design; exercise digest invalidation on a mutable fact below.
        self.assertEqual(old["content_hash"], content_digest("first content"))
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO facts (fact_id, history_id, turn_id, fact_seq, key_type,
                   entity, attribute, value, value_norm, cardinality, kind, key_text,
                   valid_from, recorded_at) VALUES ('f1','h1',?,0,'slot','user','city',
                   'Paris','paris','single','state','user.city: Paris',10,20)""",
                (turn_id,),
            )
        index_history(self.database, "h1", self.provider)
        with connection(self.database) as conn:
            before = conn.execute(
                "SELECT vec, content_hash FROM embeddings WHERE ref_type='fact' AND ref_id='f1'"
            ).fetchone()
            conn.execute("UPDATE facts SET key_text='user.city: Rome' WHERE fact_id='f1'")
        changed = index_history(self.database, "h1", self.provider)
        self.assertEqual(changed.indexed, 1)
        with connection(self.database) as conn:
            after = conn.execute(
                "SELECT vec, content_hash FROM embeddings WHERE ref_type='fact' AND ref_id='f1'"
            ).fetchone()
        self.assertNotEqual(before["content_hash"], after["content_hash"])
        self.assertNotEqual(before["vec"], after["vec"])

    def test_history_index_stores_turns_and_facts_as_float16_blobs(self) -> None:
        turn_id = self.add_turn("t1", "I live in Paris")
        insert_fact(
            self.database,
            FactInput("f1", "h1", turn_id, 0, "user", "city", "Paris", "single", 10, 20),
        )
        summary = index_history(self.database, "h1", self.provider, batch_size=1)
        self.assertEqual(summary.indexed, 2)
        with connection(self.database) as conn:
            rows = conn.execute("SELECT ref_type, vec, content_hash FROM embeddings ORDER BY ref_type").fetchall()
        self.assertEqual({row["ref_type"] for row in rows}, {"turn", "fact"})
        for row in rows:
            self.assertEqual(len(row["vec"]), BGE_M3_DIMENSION * 2)
            self.assertTrue(row["content_hash"])
            self.assertEqual(len(deserialize_vector(row["vec"])), BGE_M3_DIMENSION)

    def test_schema_initialization_adds_digest_to_legacy_embedding_table(self) -> None:
        legacy_db = str(Path(self.temp_dir.name) / "legacy.sqlite3")
        conn = sqlite3.connect(legacy_db)
        conn.execute(
            """CREATE TABLE embeddings (
               ref_type TEXT NOT NULL, ref_id TEXT NOT NULL, model TEXT NOT NULL,
               vec BLOB NOT NULL, PRIMARY KEY(ref_type, ref_id, model))"""
        )
        conn.execute(
            "INSERT INTO embeddings(ref_type, ref_id, model, vec) VALUES ('turn','legacy','old-model',x'0001')"
        )
        conn.commit()
        conn.close()
        initialize_database(legacy_db)
        with connection(legacy_db) as conn:
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(embeddings)")}
            row = conn.execute(
                "SELECT vec, content_hash FROM embeddings WHERE ref_id='legacy'"
            ).fetchone()
        self.assertIn("content_hash", cols)
        self.assertEqual(row["vec"], b"\x00\x01")
        self.assertIsNone(row["content_hash"])

    def test_bge_adapter_is_lazy_and_refuses_network_downloads(self) -> None:
        adapter = BgeM3EmbeddingProvider(
            model_path=Path(self.temp_dir.name) / "missing-model"
        )
        self.assertEqual(adapter.dimension, BGE_M3_DIMENSION)
        with self.assertRaises(EmbeddingModelUnavailable):
            _ = adapter.model_id
        with self.assertRaises(EmbeddingModelUnavailable):
            adapter.encode_query("test")

    def test_hugging_face_identity_uses_ref_metadata_not_snapshot_sort_order(self) -> None:
        cache = Path(self.temp_dir.name) / "hub"
        repository = cache / "models--BAAI--bge-m3"
        snapshots = repository / "snapshots"
        for commit in ("aaa1111", "fff9999"):
            (snapshots / commit).mkdir(parents=True)
            (snapshots / commit / "config.json").write_text(commit, encoding="utf-8")
        refs = repository / "refs"
        refs.mkdir()
        (refs / "main").write_text("aaa1111\n", encoding="utf-8")
        with mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(cache)}):
            resolved = _find_local_model_snapshot("main")
            self.assertEqual(resolved, (snapshots / "aaa1111", "aaa1111"))
            first = BgeM3EmbeddingProvider(revision="main")
            second = BgeM3EmbeddingProvider(revision="main")
            other = BgeM3EmbeddingProvider(revision="fff9999")
            self.assertEqual(first.model_id, second.model_id)
            self.assertEqual(first.model_id, f"{BGE_M3_MODEL_ID}@aaa1111")
            self.assertNotEqual(first.model_id, other.model_id)
            turn_id = self.add_turn("revision-row", "same text")
            with connection(self.database) as conn:
                for model_id in (first.model_id, other.model_id):
                    conn.execute(
                        """INSERT INTO embeddings(ref_type,ref_id,model,vec,content_hash)
                           VALUES ('turn',?,?,?,?)""",
                        (turn_id, model_id, serialize_vector(np.ones(BGE_M3_DIMENSION)), content_digest("same text")),
                    )
                persisted = conn.execute(
                    "SELECT model FROM embeddings WHERE ref_type='turn' AND ref_id=? ORDER BY model",
                    (turn_id,),
                ).fetchall()
            self.assertEqual({row["model"] for row in persisted}, {first.model_id, other.model_id})

    def test_ambiguous_or_missing_cache_revision_fails_clearly(self) -> None:
        cache = Path(self.temp_dir.name) / "hub"
        snapshots = cache / "models--BAAI--bge-m3" / "snapshots"
        (snapshots / "aaa1111").mkdir(parents=True)
        (snapshots / "zzz9999").mkdir()
        with mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(cache)}):
            self.assertIsNone(_find_local_model_snapshot("main"))
            adapter = BgeM3EmbeddingProvider(revision="main")
            with self.assertRaisesRegex(EmbeddingModelUnavailable, "revision 'main'"):
                _ = adapter.model_id

    def test_local_checkpoint_fingerprint_is_path_independent_and_content_sensitive(self) -> None:
        left = Path(self.temp_dir.name) / "left"
        right = Path(self.temp_dir.name) / "right"
        for directory in (left, right):
            directory.mkdir()
            (directory / "config.json").write_text('{"hidden_size":1024}', encoding="utf-8")
            (directory / "weights.bin").write_bytes(b"checkpoint bytes")
        first = BgeM3EmbeddingProvider(left)
        second = BgeM3EmbeddingProvider(right)
        self.assertEqual(first.model_id, second.model_id)
        (right / "weights.bin").write_bytes(b"different checkpoint")
        third = BgeM3EmbeddingProvider(right)
        self.assertNotEqual(first.model_id, third.model_id)

    def test_query_and_document_encodings_use_one_local_checkpoint_identity(self) -> None:
        checkpoint = Path(self.temp_dir.name) / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("local test checkpoint", encoding="utf-8")

        class LocalModel:
            def get_sentence_embedding_dimension(self) -> int:
                return BGE_M3_DIMENSION

            def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
                return np.ones((len(texts), BGE_M3_DIMENSION), dtype=np.float32)

        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = lambda *args, **kwargs: LocalModel()
        with mock.patch.dict(sys.modules, {"sentence_transformers": module}):
            provider = BgeM3EmbeddingProvider(checkpoint, device="cpu")
            identity = provider.model_id
            provider.encode_query("query")
            provider.encode_documents(["document"])
            self.assertEqual(provider.model_id, identity)

    def test_freshness_detects_new_sources_changed_text_and_legacy_hash_without_writing(self) -> None:
        turn_id = self.add_turn("fresh-turn", "turn content")
        insert_fact(
            self.database,
            FactInput("fresh-fact", "h1", turn_id, 0, "user", "city", "Paris", "single", 10, 20),
        )
        index_history(self.database, "h1", self.provider)
        provider_calls = len(self.provider.document_calls)
        self.assertTrue(check_index_freshness(self.database, "h1", self.provider).is_fresh)

        self.add_turn("late-turn", "new turn", time=30)
        report = check_index_freshness(self.database, "h1", self.provider)
        self.assertIn(("turn", "late-turn", "missing_embedding"), {
            (item.ref_type, item.ref_id, item.reason) for item in report.issues
        })
        with connection(self.database) as conn:
            conn.execute("UPDATE facts SET key_text='user.city: Rome' WHERE fact_id='fresh-fact'")
        report = check_index_freshness(self.database, "h1", self.provider)
        self.assertIn(("fact", "fresh-fact", "content_changed"), {
            (item.ref_type, item.ref_id, item.reason) for item in report.issues
        })
        with connection(self.database) as conn:
            conn.execute(
                "UPDATE embeddings SET content_hash=NULL WHERE ref_type='fact' AND ref_id='fresh-fact'"
            )
        report = check_index_freshness(self.database, "h1", self.provider)
        self.assertIn(("fact", "fresh-fact", "missing_content_digest"), {
            (item.ref_type, item.ref_id, item.reason) for item in report.issues
        })
        self.assertEqual(len(self.provider.document_calls), provider_calls)

    def test_freshness_ignores_quarantine_and_is_history_scoped(self) -> None:
        self.add_turn("h1-live", "live", history_id="h1")
        self.add_turn("h1-poison", "poison", history_id="h1")
        index_history(self.database, "h1", self.provider)
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO gate_events (turn_id, signal_bridge, signal_policy,
                   signal_source, signals_fired, decision, created_at)
                   VALUES ('h1-poison',1,1,0,2,'quarantine',40)"""
            )
        self.assertTrue(check_index_freshness(self.database, "h1", self.provider).is_fresh)
        self.add_turn("h2-new", "other history", history_id="h2")
        self.assertTrue(check_index_freshness(self.database, "h1", self.provider).is_fresh)
        report = check_index_freshness(self.database, "h2", self.provider)
        self.assertEqual([(issue.ref_type, issue.ref_id) for issue in report.issues], [("turn", "h2-new")])

    def test_new_fact_is_stale_but_quarantined_fact_does_not_count(self) -> None:
        turn_id = self.add_turn("fact source", "fact source")
        index_history(self.database, "h1", self.provider)
        insert_fact(
            self.database,
            FactInput("new-fact", "h1", turn_id, 0, "user", "city", "Paris", "single", 20, 30),
        )
        report = check_index_freshness(self.database, "h1", self.provider)
        self.assertIn(("fact", "new-fact", "missing_embedding"), {
            (item.ref_type, item.ref_id, item.reason) for item in report.issues
        })
        with connection(self.database) as conn:
            conn.execute("UPDATE facts SET quarantined=1 WHERE fact_id='new-fact'")
        self.assertTrue(check_index_freshness(self.database, "h1", self.provider).is_fresh)

    def test_new_eligible_trigger_fact_requires_index_refresh(self) -> None:
        turn_id = self.add_turn("trigger source", "trigger source")
        parent_id = insert_fact(
            self.database,
            FactInput("trigger-parent", "h1", turn_id, 0, "user", "seat", "aisle", "single", 10, 20),
        )
        index_history(self.database, "h1", self.provider)
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO facts (fact_id, history_id, turn_id, fact_seq, key_type,
                   key_text, parent_fact_id, valid_from, recorded_at)
                   VALUES ('trigger-child','h1',?,1,'trigger','Find a nearby seat',?,10,30)""",
                (turn_id, parent_id),
            )
        report = check_index_freshness(self.database, "h1", self.provider)
        self.assertIn(("fact", "trigger-child", "missing_embedding"), {
            (item.ref_type, item.ref_id, item.reason) for item in report.issues
        })
        index_history(self.database, "h1", self.provider)
        self.assertTrue(check_index_freshness(self.database, "h1", self.provider).is_fresh)

    def test_freshness_reports_wrong_model_identity(self) -> None:
        self.add_turn("model-turn", "model source")
        index_history(self.database, "h1", self.provider)

        class OtherProvider(FakeEmbeddingProvider):
            model_id = "other-revision"

        report = check_index_freshness(self.database, "h1", OtherProvider())
        self.assertEqual(report.issues[0].reason, "model_identity_mismatch")
        self.assertEqual(report.issues[0].stored_models, (self.provider.model_id,))

    def test_freshness_reports_reference_type_mismatch(self) -> None:
        turn_id = self.add_turn("typed-reference", "typed source")
        with connection(self.database) as conn:
            conn.execute(
                "INSERT INTO embeddings(ref_type,ref_id,model,vec,content_hash) VALUES ('fact',?,?,?,?)",
                (turn_id, self.provider.model_id, serialize_vector(np.ones(BGE_M3_DIMENSION)), content_digest("typed source")),
            )
        report = check_index_freshness(
            self.database, "h1", self.provider, ref_types=("turn",)
        )
        self.assertEqual(report.issues[0].reason, "reference_type_mismatch")

    def test_legacy_generic_bge_vector_does_not_match_resolved_revision(self) -> None:
        turn_id = self.add_turn("legacy model source", "legacy content")
        with connection(self.database) as conn:
            conn.execute(
                "INSERT INTO embeddings(ref_type,ref_id,model,vec,content_hash) VALUES ('turn',?,?,?,?)",
                (turn_id, BGE_M3_MODEL_ID, serialize_vector(np.ones(BGE_M3_DIMENSION)), content_digest("legacy content")),
            )
        checkpoint = Path(self.temp_dir.name) / "local-bge"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("revision checkpoint", encoding="utf-8")
        provider = BgeM3EmbeddingProvider(checkpoint)
        report = check_index_freshness(self.database, "h1", provider)
        self.assertEqual(report.issues[0].reason, "model_identity_mismatch")
        self.assertEqual(report.issues[0].stored_models, (BGE_M3_MODEL_ID,))

    def test_refresh_after_stale_index_is_idempotent(self) -> None:
        self.add_turn("first", "first source")
        index_history(self.database, "h1", self.provider)
        self.add_turn("second", "second source", time=20)
        report = check_index_freshness(self.database, "h1", self.provider)
        self.assertFalse(report.is_fresh)
        self.assertIsInstance(report.stale_error(), DenseIndexStaleError)
        refreshed = index_history(self.database, "h1", self.provider)
        self.assertEqual(refreshed.indexed, 1)
        self.assertTrue(check_index_freshness(self.database, "h1", self.provider).is_fresh)
        replay = index_history(self.database, "h1", self.provider)
        self.assertEqual((replay.indexed, replay.unchanged), (0, 2))


if __name__ == "__main__":
    unittest.main()
