"""SQLite-backed dense index integration tests."""

from __future__ import annotations

import hashlib
import sys
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
    DenseIndexStaleError,
    EmbeddingValidationError,
    content_digest,
    index_history,
)
from engramOS.vector_index import InMemoryVectorIndex  # noqa: E402
from tests.helpers import record_store_event  # noqa: E402


class AxisProvider:
    model_id = "axis-provider"
    dimension = BGE_M3_DIMENSION
    device = "cpu"

    def __init__(self, axes: dict[str, int]) -> None:
        self.axes = axes

    def _vector(self, text: str) -> np.ndarray:
        result = np.zeros(self.dimension, dtype=np.float32)
        result[self.axes.get(text, int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little") % self.dimension)] = 1
        return result

    def encode_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def encode_documents(self, texts: list[str] | tuple[str, ...]) -> np.ndarray:
        return np.vstack([self._vector(text) for text in texts])


class VectorIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = str(Path(self.temp_dir.name) / "vectors.sqlite3")
        initialize_database(self.database)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def add_turn(
        self, turn_id: str, history: str, content: str, when: float, *, gated: bool = True
    ) -> None:
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO turns (turn_id, history_id, session_id, turn_idx, role,
                   raw_content, source_type, event_time, recorded_at, hash)
                   VALUES (?, ?, ?, 0, 'user', ?, 'user_turn', ?, ?, ?)""",
                (turn_id, history, f"session-{turn_id}", content, when, when + 10, f"hash-{turn_id}"),
            )
            if gated:
                record_store_event(conn, turn_id, created_at=when + 10)

    def test_cosine_order_and_deterministic_ties(self) -> None:
        self.add_turn("z-turn", "h1", "tie z", 10)
        self.add_turn("a-turn", "h1", "tie a", 11)
        self.add_turn("b-turn", "h1", "other", 12)
        provider = AxisProvider({"tie z": 1, "tie a": 1, "other": 2, "query": 1})
        index_history(self.database, "h1", provider)
        index = InMemoryVectorIndex.load(
            self.database, "h1", "turn", provider.model_id
        )
        hits = index.search(provider.encode_query("query"), limit=3)
        self.assertEqual([hit.ref_id for hit in hits], ["a-turn", "z-turn", "b-turn"])
        self.assertEqual([hit.rank for hit in hits], [1, 2, 3])
        self.assertGreater(hits[0].cosine_score, hits[2].cosine_score)

    def test_allowed_ids_are_filtered_before_scoring_and_keep_stable_ranks(self) -> None:
        self.add_turn("z-turn", "h1", "tie z", 10)
        self.add_turn("a-turn", "h1", "tie a", 11)
        self.add_turn("b-turn", "h1", "other", 12)
        provider = AxisProvider({"tie z": 1, "tie a": 1, "other": 2, "query": 1})
        index_history(self.database, "h1", provider)
        index = InMemoryVectorIndex.load(self.database, "h1", "turn", provider.model_id)
        query = provider.encode_query("query")
        full = index.search(query, limit=3)
        filtered = index.search(query, limit=3, allowed_ids={"z-turn", "b-turn"})
        self.assertEqual(
            [item.ref_id for item in filtered],
            [item.ref_id for item in full if item.ref_id in {"z-turn", "b-turn"}],
        )
        self.assertEqual([item.rank for item in filtered], [1, 2])
        self.assertEqual(index.search(query, allowed_ids=set()), [])
        self.assertEqual(index.search(query, allowed_ids={"unknown"}), [])

        scored_rows: list[int] = []

        class MatrixProbe:
            def __init__(self, values: np.ndarray) -> None:
                self.values = values

            def __getitem__(self, rows) -> "MatrixProbe":
                return MatrixProbe(self.values[rows])

            def __matmul__(self, vector: np.ndarray) -> np.ndarray:
                scored_rows.append(self.values.shape[0])
                return self.values @ vector

        index._matrix = MatrixProbe(index._matrix)  # type: ignore[assignment]
        index._torch_matrix = None
        index.search(query, allowed_ids={"a-turn"})
        self.assertEqual(scored_rows, [1])

    def test_cpu_path_and_cuda_path_when_available(self) -> None:
        self.add_turn("cpu-a", "h1", "same vector", 10)
        self.add_turn("cpu-b", "h1", "other vector", 20)
        provider = AxisProvider({"same vector": 1, "query": 1, "other vector": 2})
        index_history(self.database, "h1", provider)
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None:
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                cpu_index = InMemoryVectorIndex.load(
                    self.database, "h1", "turn", provider.model_id
                )
                self.assertEqual(cpu_index.device, "cpu")
                cpu_hits = cpu_index.search(provider.encode_query("query"))
            if torch.cuda.is_available():
                cuda_index = InMemoryVectorIndex.load(
                    self.database, "h1", "turn", provider.model_id
                )
                self.assertEqual(cuda_index.device, "cuda")
                cuda_hits = cuda_index.search(provider.encode_query("query"))
                self.assertEqual(
                    [(item.ref_id, item.rank) for item in cuda_hits],
                    [(item.ref_id, item.rank) for item in cpu_hits],
                )
        else:
            cpu_index = InMemoryVectorIndex.load(
                self.database, "h1", "turn", provider.model_id
            )
            self.assertEqual(cpu_index.device, "cpu")
            cpu_hits = cpu_index.search(provider.encode_query("query"))
        self.assertEqual(cpu_hits[0].ref_id, "cpu-a")

    def test_rebuild_ignores_other_histories_and_orphans(self) -> None:
        self.add_turn("h1-turn", "h1", "history one", 10)
        self.add_turn("h2-turn", "h2", "history two", 20)
        provider = AxisProvider({})
        index_history(self.database, "h1", provider)
        index = InMemoryVectorIndex.load(self.database, "h1", "turn", provider.model_id)
        self.assertEqual(index.ref_ids, ("h1-turn",))
        index_history(self.database, "h2", provider)
        other = InMemoryVectorIndex.load(self.database, "h2", "turn", provider.model_id)
        self.assertEqual(other.ref_ids, ("h2-turn",))
        with connection(self.database) as conn:
            conn.execute(
                "INSERT INTO embeddings(ref_type, ref_id, model, vec, content_hash) "
                "SELECT 'turn', 'deleted-turn', model, vec, content_hash FROM embeddings LIMIT 1"
            )
        rebuilt = InMemoryVectorIndex.load(self.database, "h1", "turn", provider.model_id)
        self.assertEqual(rebuilt.ref_ids, ("h1-turn",))

    def test_quarantined_turn_is_removed_from_rebuilt_dense_index(self) -> None:
        self.add_turn("clean", "h1", "same content", 10)
        self.add_turn("poison", "h1", "same content", 20)
        provider = AxisProvider({})
        index_history(self.database, "h1", provider)
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO gate_events (turn_id, signal_bridge, signal_policy,
                   signal_source, signals_fired, decision, created_at)
                   VALUES ('poison',1,1,0,2,'quarantine',30)"""
            )
        index = InMemoryVectorIndex.load(self.database, "h1", "turn", provider.model_id)
        self.assertEqual(index.ref_ids, ("clean",))
        self.assertEqual([hit.ref_id for hit in index.search(provider.encode_query("same content"))], ["clean"])

    def test_corrupt_matching_model_vector_fails_safely(self) -> None:
        self.add_turn("broken", "h1", "corrupt vector source", 10)
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO embeddings (ref_type, ref_id, model, vec, content_hash)
                   VALUES ('turn','broken','axis-provider',x'0001',?)""",
                (content_digest("corrupt vector source"),),
            )
        with self.assertRaises(EmbeddingValidationError):
            InMemoryVectorIndex.load(
                self.database, "h1", "turn", "axis-provider"
            )

    def test_other_model_embeddings_fail_freshness_check(self) -> None:
        self.add_turn("only-old-model", "h1", "old model vector", 10)
        index_history(self.database, "h1", AxisProvider({}))
        with self.assertRaises(DenseIndexStaleError):
            InMemoryVectorIndex.load(
                self.database, "h1", "turn", "different-model"
            )

    def test_direct_index_load_fails_when_an_eligible_record_is_missing(self) -> None:
        self.add_turn("not-indexed", "h1", "current source", 10)
        with self.assertRaisesRegex(DenseIndexStaleError, "missing_embedding"):
            InMemoryVectorIndex.load(
                self.database, "h1", "turn", "axis-provider"
            )


if __name__ == "__main__":
    unittest.main()
