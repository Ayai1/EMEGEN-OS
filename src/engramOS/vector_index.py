"""History-scoped in-memory cosine search over persisted float16 vectors."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import sqlite3
from typing import Literal, Sequence

import numpy as np

from .db import connection
from .eligibility import fact_eligibility_predicate, turn_eligibility_predicate
from .embeddings import (
    BGE_M3_DIMENSION,
    EmbeddingValidationError,
    _check_index_freshness,
    content_digest,
    deserialize_vector,
    normalize_vector,
)

RefType = Literal["turn", "fact"]
logger = logging.getLogger(__name__)


class DenseIndexUnavailable(RuntimeError):
    """Raised when no compatible eligible embeddings are available."""


@dataclass(frozen=True, slots=True)
class DenseMatch:
    ref_type: RefType
    ref_id: str
    rank: int
    cosine_score: float
    record: dict[str, object]


class InMemoryVectorIndex:
    """One history and reference type, stored in a compact NumPy matrix."""

    def __init__(
        self,
        *,
        history_id: str,
        ref_type: RefType,
        model_id: str,
        vectors: np.ndarray,
        ref_ids: Sequence[str],
        records: Sequence[dict[str, object]],
        dimension: int = BGE_M3_DIMENSION,
    ) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape != (len(ref_ids), dimension):
            raise EmbeddingValidationError("in-memory vector matrix shape is invalid")
        if len(records) != len(ref_ids):
            raise ValueError("each vector must resolve to exactly one source record")
        if matrix.size and not np.isfinite(matrix).all():
            raise EmbeddingValidationError("in-memory vector matrix contains non-finite values")
        self.history_id = history_id
        self.ref_type = ref_type
        self.model_id = model_id
        self.dimension = dimension
        self.ref_ids = tuple(ref_ids)
        self.records = tuple(records)
        self._matrix = matrix
        self.device = "cpu"
        self._torch_matrix: object | None = None
        try:
            import torch
            if torch.cuda.is_available():
                try:
                    self._torch_matrix = torch.from_numpy(matrix).to("cuda")
                    self.device = "cuda"
                except RuntimeError:
                    logger.warning("Could not place vector index on CUDA; using CPU similarity")
        except ImportError:
            pass

    @classmethod
    def load(
        cls,
        database: str | Path,
        history_id: str,
        ref_type: RefType,
        model_id: str,
        *,
        dimension: int = BGE_M3_DIMENSION,
        timeout: float = 5.0,
        gated: bool = True,
    ) -> "InMemoryVectorIndex":
        if ref_type not in ("turn", "fact"):
            raise ValueError("ref_type must be 'turn' or 'fact'")
        if not gated and ref_type == "fact":
            raise ValueError("ungated indexing is available for turn sources only")
        if not history_id or not model_id:
            raise ValueError("history_id and model_id must be non-empty strings")
        with connection(database, timeout=timeout) as conn:
            conn.execute("BEGIN")
            report = _check_index_freshness(
                conn, history_id, model_id, (ref_type,), gated=gated
            )
            if not report.is_fresh:
                raise report.stale_error()
            return cls.load_from_connection(
                conn, history_id, ref_type, model_id,
                dimension=dimension, validate_freshness=False, gated=gated,
            )

    @classmethod
    def load_from_connection(
        cls,
        conn: sqlite3.Connection,
        history_id: str,
        ref_type: RefType,
        model_id: str,
        *,
        dimension: int = BGE_M3_DIMENSION,
        validate_freshness: bool = True,
        gated: bool = True,
    ) -> "InMemoryVectorIndex":
        """Build an index from a caller-owned SQLite snapshot."""
        if ref_type not in ("turn", "fact"):
            raise ValueError("ref_type must be 'turn' or 'fact'")
        if not gated and ref_type == "fact":
            raise ValueError("ungated indexing is available for turn sources only")
        if not history_id or not model_id:
            raise ValueError("history_id and model_id must be non-empty strings")
        if validate_freshness:
            report = _check_index_freshness(
                conn, history_id, model_id, (ref_type,), gated=gated
            )
            if not report.is_fresh:
                raise report.stale_error()
        if ref_type == "turn":
            rows = conn.execute(
                """SELECT e.ref_id, e.model, e.vec, e.content_hash,
                          t.turn_id, t.history_id, t.session_id, t.turn_idx,
                          t.role, t.raw_content, t.source_type, t.event_time,
                          t.recorded_at, t.token_count, t.base_trust, t.prev_hash, t.hash,
                          t.raw_content AS embedding_content
                   FROM embeddings AS e
                   JOIN turns AS t ON t.turn_id = e.ref_id
                   WHERE e.ref_type = 'turn' AND e.model = ? AND t.history_id = ?
                     AND """ + turn_eligibility_predicate("t", gated=gated) + " ORDER BY e.ref_id",
                (model_id, history_id),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT e.ref_id, e.model, e.vec, e.content_hash,
                          f.*, f.key_text AS embedding_content
                   FROM embeddings AS e
                   JOIN facts AS f ON f.fact_id = e.ref_id
                   WHERE e.ref_type = 'fact' AND e.model = ? AND f.history_id = ?
                     AND """ + fact_eligibility_predicate("f") + " ORDER BY e.ref_id",
                (model_id, history_id),
            ).fetchall()

        ref_ids: list[str] = []
        vectors: list[np.ndarray] = []
        records: list[dict[str, object]] = []
        for row in rows:
            if row["content_hash"] is None or row["content_hash"] != content_digest(
                str(row["embedding_content"])
            ):
                # Legacy or stale rows cannot be searched. index_history refreshes them.
                continue
            vector = deserialize_vector(
                row["vec"], dimension=dimension, model_id=model_id,
                stored_model_id=str(row["model"]),
            )
            ref_ids.append(str(row["ref_id"]))
            vectors.append(vector)
            records.append(
                {key: row[key] for key in row.keys() if key not in {"vec", "embedding_content"}}
            )
        matrix = np.vstack(vectors).astype(np.float32, copy=False) if vectors else np.empty(
            (0, dimension), dtype=np.float32
        )
        return cls(
            history_id=history_id,
            ref_type=ref_type,
            model_id=model_id,
            vectors=matrix,
            ref_ids=ref_ids,
            records=records,
            dimension=dimension,
        )

    @property
    def size(self) -> int:
        return len(self.ref_ids)

    def search(
        self,
        query_vector: Sequence[float] | np.ndarray,
        *,
        limit: int = 10,
        allowed_ids: set[str] | None = None,
    ) -> list[DenseMatch]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        query = normalize_vector(query_vector, dimension=self.dimension)
        if self.size == 0:
            return []
        indices = [
            index for index, ref_id in enumerate(self.ref_ids)
            if allowed_ids is None or ref_id in allowed_ids
        ]
        if not indices:
            return []
        if self._torch_matrix is not None:
            import torch
            try:
                query_tensor = torch.from_numpy(query).to("cuda")
                if allowed_ids is None:
                    scores = (self._torch_matrix @ query_tensor).detach().cpu().numpy()
                else:
                    selected = torch.as_tensor(indices, dtype=torch.long, device="cuda")
                    selected_matrix = torch.index_select(self._torch_matrix, 0, selected)
                    scores = (selected_matrix @ query_tensor).detach().cpu().numpy()
            except RuntimeError:
                logger.warning("CUDA similarity failed; retrying this query on CPU")
                self._torch_matrix = None
                self.device = "cpu"
                scores = self._matrix[indices] @ query
        else:
            selected_matrix = self._matrix if allowed_ids is None else self._matrix[indices]
            scores = selected_matrix @ query
        selected_order = sorted(
            range(len(indices)),
            key=lambda offset: (-float(scores[offset]), self.ref_ids[indices[offset]]),
        )
        matches: list[DenseMatch] = []
        for rank, offset in enumerate(selected_order[:limit], start=1):
            index = indices[offset]
            matches.append(
                DenseMatch(
                    self.ref_type,
                    self.ref_ids[index],
                    rank,
                    float(scores[offset]),
                    self.records[index],
                )
            )
        return matches


def require_dense_index(index: InMemoryVectorIndex) -> None:
    if index.size == 0:
        raise DenseIndexUnavailable(
            f"No current {index.ref_type} embeddings for history {index.history_id!r} "
            f"and model {index.model_id!r}; run index_history first"
        )
