"""Embedding providers, validation, persistence, and idempotent indexing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import re
import sqlite3
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from .db import connection, transaction
from .eligibility import fact_eligibility_predicate, turn_eligibility_predicate

logger = logging.getLogger(__name__)

BGE_M3_MODEL_ID = "BAAI/bge-m3"
BGE_M3_DIMENSION = 1024


class EmbeddingError(RuntimeError):
    """Base class for embedding configuration and data errors."""


class EmbeddingModelUnavailable(EmbeddingError):
    """Raised when the requested local embedding model is not available."""


class EmbeddingValidationError(EmbeddingError, ValueError):
    """Raised when an embedding is malformed or incompatible."""


class EmbeddingSourceChanged(EmbeddingError):
    """Raised when source text changes while a batch is being encoded."""


class DenseIndexStaleError(EmbeddingError):
    """Raised when one or more eligible sources lack a current dense vector."""


@dataclass(frozen=True, slots=True)
class IndexFreshnessIssue:
    ref_type: str
    ref_id: str
    reason: str
    stored_models: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IndexFreshnessReport:
    history_id: str
    model_id: str
    required_count: int
    fresh_count: int
    issues: tuple[IndexFreshnessIssue, ...]

    @property
    def is_fresh(self) -> bool:
        return not self.issues

    def stale_error(self) -> DenseIndexStaleError:
        examples = ", ".join(
            f"{issue.ref_type}:{issue.ref_id} ({issue.reason})"
            for issue in self.issues[:5]
        )
        more = "" if len(self.issues) <= 5 else f"; and {len(self.issues) - 5} more"
        return DenseIndexStaleError(
            f"Dense index for history {self.history_id!r} and model {self.model_id!r} "
            f"is stale ({len(self.issues)} of {self.required_count} eligible records): "
            f"{examples}{more}. Please run index_history(database, history_id, provider) "
            "and retry retrieval."
        )


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Provider-independent dense embedding interface."""

    model_id: str
    dimension: int
    device: str

    def encode_query(self, text: str) -> Sequence[float] | np.ndarray:
        """Encode one search query."""

    def encode_documents(
        self, texts: Sequence[str]
    ) -> Sequence[Sequence[float]] | np.ndarray:
        """Encode one or more source documents."""


class BgeM3EmbeddingProvider:
    """Lazy local BGE-M3 adapter using Sentence Transformers.

    It resolves only an existing local model directory or Hugging Face cache
    snapshot. Constructing this provider and importing EngramOS never loads
    model weights or downloads them.
    """

    dimension = BGE_M3_DIMENSION

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        revision: str = "main",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("revision must be a non-empty Hugging Face revision")
        if model_path is not None and revision != "main":
            raise ValueError("revision applies only to the Hugging Face cache, not model_path")
        self._model_path = Path(model_path).expanduser() if model_path is not None else None
        self._revision = revision
        self._requested_device = device
        self._batch_size = batch_size
        self._model: object | None = None
        self._resolved_model_path: Path | None = None
        self._model_id: str | None = None
        self.device = device or "uninitialized"

    @property
    def model_id(self) -> str:
        """Stable identity for the exact local checkpoint used by this provider."""
        self._resolve_checkpoint()
        assert self._model_id is not None
        return self._model_id

    def encode_query(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("query text must be a non-empty string")
        return self.encode_documents([text])[0]

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("document texts must be non-empty strings")
        model = self._load_model()
        try:
            result = model.encode(  # type: ignore[attr-defined]
                list(texts),
                batch_size=self._batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError("BGE-M3 document encoding failed") from exc
        vectors = np.asarray(result)
        if vectors.ndim != 2 or vectors.shape != (len(texts), self.dimension):
            raise EmbeddingValidationError(
                f"BGE-M3 returned shape {vectors.shape}, expected ({len(texts)}, {self.dimension})"
            )
        return vectors

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model
        model_path, _ = self._resolve_checkpoint()
        if not model_path.is_dir():
            raise EmbeddingModelUnavailable(
                "BGE-M3 weights are not present locally; pass an existing model directory. "
                "Automatic model downloads are disabled."
            )
        if self._model_path is not None:
            expected_fingerprint = (self._model_id or "").rsplit("local-sha256:", 1)[-1]
            if _checkpoint_fingerprint(model_path) != expected_fingerprint:
                raise EmbeddingModelUnavailable(
                    "Local BGE-M3 checkpoint contents changed after provider identity "
                    "was resolved; create a new provider to use the updated checkpoint."
                )
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise EmbeddingModelUnavailable(
                "BGE-M3 needs the optional sentence-transformers package; install the project's "
                "optional ML dependencies after reviewing their size."
            ) from exc
        selected_device = self._requested_device or _available_device()
        try:
            # A local filesystem path avoids any hub/network lookup.
            model = SentenceTransformer(
                str(model_path), device=selected_device, local_files_only=True
            )
            model_dimension = model.get_sentence_embedding_dimension()
        except Exception as exc:
            if self._requested_device is None and selected_device == "cuda":
                logger.warning("BGE-M3 CUDA load failed; retrying the local model on CPU")
                selected_device = "cpu"
                try:
                    model = SentenceTransformer(
                        str(model_path), device=selected_device, local_files_only=True
                    )
                    model_dimension = model.get_sentence_embedding_dimension()
                except Exception as cpu_exc:
                    raise EmbeddingModelUnavailable(
                        f"Could not load local BGE-M3 model at {model_path} on CUDA or CPU"
                    ) from cpu_exc
            else:
                raise EmbeddingModelUnavailable(
                    f"Could not load local BGE-M3 model at {model_path}"
                ) from exc
        if model_dimension != self.dimension:
            raise EmbeddingValidationError(
                f"Local BGE-M3 reports {model_dimension} dimensions; expected {self.dimension}"
            )
        self.device = selected_device
        self._model = model
        return model

    def _resolve_checkpoint(self) -> tuple[Path, str]:
        if self._resolved_model_path is not None and self._model_id is not None:
            return self._resolved_model_path, self._model_id
        if self._model_path is not None:
            model_path = self._model_path
            if not model_path.is_dir():
                raise EmbeddingModelUnavailable(
                    f"Local BGE-M3 checkpoint directory does not exist: {model_path}"
                )
            fingerprint = _checkpoint_fingerprint(model_path)
            model_id = f"{BGE_M3_MODEL_ID}@local-sha256:{fingerprint}"
        else:
            resolved = _find_local_model_snapshot(self._revision)
            if resolved is None:
                raise EmbeddingModelUnavailable(
                    f"BGE-M3 revision {self._revision!r} is not present in the local "
                    "Hugging Face cache. Provide an existing model_path or cache that "
                    "revision explicitly; automatic downloads are disabled."
                )
            model_path, commit = resolved
            model_id = f"{BGE_M3_MODEL_ID}@{commit}"
        self._resolved_model_path = model_path
        self._model_id = model_id
        return model_path, model_id


def _available_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _find_local_model_snapshot(revision: str = "main") -> tuple[Path, str] | None:
    """Resolve a cache revision through Hugging Face's ``refs`` metadata.

    A direct commit hash is also accepted when its snapshot directory exists.
    The resolver never guesses between snapshots or sorts directory names.
    """
    if (
        not isinstance(revision, str)
        or not revision
        or Path(revision).is_absolute()
        or ".." in Path(revision).parts
    ):
        raise ValueError("revision must be a safe non-empty cache revision")
    cache_root = Path(
        os.environ.get("HF_HUB_CACHE")
        or (Path(os.environ["HF_HOME"]) / "hub" if os.environ.get("HF_HOME") else "")
        or (Path.home() / ".cache" / "huggingface" / "hub")
    )
    repository = cache_root / "models--BAAI--bge-m3"
    snapshots = repository / "snapshots"
    if not snapshots.is_dir():
        return None
    reference_path = repository / "refs" / revision
    if reference_path.is_file():
        commit = reference_path.read_text(encoding="utf-8").strip()
    elif re.fullmatch(r"[0-9a-fA-F]{7,64}", revision):
        commit = revision
    else:
        return None
    if not re.fullmatch(r"[0-9a-fA-F]{7,64}", commit):
        raise EmbeddingModelUnavailable(
            f"Hugging Face cache ref {revision!r} does not contain a valid commit identity"
        )
    snapshot = snapshots / commit
    if not snapshot.is_dir():
        return None
    return snapshot, commit


def _checkpoint_fingerprint(model_path: Path) -> str:
    """Hash checkpoint file names and bytes, independent of the absolute path."""
    digest = hashlib.sha256()
    files = sorted(
        path for path in model_path.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(model_path).parts
    )
    if not files:
        raise EmbeddingModelUnavailable(
            f"Local BGE-M3 checkpoint directory is empty: {model_path}"
        )
    for path in files:
        relative = path.relative_to(model_path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def content_digest(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("embedding source text must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_vector(
    vector: Sequence[float] | np.ndarray,
    *,
    dimension: int = BGE_M3_DIMENSION,
) -> np.ndarray:
    """Validate and return a unit-normalized float32 dense vector."""
    try:
        values = np.asarray(vector, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise EmbeddingValidationError("vector must contain numeric values") from exc
    if values.ndim != 1 or values.shape[0] != dimension:
        raise EmbeddingValidationError(
            f"vector must have exactly {dimension} dimensions"
        )
    if not np.isfinite(values).all():
        raise EmbeddingValidationError("vector values must all be finite")
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm == 0.0:
        raise EmbeddingValidationError("vector norm must be finite and non-zero")
    return values / norm


def serialize_vector(
    vector: Sequence[float] | np.ndarray,
    *,
    dimension: int = BGE_M3_DIMENSION,
) -> bytes:
    """Normalize and serialize a vector as a little-endian float16 BLOB."""
    normalized = normalize_vector(vector, dimension=dimension)
    packed = normalized.astype("<f2", copy=False)
    if not np.isfinite(packed).all() or not np.any(packed):
        raise EmbeddingValidationError("vector cannot be represented as finite float16")
    return packed.tobytes(order="C")


def deserialize_vector(
    blob: bytes | bytearray | memoryview,
    *,
    dimension: int = BGE_M3_DIMENSION,
    model_id: str | None = None,
    stored_model_id: str | None = None,
) -> np.ndarray:
    """Validate and normalize a persisted float16 embedding BLOB."""
    if model_id is not None and stored_model_id != model_id:
        raise EmbeddingValidationError(
            f"embedding model {stored_model_id!r} does not match requested model {model_id!r}"
        )
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise EmbeddingValidationError("stored vector must be a BLOB")
    expected_bytes = dimension * np.dtype("<f2").itemsize
    if len(blob) != expected_bytes:
        raise EmbeddingValidationError(
            f"stored vector BLOB has {len(blob)} bytes, expected {expected_bytes}"
        )
    values = np.frombuffer(blob, dtype="<f2")
    if values.shape != (dimension,) or not np.isfinite(values).all():
        raise EmbeddingValidationError("stored float16 vector is invalid")
    return normalize_vector(values, dimension=dimension)


@dataclass(frozen=True, slots=True)
class IndexingSummary:
    history_id: str
    model_id: str
    indexed: int
    unchanged: int
    ineligible_during_write: int = 0


def index_history(
    database: str | Path,
    history_id: str,
    provider: EmbeddingProvider,
    *,
    batch_size: int = 32,
    timeout: float = 5.0,
    gated: bool = True,
) -> IndexingSummary:
    """Idempotently index eligible turns and fact projection rows in a history.

    Encoding happens outside SQLite write transactions. Each completed batch is
    revalidated and committed atomically, so retries refresh only changed text.
    """
    validate_embedding_provider(provider)
    if not isinstance(history_id, str) or not history_id:
        raise ValueError("history_id must be a non-empty string")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    sources = _read_index_sources(
        database, history_id, provider.model_id, timeout, gated=gated
    )
    indexed = unchanged = ineligible = 0
    pending: list[tuple[str, str, str, str]] = []

    def write_pending() -> None:
        nonlocal indexed, ineligible, pending
        if not pending:
            return
        encoded = provider.encode_documents([text for _, _, text, _ in pending])
        vectors = _validate_vector_batch(encoded, len(pending), provider.dimension)
        with transaction(database, timeout=timeout) as conn:
            for (ref_type, ref_id, text, digest), vector in zip(pending, vectors, strict=True):
                current = _read_source_text(
                    conn, history_id, ref_type, ref_id, gated=gated
                )
                if current is None:
                    ineligible += 1
                    continue
                if content_digest(current) != digest:
                    raise EmbeddingSourceChanged(
                        f"Source {ref_type}:{ref_id} changed while it was being embedded"
                    )
                conn.execute(
                    """INSERT INTO embeddings (ref_type, ref_id, model, vec, content_hash)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(ref_type, ref_id, model) DO UPDATE SET
                         vec = excluded.vec, content_hash = excluded.content_hash""",
                    (ref_type, ref_id, provider.model_id,
                     serialize_vector(vector, dimension=provider.dimension), digest),
                )
                indexed += 1
        pending = []

    for ref_type, ref_id, text, stored_digest in sources:
        digest = content_digest(text)
        if stored_digest == digest:
            unchanged += 1
            continue
        pending.append((ref_type, ref_id, text, digest))
        if len(pending) >= batch_size:
            write_pending()
    write_pending()
    logger.info(
        "Indexed history=%s model=%s indexed=%d unchanged=%d ineligible=%d",
        history_id, provider.model_id, indexed, unchanged, ineligible,
    )
    return IndexingSummary(history_id, provider.model_id, indexed, unchanged, ineligible)


def check_index_freshness(
    database: str | Path,
    history_id: str,
    provider_or_model_id: EmbeddingProvider | str,
    *,
    ref_types: Sequence[str] = ("turn", "fact"),
    timeout: float = 5.0,
    gated: bool = True,
) -> IndexFreshnessReport:
    """Read-only validation of the eligible source set against persisted vectors.

    This operation does not encode data, write rows, or repair stale embeddings.
    Pass either a provider or its exact persisted model identity. Retrieval can
    use the private connection-aware form to keep validation in its read snapshot.
    """
    if not isinstance(history_id, str) or not history_id:
        raise ValueError("history_id must be a non-empty string")
    model_id = (
        provider_or_model_id
        if isinstance(provider_or_model_id, str)
        else getattr(provider_or_model_id, "model_id", None)
    )
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("provider must expose a non-empty model_id")
    selected_types = tuple(dict.fromkeys(ref_types))
    if not selected_types or any(value not in ("turn", "fact") for value in selected_types):
        raise ValueError("ref_types must contain 'turn' and/or 'fact'")
    with connection(database, timeout=timeout) as conn:
        conn.execute("BEGIN")
        return _check_index_freshness(
            conn, history_id, model_id, selected_types, gated=gated
        )


def _check_index_freshness(
    conn: sqlite3.Connection,
    history_id: str,
    model_id: str,
    ref_types: Sequence[str] = ("turn", "fact"),
    *,
    gated: bool = True,
) -> IndexFreshnessReport:
    """Connection-aware freshness check used inside a consistent read snapshot."""
    if any(value not in ("turn", "fact") for value in ref_types):
        raise ValueError("ref_types must contain only 'turn' and/or 'fact'")
    sources: list[tuple[str, str, str]] = []
    if "turn" in ref_types:
        rows = conn.execute(
            "SELECT t.turn_id AS ref_id, t.raw_content AS content FROM turns AS t "
            "WHERE t.history_id = ? AND " + turn_eligibility_predicate("t", gated=gated) +
            " ORDER BY t.turn_id",
            (history_id,),
        ).fetchall()
        sources.extend(("turn", str(row["ref_id"]), str(row["content"])) for row in rows)
    if "fact" in ref_types:
        rows = conn.execute(
            "SELECT f.fact_id AS ref_id, f.key_text AS content FROM facts AS f "
            "WHERE f.history_id = ? AND " + fact_eligibility_predicate("f") +
            " ORDER BY f.fact_id",
            (history_id,),
        ).fetchall()
        sources.extend(("fact", str(row["ref_id"]), str(row["content"])) for row in rows)

    embeddings: list[sqlite3.Row] = []
    # Read in bounded chunks without creating a temp table or changing the DB.
    required_ids = tuple(dict.fromkeys(ref_id for _, ref_id, _ in sources))
    for offset in range(0, len(required_ids), 800):
        required = required_ids[offset:offset + 800]
        values_sql = ",".join("(?)" for _ in required)
        rows = conn.execute(
            "WITH required(ref_id) AS (VALUES " + values_sql + ") "
            "SELECT e.ref_type, e.ref_id, e.model, e.content_hash "
            "FROM embeddings AS e JOIN required AS r "
            "ON r.ref_id = e.ref_id "
            "ORDER BY e.ref_type, e.ref_id, e.model",
            required,
        ).fetchall()
        embeddings.extend(rows)

    stored: dict[tuple[str, str], list[tuple[str, str | None]]] = {}
    stored_by_ref_id: dict[str, list[tuple[str, str, str | None]]] = {}
    for row in embeddings:
        key = str(row["ref_type"]), str(row["ref_id"])
        model = str(row["model"])
        digest = row["content_hash"]
        stored.setdefault(key, []).append((model, digest))
        stored_by_ref_id.setdefault(str(row["ref_id"]), []).append(
            (str(row["ref_type"]), model, digest)
        )

    issues: list[IndexFreshnessIssue] = []
    fresh_count = 0
    for ref_type, ref_id, text in sources:
        rows = stored.get((ref_type, ref_id), [])
        matching = next((content_hash for model, content_hash in rows if model == model_id), None)
        if not rows:
            other_types = stored_by_ref_id.get(ref_id, [])
            if other_types:
                issues.append(
                    IndexFreshnessIssue(
                        ref_type, ref_id, "reference_type_mismatch",
                        tuple(sorted({model for _, model, _ in other_types})),
                    )
                )
            else:
                issues.append(IndexFreshnessIssue(ref_type, ref_id, "missing_embedding"))
        elif not any(model == model_id for model, _ in rows):
            issues.append(
                IndexFreshnessIssue(
                    ref_type, ref_id, "model_identity_mismatch",
                    tuple(sorted(model for model, _ in rows)),
                )
            )
        elif matching is None:
            issues.append(
                IndexFreshnessIssue(ref_type, ref_id, "missing_content_digest")
            )
        elif str(matching) != content_digest(text):
            issues.append(IndexFreshnessIssue(ref_type, ref_id, "content_changed"))
        else:
            fresh_count += 1
    return IndexFreshnessReport(
        history_id, model_id, len(sources), fresh_count, tuple(issues)
    )


def validate_embedding_provider(provider: EmbeddingProvider) -> None:
    """Check that an injected provider matches the persisted BGE-M3 contract."""
    model_id = getattr(provider, "model_id", None)
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("embedding provider must expose a non-empty model_id")
    dimension = getattr(provider, "dimension", None)
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension != BGE_M3_DIMENSION
    ):
        raise EmbeddingValidationError(
            f"embedding provider dimension must be {BGE_M3_DIMENSION}"
        )
    if not callable(getattr(provider, "encode_query", None)) or not callable(
        getattr(provider, "encode_documents", None)
    ):
        raise TypeError("embedding provider must implement query and document encoding")


def _validate_vector_batch(
    encoded: Sequence[Sequence[float]] | np.ndarray,
    count: int,
    dimension: int,
) -> list[np.ndarray]:
    vectors = np.asarray(encoded)
    if vectors.ndim != 2 or vectors.shape != (count, dimension):
        raise EmbeddingValidationError(
            f"provider returned shape {vectors.shape}, expected ({count}, {dimension})"
        )
    return [normalize_vector(vector, dimension=dimension) for vector in vectors]


def _read_index_sources(
    database: str | Path,
    history_id: str,
    model_id: str,
    timeout: float,
    *,
    gated: bool = True,
) -> list[tuple[str, str, str, str | None]]:
    with connection(database, timeout=timeout) as conn:
        turns = conn.execute(
            """SELECT 'turn' AS ref_type, t.turn_id AS ref_id, t.raw_content AS content,
                      e.content_hash
               FROM turns AS t
               LEFT JOIN embeddings AS e ON e.ref_type = 'turn' AND e.ref_id = t.turn_id
                                            AND e.model = ?
               WHERE t.history_id = ? AND """ + turn_eligibility_predicate("t", gated=gated) + " ORDER BY t.turn_id",
            (model_id, history_id),
        ).fetchall()
        facts = conn.execute(
            """SELECT 'fact' AS ref_type, f.fact_id AS ref_id, f.key_text AS content,
                      e.content_hash
               FROM facts AS f
               LEFT JOIN embeddings AS e ON e.ref_type = 'fact' AND e.ref_id = f.fact_id
                                            AND e.model = ?
               WHERE f.history_id = ? AND """ + fact_eligibility_predicate("f") + " ORDER BY f.fact_id",
            (model_id, history_id),
        ).fetchall()
    return [
        (str(row["ref_type"]), str(row["ref_id"]), str(row["content"]), row["content_hash"])
        for row in (*turns, *facts)
    ]


def _read_source_text(
    conn: sqlite3.Connection,
    history_id: str,
    ref_type: str,
    ref_id: str,
    *,
    gated: bool = True,
) -> str | None:
    if ref_type == "turn":
        row = conn.execute(
            f"SELECT t.raw_content AS content FROM turns AS t WHERE t.turn_id = ? "
            f"AND t.history_id = ? AND {turn_eligibility_predicate('t', gated=gated)}",
            (ref_id, history_id),
        ).fetchone()
    else:
        row = conn.execute(
            f"SELECT f.key_text AS content FROM facts AS f WHERE f.fact_id = ? "
            f"AND f.history_id = ? AND {fact_eligibility_predicate('f')}",
            (ref_id, history_id),
        ).fetchone()
    return str(row["content"]) if row is not None else None
