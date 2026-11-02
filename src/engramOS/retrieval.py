"""Quarantine-aware lexical, dense, temporal, and fused retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
import sqlite3
from typing import Literal, Mapping, Sequence

import numpy as np

from .db import connection
from .eligibility import fact_eligibility_predicate, turn_eligibility_predicate
from .embeddings import (
    EmbeddingProvider,
    _check_index_freshness,
    normalize_vector,
    validate_embedding_provider,
)
from .temporal import FactView, _retrieve_facts
from .vector_index import DenseIndexUnavailable, DenseMatch, InMemoryVectorIndex

_WORD = re.compile(r"[\w]+", re.UNICODE)


class DenseRetrievalUnavailable(DenseIndexUnavailable):
    """The selected hybrid retrieval cannot produce its required dense channel."""


def _require_fresh_index(
    conn: sqlite3.Connection,
    history_id: str,
    model_id: str,
    ref_types: Sequence[str],
    *,
    gated: bool = True,
) -> None:
    report = _check_index_freshness(
        conn, history_id, model_id, ref_types, gated=gated
    )
    if not report.is_fresh:
        raise report.stale_error()


@dataclass(frozen=True, slots=True)
class FtsCandidate:
    ref_type: Literal["turn", "fact"]
    ref_id: str
    rank: int
    bm25_score: float
    record: dict[str, object]


@dataclass(frozen=True, slots=True)
class FusedRank:
    ref_id: str
    rrf_score: float
    channels: tuple[str, ...]
    channel_ranks: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class RetrievedTurn:
    turn_id: str
    history_id: str
    session_id: str
    turn_idx: int
    role: str
    raw_content: str
    source_type: str
    event_time: float
    recorded_at: float
    dense_score: float | None
    bm25_score: float | None
    rrf_score: float
    channels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvidenceTurn:
    turn_id: str
    history_id: str
    session_id: str
    turn_idx: int
    role: str
    raw_content: str
    source_type: str
    event_time: float
    recorded_at: float


@dataclass(frozen=True, slots=True)
class RetrievedFact:
    fact: Mapping[str, object]
    parent_fact: Mapping[str, object] | None
    supporting_turns: tuple[EvidenceTurn, ...]
    source_turn: EvidenceTurn
    confirmation_turn: EvidenceTurn | None
    annotation: str | None
    dense_score: float | None
    bm25_score: float | None
    rrf_score: float
    channels: tuple[str, ...]

    @property
    def fact_id(self) -> str:
        return str(self.fact["fact_id"])


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostics:
    channel_candidate_counts: Mapping[str, int]
    embedding_model: str
    vector_dimension: int
    similarity_device: str
    rrf_k: int
    turn_candidate_limit: int
    fact_candidate_limit: int
    final_turn_count: int
    final_fact_count: int
    gated: bool = True


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    history_id: str
    view: FactView
    baseline: Literal["hybrid_rag", "recency", "memory"]
    turns: tuple[RetrievedTurn, ...]
    facts: tuple[RetrievedFact, ...]
    diagnostics: RetrievalDiagnostics


def safe_fts_query(query: str) -> str:
    """Tokenize and quote query words so raw FTS operators are never executed."""
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    tokens = list(dict.fromkeys(_WORD.findall(query)))
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)


def search_turns_fts(
    database: str | Path,
    history_id: str,
    query: str,
    *,
    limit: int = 50,
    timeout: float = 5.0,
    gated: bool = True,
) -> list[FtsCandidate]:
    _validate_scope(history_id, limit)
    with connection(database, timeout=timeout) as conn:
        return _search_turns_fts(conn, history_id, query, limit=limit, gated=gated)


def _search_turns_fts(
    conn: sqlite3.Connection,
    history_id: str,
    query: str,
    *,
    limit: int,
    gated: bool = True,
) -> list[FtsCandidate]:
    match_query = safe_fts_query(query)
    if not match_query:
        return []
    rows = conn.execute(
        """SELECT t.*, bm25(turns_fts) AS lexical_score
           FROM turns_fts
           JOIN turns AS t ON t.rowid = turns_fts.rowid
           WHERE turns_fts MATCH ? AND t.history_id = ? AND """
        + turn_eligibility_predicate("t", gated=gated)
        + " ORDER BY lexical_score ASC, t.turn_id ASC LIMIT ?",
        (match_query, history_id, limit),
    ).fetchall()
    return [
        FtsCandidate("turn", str(row["turn_id"]), rank, float(row["lexical_score"]),
                     {key: row[key] for key in row.keys() if key != "lexical_score"})
        for rank, row in enumerate(rows, start=1)
    ]


def search_facts_fts(
    database: str | Path,
    history_id: str,
    query: str,
    *,
    view: FactView = "current",
    as_of: object | None = None,
    limit: int = 50,
    entity: str | None = None,
    attribute: str | None = None,
    timeout: float = 5.0,
) -> list[FtsCandidate]:
    _validate_scope(history_id, limit)
    _validate_view(view, as_of)
    with connection(database, timeout=timeout) as conn:
        conn.execute("BEGIN")
        return _search_facts_fts(
            conn, history_id, query, view=view, as_of=as_of,
            limit=limit, entity=entity, attribute=attribute,
        )


def _search_facts_fts(
    conn: sqlite3.Connection,
    history_id: str,
    query: str,
    *,
    view: FactView,
    as_of: object | None,
    limit: int,
    entity: str | None,
    attribute: str | None,
) -> list[FtsCandidate]:
    eligible = _fact_rows_for_view(
        conn, history_id, view=view, as_of=as_of, entity=entity, attribute=attribute,
    )
    if not eligible:
        return []
    match_query = safe_fts_query(query)
    if not match_query:
        return []
    table_name = "retrieval_allowed_fact_ids"
    conn.execute(f"DROP TABLE IF EXISTS temp.{table_name}")
    conn.execute(f"CREATE TEMP TABLE {table_name} (fact_id TEXT PRIMARY KEY)")
    try:
        conn.executemany(
            f"INSERT INTO {table_name}(fact_id) VALUES (?)",
            ((fact_id,) for fact_id in eligible),
        )
        rows = conn.execute(
            """SELECT f.*, bm25(facts_fts) AS lexical_score
               FROM facts_fts
               JOIN facts AS f ON f.rowid = facts_fts.rowid
               JOIN retrieval_allowed_fact_ids AS allowed ON allowed.fact_id = f.fact_id
               WHERE facts_fts MATCH ? AND f.history_id = ? AND """
            + fact_eligibility_predicate("f")
            + " ORDER BY lexical_score ASC, f.fact_id ASC LIMIT ?",
            (match_query, history_id, limit),
        ).fetchall()
    finally:
        conn.execute(f"DROP TABLE IF EXISTS temp.{table_name}")
    results: list[FtsCandidate] = []
    for rank, row in enumerate(rows, start=1):
        if str(row["fact_id"]) not in eligible:
            continue
        record = {key: row[key] for key in row.keys() if key != "lexical_score"}
        if record.get("key_type") == "trigger":
            record["parent_fact"] = eligible.get(str(record.get("parent_fact_id")))
        else:
            # The temporal projection, not a stale persisted row, is authoritative.
            record.update(eligible[str(row["fact_id"])])
        results.append(
            FtsCandidate("fact", str(row["fact_id"]), rank, float(row["lexical_score"]), record)
        )
    valid_ids = _eligible_fact_ids(
        conn, history_id,
        [ref_id for ref_id in (result.ref_id for result in results)]
        + [str(result.record.get("parent_fact_id")) for result in results if result.record.get("parent_fact_id")],
    )
    return [result for result in results if result.ref_id in valid_ids]


def search_turns_dense(
    database: str | Path,
    history_id: str,
    query: str,
    provider: EmbeddingProvider,
    *,
    limit: int = 50,
    timeout: float = 5.0,
    gated: bool = True,
) -> list[DenseMatch]:
    _validate_scope(history_id, limit)
    validate_embedding_provider(provider)
    query_vector = _encode_query(provider, query)
    with connection(database, timeout=timeout) as conn:
        conn.execute("BEGIN")
        _require_fresh_index(
            conn, history_id, provider.model_id, ("turn",), gated=gated
        )
        index = InMemoryVectorIndex.load_from_connection(
            conn, history_id, "turn", provider.model_id,
            dimension=provider.dimension, validate_freshness=False, gated=gated,
        )
        if index.size == 0:
            return []
        matches = index.search(query_vector, limit=limit)
        eligible_ids = _eligible_turn_ids(
            conn, history_id, [match.ref_id for match in matches], gated=gated
        )
        return [match for match in matches if match.ref_id in eligible_ids]


def search_facts_dense(
    database: str | Path,
    history_id: str,
    query: str,
    provider: EmbeddingProvider,
    *,
    view: FactView = "current",
    as_of: object | None = None,
    limit: int = 50,
    entity: str | None = None,
    attribute: str | None = None,
    timeout: float = 5.0,
) -> list[DenseMatch]:
    _validate_scope(history_id, limit)
    validate_embedding_provider(provider)
    _validate_view(view, as_of)
    query_vector = _encode_query(provider, query)
    with connection(database, timeout=timeout) as conn:
        conn.execute("BEGIN")
        _require_fresh_index(conn, history_id, provider.model_id, ("fact",))
        eligible = _fact_rows_for_view(
            conn, history_id, view=view, as_of=as_of,
            entity=entity, attribute=attribute,
        )
        if not eligible:
            return []
        index = InMemoryVectorIndex.load_from_connection(
            conn, history_id, "fact", provider.model_id,
            dimension=provider.dimension, validate_freshness=False,
        )
        if index.size == 0:
            return []
        matches = index.search(query_vector, limit=limit, allowed_ids=set(eligible))
        results = [
            DenseMatch(
                match.ref_type, match.ref_id, match.rank, match.cosine_score,
                _merge_temporal_fact(match.record, eligible[match.ref_id]),
            )
            for match in matches if match.ref_id in eligible
        ]
        valid_ids = _eligible_fact_ids(
            conn, history_id,
            [result.ref_id for result in results]
            + [str(result.record.get("parent_fact_id")) for result in results
               if result.record.get("parent_fact_id")],
        )
        return [result for result in results if result.ref_id in valid_ids]


def reciprocal_rank_fusion(
    channels: Mapping[str, Sequence[object]],
    *,
    k: int = 60,
    limit: int | None = None,
) -> list[FusedRank]:
    """Fuse ranked candidates using ``sum(1 / (k + one_based_rank))``."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("RRF k must be a non-negative integer")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise ValueError("limit must be a positive integer when supplied")
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for channel_name, candidates in channels.items():
        seen: set[str] = set()
        for rank, candidate in enumerate(candidates, start=1):
            ref_id = _candidate_id(candidate)
            if ref_id in seen:
                continue
            seen.add(ref_id)
            scores[ref_id] = scores.get(ref_id, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(ref_id, {})[channel_name] = rank
    ordered = sorted(scores, key=lambda ref_id: (-scores[ref_id], ref_id))
    if limit is not None:
        ordered = ordered[:limit]
    return [
        FusedRank(ref_id, scores[ref_id], tuple(ranks[ref_id]), dict(ranks[ref_id]))
        for ref_id in ordered
    ]


def hybrid_rag(
    database: str | Path,
    history_id: str,
    query: str,
    embedding_provider: EmbeddingProvider,
    *,
    top_k: int = 10,
    candidate_limit: int = 50,
    rrf_k: int = 60,
    timeout: float = 5.0,
    gated: bool = True,
) -> RetrievalResult:
    """Return the specified dense+BM25 RRF top-k verbatim turns."""
    return _retrieve_turns(
        database, history_id, query, embedding_provider,
        top_k=top_k, candidate_limit=candidate_limit, rrf_k=rrf_k,
        timeout=timeout, baseline="hybrid_rag", gated=gated,
    )


def recency(
    database: str | Path,
    history_id: str,
    query: str,
    embedding_provider: EmbeddingProvider,
    *,
    top_k: int = 10,
    candidate_limit: int = 50,
    rrf_k: int = 60,
    timeout: float = 5.0,
    gated: bool = True,
) -> RetrievalResult:
    """Sort the exact same top-k hybrid candidate set by descending event time."""
    result = _retrieve_turns(
        database, history_id, query, embedding_provider,
        top_k=top_k, candidate_limit=candidate_limit, rrf_k=rrf_k,
        timeout=timeout, baseline="recency", gated=gated,
    )
    ordered = tuple(
        sorted(
            result.turns,
            key=lambda turn: (-turn.event_time, turn.session_id, turn.turn_idx, turn.turn_id),
        )
    )
    return RetrievalResult(
        result.history_id, result.view, "recency", ordered, result.facts,
        RetrievalDiagnostics(
            result.diagnostics.channel_candidate_counts,
            result.diagnostics.embedding_model,
            result.diagnostics.vector_dimension,
            result.diagnostics.similarity_device,
            result.diagnostics.rrf_k,
            result.diagnostics.turn_candidate_limit,
            result.diagnostics.fact_candidate_limit,
            len(ordered),
            result.diagnostics.final_fact_count,
            result.diagnostics.gated,
        ),
    )


def retrieve_memory(
    database: str | Path,
    history_id: str,
    query: str,
    embedding_provider: EmbeddingProvider,
    *,
    view: FactView = "current",
    as_of: object | None = None,
    top_k: int = 10,
    fact_top_k: int = 10,
    turn_candidate_limit: int = 50,
    fact_candidate_limit: int = 50,
    rrf_k: int = 60,
    entity: str | None = None,
    attribute: str | None = None,
    timeout: float = 5.0,
) -> RetrievalResult:
    """Retrieve fused turns and temporal facts for an explicitly selected view."""
    validate_embedding_provider(embedding_provider)
    _validate_scope(history_id, top_k)
    _validate_scope(history_id, fact_top_k)
    _validate_scope(history_id, turn_candidate_limit)
    _validate_scope(history_id, fact_candidate_limit)
    _validate_view(view, as_of)
    query_vector = _encode_query(embedding_provider, query)
    with connection(database, timeout=timeout) as conn:
        conn.execute("BEGIN")
        _require_fresh_index(
            conn, history_id, embedding_provider.model_id, ("turn", "fact")
        )
        turn_fts = _search_turns_fts(
            conn, history_id, query, limit=turn_candidate_limit
        )
        turn_index = InMemoryVectorIndex.load_from_connection(
            conn, history_id, "turn", embedding_provider.model_id,
            dimension=embedding_provider.dimension, validate_freshness=False,
        )
        if turn_index.size == 0:
            raise DenseIndexUnavailable(
                f"Hybrid retrieval requires turn embeddings for history {history_id!r} "
                f"and model {embedding_provider.model_id!r}; run index_history first"
            )
        turn_dense = turn_index.search(query_vector, limit=turn_candidate_limit)
        if not turn_dense:
            raise DenseIndexUnavailable("Hybrid retrieval found no eligible dense turn candidates")
        turns = _fuse_turns(turn_fts, turn_dense, top_k=top_k, rrf_k=rrf_k)

        fact_fts = _search_facts_fts(
            conn, history_id, query, view=view, as_of=as_of,
            limit=fact_candidate_limit, entity=entity, attribute=attribute,
        )
        eligible_facts = _fact_rows_for_view(
            conn, history_id, view=view, as_of=as_of,
            entity=entity, attribute=attribute,
        )
        fact_dense: list[DenseMatch] = []
        if eligible_facts:
            fact_index = InMemoryVectorIndex.load_from_connection(
                conn, history_id, "fact", embedding_provider.model_id,
                dimension=embedding_provider.dimension, validate_freshness=False,
            )
            if fact_index.size:
                fact_dense = [
                    DenseMatch(
                        match.ref_type, match.ref_id, match.rank, match.cosine_score,
                        _merge_temporal_fact(match.record, eligible_facts[match.ref_id]),
                    )
                    for match in fact_index.search(
                        query_vector, limit=fact_candidate_limit,
                        allowed_ids=set(eligible_facts),
                    )
                ]
        fact_fused = reciprocal_rank_fusion(
            {"fact_lexical": fact_fts, "fact_dense": fact_dense},
            k=rrf_k, limit=fact_top_k,
        )
        if view == "history":
            # History is a timeline, not a top-k semantic ranking. The lexical
            # and dense fact channels still report query candidates in diagnostics.
            fused_facts = list(_retrieve_temporal_facts(
                conn, history_id, view="history", as_of=None,
                entity=entity, attribute=attribute
            ))
        else:
            fused_facts = _expand_fact_evidence(
                conn, history_id, fact_fused, eligible_facts, fact_fts, fact_dense,
            )
        turns, fused_facts = _revalidate_results(
            conn, history_id, turns, fused_facts,
            view=view, as_of=as_of, entity=entity, attribute=attribute,
        )
        counts = {
            "turn_lexical": len(turn_fts),
            "turn_dense": len(turn_dense),
            "fact_lexical": len(fact_fts),
            "fact_dense": len(fact_dense),
        }
        diagnostics = RetrievalDiagnostics(
            counts, embedding_provider.model_id, embedding_provider.dimension,
            turn_index.device, rrf_k, turn_candidate_limit, fact_candidate_limit,
            len(turns), len(fused_facts),
        )
        return RetrievalResult(
            history_id, view, "memory", tuple(turns), tuple(fused_facts), diagnostics
        )


def retrieve_temporal_facts(
    database: str | Path,
    history_id: str,
    *,
    view: FactView = "current",
    as_of: object | None = None,
    entity: str | None = None,
    attribute: str | None = None,
    timeout: float = 5.0,
) -> tuple[RetrievedFact, ...]:
    """Return the full eligible temporal fact view with its source evidence.

    History view preserves every eligible interval in ``valid_from`` order;
    unlike query-ranked ``retrieve_memory``, this function does not truncate or
    reorder the timeline by lexical/dense relevance.
    """
    if not history_id:
        raise ValueError("history_id is required")
    _validate_view(view, as_of)
    with connection(database, timeout=timeout) as conn:
        conn.execute("BEGIN")
        return _retrieve_temporal_facts(
            conn, history_id, view=view, as_of=as_of,
            entity=entity, attribute=attribute,
        )


def _retrieve_temporal_facts(
    conn: sqlite3.Connection,
    history_id: str,
    *,
    view: FactView,
    as_of: object | None,
    entity: str | None,
    attribute: str | None,
) -> tuple[RetrievedFact, ...]:
    eligible = _fact_rows_for_view(
        conn, history_id, view=view, as_of=as_of, entity=entity, attribute=attribute,
    )
    ordered_ids = sorted(
        eligible,
        key=lambda fact_id: (
            float(eligible[fact_id].get("valid_from", 0.0)),
            str(eligible[fact_id].get("entity") or ""),
            str(eligible[fact_id].get("attribute") or ""),
            fact_id,
        ),
    )
    ordered = [FusedRank(fact_id, 0.0, (), {}) for fact_id in ordered_ids]
    expanded = _expand_fact_evidence(conn, history_id, ordered, eligible, (), ())
    return tuple(_revalidate_fact_results(
        conn, history_id, expanded, view=view, as_of=as_of,
        entity=entity, attribute=attribute,
    ))


def _retrieve_turns(
    database: str | Path,
    history_id: str,
    query: str,
    provider: EmbeddingProvider,
    *,
    top_k: int,
    candidate_limit: int,
    rrf_k: int,
    timeout: float,
    baseline: Literal["hybrid_rag", "recency"],
    gated: bool = True,
) -> RetrievalResult:
    validate_embedding_provider(provider)
    _validate_scope(history_id, top_k)
    _validate_scope(history_id, candidate_limit)
    query_vector = _encode_query(provider, query)
    with connection(database, timeout=timeout) as conn:
        conn.execute("BEGIN")
        _require_fresh_index(
            conn, history_id, provider.model_id, ("turn",), gated=gated
        )
        lexical = _search_turns_fts(
            conn, history_id, query, limit=candidate_limit, gated=gated
        )
        index = InMemoryVectorIndex.load_from_connection(
            conn, history_id, "turn", provider.model_id,
            dimension=provider.dimension, validate_freshness=False, gated=gated,
        )
        if index.size == 0:
            raise DenseRetrievalUnavailable(
                f"{baseline} requires dense turn embeddings for history {history_id!r} "
                f"and model {provider.model_id!r}; run index_history first"
            )
        dense = index.search(query_vector, limit=candidate_limit)
        if not dense:
            raise DenseRetrievalUnavailable(f"{baseline} has no eligible dense turn candidates")
        turns = _fuse_turns(lexical, dense, top_k=top_k, rrf_k=rrf_k)
        turns = _filter_turn_results(conn, history_id, turns, gated=gated)
        diagnostics = RetrievalDiagnostics(
            {"turn_lexical": len(lexical), "turn_dense": len(dense), "fact_lexical": 0, "fact_dense": 0},
            provider.model_id, provider.dimension, index.device, rrf_k,
            candidate_limit, 0, len(turns), 0, gated,
        )
        return RetrievalResult(history_id, "current", baseline, tuple(turns), (), diagnostics)


def _fuse_turns(
    lexical: Sequence[FtsCandidate],
    dense: Sequence[DenseMatch],
    *,
    top_k: int,
    rrf_k: int,
) -> list[RetrievedTurn]:
    fused = reciprocal_rank_fusion(
        {"turn_lexical": lexical, "turn_dense": dense}, k=rrf_k, limit=top_k
    )
    lex_by_id = {item.ref_id: item for item in lexical}
    dense_by_id = {item.ref_id: item for item in dense}
    turns: list[RetrievedTurn] = []
    for item in fused:
        candidate = dense_by_id.get(item.ref_id) or lex_by_id[item.ref_id]
        record = candidate.record
        lexical_item = lex_by_id.get(item.ref_id)
        dense_item = dense_by_id.get(item.ref_id)
        turns.append(
            RetrievedTurn(
                str(record["turn_id"]), str(record["history_id"]), str(record["session_id"]),
                int(record["turn_idx"]), str(record["role"]), str(record["raw_content"]),
                str(record["source_type"]), float(record["event_time"]),
                float(record["recorded_at"]),
                dense_item.cosine_score if dense_item else None,
                lexical_item.bm25_score if lexical_item else None,
                item.rrf_score, item.channels,
            )
        )
    return turns


def _fact_rows_for_view(
    conn: sqlite3.Connection,
    history_id: str,
    *,
    view: FactView,
    as_of: object | None,
    entity: str | None,
    attribute: str | None,
) -> dict[str, dict[str, object]]:
    _validate_view(view, as_of)
    slot_rows = _retrieve_facts(
        conn, history_id, view=view, as_of=as_of,  # type: ignore[arg-type]
        entity=entity, attribute=attribute,
    )
    output = {str(row["fact_id"]): dict(row) for row in slot_rows}
    if not output:
        return output
    table_name = "retrieval_view_parent_ids"
    conn.execute(f"DROP TABLE IF EXISTS temp.{table_name}")
    conn.execute(f"CREATE TEMP TABLE {table_name} (fact_id TEXT PRIMARY KEY)")
    try:
        conn.executemany(
            f"INSERT INTO {table_name}(fact_id) VALUES (?)",
            ((fact_id,) for fact_id in output),
        )
        trigger_rows = conn.execute(
            f"SELECT f.* FROM facts AS f JOIN {table_name} AS parent_view "
            "ON parent_view.fact_id = f.parent_fact_id "
            "WHERE f.history_id = ? AND f.key_type = 'trigger' AND "
            + fact_eligibility_predicate("f") + " ORDER BY f.fact_id",
            (history_id,),
        ).fetchall()
    finally:
        conn.execute(f"DROP TABLE IF EXISTS temp.{table_name}")
    for row in trigger_rows:
        trigger = {key: row[key] for key in row.keys()}
        trigger["parent_fact"] = output[str(row["parent_fact_id"])]
        output[str(row["fact_id"])] = trigger
    return output


def _merge_temporal_fact(
    stored_record: Mapping[str, object], temporal_record: Mapping[str, object]
) -> dict[str, object]:
    result = dict(stored_record)
    result.update(temporal_record)
    if result.get("key_type") == "trigger":
        result["parent_fact"] = temporal_record.get("parent_fact")
    return result


def _expand_fact_evidence(
    conn: sqlite3.Connection,
    history_id: str,
    fused: Sequence[FusedRank],
    allowed: Mapping[str, Mapping[str, object]],
    lexical: Sequence[FtsCandidate],
    dense: Sequence[DenseMatch],
) -> list[RetrievedFact]:
    support_by_id: dict[str, Mapping[str, object]] = {}
    turn_ids: set[str] = set()
    for fused_item in fused:
        selected = allowed.get(fused_item.ref_id)
        if selected is None:
            continue
        parent = (
            selected.get("parent_fact")
            if selected.get("key_type") == "trigger"
            else selected
        )
        if not isinstance(parent, Mapping):
            continue
        support_by_id[fused_item.ref_id] = parent
        source_id = parent.get("turn_id")
        confirmation_id = parent.get("last_confirmed_turn_id")
        if source_id:
            turn_ids.add(str(source_id))
        if confirmation_id:
            turn_ids.add(str(confirmation_id))
    if not turn_ids:
        return []
    ids = tuple(sorted(turn_ids))
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        "SELECT t.* FROM turns AS t WHERE t.history_id = ? AND t.turn_id IN ("
        + placeholders + ") AND " + turn_eligibility_predicate("t"),
        (history_id, *ids),
    ).fetchall()
    turns_by_id = {str(row["turn_id"]): _evidence_turn(row) for row in rows}

    output: list[RetrievedFact] = []
    lexical_by_id = {item.ref_id: item for item in lexical}
    dense_by_id = {item.ref_id: item for item in dense}
    for fused_item in fused:
        selected = allowed.get(fused_item.ref_id)
        parent = support_by_id.get(fused_item.ref_id)
        if selected is None or parent is None:
            continue
        source_id = str(parent.get("turn_id") or "")
        confirmation_id = parent.get("last_confirmed_turn_id")
        source_turn = turns_by_id.get(source_id)
        confirmation_turn = turns_by_id.get(str(confirmation_id)) if confirmation_id else None
        # A missing or quarantined provenance turn invalidates supported output.
        if source_turn is None or (confirmation_id and confirmation_turn is None):
            continue
        supporting = [source_turn]
        if confirmation_turn is not None and confirmation_turn.turn_id != source_turn.turn_id:
            supporting.append(confirmation_turn)
        fact_record = dict(selected)
        if selected.get("key_type") == "trigger":
            fact_record["parent_fact"] = dict(parent)
        annotation = None
        if selected.get("valid_until") is not None:
            annotation = f"[valid: {selected['valid_from']} -> {selected['valid_until']}]"
        lexical_item = lexical_by_id.get(fused_item.ref_id)
        dense_item = dense_by_id.get(fused_item.ref_id)
        output.append(
            RetrievedFact(
                fact_record,
                dict(parent) if selected.get("key_type") == "trigger" else None,
                tuple(supporting), source_turn, confirmation_turn, annotation,
                dense_item.cosine_score if dense_item else None,
                lexical_item.bm25_score if lexical_item else None,
                fused_item.rrf_score, fused_item.channels,
            )
        )
    return output


def _evidence_turn(row: object) -> EvidenceTurn:
    return EvidenceTurn(
        str(row["turn_id"]), str(row["history_id"]), str(row["session_id"]),
        int(row["turn_idx"]), str(row["role"]), str(row["raw_content"]),
        str(row["source_type"]), float(row["event_time"]), float(row["recorded_at"]),
    )


def _eligible_turn_ids(
    conn: sqlite3.Connection,
    history_id: str,
    turn_ids: Sequence[str],
    *,
    gated: bool = True,
) -> set[str]:
    if not turn_ids:
        return set()
    ids = tuple(dict.fromkeys(turn_ids))
    table_name = "retrieval_eligible_turn_ids"
    conn.execute(f"DROP TABLE IF EXISTS temp.{table_name}")
    conn.execute(f"CREATE TEMP TABLE {table_name} (turn_id TEXT PRIMARY KEY)")
    try:
        conn.executemany(
            f"INSERT INTO {table_name}(turn_id) VALUES (?)",
            ((turn_id,) for turn_id in ids),
        )
        rows = conn.execute(
            "SELECT t.turn_id FROM turns AS t "
            f"JOIN {table_name} AS candidate ON candidate.turn_id = t.turn_id "
            "WHERE t.history_id = ? AND " + turn_eligibility_predicate("t", gated=gated),
            (history_id,),
        ).fetchall()
    finally:
        conn.execute(f"DROP TABLE IF EXISTS temp.{table_name}")
    return {str(row["turn_id"]) for row in rows}


def _eligible_fact_ids(
    conn: sqlite3.Connection,
    history_id: str,
    fact_ids: Sequence[str],
) -> set[str]:
    if not fact_ids:
        return set()
    ids = tuple(dict.fromkeys(fact_ids))
    table_name = "retrieval_eligible_fact_ids"
    conn.execute(f"DROP TABLE IF EXISTS temp.{table_name}")
    conn.execute(f"CREATE TEMP TABLE {table_name} (fact_id TEXT PRIMARY KEY)")
    try:
        conn.executemany(
            f"INSERT INTO {table_name}(fact_id) VALUES (?)",
            ((fact_id,) for fact_id in ids),
        )
        rows = conn.execute(
            "SELECT f.fact_id FROM facts AS f "
            f"JOIN {table_name} AS candidate ON candidate.fact_id = f.fact_id "
            "WHERE f.history_id = ? AND " + fact_eligibility_predicate("f"),
            (history_id,),
        ).fetchall()
    finally:
        conn.execute(f"DROP TABLE IF EXISTS temp.{table_name}")
    return {str(row["fact_id"]) for row in rows}


def _filter_turn_results(
    conn: sqlite3.Connection,
    history_id: str,
    turns: Sequence[RetrievedTurn],
    *,
    gated: bool = True,
) -> list[RetrievedTurn]:
    eligible = _eligible_turn_ids(
        conn, history_id, [turn.turn_id for turn in turns], gated=gated
    )
    return [turn for turn in turns if turn.turn_id in eligible]


def _revalidate_fact_results(
    conn: sqlite3.Connection,
    history_id: str,
    facts: Sequence[RetrievedFact],
    *,
    view: FactView,
    as_of: object | None,
    entity: str | None,
    attribute: str | None,
) -> list[RetrievedFact]:
    if not facts:
        return []
    eligible = _fact_rows_for_view(
        conn, history_id, view=view, as_of=as_of,
        entity=entity, attribute=attribute,
    )
    candidates: list[RetrievedFact] = []
    required_turn_ids: list[str] = []
    required_fact_ids: list[str] = []
    for result in facts:
        selected = eligible.get(result.fact_id)
        if selected is None:
            continue
        parent = (
            selected.get("parent_fact")
            if selected.get("key_type") == "trigger"
            else selected
        )
        if not isinstance(parent, Mapping):
            continue
        if result.parent_fact is not None and str(parent.get("fact_id")) != str(
            result.parent_fact.get("fact_id")
        ):
            continue
        required_fact_ids.append(result.fact_id)
        required_fact_ids.append(str(parent.get("fact_id")))
        required_turn_ids.extend(turn.turn_id for turn in result.supporting_turns)
        if selected.get("turn_id"):
            required_turn_ids.append(str(selected["turn_id"]))
        source_id = parent.get("turn_id")
        confirmed_id = parent.get("last_confirmed_turn_id")
        if source_id:
            required_turn_ids.append(str(source_id))
        if confirmed_id:
            required_turn_ids.append(str(confirmed_id))
        candidates.append(result)
    eligible_turns = _eligible_turn_ids(
        conn, history_id, required_turn_ids
    )
    eligible_fact_ids = _eligible_fact_ids(
        conn, history_id, required_fact_ids
    )
    output: list[RetrievedFact] = []
    for result in candidates:
        selected = eligible[result.fact_id]
        parent = selected.get("parent_fact") if selected.get("key_type") == "trigger" else selected
        assert isinstance(parent, Mapping)
        needed = {turn.turn_id for turn in result.supporting_turns}
        source_id = parent.get("turn_id")
        confirmed_id = parent.get("last_confirmed_turn_id")
        if source_id:
            needed.add(str(source_id))
        if confirmed_id:
            needed.add(str(confirmed_id))
        needed_facts = {result.fact_id, str(parent.get("fact_id"))}
        if selected.get("turn_id"):
            needed.add(str(selected["turn_id"]))
        if needed.issubset(eligible_turns) and needed_facts.issubset(eligible_fact_ids):
            output.append(result)
    return output


def _revalidate_results(
    conn: sqlite3.Connection,
    history_id: str,
    turns: Sequence[RetrievedTurn],
    facts: Sequence[RetrievedFact],
    *,
    view: FactView,
    as_of: object | None,
    entity: str | None,
    attribute: str | None,
    gated: bool = True,
) -> tuple[list[RetrievedTurn], list[RetrievedFact]]:
    return (
        _filter_turn_results(conn, history_id, turns, gated=gated),
        _revalidate_fact_results(
            conn, history_id, facts, view=view, as_of=as_of,
            entity=entity, attribute=attribute,
        ),
    )


def _encode_query(provider: EmbeddingProvider, query: str) -> np.ndarray:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string for dense retrieval")
    return normalize_vector(provider.encode_query(query), dimension=provider.dimension)


def _candidate_id(candidate: object) -> str:
    if isinstance(candidate, str):
        return candidate
    ref_id = getattr(candidate, "ref_id", None)
    if not isinstance(ref_id, str) or not ref_id:
        raise TypeError("RRF candidates must be IDs or objects with a non-empty ref_id")
    return ref_id


def _validate_scope(history_id: str, limit: int) -> None:
    if not isinstance(history_id, str) or not history_id:
        raise ValueError("history_id is required")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("result limits must be positive integers")


def _validate_view(view: FactView, as_of: object | None) -> None:
    if view not in ("current", "as_of", "history"):
        raise ValueError("view must be 'current', 'as_of', or 'history'")
    if view == "as_of" and as_of is None:
        raise ValueError("as_of is required when view='as_of'")
    if view != "as_of" and as_of is not None:
        raise ValueError("as_of can only be supplied when view='as_of'")
