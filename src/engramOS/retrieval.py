"""Quarantine-aware lexical, dense, temporal, and fused retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
import sqlite3
import time
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
from .temporal import FactView, TemporalBoundary, _retrieve_facts, as_of_timestamp
from .query_classifier import TemporalRange
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


@dataclass(frozen=True, slots=True)
class RehydratedCandidates:
    """Fresh candidate text and the start time of its authorizing DB snapshot."""

    candidates: tuple[RetrievedTurn | RetrievedFact, ...]
    authorization_timestamp: float


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
    event_time_range: TemporalRange | None = None,
) -> list[FtsCandidate]:
    match_query = safe_fts_query(query)
    if not match_query:
        return []
    range_sql, range_params = _event_time_range_sql("t.event_time", event_time_range)
    rows = conn.execute(
        """SELECT t.*, bm25(turns_fts) AS lexical_score
           FROM turns_fts
           JOIN turns AS t ON t.rowid = turns_fts.rowid
           WHERE turns_fts MATCH ? AND t.history_id = ? AND """
        + turn_eligibility_predicate("t", gated=gated) + range_sql
        + " ORDER BY lexical_score ASC, t.turn_id ASC LIMIT ?",
        (match_query, history_id, *range_params, limit),
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


def _fuse_ranked_lists(
    channels: Mapping[str, Sequence[object]],
    *,
    k: int,
    limit: int | None,
) -> list[FusedRank]:
    """Fuse multiple available lists; preserve a lone channel's exact ranking."""
    available = {name: values for name, values in channels.items() if values}
    if len(available) > 1:
        return reciprocal_rank_fusion(available, k=k, limit=limit)
    if not available:
        return []
    name, values = next(iter(available.items()))
    output: list[FusedRank] = []
    seen: set[str] = set()
    for rank, candidate in enumerate(values, start=1):
        ref_id = _candidate_id(candidate)
        if ref_id in seen:
            continue
        seen.add(ref_id)
        output.append(FusedRank(ref_id, 0.0, (name,), {name: rank}))
        if limit is not None and len(output) >= limit:
            break
    return output


def hybrid_rag(
    database: str | Path,
    history_id: str,
    query: str,
    embedding_provider: EmbeddingProvider | None,
    *,
    top_k: int = 10,
    candidate_limit: int = 50,
    rrf_k: int = 60,
    timeout: float = 5.0,
    gated: bool = True,
    event_time_range: TemporalRange | None = None,
) -> RetrievalResult:
    """Return dense+BM25 RRF turns, optionally prefiltered by event time."""
    return _retrieve_turns(
        database, history_id, query, embedding_provider,
        top_k=top_k, candidate_limit=candidate_limit, rrf_k=rrf_k,
        timeout=timeout, baseline="hybrid_rag", gated=gated,
        event_time_range=event_time_range,
    )


def recency(
    database: str | Path,
    history_id: str,
    query: str,
    embedding_provider: EmbeddingProvider | None,
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
    embedding_provider: EmbeddingProvider | None,
    *,
    view: FactView = "current",
    as_of: object | None = None,
    history_range: TemporalRange | None = None,
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
    _validate_scope(history_id, top_k)
    _validate_scope(history_id, fact_top_k)
    _validate_scope(history_id, turn_candidate_limit)
    _validate_scope(history_id, fact_candidate_limit)
    _validate_view(view, as_of)
    _validate_history_range(view, history_range)
    turn_event_range = (
        TemporalRange(end=(
            as_of if isinstance(as_of, TemporalBoundary)
            else TemporalBoundary(as_of_timestamp(as_of))
        ))
        if view == "as_of"
        else history_range
    )
    if embedding_provider is None:
        return _retrieve_memory_lexical(
            database, history_id, query,
            view=view, as_of=as_of, history_range=history_range,
            top_k=top_k, fact_top_k=fact_top_k,
            turn_candidate_limit=turn_candidate_limit,
            fact_candidate_limit=fact_candidate_limit,
            rrf_k=rrf_k, entity=entity, attribute=attribute,
            timeout=timeout,
        )
    validate_embedding_provider(embedding_provider)
    query_vector = _encode_query(embedding_provider, query)
    with connection(database, timeout=timeout) as conn:
        conn.execute("BEGIN")
        _require_fresh_index(
            conn, history_id, embedding_provider.model_id, ("turn", "fact")
        )
        turn_fts = _search_turns_fts(
            conn, history_id, query, limit=turn_candidate_limit,
            event_time_range=turn_event_range,
        )
        turn_index = InMemoryVectorIndex.load_from_connection(
            conn, history_id, "turn", embedding_provider.model_id,
            dimension=embedding_provider.dimension, validate_freshness=False,
        )
        if turn_index.size == 0 and turn_event_range is None:
            raise DenseIndexUnavailable(
                f"Hybrid retrieval requires turn embeddings for history {history_id!r} "
                f"and model {embedding_provider.model_id!r}; run index_history first"
            )
        allowed_turn_ids = _turn_ids_for_event_range(
            conn, history_id, turn_event_range
        )
        turn_dense = turn_index.search(
            query_vector, limit=turn_candidate_limit, allowed_ids=allowed_turn_ids
        )
        if not turn_dense and turn_event_range is None:
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
        fact_fused = _fuse_ranked_lists(
            {"fact_lexical": fact_fts, "fact_dense": fact_dense},
            k=rrf_k, limit=fact_top_k,
        )
        if view == "history":
            # History is a timeline, not a top-k semantic ranking. The lexical
            # and dense fact channels still report query candidates in diagnostics.
            timeline = list(_retrieve_temporal_facts(
                conn, history_id, view="history", as_of=None,
                entity=entity, attribute=attribute
            ))
            fused_facts = _select_history_range(timeline, history_range)
        else:
            fused_facts = _expand_fact_evidence(
                conn, history_id, fact_fused, eligible_facts, fact_fts, fact_dense,
            )
        turns, fused_facts = _revalidate_results(
            conn, history_id, turns, fused_facts,
            view=view, as_of=as_of, entity=entity, attribute=attribute,
        )
        if turn_event_range is not None:
            turns = [turn for turn in turns if _in_range(turn.event_time, turn_event_range)]
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


def _retrieve_memory_lexical(
    database: str | Path,
    history_id: str,
    query: str,
    *,
    view: FactView,
    as_of: object | None,
    history_range: TemporalRange | None,
    top_k: int,
    fact_top_k: int,
    turn_candidate_limit: int,
    fact_candidate_limit: int,
    rrf_k: int,
    entity: str | None,
    attribute: str | None,
    timeout: float,
) -> RetrievalResult:
    """Retrieve from FTS5 and temporal projections without touching a vector index."""
    turn_event_range = (
        TemporalRange(end=(
            as_of if isinstance(as_of, TemporalBoundary)
            else TemporalBoundary(as_of_timestamp(as_of))
        ))
        if view == "as_of" else history_range
    )
    with connection(database, timeout=timeout) as conn:
        conn.execute("BEGIN")
        turn_fts = _search_turns_fts(
            conn, history_id, query, limit=turn_candidate_limit,
            event_time_range=turn_event_range,
        )
        turns = _fuse_turns(turn_fts, (), top_k=top_k, rrf_k=rrf_k)
        fact_fts = _search_facts_fts(
            conn, history_id, query, view=view, as_of=as_of,
            limit=fact_candidate_limit, entity=entity, attribute=attribute,
        )
        eligible_facts = _fact_rows_for_view(
            conn, history_id, view=view, as_of=as_of,
            entity=entity, attribute=attribute,
        )
        fact_fused = _fuse_ranked_lists(
            {"fact_lexical": fact_fts}, k=rrf_k, limit=fact_top_k,
        )
        if view == "history":
            timeline = list(_retrieve_temporal_facts(
                conn, history_id, view="history", as_of=None,
                entity=entity, attribute=attribute,
            ))
            fused_facts = _select_history_range(timeline, history_range)
        else:
            fused_facts = _expand_fact_evidence(
                conn, history_id, fact_fused, eligible_facts, fact_fts, (),
            )
        turns, fused_facts = _revalidate_results(
            conn, history_id, turns, fused_facts,
            view=view, as_of=as_of, entity=entity, attribute=attribute,
        )
        if turn_event_range is not None:
            turns = [turn for turn in turns if _in_range(turn.event_time, turn_event_range)]
        diagnostics = RetrievalDiagnostics(
            {
                "turn_lexical": len(turn_fts), "turn_dense": 0,
                "fact_lexical": len(fact_fts), "fact_dense": 0,
            },
            "none", 0, "none", rrf_k, turn_candidate_limit,
            fact_candidate_limit, len(turns), len(fused_facts),
        )
        return RetrievalResult(
            history_id, view, "memory", tuple(turns), tuple(fused_facts), diagnostics,
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


def rehydrate_reader_candidates(
    database: str | Path,
    history_id: str,
    result: RetrievalResult,
    ordered_ids: Sequence[tuple[str, str]],
    *,
    as_of: object | None = None,
    history_range: TemporalRange | None = None,
    authorization_timestamp: float | None = None,
    timeout: float = 5.0,
) -> RehydratedCandidates:
    """Revalidate only the original IDs and rebuild their text/provenance in a fresh snapshot."""
    if result.history_id != history_id:
        raise ValueError("retrieval result is outside the requested history")
    _validate_view(result.view, as_of)
    _validate_history_range(result.view, history_range)
    turn_range = (
        TemporalRange(end=(
            as_of if isinstance(as_of, TemporalBoundary)
            else TemporalBoundary(as_of_timestamp(as_of))
        ))
        if result.view == "as_of" else history_range
    )
    original_turns = {item.turn_id: item for item in result.turns}
    original_facts = {item.fact_id: item for item in result.facts}
    if len(original_turns) != len(result.turns) or len(original_facts) != len(result.facts):
        raise ValueError("retrieval result contains duplicate candidate IDs")
    requested_turns = [value for kind, value in ordered_ids if kind == "turn"]
    requested_facts = [value for kind, value in ordered_ids if kind == "fact"]
    if (len(set(ordered_ids)) != len(ordered_ids)
            or any(value not in original_turns for value in requested_turns)
            or any(value not in original_facts for value in requested_facts)
            or any(kind not in ("turn", "fact") for kind, _ in ordered_ids)):
        raise ValueError("reader candidate IDs must be a unique subset of the retrieval result")

    with connection(database, timeout=timeout) as conn:
        conn.execute("BEGIN")
        conn.execute(
            "SELECT 1 FROM turns WHERE history_id = ? LIMIT 1", (history_id,)
        ).fetchone()
        authorization_timestamp = (
            time.time() if authorization_timestamp is None else authorization_timestamp
        )
        eligible_turn_ids = _eligible_turn_ids(conn, history_id, requested_turns)
        if result.view == "current" and eligible_turn_ids:
            eligible_turn_ids -= _non_authoritative_conflict_turn_ids(
                conn, history_id, eligible_turn_ids
            )
        if turn_range is not None:
            eligible_turn_ids &= _turn_ids_for_event_range(conn, history_id, turn_range)
        rows_by_id: dict[str, sqlite3.Row] = {}
        if eligible_turn_ids:
            ids = tuple(eligible_turn_ids)
            marks = ",".join("?" for _ in ids)
            rows_by_id = {
                str(row["turn_id"]): row
                for row in conn.execute(
                    "SELECT t.* FROM turns AS t WHERE t.history_id = ? AND t.turn_id IN ("
                    + marks + ") AND " + turn_eligibility_predicate("t"),
                    (history_id, *ids),
                ).fetchall()
            }
        turns_by_id: dict[str, RetrievedTurn] = {}
        for turn_id in requested_turns:
            old = original_turns[turn_id]
            row = rows_by_id.get(turn_id)
            if row is not None:
                turns_by_id[turn_id] = RetrievedTurn(
                    turn_id, history_id, str(row["session_id"]), int(row["turn_idx"]),
                    str(row["role"]), str(row["raw_content"]), str(row["source_type"]),
                    float(row["event_time"]), float(row["recorded_at"]),
                    old.dense_score, old.bm25_score, old.rrf_score, old.channels,
                )

        selected_facts = _fact_rows_for_view(
            conn, history_id, view=result.view, as_of=as_of,
            entity=None, attribute=None,
        )
        if result.view == "history":
            selected_facts = {
                key: value for key, value in selected_facts.items()
                if _history_fact_selected(value, history_range)
            }
        fact_ranks = []
        for fact_id in requested_facts:
            if fact_id not in selected_facts:
                continue
            old = original_facts[fact_id]
            fact_ranks.append(FusedRank(fact_id, old.rrf_score, old.channels, {}))
        rebuilt_facts = _expand_fact_evidence(
            conn, history_id, fact_ranks, selected_facts, (), ()
        )
        rebuilt_by_id = {item.fact_id: item for item in rebuilt_facts}
        facts_by_id: dict[str, RetrievedFact] = {}
        for fact_id in requested_facts:
            fresh = rebuilt_by_id.get(fact_id)
            if fresh is None:
                continue
            old = original_facts[fact_id]
            facts_by_id[fact_id] = RetrievedFact(
                fresh.fact, fresh.parent_fact, fresh.supporting_turns,
                fresh.source_turn, fresh.confirmation_turn,
                ("[baseline before requested history range]"
                 if old.annotation == "[baseline before requested history range]"
                 else fresh.annotation),
                old.dense_score, old.bm25_score, old.rrf_score, old.channels,
            )
        candidates = tuple(
            turns_by_id[value] if kind == "turn" else facts_by_id[value]
            for kind, value in ordered_ids
            if (value in turns_by_id if kind == "turn" else value in facts_by_id)
        )
        return RehydratedCandidates(candidates, authorization_timestamp)


def _retrieve_turns(
    database: str | Path,
    history_id: str,
    query: str,
    provider: EmbeddingProvider | None,
    *,
    top_k: int,
    candidate_limit: int,
    rrf_k: int,
    timeout: float,
    baseline: Literal["hybrid_rag", "recency"],
    gated: bool = True,
    event_time_range: TemporalRange | None = None,
) -> RetrievalResult:
    _validate_scope(history_id, top_k)
    _validate_scope(history_id, candidate_limit)
    if provider is None:
        with connection(database, timeout=timeout) as conn:
            conn.execute("BEGIN")
            lexical = _search_turns_fts(
                conn, history_id, query, limit=candidate_limit, gated=gated,
                event_time_range=event_time_range,
            )
            turns = _fuse_turns(lexical, (), top_k=top_k, rrf_k=rrf_k)
            diagnostics = RetrievalDiagnostics(
                {"turn_lexical": len(lexical), "turn_dense": 0,
                 "fact_lexical": 0, "fact_dense": 0},
                "none", 0, "none", rrf_k, candidate_limit, 0,
                len(turns), 0, gated,
            )
            return RetrievalResult(
                history_id, "current", baseline, tuple(turns), (), diagnostics,
            )
    validate_embedding_provider(provider)
    query_vector = _encode_query(provider, query)
    with connection(database, timeout=timeout) as conn:
        conn.execute("BEGIN")
        _require_fresh_index(
            conn, history_id, provider.model_id, ("turn",), gated=gated
        )
        lexical = _search_turns_fts(
            conn, history_id, query, limit=candidate_limit, gated=gated,
            event_time_range=event_time_range,
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
        allowed_ids = _turn_ids_for_event_range(
            conn, history_id, event_time_range, gated=gated
        )
        dense = index.search(
            query_vector, limit=candidate_limit, allowed_ids=allowed_ids
        )
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
    fused = _fuse_ranked_lists(
        {"turn_lexical": lexical, "turn_dense": dense}, k=rrf_k, limit=top_k
    )
    lex_by_id = {item.ref_id: item for item in lexical}
    dense_by_id = {item.ref_id: item for item in dense}
    turns: list[RetrievedTurn] = []
    for item in fused:
        candidate = dense_by_id.get(item.ref_id) or lex_by_id.get(item.ref_id)
        if candidate is None:
            continue
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


def _non_authoritative_conflict_turn_ids(
    conn: sqlite3.Connection, history_id: str, turn_ids: set[str]
) -> set[str]:
    """Exclude conflicting external/tool turns from current Reader evidence.

    The turn and observation remain stored for audit/replay. This suppresses
    only a current-view Reader snapshot where a stored lower-authority claim
    conflicts with an active value that has a user-turn source or confirmation.
    """
    if not turn_ids:
        return set()
    marks = ",".join("?" for _ in turn_ids)
    candidate_eligible = turn_eligibility_predicate("candidate_turn")
    active_eligible = fact_eligibility_predicate("active_fact")
    confirmed_by_user = turn_eligibility_predicate("user_turn")
    query = f"""SELECT DISTINCT observation.turn_id
        FROM fact_observations AS observation
        JOIN turns AS candidate_turn ON candidate_turn.turn_id = observation.turn_id
        JOIN facts AS active_fact
          ON active_fact.history_id = observation.history_id
         AND active_fact.entity = observation.entity
         AND active_fact.attribute = observation.attribute
        WHERE observation.history_id = ?
          AND observation.turn_id IN ({marks})
          AND observation.cardinality = 'single'
          AND observation.quarantined = 0
          AND observation.canonical_fact_id IS NULL
          AND observation.value_norm <> active_fact.value_norm
          AND candidate_turn.source_type IN ('tool', 'external_doc')
          AND {candidate_eligible}
          AND {active_eligible}
          AND active_fact.key_type = 'slot'
          AND active_fact.cardinality = 'single'
          AND active_fact.valid_until IS NULL
          AND EXISTS (
            SELECT 1
            FROM fact_observations AS confirmation
            JOIN turns AS user_turn ON user_turn.turn_id = confirmation.turn_id
            WHERE confirmation.canonical_fact_id = active_fact.fact_id
              AND confirmation.value_norm = active_fact.value_norm
              AND confirmation.quarantined = 0
              AND user_turn.source_type = 'user_turn'
              AND {confirmed_by_user}
          )"""
    rows = conn.execute(query, (history_id, *turn_ids)).fetchall()
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


def _validate_history_range(view: FactView, history_range: TemporalRange | None) -> None:
    if history_range is not None and not isinstance(history_range, TemporalRange):
        raise TypeError("history_range must be a TemporalRange or None")
    if history_range is not None and view != "history":
        raise ValueError("history_range can only be supplied when view='history'")


def _event_time_range_sql(
    column: str, event_range: TemporalRange | None
) -> tuple[str, tuple[float, ...]]:
    if event_range is None:
        return "", ()
    clauses: list[str] = []
    values: list[float] = []
    if event_range.start is not None:
        clauses.append(f"{column} {'>=' if event_range.start.inclusive else '>'} ?")
        values.append(event_range.start.timestamp)
    if event_range.end is not None:
        clauses.append(f"{column} {'<=' if event_range.end.inclusive else '<'} ?")
        values.append(event_range.end.timestamp)
    return (" AND " + " AND ".join(clauses) if clauses else ""), tuple(values)


def _turn_ids_for_event_range(
    conn: sqlite3.Connection,
    history_id: str,
    event_range: TemporalRange | None,
    *,
    gated: bool = True,
) -> set[str] | None:
    if event_range is None:
        return None
    range_sql, params = _event_time_range_sql("t.event_time", event_range)
    rows = conn.execute(
        "SELECT t.turn_id FROM turns AS t WHERE t.history_id = ? AND "
        + turn_eligibility_predicate("t", gated=gated) + range_sql,
        (history_id, *params),
    ).fetchall()
    return {str(row["turn_id"]) for row in rows}


def _in_range(value: float, event_range: TemporalRange) -> bool:
    if event_range.start is not None:
        if value < event_range.start.timestamp or (
            value == event_range.start.timestamp and not event_range.start.inclusive
        ):
            return False
    if event_range.end is not None:
        if value > event_range.end.timestamp or (
            value == event_range.end.timestamp and not event_range.end.inclusive
        ):
            return False
    return True


def _history_fact_selected(
    fact: Mapping[str, object], history_range: TemporalRange | None
) -> bool:
    valid_from = float(fact.get("valid_from", 0.0))
    if history_range is None:
        return True
    if history_range.start is not None and (
        valid_from > history_range.start.timestamp
        or (valid_from == history_range.start.timestamp and history_range.start.inclusive)
    ) and (
        history_range.end is None
        or valid_from < history_range.end.timestamp
        or (valid_from == history_range.end.timestamp and history_range.end.inclusive)
    ):
        return True
    if history_range.start is not None and _is_history_baseline(fact, history_range.start):
        valid_until = fact.get("valid_until")
        if valid_until is None or (
            float(valid_until) >= history_range.start.timestamp
            if history_range.start.inclusive
            else float(valid_until) > history_range.start.timestamp
        ):
            return True
    if history_range.start is None:
        return _in_range(valid_from, history_range)
    if valid_from < history_range.start.timestamp:
        return False
    return _in_range(valid_from, history_range)


def _select_history_range(
    timeline: Sequence[RetrievedFact], history_range: TemporalRange | None
) -> list[RetrievedFact]:
    if history_range is None:
        return list(timeline)
    selected = [
        item for item in timeline
        if _in_range(float(item.fact.get("valid_from", 0.0)), history_range)
    ]
    if history_range.start is None:
        return selected
    baselines = [
        item for item in timeline
        if _is_history_baseline(item.fact, history_range.start)
        and (item.fact.get("valid_until") is None or (
            float(item.fact["valid_until"]) >= history_range.start.timestamp
            if history_range.start.inclusive
            else float(item.fact["valid_until"]) > history_range.start.timestamp
        ))
    ]
    selected_ids = {item.fact_id for item in selected}
    baseline_by_slot: dict[tuple[str, str, str], RetrievedFact] = {}
    for item in sorted(
        baselines,
        key=lambda candidate: (
            str(candidate.fact.get("entity") or ""),
            str(candidate.fact.get("attribute") or ""),
            -float(candidate.fact.get("valid_from", 0.0)),
        ),
    ):
        slot = (
            str(item.fact.get("entity") or ""),
            str(item.fact.get("attribute") or ""),
            str(item.fact.get("value") or "")
            if item.fact.get("cardinality") == "multi" else "",
        )
        baseline_by_slot.setdefault(slot, item)
    output = list(selected)
    for item in baseline_by_slot.values():
        if item.fact_id not in selected_ids:
            output.append(RetrievedFact(
                item.fact, item.parent_fact, item.supporting_turns, item.source_turn,
                item.confirmation_turn, "[baseline before requested history range]",
                item.dense_score, item.bm25_score, item.rrf_score, item.channels,
            ))
    return sorted(
        output,
        key=lambda item: (
            float(item.fact.get("valid_from", 0.0)),
            str(item.fact.get("entity") or ""),
            str(item.fact.get("attribute") or ""), item.fact_id,
        ),
    )


def _is_history_baseline(
    fact: Mapping[str, object], start: TemporalBoundary
) -> bool:
    valid_from = float(fact.get("valid_from", 0.0))
    # The baseline is a state already in effect before the range starts.
    # A change exactly on an exclusive start is outside the requested range,
    # but it is not a pre-range change and must not be smuggled in as baseline.
    return valid_from < start.timestamp
