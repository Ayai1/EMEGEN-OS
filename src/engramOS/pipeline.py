"""Classify, retrieve, revalidate, and answer one memory query."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Sequence

from .query_classifier import (
    QueryClassification,
    QueryClassificationError,
    QueryClassificationOutputError,
    QueryClassificationProvider,
    classify_query,
)
from .embeddings import EmbeddingProvider
from .reader import (
    _PIPELINE_CONTEXT_AUTHORITY,
    ReaderContext,
    ReaderEvidence,
    ReaderOutcome,
    ReaderProvider,
    ReaderProvenance,
    _answer_from_context,
    _label_unverified_evidence,
    _snapshot_turn,
)
from .reranking import CandidateReranker, RerankingOutcome, rerank_retrieval_result
from .retrieval import (
    RetrievalResult,
    RetrievedFact,
    RetrievedTurn,
    rehydrate_reader_candidates,
    retrieve_memory,
    RetrievedTurn,
)


@dataclass(frozen=True, slots=True)
class QueryAnswerOutcome:
    """Pipeline result with classification and retrieval diagnostics when available."""

    status: Literal["answered", "abstained", "clarification_required", "unsupported"]
    reason_code: str
    classification: QueryClassification | None
    retrieval: RetrievalResult | None
    reranking: RerankingOutcome | None
    reader: ReaderOutcome | None


def answer_query(
    database: str | Path,
    history_id: str,
    query: str,
    embedding_provider: EmbeddingProvider,
    reader_provider: ReaderProvider | None,
    *,
    question_date: date | datetime | str,
    classifier: QueryClassificationProvider | None = None,
    reranker: CandidateReranker | None = None,
    timeout: float = 5.0,
    top_k: int = 10,
    fact_top_k: int = 10,
    turn_candidate_limit: int = 50,
    fact_candidate_limit: int = 50,
    context_version: Literal["v2", "v3"] = "v3",
) -> QueryAnswerOutcome:
    """Run the guarded Stage 4 flow; no answer is persisted to memory."""
    try:
        classification = classify_query(
            query, question_date=question_date, classifier=classifier, timeout=timeout
        )
    except (QueryClassificationError, QueryClassificationOutputError, TypeError, ValueError):
        return QueryAnswerOutcome(
            "abstained", "classification_failed", None, None, None, None
        )
    if not classification.ready_for_retrieval:
        return QueryAnswerOutcome(
            "clarification_required" if classification.status == "clarification_required" else "unsupported",
            classification.reason_code, classification, None, None, None,
        )

    assert classification.view is not None
    try:
        retrieval = retrieve_memory(
            database, history_id, query, embedding_provider,
            view=classification.view, as_of=classification.as_of,
            history_range=classification.history_range,
            top_k=top_k, fact_top_k=fact_top_k,
            turn_candidate_limit=turn_candidate_limit,
            fact_candidate_limit=fact_candidate_limit, timeout=timeout,
        )
    except Exception:
        failed = _reader_abstention("retrieval_failed")
        return QueryAnswerOutcome(
            "abstained", failed.reason_code, classification, None, None, failed
        )

    if reranker is None or retrieval.view == "history":
        reranking = RerankingOutcome(
            retrieval, "skipped",
            "history_timeline_preserved" if retrieval.view == "history" else "reranker_unavailable",
            _collection_ids(retrieval, history=retrieval.view == "history"),
        )
    else:
        try:
            reranking = rerank_retrieval_result(
                query, retrieval, reranker, timeout=timeout
            )
        except Exception:
            reranking = RerankingOutcome(
                retrieval, "fallback", "reranker_failed", _collection_ids(retrieval)
            )

    result = reranking.result
    if result.view == "history":
        ordered_ids = _collection_ids(result, history=True)
        ordering = "complete_chronological_fact_changes_with_labeled_baselines_then_supplemental_query_ranked_turns"
        timeline_complete = True
    elif reranking.status == "reranked":
        ordered_ids = tuple((item.kind, item.value) for item in reranking.ordered_candidate_ids)
        ordering = "validated_combined_reranker_order"
        timeline_complete = False
    else:
        ordered_ids = _collection_ids(result)
        ordering = "retrieval_turns_then_facts"
        timeline_complete = False

    try:
        materialized = rehydrate_reader_candidates(
            database, history_id, result, ordered_ids,
            as_of=classification.as_of,
            history_range=classification.history_range,
            timeout=timeout,
        )
        evidence = _label_unverified_evidence(tuple(
            _reader_evidence(item, classification)
            for item in materialized.candidates
        ), context_version)
        context = ReaderContext(
            query=query,
            classification=classification,
            history_id=history_id,
            evidence=evidence,
            ordering=ordering,
            timeline_complete=timeline_complete,
            turn_evidence_complete=False,
            authorization_timestamp=materialized.authorization_timestamp,
            _authorization=_PIPELINE_CONTEXT_AUTHORITY,
            context_version=context_version,
        )
    except Exception:
        failed = _reader_abstention("final_revalidation_failed")
        return QueryAnswerOutcome(
            "abstained", failed.reason_code, classification, retrieval, reranking, failed
        )

    reader = _answer_from_context(context, reader_provider, timeout=timeout)
    status = "answered" if not reader.abstained else "abstained"
    return QueryAnswerOutcome(
        status, reader.reason_code, classification, retrieval, reranking, reader
    )


def answer_turn_baseline_query(
    query: str,
    history_id: str,
    turns: Sequence[RetrievedTurn],
    classification: QueryClassification,
    reader_provider: ReaderProvider | None,
    *,
    ordering: Literal["relevance", "reverse_chronological"],
    timeout: float = 5.0,
    context_version: Literal["v2", "v3"] = "v2",
) -> ReaderOutcome:
    """Evaluation-only Reader route for already-retrieved turn baselines.

    This intentionally preserves each baseline's retrieval policy: it is
    ungated and unrevalidated here. Callers must provide the exact candidate
    turns to compare. Production callers should use :func:`answer_query`.
    """
    if ordering not in ("relevance", "reverse_chronological"):
        raise ValueError("unsupported baseline ordering label")
    evidence = _label_unverified_evidence(
        tuple(_snapshot_turn(turn) for turn in turns), context_version
    )
    authorization_timestamp = max(
        (float(turn.event_time) for turn in turns), default=0.0
    )
    context = ReaderContext(
        query=query,
        classification=classification,
        history_id=history_id,
        evidence=evidence,
        ordering=ordering,
        timeline_complete=False,
        turn_evidence_complete=False,
        authorization_timestamp=authorization_timestamp,
        _authorization=_PIPELINE_CONTEXT_AUTHORITY,
        context_version=context_version,
    )
    return _answer_from_context(context, reader_provider, timeout=timeout)


def _collection_ids(
    result: RetrievalResult, *, history: bool = False
) -> tuple[tuple[str, str], ...]:
    turns = tuple(("turn", item.turn_id) for item in result.turns)
    facts = tuple(("fact", item.fact_id) for item in result.facts)
    return facts + turns if history else turns + facts


def _reader_evidence(
    candidate: RetrievedTurn | RetrievedFact,
    classification: QueryClassification,
) -> ReaderEvidence:
    from .reader import _snapshot_fact, _snapshot_turn

    if isinstance(candidate, RetrievedTurn):
        return _snapshot_turn(candidate)
    snapshot = _snapshot_fact(candidate)
    if classification.view != "as_of" or classification.as_of is None:
        return snapshot
    cutoff = classification.as_of
    allowed_provenance: list[ReaderProvenance] = []
    retrospective = False
    valid_from = candidate.fact.get("valid_from")
    fact_recorded_at = candidate.fact.get("recorded_at")
    if (type(valid_from) in (int, float)
            and type(fact_recorded_at) in (int, float)
            and float(fact_recorded_at) > float(valid_from)):
        retrospective = True
    for source in snapshot.provenance:
        after_cutoff = (
            source.event_time > cutoff.timestamp
            or (source.event_time == cutoff.timestamp and not cutoff.inclusive)
        )
        if after_cutoff:
            allowed_provenance.append(replace(source, text=""))
        else:
            allowed_provenance.append(source)
        if type(valid_from) in (int, float) and (
            source.event_time > float(valid_from) or source.recorded_at > float(valid_from)
        ):
            retrospective = True
    metadata = dict(snapshot.metadata)
    metadata["retrospective"] = retrospective
    metadata["valid_time"] = candidate.fact.get("valid_from")
    metadata["recorded_at"] = candidate.fact.get("recorded_at")
    return ReaderEvidence(
        snapshot.evidence_id,
        snapshot.text,
        MappingProxyType(metadata),
        tuple(allowed_provenance),
    )


def _reader_abstention(reason: str) -> ReaderOutcome:
    return ReaderOutcome("I don't know.", (), True, reason)  # type: ignore[arg-type]
