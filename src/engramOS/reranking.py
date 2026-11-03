"""Candidate-only reranking for Phase 4A retrieval results.

Reranking can change ordering only. Providers receive immutable snapshots and
typed identifiers; validated output is applied to the original candidate
objects. Any provider, timeout, or validation failure returns the original
retrieval result unchanged. History views are returned unchanged to preserve
their complete temporal timeline.
"""

from __future__ import annotations

from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, replace
import logging
import math
from types import MappingProxyType
from threading import BoundedSemaphore, Thread
from typing import Literal, Mapping, Protocol

from .retrieval import RetrievalResult, RetrievedFact, RetrievedTurn

CandidateKind = Literal["turn", "fact"]
RerankingStatus = Literal["reranked", "skipped", "fallback"]
_Primitive = str | int | float | bool | None
_logger = logging.getLogger(__name__)
_MAX_CONCURRENT_RERANKERS = 4
_RERANKER_WORKER_SLOTS = BoundedSemaphore(_MAX_CONCURRENT_RERANKERS)


@dataclass(frozen=True, slots=True)
class CandidateId:
    """Immutable typed identity; equal raw IDs of different kinds stay distinct."""

    kind: CandidateKind
    value: str

    def __post_init__(self) -> None:
        if type(self.kind) is not str or self.kind not in ("turn", "fact"):
            raise ValueError("candidate kind must be 'turn' or 'fact'")
        if type(self.value) is not str or not self.value.strip():
            raise ValueError("candidate value must be a non-empty string")

    @property
    def stable_key(self) -> tuple[str, str]:
        """Return a deterministic typed key for tie breaking and diagnostics."""
        return (self.kind, self.value)

    def __str__(self) -> str:
        return f"{self.kind}:{self.value}"


@dataclass(frozen=True, slots=True)
class RerankCandidate:
    """Read-only candidate snapshot supplied to an injected reranker."""

    candidate_id: CandidateId
    original_rank: int
    text: str
    metadata: Mapping[str, _Primitive]


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """Provider output assigning a numeric relevance score to one candidate."""

    candidate_id: CandidateId
    score: int | float


class CandidateReranker(Protocol):
    """Provider-independent synchronous reranker contract.

    Implementations should honor ``timeout`` themselves. The adapter also
    enforces a caller-visible timeout and ignores any late provider result.
    """

    def rerank(
        self,
        query: str,
        candidates: tuple[RerankCandidate, ...],
        *,
        timeout: float,
    ) -> tuple[CandidateScore, ...] | list[CandidateScore]:
        """Return one score for every submitted candidate identifier."""
        ...


@dataclass(frozen=True, slots=True)
class RerankingOutcome:
    """Retrieval result and a content-free diagnostic for the reranking step."""

    result: RetrievalResult
    status: RerankingStatus
    reason: str
    ordered_candidate_ids: tuple[CandidateId, ...]


class _InvalidOutput(ValueError):
    pass


class _ProviderTimedOut(TimeoutError):
    pass


class _ProviderCapacityExhausted(RuntimeError):
    pass


class _ProviderRaised(Exception):
    def __init__(self, error_type: str) -> None:
        super().__init__(error_type)
        self.error_type = error_type


def rerank_retrieval_result(
    query: str,
    result: RetrievalResult,
    reranker: CandidateReranker,
    *,
    timeout: float = 5.0,
) -> RerankingOutcome:
    """Rerank an existing Phase 4A result without changing its evidence.

    Only ``current`` and ``as_of`` results are eligible for reordering. A
    ``history`` result is returned as-is because its fact tuple is a complete,
    chronologically ordered timeline. Candidate IDs are unique across the
    combined turn/fact pool or the operation is skipped. Invalid provider
    output is rejected as a whole and falls back to the exact input object.

    ``timeout`` must be a positive finite Python integer or float. Timeout
    expiration returns promptly; a non-cooperative provider may finish in its
    daemon worker later, but it only has immutable candidate snapshots. At most
    four provider workers can be active at once; saturated calls fail closed
    without creating or queueing another worker.
    """
    _validate_timeout(timeout)
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not isinstance(result, RetrievalResult):
        raise TypeError("result must be a RetrievalResult")

    original_ids = _candidate_ids_for_diagnostics(result)
    if result.view == "history":
        return RerankingOutcome(result, "skipped", "history_timeline_preserved", original_ids)
    if not result.turns and not result.facts:
        return RerankingOutcome(result, "skipped", "no_candidates", ())

    try:
        candidates = _candidate_snapshots(result)
    except (TypeError, ValueError, KeyError, AttributeError):
        return _fallback(result, original_ids, "invalid_candidate_snapshot")

    submitted_ids = tuple(candidate.candidate_id for candidate in candidates)
    if len(set(submitted_ids)) != len(submitted_ids):
        return RerankingOutcome(
            result, "skipped", "duplicate_input_candidate_ids", original_ids
        )

    try:
        raw_output = _invoke_with_timeout(
            reranker, query, candidates, timeout=timeout
        )
    except _ProviderTimedOut:
        return _fallback(result, original_ids, "provider_timeout")
    except _ProviderCapacityExhausted:
        return _fallback(result, original_ids, "provider_capacity_exhausted")
    except _ProviderRaised as exc:
        return _fallback(result, original_ids, "provider_exception", exc.error_type)
    except Exception as exc:
        return _fallback(
            result, original_ids, "provider_exception", type(exc).__name__
        )

    try:
        scores = _validate_scores(raw_output, candidates)
    except _InvalidOutput as exc:
        return _fallback(result, original_ids, str(exc))
    except Exception as exc:
        return _fallback(result, original_ids, "malformed_output", type(exc).__name__)

    ordered_ids = tuple(
        item.candidate_id
        for item in sorted(
            scores,
            key=lambda item: (
                -item.score,
                item.original_rank,
                item.candidate_id.stable_key,
            ),
        )
    )
    original_by_id = {
        candidate.candidate_id: source
        for candidate, source in _paired_candidates(result, candidates)
    }
    reordered = replace(
        result,
        turns=tuple(
            original_by_id[candidate_id]
            for candidate_id in ordered_ids
            if candidate_id.kind == "turn"
        ),
        facts=tuple(
            original_by_id[candidate_id]
            for candidate_id in ordered_ids
            if candidate_id.kind == "fact"
        ),
    )
    return RerankingOutcome(reordered, "reranked", "validated_permutation", ordered_ids)


def _validate_timeout(timeout: float) -> None:
    if type(timeout) not in (int, float):
        raise TypeError("timeout must be a positive finite int or float")
    try:
        valid = math.isfinite(timeout) and timeout > 0
    except (OverflowError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("timeout must be a positive finite int or float")


def _candidate_snapshots(
    result: RetrievalResult,
) -> tuple[RerankCandidate, ...]:
    snapshots: list[RerankCandidate] = []
    for rank, turn in enumerate(result.turns):
        if (
            not isinstance(turn, RetrievedTurn)
            or type(turn.turn_id) is not str
            or type(turn.raw_content) is not str
        ):
            raise TypeError("invalid turn candidate")
        candidate_id = CandidateId("turn", turn.turn_id)
        metadata = _readonly_metadata(
            {
                "history_id": turn.history_id,
                "session_id": turn.session_id,
                "turn_idx": turn.turn_idx,
                "role": turn.role,
                "source_type": turn.source_type,
                "event_time": turn.event_time,
                "recorded_at": turn.recorded_at,
                "original_rank": rank,
            }
        )
        snapshots.append(
            RerankCandidate(candidate_id, rank, turn.raw_content, metadata)
        )

    for rank, fact_result in enumerate(result.facts):
        if not isinstance(fact_result, RetrievedFact):
            raise TypeError("invalid fact candidate")
        fact_id = fact_result.fact.get("fact_id")
        fact_text = fact_result.fact.get("key_text")
        if type(fact_id) is not str or type(fact_text) is not str:
            raise ValueError("fact candidate requires string fact_id and key_text")
        candidate_id = CandidateId("fact", fact_id)
        source = fact_result.source_turn
        confirmation = fact_result.confirmation_turn
        metadata = _readonly_metadata(
            {
                "history_id": fact_result.fact.get("history_id"),
                "entity": fact_result.fact.get("entity"),
                "attribute": fact_result.fact.get("attribute"),
                "cardinality": fact_result.fact.get("cardinality"),
                "key_type": fact_result.fact.get("key_type"),
                "parent_fact_id": fact_result.fact.get("parent_fact_id"),
                "valid_from": fact_result.fact.get("valid_from"),
                "valid_until": fact_result.fact.get("valid_until"),
                "source_turn_id": source.turn_id,
                "source_event_time": source.event_time,
                "source_recorded_at": source.recorded_at,
                "confirmation_turn_id": (
                    confirmation.turn_id if confirmation is not None else None
                ),
                "confirmation_event_time": (
                    confirmation.event_time if confirmation is not None else None
                ),
                "original_rank": rank,
            }
        )
        snapshots.append(
            RerankCandidate(candidate_id, rank, fact_text, metadata)
        )
    return tuple(snapshots)


def _readonly_metadata(values: Mapping[str, object]) -> Mapping[str, _Primitive]:
    copied: dict[str, _Primitive] = {}
    for key, value in values.items():
        if value is None or type(value) in (str, int, bool):
            copied[key] = value  # type: ignore[assignment]
        elif type(value) is float and math.isfinite(value):
            copied[key] = value
    return MappingProxyType(copied)


def _paired_candidates(
    result: RetrievalResult,
    candidates: tuple[RerankCandidate, ...],
) -> tuple[tuple[RerankCandidate, RetrievedTurn | RetrievedFact], ...]:
    return tuple(zip(candidates, (*result.turns, *result.facts), strict=True))


def _invoke_with_timeout(
    reranker: CandidateReranker,
    query: str,
    candidates: tuple[RerankCandidate, ...],
    *,
    timeout: float,
) -> object:
    if not _RERANKER_WORKER_SLOTS.acquire(blocking=False):
        raise _ProviderCapacityExhausted

    try:
        future: Future[object] = Future()

        def run_provider() -> None:
            try:
                output = reranker.rerank(query, candidates, timeout=timeout)
            except BaseException as exc:
                # Do not let provider-raised SystemExit/KeyboardInterrupt escape
                # the worker thread or terminate the host application.
                future.set_exception(_ProviderRaised(type(exc).__name__))
            else:
                future.set_result(output)
            finally:
                _RERANKER_WORKER_SLOTS.release()

        worker = Thread(target=run_provider, name="engram-rerank", daemon=True)
        worker.start()
    except BaseException:
        _RERANKER_WORKER_SLOTS.release()
        raise
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError as exc:
        raise _ProviderTimedOut from exc


@dataclass(frozen=True, slots=True)
class _ValidatedScore:
    candidate_id: CandidateId
    score: float
    original_rank: int


def _validate_scores(
    output: object,
    candidates: tuple[RerankCandidate, ...],
) -> tuple[_ValidatedScore, ...]:
    # Restrict provider output to built-in finite containers so validation does
    # not invoke lazy or user-defined iteration outside the provider timeout.
    if type(output) not in (tuple, list):
        raise _InvalidOutput("malformed_output")

    submitted = {candidate.candidate_id: candidate for candidate in candidates}
    seen: set[CandidateId] = set()
    validated: list[_ValidatedScore] = []
    for item in output:
        if type(item) is not CandidateScore:
            raise _InvalidOutput("malformed_output")
        raw_candidate_id = item.candidate_id
        if not _is_valid_candidate_id(raw_candidate_id):
            raise _InvalidOutput("invalid_candidate_id_type")
        # Canonicalize the provider's identifier into a fresh immutable value.
        candidate_id = CandidateId(raw_candidate_id.kind, raw_candidate_id.value)
        if candidate_id not in submitted:
            raise _InvalidOutput("unknown_candidate_id")
        if candidate_id in seen:
            raise _InvalidOutput("duplicate_candidate_id")
        score = item.score
        if type(score) not in (int, float):
            raise _InvalidOutput("invalid_score_type")
        try:
            numeric_score = float(score)
        except (OverflowError, TypeError, ValueError):
            raise _InvalidOutput("invalid_score_type") from None
        if not math.isfinite(numeric_score):
            raise _InvalidOutput("non_finite_score")
        seen.add(candidate_id)
        validated.append(
            _ValidatedScore(candidate_id, numeric_score, submitted[candidate_id].original_rank)
        )

    if len(output) != len(candidates) or seen != set(submitted):
        raise _InvalidOutput("candidate_set_mismatch")
    return tuple(validated)


def _is_valid_candidate_id(candidate_id: object) -> bool:
    try:
        return (
            type(candidate_id) is CandidateId
            and type(candidate_id.kind) is str
            and candidate_id.kind in ("turn", "fact")
            and type(candidate_id.value) is str
            and bool(candidate_id.value.strip())
        )
    except Exception:
        return False


def _candidate_ids_for_diagnostics(
    result: RetrievalResult,
) -> tuple[CandidateId, ...]:
    try:
        return tuple(
            [CandidateId("turn", turn.turn_id) for turn in result.turns]
            + [
                CandidateId("fact", fact.fact["fact_id"])
                for fact in result.facts
            ]
        )
    except (TypeError, ValueError, KeyError, AttributeError):
        return ()


def _fallback(
    result: RetrievalResult,
    original_ids: tuple[CandidateId, ...],
    reason: str,
    error_type: str | None = None,
) -> RerankingOutcome:
    if error_type is None:
        _logger.warning("Reranking fallback: %s", reason)
    else:
        _logger.warning(
            "Reranking fallback: %s (error_type=%s)", reason, error_type
        )
    return RerankingOutcome(result, "fallback", reason, original_ids)
