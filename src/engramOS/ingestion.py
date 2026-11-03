"""Validated, recoverable turn ingestion and deterministic fact application."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import logging
import math
import sqlite3
import time as clock
from typing import Literal, Sequence

from .db import connection, transaction
from .eligibility import (
    fact_eligibility_predicate,
    turn_has_gate_decision,
    turn_has_quarantine_event,
    turn_is_eligible,
)
from .extraction import (
    ExtractedFact,
    ExtractionTurn,
    FactExtractor,
    build_extraction_prompt,
    extract_and_validate_with_bounded_retries,
    serialize_extraction,
    validate_extraction_response,
)
from .gate import SignalEvaluator, evaluate_and_record_turn
from .normalize import normalize_slot_name
from .provenance import (
    ALLOWED_ROLES,
    ALLOWED_SOURCE_TYPES,
    ProvenanceError,
    validate_role_source_type,
)
from .temporal import (
    FactInput,
    UngatedSourceTurnError,
    calculate_valid_from,
    insert_fact,
)

logger = logging.getLogger(__name__)

_ROLES = ALLOWED_ROLES
_SQLITE_INTEGER_MAX = (1 << 63) - 1
_SOURCE_TYPES = ALLOWED_SOURCE_TYPES
_DEFAULT_SOURCE_TYPE_BY_ROLE = {
    "user": "user_turn",
    "assistant": "assistant_turn",
    "system": "system_policy",
    "tool": "tool",
}
@dataclass(frozen=True, slots=True)
class TurnInput:
    """Caller-supplied source turn; derived identity and times are validated."""

    history_id: str
    session_id: str
    turn_idx: int
    role: str
    raw_content: str
    session_unix_ts: float
    source_type: str | None = None
    session_date: str | None = None
    token_count: int | None = None
    turn_id: str | None = None


@dataclass(frozen=True, slots=True)
class IngestionResult:
    turn_id: str
    state: Literal["completed", "quarantined"]
    gate_event_id: int
    extracted_count: int
    applied_count: int
    observation_ids: tuple[str, ...]
    replayed: bool


@dataclass(frozen=True, slots=True)
class PersistedTurn:
    """Result of persisting an unvetted turn for a baseline run."""

    turn_id: str
    replayed: bool


@dataclass(frozen=True, slots=True)
class _GateStatus:
    event_id: int
    decision: Literal["store", "quarantine"]


@dataclass(slots=True)
class _TurnWork:
    validated: _ValidatedTurn
    was_present: bool
    gate_event: _GateStatus
    job: sqlite3.Row
    candidates: tuple[ExtractedFact, ...] | None = None


@dataclass(frozen=True, slots=True)
class _ValidatedTurn:
    turn_id: str
    history_id: str
    session_id: str
    session_date: str
    session_unix_ts: float
    turn_idx: int
    role: str
    raw_content: str
    source_type: str
    token_count: int | None
    event_time: float
    payload_hash: str


class InvalidTurnError(ValueError):
    """Raised when source turn metadata is malformed or inconsistent."""


class TurnConflictError(ValueError):
    """Raised when a stable turn identity is submitted with different content."""


class IngestionOrderError(ValueError):
    """Raised when a history turn would break its append-only provenance chain."""


class MissingGateEvaluator(RuntimeError):
    """Raised when a new turn has no configured fail-closed gate evaluator."""


class MissingFactExtractor(RuntimeError):
    """Raised when an approved turn has no configured extractor."""


class QuarantinedTurnError(RuntimeError):
    """Raised after a persisted quarantine decision blocks extraction."""

    def __init__(self, turn_id: str, gate_event_id: int) -> None:
        self.turn_id = turn_id
        self.gate_event_id = gate_event_id
        super().__init__(f"Turn {turn_id!r} is quarantined by gate event {gate_event_id}")


class ExtractionIdentityConflict(ValueError):
    """Raised if a staged extraction is replaced by different validated data."""


class TemporalWriteError(RuntimeError):
    """Raised when an accepted fact cannot be applied through the temporal API."""


class IngestionRecoveryError(RuntimeError):
    """Raised when persisted ingestion state conflicts with gate/fact history."""


def ingest_turn(
    database: str,
    turn: TurnInput,
    *,
    evaluator: SignalEvaluator | None = None,
    extractor: FactExtractor | None = None,
    recorded_at: float | None = None,
    max_extraction_attempts: int = 2,
    timeout: float = 5.0,
    strict_untrusted: bool = False,
    slot_aliases: bool = False,
) -> IngestionResult:
    """Ingest one turn through the same staged batch pipeline as ``ingest_turns``."""
    return ingest_turns(
        database,
        (turn,),
        evaluator=evaluator,
        extractor=extractor,
        recorded_at=recorded_at,
        max_extraction_attempts=max_extraction_attempts,
        timeout=timeout,
        strict_untrusted=strict_untrusted,
        slot_aliases=slot_aliases,
    )[0]


def ingest_turns(
    database: str,
    turns: Sequence[TurnInput],
    *,
    evaluator: SignalEvaluator | None = None,
    extractor: FactExtractor | None = None,
    recorded_at: float | None = None,
    quarantine_policy: Literal["raise", "isolate"] = "raise",
    max_extraction_attempts: int = 2,
    timeout: float = 5.0,
    strict_untrusted: bool = False,
    slot_aliases: bool = False,
) -> tuple[IngestionResult, ...]:
    """Ingest a deterministic batch, validating all extractions before writes.

    Turns within one history are sorted by session timestamp, session ID, and
    turn index. Gate decisions and every candidate extraction are completed
    before the first temporal write, so an invalid later response cannot leave
    earlier candidates from this batch partially applied.
    """
    if quarantine_policy not in ("raise", "isolate"):
        raise ValueError("quarantine_policy must be 'raise' or 'isolate'")
    if type(strict_untrusted) is not bool or type(slot_aliases) is not bool:
        raise TypeError("strict_untrusted and slot_aliases must be bool values")
    ordered = _validate_turn_batch(turns, recorded_at=recorded_at)
    if not ordered:
        return ()
    return _ingest_validated_batch(
        database,
        ordered,
        evaluator=evaluator,
        extractor=extractor,
        recorded_at=recorded_at,
        quarantine_policy=quarantine_policy,
        max_extraction_attempts=max_extraction_attempts,
        timeout=timeout,
        strict_untrusted=strict_untrusted,
        slot_aliases=slot_aliases,
    )


def ingest_turns_only(
    database: str,
    turns: Sequence[TurnInput],
    *,
    recorded_at: float | None = None,
    timeout: float = 5.0,
) -> tuple[PersistedTurn, ...]:
    """Persist turns for baseline use only; stored turns are unvetted.

    This path creates ``pending_gate`` jobs without running a gate or extractor.
    Use ``ingest_turns`` later to upgrade the same turns through normal review.
    """
    ordered = _validate_turn_batch(turns, recorded_at=recorded_at)
    return tuple(
        PersistedTurn(
            turn.turn_id,
            _persist_turn(database, turn, recorded_at=recorded_at, timeout=timeout),
        )
        for turn in ordered
    )


def _validate_turn_batch(
    turns: Sequence[TurnInput], *, recorded_at: float | None
) -> tuple[_ValidatedTurn, ...]:
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        raise TypeError("turns must be a sequence of TurnInput values")
    if not turns:
        return ()
    validated = tuple(_validate_turn(turn, recorded_at=recorded_at) for turn in turns)
    histories = {turn.history_id for turn in validated}
    if len(histories) != 1:
        raise InvalidTurnError("one ingestion batch must contain exactly one history_id")
    by_id: dict[str, str] = {}
    session_metadata: dict[tuple[str, str], tuple[float, str]] = {}
    for turn in validated:
        previous = by_id.setdefault(turn.turn_id, turn.payload_hash)
        if previous != turn.payload_hash:
            raise TurnConflictError(
                f"turn ID {turn.turn_id!r} appears with different payloads in one batch"
            )
        session_key = (turn.history_id, turn.session_id)
        prior_metadata = session_metadata.setdefault(
            session_key, (turn.session_unix_ts, turn.session_date)
        )
        if (
            not math.isclose(prior_metadata[0], turn.session_unix_ts, abs_tol=1e-9)
            or prior_metadata[1] != turn.session_date
        ):
            raise InvalidTurnError(
                "session timestamp/date must be consistent across turns in one batch"
            )
    return tuple(sorted(
        {turn.turn_id: turn for turn in validated}.values(),
        key=lambda item: (item.event_time, item.session_id, item.turn_idx, item.turn_id),
    ))


def _ingest_validated_batch(
    database: str,
    turns: Sequence[_ValidatedTurn],
    *,
    evaluator: SignalEvaluator | None,
    extractor: FactExtractor | None,
    recorded_at: float | None,
    quarantine_policy: Literal["raise", "isolate"],
    max_extraction_attempts: int,
    timeout: float,
    strict_untrusted: bool,
    slot_aliases: bool,
) -> tuple[IngestionResult, ...]:
    # Stage source records independently. A crash leaves valid, resumable
    # pending_gate jobs and never exposes partially written turn metadata.
    presence = {
        turn.turn_id: _persist_turn(
            database, turn, recorded_at=recorded_at, timeout=timeout
        )
        for turn in turns
    }
    for turn in turns:
        logger.info(
            "Persisted source turn %s (existing=%s)",
            turn.turn_id,
            presence[turn.turn_id],
        )

    # Gate every source before any extraction starts. gate_events are the
    # authoritative recovery log; _approve_or_quarantine reuses prior stores
    # and makes any prior quarantine sticky.
    work: list[_TurnWork] = []
    results_by_id: dict[str, IngestionResult] = {}
    for turn in turns:
        try:
            gate_event = _approve_or_quarantine(
                database,
                turn,
                evaluator=evaluator,
                recorded_at=recorded_at,
                timeout=timeout,
                strict_untrusted=strict_untrusted,
            )
            quarantine_event_id = _gate_outcome_database(
                database, turn.turn_id, timeout=timeout
            )
            if quarantine_event_id is not None:
                _set_state(
                    database, turn.turn_id, "quarantined", recorded_at=recorded_at, timeout=timeout
                )
                raise QuarantinedTurnError(turn.turn_id, quarantine_event_id)
            if gate_event.decision != "store":
                raise IngestionRecoveryError(
                    "gate returned quarantine without an authoritative quarantine event"
                )
            job = _job(database, turn.turn_id, timeout=timeout)
            if job["state"] == "quarantined":
                raise IngestionRecoveryError(
                    "quarantined job has no authoritative quarantine event"
                )
            if job["state"] == "pending_gate":
                raise IngestionRecoveryError(
                    "gate event exists but ingestion state is still pending_gate"
                )
            if job["state"] not in ("approved", "staged", "completed"):
                raise IngestionRecoveryError(f"unsupported ingestion state {job['state']!r}")
            work.append(_TurnWork(turn, presence[turn.turn_id], gate_event, job))
        except QuarantinedTurnError as exc:
            if quarantine_policy != "isolate" or exc.turn_id != turn.turn_id:
                raise
            results_by_id[turn.turn_id] = _quarantined_result(
                turn.turn_id, exc.gate_event_id, replayed=presence[turn.turn_id]
            )
            continue
        logger.info(
            "Turn %s passed gate decision %s (event=%d)",
            turn.turn_id,
            gate_event.decision,
            gate_event.event_id,
        )

    # Collect and validate all responses before staging or applying any facts.
    # Existing staged JSON is revalidated and checked against its content hash.
    extraction_work: list[_TurnWork] = []
    for item in work:
        # Isolation rechecks earlier turns before adding their un-applied
        # candidates to a later prompt's slot set.
        if quarantine_policy == "isolate":
            surviving: list[_TurnWork] = []
            for previous in extraction_work:
                quarantine_event_id = _gate_outcome_database(
                    database, previous.validated.turn_id, timeout=timeout
                )
                if quarantine_event_id is None:
                    surviving.append(previous)
                else:
                    _set_state(
                        database,
                        previous.validated.turn_id,
                        "quarantined",
                        recorded_at=recorded_at,
                        timeout=timeout,
                    )
                    results_by_id[previous.validated.turn_id] = _quarantined_result(
                        previous.validated.turn_id,
                        quarantine_event_id,
                        replayed=previous.was_present,
                    )
            extraction_work = surviving
        working_slots = set(
            _existing_slots(
                database, turns[0].history_id, timeout=timeout,
                **({"slot_aliases": True} if slot_aliases else {}),
            )
        )
        for previous in extraction_work:
            if previous.candidates is not None:
                working_slots.update(
                    (candidate.entity, candidate.attribute)
                    for candidate in previous.candidates
                )

        if quarantine_policy == "isolate":
            try:
                quarantine_event_id = _gate_outcome_database(
                    database, item.validated.turn_id, timeout=timeout
                )
                if quarantine_event_id is not None:
                    raise QuarantinedTurnError(
                        item.validated.turn_id,
                        quarantine_event_id,
                    )
            except QuarantinedTurnError as exc:
                if exc.turn_id != item.validated.turn_id:
                    raise
                _set_state(
                    database,
                    item.validated.turn_id,
                    "quarantined",
                    recorded_at=recorded_at,
                    timeout=timeout,
                )
                results_by_id[item.validated.turn_id] = _quarantined_result(
                    item.validated.turn_id,
                    exc.gate_event_id,
                    replayed=item.was_present,
                )
                continue
        if item.job["state"] == "completed":
            extraction_work.append(item)
            continue
        if item.job["extraction_json"] is not None:
            item.candidates = _load_staged_candidates(item.job, item.validated)
        else:
            if item.job["state"] != "approved":
                raise IngestionRecoveryError(
                    f"cannot extract from ingestion state {item.job['state']!r}"
                )
            if extractor is None:
                raise MissingFactExtractor(
                    f"Turn {item.validated.turn_id!r} is approved but no FactExtractor is configured"
                )
            context = _extraction_turn(item.validated)
            prompt = build_extraction_prompt(context, tuple(sorted(working_slots)))
            item.candidates = extract_and_validate_with_bounded_retries(
                extractor,
                context,
                prompt,
                max_attempts=max_extraction_attempts,
            )
        item.candidates = tuple(
            sorted(
                item.candidates,
                key=lambda candidate: (
                    candidate.session_id,
                    candidate.turn_idx,
                    candidate.fact_seq,
                ),
            )
        )
        expected_count = item.job["fact_count"]
        if expected_count is not None and len(item.candidates) != int(expected_count):
            raise IngestionRecoveryError(
                "staged extraction fact count does not match its ledger"
            )
        working_slots.update((candidate.entity, candidate.attribute) for candidate in item.candidates)
        extraction_work.append(item)

    # Persist every newly accepted extraction before beginning temporal writes.
    staged_work: list[_TurnWork] = []
    for item in extraction_work:
        if item.job["state"] == "completed" or item.job["extraction_json"] is not None:
            staged_work.append(item)
            continue
        if item.candidates is None:
            raise IngestionRecoveryError("validated extraction candidates are missing")
        try:
            item.job = _stage_extraction(
                database,
                item.validated.turn_id,
                serialize_extraction(item.candidates),
                recorded_at=recorded_at,
                timeout=timeout,
            )
        except QuarantinedTurnError as exc:
            if quarantine_policy != "isolate" or exc.turn_id != item.validated.turn_id:
                raise
            results_by_id[item.validated.turn_id] = _quarantined_result(
                item.validated.turn_id,
                exc.gate_event_id,
                replayed=item.was_present,
            )
            continue
        staged_work.append(item)
        logger.info(
            "Staged %d validated facts for turn %s",
            int(item.job["fact_count"]),
            item.validated.turn_id,
        )

    for item in staged_work:
        if item.job["state"] == "completed" or item.candidates is None:
            continue
        try:
            _apply_batch_facts(
                database,
                ((item.validated, item.candidates),),
                recorded_at=recorded_at,
                timeout=timeout,
                **({"slot_aliases": True} if slot_aliases else {}),
            )
        except QuarantinedTurnError as exc:
            if quarantine_policy != "isolate" or exc.turn_id != item.validated.turn_id:
                raise
            results_by_id[item.validated.turn_id] = _quarantined_result(
                item.validated.turn_id,
                exc.gate_event_id,
                replayed=item.was_present,
            )

    for item in staged_work:
        if item.validated.turn_id in results_by_id:
            continue
        if item.job["state"] == "completed":
            quarantine_event_id = _gate_outcome_database(
                database, item.validated.turn_id, timeout=timeout
            )
            if quarantine_event_id is not None:
                _set_state(
                    database,
                    item.validated.turn_id,
                    "quarantined",
                    recorded_at=recorded_at,
                    timeout=timeout,
                )
                results_by_id[item.validated.turn_id] = _quarantined_result(
                    item.validated.turn_id, quarantine_event_id, replayed=item.was_present
                )
            else:
                results_by_id[item.validated.turn_id] = _completed_result(
                    database,
                    item.validated.turn_id,
                    item.gate_event.event_id,
                    replayed=True,
                    timeout=timeout,
                )
        else:
            try:
                result = _complete_job(
                    database,
                    item.validated.turn_id,
                    item.gate_event.event_id,
                    replayed=item.was_present,
                    recorded_at=recorded_at,
                    timeout=timeout,
                )
            except QuarantinedTurnError as exc:
                if quarantine_policy != "isolate" or exc.turn_id != item.validated.turn_id:
                    raise
                results_by_id[item.validated.turn_id] = _quarantined_result(
                    item.validated.turn_id,
                    exc.gate_event_id,
                    replayed=item.was_present,
                )
                continue
            results_by_id[item.validated.turn_id] = result
            logger.info(
                "Completed ingestion for turn %s (%d facts applied)",
                item.validated.turn_id,
                result.applied_count,
            )
    return tuple(results_by_id[turn.turn_id] for turn in turns)


def _validate_turn(turn: TurnInput, *, recorded_at: float | None) -> _ValidatedTurn:
    if not isinstance(turn, TurnInput):
        raise TypeError("turn must be a TurnInput")
    for name in ("history_id", "session_id"):
        value = getattr(turn, name)
        if not isinstance(value, str) or not value or value != value.strip():
            raise InvalidTurnError(f"{name} must be non-empty text without outer whitespace")
        if ":" in value:
            raise InvalidTurnError(f"{name} cannot contain ':' in the canonical turn ID format")
    if (
        isinstance(turn.turn_idx, bool)
        or not isinstance(turn.turn_idx, int)
        or not 0 <= turn.turn_idx <= _SQLITE_INTEGER_MAX
    ):
        raise InvalidTurnError("turn_idx must be a non-negative SQLite integer")
    source_type = _normalize_and_validate_role_source_type(turn.role, turn.source_type)
    if not isinstance(turn.raw_content, str) or not turn.raw_content.strip():
        raise InvalidTurnError("raw_content must be non-empty text")
    if isinstance(turn.session_unix_ts, bool):
        raise InvalidTurnError("session_unix_ts must be a finite number")
    try:
        session_unix_ts = float(turn.session_unix_ts)
    except (OverflowError, TypeError, ValueError) as exc:
        raise InvalidTurnError("session_unix_ts must be a finite number") from exc
    if not math.isfinite(session_unix_ts):
        raise InvalidTurnError("session_unix_ts must be finite")
    try:
        event_time = session_unix_ts + float(turn.turn_idx)
    except OverflowError as exc:
        raise InvalidTurnError("computed event_time must be finite") from exc
    if not math.isfinite(event_time):
        raise InvalidTurnError("computed event_time must be finite")
    session_date = turn.session_date
    if session_date is None:
        try:
            session_date = datetime.fromtimestamp(
                session_unix_ts, tz=timezone.utc
            ).date().isoformat()
        except (OverflowError, OSError, ValueError) as exc:
            raise InvalidTurnError("session_unix_ts is outside the supported date range") from exc
    if not isinstance(session_date, str):
        raise InvalidTurnError("session_date must be an ISO date string")
    try:
        parsed_date = date.fromisoformat(session_date)
    except ValueError as exc:
        raise InvalidTurnError("session_date must be a valid YYYY-MM-DD date") from exc
    if parsed_date.isoformat() != session_date:
        raise InvalidTurnError("session_date must use YYYY-MM-DD format")
    token_count = turn.token_count
    if token_count is not None and (
        isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0
        or token_count > _SQLITE_INTEGER_MAX
    ):
        raise InvalidTurnError("token_count must be null or a non-negative SQLite integer")
    expected_id = f"{turn.history_id}:{turn.session_id}:{turn.turn_idx}"
    turn_id = expected_id if turn.turn_id is None else turn.turn_id
    if not isinstance(turn_id, str) or turn_id != expected_id:
        raise InvalidTurnError(f"turn_id must equal {expected_id!r}")
    payload = {
        "turn_id": turn_id,
        "history_id": turn.history_id,
        "session_id": turn.session_id,
        "session_date": session_date,
        "session_unix_ts": session_unix_ts,
        "turn_idx": turn.turn_idx,
        "role": turn.role,
        "raw_content": turn.raw_content,
        "source_type": source_type,
        "token_count": token_count,
    }
    payload_hash = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if recorded_at is not None:
        _finite_time(recorded_at, "recorded_at")
    return _ValidatedTurn(
        turn_id=turn_id,
        history_id=turn.history_id,
        session_id=turn.session_id,
        session_date=session_date,
        session_unix_ts=session_unix_ts,
        turn_idx=turn.turn_idx,
        role=turn.role,
        raw_content=turn.raw_content,
        source_type=source_type,
        token_count=token_count,
        event_time=event_time,
        payload_hash=payload_hash,
    )


def _persist_turn(
    database: str,
    turn: _ValidatedTurn,
    *,
    recorded_at: float | None,
    timeout: float,
) -> bool:
    resolved_source_type = _normalize_and_validate_role_source_type(
        turn.role, turn.source_type
    )
    if resolved_source_type != turn.source_type:
        raise InvalidTurnError(
            "validated turn must carry its resolved source_type: "
            f"role={turn.role!r}, source_type={turn.source_type!r}"
        )
    now = _finite_time(clock.time() if recorded_at is None else recorded_at, "recorded_at")
    with transaction(database, timeout=timeout) as conn:
        job = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE turn_id = ?", (turn.turn_id,)
        ).fetchone()
        if job is not None:
            if job["payload_hash"] != turn.payload_hash:
                raise TurnConflictError(
                    f"turn ID {turn.turn_id!r} was already ingested with a different payload"
                )
            if (
                float(job["session_unix_ts"]) != turn.session_unix_ts
                or job["session_date"] != turn.session_date
            ):
                raise TurnConflictError("stored session metadata does not match turn payload")
            return True

        existing = conn.execute(
            "SELECT * FROM turns WHERE turn_id = ?", (turn.turn_id,)
        ).fetchone()
        if existing is not None:
            _assert_existing_turn_matches(existing, turn)
            expected_hash = _turn_hash(existing["prev_hash"], turn.raw_content)
            if existing["hash"] != expected_hash:
                raise TurnConflictError("existing source turn has a non-matching provenance hash")
            conn.execute(
                """INSERT INTO ingestion_jobs (
                       turn_id, payload_hash, session_unix_ts, session_date, state, updated_at
                   ) VALUES (?, ?, ?, ?, 'pending_gate', ?)""",
                (turn.turn_id, turn.payload_hash, turn.session_unix_ts, turn.session_date, now),
            )
            return True

        _assert_session_consistent(conn, turn)
        collision = conn.execute(
            """SELECT turn_id FROM turns
               WHERE history_id = ? AND session_id = ? AND turn_idx = ?""",
            (turn.history_id, turn.session_id, turn.turn_idx),
        ).fetchone()
        if collision is not None:
            raise TurnConflictError(
                f"session turn index is already assigned to {collision['turn_id']!r}"
            )
        previous = conn.execute(
            """SELECT turn_id, event_time, session_id, turn_idx, hash
               FROM turns WHERE history_id = ?
               ORDER BY event_time DESC, session_id DESC, turn_idx DESC, turn_id DESC
               LIMIT 1""",
            (turn.history_id,),
        ).fetchone()
        if previous is not None:
            previous_key = (
                float(previous["event_time"]),
                str(previous["session_id"]),
                int(previous["turn_idx"]),
                str(previous["turn_id"]),
            )
            new_key = (turn.event_time, turn.session_id, turn.turn_idx, turn.turn_id)
            if new_key <= previous_key:
                raise IngestionOrderError(
                    "turns must append in chronological event_time/session/turn order"
                )
            prev_hash = str(previous["hash"])
        else:
            prev_hash = None
        turn_hash = _turn_hash(prev_hash, turn.raw_content)
        conn.execute(
            """INSERT INTO turns (
                   turn_id, history_id, session_id, turn_idx, role, raw_content,
                   token_count, source_type, event_time, recorded_at, prev_hash, hash
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                turn.turn_id,
                turn.history_id,
                turn.session_id,
                turn.turn_idx,
                turn.role,
                turn.raw_content,
                turn.token_count,
                turn.source_type,
                turn.event_time,
                now,
                prev_hash,
                turn_hash,
            ),
        )
        conn.execute(
            """INSERT INTO ingestion_jobs (
                   turn_id, payload_hash, session_unix_ts, session_date, state, updated_at
               ) VALUES (?, ?, ?, ?, 'pending_gate', ?)""",
            (turn.turn_id, turn.payload_hash, turn.session_unix_ts, turn.session_date, now),
        )
        return False


def _assert_existing_turn_matches(row: sqlite3.Row, turn: _ValidatedTurn) -> None:
    actual = (
        row["history_id"], row["session_id"], row["turn_idx"], row["role"],
        row["raw_content"], row["token_count"], row["source_type"], row["event_time"],
    )
    expected = (
        turn.history_id, turn.session_id, turn.turn_idx, turn.role,
        turn.raw_content, turn.token_count, turn.source_type, turn.event_time,
    )
    if actual != expected:
        raise TurnConflictError(f"turn ID {turn.turn_id!r} conflicts with existing source data")


def _normalize_and_validate_role_source_type(
    role: str, source_type: str | None
) -> str:
    """Resolve an omitted source type and enforce the asymmetric trust labels."""
    if not isinstance(role, str) or role not in _ROLES:
        raise InvalidTurnError(f"role must be one of {sorted(_ROLES)}")
    if source_type is None:
        source_type = _DEFAULT_SOURCE_TYPE_BY_ROLE[role]
    try:
        validate_role_source_type(role, source_type)
    except ProvenanceError as exc:
        raise InvalidTurnError(str(exc)) from exc
    return source_type


def _assert_session_consistent(conn: sqlite3.Connection, turn: _ValidatedTurn) -> None:
    rows = conn.execute(
        """SELECT j.session_unix_ts, j.session_date
           FROM ingestion_jobs AS j JOIN turns AS t ON t.turn_id = j.turn_id
           WHERE t.history_id = ? AND t.session_id = ?""",
        (turn.history_id, turn.session_id),
    ).fetchall()
    for row in rows:
        if (
            not math.isclose(float(row["session_unix_ts"]), turn.session_unix_ts, abs_tol=1e-9)
            or row["session_date"] != turn.session_date
        ):
            raise InvalidTurnError(
                "session timestamp/date must be consistent across its turns"
            )
    existing_turns = conn.execute(
        """SELECT event_time, turn_idx FROM turns
           WHERE history_id = ? AND session_id = ?
           ORDER BY turn_idx""",
        (turn.history_id, turn.session_id),
    ).fetchall()
    for existing in existing_turns:
        stored_session_ts = float(existing["event_time"]) - int(existing["turn_idx"])
        if not math.isclose(stored_session_ts, turn.session_unix_ts, abs_tol=1e-9):
            raise InvalidTurnError("existing session turn timestamps are inconsistent")


def _approve_or_quarantine(
    database: str,
    turn: _ValidatedTurn,
    *,
    evaluator: SignalEvaluator | None,
    recorded_at: float | None,
    timeout: float,
    strict_untrusted: bool,
) -> _GateStatus:
    with connection(database, timeout=timeout) as conn:
        events = conn.execute(
            """SELECT event_id, decision, signals_fired, created_at FROM gate_events
               WHERE turn_id = ? ORDER BY event_id""",
            (turn.turn_id,),
        ).fetchall()
        job = conn.execute(
            "SELECT state FROM ingestion_jobs WHERE turn_id = ?", (turn.turn_id,)
        ).fetchone()
    if job is None:
        raise IngestionRecoveryError("source turn has no ingestion job")
    quarantine = next((row for row in events if row["decision"] == "quarantine"), None)
    if quarantine is not None:
        _set_state(
            database, turn.turn_id, "quarantined", recorded_at=recorded_at, timeout=timeout
        )
        return _GateStatus(int(quarantine["event_id"]), "quarantine")
    stores = [row for row in events if row["decision"] == "store"]
    if stores:
        if job["state"] == "pending_gate":
            _set_state(
                database, turn.turn_id, "approved", recorded_at=recorded_at, timeout=timeout
            )
        elif job["state"] == "quarantined":
            raise IngestionRecoveryError("job is quarantined but gate history has no quarantine event")
        return _GateStatus(int(stores[-1]["event_id"]), "store")
    if job["state"] in ("approved", "staged", "completed"):
        raise IngestionRecoveryError("approved ingestion has no corresponding gate event")
    if job["state"] == "quarantined":
        raise IngestionRecoveryError("quarantined job has no authoritative quarantine event")
    if evaluator is None:
        raise MissingGateEvaluator(
            f"Turn {turn.turn_id!r} is pending and no SignalEvaluator is configured"
        )
    event = evaluate_and_record_turn(
        database,
        turn.turn_id,
        evaluator,
        created_at=recorded_at,
        timeout=timeout,
        strict_untrusted=strict_untrusted,
    )
    _set_state(
        database,
        turn.turn_id,
        "quarantined" if event.decision == "quarantine" else "approved",
        recorded_at=recorded_at,
        timeout=timeout,
    )
    return _GateStatus(event.event_id, "quarantine" if event.decision == "quarantine" else "store")


def turn_is_eligible_database(database: str, turn_id: str, *, timeout: float) -> bool:
    with connection(database, timeout=timeout) as conn:
        return turn_is_eligible(conn, turn_id)


def _gate_outcome(conn: sqlite3.Connection, turn_id: str) -> int | None:
    """Return the authoritative quarantine ID or require an eligible store."""
    if turn_has_quarantine_event(conn, turn_id):
        return _quarantine_event_id(conn, turn_id)
    if not turn_has_gate_decision(conn, turn_id):
        raise IngestionRecoveryError(f"Turn {turn_id!r} has no gate decision")
    if not turn_is_eligible(conn, turn_id):
        raise IngestionRecoveryError(
            f"Turn {turn_id!r} has a gate decision but no eligible store event"
        )
    return None


def _gate_outcome_database(database: str, turn_id: str, *, timeout: float) -> int | None:
    with connection(database, timeout=timeout) as conn:
        return _gate_outcome(conn, turn_id)


def _existing_slots(
    database: str,
    history_id: str,
    *,
    timeout: float,
    slot_aliases: bool = False,
) -> tuple[tuple[str, str], ...]:
    with connection(database, timeout=timeout) as conn:
        rows = conn.execute(
            """SELECT DISTINCT f.entity, f.attribute
               FROM facts AS f
               WHERE f.history_id = ? AND f.key_type = 'slot'
                 AND f.entity IS NOT NULL AND f.attribute IS NOT NULL AND """
            + fact_eligibility_predicate("f")
            + " ORDER BY f.entity, f.attribute",
            (history_id,),
        ).fetchall()
    return tuple(
        normalize_slot_name(
            str(row["entity"]), str(row["attribute"]), use_aliases=slot_aliases
        )
        for row in rows
    )


def _stage_extraction(
    database: str,
    turn_id: str,
    extraction_json: str,
    *,
    recorded_at: float | None,
    timeout: float,
) -> sqlite3.Row:
    extraction_hash = hashlib.sha256(extraction_json.encode("utf-8")).hexdigest()
    count = len(json.loads(extraction_json))
    now = _write_timestamp(recorded_at)
    quarantine_event_id: int | None = None
    staged: sqlite3.Row | None = None
    with transaction(database, timeout=timeout) as conn:
        job = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE turn_id = ?", (turn_id,)
        ).fetchone()
        if job is None:
            raise IngestionRecoveryError("cannot stage extraction without ingestion job")
        quarantine_event_id = _gate_outcome(conn, turn_id)
        if quarantine_event_id is not None:
            conn.execute(
                "UPDATE ingestion_jobs SET state='quarantined', updated_at=? WHERE turn_id=?",
                (now, turn_id),
            )
        elif job["extraction_json"] is not None:
            if job["extraction_hash"] != extraction_hash:
                raise ExtractionIdentityConflict(
                    f"turn {turn_id!r} already has a different staged extraction"
                )
            return job
        elif job["state"] != "approved":
            raise IngestionRecoveryError(
                f"cannot stage extraction from ingestion state {job['state']!r}"
            )
        else:
            conn.execute(
                """UPDATE ingestion_jobs
                   SET state='staged', extraction_json=?, extraction_hash=?,
                       fact_count=?, applied_count=0, updated_at=?
                   WHERE turn_id=? AND state='approved' AND extraction_json IS NULL""",
                (extraction_json, extraction_hash, count, now, turn_id),
            )
            staged = conn.execute(
                "SELECT * FROM ingestion_jobs WHERE turn_id = ?", (turn_id,)
            ).fetchone()
    if quarantine_event_id is not None:
        raise QuarantinedTurnError(turn_id, quarantine_event_id)
    if staged is None:
        raise IngestionRecoveryError("staged extraction was not persisted")
    return staged


def _load_staged_candidates(
    job: sqlite3.Row,
    turn: _ValidatedTurn,
) -> tuple[ExtractedFact, ...]:
    payload = job["extraction_json"]
    expected_hash = job["extraction_hash"]
    if not isinstance(payload, str) or not isinstance(expected_hash, str):
        raise IngestionRecoveryError("staged extraction payload or hash is missing")
    actual_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if actual_hash != expected_hash:
        raise IngestionRecoveryError("staged extraction payload hash does not match its ledger")
    candidates = validate_extraction_response(payload, _extraction_turn(turn))
    candidates = tuple(
        sorted(candidates, key=lambda fact: (fact.session_id, fact.turn_idx, fact.fact_seq))
    )
    if len(candidates) != int(job["fact_count"]):
        raise IngestionRecoveryError("staged extraction fact count does not match its ledger")
    return candidates


def _apply_batch_facts(
    database: str,
    fact_batches: Sequence[tuple[_ValidatedTurn, tuple[ExtractedFact, ...] | None]],
    *,
    recorded_at: float | None,
    timeout: float,
    slot_aliases: bool = False,
) -> None:
    if not fact_batches:
        return
    history_ids = {turn.history_id for turn, _ in fact_batches}
    if len(history_ids) != 1:
        raise IngestionRecoveryError("temporal write batch crosses history boundaries")
    history_id = next(iter(history_ids))
    registry = set(
        _existing_slots(
            database,
            history_id,
            timeout=timeout,
            **({"slot_aliases": True} if slot_aliases else {}),
        )
    )
    for turn, candidates in fact_batches:
        if candidates is None:
            raise IngestionRecoveryError("cannot apply an unstaged extraction")
        fact_recorded_at = (
            _finite_time(recorded_at, "recorded_at")
            if recorded_at is not None
            else _persisted_turn_recorded_at(database, turn.turn_id, timeout=timeout)
        )
        for candidate in candidates:
            if (candidate.session_id, candidate.turn_idx) != (turn.session_id, turn.turn_idx):
                raise IngestionRecoveryError(
                    "staged fact provenance no longer matches its source turn"
                )
            quarantine_event_id = _gate_outcome_database(
                database, turn.turn_id, timeout=timeout
            )
            if quarantine_event_id is not None:
                _set_state(
                    database,
                    turn.turn_id,
                    "quarantined",
                    recorded_at=recorded_at,
                    timeout=timeout,
                )
                raise QuarantinedTurnError(
                    turn.turn_id,
                    quarantine_event_id,
                )

            normalized_slot = normalize_slot_name(
                candidate.entity, candidate.attribute, use_aliases=slot_aliases
            )
            # Current canonicalization is the specification's exact-name match.
            # No fuzzy merge is attempted before the calibrated local matcher exists.
            canonical_entity, canonical_attribute = next(
                (slot for slot in registry if slot == normalized_slot), normalized_slot
            )
            fact_id = f"{turn.turn_id}:fact:{candidate.fact_seq}"
            fact = FactInput(
                fact_id=fact_id,
                history_id=turn.history_id,
                turn_id=turn.turn_id,
                fact_seq=candidate.fact_seq,
                entity=canonical_entity,
                attribute=canonical_attribute,
                value=candidate.value,
                cardinality=candidate.cardinality,  # type: ignore[arg-type]
                valid_from=calculate_valid_from(
                    turn.session_unix_ts,
                    turn.turn_idx,
                    candidate.fact_seq,
                    as_of=candidate.as_of,
                ),
                recorded_at=fact_recorded_at,
                kind=candidate.kind,
                confidence=candidate.confidence,
            )
            try:
                insert_fact(database, fact, timeout=timeout)
            except UngatedSourceTurnError as exc:
                raise IngestionRecoveryError(str(exc)) from exc
            except (sqlite3.Error, ValueError) as exc:
                raise TemporalWriteError(
                    f"failed to apply fact sequence {candidate.fact_seq} for turn {turn.turn_id!r}"
                ) from exc
            registry.add((canonical_entity, canonical_attribute))
            _record_applied_fact(
                database,
                turn.turn_id,
                candidate.fact_seq,
                recorded_at=recorded_at,
                timeout=timeout,
            )


def _record_applied_fact(
    database: str,
    turn_id: str,
    fact_seq: int,
    *,
    recorded_at: float | None,
    timeout: float,
) -> None:
    quarantine_event_id: int | None = None
    with transaction(database, timeout=timeout) as conn:
        job = conn.execute(
            "SELECT state, applied_count FROM ingestion_jobs WHERE turn_id = ?", (turn_id,)
        ).fetchone()
        if job is None:
            raise IngestionRecoveryError("missing ingestion job while recording fact progress")
        quarantine_event_id = _gate_outcome(conn, turn_id)
        if quarantine_event_id is not None:
            conn.execute(
                "UPDATE ingestion_jobs SET state='quarantined', updated_at=? WHERE turn_id=?",
                (_write_timestamp(recorded_at), turn_id),
            )
        elif job["state"] != "staged":
            raise IngestionRecoveryError("fact progress can only advance a staged extraction")
        else:
            conn.execute(
                "UPDATE ingestion_jobs SET applied_count=?, updated_at=? WHERE turn_id=?",
                (
                    max(int(job["applied_count"]), fact_seq + 1),
                    _write_timestamp(recorded_at),
                    turn_id,
                ),
            )
    if quarantine_event_id is not None:
        raise QuarantinedTurnError(turn_id, quarantine_event_id)


def _complete_job(
    database: str,
    turn_id: str,
    gate_event_id: int,
    *,
    replayed: bool,
    recorded_at: float | None,
    timeout: float,
) -> IngestionResult:
    quarantine_event_id: int | None = None
    with transaction(database, timeout=timeout) as conn:
        job = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE turn_id = ?", (turn_id,)
        ).fetchone()
        if job is None:
            raise IngestionRecoveryError("missing ingestion job before completion")
        quarantine_event_id = _gate_outcome(conn, turn_id)
        if quarantine_event_id is not None:
            conn.execute(
                "UPDATE ingestion_jobs SET state='quarantined', updated_at=? WHERE turn_id=?",
                (_write_timestamp(recorded_at), turn_id),
            )
        elif job["state"] != "staged":
            raise IngestionRecoveryError("only staged extraction can be completed")
        elif int(job["applied_count"]) != int(job["fact_count"]):
            raise IngestionRecoveryError("not all staged facts have been applied")
        else:
            conn.execute(
                "UPDATE ingestion_jobs SET state='completed', updated_at=? WHERE turn_id=?",
                (_write_timestamp(recorded_at), turn_id),
            )
    if quarantine_event_id is not None:
        raise QuarantinedTurnError(turn_id, quarantine_event_id)
    return _completed_result(
        database, turn_id, gate_event_id, replayed=replayed, timeout=timeout
    )


def _completed_result(
    database: str,
    turn_id: str,
    gate_event_id: int,
    *,
    replayed: bool,
    timeout: float,
) -> IngestionResult:
    job = _job(database, turn_id, timeout=timeout)
    if job["state"] != "completed":
        raise IngestionRecoveryError("completion result requested for non-completed job")
    ids = _staged_fact_ids(turn_id, job["extraction_json"])
    return IngestionResult(
        turn_id=turn_id,
        state="completed",
        gate_event_id=gate_event_id,
        extracted_count=int(job["fact_count"]),
        applied_count=int(job["applied_count"]),
        observation_ids=ids,
        replayed=replayed,
    )


def _quarantined_result(
    turn_id: str, gate_event_id: int, *, replayed: bool
) -> IngestionResult:
    return IngestionResult(
        turn_id=turn_id,
        state="quarantined",
        gate_event_id=gate_event_id,
        extracted_count=0,
        applied_count=0,
        observation_ids=(),
        replayed=replayed,
    )


def _staged_fact_ids(turn_id: str, payload: str) -> tuple[str, ...]:
    rows = json.loads(payload)
    return tuple(f"{turn_id}:fact:{seq}" for seq, _ in enumerate(rows))


def _set_state(
    database: str,
    turn_id: str,
    state: str,
    *,
    recorded_at: float | None,
    timeout: float,
) -> None:
    with transaction(database, timeout=timeout) as conn:
        current = conn.execute(
            "SELECT state FROM ingestion_jobs WHERE turn_id = ?", (turn_id,)
        ).fetchone()
        if current is None:
            raise IngestionRecoveryError("missing ingestion job while changing state")
        if current["state"] == "quarantined" and state != "quarantined":
            raise IngestionRecoveryError("quarantine state cannot be reversed")
        conn.execute(
            "UPDATE ingestion_jobs SET state=?, updated_at=? WHERE turn_id=?",
            (state, _write_timestamp(recorded_at), turn_id),
        )


def _job(database: str, turn_id: str, *, timeout: float) -> sqlite3.Row:
    with connection(database, timeout=timeout) as conn:
        row = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE turn_id = ?", (turn_id,)
        ).fetchone()
    if row is None:
        raise IngestionRecoveryError("missing persistent ingestion state")
    return row


def _quarantine_event_id(conn: sqlite3.Connection, turn_id: str) -> int:
    row = conn.execute(
        "SELECT event_id FROM gate_events WHERE turn_id=? AND decision='quarantine' "
        "ORDER BY event_id LIMIT 1",
        (turn_id,),
    ).fetchone()
    if row is None:
        raise IngestionRecoveryError("turn is ineligible without a quarantine event")
    return int(row["event_id"])


def _get_quarantine_event_id(database: str, turn_id: str, *, timeout: float) -> int:
    with connection(database, timeout=timeout) as conn:
        return _quarantine_event_id(conn, turn_id)


def _persisted_turn_recorded_at(database: str, turn_id: str, *, timeout: float) -> float:
    with connection(database, timeout=timeout) as conn:
        row = conn.execute(
            "SELECT recorded_at FROM turns WHERE turn_id = ?", (turn_id,)
        ).fetchone()
    if row is None:
        raise IngestionRecoveryError("source turn is missing while resolving recorded_at")
    return _finite_time(row["recorded_at"], "recorded_at")


def _extraction_turn(turn: _ValidatedTurn) -> ExtractionTurn:
    return ExtractionTurn(
        turn_id=turn.turn_id,
        history_id=turn.history_id,
        session_id=turn.session_id,
        session_date=turn.session_date,
        session_unix_ts=turn.session_unix_ts,
        turn_idx=turn.turn_idx,
        role=turn.role,
        raw_content=turn.raw_content,
        source_type=turn.source_type,
        event_time=turn.event_time,
    )


def _turn_hash(prev_hash: str | None, raw_content: str) -> str:
    return hashlib.sha256((prev_hash or "").encode("utf-8") + raw_content.encode("utf-8")).hexdigest()


def _finite_time(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise InvalidTurnError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise InvalidTurnError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise InvalidTurnError(f"{name} must be finite")
    return result


def _write_timestamp(recorded_at: float | None) -> float:
    return _finite_time(
        clock.time() if recorded_at is None else recorded_at,
        "recorded_at",
    )
