"""Trust-gate evaluation, pure 2-of-3 decisions, and atomic persistence.

The bridge and policy signal detectors are injected interfaces because the
project specifications do not define their thresholds or detection algorithms.
The source-provenance signal is implemented exactly from ``BUILD_PLAN.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import math
import sqlite3
import time
from typing import Callable, Protocol

from .db import connection, transaction
from .provenance import ProvenanceError, validate_role_source_type
from .temporal import rebuild_slot_projection

logger = logging.getLogger(__name__)

_UNTRUSTED_SOURCE_TYPES = frozenset({"tool", "external_doc"})


class SignalStatus(str, Enum):
    FIRED = "fired"
    CLEAR = "clear"
    MISSING = "missing"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SignalResult:
    """One signal outcome. Missing or failed results are never treated as clear."""

    status: SignalStatus
    detail: str | None = None

    def __post_init__(self) -> None:
        try:
            status = SignalStatus(self.status)
        except ValueError as exc:
            raise ValueError(f"Unsupported signal status: {self.status!r}") from exc
        object.__setattr__(self, "status", status)
        if status in (SignalStatus.MISSING, SignalStatus.FAILED) and not self.detail:
            raise ValueError(f"{status.value} signal results require a detail")

    @property
    def fired(self) -> bool:
        if self.status == SignalStatus.FIRED:
            return True
        if self.status == SignalStatus.CLEAR:
            return False
        raise IncompleteSignalEvaluation((self.status.value,))


@dataclass(frozen=True, slots=True)
class GateSignals:
    signal_bridge: SignalResult
    signal_policy: SignalResult
    signal_source: SignalResult

    def named(self) -> tuple[tuple[str, SignalResult], ...]:
        return (
            ("signal_bridge", self.signal_bridge),
            ("signal_policy", self.signal_policy),
            ("signal_source", self.signal_source),
        )


@dataclass(frozen=True, slots=True)
class GateEvaluation:
    turn_id: str
    history_id: str
    source_type: str
    signals: GateSignals
    input_fingerprint: str


@dataclass(frozen=True, slots=True)
class GateDecision:
    decision: str
    signals_fired: int


@dataclass(frozen=True, slots=True)
class GateEvent:
    event_id: int
    turn_id: str
    decision: str
    signals_fired: int
    base_trust_before: float
    base_trust_after: float
    created_at: float


class IncompleteSignalEvaluation(ValueError):
    """Raised when a 2-of-3 decision cannot be made from three valid results."""

    def __init__(self, signals: tuple[str, ...]) -> None:
        self.signals = signals
        super().__init__("Incomplete gate signals: " + ", ".join(signals))


class GatePersistenceError(ValueError):
    """Raised when a gate evaluation does not match the immutable source turn."""


class InconsistentTurnProvenanceError(GatePersistenceError):
    """Raised when persisted role/source metadata cannot receive a gate event."""


class GateInputsChangedError(RuntimeError):
    """Raised when mutable gate inputs change during or after evaluation."""


class SignalEvaluator(Protocol):
    """Injected detectors and a cheap fingerprint of their mutable inputs.

    ``fingerprint_inputs`` must cover every mutable database input used by the
    bridge and policy evaluators. It must be deterministic and local; it must
    not make model or network calls. The source turn itself is revalidated by
    the gate and is immutable apart from ``base_trust``.
    """

    def fingerprint_inputs(
        self, conn: sqlite3.Connection, turn: sqlite3.Row
    ) -> str:
        """Return a stable token for the mutable inputs used by both detectors."""

    def evaluate_bridge(
        self, conn: sqlite3.Connection, turn: sqlite3.Row
    ) -> SignalResult:
        """Evaluate bridging-similarity across shortening query prefixes."""

    def evaluate_policy(
        self, conn: sqlite3.Connection, turn: sqlite3.Row
    ) -> SignalResult:
        """Evaluate contradiction against pinned system/user policy facts."""


def evaluate_source_signal(source_type: str) -> SignalResult:
    """Fire for the specified untrusted provenance types only."""
    if not isinstance(source_type, str):
        raise TypeError("source_type must be a string")
    return SignalResult(
        SignalStatus.FIRED
        if source_type in _UNTRUSTED_SOURCE_TYPES
        else SignalStatus.CLEAR
    )


def evaluate_turn(
    database: str,
    turn_id: str,
    evaluator: SignalEvaluator,
    *,
    timeout: float = 5.0,
) -> GateEvaluation:
    """Evaluate signals without persisting events or changing trust.

    Bridge and policy evaluation errors are detected and logged. The function
    raises ``IncompleteSignalEvaluation`` rather than returning a partial result
    that a caller might mistake for a store decision.
    """
    if not isinstance(turn_id, str):
        raise TypeError("turn_id must be a string")
    if not turn_id:
        raise ValueError("turn_id must be a non-empty string")
    with connection(database, timeout=timeout) as conn:
        turn = conn.execute(
            "SELECT turn_id, history_id, role, source_type, raw_content, event_time, "
            "session_id, turn_idx FROM turns WHERE turn_id = ?",
            (turn_id,),
        ).fetchone()
        if turn is None:
            raise ValueError(f"Unknown source turn: {turn_id}")
        _validate_persisted_provenance(turn)

        input_fingerprint = _input_fingerprint(evaluator, conn, turn)

        bridge = _evaluate_injected(
            "signal_bridge", lambda: evaluator.evaluate_bridge(conn, turn)
        )
        policy = _evaluate_injected(
            "signal_policy", lambda: evaluator.evaluate_policy(conn, turn)
        )
        source = evaluate_source_signal(str(turn["source_type"]))
        if _input_fingerprint(evaluator, conn, turn) != input_fingerprint:
            raise GateInputsChangedError(
                "gate inputs changed while signals were being evaluated"
            )
        evaluated = GateEvaluation(
            turn_id=str(turn["turn_id"]),
            history_id=str(turn["history_id"]),
            source_type=str(turn["source_type"]),
            signals=GateSignals(bridge, policy, source),
            input_fingerprint=input_fingerprint,
        )
        # Fail at the evaluator boundary so callers cannot mistake an
        # incomplete result for a store decision and continue ingestion.
        decide_gate(evaluated.signals)
        return evaluated


def _evaluate_injected(
    name: str, evaluate: Callable[[], SignalResult | None]
) -> SignalResult:
    try:
        result = evaluate()
        if result is None:
            return SignalResult(SignalStatus.MISSING, "evaluator returned no result")
        if not isinstance(result, SignalResult):
            raise TypeError("evaluator must return SignalResult")
        return result
    except Exception as exc:
        logger.warning("%s evaluation failed (%s)", name, type(exc).__name__)
        return SignalResult(SignalStatus.FAILED, type(exc).__name__)


def decide_gate(signals: GateSignals) -> GateDecision:
    """Apply the specified 2-of-3 policy to three completed signal results."""
    if not isinstance(signals, GateSignals):
        raise TypeError("signals must be GateSignals")
    invalid: list[str] = []
    for name, result in signals.named():
        if not isinstance(result, SignalResult):
            invalid.append(name)
        elif result.status not in (SignalStatus.FIRED, SignalStatus.CLEAR):
            invalid.append(name)
    if invalid:
        raise IncompleteSignalEvaluation(tuple(invalid))

    fired = sum(result.status == SignalStatus.FIRED for _, result in signals.named())
    return GateDecision(
        decision="quarantine" if fired >= 2 else "store",
        signals_fired=fired,
    )


def _input_fingerprint(
    evaluator: SignalEvaluator,
    conn: sqlite3.Connection,
    turn: sqlite3.Row,
) -> str:
    fingerprint_method = getattr(evaluator, "fingerprint_inputs", None)
    if not callable(fingerprint_method):
        raise TypeError("evaluator must implement fingerprint_inputs")
    fingerprint = fingerprint_method(conn, turn)
    if not isinstance(fingerprint, str) or not fingerprint:
        raise TypeError("fingerprint_inputs must return a non-empty string")
    return fingerprint


def evaluate_and_record_turn(
    database: str,
    turn_id: str,
    evaluator: SignalEvaluator,
    *,
    created_at: float | None = None,
    timeout: float = 5.0,
) -> GateEvent:
    """Evaluate a turn's signals and atomically record/apply the gate decision.

    This is the canonical application-facing operation. Callers supply neither
    signal outcomes nor a preconstructed evaluation. Signal evaluation occurs
    outside the SQLite write transaction; persistence revalidates source-turn
    identity/provenance and the evaluator's mutable-input fingerprint.
    """
    evaluation = evaluate_turn(database, turn_id, evaluator, timeout=timeout)
    return _record_gate_decision(
        database,
        evaluation,
        evaluator,
        created_at=created_at,
        timeout=timeout,
    )


def _record_gate_decision(
    database: str,
    evaluation: GateEvaluation,
    evaluator: SignalEvaluator,
    *,
    created_at: float | None = None,
    timeout: float = 5.0,
) -> GateEvent:
    """Internal atomic persistence for a completed gate evaluation.

    Application code should use ``evaluate_and_record_turn``. Quarantine is
    sticky: a subsequent store event never clears flags or raises trust again.
    """
    decision = decide_gate(evaluation.signals)
    timestamp = _finite_timestamp(time.time() if created_at is None else created_at)

    with transaction(database, timeout=timeout) as conn:
        turn = conn.execute(
            "SELECT * FROM turns WHERE turn_id = ?",
            (evaluation.turn_id,),
        ).fetchone()
        if turn is None:
            raise GatePersistenceError(f"Unknown source turn: {evaluation.turn_id}")
        if turn["history_id"] != evaluation.history_id:
            raise GatePersistenceError("Evaluation history does not match source turn")
        if turn["source_type"] != evaluation.source_type:
            raise GatePersistenceError("Evaluation source type does not match source turn")
        if _input_fingerprint(evaluator, conn, turn) != evaluation.input_fingerprint:
            raise GateInputsChangedError(
                "gate inputs changed after signal evaluation"
            )
        expected_source = evaluate_source_signal(str(turn["source_type"]))
        if evaluation.signals.signal_source != expected_source:
            raise GatePersistenceError("Evaluation source signal does not match source type")
        _assert_source_rows_match_history(conn, evaluation.turn_id, evaluation.history_id)

        trust_before = _finite_timestamp(turn["base_trust"])
        if not 0.0 <= trust_before <= 1.0:
            raise GatePersistenceError("Source turn base_trust must be in [0, 1]")
        signal_values = tuple(
            1.0 if result.status == SignalStatus.FIRED else 0.0
            for _, result in evaluation.signals.named()
        )
        _validate_persisted_provenance(turn)
        cursor = conn.execute(
            """INSERT INTO gate_events (
                   turn_id, signal_bridge, signal_policy, signal_source,
                   signals_fired, decision, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                evaluation.turn_id,
                signal_values[0],
                signal_values[1],
                signal_values[2],
                decision.signals_fired,
                decision.decision,
                timestamp,
            ),
        )
        event_id = int(cursor.lastrowid)
        trust_after = trust_before

        if decision.decision == "quarantine":
            reason = f"gate event {event_id}: {decision.signals_fired}/3 signals fired"
            affected_slots = conn.execute(
                """SELECT DISTINCT history_id, entity, attribute
                   FROM fact_observations
                   WHERE turn_id = ? AND cardinality = 'single'""",
                (evaluation.turn_id,),
            ).fetchall()
            conn.execute(
                """UPDATE facts
                   SET quarantined = 1,
                       quarantine_reason = COALESCE(quarantine_reason, ?)
                   WHERE turn_id = ?""",
                (reason, evaluation.turn_id),
            )
            conn.execute(
                """UPDATE fact_observations
                   SET quarantined = 1,
                       quarantine_reason = COALESCE(quarantine_reason, ?)
                   WHERE turn_id = ?""",
                (reason, evaluation.turn_id),
            )
            # The source text remains immutable. Quarantined turns carry no
            # trust for future ranking, even before route-level eligibility.
            conn.execute(
                "UPDATE turns SET base_trust = 0.0 WHERE turn_id = ?",
                (evaluation.turn_id,),
            )
            trust_after = 0.0

            # Rebuild coalesced slot projections so quarantined confirmations
            # cannot leave stale counts, dates, or interval boundaries behind.
            for slot in affected_slots:
                if slot["history_id"] != evaluation.history_id:
                    raise GatePersistenceError(
                        "Observation history does not match source turn"
                    )
                rebuild_slot_projection(
                    conn,
                    str(slot["history_id"]),
                    str(slot["entity"]),
                    str(slot["attribute"]),
                    timestamp,
                )

        event = GateEvent(
            event_id=event_id,
            turn_id=evaluation.turn_id,
            decision=decision.decision,
            signals_fired=decision.signals_fired,
            base_trust_before=trust_before,
            base_trust_after=trust_after,
            created_at=timestamp,
        )

    logger.info(
        "Recorded gate event %s for turn %s: %s (%d/3); base_trust %.3f -> %.3f",
        event.event_id,
        event.turn_id,
        event.decision,
        event.signals_fired,
        event.base_trust_before,
        event.base_trust_after,
    )
    return event


def _assert_source_rows_match_history(
    conn: sqlite3.Connection, turn_id: str, history_id: str
) -> None:
    for table in ("facts", "fact_observations"):
        row = conn.execute(
            f"SELECT history_id FROM {table} WHERE turn_id = ? "
            "AND history_id != ? LIMIT 1",
            (turn_id, history_id),
        ).fetchone()
        if row is not None:
            raise GatePersistenceError(
                f"A {table} row for turn {turn_id!r} crosses history boundaries"
            )


def _validate_persisted_provenance(turn: sqlite3.Row) -> None:
    try:
        validate_role_source_type(str(turn["role"]), str(turn["source_type"]))
    except ProvenanceError as exc:
        raise InconsistentTurnProvenanceError(
            "Persisted turn provenance is inconsistent: "
            f"role={turn['role']!r}, source_type={turn['source_type']!r}: {exc}"
        ) from exc


def _finite_timestamp(value: float) -> float:
    if isinstance(value, bool):
        raise TypeError("timestamp must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("timestamp must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError("timestamp must be finite")
    return result
