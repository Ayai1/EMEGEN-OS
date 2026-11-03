"""Deterministic slot-fact interval management and historical retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
import math
import re
import sqlite3
from typing import Literal

from .db import connection, transaction
from .eligibility import (
    observation_eligibility_predicate,
    turn_eligibility_predicate,
    turn_has_gate_decision,
    turn_has_quarantine_event,
)
from .normalize import normalize_slot_name, normalize_value

FactView = Literal["current", "as_of", "history"]
_KINDS = {"state", "preference", "event", "constraint", "other"}
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class TemporalInvariantError(ValueError):
    """Raised when a temporal write would make slot history ambiguous."""


class UngatedSourceTurnError(TemporalInvariantError):
    """Raised when a fact write references a turn without a gate decision."""


@dataclass(frozen=True, slots=True)
class FactInput:
    """A locally validated slot fact; ``valid_from`` is event time, not wall time."""

    fact_id: str
    history_id: str
    turn_id: str
    fact_seq: int
    entity: str
    attribute: str
    value: str
    cardinality: Literal["single", "multi"]
    valid_from: float
    recorded_at: float
    kind: str | None = "state"
    key_text: str | None = None
    trust: float = 0.6
    quarantined: bool = False
    quarantine_reason: str | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class TemporalBoundary:
    """A normalized temporal endpoint and its explicit inclusivity.

    For an ``as_of`` query, an inclusive boundary selects the state at the
    instant. An exclusive boundary selects the left limit immediately before
    the instant, without approximating it by subtracting a floating-point
    epsilon. The same value type can describe endpoints of history ranges.
    """

    timestamp: float
    inclusive: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, (int, float)):
            raise TypeError("Temporal boundary timestamp must be finite")
        timestamp = float(self.timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("Temporal boundary timestamp must be finite")
        if not isinstance(self.inclusive, bool):
            raise TypeError("Temporal boundary inclusivity must be a boolean")
        object.__setattr__(self, "timestamp", timestamp)


def as_of_timestamp(value: float | int | date | datetime | str | TemporalBoundary) -> float:
    """Convert an as-of value to Unix seconds.

    A date or ``YYYY-MM-DD`` string means 00:00:00 UTC at the start of that
    date. Explicit timestamps must include a UTC offset; naive datetimes are
    rejected so callers cannot accidentally depend on the host timezone. A
    ``TemporalBoundary`` returns its timestamp only; pass the boundary object
    directly to temporal retrieval to preserve its inclusivity.
    """
    if isinstance(value, TemporalBoundary):
        return value.timestamp
    if isinstance(value, bool):
        raise TypeError("Boolean values are not timestamps")
    if isinstance(value, (int, float)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("As-of timestamp must be finite")
        return result
    if isinstance(value, datetime):
        stamp = value
    elif isinstance(value, date):
        stamp = datetime.combine(value, time.min, tzinfo=timezone.utc)
    elif isinstance(value, str):
        if _ISO_DATE.fullmatch(value):
            day = date.fromisoformat(value)
            stamp = datetime.combine(day, time.min, tzinfo=timezone.utc)
        else:
            iso_value = value[:-1] + "+00:00" if value.endswith("Z") else value
            stamp = datetime.fromisoformat(iso_value)
    else:
        raise TypeError("As-of value must be a timestamp, date, datetime, or ISO string")

    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Explicit as-of datetimes must include a timezone offset")
    result = stamp.timestamp()
    if not math.isfinite(result):
        raise ValueError("As-of timestamp must be finite")
    return result


def calculate_valid_from(
    session_unix_ts: float,
    turn_idx: int,
    fact_seq: int,
    *,
    as_of: date | str | None = None,
) -> float:
    """Apply the deterministic timestamp formulas from ``prompts.md``."""
    base = float(session_unix_ts)
    if not math.isfinite(base):
        raise ValueError("Session timestamp must be finite")
    if isinstance(turn_idx, bool) or not isinstance(turn_idx, int) or turn_idx < 0:
        raise ValueError("turn_idx must be a non-negative integer")
    if isinstance(fact_seq, bool) or not isinstance(fact_seq, int) or fact_seq < 0:
        raise ValueError("fact_seq must be a non-negative integer")
    if as_of is None:
        return base + turn_idx + fact_seq * 1e-3
    historical_base = as_of_timestamp(as_of)
    return historical_base + turn_idx * 1e-3 + fact_seq * 1e-6


def insert_fact(database: str, fact: FactInput, *, timeout: float = 5.0) -> str | None:
    """Insert or replay one slot fact atomically and return its canonical interval ID.

    The observation ledger preserves every source fact. ``facts`` stores the
    coalesced interval projection. Rebuilding that projection from observations
    makes out-of-order insertion independent of arrival order.
    """
    normalized = _normalize_input(fact)
    with transaction(database, timeout=timeout) as conn:
        return _insert_fact(conn, normalized)


def _normalize_input(fact: FactInput) -> FactInput:
    if not isinstance(fact, FactInput):
        raise TypeError("fact must be a FactInput")
    for name in ("fact_id", "history_id", "turn_id"):
        if not getattr(fact, name) or not isinstance(getattr(fact, name), str):
            raise ValueError(f"{name} must be a non-empty string")
    if isinstance(fact.fact_seq, bool) or not isinstance(fact.fact_seq, int) or fact.fact_seq < 0:
        raise ValueError("fact_seq must be a non-negative integer")
    if fact.cardinality not in ("single", "multi"):
        raise ValueError("cardinality must be 'single' or 'multi'")
    if fact.kind is not None and fact.kind not in _KINDS:
        raise ValueError(f"kind must be one of {sorted(_KINDS)}")
    if not isinstance(fact.value, str) or not fact.value.strip():
        raise ValueError("value must be a non-empty string")
    entity, attribute = normalize_slot_name(fact.entity, fact.attribute)
    if not entity or not attribute:
        raise ValueError("entity and attribute must normalize to non-empty slot names")
    value_norm = normalize_value(fact.value)
    if not value_norm:
        raise ValueError("value must normalize to a non-empty string")
    valid_from = _finite_number(fact.valid_from, "valid_from")
    recorded_at = _finite_number(fact.recorded_at, "recorded_at")
    trust = _finite_number(fact.trust, "trust")
    if not 0.0 <= trust <= 1.0:
        raise ValueError("trust must be in [0, 1]")
    confidence = fact.confidence
    if confidence is not None:
        confidence = _finite_number(confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
    key_text = fact.key_text.strip() if fact.key_text is not None else ""
    if not key_text:
        key_text = f"{entity}.{attribute}: {fact.value}"
    return FactInput(
        fact_id=fact.fact_id,
        history_id=fact.history_id,
        turn_id=fact.turn_id,
        fact_seq=fact.fact_seq,
        entity=entity,
        attribute=attribute,
        value=fact.value,
        cardinality=fact.cardinality,
        valid_from=valid_from,
        recorded_at=recorded_at,
        kind=fact.kind,
        key_text=key_text,
        trust=trust,
        quarantined=bool(fact.quarantined),
        quarantine_reason=fact.quarantine_reason,
        confidence=confidence,
    )


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _insert_fact(conn: sqlite3.Connection, fact: FactInput) -> str | None:
    source = conn.execute(
        "SELECT history_id, event_time, session_id, turn_idx "
        "FROM turns WHERE turn_id = ?",
        (fact.turn_id,),
    ).fetchone()
    if source is None:
        raise ValueError(f"Unknown source turn: {fact.turn_id}")
    if source["history_id"] != fact.history_id:
        raise ValueError("Fact history_id does not match its source turn")
    if not turn_has_gate_decision(conn, fact.turn_id):
        raise UngatedSourceTurnError(
            f"Source turn {fact.turn_id!r} has no gate decision"
        )

    _backfill_legacy_facts(conn, fact.history_id, fact.entity, fact.attribute)
    existing = conn.execute(
        "SELECT * FROM fact_observations WHERE observation_id = ?",
        (fact.fact_id,),
    ).fetchone()
    if existing is not None:
        _assert_same_observation(existing, fact)
        if fact.cardinality == "single" and not existing["quarantined"]:
            _reconcile_slot(conn, fact.history_id, fact.entity, fact.attribute, fact.recorded_at)
            existing = conn.execute(
                "SELECT canonical_fact_id FROM fact_observations WHERE observation_id = ?",
                (fact.fact_id,),
            ).fetchone()
        return existing["canonical_fact_id"]

    same_source = conn.execute(
        "SELECT observation_id FROM fact_observations WHERE turn_id = ? AND fact_seq = ?",
        (fact.turn_id, fact.fact_seq),
    ).fetchone()
    if same_source is not None:
        raise TemporalInvariantError(
            f"Turn {fact.turn_id!r} fact_seq {fact.fact_seq} is already assigned "
            f"to observation {same_source['observation_id']!r}"
        )

    fact_row = conn.execute(
        "SELECT history_id, turn_id, entity, attribute, value_norm, valid_from "
        "FROM facts WHERE fact_id = ?",
        (fact.fact_id,),
    ).fetchone()
    if fact_row is not None:
        _backfill_legacy_facts(conn, fact_row["history_id"], fact_row["entity"], fact_row["attribute"])
        existing = conn.execute(
            "SELECT * FROM fact_observations WHERE observation_id = ?",
            (fact.fact_id,),
        ).fetchone()
        if existing is not None:
            _assert_same_observation(existing, fact)
            return existing["canonical_fact_id"]
        raise TemporalInvariantError(f"fact_id {fact.fact_id!r} is already in use")

    source_is_quarantined = turn_has_quarantine_event(conn, fact.turn_id)
    quarantined = fact.quarantined or source_is_quarantined
    quarantine_reason = fact.quarantine_reason
    if source_is_quarantined and not quarantine_reason:
        quarantine_reason = "source turn has a quarantine gate event"

    conn.execute(
        """INSERT INTO fact_observations (
               observation_id, history_id, turn_id, fact_seq, entity, attribute,
               value, value_norm, cardinality, kind, key_text, valid_at,
               effective_at, canonical_fact_id, last_confirmed_at,
               last_confirmed_turn_id, confirmation_count, recorded_at, trust,
               quarantined, quarantine_reason, confidence
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, 1, ?, ?, ?, ?, ?)""",
        (
            fact.fact_id,
            fact.history_id,
            fact.turn_id,
            fact.fact_seq,
            fact.entity,
            fact.attribute,
            fact.value,
            normalize_value(fact.value),
            fact.cardinality,
            fact.kind,
            fact.key_text,
            fact.valid_from,
            fact.valid_from,
            fact.valid_from,
            fact.turn_id,
            fact.recorded_at,
            fact.trust,
            int(quarantined),
            quarantine_reason,
            fact.confidence,
        ),
    )

    if fact.cardinality == "multi" or quarantined:
        _insert_standalone_fact(conn, fact, quarantined, quarantine_reason)
        conn.execute(
            "UPDATE fact_observations SET canonical_fact_id = ? WHERE observation_id = ?",
            (fact.fact_id, fact.fact_id),
        )
        return fact.fact_id

    _reconcile_slot(conn, fact.history_id, fact.entity, fact.attribute, fact.recorded_at)
    row = conn.execute(
        "SELECT canonical_fact_id FROM fact_observations WHERE observation_id = ?",
        (fact.fact_id,),
    ).fetchone()
    if row is None:
        raise TemporalInvariantError("Inserted observation was not assigned to an interval")
    # None is a deliberate non-projection: an untrusted conflicting observation
    # remains in the append-only ledger but is not an active fact.
    return row["canonical_fact_id"]


def _assert_same_observation(existing: sqlite3.Row, fact: FactInput) -> None:
    expected = (
        fact.history_id,
        fact.turn_id,
        fact.fact_seq,
        fact.entity,
        fact.attribute,
        fact.value,
        normalize_value(fact.value),
        fact.cardinality,
        fact.valid_from,
    )
    actual = (
        existing["history_id"],
        existing["turn_id"],
        existing["fact_seq"],
        existing["entity"],
        existing["attribute"],
        existing["value"],
        existing["value_norm"],
        existing["cardinality"],
        existing["valid_at"],
    )
    if actual != expected:
        raise TemporalInvariantError(
            f"Observation ID {fact.fact_id!r} was replayed with different fact data"
        )


def _turn_eligible(conn: sqlite3.Connection, turn_id: str) -> bool:
    row = conn.execute(
        f"SELECT {turn_eligibility_predicate('t')} AS eligible "
        "FROM turns AS t WHERE t.turn_id = ?",
        (turn_id,),
    ).fetchone()
    return row is not None and bool(row["eligible"])


def _backfill_legacy_facts(
    conn: sqlite3.Connection, history_id: str, entity: str, attribute: str
) -> None:
    """Preserve pre-ledger slot rows when an existing database is upgraded."""
    rows = conn.execute(
        """SELECT f.* FROM facts AS f
           WHERE f.history_id = ? AND f.key_type = 'slot'
             AND NOT EXISTS (
               SELECT 1 FROM fact_observations AS o WHERE o.observation_id = f.fact_id
             )""",
        (history_id,),
    ).fetchall()
    for row in rows:
        normalized_entity, normalized_attribute = normalize_slot_name(
            row["entity"] or "", row["attribute"] or ""
        )
        if (normalized_entity, normalized_attribute) != (entity, attribute):
            continue
        turn = conn.execute(
            "SELECT history_id FROM turns WHERE turn_id = ?", (row["turn_id"],)
        ).fetchone()
        if turn is None or turn["history_id"] != row["history_id"]:
            raise TemporalInvariantError(
                f"Legacy fact {row['fact_id']!r} has inconsistent source history"
            )
        conn.execute(
            """INSERT OR IGNORE INTO fact_observations (
                   observation_id, history_id, turn_id, fact_seq, entity, attribute,
                   value, value_norm, cardinality, kind, key_text, valid_at,
                   effective_at, canonical_fact_id, last_confirmed_at,
                   last_confirmed_turn_id, confirmation_count, recorded_at, trust,
                   quarantined, quarantine_reason, confidence
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row["fact_id"], row["history_id"], row["turn_id"], row["fact_seq"],
                normalized_entity, normalized_attribute, row["value"],
                row["value_norm"] or normalize_value(row["value"]),
                row["cardinality"] or "single", row["kind"], row["key_text"],
                row["valid_from"], row["valid_from"], row["fact_id"],
                row["last_confirmed_at"] or row["valid_from"],
                row["last_confirmed_turn_id"] or row["turn_id"],
                max(1, int(row["confirmation_count"])), row["recorded_at"],
                row["trust"], row["quarantined"], row["quarantine_reason"], row["confidence"],
            ),
        )


def _insert_standalone_fact(
    conn: sqlite3.Connection,
    fact: FactInput,
    quarantined: bool,
    quarantine_reason: str | None,
) -> None:
    conn.execute(
        """INSERT INTO facts (
               fact_id, history_id, turn_id, fact_seq, key_type, entity, attribute,
               value, value_norm, cardinality, kind, key_text, valid_from,
               last_confirmed_at, last_confirmed_turn_id, confirmation_count,
               recorded_at, trust, quarantined, quarantine_reason, confidence
           ) VALUES (?, ?, ?, ?, 'slot', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)""",
        (
            fact.fact_id, fact.history_id, fact.turn_id, fact.fact_seq,
            fact.entity, fact.attribute, fact.value, normalize_value(fact.value),
            fact.cardinality, fact.kind, fact.key_text, fact.valid_from,
            fact.valid_from, fact.turn_id, fact.recorded_at, fact.trust,
            int(quarantined), quarantine_reason, fact.confidence,
        ),
    )


def _eligible_observations(
    conn: sqlite3.Connection,
    history_id: str,
    entity: str | None = None,
    attribute: str | None = None,
    cardinality: str | None = None,
) -> list[sqlite3.Row]:
    where = [
        "o.history_id = ?",
        observation_eligibility_predicate("o"),
    ]
    params: list[object] = [history_id]
    if entity is not None:
        where.append("o.entity = ?")
        params.append(entity)
    if attribute is not None:
        where.append("o.attribute = ?")
        params.append(attribute)
    if cardinality is not None:
        where.append("o.cardinality = ?")
        params.append(cardinality)
    return conn.execute(
        """SELECT o.*, t.event_time AS source_event_time,
                  t.source_type AS source_type,
                  t.session_id AS source_session_id, t.turn_idx AS source_turn_idx
           FROM fact_observations AS o
           JOIN turns AS t ON t.turn_id = o.turn_id
           WHERE """ + " AND ".join(where) +
        " ORDER BY o.entity, o.attribute, o.valid_at, t.event_time, t.session_id, "
        "t.turn_idx, o.fact_seq, o.observation_id",
        params,
    ).fetchall()


def _reconcile_slot(
    conn: sqlite3.Connection, history_id: str, entity: str, attribute: str, recorded_at: float
) -> None:
    """Rebuild one single-valued slot's projection with provenance-aware updates.

    ``user_turn`` has higher authority than ``tool`` and ``external_doc``,
    independently of the numeric trust values. A conflicting lower-authority
    observation remains in the append-only observation ledger but is excluded
    from the slot projection until a user turn confirms or replaces it.
    """
    observations = _eligible_observations(
        conn, history_id, entity, attribute, cardinality="single"
    )
    observations, blocked_observation_ids = _resolve_source_authority(observations)
    for obs in observations:
        conn.execute(
            "UPDATE fact_observations SET effective_at = ? WHERE observation_id = ?",
            (obs["valid_at"], obs["observation_id"]),
        )

    # Resolve exact timestamp collisions in stable accepted-source order.
    # nextafter keeps projected events ordered without inventing a fixed-size
    # epsilon at epoch scale.
    previous: float | None = None
    resolved: list[dict[str, object]] = []
    for obs in observations:
        observation_id = str(obs["observation_id"])
        effective = float(obs["valid_at"])
        if previous is not None and effective <= previous:
            effective = math.nextafter(previous, math.inf)
        if not math.isfinite(effective):
            raise TemporalInvariantError("Could not resolve a finite event-time tie")
        previous = effective
        item = dict(obs)
        item["effective_at"] = effective
        resolved.append(item)
        conn.execute(
            "UPDATE fact_observations SET effective_at = ? WHERE observation_id = ?",
            (effective, observation_id),
        )

    # Consecutive normalized-equal observations are one logical interval.
    runs: list[list[dict[str, object]]] = []
    for item in resolved:
        if runs and runs[-1][-1]["value_norm"] == item["value_norm"]:
            runs[-1].append(item)
        else:
            runs.append([item])

    desired_ids = {str(run[0]["observation_id"]) for run in runs}
    old_rows = conn.execute(
        """SELECT DISTINCT f.* FROM facts AS f
           WHERE f.history_id = ? AND f.key_type = 'slot' AND f.cardinality = 'single'
             AND (
               (f.entity = ? AND f.attribute = ?)
               OR EXISTS (
                 SELECT 1 FROM fact_observations AS o
                 WHERE o.canonical_fact_id = f.fact_id AND o.history_id = ?
                   AND o.entity = ? AND o.attribute = ?
               )
             )""",
        (history_id, entity, attribute, history_id, entity, attribute),
    ).fetchall()
    old_by_id = {str(row["fact_id"]): row for row in old_rows}

    # Validate all desired primary keys before mutating relationships.
    for desired_id in desired_ids:
        collision = conn.execute(
            "SELECT history_id, entity, attribute, key_type FROM facts WHERE fact_id = ?",
            (desired_id,),
        ).fetchone()
        if collision is not None:
            same_legacy_slot = (
                desired_id in old_by_id
                and collision["history_id"] == history_id
                and collision["key_type"] == "slot"
                and normalize_slot_name(collision["entity"] or "", collision["attribute"] or "")
                == (entity, attribute)
            )
            same_slot = (
                collision["history_id"] == history_id
                and collision["entity"] == entity
                and collision["attribute"] == attribute
                and collision["key_type"] == "slot"
            )
            if not (same_legacy_slot or same_slot):
                raise TemporalInvariantError(
                    f"Fact ID {desired_id!r} is already used by another slot"
                )

    # Snapshot each old row's observations so child cues can be reparented if a
    # backdated event changes which observation starts a normalized interval.
    old_observations: dict[str, list[str]] = {}
    for old_id in old_by_id:
        old_observations[old_id] = [
            str(row[0]) for row in conn.execute(
                "SELECT observation_id FROM fact_observations WHERE canonical_fact_id = ?",
                (old_id,),
            )
        ]

    # Remove interval links before replacing/deleting any referenced fact rows.
    conn.execute(
        """UPDATE facts SET superseded_by = NULL
           WHERE history_id = ? AND entity = ? AND attribute = ? AND key_type = 'slot'
             AND cardinality = 'single'""",
        (history_id, entity, attribute),
    )
    mapping: dict[str, str] = {}
    desired_rows: list[dict[str, object]] = []
    for index, run in enumerate(runs):
        first = run[0]
        last = max(
            run,
            key=lambda item: (
                float(item["last_confirmed_at"]),
                float(item["source_event_time"]),
                str(item["source_session_id"]),
                int(item["source_turn_idx"]),
                int(item["fact_seq"]),
                str(item["observation_id"]),
            ),
        )
        next_run = runs[index + 1] if index + 1 < len(runs) else None
        interval_end = float(next_run[0]["effective_at"]) if next_run else None
        fact_id = str(first["observation_id"])
        for item in run:
            obs_id = str(item["observation_id"])
            mapping[obs_id] = fact_id
        run_recorded = max(float(item["recorded_at"]) for item in run)
        expired_at = None
        if next_run:
            expired_at = max(run_recorded, float(next_run[0]["recorded_at"]))
        desired_rows.append(
            {
                "fact_id": fact_id,
                "history_id": history_id,
                "turn_id": first["turn_id"],
                "fact_seq": first["fact_seq"],
                "entity": entity,
                "attribute": attribute,
                "value": first["value"],
                "value_norm": first["value_norm"],
                "kind": first["kind"],
                "key_text": first["key_text"],
                "valid_from": first["effective_at"],
                "valid_until": interval_end,
                "next_fact_id": str(next_run[0]["observation_id"]) if next_run else None,
                "last_confirmed_at": last["last_confirmed_at"],
                "last_confirmed_turn_id": last["last_confirmed_turn_id"],
                "confirmation_count": sum(int(item["confirmation_count"]) for item in run),
                "recorded_at": first["recorded_at"],
                "expired_at": expired_at,
                "trust": first["trust"],
                "confidence": first["confidence"],
            }
        )

    # Keep vector rows when a canonical interval is unchanged. Remove embeddings
    # only when an interval disappears or its indexed text changes.
    desired_by_id = {str(row["fact_id"]): row for row in desired_rows}
    for old_id, old in old_by_id.items():
        desired = desired_by_id.get(old_id)
        if desired is None or old["key_text"] != desired["key_text"]:
            conn.execute(
                "DELETE FROM embeddings WHERE ref_type = 'fact' AND ref_id = ?",
                (old_id,),
            )

    # Install every target row before adding supersession FKs between them.
    for row in desired_rows:
        _upsert_interval_fact(conn, row)
    for observation_id, canonical_id in mapping.items():
        conn.execute(
            "UPDATE fact_observations SET canonical_fact_id = ? WHERE observation_id = ?",
            (canonical_id, observation_id),
        )
    for observation_id in blocked_observation_ids:
        conn.execute(
            "UPDATE fact_observations SET canonical_fact_id = NULL WHERE observation_id = ?",
            (observation_id,),
        )

    eligible_observation_ids = set(mapping) | blocked_observation_ids
    obsolete_ids = set(old_by_id) - desired_ids
    for old_id in sorted(obsolete_ids):
        old = old_by_id[old_id]
        if old_id not in eligible_observation_ids:
            # Keep source rows excluded only for lacking an approval decision;
            # exclusion by itself is not evidence of quarantine. A genuine
            # observation quarantine or source-turn quarantine remains sticky.
            quarantined_observation = conn.execute(
                """SELECT 1 FROM fact_observations AS o
                   WHERE o.history_id = ?
                     AND (o.observation_id = ? OR o.canonical_fact_id = ?)
                     AND (o.quarantined = 1 OR EXISTS (
                       SELECT 1 FROM gate_events AS q
                       WHERE q.turn_id = o.turn_id AND q.decision = 'quarantine'
                     ))
                   LIMIT 1""",
                (history_id, old_id, old_id),
            ).fetchone()
            if quarantined_observation is not None:
                conn.execute(
                    """UPDATE facts SET quarantined = 1,
                              quarantine_reason = COALESCE(
                                  quarantine_reason, 'source observation is quarantined'
                              ), superseded_by = NULL
                       WHERE fact_id = ?""",
                    (old_id,),
                )
            else:
                conn.execute(
                    "UPDATE facts SET superseded_by = NULL WHERE fact_id = ?",
                    (old_id,),
                )
            continue

        target_id = mapping.get(old_id)
        if target_id is None:
            for observation_id in old_observations.get(old_id, []):
                if observation_id in mapping:
                    target_id = mapping[observation_id]
                    break
        if target_id is None:
            target_id = _replacement_for_old_fact(old, desired_rows)
        children = conn.execute(
            "SELECT fact_id FROM facts WHERE parent_fact_id = ?", (old_id,)
        ).fetchall()
        if target_id is not None:
            conn.execute(
                "UPDATE facts SET parent_fact_id = ? WHERE parent_fact_id = ?",
                (target_id, old_id),
            )
        elif children:
            conn.execute(
                """UPDATE facts SET parent_fact_id = NULL, quarantined = 1,
                          quarantine_reason = COALESCE(quarantine_reason, 'parent interval removed')
                   WHERE parent_fact_id = ?""",
                (old_id,),
            )
        conn.execute("UPDATE facts SET superseded_by = NULL WHERE superseded_by = ?", (old_id,))
        conn.execute("DELETE FROM facts WHERE fact_id = ?", (old_id,))

    # Set the successor relation only after all canonical fact IDs exist.
    for row in desired_rows:
        conn.execute(
            "UPDATE facts SET superseded_by = ? WHERE fact_id = ?",
            (row["next_fact_id"], row["fact_id"]),
        )
        conn.execute(
            """UPDATE facts AS trigger_fact
               SET valid_from = ?, valid_until = ?, expired_at = ?
               WHERE trigger_fact.key_type = 'trigger' AND trigger_fact.parent_fact_id = ?""",
            (row["valid_from"], row["valid_until"], row["expired_at"], row["fact_id"]),
        )


def _resolve_source_authority(
    observations: list[sqlite3.Row],
) -> tuple[list[sqlite3.Row], set[str]]:
    """Exclude lower-authority conflicting events after a user-established value.

    Numeric ``trust`` is intentionally not used: current fixture and production
    source classes can carry equal scores. ``user_turn`` is the authority class;
    ``tool`` and ``external_doc`` can update their own untrusted slot chain but
    cannot replace a user-established value. A later user observation of the
    proposed value records the confirmation and promotes it to user authority.
    """
    accepted: list[sqlite3.Row] = []
    blocked: set[str] = set()
    current_value: str | None = None
    current_authority: str | None = None
    for observation in observations:
        observation_id = str(observation["observation_id"])
        source_type = str(observation["source_type"])
        value_norm = str(observation["value_norm"])
        if (
            current_value is not None
            and value_norm != current_value
            and source_type in {"tool", "external_doc"}
            and current_authority == "user_turn"
        ):
            blocked.add(observation_id)
            continue
        accepted.append(observation)
        if current_value is None or value_norm != current_value:
            current_value = value_norm
            current_authority = source_type
        elif source_type == "user_turn":
            current_authority = "user_turn"
    return accepted, blocked


def rebuild_slot_projection(
    conn: sqlite3.Connection,
    history_id: str,
    entity: str,
    attribute: str,
    recorded_at: float,
) -> None:
    """Rebuild one slot projection inside the caller's active transaction.

    Gate persistence uses this after marking source observations ineligible so
    confirmation metadata and interval boundaries cannot retain quarantined
    observations. This function never opens or commits a connection.
    """
    if not history_id:
        raise ValueError("history_id must be a non-empty string")
    normalized_entity, normalized_attribute = normalize_slot_name(entity, attribute)
    if not normalized_entity or not normalized_attribute:
        raise ValueError("entity and attribute must normalize to non-empty slot names")
    _reconcile_slot(
        conn,
        history_id,
        normalized_entity,
        normalized_attribute,
        _finite_number(recorded_at, "recorded_at"),
    )


def _replacement_for_old_fact(
    old: sqlite3.Row, desired_rows: list[dict[str, object]]
) -> str | None:
    candidates = [
        row for row in desired_rows
        if row["entity"] == old["entity"]
        and row["attribute"] == old["attribute"]
        and row["value_norm"] == old["value_norm"]
    ]
    if not candidates:
        return None
    old_start = float(old["valid_from"])
    for row in candidates:
        end = row["valid_until"]
        if float(row["valid_from"]) <= old_start and (end is None or old_start < float(end)):
            return str(row["fact_id"])
    return str(min(candidates, key=lambda row: abs(float(row["valid_from"]) - old_start))["fact_id"])


def _upsert_interval_fact(conn: sqlite3.Connection, row: dict[str, object]) -> None:
    conn.execute(
        """INSERT INTO facts (
               fact_id, history_id, turn_id, fact_seq, key_type, entity, attribute,
               value, value_norm, cardinality, kind, key_text, valid_from, valid_until,
               superseded_by, last_confirmed_at, last_confirmed_turn_id,
               confirmation_count, recorded_at, expired_at, trust, quarantined, confidence
           ) VALUES (?, ?, ?, ?, 'slot', ?, ?, ?, ?, 'single', ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, 0, ?)
           ON CONFLICT(fact_id) DO UPDATE SET
               history_id = excluded.history_id,
               turn_id = excluded.turn_id,
               fact_seq = excluded.fact_seq,
               key_type = excluded.key_type,
               entity = excluded.entity,
               attribute = excluded.attribute,
               value = excluded.value,
               value_norm = excluded.value_norm,
               cardinality = excluded.cardinality,
               kind = excluded.kind,
               key_text = excluded.key_text,
               parent_fact_id = NULL,
               valid_from = excluded.valid_from,
               valid_until = excluded.valid_until,
               superseded_by = NULL,
               last_confirmed_at = excluded.last_confirmed_at,
               last_confirmed_turn_id = excluded.last_confirmed_turn_id,
               confirmation_count = excluded.confirmation_count,
               recorded_at = excluded.recorded_at,
               expired_at = excluded.expired_at,
               trust = excluded.trust,
               quarantined = 0,
               quarantine_reason = NULL,
               confidence = excluded.confidence""",
        (
            row["fact_id"], row["history_id"], row["turn_id"], row["fact_seq"],
            row["entity"], row["attribute"], row["value"], row["value_norm"],
            row["kind"], row["key_text"], row["valid_from"], row["valid_until"],
            row["last_confirmed_at"], row["last_confirmed_turn_id"],
            row["confirmation_count"], row["recorded_at"], row["expired_at"],
            row["trust"], row["confidence"],
        ),
    )


def _ordered_timeline(
    observations: list[sqlite3.Row],
) -> dict[tuple[str, str, str], list[dict[str, object]]]:
    by_slot: dict[tuple[str, str, str], list[sqlite3.Row]] = {}
    for row in observations:
        key = (str(row["entity"]), str(row["attribute"]), str(row["cardinality"]))
        by_slot.setdefault(key, []).append(row)

    timelines: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for slot, rows in by_slot.items():
        rows.sort(
            key=lambda row: (
                float(row["valid_at"]),
                float(row["source_event_time"]),
                str(row["source_session_id"]),
                int(row["source_turn_idx"]),
                int(row["fact_seq"]),
                str(row["observation_id"]),
            )
        )
        previous: float | None = None
        timeline: list[dict[str, object]] = []
        for row in rows:
            effective = float(row["valid_at"])
            if slot[2] == "single" and previous is not None and effective <= previous:
                effective = math.nextafter(previous, math.inf)
            if not math.isfinite(effective):
                raise TemporalInvariantError("Could not resolve a finite event-time tie")
            item = dict(row)
            item["effective_at"] = effective
            timeline.append(item)
            previous = effective
        timelines[slot] = timeline
    return timelines


def retrieve_facts(
    database: str,
    history_id: str,
    *,
    view: FactView = "current",
    as_of: float | int | date | datetime | str | TemporalBoundary | None = None,
    entity: str | None = None,
    attribute: str | None = None,
    timeout: float = 5.0,
) -> list[dict[str, object]]:
    """Retrieve eligible slot facts for current, as-of, or history views.

    `current` returns each single-valued slot's latest eligible interval and
    all eligible multi-valued observations. `as_of` uses half-open intervals;
    a `TemporalBoundary(inclusive=False)` selects the left limit before its
    timestamp without floating-point epsilon arithmetic. `history` returns
    normalized intervals in deterministic event order.
    """
    with connection(database, timeout=timeout) as conn:
        return _retrieve_facts(
            conn, history_id, view=view, as_of=as_of,
            entity=entity, attribute=attribute,
        )


def _retrieve_facts(
    conn: sqlite3.Connection,
    history_id: str,
    *,
    view: FactView = "current",
    as_of: float | int | date | datetime | str | TemporalBoundary | None = None,
    entity: str | None = None,
    attribute: str | None = None,
) -> list[dict[str, object]]:
    """Connection-aware temporal projection used by snapshot retrieval."""
    if view not in ("current", "as_of", "history"):
        raise ValueError("view must be 'current', 'as_of', or 'history'")
    if view == "as_of" and as_of is None:
        raise ValueError("as_of is required when view='as_of'")
    if view != "as_of" and as_of is not None:
        raise ValueError("as_of can only be supplied when view='as_of'")
    boundary = as_of if isinstance(as_of, TemporalBoundary) else None
    timestamp = as_of_timestamp(as_of) if as_of is not None else None
    if not history_id:
        raise ValueError("history_id must be a non-empty string")
    if (entity is None) != (attribute is None):
        raise ValueError("entity and attribute must be supplied together")
    normalized_entity = normalized_attribute = None
    if entity is not None and attribute is not None:
        normalized_entity, normalized_attribute = normalize_slot_name(entity, attribute)
    observations = _eligible_observations(
        conn,
        history_id,
        normalized_entity,
        normalized_attribute,
    )
    observations, _ = _resolve_source_authority(observations)
    timelines = _ordered_timeline(observations)
    projected: list[dict[str, object]] = []
    for (slot_entity, slot_attribute, cardinality), timeline in timelines.items():
        single_runs: list[list[dict[str, object]]] = []
        multi_rows: list[dict[str, object]] = []
        for item in timeline:
            if cardinality == "multi":
                multi_rows.append(item)
            elif single_runs and single_runs[-1][-1]["value_norm"] == item["value_norm"]:
                single_runs[-1].append(item)
            else:
                single_runs.append([item])

        intervals: list[dict[str, object]] = []
        for index, run in enumerate(single_runs):
            first, last = run[0], run[-1]
            next_run = single_runs[index + 1] if index + 1 < len(single_runs) else None
            until = float(next_run[0]["effective_at"]) if next_run else None
            intervals.append(
                {
                    **dict(first),
                    "fact_id": first["observation_id"],
                    "valid_from": first["effective_at"],
                    "valid_until": until,
                    "effective_valid_until": until,
                    "superseded_by": next_run[0]["observation_id"] if next_run else None,
                    "last_confirmed_at": last["last_confirmed_at"],
                    "last_confirmed_turn_id": last["last_confirmed_turn_id"],
                    "confirmation_count": sum(int(item["confirmation_count"]) for item in run),
                    "quarantined": 0,
                    "expired_at": (
                        max(
                            max(float(entry["recorded_at"]) for entry in run),
                            float(next_run[0]["recorded_at"]),
                        )
                        if next_run
                        else None
                    ),
                }
            )
        if view == "current":
            if intervals:
                projected.append(intervals[-1])
            projected.extend(_project_multi(item) for item in multi_rows)
            continue
        if view == "as_of":
            assert timestamp is not None
            if boundary is None or boundary.inclusive:
                projected.extend(
                    row for row in intervals
                    if float(row["valid_from"]) <= timestamp
                    and (row["valid_until"] is None or float(row["valid_until"]) > timestamp)
                )
                projected.extend(
                    _project_multi(item)
                    for item in multi_rows if float(item["effective_at"]) <= timestamp
                )
            else:
                # For a half-open fact interval [valid_from, valid_until), its
                # left-limit at T is present when valid_from < T <= valid_until.
                projected.extend(
                    row for row in intervals
                    if float(row["valid_from"]) < timestamp
                    and (row["valid_until"] is None or float(row["valid_until"]) >= timestamp)
                )
                projected.extend(
                    _project_multi(item)
                    for item in multi_rows if float(item["effective_at"]) < timestamp
                )
            continue
        projected.extend(intervals)
        projected.extend(_project_multi(item) for item in multi_rows)

    projected.sort(
        key=lambda row: (
            float(row["valid_from"]),
            str(row["entity"]),
            str(row["attribute"]),
            str(row["fact_id"]),
        )
    )
    return projected


def _project_multi(item: dict[str, object]) -> dict[str, object]:
    return {
        **item,
        "fact_id": item["observation_id"],
        "valid_from": item["effective_at"],
        "valid_until": None,
        "effective_valid_until": None,
        "superseded_by": None,
        "last_confirmed_at": item["last_confirmed_at"],
        "last_confirmed_turn_id": item["last_confirmed_turn_id"],
        "confirmation_count": int(item["confirmation_count"]),
        "quarantined": 0,
        "expired_at": None,
    }


def get_current_facts(
    database: str,
    history_id: str,
    *,
    entity: str | None = None,
    attribute: str | None = None,
    timeout: float = 5.0,
) -> list[dict[str, object]]:
    return retrieve_facts(
        database, history_id, view="current", entity=entity, attribute=attribute, timeout=timeout
    )


def get_facts_as_of(
    database: str,
    history_id: str,
    as_of: float | int | date | datetime | str | TemporalBoundary,
    *,
    entity: str | None = None,
    attribute: str | None = None,
    timeout: float = 5.0,
) -> list[dict[str, object]]:
    return retrieve_facts(
        database, history_id, view="as_of", as_of=as_of,
        entity=entity, attribute=attribute, timeout=timeout,
    )


def get_fact_history(
    database: str,
    history_id: str,
    *,
    entity: str | None = None,
    attribute: str | None = None,
    timeout: float = 5.0,
) -> list[dict[str, object]]:
    return retrieve_facts(
        database, history_id, view="history", entity=entity, attribute=attribute, timeout=timeout
    )
