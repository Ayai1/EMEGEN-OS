"""Shared, fail-closed eligibility rules for memory retrieval."""

from __future__ import annotations

import re
import sqlite3

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _alias(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid SQL table alias: {value!r}")
    return value


def turn_eligibility_predicate(turn_alias: str = "t", *, gated: bool = True) -> str:
    """Return the shared SQL predicate for a source turn.

    ``gated=False`` is reserved for explicitly ungated turn-only baselines.
    Under D1, gated turns require at least one store event and no quarantine
    event. Quarantine is sticky: a later store event cannot reopen the turn.
    """
    alias = _alias(turn_alias)
    if not gated:
        return "1 = 1"
    return (
        "(EXISTS (SELECT 1 FROM gate_events AS store_event "
        f"WHERE store_event.turn_id = {alias}.turn_id "
        "AND store_event.decision = 'store') "
        "AND NOT EXISTS (SELECT 1 FROM gate_events AS quarantine_event "
        f"WHERE quarantine_event.turn_id = {alias}.turn_id "
        "AND quarantine_event.decision = 'quarantine'))"
    )


def turn_has_quarantine_event(conn: sqlite3.Connection, turn_id: str) -> bool:
    """Return whether an actual quarantine decision exists for this turn."""
    row = conn.execute(
        "SELECT 1 FROM gate_events WHERE turn_id = ? AND decision = 'quarantine' LIMIT 1",
        (turn_id,),
    ).fetchone()
    return row is not None


def turn_has_gate_decision(conn: sqlite3.Connection, turn_id: str) -> bool:
    """Return whether the gate has recorded either a store or quarantine event."""
    row = conn.execute(
        "SELECT 1 FROM gate_events WHERE turn_id = ? "
        "AND decision IN ('store', 'quarantine') LIMIT 1",
        (turn_id,),
    ).fetchone()
    return row is not None


def fact_eligibility_predicate(fact_alias: str = "f") -> str:
    """Return the common eligibility predicate for slot, trigger, or FTS rows.

    It checks the fact flag, source-turn gate history, and (for trigger rows)
    the parent fact. Retrieval routes should reuse this predicate instead of
    implementing route-specific quarantine rules.
    """
    alias = _alias(fact_alias)
    source_turn_ok = turn_eligibility_predicate("source_turn")
    parent_turn_ok = turn_eligibility_predicate("parent_turn")
    return (
        f"{alias}.quarantined = 0 "
        "AND EXISTS (SELECT 1 FROM turns AS source_turn "
        f"WHERE source_turn.turn_id = {alias}.turn_id "
        f"AND source_turn.history_id = {alias}.history_id AND {source_turn_ok}) "
        f"AND ({alias}.parent_fact_id IS NULL OR EXISTS ("
        "SELECT 1 FROM facts AS parent_fact "
        "JOIN turns AS parent_turn ON parent_turn.turn_id = parent_fact.turn_id "
        f"WHERE parent_fact.fact_id = {alias}.parent_fact_id "
        f"AND parent_fact.history_id = {alias}.history_id "
        "AND parent_turn.history_id = parent_fact.history_id "
        "AND parent_fact.quarantined = 0 "
        f"AND {parent_turn_ok}))"
    )


def observation_eligibility_predicate(observation_alias: str = "o") -> str:
    """Return the shared policy for lossless temporal source observations.

    Observation rows retain per-source provenance even when several identical
    observations project into one fact interval. A quarantine on the source
    event or its own materialized fact excludes that observation without
    suppressing unrelated clean confirmations in the same logical interval.
    """
    alias = _alias(observation_alias)
    return (
        f"{alias}.quarantined = 0 "
        "AND EXISTS (SELECT 1 FROM turns AS observation_turn "
        f"WHERE observation_turn.turn_id = {alias}.turn_id "
        f"AND observation_turn.history_id = {alias}.history_id "
        f"AND {turn_eligibility_predicate('observation_turn')}) "
        "AND NOT EXISTS (SELECT 1 FROM facts AS source_fact "
        f"WHERE source_fact.fact_id = {alias}.observation_id "
        f"AND source_fact.history_id = {alias}.history_id "
        "AND source_fact.quarantined = 1)"
    )


def turn_is_eligible(conn: sqlite3.Connection, turn_id: str) -> bool:
    """Check source-turn eligibility using the same policy as retrieval SQL."""
    row = conn.execute(
        f"SELECT {turn_eligibility_predicate('t')} AS eligible "
        "FROM turns AS t WHERE t.turn_id = ?",
        (turn_id,),
    ).fetchone()
    return row is not None and bool(row["eligible"])


def fact_is_eligible(conn: sqlite3.Connection, fact_id: str) -> bool:
    """Check fact eligibility using the shared fact predicate."""
    row = conn.execute(
        f"SELECT {fact_eligibility_predicate('f')} AS eligible "
        "FROM facts AS f JOIN turns AS t ON t.turn_id = f.turn_id "
        "WHERE f.fact_id = ?",
        (fact_id,),
    ).fetchone()
    return row is not None and bool(row["eligible"])
