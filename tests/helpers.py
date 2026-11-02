"""Shared fixture helpers for tests that seed source turns directly."""

from __future__ import annotations

import sqlite3


def record_store_event(
    conn: sqlite3.Connection,
    turn_id: str,
    *,
    created_at: float = 0.0,
) -> int:
    """Record a clean store decision for a directly seeded turn."""
    cursor = conn.execute(
        """INSERT INTO gate_events (
               turn_id, signal_bridge, signal_policy, signal_source,
               signals_fired, decision, created_at
           ) VALUES (?, 0.0, 0.0, 0.0, 0, 'store', ?)""",
        (turn_id, created_at),
    )
    return int(cursor.lastrowid)
