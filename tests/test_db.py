"""Integration tests for the EngramOS SQLite foundation."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engramOS.db import connection, initialize_database, transaction  # noqa: E402


def insert_turn(conn: sqlite3.Connection, *, turn_id: str = "h:s:0") -> None:
    conn.execute(
        """INSERT INTO turns (
               turn_id, history_id, session_id, turn_idx, role, raw_content,
               event_time, recorded_at, hash
           ) VALUES (?, 'h', 's', 0, 'user', 'rotate secret project token',
                     10.0, 10.0, 'turn-hash')""",
        (turn_id,),
    )


def match_count(conn: sqlite3.Connection, table: str, term: str) -> int:
    return conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {table} MATCH ?", (term,)
    ).fetchone()[0]


class DatabaseFoundationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Path(self.temp_dir.name) / "engram.sqlite3"
        initialize_database(self.database)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_schema_initialization_is_idempotent(self) -> None:
        initialize_database(self.database)
        with connection(self.database) as conn:
            names = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
                )
            }
        self.assertTrue({"turns", "facts", "turns_fts", "facts_fts", "budget_summary"} <= names)

    def test_foreign_keys_are_enabled_on_every_connection(self) -> None:
        for _ in range(2):
            with connection(self.database) as conn:
                self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
                with self.assertRaises(sqlite3.IntegrityError):
                    conn.execute(
                        """INSERT INTO facts (
                               fact_id, history_id, turn_id, key_type, key_text,
                               valid_from, recorded_at
                           ) VALUES ('orphan', 'h', 'missing-turn', 'slot', 'key', 1, 1)"""
                    )

    def test_transaction_rolls_back_all_writes_on_failure(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            with transaction(self.database) as conn:
                insert_turn(conn)
                raise RuntimeError("injected failure")

        with connection(self.database) as conn:
            count = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
        self.assertEqual(count, 0)

    def test_turn_metadata_is_immutable_but_base_trust_is_mutable(self) -> None:
        with connection(self.database) as conn:
            insert_turn(conn)

        with connection(self.database) as conn:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("UPDATE turns SET raw_content = 'rewritten' WHERE turn_id = 'h:s:0'")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("UPDATE turns SET event_time = 99 WHERE turn_id = 'h:s:0'")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("DELETE FROM turns WHERE turn_id = 'h:s:0'")
            conn.execute("UPDATE turns SET base_trust = 0.1 WHERE turn_id = 'h:s:0'")
            row = conn.execute(
                "SELECT raw_content, event_time, base_trust FROM turns WHERE turn_id = 'h:s:0'"
            ).fetchone()

        self.assertEqual(row["raw_content"], "rotate secret project token")
        self.assertEqual(row["event_time"], 10.0)
        self.assertEqual(row["base_trust"], 0.1)

    def test_fts_tracks_turn_insert_and_fact_insert_update_delete(self) -> None:
        with connection(self.database) as conn:
            insert_turn(conn)
            self.assertEqual(match_count(conn, "turns_fts", "secret"), 1)

            conn.execute(
                """INSERT INTO facts (
                       fact_id, history_id, turn_id, key_type, entity, attribute,
                       value, value_norm, key_text, valid_from, recorded_at
                   ) VALUES ('f1', 'h', 'h:s:0', 'slot', 'user', 'project',
                             'alpha', 'alpha', 'oldmarker project alpha', 10, 10)"""
            )
            self.assertEqual(match_count(conn, "facts_fts", "oldmarker"), 1)

            conn.execute("UPDATE facts SET key_text = 'newmarker project alpha' WHERE fact_id = 'f1'")
            self.assertEqual(match_count(conn, "facts_fts", "oldmarker"), 0)
            self.assertEqual(match_count(conn, "facts_fts", "newmarker"), 1)

            conn.execute("DELETE FROM facts WHERE fact_id = 'f1'")
            self.assertEqual(match_count(conn, "facts_fts", "newmarker"), 0)


if __name__ == "__main__":
    unittest.main()
