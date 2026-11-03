"""SQLite connection and schema management for EngramOS."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import re
import sqlite3
from pathlib import Path
from typing import Iterator

from .normalize import normalize_slot_name, normalize_value

logger = logging.getLogger(__name__)

DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema.sql"
_CONNECTION_PRAGMA = re.compile(
    r"(?im)^[ \t]*PRAGMA\s+(?:journal_mode|foreign_keys)\s*=\s*[^;]+;[ \t]*$"
)


class DatabaseConfigurationError(RuntimeError):
    """Raised when SQLite cannot enforce a required connection setting."""


def connect(database: str | Path, *, timeout: float = 5.0) -> sqlite3.Connection:
    """Open a SQLite connection with foreign-key checks enabled and verified.

    All application code should open connections through this function so the
    schema's foreign-key constraints apply to every connection, not only the
    connection that initialized the database.
    """
    db_name = str(database)
    conn = sqlite3.connect(db_name, timeout=timeout)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        row = conn.execute("PRAGMA foreign_keys").fetchone()
        if row is None or row[0] != 1:
            raise DatabaseConfigurationError(
                "SQLite foreign-key enforcement could not be enabled"
            )
    except BaseException:
        conn.close()
        raise
    return conn


@contextmanager
def connection(
    database: str | Path, *, timeout: float = 5.0
) -> Iterator[sqlite3.Connection]:
    """Yield a configured connection, committing on success and rolling back on error."""
    conn = connect(database, timeout=timeout)
    try:
        yield conn
    except BaseException:
        conn.rollback()
        raise
    else:
        conn.commit()
    finally:
        conn.close()


@contextmanager
def transaction(
    database: str | Path,
    *,
    timeout: float = 5.0,
    immediate: bool = True,
) -> Iterator[sqlite3.Connection]:
    """Run work atomically in a fresh connection.

    ``BEGIN IMMEDIATE`` reserves the SQLite write lock before the operation
    reads state, which is useful for later interval and idempotency writes.
    """
    with connection(database, timeout=timeout) as conn:
        conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        yield conn


def initialize_database(
    database: str | Path,
    *,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    timeout: float = 5.0,
) -> None:
    """Create the schema atomically and enable WAL for file-backed databases.

    The schema retains its PRAGMA declarations for direct SQLite use. They are
    removed from the initialization script because journal mode must be set
    outside a transaction, while foreign keys are configured by ``connect``.
    """
    schema_file = Path(schema_path)
    schema_sql = schema_file.read_text(encoding="utf-8")
    schema_sql = _CONNECTION_PRAGMA.sub("", schema_sql)

    conn = connect(database, timeout=timeout)
    try:
        if str(database) != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL").fetchone()
        conn.executescript(f"BEGIN IMMEDIATE;\n{schema_sql}\nCOMMIT;")
        conn.execute("BEGIN IMMEDIATE")
        _ensure_embedding_content_hash_column(conn)
        _ensure_llm_cache_attribution_columns(conn)
        _rebuild_budget_summary_view(conn)
        _backfill_fact_observations(conn)
        conn.commit()
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        logger.exception("Failed to initialize EngramOS SQLite schema")
        raise
    finally:
        conn.close()

    logger.info("Initialized EngramOS SQLite schema")


def _backfill_fact_observations(conn: sqlite3.Connection) -> None:
    """Migrate pre-existing slot rows into the additive observation ledger."""
    rows = conn.execute(
        """SELECT f.* FROM facts AS f
           WHERE f.key_type = 'slot'
             AND NOT EXISTS (
               SELECT 1 FROM fact_observations AS o WHERE o.observation_id = f.fact_id
             )"""
    ).fetchall()
    for row in rows:
        entity, attribute = normalize_slot_name(row["entity"] or "", row["attribute"] or "")
        if not entity or not attribute or not row["value"]:
            raise DatabaseConfigurationError(
                f"Cannot preserve malformed legacy slot fact {row['fact_id']!r}"
            )
        if row["cardinality"] not in ("single", "multi"):
            raise DatabaseConfigurationError(
                f"Legacy slot fact {row['fact_id']!r} has invalid cardinality"
            )
        last_turn_id = row["last_confirmed_turn_id"] or row["turn_id"]
        source = conn.execute(
            "SELECT history_id FROM turns WHERE turn_id = ?", (row["turn_id"],)
        ).fetchone()
        last_source = conn.execute(
            "SELECT history_id FROM turns WHERE turn_id = ?", (last_turn_id,)
        ).fetchone()
        if (
            source is None
            or last_source is None
            or source["history_id"] != row["history_id"]
            or last_source["history_id"] != row["history_id"]
        ):
            raise DatabaseConfigurationError(
                f"Legacy slot fact {row['fact_id']!r} has inconsistent turn provenance"
            )
        conn.execute(
            """INSERT INTO fact_observations (
                   observation_id, history_id, turn_id, fact_seq, entity, attribute,
                   value, value_norm, cardinality, kind, key_text, valid_at,
                   effective_at, canonical_fact_id, last_confirmed_at,
                   last_confirmed_turn_id, confirmation_count, recorded_at, trust,
                   quarantined, quarantine_reason, confidence
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row["fact_id"], row["history_id"], row["turn_id"], row["fact_seq"],
                entity, attribute, row["value"], row["value_norm"] or normalize_value(row["value"]),
                row["cardinality"], row["kind"], row["key_text"], row["valid_from"],
                row["valid_from"], row["fact_id"],
                row["last_confirmed_at"] if row["last_confirmed_at"] is not None else row["valid_from"],
                last_turn_id, max(1, int(row["confirmation_count"])), row["recorded_at"],
                row["trust"], row["quarantined"], row["quarantine_reason"], row["confidence"],
            ),
        )


def _ensure_embedding_content_hash_column(conn: sqlite3.Connection) -> None:
    """Add the nullable vector source digest to existing databases.

    Legacy vectors remain in place for audit, but a NULL digest makes them
    ineligible for retrieval until the next indexing pass refreshes them.
    """
    columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(embeddings)").fetchall()
    }
    if "content_hash" not in columns:
        conn.execute("ALTER TABLE embeddings ADD COLUMN content_hash TEXT")


def _ensure_llm_cache_attribution_columns(conn: sqlite3.Connection) -> None:
    """Add run/phase attribution to databases created by earlier builds."""
    columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(llm_cache)").fetchall()
    }
    if "phase_tag" not in columns:
        conn.execute(
            "ALTER TABLE llm_cache ADD COLUMN phase_tag TEXT NOT NULL DEFAULT 'unspecified'"
        )
    if "run_id" not in columns:
        conn.execute(
            "ALTER TABLE llm_cache ADD COLUMN run_id TEXT NOT NULL DEFAULT 'unspecified'"
        )
    ledger_columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(llm_call_ledger)").fetchall()
    }
    if "model" not in ledger_columns:
        conn.execute(
            "ALTER TABLE llm_call_ledger ADD COLUMN model TEXT NOT NULL DEFAULT 'unspecified'"
        )


def _rebuild_budget_summary_view(conn: sqlite3.Connection) -> None:
    """Keep summaries inclusive of failed costs and outstanding reservations."""
    conn.execute("DROP VIEW IF EXISTS budget_summary")
    conn.execute(
        """CREATE VIEW budget_summary AS
           SELECT task_tag, model, phase_tag, run_id, COUNT(*) AS api_calls,
                  SUM(in_tokens) AS total_in_tokens,
                  SUM(out_tokens) AS total_out_tokens,
                  ROUND(SUM(credits_spent), 4) AS credits_spent,
                  ROUND(SUM(credits_reserved), 4) AS credits_reserved
           FROM (
             SELECT task_tag, model, phase_tag, run_id, in_tokens, out_tokens,
                    cost_credits AS credits_spent, 0.0 AS credits_reserved
             FROM llm_cache
             UNION ALL
             SELECT task_tag, model, phase_tag, run_id, 0 AS in_tokens, 0 AS out_tokens,
                    CASE WHEN status='failed' THEN actual_credits ELSE 0.0 END,
                    CASE WHEN status='reserved' THEN reserved_credits ELSE 0.0 END
             FROM llm_call_ledger WHERE status IN ('reserved','failed')
           ) GROUP BY task_tag, model, phase_tag, run_id"""
    )
