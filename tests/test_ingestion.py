"""File-backed SQLite integration tests for turn ingestion and recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engramOS.db import connection, initialize_database  # noqa: E402
from engramOS.extraction import (  # noqa: E402
    ExtractedFact,
    ExtractionExecutionError,
    ExtractionResponseError,
)
from engramOS.gate import (  # noqa: E402
    IncompleteSignalEvaluation,
    SignalResult,
    SignalStatus,
    evaluate_and_record_turn,
    evaluate_turn,
)
import engramOS.gate as gate_module  # noqa: E402
from engramOS.eligibility import turn_is_eligible  # noqa: E402
from engramOS.retrieval import search_turns_fts  # noqa: E402
from engramOS import ingestion as ingestion_module  # noqa: E402
from engramOS.ingestion import (  # noqa: E402
    IngestionOrderError,
    IngestionRecoveryError,
    InvalidTurnError,
    MissingFactExtractor,
    MissingGateEvaluator,
    PersistedTurn,
    QuarantinedTurnError,
    TemporalWriteError,
    TurnConflictError,
    TurnInput,
    ingest_turn,
    ingest_turns,
    ingest_turns_only,
)
from engramOS.temporal import get_current_facts, get_fact_history  # noqa: E402


def fact(**updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "session_id": "session-1",
        "turn_idx": 0,
        "entity": "user",
        "attribute": "home_city",
        "value": "Paris",
        "cardinality": "single",
        "kind": "state",
        "as_of": None,
        "replaces": None,
        "confidence": 0.9,
    }
    result.update(updates)
    return result


class FakeEvaluator:
    def __init__(self, bridge: bool = False, policy: bool = False) -> None:
        self.bridge = bridge
        self.policy = policy
        self.calls = 0

    def fingerprint_inputs(self, conn: sqlite3.Connection, turn: sqlite3.Row) -> str:
        return "test-fingerprint-v1"

    def evaluate_bridge(self, conn: sqlite3.Connection, turn: sqlite3.Row) -> SignalResult:
        self.calls += 1
        return SignalResult(SignalStatus.FIRED if self.bridge else SignalStatus.CLEAR)

    def evaluate_policy(self, conn: sqlite3.Connection, turn: sqlite3.Row) -> SignalResult:
        self.calls += 1
        return SignalResult(SignalStatus.FIRED if self.policy else SignalStatus.CLEAR)


class RaisingEvaluator(FakeEvaluator):
    def evaluate_bridge(self, conn: sqlite3.Connection, turn: sqlite3.Row) -> SignalResult:
        self.calls += 1
        raise RuntimeError("detector unavailable")


class FakeExtractor:
    def __init__(self, response: object = "[]", *, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls = 0
        self.turns: list[object] = []

    def extract(self, turn: object, prompt: object) -> str:
        self.calls += 1
        self.turns.append(turn)
        if self.error is not None:
            raise self.error
        return self.response  # type: ignore[return-value]


class SequencedExtractor:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.prompts: list[object] = []

    def extract(self, turn: object, prompt: object) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


class PerTurnEvaluator:
    def __init__(self, quarantine_turn_ids: set[str]) -> None:
        self.quarantine_turn_ids = quarantine_turn_ids

    def fingerprint_inputs(self, conn: sqlite3.Connection, turn: sqlite3.Row) -> str:
        return "per-turn-fingerprint-v1"

    def evaluate_bridge(self, conn: sqlite3.Connection, turn: sqlite3.Row) -> SignalResult:
        return SignalResult(
            SignalStatus.FIRED
            if turn["turn_id"] in self.quarantine_turn_ids
            else SignalStatus.CLEAR
        )

    def evaluate_policy(self, conn: sqlite3.Connection, turn: sqlite3.Row) -> SignalResult:
        return SignalResult(SignalStatus.CLEAR)


class StubClock:
    def __init__(self, *values: float) -> None:
        self.values = iter(values)
        self.calls = 0
        self.last = values[-1] if values else 0.0

    def time(self) -> float:
        self.calls += 1
        try:
            self.last = next(self.values)
        except StopIteration:
            self.last += 1.0
        return self.last


class TurnIngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = str(Path(self.temp_dir.name) / "ingestion.sqlite3")
        initialize_database(self.database)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_turn(
        self,
        *,
        history: str = "history-1",
        session: str = "session-1",
        index: int = 0,
        session_ts: float = 1_700_000_000.0,
        content: str = "I live in Paris.",
        role: str = "user",
        source: str = "user_turn",
        session_date: str | None = "2023-11-14",
        turn_id: str | None = None,
        token_count: int | None = None,
    ) -> TurnInput:
        return TurnInput(
            history_id=history,
            session_id=session,
            turn_idx=index,
            role=role,
            raw_content=content,
            session_unix_ts=session_ts,
            source_type=source,
            session_date=session_date,
            turn_id=turn_id,
            token_count=token_count,
        )

    def extractor(self, rows: list[dict[str, object]] | None = None) -> FakeExtractor:
        return FakeExtractor(json.dumps(rows if rows is not None else [fact()]))

    def rows(self, sql: str, params: tuple[object, ...] = ()) -> list[sqlite3.Row]:
        with connection(self.database) as conn:
            return conn.execute(sql, params).fetchall()

    def test_valid_user_turn_preserves_content_and_provenance_hash(self) -> None:
        turn = self.make_turn(content="  I live in Paris.\n")
        result = ingest_turn(
            self.database, turn, evaluator=FakeEvaluator(), extractor=self.extractor()
        )
        stored = self.rows("SELECT * FROM turns WHERE turn_id=?", (result.turn_id,))[0]
        expected_hash = hashlib.sha256(turn.raw_content.encode("utf-8")).hexdigest()
        self.assertEqual(stored["raw_content"], turn.raw_content)
        self.assertEqual(stored["event_time"], turn.session_unix_ts)
        self.assertEqual(stored["hash"], expected_hash)
        self.assertEqual(result.state, "completed")

    def test_assistant_tool_and_external_document_turns_are_supported(self) -> None:
        samples = (
            ("assistant", "assistant_turn"),
            ("tool", "tool"),
            ("user", "external_doc"),
        )
        for idx, (role, source) in enumerate(samples):
            with self.subTest(role=role, source=source):
                turn = self.make_turn(
                    history=f"history-{idx}",
                    session=f"session-{idx}",
                    role=role,
                    source=source,
                    content=f"source {source}",
                )
                extractor = self.extractor([])
                result = ingest_turn(
                    self.database, turn, evaluator=FakeEvaluator(), extractor=extractor
                )
                self.assertEqual(result.extracted_count, 0)
                self.assertEqual(extractor.calls, 1)
                gate = self.rows(
                    "SELECT signals_fired, decision FROM gate_events WHERE turn_id=?",
                    (result.turn_id,),
                )[0]
                self.assertEqual(gate["signals_fired"], 1 if source in ("tool", "external_doc") else 0)
                self.assertEqual(gate["decision"], "store")

    def test_invalid_incomplete_and_noncanonical_turn_inputs_are_rejected(self) -> None:
        samples = (
            self.make_turn(content="  "),
            self.make_turn(role="robot"),
            self.make_turn(source="webpage"),
            self.make_turn(index=-1),
            self.make_turn(index=10**1000),
            self.make_turn(token_count=1 << 63),
            self.make_turn(history="bad:id"),
            self.make_turn(session_date="2023-02-30"),
            self.make_turn(turn_id="forged"),
        )
        for turn in samples:
            with self.subTest(turn=turn):
                with self.assertRaises((InvalidTurnError, TypeError)):
                    ingest_turn(self.database, turn, evaluator=FakeEvaluator(), extractor=self.extractor())
        self.assertEqual(self.rows("SELECT * FROM turns"), [])

    def test_history_session_and_turn_order_are_consistent(self) -> None:
        first = self.make_turn(index=0)
        ingest_turn(self.database, first, evaluator=FakeEvaluator(), extractor=self.extractor([]))
        with self.assertRaises(InvalidTurnError):
            ingest_turn(
                self.database,
                self.make_turn(index=1, session_ts=first.session_unix_ts + 5),
                evaluator=FakeEvaluator(),
                extractor=self.extractor([]),
            )
        with self.assertRaises(IngestionOrderError):
            ingest_turn(
                self.database,
                self.make_turn(index=0, session="earlier", session_ts=first.session_unix_ts - 10),
                evaluator=FakeEvaluator(),
                extractor=self.extractor([]),
            )
        with self.assertRaises(TurnConflictError):
            ingest_turn(
                self.database,
                self.make_turn(index=0, content="different source text"),
                evaluator=FakeEvaluator(),
                extractor=self.extractor([]),
            )

    def test_inconsistent_session_metadata_in_batch_is_rejected_before_inserts(self) -> None:
        turns = (
            self.make_turn(index=0),
            self.make_turn(index=1, session_ts=1_700_000_010.0),
        )
        with self.assertRaises(InvalidTurnError):
            ingest_turns(
                self.database,
                turns,
                evaluator=FakeEvaluator(),
                extractor=self.extractor([]),
            )
        self.assertEqual(self.rows("SELECT * FROM turns"), [])

    def test_same_payload_replay_is_idempotent_and_different_payload_conflicts(self) -> None:
        turn = self.make_turn()
        evaluator, extractor = FakeEvaluator(), self.extractor()
        first = ingest_turn(self.database, turn, evaluator=evaluator, extractor=extractor)
        replay = ingest_turn(self.database, turn, evaluator=FakeEvaluator(), extractor=self.extractor([]))
        self.assertFalse(first.replayed)
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.observation_ids, first.observation_ids)
        self.assertEqual(len(self.rows("SELECT * FROM turns")), 1)
        self.assertEqual(len(self.rows("SELECT * FROM gate_events")), 1)
        self.assertEqual(len(self.rows("SELECT * FROM fact_observations")), 1)
        self.assertEqual(evaluator.calls, 2)
        with self.assertRaises(TurnConflictError):
            ingest_turn(
                self.database,
                self.make_turn(content="I moved to Rome."),
                evaluator=FakeEvaluator(),
                extractor=self.extractor([]),
            )

    def test_batch_turns_are_ordered_before_hash_chain_and_processing(self) -> None:
        later = self.make_turn(session="session-b", session_ts=200.0, content="later")
        earlier = self.make_turn(session="session-a", session_ts=100.0, content="earlier")
        results = ingest_turns(
            self.database,
            (later, earlier),
            evaluator=FakeEvaluator(),
            extractor=self.extractor([]),
            recorded_at=500.0,
        )
        self.assertEqual([result.turn_id for result in results], ["history-1:session-a:0", "history-1:session-b:0"])
        rows = self.rows("SELECT turn_id, prev_hash, hash FROM turns ORDER BY event_time")
        self.assertIsNone(rows[0]["prev_hash"])
        self.assertEqual(rows[1]["prev_hash"], rows[0]["hash"])

    def test_turns_only_persists_exact_rows_hash_chain_and_pending_jobs(self) -> None:
        turns = (
            self.make_turn(
                session="later", index=1, session_ts=200.0, content="second",
                role="assistant", source="assistant_turn",
            ),
            self.make_turn(
                session="earlier", index=0, session_ts=100.0, content="first",
                source="external_doc",
            ),
        )
        results = ingest_turns_only(self.database, turns, recorded_at=500.0)
        self.assertEqual(
            results,
            (
                PersistedTurn("history-1:earlier:0", False),
                PersistedTurn("history-1:later:1", False),
            ),
        )
        rows = self.rows(
            "SELECT turn_id, event_time, source_type, raw_content, prev_hash, hash "
            "FROM turns ORDER BY event_time, session_id, turn_idx, turn_id"
        )
        self.assertEqual(
            [(row["turn_id"], row["event_time"], row["source_type"]) for row in rows],
            [
                ("history-1:earlier:0", 100.0, "external_doc"),
                ("history-1:later:1", 201.0, "assistant_turn"),
            ],
        )
        previous_hash = ""
        for row in rows:
            expected_hash = hashlib.sha256(
                (previous_hash + row["raw_content"]).encode("utf-8")
            ).hexdigest()
            self.assertEqual(row["prev_hash"], previous_hash or None)
            self.assertEqual(row["hash"], expected_hash)
            previous_hash = expected_hash
        self.assertEqual(self.rows("SELECT * FROM gate_events"), [])
        self.assertEqual(self.rows("SELECT * FROM facts"), [])
        self.assertEqual(self.rows("SELECT * FROM fact_observations"), [])
        self.assertEqual(
            [row["state"] for row in self.rows(
                "SELECT state FROM ingestion_jobs ORDER BY turn_id"
            )],
            ["pending_gate", "pending_gate"],
        )

    def test_turns_only_replay_conflict_order_and_batch_validation(self) -> None:
        first = self.make_turn(session="a", session_ts=100.0, content="first")
        second = self.make_turn(session="b", session_ts=200.0, content="second")
        original = ingest_turns_only(self.database, (first, second), recorded_at=500.0)
        replay = ingest_turns_only(self.database, (first, second), recorded_at=600.0)
        self.assertEqual([item.replayed for item in original], [False, False])
        self.assertEqual([item.replayed for item in replay], [True, True])
        self.assertEqual(len(self.rows("SELECT * FROM turns")), 2)
        self.assertEqual(len(self.rows("SELECT * FROM ingestion_jobs")), 2)
        with self.assertRaises(TurnConflictError):
            ingest_turns_only(
                self.database,
                (self.make_turn(session="a", session_ts=100.0, content="conflict"),),
            )

        order_db = str(Path(self.temp_dir.name) / "order.sqlite3")
        initialize_database(order_db)
        ingest_turns_only(order_db, (second,), recorded_at=500.0)
        with self.assertRaises(IngestionOrderError):
            ingest_turns_only(order_db, (first,), recorded_at=600.0)

        validation_db = str(Path(self.temp_dir.name) / "validation.sqlite3")
        initialize_database(validation_db)
        with self.assertRaises(InvalidTurnError):
            ingest_turns_only(
                validation_db,
                (
                    self.make_turn(history="history-a", session="a", session_ts=100),
                    self.make_turn(history="history-b", session="b", session_ts=200),
                ),
            )
        with self.assertRaises((InvalidTurnError, TypeError)):
            ingest_turns_only(
                validation_db,
                (
                    self.make_turn(session="a", session_ts=100),
                    self.make_turn(session="b", session_ts=200, content="  "),
                ),
            )
        with connection(validation_db) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM ingestion_jobs").fetchone()[0], 0)

    def test_turns_only_records_can_be_upgraded_through_normal_ingestion(self) -> None:
        turns = (
            self.make_turn(session="a", session_ts=100.0, content="I live in Paris."),
            self.make_turn(session="b", index=1, session_ts=100.0, content="I enjoy reading."),
        )
        baseline = ingest_turns_only(self.database, turns, recorded_at=500.0)
        self.assertEqual([item.replayed for item in baseline], [False, False])
        extractor = FakeExtractor("[]")
        upgraded = ingest_turns(
            self.database,
            turns,
            evaluator=FakeEvaluator(),
            extractor=extractor,
            recorded_at=600.0,
        )
        self.assertEqual([item.turn_id for item in upgraded], [item.turn_id for item in baseline])
        self.assertEqual([item.replayed for item in upgraded], [True, True])
        self.assertEqual(extractor.calls, 2)
        self.assertEqual(
            [row["state"] for row in self.rows(
                "SELECT state FROM ingestion_jobs ORDER BY turn_id"
            )],
            ["completed", "completed"],
        )
        self.assertEqual(
            [row["decision"] for row in self.rows(
                "SELECT decision FROM gate_events ORDER BY event_id"
            )],
            ["store", "store"],
        )

    def test_turns_only_pending_state_upgrades_to_store_eligibility(self) -> None:
        turn = self.make_turn(
            history="turns-only-eligibility", session="turns-only-eligibility",
            content="upgrade eligibility marker",
        )
        persisted = ingest_turns_only(self.database, (turn,))[0]
        with connection(self.database) as conn:
            self.assertFalse(turn_is_eligible(conn, persisted.turn_id))
        self.assertEqual(
            {item.ref_id for item in search_turns_fts(
                self.database, turn.history_id, "upgrade eligibility", gated=False
            )},
            {persisted.turn_id},
        )
        self.assertEqual(
            search_turns_fts(self.database, turn.history_id, "upgrade eligibility"), []
        )
        result = ingest_turns(
            self.database, (turn,), evaluator=FakeEvaluator(),
            extractor=FakeExtractor("[]"),
        )[0]
        self.assertEqual(result.state, "completed")
        with connection(self.database) as conn:
            self.assertTrue(turn_is_eligible(conn, persisted.turn_id))
        self.assertEqual(
            {item.ref_id for item in search_turns_fts(
                self.database, turn.history_id, "upgrade eligibility"
            )},
            {persisted.turn_id},
        )

    def test_duplicate_identical_turn_in_batch_is_processed_once(self) -> None:
        turn = self.make_turn()
        evaluator = FakeEvaluator()
        extractor = self.extractor()
        results = ingest_turns(
            self.database,
            (turn, turn),
            evaluator=evaluator,
            extractor=extractor,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(extractor.calls, 1)
        self.assertEqual(evaluator.calls, 2)
        self.assertEqual(len(self.rows("SELECT * FROM gate_events")), 1)
        self.assertEqual(len(self.rows("SELECT * FROM fact_observations")), 1)

    def test_multi_turn_batch_updates_slots_sequentially_and_preserves_temporal_order(self) -> None:
        first = self.make_turn(index=0, content="I live in Paris.")
        second = self.make_turn(index=1, content="I moved to Rome.")
        extractor = SequencedExtractor(
            json.dumps([fact(value="Paris")]),
            json.dumps([fact(turn_idx=1, value="Rome", replaces="Paris")]),
        )
        results = ingest_turns(
            self.database,
            (second, first),
            evaluator=FakeEvaluator(),
            extractor=extractor,
        )
        self.assertEqual(
            [result.turn_id for result in results],
            ["history-1:session-1:0", "history-1:session-1:1"],
        )
        self.assertIn("user.home_city", extractor.prompts[1].system)
        history = get_fact_history(
            self.database, "history-1", entity="user", attribute="home_city"
        )
        self.assertEqual([row["value"] for row in history], ["Paris", "Rome"])
        self.assertEqual(len(self.rows("SELECT * FROM gate_events")), 2)

    def test_invalid_later_batch_extraction_prevents_all_temporal_writes(self) -> None:
        first = self.make_turn(index=0)
        second = self.make_turn(index=1, content="I moved to Rome.")
        extractor = SequencedExtractor(
            json.dumps([fact(value="Paris")]),
            json.dumps([fact(turn_idx=9, value="Rome")]),
            json.dumps([fact(turn_idx=9, value="Rome")]),
        )
        with self.assertRaises(ExtractionResponseError):
            ingest_turns(
                self.database,
                (first, second),
                evaluator=FakeEvaluator(),
                extractor=extractor,
            )
        self.assertEqual(self.rows("SELECT * FROM fact_observations"), [])
        self.assertEqual(self.rows("SELECT * FROM facts"), [])
        self.assertEqual(
            [row["state"] for row in self.rows("SELECT state FROM ingestion_jobs ORDER BY turn_id")],
            ["approved", "approved"],
        )

    def test_missing_or_failed_gate_evaluation_never_reaches_extractor(self) -> None:
        turn = self.make_turn()
        extractor = self.extractor()
        with self.assertRaises(MissingGateEvaluator):
            ingest_turn(self.database, turn, extractor=extractor)
        self.assertEqual(extractor.calls, 0)
        self.assertEqual(self.rows("SELECT * FROM gate_events"), [])
        with self.assertRaises(ValueError):
            ingest_turn(self.database, turn, evaluator=RaisingEvaluator(), extractor=extractor)
        self.assertEqual(extractor.calls, 0)
        self.assertEqual(self.rows("SELECT * FROM gate_events"), [])

    def test_failed_extractor_recovery_reuses_store_gate_event(self) -> None:
        turn = self.make_turn()
        with self.assertRaises(ExtractionExecutionError):
            ingest_turn(
                self.database,
                turn,
                evaluator=FakeEvaluator(),
                extractor=FakeExtractor(error=RuntimeError("local provider unavailable")),
            )
        before = self.rows("SELECT event_id FROM gate_events WHERE turn_id=?", ("history-1:session-1:0",))
        self.assertEqual(len(before), 1)
        result = ingest_turn(
            self.database,
            turn,
            evaluator=None,
            extractor=self.extractor([fact()]),
        )
        after = self.rows("SELECT event_id FROM gate_events WHERE turn_id=?", ("history-1:session-1:0",))
        self.assertEqual(len(after), 1)
        self.assertEqual(result.gate_event_id, before[0]["event_id"])

    def test_quarantine_stops_before_extraction_and_is_sticky(self) -> None:
        turn = self.make_turn(source="tool", role="tool")
        extractor = self.extractor()
        with self.assertRaises(QuarantinedTurnError):
            ingest_turn(
                self.database,
                turn,
                evaluator=FakeEvaluator(bridge=True),
                extractor=extractor,
            )
        self.assertEqual(extractor.calls, 0)
        self.assertEqual(self.rows("SELECT * FROM facts"), [])
        quarantine_event = self.rows("SELECT event_id FROM gate_events WHERE decision='quarantine'")[0]
        # A later store event remains in the audit history and cannot revoke quarantine.
        evaluate_and_record_turn(self.database, "history-1:session-1:0", FakeEvaluator())
        retry_extractor = self.extractor()
        with self.assertRaises(QuarantinedTurnError):
            ingest_turn(self.database, turn, extractor=retry_extractor)
        self.assertEqual(retry_extractor.calls, 0)
        events = self.rows("SELECT decision FROM gate_events ORDER BY event_id")
        self.assertEqual([row["decision"] for row in events], ["quarantine", "store"])
        self.assertIsNotNone(quarantine_event)

    def test_existing_store_gate_event_is_reconciled_and_reused(self) -> None:
        turn = self.make_turn()
        with self.assertRaises(MissingGateEvaluator):
            ingest_turn(self.database, turn)
        event = evaluate_and_record_turn(
            self.database, "history-1:session-1:0", FakeEvaluator()
        )
        result = ingest_turn(self.database, turn, extractor=self.extractor([]))
        self.assertEqual(result.gate_event_id, event.event_id)
        self.assertEqual(len(self.rows("SELECT * FROM gate_events")), 1)
        state = self.rows("SELECT state FROM ingestion_jobs")[0]["state"]
        self.assertEqual(state, "completed")

    def test_extraction_is_validated_as_a_whole_before_any_fact_write(self) -> None:
        response = json.dumps([fact(), fact(turn_idx=9, value="Rome")])
        with self.assertRaises(ExtractionResponseError):
            ingest_turn(
                self.database,
                self.make_turn(),
                evaluator=FakeEvaluator(),
                extractor=FakeExtractor(response),
            )
        self.assertEqual(self.rows("SELECT * FROM fact_observations"), [])
        state = self.rows("SELECT state, extraction_json FROM ingestion_jobs")[0]
        self.assertEqual(state["state"], "approved")
        self.assertIsNone(state["extraction_json"])

    def test_empty_extraction_is_successfully_staged_and_completed(self) -> None:
        result = ingest_turn(
            self.database,
            self.make_turn(),
            evaluator=FakeEvaluator(),
            extractor=self.extractor([]),
        )
        self.assertEqual(result.extracted_count, 0)
        self.assertEqual(result.applied_count, 0)
        self.assertEqual(self.rows("SELECT state FROM ingestion_jobs")[0]["state"], "completed")

    def test_facts_enter_temporal_engine_sequentially_for_same_slot(self) -> None:
        response = [fact(value="Paris"), fact(value="Rome", replaces="Paris")]
        result = ingest_turn(
            self.database,
            self.make_turn(),
            evaluator=FakeEvaluator(),
            extractor=self.extractor(response),
        )
        history = get_fact_history(self.database, "history-1", entity="user", attribute="home_city")
        self.assertEqual([row["value"] for row in history], ["Paris", "Rome"])
        self.assertEqual(history[0]["valid_until"], history[1]["valid_from"])
        current = get_current_facts(self.database, "history-1", entity="user", attribute="home_city")
        self.assertEqual([row["value"] for row in current], ["Rome"])
        self.assertEqual(result.applied_count, 2)

    def test_backdated_extraction_uses_temporal_api(self) -> None:
        result = ingest_turn(
            self.database,
            self.make_turn(),
            evaluator=FakeEvaluator(),
            extractor=self.extractor([fact(as_of="2020-01-02")]),
        )
        history = get_fact_history(self.database, "history-1", entity="user", attribute="home_city")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["valid_from"], 1577923200.0 + 0.0)
        self.assertEqual(result.applied_count, 1)

    def test_future_as_of_extraction_fails_before_staging_or_fact_writes(self) -> None:
        turn = self.make_turn(content="I started living in Paris in the future")
        turn_id = f"{turn.history_id}:{turn.session_id}:{turn.turn_idx}"

        with self.assertRaises(ExtractionResponseError):
            ingest_turn(
                self.database,
                turn,
                evaluator=FakeEvaluator(),
                extractor=self.extractor([fact(as_of="2099-01-01")]),
            )

        facts = self.rows("SELECT fact_id FROM facts WHERE turn_id=?", (turn_id,))
        observations = self.rows(
            "SELECT observation_id FROM fact_observations WHERE turn_id=?", (turn_id,)
        )
        job = self.rows(
            "SELECT state, extraction_json FROM ingestion_jobs WHERE turn_id=?",
            (turn_id,),
        )[0]
        source_turns = self.rows("SELECT turn_id FROM turns WHERE turn_id=?", (turn_id,))
        gate_events = self.rows(
            "SELECT decision, signals_fired FROM gate_events WHERE turn_id=?",
            (turn_id,),
        )

        self.assertEqual(facts, [])
        self.assertEqual(observations, [])
        self.assertEqual((job["state"], job["extraction_json"]), ("approved", None))
        self.assertEqual([row["turn_id"] for row in source_turns], [turn_id])
        self.assertEqual(
            [(row["decision"], row["signals_fired"]) for row in gate_events],
            [("store", 0)],
        )

    def test_interruption_after_extraction_before_staging_can_retry(self) -> None:
        turn = self.make_turn()
        extractor = self.extractor()
        with patch("engramOS.ingestion._stage_extraction", side_effect=RuntimeError("crash before stage")):
            with self.assertRaisesRegex(RuntimeError, "crash before stage"):
                ingest_turn(self.database, turn, evaluator=FakeEvaluator(), extractor=extractor)
        self.assertEqual(self.rows("SELECT * FROM fact_observations"), [])
        result = ingest_turn(self.database, turn, extractor=self.extractor())
        self.assertEqual(result.state, "completed")
        self.assertEqual(len(self.rows("SELECT * FROM gate_events")), 1)
        self.assertEqual(len(self.rows("SELECT * FROM fact_observations")), 1)

    def test_partial_fact_batch_failure_recovers_without_duplicate_observations(self) -> None:
        turn = self.make_turn()
        response = [fact(value="Paris"), fact(value="Rome")]
        with connection(self.database) as conn:
            conn.execute(
                """CREATE TRIGGER fail_second_observation BEFORE INSERT ON fact_observations
                   WHEN NEW.fact_seq = 1 BEGIN SELECT RAISE(ABORT, 'simulated interruption'); END"""
            )
        with self.assertRaises(TemporalWriteError):
            ingest_turn(
                self.database,
                turn,
                evaluator=FakeEvaluator(),
                extractor=self.extractor(response),
            )
        self.assertEqual(len(self.rows("SELECT * FROM fact_observations")), 1)
        job = self.rows("SELECT state, applied_count FROM ingestion_jobs")[0]
        self.assertEqual((job["state"], job["applied_count"]), ("staged", 1))
        with connection(self.database) as conn:
            conn.execute("DROP TRIGGER fail_second_observation")
        result = ingest_turn(self.database, turn, extractor=self.extractor([]))
        self.assertEqual(result.applied_count, 2)
        self.assertEqual(len(self.rows("SELECT * FROM fact_observations")), 2)
        self.assertEqual(len(self.rows("SELECT * FROM gate_events")), 1)

    def test_failure_before_completion_replays_temporal_writes_idempotently(self) -> None:
        turn = self.make_turn()
        with connection(self.database) as conn:
            conn.execute(
                """CREATE TRIGGER fail_completion BEFORE UPDATE OF state ON ingestion_jobs
                   WHEN NEW.state = 'completed'
                   BEGIN SELECT RAISE(ABORT, 'simulated interruption'); END"""
            )
        with self.assertRaises(sqlite3.IntegrityError):
            ingest_turn(
                self.database,
                turn,
                evaluator=FakeEvaluator(),
                extractor=self.extractor(),
            )
        self.assertEqual(len(self.rows("SELECT * FROM fact_observations")), 1)
        with connection(self.database) as conn:
            conn.execute("DROP TRIGGER fail_completion")
        result = ingest_turn(self.database, turn, extractor=self.extractor([]))
        self.assertTrue(result.replayed)
        self.assertEqual(len(self.rows("SELECT * FROM fact_observations")), 1)
        self.assertEqual(len(self.rows("SELECT * FROM gate_events")), 1)

    def test_cross_history_facts_remain_isolated(self) -> None:
        for history, city in (("history-1", "Paris"), ("history-2", "Rome")):
            turn = self.make_turn(history=history, content=f"I live in {city}")
            ingest_turn(
                self.database,
                turn,
                evaluator=FakeEvaluator(),
                extractor=self.extractor([fact(value=city)]),
            )
        self.assertEqual(
            [row["value"] for row in get_current_facts(self.database, "history-1")], ["Paris"]
        )
        self.assertEqual(
            [row["value"] for row in get_current_facts(self.database, "history-2")], ["Rome"]
        )

    def test_isolate_quarantines_middle_turn_and_replays_results_in_order(self) -> None:
        other_history = self.make_turn(
            history="history-other", session="other", session_ts=50.0, content="other history"
        )
        ingest_turn(
            self.database,
            other_history,
            evaluator=FakeEvaluator(),
            extractor=self.extractor(
                [fact(session_id="other", attribute="home_city", value="Rome")]
            ),
        )
        turns = (
            self.make_turn(session="a", session_ts=100.0, content="clean before"),
            self.make_turn(
                session="b", session_ts=200.0, role="tool", source="tool",
                content="untrusted middle",
            ),
            self.make_turn(session="c", session_ts=300.0, content="clean after"),
        )
        middle_id = "history-1:b:0"
        extractor = SequencedExtractor(
            json.dumps([fact(session_id="a", attribute="home_city", value="Paris")]),
            json.dumps([fact(session_id="c", attribute="hobby", value="Reading")]),
        )
        results = ingest_turns(
            self.database,
            turns,
            evaluator=PerTurnEvaluator({middle_id}),
            extractor=extractor,
            quarantine_policy="isolate",
        )

        self.assertEqual(
            [(item.turn_id, item.state) for item in results],
            [
                ("history-1:a:0", "completed"),
                (middle_id, "quarantined"),
                ("history-1:c:0", "completed"),
            ],
        )
        quarantined = results[1]
        self.assertGreater(quarantined.gate_event_id, 0)
        self.assertEqual(
            (
                quarantined.extracted_count,
                quarantined.applied_count,
                quarantined.observation_ids,
                quarantined.replayed,
            ),
            (0, 0, (), False),
        )
        self.assertEqual(len(extractor.prompts), 2)
        self.assertEqual(
            [row["value"] for row in get_current_facts(
                self.database, "history-1", entity="user", attribute="home_city"
            )],
            ["Paris"],
        )
        self.assertEqual(
            [row["value"] for row in get_current_facts(
                self.database, "history-1", entity="user", attribute="hobby"
            )],
            ["Reading"],
        )
        with connection(self.database) as conn:
            middle_job = conn.execute(
                "SELECT state FROM ingestion_jobs WHERE turn_id=?", (middle_id,)
            ).fetchone()
            middle_event = conn.execute(
                "SELECT event_id, decision FROM gate_events WHERE turn_id=? ORDER BY event_id",
                (middle_id,),
            ).fetchall()
            middle_fact_count = conn.execute(
                "SELECT COUNT(*) FROM facts WHERE turn_id=?", (middle_id,)
            ).fetchone()[0]
            middle_observation_count = conn.execute(
                "SELECT COUNT(*) FROM fact_observations WHERE turn_id=?", (middle_id,)
            ).fetchone()[0]
        self.assertEqual(middle_job["state"], "quarantined")
        self.assertEqual(
            [(row["event_id"], row["decision"]) for row in middle_event],
            [(quarantined.gate_event_id, "quarantine")],
        )
        self.assertEqual((middle_fact_count, middle_observation_count), (0, 0))
        with connection(self.database) as conn:
            observation_rows = conn.execute(
                "SELECT observation_id, value FROM fact_observations "
                "WHERE history_id='history-1' ORDER BY turn_id"
            ).fetchall()
        self.assertEqual(
            [(row["observation_id"], row["value"]) for row in observation_rows],
            [
                ("history-1:a:0:fact:0", "Paris"),
                ("history-1:c:0:fact:0", "Reading"),
            ],
        )
        self.assertEqual(len(observation_rows), 2)
        self.assertEqual(
            [
                row["value"]
                for row in get_current_facts(
                    self.database,
                    "history-other",
                    entity="user",
                    attribute="home_city",
                )
            ],
            ["Rome"],
        )

        replay = ingest_turns(
            self.database,
            turns,
            extractor=self.extractor([]),
            quarantine_policy="isolate",
        )
        self.assertEqual(
            [(item.turn_id, item.state, item.replayed) for item in replay],
            [
                ("history-1:a:0", "completed", True),
                (middle_id, "quarantined", True),
                ("history-1:c:0", "completed", True),
            ],
        )
        self.assertEqual(len(self.rows("SELECT * FROM turns")), 4)
        self.assertEqual(len(self.rows("SELECT * FROM gate_events")), 4)
        self.assertEqual(len(self.rows("SELECT * FROM fact_observations")), 3)

    def test_isolated_quarantine_slot_is_absent_from_later_existing_slots_prompt(self) -> None:
        before = self.make_turn(session="a", session_ts=100.0, content="clean before")
        middle = self.make_turn(
            session="b", session_ts=200.0, role="tool", source="tool",
            content="untrusted source had a secret code",
        )
        after = self.make_turn(session="c", session_ts=300.0, content="clean after")
        ingest_turn(
            self.database,
            before,
            evaluator=FakeEvaluator(),
            extractor=self.extractor([fact(session_id="a", attribute="home_city", value="Paris")]),
        )
        ingest_turn(
            self.database,
            middle,
            evaluator=FakeEvaluator(),
            extractor=self.extractor([fact(session_id="b", attribute="secret_code", value="1234")]),
        )
        quarantine = evaluate_and_record_turn(
            self.database,
            "history-1:b:0",
            FakeEvaluator(bridge=True),
            created_at=900.0,
        )
        self.assertEqual(quarantine.decision, "quarantine")

        later_extractor = SequencedExtractor(
            json.dumps([fact(session_id="c", attribute="hobby", value="Reading")])
        )
        results = ingest_turns(
            self.database,
            (before, middle, after),
            evaluator=FakeEvaluator(),
            extractor=later_extractor,
            quarantine_policy="isolate",
        )
        self.assertEqual([item.state for item in results], ["completed", "quarantined", "completed"])
        self.assertEqual(len(later_extractor.prompts), 1)
        prompt = later_extractor.prompts[0].system
        self.assertIn("user.home_city", prompt)
        self.assertNotIn("user.secret_code", prompt)

    def test_default_raise_policy_still_stops_the_batch_on_quarantine(self) -> None:
        turns = (
            self.make_turn(session="a", session_ts=100.0),
            self.make_turn(session="b", session_ts=200.0, role="tool", source="tool"),
            self.make_turn(session="c", session_ts=300.0),
        )
        with self.assertRaises(QuarantinedTurnError):
            ingest_turns(
                self.database,
                turns,
                evaluator=PerTurnEvaluator({"history-1:b:0"}),
                extractor=self.extractor([]),
                quarantine_policy="raise",
            )
        self.assertEqual(len(self.rows("SELECT * FROM fact_observations")), 0)

    def test_isolate_handles_quarantine_detected_mid_apply_and_keeps_other_turns(self) -> None:
        turns = (
            self.make_turn(session="a", session_ts=100.0),
            self.make_turn(session="b", session_ts=200.0),
            self.make_turn(session="c", session_ts=300.0),
        )
        response = SequencedExtractor(
            json.dumps([fact(session_id="a", attribute="city", value="Paris")]),
            json.dumps([fact(session_id="b", attribute="city", value="Rome")]),
            json.dumps([fact(session_id="c", attribute="city", value="Tokyo")]),
        )
        original_apply = ingestion_module._apply_batch_facts
        middle_id = "history-1:b:0"
        injected = False

        def quarantine_between_gate_and_apply(
            database: str,
            batches: object,
            *,
            recorded_at: float | None,
            timeout: float,
        ) -> None:
            nonlocal injected
            first_turn_id = batches[0][0].turn_id  # type: ignore[index]
            if first_turn_id == middle_id and not injected:
                injected = True
                evaluate_and_record_turn(
                    database,
                    middle_id,
                    FakeEvaluator(bridge=True, policy=True),
                    created_at=700.0,
                )
            original_apply(
                database,
                batches,
                recorded_at=recorded_at,
                timeout=timeout,
            )  # type: ignore[arg-type]

        with patch.object(
            ingestion_module,
            "_apply_batch_facts",
            side_effect=quarantine_between_gate_and_apply,
        ):
            results = ingest_turns(
                self.database,
                turns,
                evaluator=FakeEvaluator(),
                extractor=response,
                quarantine_policy="isolate",
            )
        self.assertTrue(injected)
        self.assertEqual([item.state for item in results], ["completed", "quarantined", "completed"])
        self.assertEqual(
            [row["value"] for row in get_fact_history(self.database, "history-1")],
            ["Paris", "Tokyo"],
        )
        self.assertEqual(
            [row["value"] for row in get_current_facts(self.database, "history-1")],
            ["Tokyo"],
        )
        with connection(self.database) as conn:
            middle = conn.execute(
                "SELECT state FROM ingestion_jobs WHERE turn_id=?", (middle_id,)
            ).fetchone()
            middle_fact = conn.execute(
                "SELECT quarantined FROM facts WHERE turn_id=?", (middle_id,)
            ).fetchall()
        self.assertEqual(middle["state"], "quarantined")
        self.assertEqual(middle_fact, [])

    def test_isolate_propagates_unrelated_quarantine_and_extractor_errors(self) -> None:
        turn = self.make_turn()
        with self.assertRaisesRegex(RuntimeError, "extractor failed"):
            ingest_turns(
                self.database,
                (turn,),
                evaluator=FakeEvaluator(),
                extractor=FakeExtractor(error=RuntimeError("extractor failed")),
                quarantine_policy="isolate",
            )

        def different_turn_quarantine(*args: object, **kwargs: object) -> None:
            raise QuarantinedTurnError("another-turn", 999)

        with patch.object(
            ingestion_module,
            "_apply_batch_facts",
            side_effect=different_turn_quarantine,
        ):
            with self.assertRaisesRegex(QuarantinedTurnError, "another-turn"):
                ingest_turns(
                    self.database,
                    (turn,),
                    extractor=self.extractor([fact()]),
                    quarantine_policy="isolate",
                )

        with patch.object(
            ingestion_module,
            "_apply_batch_facts",
            side_effect=sqlite3.OperationalError("database failure"),
        ):
            with self.assertRaisesRegex(sqlite3.OperationalError, "database failure"):
                ingest_turns(
                    self.database,
                    (turn,),
                    extractor=self.extractor([]),
                    quarantine_policy="isolate",
                )

    def test_malformed_batch_persists_nothing_under_both_quarantine_policies(self) -> None:
        for policy in ("raise", "isolate"):
            with self.subTest(policy=policy):
                database = str(Path(self.temp_dir.name) / f"malformed-{policy}.sqlite3")
                initialize_database(database)
                turns = (
                    self.make_turn(session="a", session_ts=100.0),
                    self.make_turn(session="b", session_ts=200.0, content="  "),
                )
                with self.assertRaises(InvalidTurnError):
                    ingest_turns(
                        database,
                        turns,
                        evaluator=FakeEvaluator(),
                        extractor=self.extractor([]),
                        quarantine_policy=policy,  # type: ignore[call-arg]
                    )
                with connection(database) as conn:
                    self.assertEqual(conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0], 0)
                    self.assertEqual(
                        conn.execute("SELECT COUNT(*) FROM ingestion_jobs").fetchone()[0], 0
                    )
                    self.assertEqual(conn.execute("SELECT COUNT(*) FROM gate_events").fetchone()[0], 0)

    def test_supplied_recorded_at_propagates_without_clock_reads(self) -> None:
        instant = 1_234_567_890.0
        turns = (
            self.make_turn(session="a", session_ts=100.0, content="Paris then Rome"),
            self.make_turn(session="b", session_ts=200.0, content="now London"),
        )
        extractor = SequencedExtractor(
            json.dumps(
                [
                    fact(session_id="a", attribute="home_city", value="Paris"),
                    fact(session_id="a", attribute="home_city", value="Rome"),
                ]
            ),
            json.dumps([fact(session_id="b", attribute="home_city", value="London")]),
        )
        ingestion_clock = StubClock(8000.0)
        gate_clock = StubClock(9000.0)
        with patch("engramOS.ingestion.clock", ingestion_clock), patch(
            "engramOS.gate.time", SimpleNamespace(time=gate_clock.time)
        ):
            ingest_turns(
                self.database,
                turns,
                evaluator=FakeEvaluator(),
                extractor=extractor,
                recorded_at=instant,
            )
        self.assertEqual((ingestion_clock.calls, gate_clock.calls), (0, 0))
        turn_rows = self.rows("SELECT recorded_at FROM turns ORDER BY event_time")
        observations = self.rows(
            "SELECT recorded_at FROM fact_observations ORDER BY effective_at, observation_id"
        )
        facts = self.rows(
            "SELECT recorded_at, expired_at FROM facts WHERE key_type='slot' ORDER BY valid_from"
        )
        events = self.rows("SELECT created_at FROM gate_events ORDER BY event_id")
        jobs = self.rows("SELECT updated_at FROM ingestion_jobs ORDER BY turn_id")
        self.assertEqual([row["recorded_at"] for row in turn_rows], [instant, instant])
        self.assertEqual([row["recorded_at"] for row in observations], [instant] * 3)
        self.assertEqual(
            [(row["recorded_at"], row["expired_at"]) for row in facts],
            [(instant, instant), (instant, instant), (instant, None)],
        )
        self.assertEqual([row["created_at"] for row in events], [instant, instant])
        self.assertEqual([row["updated_at"] for row in jobs], [instant, instant])

    def test_omitted_recorded_at_uses_each_turn_timestamp_and_replay_is_stable(self) -> None:
        turns = (
            self.make_turn(session="a", session_ts=100.0),
            self.make_turn(session="b", session_ts=200.0),
        )
        extractor = SequencedExtractor(
            json.dumps([fact(session_id="a", attribute="home_city", value="Paris")]),
            json.dumps([fact(session_id="b", attribute="hobby", value="Reading")]),
        )
        first_clock = StubClock(1001.0, 1002.0)
        gate_clock = StubClock(5000.0)
        with patch("engramOS.ingestion.clock", first_clock), patch(
            "engramOS.gate.time", SimpleNamespace(time=gate_clock.time)
        ):
            ingest_turns(self.database, turns, evaluator=FakeEvaluator(), extractor=extractor)
        turn_times = {
            row["turn_id"]: row["recorded_at"]
            for row in self.rows("SELECT turn_id, recorded_at FROM turns")
        }
        observation_rows = self.rows(
            "SELECT turn_id, recorded_at FROM fact_observations ORDER BY turn_id"
        )
        self.assertEqual(gate_clock.calls, 2)
        self.assertEqual(
            [(row["turn_id"], row["recorded_at"]) for row in observation_rows],
            sorted(turn_times.items()),
        )

        original_observation_times = [row["recorded_at"] for row in observation_rows]
        replay_clock = StubClock(7001.0, 7002.0)
        with patch("engramOS.ingestion.clock", replay_clock):
            first_replay = ingest_turns(self.database, turns)
            second_replay = ingest_turns(self.database, turns)
        self.assertEqual([item.replayed for item in first_replay], [True, True])
        self.assertEqual([item.replayed for item in second_replay], [True, True])
        self.assertEqual(
            [row["recorded_at"] for row in self.rows(
                "SELECT recorded_at FROM fact_observations ORDER BY turn_id"
            )],
            original_observation_times,
        )

    def test_staged_and_partial_recovery_use_explicit_or_stored_recorded_at(self) -> None:
        instant = 123.0
        staged_db = str(Path(self.temp_dir.name) / "staged-recorded.sqlite3")
        initialize_database(staged_db)
        staged_turn = self.make_turn(session="staged", session_ts=100.0)
        staged_extractor = self.extractor(
            [fact(session_id="staged", value="Paris"), fact(session_id="staged", value="Rome")]
        )
        with patch(
            "engramOS.ingestion._apply_batch_facts",
            side_effect=RuntimeError("stop after staging"),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop after staging"):
                ingest_turn(
                    staged_db,
                    staged_turn,
                    evaluator=FakeEvaluator(),
                    extractor=staged_extractor,
                    recorded_at=instant,
                )
        retry = ingest_turn(
            staged_db,
            staged_turn,
            extractor=self.extractor([]),
            recorded_at=456.0,
        )
        self.assertEqual(retry.state, "completed")
        with connection(staged_db) as conn:
            self.assertEqual(
                [row[0] for row in conn.execute(
                    "SELECT recorded_at FROM fact_observations ORDER BY fact_seq"
                )],
                [456.0, 456.0],
            )
            self.assertEqual(
                conn.execute("SELECT recorded_at FROM turns").fetchone()[0], instant
            )

        partial_db = str(Path(self.temp_dir.name) / "partial-recorded.sqlite3")
        initialize_database(partial_db)
        partial_turn = self.make_turn(session="partial", session_ts=300.0)
        with connection(partial_db) as conn:
            conn.execute(
                """CREATE TRIGGER fail_second_recorded_observation
                   BEFORE INSERT ON fact_observations WHEN NEW.fact_seq = 1
                   BEGIN SELECT RAISE(ABORT, 'stop after first observation'); END"""
            )
        with self.assertRaises(TemporalWriteError):
            ingest_turn(
                partial_db,
                partial_turn,
                evaluator=FakeEvaluator(),
                extractor=self.extractor(
                    [
                        fact(session_id="partial", value="Paris"),
                        fact(session_id="partial", value="Rome"),
                    ]
                ),
                recorded_at=instant,
            )
        with connection(partial_db) as conn:
            conn.execute("DROP TRIGGER fail_second_recorded_observation")
        recovered = ingest_turn(partial_db, partial_turn, extractor=self.extractor([]))
        self.assertEqual(recovered.state, "completed")
        with connection(partial_db) as conn:
            observations = conn.execute(
                "SELECT recorded_at FROM fact_observations ORDER BY fact_seq"
            ).fetchall()
            count = conn.execute("SELECT COUNT(*) FROM fact_observations").fetchone()[0]
            stored_turn_time = conn.execute("SELECT recorded_at FROM turns").fetchone()[0]
        self.assertEqual(count, 2)
        self.assertEqual([row["recorded_at"] for row in observations], [instant, instant])
        self.assertEqual(stored_turn_time, instant)

    def test_role_source_matrix_through_turns_only(self) -> None:
        roles = ("user", "assistant", "system", "tool")
        source_types = ("user_turn", "assistant_turn", "system_policy", "tool", "external_doc")
        accepted = {
            ("user", "user_turn"), ("user", "tool"), ("user", "external_doc"),
            ("assistant", "assistant_turn"), ("assistant", "tool"), ("assistant", "external_doc"),
            ("system", "system_policy"), ("system", "tool"), ("system", "external_doc"),
            ("tool", "tool"), ("tool", "external_doc"),
        }
        for role in roles:
            for source_type in source_types:
                pair = (role, source_type)
                with self.subTest(role=role, source_type=source_type):
                    database = str(
                        Path(self.temp_dir.name) / f"matrix-only-{role}-{source_type}.sqlite3"
                    )
                    initialize_database(database)
                    turn = self.make_turn(
                        history="matrix-only",
                        session="session",
                        role=role,
                        source=source_type,
                    )
                    if pair in accepted:
                        persisted = ingest_turns_only(database, (turn,))
                        self.assertEqual(len(persisted), 1)
                        with connection(database) as conn:
                            row = conn.execute(
                                "SELECT role, source_type FROM turns WHERE turn_id=?",
                                (persisted[0].turn_id,),
                            ).fetchone()
                        self.assertEqual((row["role"], row["source_type"]), pair)
                    else:
                        with self.assertRaises(InvalidTurnError):
                            ingest_turns_only(database, (turn,))
                        with connection(database) as conn:
                            for table in (
                                "turns", "ingestion_jobs", "gate_events", "facts",
                                "fact_observations",
                            ):
                                self.assertEqual(
                                    conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0],
                                    0,
                                    table,
                                )

    def test_role_source_matrix_through_ingest_turns_matches_turns_only(self) -> None:
        roles = ("user", "assistant", "system", "tool")
        source_types = ("user_turn", "assistant_turn", "system_policy", "tool", "external_doc")
        accepted = {
            ("user", "user_turn"), ("user", "tool"), ("user", "external_doc"),
            ("assistant", "assistant_turn"), ("assistant", "tool"), ("assistant", "external_doc"),
            ("system", "system_policy"), ("system", "tool"), ("system", "external_doc"),
            ("tool", "tool"), ("tool", "external_doc"),
        }
        for role in roles:
            for source_type in source_types:
                pair = (role, source_type)
                expected = pair in accepted
                with self.subTest(role=role, source_type=source_type):
                    only_database = str(
                        Path(self.temp_dir.name) / f"matrix-compare-only-{role}-{source_type}.sqlite3"
                    )
                    normal_database = str(
                        Path(self.temp_dir.name) / f"matrix-normal-{role}-{source_type}.sqlite3"
                    )
                    initialize_database(only_database)
                    initialize_database(normal_database)
                    turn = self.make_turn(
                        history="matrix-normal",
                        session="session",
                        role=role,
                        source=source_type,
                    )
                    try:
                        only_accepted = bool(ingest_turns_only(only_database, (turn,)))
                    except InvalidTurnError:
                        only_accepted = False
                    evaluator = FakeEvaluator()
                    extractor = FakeExtractor("[]")
                    try:
                        ingest_turns(
                            normal_database,
                            (turn,),
                            evaluator=evaluator,
                            extractor=extractor,
                        )
                        normal_accepted = True
                    except InvalidTurnError:
                        normal_accepted = False
                    self.assertEqual(only_accepted, normal_accepted)
                    self.assertEqual(only_accepted, expected)
                    if expected:
                        with connection(only_database) as conn:
                            self.assertEqual(
                                conn.execute(
                                    "SELECT source_type FROM turns"
                                ).fetchone()[0],
                                source_type,
                            )
                        with connection(normal_database) as conn:
                            self.assertEqual(
                                conn.execute(
                                    "SELECT source_type FROM turns"
                                ).fetchone()[0],
                                source_type,
                            )
                    else:
                        self.assertEqual(evaluator.calls, 0)
                        self.assertEqual(extractor.calls, 0)
                        for database in (only_database, normal_database):
                            with connection(database) as conn:
                                for table in (
                                    "turns", "ingestion_jobs", "gate_events", "facts",
                                    "fact_observations",
                                ):
                                    self.assertEqual(
                                        conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0],
                                        0,
                                        (database, table),
                                    )

    def test_single_turn_entry_point_rejects_tool_claiming_user_turn(self) -> None:
        rejected = self.make_turn(
            history="single-rejected", role="tool", source="user_turn"
        )
        evaluator = FakeEvaluator()
        extractor = FakeExtractor("[]")
        with self.assertRaises(InvalidTurnError):
            ingest_turn(
                self.database, rejected, evaluator=evaluator, extractor=extractor
            )
        self.assertEqual((evaluator.calls, extractor.calls), (0, 0))
        for table in ("turns", "ingestion_jobs", "gate_events", "facts", "fact_observations"):
            self.assertEqual(self.rows(f"SELECT COUNT(*) FROM {table}")[0][0], 0)

        accepted = self.make_turn(
            history="single-accepted", role="assistant", source="assistant_turn"
        )
        result = ingest_turn(
            self.database,
            accepted,
            evaluator=FakeEvaluator(),
            extractor=FakeExtractor("[]"),
        )
        self.assertEqual(result.state, "completed")

    def test_role_source_rejection_message_names_both_values(self) -> None:
        turn = self.make_turn(role="tool", source="user_turn")
        with self.assertRaises(InvalidTurnError) as raised:
            ingest_turns_only(self.database, (turn,))
        message = str(raised.exception)
        self.assertIn("tool", message)
        self.assertIn("user_turn", message)

    def test_invalid_role_source_pair_makes_both_batch_entry_points_atomic(self) -> None:
        turns = (
            self.make_turn(history="atomic", session="same", index=0, session_ts=100.0,
                           role="user", source="user_turn"),
            self.make_turn(history="atomic", session="same", index=1, session_ts=100.0,
                           role="assistant", source="assistant_turn"),
            self.make_turn(history="atomic", session="same", index=2, session_ts=100.0,
                           role="tool", source="tool"),
            self.make_turn(history="atomic", session="same", index=3, session_ts=100.0,
                           role="system", source="user_turn"),
        )
        only_database = str(Path(self.temp_dir.name) / "atomic-only.sqlite3")
        normal_database = str(Path(self.temp_dir.name) / "atomic-normal.sqlite3")
        initialize_database(only_database)
        initialize_database(normal_database)
        with self.assertRaises(InvalidTurnError):
            ingest_turns_only(only_database, turns)
        evaluator = FakeEvaluator()
        extractor = FakeExtractor("[]")
        with self.assertRaises(InvalidTurnError):
            ingest_turns(
                normal_database,
                turns,
                evaluator=evaluator,
                extractor=extractor,
            )
        self.assertEqual((evaluator.calls, extractor.calls), (0, 0))
        for database in (only_database, normal_database):
            with connection(database) as conn:
                for table in (
                    "turns", "ingestion_jobs", "gate_events", "facts", "fact_observations",
                ):
                    self.assertEqual(
                        conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0],
                        0,
                        (database, table),
                    )

    def test_omitted_source_type_derives_per_role_and_replays_as_explicit(self) -> None:
        expected_source_types = {
            "user": "user_turn",
            "assistant": "assistant_turn",
            "system": "system_policy",
            "tool": "tool",
        }
        for role, expected_source in expected_source_types.items():
            with self.subTest(role=role):
                default_database = str(Path(self.temp_dir.name) / f"default-{role}.sqlite3")
                explicit_database = str(Path(self.temp_dir.name) / f"explicit-{role}.sqlite3")
                initialize_database(default_database)
                initialize_database(explicit_database)
                values = {
                    "history_id": f"default-{role}",
                    "session_id": "session",
                    "turn_idx": 0,
                    "role": role,
                    "raw_content": f"same content for {role}",
                    "session_unix_ts": 100.0,
                    "session_date": "1970-01-01",
                }
                default_turn = TurnInput(**values)
                explicit_turn = TurnInput(**values, source_type=expected_source)
                ingest_turns_only(default_database, (default_turn,), recorded_at=500.0)
                ingest_turns_only(explicit_database, (explicit_turn,), recorded_at=500.0)
                row_query = (
                    "SELECT turn_id, history_id, session_id, turn_idx, role, raw_content, "
                    "source_type, event_time, recorded_at, prev_hash, hash FROM turns"
                )
                with connection(default_database) as default_conn, connection(explicit_database) as explicit_conn:
                    default_row = tuple(default_conn.execute(row_query).fetchone())
                    explicit_row = tuple(explicit_conn.execute(row_query).fetchone())
                    default_payload_hash = default_conn.execute(
                        "SELECT payload_hash FROM ingestion_jobs"
                    ).fetchone()[0]
                    explicit_payload_hash = explicit_conn.execute(
                        "SELECT payload_hash FROM ingestion_jobs"
                    ).fetchone()[0]
                self.assertEqual(default_row, explicit_row)
                self.assertEqual(default_row[6], expected_source)
                self.assertEqual(default_payload_hash, explicit_payload_hash)
                self.assertTrue(
                    ingest_turns_only(default_database, (explicit_turn,), recorded_at=600.0)[0].replayed
                )
                self.assertTrue(
                    ingest_turns_only(explicit_database, (default_turn,), recorded_at=600.0)[0].replayed
                )

    def test_source_gate_behavior_for_external_document_and_tool_pairs_is_unchanged(self) -> None:
        turns = (
            self.make_turn(
                history="source-gate-user-doc",
                role="user",
                source="external_doc",
                content="pasted meeting notes",
            ),
            self.make_turn(
                history="source-gate-tool",
                role="tool",
                source="tool",
                content="tool result",
            ),
        )
        for turn in turns:
            result = ingest_turns(
                self.database,
                (turn,),
                evaluator=FakeEvaluator(),
                extractor=FakeExtractor("[]"),
            )[0]
            self.assertEqual(result.state, "completed")
        events = self.rows(
            "SELECT turn_id, signal_source, signals_fired, decision FROM gate_events ORDER BY event_id"
        )
        self.assertEqual(
            [(row["signal_source"], row["signals_fired"], row["decision"]) for row in events],
            [(1.0, 1, "store"), (1.0, 1, "store")],
        )

    def test_omitted_recorded_at_clocks_do_not_replace_source_times_on_projection(self) -> None:
        turns = (
            self.make_turn(
                history="omitted-clock-projection",
                session="same-session",
                index=0,
                session_ts=100.0,
                content="I live in Paris",
            ),
            self.make_turn(
                history="omitted-clock-projection",
                session="same-session",
                index=1,
                session_ts=100.0,
                content="I moved to Rome",
            ),
        )
        # Seed pending turns with separate transaction timestamps so the clocks
        # used by the omitted-recorded_at replay cannot be confused with them.
        ingest_turns_only(self.database, (turns[0],), recorded_at=700.0)
        ingest_turns_only(self.database, (turns[1],), recorded_at=800.0)
        extractor = SequencedExtractor(
            json.dumps([fact(session_id="same-session", turn_idx=0, value="Paris")]),
            json.dumps([fact(session_id="same-session", turn_idx=1, value="Rome")]),
        )
        ingestion_values = tuple(float(value) for value in range(2001, 2011))
        gate_values = (3001.0, 3002.0)
        ingestion_clock = StubClock(*ingestion_values)
        gate_clock = StubClock(*gate_values)
        with patch("engramOS.ingestion.clock", ingestion_clock), patch(
            "engramOS.gate.time", SimpleNamespace(time=gate_clock.time)
        ):
            results = ingest_turns(
                self.database,
                turns,
                evaluator=FakeEvaluator(),
                extractor=extractor,
            )
            self.assertEqual([item.replayed for item in results], [True, True])
            self.assertEqual(ingestion_clock.calls, 10)
            self.assertEqual(gate_clock.calls, 2)

            turns_rows = self.rows(
                "SELECT turn_id, event_time, recorded_at FROM turns ORDER BY turn_idx"
            )
            observation_rows = self.rows(
                "SELECT o.observation_id, o.recorded_at, t.recorded_at AS turn_recorded_at "
                "FROM fact_observations AS o JOIN turns AS t ON t.turn_id=o.turn_id "
                "ORDER BY t.turn_idx"
            )
            fact_rows = self.rows(
                "SELECT value, recorded_at, expired_at FROM facts "
                "WHERE key_type='slot' ORDER BY valid_from"
            )
            event_rows = self.rows(
                "SELECT t.turn_idx, g.created_at, g.decision, g.signals_fired "
                "FROM gate_events AS g JOIN turns AS t ON t.turn_id=g.turn_id "
                "ORDER BY t.turn_idx"
            )
            job_rows = self.rows(
                "SELECT t.turn_idx, j.state, j.updated_at FROM ingestion_jobs AS j "
                "JOIN turns AS t ON t.turn_id=j.turn_id ORDER BY t.turn_idx"
            )
            self.assertEqual(
                [(row["event_time"], row["recorded_at"]) for row in turns_rows],
                [(100.0, 700.0), (101.0, 800.0)],
            )
            self.assertEqual(
                [
                    (row["observation_id"], row["recorded_at"], row["turn_recorded_at"])
                    for row in observation_rows
                ],
                [
                    ("omitted-clock-projection:same-session:0:fact:0", 700.0, 700.0),
                    ("omitted-clock-projection:same-session:1:fact:0", 800.0, 800.0),
                ],
            )
            self.assertEqual(
                [(row["value"], row["recorded_at"], row["expired_at"]) for row in fact_rows],
                [("Paris", 700.0, 800.0), ("Rome", 800.0, None)],
            )
            self.assertEqual(
                [(row["turn_idx"], row["created_at"], row["decision"], row["signals_fired"])
                 for row in event_rows],
                [(0, 3001.0, "store", 0), (1, 3002.0, "store", 0)],
            )
            self.assertEqual(
                [(row["turn_idx"], row["state"], row["updated_at"]) for row in job_rows],
                [(0, "completed", 2009.0), (1, "completed", 2010.0)],
            )
            stub_values = set(ingestion_values) | set(gate_values)
            persisted_fact_times = [row["recorded_at"] for row in fact_rows]
            persisted_observation_times = [row["recorded_at"] for row in observation_rows]
            self.assertTrue(all(value not in stub_values for value in persisted_fact_times))
            self.assertTrue(all(value not in stub_values for value in persisted_observation_times))

            before_facts = [tuple(row) for row in fact_rows]
            before_observations = [tuple(row) for row in observation_rows]
            replay = ingest_turns(self.database, turns)
            self.assertEqual([item.replayed for item in replay], [True, True])
            self.assertEqual(
                [tuple(row) for row in self.rows(
                    "SELECT value, recorded_at, expired_at FROM facts "
                    "WHERE key_type='slot' ORDER BY valid_from"
                )],
                before_facts,
            )
        self.assertEqual(
            [tuple(row) for row in self.rows(
                "SELECT o.observation_id, o.recorded_at, t.recorded_at AS turn_recorded_at "
                "FROM fact_observations AS o JOIN turns AS t ON t.turn_id=o.turn_id "
                "ORDER BY t.turn_idx"
            )],
            before_observations,
        )

    def test_provenance_matrix_matches_both_ingestion_entrypoints_and_gate(self) -> None:
        pairs = {
            ("user", "user_turn"): True,
            ("user", "assistant_turn"): False,
            ("user", "system_policy"): False,
            ("user", "tool"): True,
            ("user", "external_doc"): True,
            ("assistant", "user_turn"): False,
            ("assistant", "assistant_turn"): True,
            ("assistant", "system_policy"): False,
            ("assistant", "tool"): True,
            ("assistant", "external_doc"): True,
            ("system", "user_turn"): False,
            ("system", "assistant_turn"): False,
            ("system", "system_policy"): True,
            ("system", "tool"): True,
            ("system", "external_doc"): True,
            ("tool", "user_turn"): False,
            ("tool", "assistant_turn"): False,
            ("tool", "system_policy"): False,
            ("tool", "tool"): True,
            ("tool", "external_doc"): True,
        }
        gate_error = getattr(
            gate_module, "InconsistentTurnProvenanceError", ValueError
        )
        for idx, ((role, source_type), accepted) in enumerate(pairs.items()):
            turn = self.make_turn(
                history=f"provenance-parity-{idx}",
                session=f"parity-session-{idx}",
                role=role,
                source=source_type,
                content=f"parity content {role} {source_type}",
            )
            with self.subTest(role=role, source_type=source_type, accepted=accepted):
                if accepted:
                    ingest_turns_only(self.database, (turn,))
                    result = ingest_turns(
                        self.database,
                        (turn,),
                        evaluator=FakeEvaluator(),
                        extractor=FakeExtractor("[]"),
                    )[0]
                    self.assertEqual(result.state, "completed")
                    gate_evaluation = evaluate_turn(
                        self.database, result.turn_id, FakeEvaluator()
                    )
                    self.assertEqual(gate_evaluation.source_type, source_type)
                else:
                    with self.assertRaises(InvalidTurnError):
                        ingest_turns_only(self.database, (turn,))
                    with self.assertRaises(InvalidTurnError):
                        ingest_turns(
                            self.database,
                            (turn,),
                            evaluator=FakeEvaluator(),
                            extractor=FakeExtractor("[]"),
                        )
                    turn_id = f"{turn.history_id}:{turn.session_id}:{turn.turn_idx}"
                    with connection(self.database) as conn:
                        conn.execute(
                            """INSERT INTO turns (turn_id, history_id, session_id,
                               turn_idx, role, raw_content, source_type, event_time,
                               recorded_at, hash)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (turn_id, turn.history_id, turn.session_id, turn.turn_idx,
                             role, turn.raw_content, source_type,
                             turn.session_unix_ts, turn.session_unix_ts + 10,
                             f"invalid-{idx}"),
                        )
                    with self.assertRaises(gate_error):
                        evaluate_turn(self.database, turn_id, FakeEvaluator())

        invalid_inputs = (
            ("invalid-role", "observer", "user_turn"),
            ("invalid-source", "user", "unverified_source"),
        )
        for idx, (label, role, source_type) in enumerate(invalid_inputs):
            turn = self.make_turn(
                history=f"unknown-parity-{label}", session=f"unknown-{label}",
                role=role, source=source_type,
            )
            with self.subTest(role=role, source_type=source_type):
                with self.assertRaises(InvalidTurnError):
                    ingest_turns_only(self.database, (turn,))
                with self.assertRaises(InvalidTurnError):
                    ingest_turns(
                        self.database, (turn,), evaluator=FakeEvaluator(),
                        extractor=FakeExtractor("[]"),
                    )
                turn_id = f"unknown-gate:{label}:0"
                with connection(self.database) as conn:
                    conn.execute(
                        """INSERT INTO turns (turn_id, history_id, session_id, turn_idx,
                           role, raw_content, source_type, event_time, recorded_at, hash)
                           VALUES (?, 'unknown-gate', ?, 0, ?, 'unknown metadata', ?, 1, 1, ?)""",
                        (turn_id, label, role, source_type, f"unknown-hash-{idx}"),
                    )
                with self.assertRaises(gate_error):
                    evaluate_turn(self.database, turn_id, FakeEvaluator())

    def test_failed_gate_leaves_pending_turn_unquarantined_and_can_retry(self) -> None:
        turn = self.make_turn(
            history="failed-gate-history", session="failed-gate-session",
            content="pending recovery marker",
        )
        turn_id = f"{turn.history_id}:{turn.session_id}:{turn.turn_idx}"
        extractor = FakeExtractor("[]")
        with self.assertRaises(IncompleteSignalEvaluation):
            ingest_turns(
                self.database, (turn,), evaluator=RaisingEvaluator(), extractor=extractor,
                quarantine_policy="isolate",
            )
        self.assertEqual(extractor.calls, 0)
        with connection(self.database) as conn:
            job = conn.execute(
                "SELECT state FROM ingestion_jobs WHERE turn_id=?", (turn_id,)
            ).fetchone()
            source = conn.execute(
                "SELECT base_trust FROM turns WHERE turn_id=?", (turn_id,)
            ).fetchone()
            self.assertEqual(job["state"], "pending_gate")
            self.assertEqual(source["base_trust"], 0.6)
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM gate_events WHERE turn_id=? AND decision='quarantine'",
                    (turn_id,),
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM facts WHERE turn_id=?", (turn_id,)).fetchone()[0],
                0,
            )
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM fact_observations WHERE turn_id=?", (turn_id,)
                ).fetchone()[0],
                0,
            )
            self.assertFalse(turn_is_eligible(conn, turn_id))
        self.assertEqual(
            {item.ref_id for item in search_turns_fts(
                self.database, turn.history_id, "pending recovery", gated=False
            )},
            {turn_id},
        )
        self.assertEqual(
            search_turns_fts(self.database, turn.history_id, "pending recovery"), []
        )
        result = ingest_turns(
            self.database, (turn,), evaluator=FakeEvaluator(), extractor=FakeExtractor("[]")
        )[0]
        self.assertEqual(result.state, "completed")
        with connection(self.database) as conn:
            self.assertTrue(turn_is_eligible(conn, turn_id))

    def test_isolation_recheck_with_missing_gate_is_recovery_not_quarantine(self) -> None:
        turn = self.make_turn(
            history="isolation-gate-history", session="isolation-gate-session",
            content="isolation marker",
        )
        turn_id = f"{turn.history_id}:{turn.session_id}:{turn.turn_idx}"
        with patch.object(
            ingestion_module,
            "turn_has_gate_decision",
            create=True,
            side_effect=[True, False],
        ) as has_gate:
            with self.assertRaises(IngestionRecoveryError):
                ingest_turns(
                    self.database, (turn,), evaluator=FakeEvaluator(),
                    extractor=FakeExtractor("[]"), quarantine_policy="isolate",
                )
        self.assertIsNotNone(has_gate)
        self.assertEqual(has_gate.call_count, 2)
        with connection(self.database) as conn:
            state = conn.execute(
                "SELECT state FROM ingestion_jobs WHERE turn_id=?", (turn_id,)
            ).fetchone()[0]
            self.assertEqual(state, "approved")
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM gate_events WHERE turn_id=? AND decision='quarantine'",
                    (turn_id,),
                ).fetchone()[0],
                0,
            )

    def test_missing_gate_at_staging_application_and_completion_checkpoints(self) -> None:

        checkpoints = ("stage", "apply", "record", "complete", "replay")
        for index, checkpoint in enumerate(checkpoints):
            with self.subTest(checkpoint=checkpoint):
                database = str(Path(self.temp_dir.name) / f"missing-gate-{checkpoint}.sqlite3")
                initialize_database(database)
                current = self.make_turn(
                    history=f"checkpoint-history-{index}",
                    session=f"checkpoint-session-{index}",
                    content=f"checkpoint {checkpoint}",
                )
                current_id = f"{current.history_id}:{current.session_id}:0"
                ingest_turns_only(database, (current,))
                validated = ingestion_module._validate_turn(current, recorded_at=None)
                if checkpoint in ("apply", "record", "complete"):
                    with connection(database) as conn:
                        conn.execute(
                            """UPDATE ingestion_jobs SET state='staged', extraction_json='[]',
                               extraction_hash=?, fact_count=1, applied_count=0
                               WHERE turn_id=?""",
                            (hashlib.sha256(b"[]").hexdigest(), current_id),
                        )
                if checkpoint == "replay":
                    with connection(database) as conn:
                        conn.execute(
                            """UPDATE ingestion_jobs SET state='completed', extraction_json='[]',
                               extraction_hash=?, fact_count=0, applied_count=0
                               WHERE turn_id=?""",
                            (hashlib.sha256(b"[]").hexdigest(), current_id),
                        )
                    with self.assertRaises(IngestionRecoveryError):
                        ingest_turns(
                            database, (current,), evaluator=None, extractor=None,
                            quarantine_policy="isolate",
                        )
                    self.assertEqual(
                        self._state_from(database, current_id), "completed"
                    )
                    continue
                candidate = ExtractedFact(
                    session_id=current.session_id, turn_idx=current.turn_idx,
                    entity="user", attribute="city", value="Paris",
                    cardinality="single", kind="state", as_of=None, replaces=None,
                    confidence=0.9, fact_seq=0,
                )
                before = self._checkpoint_snapshot(database, current_id)
                with self.assertRaises(IngestionRecoveryError) as caught:
                    if checkpoint == "stage":
                        ingestion_module._stage_extraction(
                            database, current_id, "[]", recorded_at=900, timeout=5
                        )
                    elif checkpoint == "apply":
                        ingestion_module._apply_batch_facts(
                            database, ((validated, (candidate,)),),
                            recorded_at=900, timeout=5,
                        )
                    elif checkpoint == "record":
                        ingestion_module._record_applied_fact(
                            database, current_id, 0, recorded_at=900, timeout=5
                        )
                    else:
                        ingestion_module._complete_job(
                            database, current_id, 1, replayed=False,
                            recorded_at=900, timeout=5,
                        )
                self.assertNotIsInstance(caught.exception, QuarantinedTurnError)
                self.assertNotIsInstance(caught.exception, TemporalWriteError)
                self.assertEqual(self._checkpoint_snapshot(database, current_id), before)

    def test_gate_provenance_failure_propagates_under_isolate(self) -> None:
        turn = self.make_turn(
            history="gate-provenance-history", session="gate-provenance-session"
        )
        error_type = getattr(gate_module, "InconsistentTurnProvenanceError", ValueError)
        with patch.object(
            ingestion_module,
            "evaluate_and_record_turn",
            side_effect=error_type("forced inconsistent persisted provenance"),
        ):
            with self.assertRaises(error_type) as caught:
                ingest_turns(
                    self.database, (turn,), evaluator=FakeEvaluator(),
                    extractor=FakeExtractor("[]"), quarantine_policy="isolate",
                )
        self.assertEqual(type(caught.exception).__name__, "InconsistentTurnProvenanceError")
        turn_id = f"{turn.history_id}:{turn.session_id}:{turn.turn_idx}"
        with connection(self.database) as conn:
            self.assertEqual(
                conn.execute(
                    "SELECT state FROM ingestion_jobs WHERE turn_id=?", (turn_id,)
                ).fetchone()[0],
                "pending_gate",
            )
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM gate_events WHERE turn_id=?", (turn_id,)
                ).fetchone()[0],
                0,
            )
            self.assertFalse(turn_is_eligible(conn, turn_id))

    def _state_from(self, database: str, turn_id: str) -> str:
        with connection(database) as conn:
            return str(conn.execute(
                "SELECT state FROM ingestion_jobs WHERE turn_id=?", (turn_id,)
            ).fetchone()[0])

    def _checkpoint_snapshot(self, database: str, turn_id: str) -> tuple[object, ...]:
        with connection(database) as conn:
            return (
                tuple(tuple(row) for row in conn.execute(
                    "SELECT * FROM ingestion_jobs WHERE turn_id=?", (turn_id,)
                ).fetchall()),
                tuple(tuple(row) for row in conn.execute(
                    "SELECT * FROM facts ORDER BY fact_id"
                ).fetchall()),
                tuple(tuple(row) for row in conn.execute(
                    "SELECT * FROM fact_observations ORDER BY observation_id"
                ).fetchall()),
                tuple(tuple(row) for row in conn.execute(
                    "SELECT * FROM embeddings ORDER BY ref_type, ref_id, model"
                ).fetchall()),
                tuple(tuple(row) for row in conn.execute(
                    "SELECT * FROM gate_events WHERE turn_id=? ORDER BY event_id", (turn_id,)
                ).fetchall()),
                conn.execute(
                    "SELECT base_trust FROM turns WHERE turn_id=?", (turn_id,)
                ).fetchone()[0],
            )


if __name__ == "__main__":
    unittest.main()
