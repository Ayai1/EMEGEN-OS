"""Decision and persistence integration tests for the provenance gate."""

from __future__ import annotations

from itertools import product
import sqlite3
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engramOS.db import connection, initialize_database  # noqa: E402
import engramOS.gate as gate_module  # noqa: E402
from engramOS.eligibility import (  # noqa: E402
    fact_eligibility_predicate,
    observation_eligibility_predicate,
    turn_eligibility_predicate,
    turn_is_eligible,
)
from engramOS.gate import (  # noqa: E402
    GateEvaluation,
    GateSignals,
    GateInputsChangedError,
    IncompleteSignalEvaluation,
    SignalResult,
    SignalStatus,
    decide_gate,
    evaluate_and_record_turn,
    evaluate_source_signal,
    evaluate_turn,
    _record_gate_decision,
)
from engramOS.temporal import (  # noqa: E402
    FactInput,
    get_current_facts,
    get_fact_history,
    insert_fact,
)
from tests.helpers import record_store_event  # noqa: E402


def outcome(fired: bool) -> SignalResult:
    return SignalResult(SignalStatus.FIRED if fired else SignalStatus.CLEAR)


def signals(bridge: bool, policy: bool, source: bool) -> GateSignals:
    return GateSignals(outcome(bridge), outcome(policy), outcome(source))


class FakeEvaluator:
    def __init__(self, bridge: object, policy: object) -> None:
        self.bridge = bridge
        self.policy = policy
        self.fingerprint = "gate-inputs-v1"
        self.calls: list[str] = []

    def fingerprint_inputs(self, conn: sqlite3.Connection, turn: sqlite3.Row) -> str:
        self.calls.append("fingerprint")
        return self.fingerprint

    def evaluate_bridge(self, conn: sqlite3.Connection, turn: sqlite3.Row) -> object:
        self.calls.append("bridge")
        return self.bridge

    def evaluate_policy(self, conn: sqlite3.Connection, turn: sqlite3.Row) -> object:
        self.calls.append("policy")
        return self.policy


class RaisingEvaluator:
    def fingerprint_inputs(self, conn: sqlite3.Connection, turn: sqlite3.Row) -> str:
        return "gate-inputs-v1"

    def evaluate_bridge(self, conn: sqlite3.Connection, turn: sqlite3.Row) -> SignalResult:
        raise RuntimeError("bridge service unavailable")

    def evaluate_policy(self, conn: sqlite3.Connection, turn: sqlite3.Row) -> SignalResult:
        return outcome(False)


class GateDecisionTests(unittest.TestCase):
    def test_all_eight_signal_combinations_follow_two_of_three_rule(self) -> None:
        for bridge, policy, source in product((False, True), repeat=3):
            with self.subTest(bridge=bridge, policy=policy, source=source):
                decision = decide_gate(signals(bridge, policy, source))
                fired = sum((bridge, policy, source))
                self.assertEqual(decision.signals_fired, fired)
                self.assertEqual(decision.decision, "quarantine" if fired >= 2 else "store")

    def test_repeated_decisions_are_deterministic(self) -> None:
        sample = signals(True, False, True)
        self.assertEqual(decide_gate(sample), decide_gate(sample))
        self.assertEqual(decide_gate(sample).decision, "quarantine")

    def test_missing_or_failed_signal_cannot_be_counted_or_decided(self) -> None:
        for unavailable in (
            SignalResult(SignalStatus.MISSING, "no policy evaluator result"),
            SignalResult(SignalStatus.FAILED, "policy evaluator failed"),
        ):
            with self.subTest(status=unavailable.status):
                with self.assertRaises(IncompleteSignalEvaluation):
                    decide_gate(GateSignals(outcome(True), unavailable, outcome(False)))

    def test_source_signal_uses_only_specified_provenance_types(self) -> None:
        for source_type in ("tool", "external_doc"):
            self.assertEqual(evaluate_source_signal(source_type).status, SignalStatus.FIRED)
        for source_type in ("user_turn", "assistant_turn", "system_policy", "external"):
            self.assertEqual(evaluate_source_signal(source_type).status, SignalStatus.CLEAR)


class GatePersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = str(Path(self.temp_dir.name) / "gate.sqlite3")
        initialize_database(self.database)
        self.serial = 0

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def add_turn(
        self,
        turn_id: str,
        *,
        history_id: str = "history-1",
        source_type: str = "user_turn",
        raw_content: str = "I live in Paris",
        event_time: float = 100.0,
    ) -> None:
        session_id = f"session-{turn_id}"
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO turns (
                       turn_id, history_id, session_id, turn_idx, role, raw_content,
                       source_type, event_time, recorded_at, hash
                   ) VALUES (?, ?, ?, 0, 'user', ?, ?, ?, ?, ?)""",
                (
                    turn_id,
                    history_id,
                    session_id,
                    raw_content,
                    source_type,
                    event_time,
                    event_time + 1000,
                    f"hash-{turn_id}",
                ),
            )

    def add_fact(
        self,
        fact_id: str,
        turn_id: str,
        value: str,
        *,
        history_id: str = "history-1",
        when: float = 100.0,
        attribute: str = "city",
    ) -> None:
        self.serial += 1
        insert_fact(
            self.database,
            FactInput(
                fact_id=fact_id,
                history_id=history_id,
                turn_id=turn_id,
                fact_seq=0,
                entity="user",
                attribute=attribute,
                value=value,
                cardinality="single",
                valid_from=when,
                recorded_at=when + 1000,
            ),
        )

    def evaluate(
        self,
        turn_id: str,
        *,
        bridge: SignalResult | None,
        policy: SignalResult | None,
    ) -> GateEvaluation:
        return evaluate_turn(self.database, turn_id, FakeEvaluator(bridge, policy))

    def record(
        self,
        turn_id: str,
        *,
        bridge: object,
        policy: object,
        created_at: float,
        evaluator: object | None = None,
    ):
        configured = evaluator or FakeEvaluator(bridge, policy)
        return evaluate_and_record_turn(
            self.database,
            turn_id,
            configured,
            created_at=created_at,
        )

    def _persistence_snapshot(self) -> dict[str, tuple[tuple[object, ...], ...]]:
        with connection(self.database) as conn:
            return {
                table: tuple(
                    tuple(row)
                    for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                )
                for table in (
                    "gate_events", "turns", "facts", "fact_observations", "embeddings"
                )
            }

    def test_canonical_api_evaluates_signals_and_persists_result(self) -> None:
        self.add_turn("h1:tool:0", source_type="tool")
        evaluator = FakeEvaluator(outcome(False), outcome(True))
        event = self.record(
            "h1:tool:0",
            bridge=outcome(False),
            policy=outcome(True),
            evaluator=evaluator,
            created_at=500.0,
        )
        self.assertEqual((event.decision, event.signals_fired), ("quarantine", 2))
        self.assertEqual(
            evaluator.calls,
            ["fingerprint", "bridge", "policy", "fingerprint", "fingerprint"],
        )
        with connection(self.database) as conn:
            row = conn.execute(
                "SELECT decision, signal_bridge, signal_policy, signal_source "
                "FROM gate_events WHERE turn_id='h1:tool:0'"
            ).fetchone()
        self.assertEqual(tuple(row), ("quarantine", 0.0, 1.0, 1.0))

    def test_public_api_rejects_caller_supplied_gate_evaluation(self) -> None:
        self.add_turn("h1:s:0")
        fabricated = GateEvaluation(
            "h1:s:0", "history-1", "user_turn", signals(True, True, False), "fake"
        )
        with self.assertRaisesRegex(TypeError, "turn_id must be a string"):
            evaluate_and_record_turn(
                self.database, fabricated, FakeEvaluator(outcome(False), outcome(False))
            )
        with connection(self.database) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM gate_events").fetchone()[0], 0)

    def test_missing_and_invalid_signal_results_cannot_create_store_event(self) -> None:
        self.add_turn("h1:s:0")
        for bridge in (None, object()):
            with self.subTest(bridge=bridge):
                with self.assertRaises(IncompleteSignalEvaluation):
                    self.record(
                        "h1:s:0", bridge=bridge, policy=outcome(False), created_at=500.0
                    )
        with connection(self.database) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM gate_events").fetchone()[0], 0)

    def test_gate_inputs_are_revalidated_inside_persistence_transaction(self) -> None:
        self.add_turn("h1:s:0")

        class ChangesBeforePersistence(FakeEvaluator):
            def fingerprint_inputs(
                self, conn: sqlite3.Connection, turn: sqlite3.Row
            ) -> str:
                self.calls.append("fingerprint")
                return (
                    "gate-inputs-v1"
                    if self.calls.count("fingerprint") <= 2
                    else "gate-inputs-v2"
                )

        evaluator = ChangesBeforePersistence(outcome(False), outcome(False))
        with self.assertRaises(GateInputsChangedError):
            self.record(
                "h1:s:0",
                bridge=outcome(False),
                policy=outcome(False),
                evaluator=evaluator,
                created_at=500.0,
            )
        with connection(self.database) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM gate_events").fetchone()[0], 0)

    def test_store_event_is_persisted_and_keeps_base_trust(self) -> None:
        self.add_turn("h1:s1:0")
        event = self.record(
            "h1:s1:0", bridge=outcome(False), policy=outcome(False), created_at=500.0
        )
        self.assertEqual((event.decision, event.signals_fired), ("store", 0))
        self.assertEqual((event.base_trust_before, event.base_trust_after), (0.6, 0.6))
        with connection(self.database) as conn:
            row = conn.execute(
                "SELECT * FROM gate_events WHERE event_id = ?", (event.event_id,)
            ).fetchone()
            turn = conn.execute(
                "SELECT base_trust FROM turns WHERE turn_id = 'h1:s1:0'"
            ).fetchone()
            with self.assertRaisesRegex(sqlite3.IntegrityError, "gate event history is immutable"):
                conn.execute(
                    "UPDATE gate_events SET decision='quarantine' WHERE event_id=?",
                    (event.event_id,),
                )
            with self.assertRaisesRegex(sqlite3.IntegrityError, "gate event history is immutable"):
                conn.execute("DELETE FROM gate_events WHERE event_id=?", (event.event_id,))
        self.assertEqual(row["decision"], "store")
        self.assertEqual((row["signal_bridge"], row["signal_policy"], row["signal_source"]), (0, 0, 0))
        self.assertEqual(row["created_at"], 500.0)
        self.assertEqual(turn["base_trust"], 0.6)

    def test_quarantine_marks_fact_and_observation_and_filters_turn_fts(self) -> None:
        self.add_turn(
            "h1:tool:0", source_type="tool", raw_content="poisoned project endpoint secret"
        )
        self.add_turn("h1:cue:0", raw_content="Please remember to check this later")
        with connection(self.database) as conn:
            record_store_event(conn, "h1:tool:0", created_at=1100.0)
            record_store_event(conn, "h1:cue:0", created_at=1200.0)
        self.add_fact("poison", "h1:tool:0", "malicious endpoint")
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO facts (
                       fact_id, history_id, turn_id, key_type, key_text, parent_fact_id,
                       valid_from, recorded_at
                   ) VALUES ('poison-trigger', 'history-1', 'h1:cue:0', 'trigger',
                             'later request cue', 'poison', 100, 1100)"""
            )
        before = get_current_facts(self.database, "history-1")
        self.assertEqual([row["value"] for row in before], ["malicious endpoint"])

        event = self.record(
            "h1:tool:0", bridge=outcome(True), policy=outcome(False), created_at=700.0
        )
        self.assertEqual((event.decision, event.signals_fired), ("quarantine", 2))
        self.assertEqual((event.base_trust_before, event.base_trust_after), (0.6, 0.0))
        self.assertEqual(get_current_facts(self.database, "history-1"), [])

        with connection(self.database) as conn:
            fact = conn.execute("SELECT quarantined FROM facts WHERE fact_id='poison'").fetchone()
            observation = conn.execute(
                "SELECT quarantined FROM fact_observations WHERE observation_id='poison'"
            ).fetchone()
            turn = conn.execute(
                "SELECT raw_content, event_time, base_trust FROM turns WHERE turn_id='h1:tool:0'"
            ).fetchone()
            fact_eligible = conn.execute(
                f"SELECT {fact_eligibility_predicate('f')} AS eligible "
                "FROM facts AS f WHERE f.fact_id='poison'"
            ).fetchone()["eligible"]
            observation_eligible = conn.execute(
                f"SELECT {observation_eligibility_predicate('o')} AS eligible "
                "FROM fact_observations AS o WHERE o.observation_id='poison'"
            ).fetchone()["eligible"]
            trigger_eligible = conn.execute(
                f"SELECT {fact_eligibility_predicate('f')} AS eligible "
                "FROM facts AS f WHERE f.fact_id='poison-trigger'"
            ).fetchone()["eligible"]
            fts_candidates = conn.execute(
                """SELECT t.turn_id FROM turns_fts
                   JOIN turns AS t ON t.rowid = turns_fts.rowid
                   WHERE turns_fts MATCH ? AND """
                + turn_eligibility_predicate("t"),
                ("endpoint",),
            ).fetchall()
        self.assertEqual(fact["quarantined"], 1)
        self.assertEqual(observation["quarantined"], 1)
        self.assertEqual(fact_eligible, 0)
        self.assertEqual(observation_eligible, 0)
        self.assertEqual(trigger_eligible, 0)
        self.assertEqual(turn["raw_content"], "poisoned project endpoint secret")
        self.assertEqual(turn["event_time"], 100.0)
        self.assertEqual(turn["base_trust"], 0.0)
        self.assertEqual(fts_candidates, [])
        with connection(self.database) as conn:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("UPDATE turns SET raw_content='changed' WHERE turn_id='h1:tool:0'")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("UPDATE turns SET event_time=101 WHERE turn_id='h1:tool:0'")

    def test_quarantine_rebuilds_confirmation_metadata_from_eligible_observations(self) -> None:
        self.add_turn("h1:a:0", raw_content="I live in Paris", event_time=100.0)
        self.add_turn("h1:b:0", raw_content="I still live in Paris", event_time=200.0)
        self.add_turn("h1:c:0", raw_content="I moved to Rome", event_time=300.0)
        with connection(self.database) as conn:
            for turn_id, timestamp in (("h1:a:0", 1100.0), ("h1:b:0", 1200.0), ("h1:c:0", 1300.0)):
                record_store_event(conn, turn_id, created_at=timestamp)
        self.add_fact("a", "h1:a:0", "Paris", when=100.0)
        self.add_fact("a-confirm", "h1:b:0", "Paris", when=200.0)
        self.add_fact("b", "h1:c:0", "Rome", when=300.0)
        before = get_fact_history(
            self.database, "history-1", entity="user", attribute="city"
        )
        paris_before = next(row for row in before if row["value"] == "Paris")
        self.assertEqual(paris_before["confirmation_count"], 2)
        self.assertEqual(paris_before["last_confirmed_turn_id"], "h1:b:0")

        event = self.record(
            "h1:b:0", bridge=outcome(True), policy=outcome(True), created_at=900.0
        )
        self.assertEqual(event.decision, "quarantine")
        rows = get_current_facts(self.database, "history-1", entity="user", attribute="city")
        self.assertEqual(rows[0]["value"], "Rome")
        history = get_fact_history(
            self.database, "history-1", entity="user", attribute="city"
        )
        paris = next(row for row in history if row["value"] == "Paris")
        self.assertEqual(paris["confirmation_count"], 1)
        self.assertEqual(paris["last_confirmed_turn_id"], "h1:a:0")
        with connection(self.database) as conn:
            canonical = conn.execute("SELECT * FROM facts WHERE fact_id='a'").fetchone()
        self.assertEqual(canonical["confirmation_count"], 1)
        self.assertEqual(canonical["last_confirmed_turn_id"], "h1:a:0")

    def test_successful_quarantine_updates_projection_and_related_rows(self) -> None:
        self.add_turn("h1:tool:0", source_type="tool", event_time=100.0)
        self.add_turn("h1:confirm:0", event_time=200.0)
        self.add_turn("h1:next:0", event_time=300.0)
        with connection(self.database) as conn:
            record_store_event(conn, "h1:tool:0", created_at=1100.0)
            record_store_event(conn, "h1:confirm:0", created_at=1200.0)
            record_store_event(conn, "h1:next:0", created_at=1300.0)
        self.add_fact("first", "h1:tool:0", "Paris", when=100.0)
        self.add_fact("confirm", "h1:confirm:0", "Paris", when=200.0)
        self.add_fact("next", "h1:next:0", "Rome", when=300.0)
        with connection(self.database) as conn:
            conn.execute(
                "INSERT INTO embeddings(ref_type, ref_id, model, vec) VALUES ('fact', 'first', 'm', X'01')"
            )
            conn.execute(
                """INSERT INTO facts (
                       fact_id, history_id, turn_id, key_type, key_text, parent_fact_id,
                       valid_from, recorded_at
                   ) VALUES ('trigger-child', 'history-1', 'h1:confirm:0', 'trigger',
                             'follow up', 'first', 150, 1150)"""
            )

        with connection(self.database) as conn:
            baseline_event_count = conn.execute("SELECT COUNT(*) FROM gate_events").fetchone()[0]
        event = self.record(
            "h1:tool:0", bridge=outcome(True), policy=outcome(False), created_at=800.0
        )
        history = get_fact_history(
            self.database, "history-1", entity="user", attribute="city"
        )
        self.assertEqual(
            [
                (
                    row["fact_id"],
                    row["value"],
                    row["valid_from"],
                    row["valid_until"],
                    row["confirmation_count"],
                    row["last_confirmed_turn_id"],
                )
                for row in history
            ],
            [
                ("confirm", "Paris", 200.0, 300.0, 1, "h1:confirm:0"),
                ("next", "Rome", 300.0, None, 1, "h1:next:0"),
            ],
        )
        self.assertEqual((event.decision, event.signals_fired, event.created_at), ("quarantine", 2, 800.0))
        with connection(self.database) as conn:
            event_count = conn.execute("SELECT COUNT(*) FROM gate_events").fetchone()[0]
            trust = conn.execute(
                "SELECT base_trust FROM turns WHERE turn_id='h1:tool:0'"
            ).fetchone()[0]
            source_fact = conn.execute(
                "SELECT quarantined, quarantine_reason FROM facts WHERE fact_id='first'"
            ).fetchone()
            source_observation = conn.execute(
                "SELECT quarantined, quarantine_reason FROM fact_observations WHERE observation_id='first'"
            ).fetchone()
            vector_count = conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE ref_type='fact' AND ref_id='first'"
            ).fetchone()[0]
            child = conn.execute(
                "SELECT parent_fact_id FROM facts WHERE fact_id='trigger-child'"
            ).fetchone()
        self.assertEqual(event_count, baseline_event_count + 1)
        self.assertEqual(trust, 0.0)
        self.assertEqual(source_fact["quarantined"], 1)
        self.assertIn("gate event", source_fact["quarantine_reason"])
        self.assertEqual(source_observation["quarantined"], 1)
        self.assertIn("gate event", source_observation["quarantine_reason"])
        self.assertEqual(vector_count, 0)
        self.assertEqual(child["parent_fact_id"], "first")

    def test_gate_history_is_append_only_and_quarantine_is_sticky(self) -> None:
        self.add_turn("h1:s:0")
        with connection(self.database) as conn:
            seed_event_id = record_store_event(conn, "h1:s:0", created_at=50.0)
            baseline_event_count = conn.execute("SELECT COUNT(*) FROM gate_events").fetchone()[0]
        self.add_fact("f", "h1:s:0", "Paris")
        store = self.record(
            "h1:s:0", bridge=outcome(False), policy=outcome(False), created_at=100.0
        )
        quarantine = self.record(
            "h1:s:0", bridge=outcome(True), policy=outcome(True), created_at=200.0
        )
        later_store = self.record(
            "h1:s:0", bridge=outcome(False), policy=outcome(False), created_at=300.0
        )
        self.assertLess(store.event_id, quarantine.event_id)
        self.assertLess(quarantine.event_id, later_store.event_id)
        with connection(self.database) as conn:
            self.assertFalse(turn_is_eligible(conn, "h1:s:0"))
        self.assertEqual(get_current_facts(self.database, "history-1"), [])
        with connection(self.database) as conn:
            events = conn.execute(
                "SELECT decision FROM gate_events WHERE turn_id='h1:s:0' AND event_id>? ORDER BY event_id",
                (seed_event_id,),
            ).fetchall()
            event_count = conn.execute("SELECT COUNT(*) FROM gate_events").fetchone()[0]
            turn = conn.execute(
                "SELECT base_trust FROM turns WHERE turn_id='h1:s:0'"
            ).fetchone()
        self.assertEqual([row["decision"] for row in events], ["store", "quarantine", "store"])
        self.assertEqual(event_count, baseline_event_count + 3)
        self.assertEqual(turn["base_trust"], 0.0)

    def test_quarantine_is_history_scoped(self) -> None:
        self.add_turn("h1:s:0", history_id="history-1")
        self.add_turn("h2:s:0", history_id="history-2")
        with connection(self.database) as conn:
            record_store_event(conn, "h1:s:0", created_at=100.0)
            record_store_event(conn, "h2:s:0", created_at=200.0)
        self.add_fact("h1-f", "h1:s:0", "Paris", history_id="history-1")
        self.add_fact("h2-f", "h2:s:0", "Paris", history_id="history-2")
        self.record(
            "h1:s:0", bridge=outcome(True), policy=outcome(True), created_at=500.0
        )
        self.assertEqual(get_current_facts(self.database, "history-1"), [])
        self.assertEqual(
            [row["value"] for row in get_current_facts(self.database, "history-2")],
            ["Paris"],
        )

    def test_evaluator_failure_creates_no_decision_or_trust_update(self) -> None:
        self.add_turn("h1:s:0", source_type="tool")
        with self.assertRaises(IncompleteSignalEvaluation):
            evaluate_and_record_turn(self.database, "h1:s:0", RaisingEvaluator())
        with connection(self.database) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM gate_events").fetchone()[0], 0)
            trust = conn.execute(
                "SELECT base_trust FROM turns WHERE turn_id='h1:s:0'"
            ).fetchone()[0]
        self.assertEqual(trust, 0.6)

    def test_source_result_mismatch_is_rejected(self) -> None:
        self.add_turn("h1:s:0", source_type="tool")
        forged = GateEvaluation(
            "h1:s:0", "history-1", "user_turn", signals(False, False, False), "gate-inputs-v1"
        )
        with self.assertRaisesRegex(ValueError, "source type"):
            _record_gate_decision(
                self.database,
                forged,
                FakeEvaluator(outcome(False), outcome(False)),
                created_at=100.0,
            )
        with connection(self.database) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM gate_events").fetchone()[0], 0)

    def test_source_signal_mismatch_is_rejected(self) -> None:
        self.add_turn("h1:s:0", source_type="tool")
        forged = GateEvaluation(
            "h1:s:0", "history-1", "tool", signals(False, False, False), "gate-inputs-v1"
        )
        with self.assertRaisesRegex(ValueError, "source signal"):
            _record_gate_decision(
                self.database,
                forged,
                FakeEvaluator(outcome(False), outcome(False)),
                created_at=100.0,
            )
        with connection(self.database) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM gate_events").fetchone()[0], 0)

    def test_gate_event_trust_and_quarantine_changes_roll_back_together(self) -> None:
        self.add_turn("h1:s:0", source_type="tool")
        with connection(self.database) as conn:
            record_store_event(conn, "h1:s:0", created_at=100.0)
            baseline_event_count = conn.execute("SELECT COUNT(*) FROM gate_events").fetchone()[0]
        self.add_fact("f", "h1:s:0", "Paris")
        with connection(self.database) as conn:
            conn.execute(
                """CREATE TRIGGER fail_gate_trust_update
                   BEFORE UPDATE OF base_trust ON turns
                   WHEN NEW.turn_id = 'h1:s:0'
                   BEGIN SELECT RAISE(ABORT, 'injected trust update failure'); END"""
            )
        with self.assertRaisesRegex(sqlite3.IntegrityError, "injected trust update failure"):
            self.record(
                "h1:s:0", bridge=outcome(True), policy=outcome(False), created_at=500.0
            )
        with connection(self.database) as conn:
            event_count = conn.execute("SELECT COUNT(*) FROM gate_events").fetchone()[0]
            fact = conn.execute("SELECT quarantined FROM facts WHERE fact_id='f'").fetchone()
            observation = conn.execute(
                "SELECT quarantined FROM fact_observations WHERE observation_id='f'"
            ).fetchone()
            turn = conn.execute(
                "SELECT base_trust FROM turns WHERE turn_id='h1:s:0'"
            ).fetchone()
        self.assertEqual(event_count, baseline_event_count)
        self.assertEqual(fact["quarantined"], 0)
        self.assertEqual(observation["quarantined"], 0)
        self.assertEqual(turn["base_trust"], 0.6)

    def test_quarantine_propagation_failures_roll_back_the_entire_gate_decision(self) -> None:
        self.add_turn("h1:tool:0", source_type="tool")
        self.add_turn("h1:confirm:0", event_time=200.0)
        self.add_turn("h1:next:0", event_time=300.0)
        with connection(self.database) as conn:
            record_store_event(conn, "h1:tool:0", created_at=1100.0)
            record_store_event(conn, "h1:confirm:0", created_at=1200.0)
            record_store_event(conn, "h1:next:0", created_at=1300.0)
        self.add_fact("f", "h1:tool:0", "Paris", when=100.0)
        self.add_fact("f-confirm", "h1:confirm:0", "Paris", when=200.0)
        self.add_fact("g", "h1:next:0", "Rome", when=300.0)
        with connection(self.database) as conn:
            conn.execute(
                "INSERT INTO embeddings(ref_type, ref_id, model, vec) VALUES ('fact', 'f', 'm', X'01')"
            )
            conn.execute(
                """INSERT INTO facts (
                       fact_id, history_id, turn_id, key_type, key_text, parent_fact_id,
                       valid_from, recorded_at
                   ) VALUES ('child-trigger', 'history-1', 'h1:confirm:0', 'trigger',
                             'remind me', 'f', 150, 1150)"""
            )
        self.record(
            "h1:tool:0", bridge=outcome(False), policy=outcome(False), created_at=400.0
        )
        before = self._persistence_snapshot()

        trigger_failures = (
            ("facts", "BEFORE UPDATE OF quarantined ON facts", "NEW.quarantined = 1"),
            (
                "observations",
                "BEFORE UPDATE OF quarantined ON fact_observations",
                "NEW.quarantined = 1",
            ),
            (
                "trust",
                "BEFORE UPDATE OF base_trust ON turns",
                "NEW.turn_id = 'h1:tool:0'",
            ),
        )
        for name, when_sql, condition in trigger_failures:
            with self.subTest(stage=name):
                with connection(self.database) as conn:
                    conn.execute(
                        f"""CREATE TRIGGER injected_{name}
                            {when_sql}
                            WHEN {condition}
                            BEGIN SELECT RAISE(ABORT, 'injected {name} failure'); END"""
                    )
                with self.assertRaisesRegex(sqlite3.IntegrityError, f"injected {name} failure"):
                    self.record(
                        "h1:tool:0",
                        bridge=outcome(True),
                        policy=outcome(False),
                        created_at=500.0,
                    )
                self.assertEqual(self._persistence_snapshot(), before)
                with connection(self.database) as conn:
                    conn.execute(f"DROP TRIGGER injected_{name}")

        from engramOS.temporal import rebuild_slot_projection as real_rebuild

        def rebuild_then_fail(*args: object, **kwargs: object) -> None:
            real_rebuild(*args, **kwargs)  # type: ignore[arg-type]
            raise RuntimeError("injected slot rebuild failure")

        with patch("engramOS.gate.rebuild_slot_projection", side_effect=rebuild_then_fail):
            with self.assertRaisesRegex(RuntimeError, "injected slot rebuild failure"):
                self.record(
                    "h1:tool:0",
                    bridge=outcome(True),
                    policy=outcome(False),
                    created_at=600.0,
                )
        self.assertEqual(self._persistence_snapshot(), before)


    def test_gate_enforces_hard_coded_role_source_matrix_before_evaluator(self) -> None:
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
        expected_error = getattr(
            gate_module, "InconsistentTurnProvenanceError", ValueError
        )
        for idx, ((role, source_type), expected) in enumerate(pairs.items()):
            turn_id = f"matrix:{idx}"
            with connection(self.database) as conn:
                conn.execute(
                    """INSERT INTO turns (turn_id, history_id, session_id, turn_idx,
                       role, raw_content, source_type, event_time, recorded_at,
                       base_trust, hash)
                       VALUES (?, 'matrix-history', ?, 0, ?, 'matrix content', ?,
                       ?, ?, 0.6, ?)""",
                    (turn_id, f"session-{idx}", role, source_type,
                     1000 + idx, 2000 + idx, f"hash-{idx}"),
                )
            with self.subTest(role=role, source_type=source_type, accepted=expected):
                if expected:
                    evaluator = FakeEvaluator(outcome(False), outcome(False))
                    event = evaluate_and_record_turn(
                        self.database, turn_id, evaluator, created_at=3000 + idx
                    )
                    self.assertEqual(event.decision, "store")
                    with connection(self.database) as conn:
                        self.assertTrue(turn_is_eligible(conn, turn_id))
                else:
                    evaluator = FakeEvaluator(outcome(False), outcome(False))
                    with self.assertRaises(expected_error) as caught:
                        evaluate_turn(self.database, turn_id, evaluator)
                    self.assertEqual(
                        type(caught.exception).__name__,
                        "InconsistentTurnProvenanceError",
                    )
                    self.assertEqual(evaluator.calls, [])
                    with self.assertRaises(expected_error):
                        evaluate_and_record_turn(self.database, turn_id, evaluator)
                    self.assertEqual(evaluator.calls, [])
                    with connection(self.database) as conn:
                        self.assertEqual(
                            conn.execute(
                                "SELECT COUNT(*) FROM gate_events WHERE turn_id=?",
                                (turn_id,),
                            ).fetchone()[0],
                            0,
                        )
                        self.assertEqual(
                            conn.execute(
                                "SELECT base_trust FROM turns WHERE turn_id=?",
                                (turn_id,),
                            ).fetchone()[0],
                            0.6,
                        )
                        self.assertFalse(turn_is_eligible(conn, turn_id))

        invalid_rows = (
            ("unknown-role", "observer", "user_turn"),
            ("unknown-source", "user", "unverified_source"),
        )
        for idx, (turn_id, role, source_type) in enumerate(invalid_rows):
            with connection(self.database) as conn:
                conn.execute(
                    """INSERT INTO turns (turn_id, history_id, session_id, turn_idx,
                       role, raw_content, source_type, event_time, recorded_at, hash)
                       VALUES (?, 'invalid-history', ?, 0, ?, 'invalid metadata', ?, 1, 1, ?)""",
                    (turn_id, turn_id, role, source_type, f"invalid-hash-{idx}"),
                )
            evaluator = FakeEvaluator(outcome(False), outcome(False))
            with self.subTest(role=role, source_type=source_type):
                with self.assertRaises(expected_error):
                    evaluate_turn(self.database, turn_id, evaluator)
                self.assertEqual(evaluator.calls, [])
                with connection(self.database) as conn:
                    self.assertEqual(
                        conn.execute(
                            "SELECT COUNT(*) FROM gate_events WHERE turn_id=?", (turn_id,)
                        ).fetchone()[0],
                        0,
                    )

    def test_gate_persistence_rechecks_provenance_before_writing(self) -> None:
        self.add_turn("persistence-recheck")
        error_type = getattr(gate_module, "ProvenanceError", ValueError)
        inconsistent_type = getattr(
            gate_module, "InconsistentTurnProvenanceError", ValueError
        )
        evaluator = FakeEvaluator(outcome(False), outcome(False))
        with patch.object(
            gate_module,
            "validate_role_source_type",
            create=True,
            side_effect=[None, error_type("forced second-check failure")],
        ) as validate:
            with self.assertRaises(inconsistent_type) as caught:
                evaluate_and_record_turn(
                    self.database, "persistence-recheck", evaluator, created_at=500
                )
        self.assertEqual(type(caught.exception).__name__, "InconsistentTurnProvenanceError")
        self.assertEqual(validate.call_count, 2)
        with connection(self.database) as conn:
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM gate_events WHERE turn_id='persistence-recheck'"
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                conn.execute(
                    "SELECT base_trust FROM turns WHERE turn_id='persistence-recheck'"
                ).fetchone()[0],
                0.6,
            )


if __name__ == "__main__":
    unittest.main()
