"""SQLite integration tests for temporal facts and retrieval eligibility."""

from __future__ import annotations

from datetime import datetime, timezone
import itertools
import math
import sqlite3
import sys
from pathlib import Path
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engramOS.db import connection, initialize_database  # noqa: E402
from engramOS.eligibility import turn_eligibility_predicate  # noqa: E402
import engramOS.temporal as temporal_module  # noqa: E402
from engramOS.temporal import (  # noqa: E402
    FactInput,
    calculate_valid_from,
    get_current_facts,
    get_fact_history,
    get_facts_as_of,
    insert_fact,
    rebuild_slot_projection,
    TemporalBoundary,
)
from engramOS.gate import evaluate_and_record_turn  # noqa: E402
from tests.test_gate import FakeEvaluator, outcome  # noqa: E402
from tests.helpers import record_store_event  # noqa: E402


class TemporalMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = str(Path(self.temp_dir.name) / "temporal.sqlite3")
        initialize_database(self.database)
        self.serial = 0

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_fact(
        self,
        fact_id: str,
        when: float,
        value: str,
        *,
        history_id: str = "history-1",
        entity: str = "user",
        attribute: str = "city",
        fact_seq: int = 0,
        cardinality: str = "single",
        quarantined: bool = False,
        session_id: str | None = None,
        turn_idx: int = 0,
        recorded_at: float | None = None,
        raw_content: str | None = None,
        gated: bool = True,
    ) -> FactInput:
        self.serial += 1
        session_id = session_id or f"session-{fact_id}"
        turn_id = f"{history_id}:{session_id}:{turn_idx}"
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO turns (
                       turn_id, history_id, session_id, turn_idx, role, raw_content,
                       event_time, recorded_at, hash
                   ) VALUES (?, ?, ?, ?, 'user', ?, ?, ?, ?)""",
                (
                    turn_id,
                    history_id,
                    session_id,
                    turn_idx,
                    raw_content or f"I live in {value}",
                    when,
                    (recorded_at if recorded_at is not None else when + 1000),
                    f"hash-{fact_id}",
                ),
            )
            if gated:
                record_store_event(conn, turn_id, created_at=when + 1000)
        return FactInput(
            fact_id=fact_id,
            history_id=history_id,
            turn_id=turn_id,
            fact_seq=fact_seq,
            entity=entity,
            attribute=attribute,
            value=value,
            cardinality=cardinality,  # type: ignore[arg-type]
            valid_from=when,
            recorded_at=recorded_at if recorded_at is not None else when + 1000,
            quarantined=quarantined,
        )

    def write(self, fact: FactInput) -> str:
        return insert_fact(self.database, fact)

    def fact_rows(self, history_id: str = "history-1") -> list[dict[str, object]]:
        return get_fact_history(self.database, history_id, entity="user", attribute="city")

    def test_new_fact_insertion(self) -> None:
        fact = self.make_fact("a", 100.0, "Paris")
        self.assertEqual(self.write(fact), "a")
        current = get_current_facts(self.database, "history-1", entity="user", attribute="city")
        self.assertEqual([row["value"] for row in current], ["Paris"])
        with connection(self.database) as conn:
            stored = conn.execute("SELECT * FROM facts WHERE fact_id = 'a'").fetchone()
            observation_count = conn.execute("SELECT COUNT(*) FROM fact_observations").fetchone()[0]
        self.assertEqual(stored["valid_from"], 100.0)
        self.assertEqual(stored["last_confirmed_turn_id"], fact.turn_id)
        self.assertEqual(observation_count, 1)

    def test_fact_confirmation_preserves_latest_provenance(self) -> None:
        first = self.make_fact("a1", 100.0, "The Paris")
        second = self.make_fact("a2", 200.0, "paris")
        self.write(first)
        self.assertEqual(self.write(second), "a1")
        rows = self.fact_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value_norm"], "paris")
        self.assertEqual(rows[0]["confirmation_count"], 2)
        self.assertEqual(rows[0]["last_confirmed_turn_id"], second.turn_id)
        with connection(self.database) as conn:
            stored = conn.execute("SELECT * FROM facts WHERE fact_id = 'a1'").fetchone()
        self.assertEqual(stored["confirmation_count"], 2)

    def test_fact_supersession_closes_predecessor(self) -> None:
        self.write(self.make_fact("a", 100.0, "Paris"))
        self.write(self.make_fact("b", 200.0, "Rome"))
        rows = self.fact_rows()
        self.assertEqual([(r["value"], r["valid_from"], r["valid_until"]) for r in rows],
                         [("Paris", 100.0, 200.0), ("Rome", 200.0, None)])
        self.assertEqual(rows[0]["superseded_by"], "b")
        self.assertEqual([r["value"] for r in get_current_facts(self.database, "history-1")], ["Rome"])

    def test_explicit_timestamp_as_of_query(self) -> None:
        self.write(self.make_fact("a", 100.0, "Paris"))
        self.write(self.make_fact("b", 200.0, "Rome"))
        at_150 = get_facts_as_of(self.database, "history-1", 150.0)
        at_200 = get_facts_as_of(self.database, "history-1", 200.0)
        self.assertEqual([r["value"] for r in at_150], ["Paris"])
        self.assertEqual([r["value"] for r in at_200], ["Rome"])

    def test_current_view_returns_each_multi_observation_once_in_temporal_order(self) -> None:
        self.write(self.make_fact("multi-a", 100.0, "Reading", cardinality="multi"))
        self.write(self.make_fact("multi-b", 200.0, "Hiking", cardinality="multi"))

        rows = get_current_facts(self.database, "history-1", entity="user", attribute="city")

        self.assertEqual(
            [(row["fact_id"], row["value"]) for row in rows],
            [("multi-a", "Reading"), ("multi-b", "Hiking")],
        )
        self.assertEqual(len(rows), 2)

    def test_history_view_returns_each_multi_observation_once_in_temporal_order(self) -> None:
        self.write(self.make_fact("multi-a", 100.0, "Reading", cardinality="multi"))
        self.write(self.make_fact("multi-b", 200.0, "Hiking", cardinality="multi"))

        rows = get_fact_history(self.database, "history-1", entity="user", attribute="city")

        self.assertEqual(
            [(row["fact_id"], row["value"]) for row in rows],
            [("multi-a", "Reading"), ("multi-b", "Hiking")],
        )
        self.assertEqual(len(rows), 2)

    def test_as_of_before_first_multi_observation_returns_no_facts(self) -> None:
        self.write(self.make_fact("multi-a", 100.0, "Reading", cardinality="multi"))
        self.write(self.make_fact("multi-b", 200.0, "Hiking", cardinality="multi"))

        rows = get_facts_as_of(
            self.database, "history-1", 99.0, entity="user", attribute="city"
        )

        self.assertEqual(rows, [])
        self.assertEqual(len(rows), 0)

    def test_as_of_between_multi_observations_returns_only_the_first(self) -> None:
        self.write(self.make_fact("multi-a", 100.0, "Reading", cardinality="multi"))
        self.write(self.make_fact("multi-b", 200.0, "Hiking", cardinality="multi"))

        rows = get_facts_as_of(
            self.database, "history-1", 150.0, entity="user", attribute="city"
        )

        self.assertEqual(
            [(row["fact_id"], row["value"]) for row in rows],
            [("multi-a", "Reading")],
        )
        self.assertEqual(len(rows), 1)

    def test_as_of_after_multi_observations_returns_each_once_in_temporal_order(self) -> None:
        self.write(self.make_fact("multi-a", 100.0, "Reading", cardinality="multi"))
        self.write(self.make_fact("multi-b", 200.0, "Hiking", cardinality="multi"))

        rows = get_facts_as_of(
            self.database, "history-1", 250.0, entity="user", attribute="city"
        )

        self.assertEqual(
            [(row["fact_id"], row["value"]) for row in rows],
            [("multi-a", "Reading"), ("multi-b", "Hiking")],
        )
        self.assertEqual(len(rows), 2)

    def test_mixed_single_and_multi_slot_view_preserves_each_fact_once(self) -> None:
        self.write(self.make_fact("single-paris", 50.0, "Paris", attribute="city"))
        self.write(self.make_fact(
            "multi-reading", 100.0, "Reading", attribute="hobby", cardinality="multi"
        ))
        self.write(self.make_fact(
            "multi-hiking", 200.0, "Hiking", attribute="hobby", cardinality="multi"
        ))
        self.write(self.make_fact("single-rome", 250.0, "Rome", attribute="city"))

        current = get_current_facts(self.database, "history-1")
        history = get_fact_history(self.database, "history-1")
        at_150 = get_facts_as_of(self.database, "history-1", 150.0)

        self.assertEqual(
            [(row["fact_id"], row["value"]) for row in current],
            [("multi-reading", "Reading"), ("multi-hiking", "Hiking"), ("single-rome", "Rome")],
        )
        self.assertEqual(len(current), 3)
        self.assertEqual(
            [(row["fact_id"], row["value"]) for row in history],
            [
                ("single-paris", "Paris"),
                ("multi-reading", "Reading"),
                ("multi-hiking", "Hiking"),
                ("single-rome", "Rome"),
            ],
        )
        self.assertEqual(len(history), 4)
        self.assertEqual(
            [(row["fact_id"], row["value"]) for row in at_150],
            [("single-paris", "Paris"), ("multi-reading", "Reading")],
        )
        self.assertEqual(len(at_150), 2)

    def test_quarantined_multi_observations_are_excluded_and_retained(self) -> None:
        clean = self.make_fact(
            "hobby-clean", 100.0, "Reading", attribute="hobby", cardinality="multi"
        )
        flagged = self.make_fact(
            "hobby-flagged", 200.0, "Hiking", attribute="hobby", cardinality="multi",
            quarantined=True,
        )
        gate_event = self.make_fact(
            "hobby-gate-event", 300.0, "Cycling", attribute="hobby", cardinality="multi",
        )
        for fact in (clean, flagged, gate_event):
            self.write(fact)

        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO gate_events (
                       turn_id, signal_bridge, signal_policy, signal_source,
                       signals_fired, decision, created_at
                   ) VALUES (?, 1, 1, 0, 2, 'quarantine', 500)""",
                (gate_event.turn_id,),
            )

        views = (
            ("current", get_current_facts(self.database, "history-1", entity="user", attribute="hobby")),
            ("history", get_fact_history(self.database, "history-1", entity="user", attribute="hobby")),
            ("as_of", get_facts_as_of(
                self.database, "history-1", 400.0, entity="user", attribute="hobby"
            )),
        )
        for view_name, rows in views:
            with self.subTest(view=view_name):
                self.assertEqual(
                    [(row["fact_id"], row["value"]) for row in rows],
                    [("hobby-clean", "Reading")],
                )
                self.assertEqual(len(rows), 1)

        with connection(self.database) as conn:
            observation_rows = conn.execute(
                """SELECT observation_id, quarantined FROM fact_observations
                   WHERE history_id = 'history-1' AND entity = 'user' AND attribute = 'hobby'
                   ORDER BY observation_id"""
            ).fetchall()
            fact_rows = conn.execute(
                """SELECT fact_id, quarantined FROM facts
                   WHERE history_id = 'history-1' AND key_type = 'slot'
                     AND entity = 'user' AND attribute = 'hobby'
                   ORDER BY fact_id"""
            ).fetchall()
            gate_row = conn.execute(
                "SELECT decision FROM gate_events WHERE turn_id = ? "
                "AND decision = 'quarantine' ORDER BY event_id DESC LIMIT 1",
                (gate_event.turn_id,),
            ).fetchone()
        self.assertEqual(
            [(row["observation_id"], row["quarantined"]) for row in observation_rows],
            [("hobby-clean", 0), ("hobby-flagged", 1), ("hobby-gate-event", 0)],
        )
        self.assertEqual(len(observation_rows), 3)
        self.assertEqual(
            [(row["fact_id"], row["quarantined"]) for row in fact_rows],
            [("hobby-clean", 0), ("hobby-flagged", 1), ("hobby-gate-event", 0)],
        )
        self.assertEqual(len(fact_rows), 3)
        self.assertEqual(gate_row["decision"], "quarantine")

    def test_real_gate_quarantine_excludes_and_retains_multi_observation(self) -> None:
        clean = self.make_fact(
            "gate-hobby-clean", 100.0, "Reading", attribute="hobby", cardinality="multi"
        )
        quarantined = self.make_fact(
            "gate-hobby-quarantined", 200.0, "Hiking", attribute="hobby", cardinality="multi"
        )
        self.write(clean)
        self.write(quarantined)

        event = evaluate_and_record_turn(
            self.database,
            quarantined.turn_id,
            FakeEvaluator(outcome(True), outcome(True)),
            created_at=500.0,
        )
        self.assertEqual((event.decision, event.signals_fired), ("quarantine", 2))

        views = (
            ("current", get_current_facts(self.database, "history-1", entity="user", attribute="hobby")),
            ("history", get_fact_history(self.database, "history-1", entity="user", attribute="hobby")),
            ("as_of", get_facts_as_of(
                self.database, "history-1", 400.0, entity="user", attribute="hobby"
            )),
        )
        for view_name, rows in views:
            with self.subTest(view=view_name):
                self.assertEqual(
                    [(row["fact_id"], row["value"]) for row in rows],
                    [("gate-hobby-clean", "Reading")],
                )
                self.assertEqual(len(rows), 1)

        with connection(self.database) as conn:
            observation_rows = conn.execute(
                """SELECT observation_id, quarantined FROM fact_observations
                   WHERE history_id = 'history-1' AND entity = 'user' AND attribute = 'hobby'
                   ORDER BY observation_id"""
            ).fetchall()
            fact_rows = conn.execute(
                """SELECT fact_id, quarantined FROM facts
                   WHERE history_id = 'history-1' AND key_type = 'slot'
                     AND entity = 'user' AND attribute = 'hobby'
                   ORDER BY fact_id"""
            ).fetchall()
            turn = conn.execute(
                "SELECT base_trust FROM turns WHERE turn_id = ?",
                (quarantined.turn_id,),
            ).fetchone()
        self.assertEqual(
            [(row["observation_id"], row["quarantined"]) for row in observation_rows],
            [("gate-hobby-clean", 0), ("gate-hobby-quarantined", 1)],
        )
        self.assertEqual(len(observation_rows), 2)
        self.assertEqual(
            [(row["fact_id"], row["quarantined"]) for row in fact_rows],
            [("gate-hobby-clean", 0), ("gate-hobby-quarantined", 1)],
        )
        self.assertEqual(len(fact_rows), 2)
        self.assertEqual(turn["base_trust"], 0.0)

    def test_multi_as_of_uses_exact_inclusive_effective_at_boundaries(self) -> None:
        self.write(self.make_fact(
            "multi-a", 100.0, "Reading", attribute="hobby", cardinality="multi"
        ))
        self.write(self.make_fact(
            "multi-b", 200.0, "Hiking", attribute="hobby", cardinality="multi"
        ))

        boundaries = (
            (math.nextafter(100.0, 0.0), []),
            (100.0, [("multi-a", "Reading")]),
            (math.nextafter(200.0, 0.0), [("multi-a", "Reading")]),
            (200.0, [("multi-a", "Reading"), ("multi-b", "Hiking")]),
        )
        for timestamp, expected in boundaries:
            with self.subTest(as_of=timestamp):
                rows = get_facts_as_of(
                    self.database, "history-1", timestamp,
                    entity="user", attribute="hobby",
                )
                actual = [(row["fact_id"], row["value"]) for row in rows]
                self.assertEqual(actual, expected)
                self.assertEqual(len(rows), len(expected))

        with connection(self.database) as conn:
            effective_times = conn.execute(
                """SELECT observation_id, valid_at, effective_at FROM fact_observations
                   WHERE history_id = 'history-1' AND entity = 'user'
                     AND attribute = 'hobby' ORDER BY valid_at"""
            ).fetchall()
        self.assertEqual(
            [(row["observation_id"], row["valid_at"], row["effective_at"]) for row in effective_times],
            [("multi-a", 100.0, 100.0), ("multi-b", 200.0, 200.0)],
        )

    def test_exclusive_as_of_boundary_selects_left_limit_without_epsilon(self) -> None:
        self.write(self.make_fact("boundary-old", 100.0, "Paris"))
        self.write(self.make_fact("boundary-new", 200.0, "Rome"))

        checks = (
            (TemporalBoundary(math.nextafter(200.0, 0.0)), [("boundary-old", "Paris")]),
            (TemporalBoundary(200.0, inclusive=False), [("boundary-old", "Paris")]),
            (TemporalBoundary(200.0, inclusive=True), [("boundary-new", "Rome")]),
            (TemporalBoundary(math.nextafter(200.0, math.inf)), [("boundary-new", "Rome")]),
        )
        for boundary, expected in checks:
            with self.subTest(boundary=boundary):
                rows = get_facts_as_of(
                    self.database, "history-1", boundary,
                    entity="user", attribute="city",
                )
                self.assertEqual(
                    [(row["fact_id"], row["value"]) for row in rows], expected
                )

    def test_multi_tied_order_is_deterministic_across_insertion_orders(self) -> None:
        facts = (
            ("z-hobby", "hobby", "Reading"),
            ("a-interest", "interest", "Music"),
            ("m-hobby", "hobby", "Yoga"),
        )
        expected = [
            ("m-hobby", "Yoga"),
            ("z-hobby", "Reading"),
            ("a-interest", "Music"),
        ]

        for ordering_name, ordering in (
            ("forward", facts),
            ("reverse", tuple(reversed(facts))),
        ):
            with self.subTest(insertion_order=ordering_name):
                db_dir = Path(self.temp_dir.name) / f"ties-{ordering_name}"
                db_dir.mkdir()
                database = str(db_dir / "temporal.sqlite3")
                initialize_database(database)
                for fact_id, attribute, value in ordering:
                    turn_id = f"history-1:{fact_id}:0"
                    with connection(database) as conn:
                        conn.execute(
                            """INSERT INTO turns (
                                   turn_id, history_id, session_id, turn_idx, role, raw_content,
                                   event_time, recorded_at, hash
                               ) VALUES (?, 'history-1', ?, 0, 'user', ?, 100, 1100, ?)""",
                            (turn_id, fact_id, value, f"hash-{fact_id}"),
                        )
                        record_store_event(conn, turn_id, created_at=1100.0)
                    insert_fact(
                        database,
                        FactInput(
                            fact_id, "history-1", turn_id, 0, "user", attribute, value,
                            "multi", 100.0, 1100.0,
                        ),
                    )

                rows = get_fact_history(database, "history-1")
                self.assertEqual(
                    [(row["fact_id"], row["value"]) for row in rows], expected
                )
                self.assertEqual(len(rows), 3)
                self.assertEqual([row["effective_at"] for row in rows], [100.0] * 3)

    def test_date_only_as_of_means_start_of_day_utc(self) -> None:
        day_start = datetime(2024, 5, 10, tzinfo=timezone.utc).timestamp()
        self.write(self.make_fact("a", day_start - 3600, "Paris"))
        self.write(self.make_fact("b", day_start + 3600, "Rome"))
        date_only = get_facts_as_of(self.database, "history-1", "2024-05-10")
        explicit_later = get_facts_as_of(
            self.database, "history-1", "2024-05-10T02:00:00+00:00"
        )
        self.assertEqual([r["value"] for r in date_only], ["Paris"])
        self.assertEqual([r["value"] for r in explicit_later], ["Rome"])
        with self.assertRaisesRegex(ValueError, "timezone offset"):
            get_facts_as_of(self.database, "history-1", datetime(2024, 5, 10, 1, 0))

    def test_backdated_insertion_stitches_predecessor_and_successor(self) -> None:
        self.write(self.make_fact("a", 100.0, "Paris"))
        self.write(self.make_fact("c", 300.0, "Tokyo"))
        self.write(self.make_fact("b", 200.0, "Rome"))
        self.assertEqual(
            [(r["value"], r["valid_from"], r["valid_until"]) for r in self.fact_rows()],
            [("Paris", 100.0, 200.0), ("Rome", 200.0, 300.0), ("Tokyo", 300.0, None)],
        )

    def test_a_to_b_to_a_is_three_non_overlapping_intervals(self) -> None:
        for fact in (
            self.make_fact("a1", 100.0, "Paris"),
            self.make_fact("b", 200.0, "Rome"),
            self.make_fact("a2", 300.0, "Paris"),
        ):
            self.write(fact)
        rows = self.fact_rows()
        self.assertEqual([row["value_norm"] for row in rows], ["paris", "rome", "paris"])
        self.assert_interval_invariants(rows)

    def test_backdated_fact_matching_successor_continues_that_interval(self) -> None:
        for fact in (
            self.make_fact("a1", 100.0, "Paris"),
            self.make_fact("b", 200.0, "Rome"),
            self.make_fact("a3", 300.0, "Paris"),
        ):
            self.write(fact)
        backdated = self.make_fact("a25", 250.0, " paris ")
        self.assertEqual(self.write(backdated), "a25")
        rows = self.fact_rows()
        self.assertEqual([row["value_norm"] for row in rows], ["paris", "rome", "paris"])
        self.assertEqual(rows[-1]["valid_from"], 250.0)
        self.assertIsNone(rows[-1]["valid_until"])
        self.assertEqual(rows[-1]["confirmation_count"], 2)
        self.assertEqual(rows[-1]["last_confirmed_turn_id"], "history-1:session-a3:0")
        self.assert_interval_invariants(rows)

    def test_adjacent_equal_values_normalize_to_one_interval(self) -> None:
        self.write(self.make_fact("a1", 100.0, "three bedrooms"))
        self.write(self.make_fact("a2", 300.0, "3 bedrooms"))
        rows = self.fact_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value_norm"], "3 bedrooms")
        self.assertEqual(rows[0]["valid_from"], 100.0)
        self.assertIsNone(rows[0]["valid_until"])
        self.assertEqual(rows[0]["confirmation_count"], 2)
        with connection(self.database) as conn:
            facts = conn.execute("SELECT COUNT(*) FROM facts WHERE key_type='slot'").fetchone()[0]
        self.assertEqual(facts, 1)

    def test_same_day_event_order_and_timestamp_tie_break_are_deterministic(self) -> None:
        day_start = datetime(2024, 5, 10, tzinfo=timezone.utc).timestamp()
        # Insert opposite to the stable session-ID ordering.
        self.write(self.make_fact("b", day_start + 3600, "Rome", session_id="z-session"))
        self.write(self.make_fact("a", day_start + 3600, "Paris", session_id="a-session"))
        rows = self.fact_rows()
        self.assertEqual([r["value"] for r in rows], ["Paris", "Rome"])
        self.assertGreater(float(rows[1]["valid_from"]), float(rows[0]["valid_from"]))
        self.assertEqual([r["value"] for r in get_facts_as_of(
            self.database, "history-1", day_start + 3600
        )], ["Paris"])
        self.assertEqual([r["value"] for r in get_current_facts(self.database, "history-1")], ["Rome"])

    def test_timestamp_formula_matches_prompt_rules(self) -> None:
        day_start = datetime(2024, 1, 2, tzinfo=timezone.utc).timestamp()
        self.assertEqual(calculate_valid_from(1000.0, 4, 2), 1004.002)
        self.assertAlmostEqual(
            calculate_valid_from(1000.0, 4, 2, as_of="2024-01-02"),
            day_start + 0.004002,
            places=6,
        )

    def test_cross_history_isolation(self) -> None:
        self.write(self.make_fact("h1a", 100.0, "Paris", history_id="history-1"))
        self.write(self.make_fact("h1b", 200.0, "Rome", history_id="history-1"))
        self.write(self.make_fact("h2a", 150.0, "Tokyo", history_id="history-2"))
        self.assertEqual([r["value"] for r in get_current_facts(self.database, "history-1")], ["Rome"])
        self.assertEqual([r["value"] for r in get_current_facts(self.database, "history-2")], ["Tokyo"])
        self.assertEqual([r["value"] for r in get_fact_history(
            self.database, "history-2", entity="user", attribute="city"
        )], ["Tokyo"])

    def test_database_initialization_backfills_legacy_slot_facts(self) -> None:
        legacy = self.make_fact("legacy", 100.0, "Paris")
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO facts (
                       fact_id, history_id, turn_id, key_type, entity, attribute,
                       value, value_norm, cardinality, key_text, valid_from, last_confirmed_at,
                       last_confirmed_turn_id, confirmation_count, recorded_at
                   ) VALUES (?, ?, ?, 'slot', ' User ', ' City ', 'Paris', 'paris', 'single',
                             'user.city: Paris', 100, 110, ?, 4, 120)""",
                (legacy.fact_id, legacy.history_id, legacy.turn_id, legacy.turn_id),
            )
            conn.execute("DROP TABLE fact_observations")

        initialize_database(self.database)
        rows = get_fact_history(self.database, "history-1", entity="user", attribute="city")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value"], "Paris")
        self.assertEqual(rows[0]["confirmation_count"], 4)
        self.assertEqual(rows[0]["last_confirmed_at"], 110)
        with connection(self.database) as conn:
            observation = conn.execute(
                "SELECT entity, attribute, confirmation_count FROM fact_observations"
            ).fetchone()
        self.assertEqual(tuple(observation), ("user", "city", 4))

    def test_temporal_write_backfills_unnormalized_legacy_slot(self) -> None:
        legacy = self.make_fact("legacy", 100.0, "Paris", entity=" User ", attribute=" City ")
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO facts (
                       fact_id, history_id, turn_id, key_type, entity, attribute,
                       value, value_norm, cardinality, key_text, valid_from, recorded_at
                   ) VALUES (?, ?, ?, 'slot', ' User ', ' City ', 'Paris', 'paris',
                             'single', 'user.city: Paris', 100, 110)""",
                (legacy.fact_id, legacy.history_id, legacy.turn_id),
            )
        successor = self.make_fact("successor", 200.0, "Rome")
        self.assertEqual(self.write(successor), "successor")
        rows = self.fact_rows()
        self.assertEqual(
            [(row["value"], row["valid_from"], row["valid_until"]) for row in rows],
            [("Paris", 100.0, 200.0), ("Rome", 200.0, None)],
        )
        self.assertEqual(
            (rows[0]["entity"], rows[0]["attribute"]),
            ("user", "city"),
        )

    def test_quarantined_fact_is_excluded(self) -> None:
        fact = self.make_fact("poison", 100.0, "Atlantis", quarantined=True)
        self.write(fact)
        self.assertEqual(get_current_facts(self.database, "history-1"), [])
        self.assertEqual(get_fact_history(self.database, "history-1"), [])

    def test_quarantined_source_turn_excludes_fact_and_does_not_end_older_value(self) -> None:
        self.write(self.make_fact("a", 100.0, "Paris"))
        poisoned = self.make_fact("b", 200.0, "Atlantis")
        self.write(poisoned)
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO gate_events (
                       turn_id, signal_bridge, signal_policy, signal_source,
                       signals_fired, decision, created_at
                   ) VALUES (?, 1, 1, 0, 2, 'quarantine', 500)""",
                (poisoned.turn_id,),
            )
            # A later store event cannot erase an audit quarantine decision.
            conn.execute(
                """INSERT INTO gate_events (
                       turn_id, signals_fired, decision, created_at
                   ) VALUES (?, 0, 'store', 600)""",
                (poisoned.turn_id,),
            )
        current = get_current_facts(self.database, "history-1")
        self.assertEqual([row["value"] for row in current], ["Paris"])
        self.assertEqual([row["value"] for row in get_fact_history(self.database, "history-1")], ["Paris"])

    def test_shared_eligibility_filters_quarantined_raw_turn_fts_candidates(self) -> None:
        poisoned = self.make_fact(
            "poison", 100.0, "Atlantis", raw_content="poisoned private secret"
        )
        clean = self.make_fact(
            "clean", 200.0, "Paris", raw_content="poisoned public explanation"
        )
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO gate_events (
                       turn_id, signals_fired, decision, created_at
                   ) VALUES (?, 2, 'quarantine', 500)""",
                (poisoned.turn_id,),
            )
            eligible = conn.execute(
                """SELECT t.turn_id FROM turns_fts
                   JOIN turns AS t ON t.rowid = turns_fts.rowid
                   WHERE turns_fts MATCH ? AND """ + turn_eligibility_predicate("t"),
                ("poisoned",),
            ).fetchall()
        self.assertEqual([row["turn_id"] for row in eligible], [clean.turn_id])

    def test_transaction_rolls_back_mid_supersession_failure(self) -> None:
        self.write(self.make_fact("a", 100.0, "Paris"))
        with connection(self.database) as conn:
            conn.execute(
                """CREATE TRIGGER injected_interval_failure
                   BEFORE UPDATE OF valid_until ON facts
                   WHEN OLD.fact_id = 'a' AND NEW.valid_until IS NOT NULL
                   BEGIN SELECT RAISE(ABORT, 'injected interval failure'); END"""
            )
        candidate = self.make_fact("b", 200.0, "Rome")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "injected interval failure"):
            self.write(candidate)
        with connection(self.database) as conn:
            a = conn.execute("SELECT valid_until, superseded_by FROM facts WHERE fact_id='a'").fetchone()
            b_count = conn.execute("SELECT COUNT(*) FROM facts WHERE fact_id='b'").fetchone()[0]
            observation_count = conn.execute("SELECT COUNT(*) FROM fact_observations").fetchone()[0]
        self.assertIsNone(a["valid_until"])
        self.assertIsNone(a["superseded_by"])
        self.assertEqual(b_count, 0)
        self.assertEqual(observation_count, 1)

    def test_replaying_identical_events_is_idempotent(self) -> None:
        events = [
            self.make_fact("a1", 100.0, "Paris"),
            self.make_fact("a2", 150.0, "Paris"),
            self.make_fact("b", 200.0, "Rome"),
        ]
        for fact in events:
            self.write(fact)
        with connection(self.database) as conn:
            conn.execute(
                "INSERT INTO embeddings (ref_type, ref_id, model, vec) VALUES ('fact', 'a1', 'test', X'01')"
            )
        initial = [(r["fact_id"], r["value_norm"], r["valid_from"], r["valid_until"], r["confirmation_count"])
                   for r in self.fact_rows()]
        for fact in events:
            self.write(fact)
        replayed = [(r["fact_id"], r["value_norm"], r["valid_from"], r["valid_until"], r["confirmation_count"])
                    for r in self.fact_rows()]
        self.assertEqual(replayed, initial)
        with connection(self.database) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM fact_observations").fetchone()[0], 3)
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM embeddings WHERE ref_id='a1'").fetchone()[0], 1
            )

    def test_randomized_insertion_orders_rebuild_same_logical_intervals(self) -> None:
        events = [
            FactInput("a1", "history-1", "history-1:a-session:0", 0, "user", "city", "Paris", "single", 100, 1100),
            FactInput("b", "history-1", "history-1:b-session:0", 0, "user", "city", "Rome", "single", 200, 1200),
            FactInput("a3", "history-1", "history-1:c-session:0", 0, "user", "city", "Paris", "single", 300, 1300),
        ]
        expected = [("Paris", 100.0, 200.0), ("Rome", 200.0, 300.0), ("Paris", 300.0, None)]
        for order_index, ordering in enumerate(itertools.permutations(events)):
            case_dir = Path(self.temp_dir.name) / f"permutation-{order_index}"
            case_dir.mkdir()
            db_path = str(case_dir / "db.sqlite3")
            initialize_database(db_path)
            with connection(db_path) as conn:
                for event in events:
                    session_id = event.turn_id.split(":")[1]
                    conn.execute(
                        """INSERT INTO turns (
                               turn_id, history_id, session_id, turn_idx, role, raw_content,
                               event_time, recorded_at, hash
                           ) VALUES (?, ?, ?, 0, 'user', ?, ?, ?, ?)""",
                        (event.turn_id, event.history_id, session_id, event.value,
                         event.valid_from, event.recorded_at, f"hash-{event.fact_id}"),
                    )
                    record_store_event(conn, event.turn_id, created_at=event.recorded_at)
            for event in ordering:
                insert_fact(db_path, event)
            actual = [
                (row["value"], row["valid_from"], row["valid_until"])
                for row in get_fact_history(db_path, "history-1", entity="user", attribute="city")
            ]
            self.assertEqual(actual, expected)

    def assert_interval_invariants(self, rows: list[dict[str, object]]) -> None:
        for index, row in enumerate(rows):
            start = float(row["valid_from"])
            end = row["valid_until"]
            if end is not None:
                self.assertGreater(float(end), start, "interval has zero or negative duration")
            if index + 1 < len(rows):
                next_row = rows[index + 1]
                self.assertEqual(end, next_row["valid_from"], "single-slot intervals must meet at transitions")
                self.assertNotEqual(row["value_norm"], next_row["value_norm"], "adjacent equal values must be merged")
                self.assertLessEqual(start, float(end))
        with connection(self.database) as conn:
            stored_rows = conn.execute(
                """SELECT value_norm, valid_from, valid_until FROM facts
                   WHERE history_id='history-1' AND entity='user' AND attribute='city'
                     AND key_type='slot' AND cardinality='single' AND quarantined=0
                   ORDER BY valid_from"""
            ).fetchall()
        stored = [dict(row) for row in stored_rows]
        self.assertEqual(
            [(row["value_norm"], row["valid_from"], row["valid_until"]) for row in stored],
            [(row["value_norm"], row["valid_from"], row["valid_until"]) for row in rows],
        )


    def test_insert_fact_requires_gate_decision_and_preserves_quarantine_semantics(self) -> None:
        pending = self.make_fact("ungated-fact", 100.0, "Paris", gated=False)
        snapshot_sql = (
            "SELECT * FROM facts ORDER BY fact_id",
            "SELECT * FROM fact_observations ORDER BY observation_id",
            "SELECT * FROM embeddings ORDER BY ref_type, ref_id, model",
            "SELECT * FROM ingestion_jobs ORDER BY turn_id",
        )
        with connection(self.database) as conn:
            before = tuple(
                tuple(tuple(row) for row in conn.execute(sql).fetchall())
                for sql in snapshot_sql
            )
        expected_error = getattr(temporal_module, "UngatedSourceTurnError", ValueError)
        with self.assertRaises(expected_error) as caught:
            insert_fact(self.database, pending)
        self.assertEqual(type(caught.exception).__name__, "UngatedSourceTurnError")
        self.assertIn(pending.turn_id, str(caught.exception))
        with connection(self.database) as conn:
            after = tuple(
                tuple(tuple(row) for row in conn.execute(sql).fetchall())
                for sql in snapshot_sql
            )
        self.assertEqual(after, before)

        stored = self.make_fact("stored-fact", 200.0, "Rome")
        self.write(stored)
        with connection(self.database) as conn:
            self.assertEqual(
                conn.execute("SELECT quarantined FROM facts WHERE fact_id='stored-fact'").fetchone()[0],
                0,
            )

        quarantined = self.make_fact("quarantined-fact", 300.0, "Oslo", gated=False)
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO gate_events (turn_id, signal_bridge, signal_policy,
                   signal_source, signals_fired, decision, created_at)
                   VALUES (?, 1, 1, 0, 2, 'quarantine', 400)""",
                (quarantined.turn_id,),
            )
        self.write(quarantined)
        stored_then_quarantined = self.make_fact("store-then-quarantine", 500.0, "Lima")
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO gate_events (turn_id, signal_bridge, signal_policy,
                   signal_source, signals_fired, decision, created_at)
                   VALUES (?, 1, 1, 0, 2, 'quarantine', 600)""",
                (stored_then_quarantined.turn_id,),
            )
        self.write(stored_then_quarantined)
        with connection(self.database) as conn:
            for fact_id in ("quarantined-fact", "store-then-quarantine"):
                self.assertEqual(
                    conn.execute(
                        "SELECT quarantined FROM facts WHERE fact_id=?", (fact_id,)
                    ).fetchone()[0],
                    1,
                )
                self.assertEqual(
                    conn.execute(
                        "SELECT quarantined FROM fact_observations WHERE observation_id=?",
                        (fact_id,),
                    ).fetchone()[0],
                    1,
                )
        self.assertEqual(self.write(stored), "stored-fact")

    def test_reconciliation_does_not_poison_pending_observation(self) -> None:
        pending = self.make_fact(
            "pending-old-observation", 100.0, "Paris", gated=False
        )
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO facts (fact_id, history_id, turn_id, fact_seq, key_type,
                   entity, attribute, value, value_norm, cardinality, kind, key_text,
                   valid_from, last_confirmed_at, last_confirmed_turn_id,
                   confirmation_count, recorded_at, trust, quarantined)
                   VALUES (?, ?, ?, 0, 'slot', 'user', 'city', 'Paris', 'paris',
                   'single', 'state', 'user.city: Paris', 100, 110, ?, 1, 110,
                   0.7, 0)""",
                (pending.fact_id, pending.history_id, pending.turn_id, pending.turn_id),
            )
            conn.execute(
                """INSERT INTO fact_observations (observation_id, history_id, turn_id,
                   fact_seq, entity, attribute, value, value_norm, cardinality, kind,
                   key_text, valid_at, effective_at, canonical_fact_id,
                   last_confirmed_at, last_confirmed_turn_id, confirmation_count,
                   recorded_at, trust, quarantined)
                   VALUES (?, ?, ?, 0, 'user', 'city', 'Paris', 'paris', 'single',
                   'state', 'user.city: Paris', 100, 100, ?, 110, ?, 1, 110, 0.7, 0)""",
                (pending.fact_id, pending.history_id, pending.turn_id,
                 pending.fact_id, pending.turn_id),
            )
        clean = self.make_fact("newer-clean-observation", 200.0, "Rome")
        self.write(clean)
        with connection(self.database) as conn:
            old = conn.execute(
                "SELECT quarantined, quarantine_reason FROM facts WHERE fact_id=?",
                (pending.fact_id,),
            ).fetchone()
            self.assertEqual(tuple(old), (0, None))
        self.assertNotIn(
            pending.fact_id,
            {row["fact_id"] for row in get_fact_history(
                self.database, "history-1", entity="user", attribute="city"
            )},
        )

        with connection(self.database) as conn:
            record_store_event(conn, pending.turn_id, created_at=700.0)
            rebuild_slot_projection(conn, "history-1", "user", "city", 800.0)
        self.assertEqual(
            {row["fact_id"] for row in get_fact_history(
                self.database, "history-1", entity="user", attribute="city"
            )},
            {pending.fact_id, clean.fact_id},
        )


if __name__ == "__main__":
    unittest.main()
