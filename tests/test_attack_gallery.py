"""Attack gallery builder uses only saved suite, results, and SQLite data."""

import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from engramOS.db import connection, initialize_database  # noqa: E402
from scripts.build_attack_gallery import build_gallery  # noqa: E402


class AttackGalleryBuilderTests(unittest.TestCase):
    def test_builds_attack_and_control_outcomes_from_saved_data_only(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            database = directory / "gallery.sqlite3"
            suite_path = directory / "suite.json"
            results_path = directory / "results.json"
            initialize_database(str(database))
            with connection(str(database)) as conn:
                conn.execute(
                    """INSERT INTO turns (turn_id, history_id, session_id, turn_idx,
                       role, raw_content, source_type, event_time, recorded_at, hash)
                       VALUES ('h:a01:0', 'h', 'a01', 0, 'user', 'Injected value',
                               'external_doc', 100.0, 101.0, 'attack-hash')"""
                )
                conn.execute(
                    """INSERT INTO turns (turn_id, history_id, session_id, turn_idx,
                       role, raw_content, source_type, event_time, recorded_at, hash)
                       VALUES ('h:b01:0', 'h', 'b01', 1, 'user', 'Benign value',
                               'external_doc', 102.0, 103.0, 'control-hash')"""
                )
                conn.execute(
                    """INSERT INTO gate_events (turn_id, signal_bridge, signal_policy,
                       signal_source, signals_fired, decision, created_at)
                       VALUES ('h:a01:0', 1.0, 0.0, 1.0, 2, 'quarantine', 101.0)"""
                )
                conn.execute(
                    """INSERT INTO gate_events (turn_id, signal_bridge, signal_policy,
                       signal_source, signals_fired, decision, created_at)
                       VALUES ('h:b01:0', 0.0, 0.0, 1.0, 1, 'store', 103.0)"""
                )

            suite = {
                "histories": [{
                    "history_id": "h",
                    "turns": [
                        {"session": "a01", "date": "2026-01-01", "text": "Injected value",
                         "source_type": "external_doc", "facts": []},
                        {"session": "b01", "date": "2026-01-02", "text": "Benign value",
                         "source_type": "external_doc", "facts": []},
                    ],
                }],
                "questions": [{
                    "question_id": "q1", "axis": "attack_adaptive", "query": "What is it?",
                    "expected_answer": "safe", "attack_turn_id": "h:a01:0",
                    "benign_turn_id": "h:b01:0",
                }],
                "attack_claims": {"q1": "unsafe"},
            }
            systems = ("recency", "hybrid_rag", "hybrid_rag_date_filter", "engram")
            results = {
                "run_id": "gallery-test",
                "suite": {"questions": 1},
                "question_results": [
                    {
                        "question_id": "q1", "system": system, "answer": system,
                        "abstained": False, "correct": system == "engram",
                        "provider_failure": False, "provider_failure_types": [],
                        "retrieved_ids": ["turn:h:a01:0"] if system == "recency" else [],
                        "attack_claim_adopted": system == "recency",
                        "quarantine_leak": system == "recency",
                    }
                    for system in systems
                ],
            }
            suite_path.write_text(json.dumps(suite), encoding="utf-8")
            results_path.write_text(json.dumps(results), encoding="utf-8")

            with patch.object(
                socket, "socket", side_effect=AssertionError("gallery attempted network I/O")
            ), patch(
                "engramOS.llm_cache.cached_generate",
                side_effect=AssertionError("gallery invoked a provider"),
            ):
                gallery = build_gallery(
                    suite_path=suite_path,
                    results_path=results_path,
                    database_path=database,
                    run_id="gallery-test",
                )

            self.assertEqual(gallery["attack_case_count"], 1)
            self.assertEqual(gallery["control_case_count"], 1)
            attack = gallery["attacks"][0]
            self.assertEqual(attack["gate"]["signals"], {
                "signal_source": True,
                "signal_policy": False,
                "signal_bridge": True,
            })
            self.assertEqual(attack["gate"]["decision"], "quarantine")
            self.assertTrue(attack["outcomes"]["recency"]["target_turn_reached_reader"])
            self.assertTrue(attack["outcomes"]["recency"]["claim_adopted"])
            self.assertFalse(attack["outcomes"]["engram"]["target_turn_reached_reader"])
            control = gallery["controls"][0]
            self.assertEqual(control["gate"]["decision"], "store")
            self.assertIsNone(control["outcomes"]["engram"]["claim_adopted"])


if __name__ == "__main__":
    unittest.main()
