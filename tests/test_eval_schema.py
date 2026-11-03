"""Additive evaluation-result table initialization coverage."""

from pathlib import Path
import sqlite3
import tempfile
import unittest

from engramOS.db import initialize_database


class EvalSchemaInitializationTests(unittest.TestCase):
    def test_eval_question_results_initializes_idempotently_on_fresh_database(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            database = str(Path(root) / "fresh.sqlite3")
            initialize_database(database)
            initialize_database(database)
            conn = sqlite3.connect(database)
            try:
                names = {
                    row[1] for row in conn.execute("PRAGMA table_info(eval_question_results)")
                }
            finally:
                conn.close()
            self.assertTrue(
                {
                    "run_id", "system", "question_id", "axis", "answer_text",
                    "abstained", "cited_ids_json", "retrieved_ids_json",
                    "expected_answer", "expected_label", "correct",
                    "gate_outcome_json", "prompt_tokens", "completion_tokens",
                    "latency_ms", "cost_credits", "cache_request_hash",
                }.issubset(names)
            )

    def test_initialization_adds_eval_question_results_to_preexisting_database(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            database = str(Path(root) / "preexisting.sqlite3")
            initialize_database(database)
            conn = sqlite3.connect(database)
            try:
                conn.execute("DROP TABLE eval_question_results")
            finally:
                conn.close()
            initialize_database(database)
            conn = sqlite3.connect(database)
            try:
                row = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    ("eval_question_results",),
                ).fetchone()
            finally:
                conn.close()
            self.assertEqual(row[0], "eval_question_results")


if __name__ == "__main__":
    unittest.main()
