"""Tests for resuming partial evals without losing provider-failure rows."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from engramOS.db import connection, initialize_database, transaction  # noqa: E402
from scripts.run_eval import _seed_resume_results  # noqa: E402


class EvaluationResumeTests(unittest.TestCase):
    def test_resume_copies_successes_and_preserves_superseded_failure(self) -> None:
        with TemporaryDirectory() as folder:
            database = str(Path(folder) / "resume.sqlite3")
            initialize_database(database)
            old_run = "old-partial"
            new_run = "continuation-300s"
            with transaction(database) as conn:
                for question_id, provider_failure, answer, cache_hash in (
                    ("done", False, "valid answer", "request-hash"),
                    ("timed_out", True, "I don't know.", None),
                ):
                    config = {
                        "provider_failure": provider_failure,
                        "provider_failure_types": ["reader_timeout"] if provider_failure else [],
                        "call_failures": ["reader_timeout"] if provider_failure else [],
                    }
                    conn.execute(
                        """INSERT INTO eval_runs (
                               run_id, system, question_id, question_type, split,
                               reader_model, judge_model, reader_prompt_hash,
                               judge_prompt_hash, run_config_json, answer, correct,
                               evidence_recall_at_5, prompt_tokens, out_tokens,
                               latency_ms, created_at
                           ) VALUES (?, 'engram', ?, 'authored', 'test', 'local',
                                     'rules', 'reader-hash', 'judge-hash', ?, ?, ?,
                                     1.0, 5, 4, 120, 1.0)""",
                        (old_run, question_id, json.dumps(config), answer,
                         0 if provider_failure else 1),
                    )
                    conn.execute(
                        """INSERT INTO eval_question_results (
                               run_id, system, question_id, axis, answer_text,
                               abstained, cited_ids_json, retrieved_ids_json,
                               expected_answer, expected_label, correct,
                               gate_outcome_json, prompt_tokens, completion_tokens,
                               latency_ms, cost_credits, cache_request_hash
                           ) VALUES (?, 'engram', ?, 'authored', ?, ?, '[]', '[]',
                                     'valid', 'answer_contains', ?, '{}', 5, 4, 120,
                                     0.0, ?)""",
                        (old_run, question_id, answer, int(provider_failure),
                         0 if provider_failure else 1, cache_hash),
                    )

            resumed = _seed_resume_results(
                database,
                source_run_id=old_run,
                target_run_id=new_run,
                previous_timeout_seconds=180.0,
                new_timeout_seconds=300.0,
            )

            self.assertEqual(resumed["reused_results"], 1)
            self.assertEqual(resumed["superseded_failures"], [{
                "system": "engram",
                "question_id": "timed_out",
                "provider_failure_types": ["reader_timeout"],
                "provider_timeout_seconds": 180.0,
            }])
            with connection(database) as conn:
                old_failure = conn.execute(
                    """SELECT q.answer_text, q.correct, e.run_config_json
                       FROM eval_question_results q JOIN eval_runs e
                         USING (run_id, system, question_id)
                       WHERE q.run_id=? AND q.question_id='timed_out'""",
                    (old_run,),
                ).fetchone()
                new_success = conn.execute(
                    """SELECT q.answer_text, q.correct, e.run_config_json
                       FROM eval_question_results q JOIN eval_runs e
                         USING (run_id, system, question_id)
                       WHERE q.run_id=? AND q.question_id='done'""",
                    (new_run,),
                ).fetchone()
                new_failure_count = conn.execute(
                    "SELECT COUNT(*) FROM eval_question_results WHERE run_id=? AND question_id='timed_out'",
                    (new_run,),
                ).fetchone()[0]
                old_count = conn.execute(
                    "SELECT COUNT(*) FROM eval_question_results WHERE run_id=?",
                    (old_run,),
                ).fetchone()[0]
            self.assertEqual(old_failure["answer_text"], "I don't know.")
            self.assertEqual(old_failure["correct"], 0)
            old_config = json.loads(old_failure["run_config_json"])
            self.assertTrue(old_config["superseded"])
            self.assertEqual(old_config["superseded_by_run_id"], new_run)
            self.assertEqual(old_config["superseded_by_timeout_seconds"], 300.0)
            self.assertEqual(new_success["answer_text"], "valid answer")
            self.assertEqual(new_success["correct"], 1)
            new_config = json.loads(new_success["run_config_json"])
            self.assertTrue(new_config["resumed_persisted_result"])
            self.assertEqual(new_config["continuation_from_run_id"], old_run)
            self.assertEqual(new_failure_count, 0)
            self.assertEqual(old_count, 2)


if __name__ == "__main__":
    unittest.main()
