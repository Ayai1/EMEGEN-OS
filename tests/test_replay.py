"""Replay CLI reads SQLite snapshots and never invokes a provider or network."""

from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from engramOS.db import connection, initialize_database  # noqa: E402
from scripts.replay import render_report  # noqa: E402


class ReplayTests(unittest.TestCase):
    def test_replay_reads_all_requested_evidence_without_provider_or_network(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            database = str(Path(root) / "replay.sqlite3")
            initialize_database(database)
            with connection(database) as conn:
                conn.execute(
                    """INSERT INTO turns (turn_id, history_id, session_id, turn_idx,
                       role, raw_content, source_type, event_time, recorded_at, hash)
                       VALUES ('turn-1', 'h', 's1', 0, 'user', 'My city is Pune.',
                               'user_turn', 100.0, 101.0, 'hash-1')"""
                )
                conn.execute(
                    """INSERT INTO gate_events (turn_id, signal_bridge, signal_policy,
                       signal_source, signals_fired, decision, created_at)
                       VALUES ('turn-1', 0.0, 0.0, 0.0, 0, 'store', 101.0)"""
                )
                conn.execute(
                    """INSERT INTO facts (fact_id, history_id, turn_id, key_type, entity,
                       attribute, value, value_norm, cardinality, kind, key_text,
                       valid_from, recorded_at, trust)
                       VALUES ('fact-1', 'h', 'turn-1', 'slot', 'user', 'city',
                               'Pune', 'pune', 'single', 'state', 'user city is Pune',
                               100.0, 101.0, 0.9)"""
                )
                conn.execute(
                    """INSERT INTO fact_observations (observation_id, history_id, turn_id,
                       fact_seq, entity, attribute, value, value_norm, cardinality, kind,
                       key_text, valid_at, effective_at, canonical_fact_id,
                       last_confirmed_at, last_confirmed_turn_id, recorded_at)
                       VALUES ('obs-1', 'h', 'turn-1', 0, 'user', 'city', 'Pune', 'pune',
                               'single', 'state', 'user city is Pune', 100.0, 100.0,
                               'fact-1', 100.0, 'turn-1', 101.0)"""
                )
                conn.execute(
                    """INSERT INTO eval_runs (run_id, system, question_id, question_type,
                       split, reader_model, judge_model, reader_prompt_hash,
                       judge_prompt_hash, run_config_json, answer, correct, created_at)
                       VALUES ('r1', 'engram', 'q1', 'present_state', 'test', 'fake-reader',
                               'rule-grader', 'reader-hash', 'judge-hash', '{}',
                               'Pune', 1, 200.0)"""
                )
                conn.execute(
                    """INSERT INTO eval_question_results (run_id, system, question_id,
                       axis, answer_text, abstained, cited_ids_json, retrieved_ids_json,
                       expected_answer, expected_label, correct, gate_outcome_json,
                       prompt_tokens, completion_tokens, latency_ms, cost_credits,
                       cache_request_hash)
                       VALUES ('r1', 'engram', 'q1', 'present_state', 'Pune', 0,
                               '["fact-1"]', '["turn-1"]', 'Pune', 'answer', 1,
                               '{"signals_fired":0}', 8, 2, 12, 0.0, 'request-hash')"""
                )
                conn.execute(
                    """INSERT INTO llm_cache (model, prompt_hash, temperature, task_tag,
                       response, in_tokens, out_tokens, cost_credits, latency_ms,
                       phase_tag, run_id, created_at)
                       VALUES ('fake-reader', 'request-hash', 0.0, 'reader', 'Pune',
                               8, 2, 0.0, 12, 'eval', 'r1', 200.0)"""
                )

            with patch.object(
                socket, "socket", side_effect=AssertionError("replay attempted network I/O")
            ), patch(
                "engramOS.llm_cache.cached_generate",
                side_effect=AssertionError("replay invoked a provider"),
            ):
                report = render_report(
                    database,
                    history_id="h",
                    entity="user",
                    attribute="city",
                    question_id="q1",
                    run_id="r1",
                )

            self.assertIn("INTERVAL TIMELINE", report)
            self.assertIn("fact-1 | Pune", report)
            self.assertIn("obs-1", report)
            self.assertIn("GATE AUDIT", report)
            self.assertIn("0.0 | 0.0 | 0.0 | 0 | store", report)
            self.assertIn("SIDE-BY-SIDE EVAL ANSWERS", report)
            self.assertIn("engram | Pune | 0 | 1", report)
            self.assertIn("LLM CACHE REQUESTS", report)
            self.assertIn("fake-reader | reader | eval | r1 | 8/2", report)


if __name__ == "__main__":
    unittest.main()
