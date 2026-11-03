"""Locked suite shape, local fixture extraction, and date-filter baseline tests."""

from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from engramOS.db import connection, initialize_database  # noqa: E402
from engramOS.embeddings import BGE_M3_DIMENSION, index_history  # noqa: E402
from engramOS.extraction import ExtractionPrompt, ExtractionTurn  # noqa: E402
from engramOS.query_classifier import TemporalRange  # noqa: E402
from engramOS.retrieval import hybrid_rag  # noqa: E402
from engramOS.temporal import TemporalBoundary  # noqa: E402
from eval.loader import (  # noqa: E402
    DEFAULT_SUITE_PATH,
    FixtureFactExtractor,
    iter_history_turns,
    load_suite,
    suite_stats,
)
from scripts.run_eval import run_evaluation  # noqa: E402
from tests.helpers import record_store_event  # noqa: E402


class FakeEmbedding:
    model_id = "fake-date-filter-1024"
    dimension = BGE_M3_DIMENSION
    device = "cpu"

    def vector(self, text: str) -> np.ndarray:
        out = np.zeros(self.dimension, dtype=np.float32)
        index = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little") % self.dimension
        out[index] = 1.0
        return out

    def encode_query(self, text: str) -> np.ndarray:
        return self.vector(text)

    def encode_documents(self, texts: list[str] | tuple[str, ...]) -> np.ndarray:
        return np.vstack([self.vector(text) for text in texts]) if texts else np.empty((0, self.dimension))


class AuthoredSuiteTests(unittest.TestCase):
    def test_locked_suite_has_five_histories_fifty_turns_and_twenty_questions(self) -> None:
        suite = load_suite(DEFAULT_SUITE_PATH)
        stats = suite_stats(suite)
        self.assertEqual((stats.histories, stats.turns, stats.questions), (5, 50, 20))
        self.assertIn("exclusive_before", stats.axes)
        self.assertIn("standard_poisoning", stats.axes)
        self.assertIn("unsupported_future", stats.axes)

    def test_fixture_extractor_returns_section_a_payload_without_text_extraction(self) -> None:
        suite = load_suite(DEFAULT_SUITE_PATH)
        history = suite["histories"][1]
        turn_input = next(iter_history_turns(history))
        fact_turn = history["turns"][1]
        source_turn = ExtractionTurn(
            turn_id=turn_input.turn_id,
            history_id=turn_input.history_id,
            session_id=turn_input.session_id,
            session_date=turn_input.session_date,
            session_unix_ts=turn_input.session_unix_ts,
            turn_idx=turn_input.turn_idx,
            role=turn_input.role,
            raw_content=turn_input.raw_content,
            source_type=turn_input.source_type or "user_turn",
            event_time=turn_input.session_unix_ts,
        )
        # The journal fixture is the second entry in its history.
        source_turn = ExtractionTurn(
            **{
                **{name: getattr(source_turn, name) for name in source_turn.__dataclass_fields__},
                "turn_id": f"{history['history_id']}:{fact_turn['session']}:0",
                "session_id": fact_turn["session"],
                "session_date": fact_turn["date"],
                "session_unix_ts": 1_771_113_600.0,
                "raw_content": fact_turn["text"],
            }
        )
        payload = json.loads(
            FixtureFactExtractor(history).extract(source_turn, ExtractionPrompt("", ""))
        )
        self.assertEqual(payload[0]["attribute"], "project_started")
        self.assertEqual(payload[0]["as_of"], "2026-01-15")
        self.assertEqual(
            set(payload[0]),
            {"session_id", "turn_idx", "entity", "attribute", "value", "cardinality",
             "kind", "as_of", "replaces", "confidence"},
        )

    def test_event_time_filter_runs_before_rrf_for_ungated_date_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            database = str(Path(root) / "date-filter.sqlite3")
            initialize_database(database)
            with connection(database) as conn:
                for idx, (turn_id, text, event_time) in enumerate((
                    ("old", "purple notebook", 100.0),
                    ("middle", "blue notebook", 200.0),
                    ("new", "orange notebook", 300.0),
                )):
                    conn.execute(
                        """INSERT INTO turns (turn_id, history_id, session_id, turn_idx,
                           role, raw_content, source_type, event_time, recorded_at, hash)
                           VALUES (?, 'h', ?, ?, 'user', ?, 'user_turn', ?, ?, ?)""",
                        (turn_id, f"s{idx}", idx, text, event_time, event_time + 1, turn_id),
                    )
                    record_store_event(conn, turn_id, created_at=event_time + 1)
            provider = FakeEmbedding()
            index_history(database, "h", provider, gated=False)
            result = hybrid_rag(
                database, "h", "orange notebook", provider,
                top_k=1, candidate_limit=10, gated=False,
                event_time_range=TemporalRange(end=TemporalBoundary(200.0)),
            )
            self.assertEqual(len(result.turns), 1)
            self.assertLessEqual(result.turns[0].event_time, 200.0)
            self.assertNotEqual(result.turns[0].turn_id, "new")

    def test_offline_runner_persists_each_system_question_and_zero_cost(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            database = str(Path(root) / "offline-eval.sqlite3")
            summary = run_evaluation(
                database=database,
                run_id="offline-test-v1",
                suite_path=DEFAULT_SUITE_PATH,
                embedding_kind="fake",
            )
            self.assertEqual(summary["suite"]["histories"], 5)
            self.assertEqual(summary["suite"]["turns"], 50)
            self.assertEqual(summary["persisted_question_results"], 80)
            self.assertEqual(summary["total_credits"]["total"], 0.0)
            self.assertFalse(summary["failures"])
            with connection(database) as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM eval_question_results WHERE run_id=?",
                    ("offline-test-v1",),
                ).fetchone()[0]
                local_calls = conn.execute(
                    "SELECT COUNT(*) FROM llm_cache WHERE cost_credits != 0"
                ).fetchone()[0]
            self.assertEqual(count, 80)
            self.assertEqual(local_calls, 0)


if __name__ == "__main__":
    unittest.main()
