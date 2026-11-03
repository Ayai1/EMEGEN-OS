"""Database identity checks for explicit lexical-only evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from engramOS.db import connection, initialize_database  # noqa: E402
from scripts.run_eval import (  # noqa: E402
    _ensure_embedding_identity,
    _embedding_provider,
    run_evaluation,
)


class NoEmbeddingIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = str(Path(self.temp_dir.name) / "identity.sqlite3")
        initialize_database(self.database)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_none_database_rejects_bge_identity(self) -> None:
        self.assertIsNone(_embedding_provider("none"))
        _ensure_embedding_identity(self.database, "none")
        with self.assertRaisesRegex(ValueError, "different embedding identity"):
            _ensure_embedding_identity(self.database, "BAAI/bge-m3@fixture")

    def test_bge_database_rejects_none_identity(self) -> None:
        _ensure_embedding_identity(self.database, "BAAI/bge-m3@fixture")
        with self.assertRaisesRegex(ValueError, "different embedding identity"):
            _ensure_embedding_identity(self.database, "none")

    def test_offline_none_run_records_embedding_identity_without_indexing(self) -> None:
        suite = PROJECT_ROOT / "eval" / "suite_v2.json"
        summary = run_evaluation(
            database=self.database,
            run_id="test-lexical-only-v3",
            suite_path=suite,
            embedding_kind="none",
            mode="offline",
            question_ids={"city_current"},
            systems=("engram",),
            context_version="v3",
        )
        self.assertEqual(summary["persisted_question_results"], 1)
        self.assertEqual(summary["provider_failure_result_count"], 0)
        with connection(self.database) as conn:
            row = conn.execute(
                """SELECT run_config_json FROM eval_runs
                   WHERE run_id='test-lexical-only-v3' AND system='engram'
                     AND question_id='city_current'"""
            ).fetchone()
            embedding_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        config = json.loads(row["run_config_json"])
        self.assertEqual(config["embedding"], "none")
        self.assertEqual(config["embedding_model"], "none")
        self.assertEqual(embedding_count, 0)


if __name__ == "__main__":
    unittest.main()
