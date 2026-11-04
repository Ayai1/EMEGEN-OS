"""Safety and fake-provider tests for the post-hoc v3 runner."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from engramOS.db import connection  # noqa: E402
from eval.loader import load_suite  # noqa: E402
from scripts.run_posthoc_v3 import (  # noqa: E402
    DEFAULT_SUITE,
    FORBIDDEN_DATABASE_NAMES,
    FORBIDDEN_RUN_IDS,
    build_parser,
    execute,
    result_paths,
    smoke_question_ids,
    validate_run_guards,
    validate_run_id,
)


class PosthocV3GuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.database = self.root / "new-v3.sqlite3"
        self.results = self.root / "results.json"
        self.progress = self.root / "progress.jsonl"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def validate(self, *, run_id: str = "q4-post-hoc-v3-test", resume: bool = False) -> None:
        validate_run_guards(
            database=self.database,
            run_id=run_id,
            results_path=self.results,
            progress_path=self.progress,
            resume=resume,
        )

    def test_database_and_run_id_are_required(self) -> None:
        parser = build_parser()
        for argv in ([], ["--database", str(self.database)]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    parser.parse_args(argv)
                self.assertEqual(raised.exception.code, 2)

    def test_locked_and_v1_v2_run_ids_are_rejected(self) -> None:
        for run_id in (
            "q4-locked-anything",
            *sorted(FORBIDDEN_RUN_IDS),
        ):
            with self.subTest(run_id=run_id):
                with self.assertRaisesRegex(ValueError, "protected run ID"):
                    validate_run_id(run_id)

    def test_run_id_must_use_v3_prefix(self) -> None:
        with self.assertRaisesRegex(ValueError, "must start"):
            validate_run_id("q4-post-hoc-v2-new")
        with self.assertRaisesRegex(ValueError, "must start"):
            validate_run_id("q4-post-hoc-v3-../escape")

    def test_v1_and_v2_database_filenames_are_rejected(self) -> None:
        for name in FORBIDDEN_DATABASE_NAMES:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "protected database filename"):
                    validate_run_guards(
                        database=self.root / name,
                        run_id="q4-post-hoc-v3-test",
                        results_path=self.results,
                        progress_path=self.progress,
                        resume=True,
                    )

    def test_existing_database_requires_resume(self) -> None:
        self.database.write_bytes(b"temporary")
        with self.assertRaisesRegex(FileExistsError, "database already exists"):
            self.validate()
        self.validate(resume=True)

    def test_existing_results_and_progress_each_require_resume(self) -> None:
        self.results.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "results path already exists"):
            self.validate()
        self.results.unlink()
        self.progress.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "progress path already exists"):
            self.validate()
        self.validate(resume=True)

    def test_default_paths_follow_v3_label_convention(self) -> None:
        results, progress = result_paths("q4-post-hoc-v3-engram-v1")
        self.assertEqual(results, PROJECT_ROOT / "eval" / "results_posthoc_v3_engram-v1.json")
        self.assertEqual(progress, PROJECT_ROOT / "logs" / "posthoc_v3_engram-v1_progress.jsonl")

    def test_smoke_preset_selects_two_a_b_a_axes_and_exact_attack(self) -> None:
        ids = smoke_question_ids(load_suite(DEFAULT_SUITE))
        self.assertEqual(len(ids), 3)
        self.assertEqual(ids.count("attack_ext_subtle_jacket"), 1)
        selected = [q for q in load_suite(DEFAULT_SUITE)["questions"] if q["question_id"] in ids]
        self.assertEqual(sum(q["axis"] == "a_b_a_supersession" for q in selected), 2)

    def test_help_includes_server_smoke_full_and_heldout_commands(self) -> None:
        with redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as raised:
                build_parser().parse_args(["--help"])
        self.assertEqual(raised.exception.code, 0)
        help_text = output.getvalue()
        self.assertIn("scripts/local_server.ps1 -GpuLayers 32", help_text)
        self.assertIn("--preset smoke", help_text)
        self.assertIn("q4-post-hoc-v3-engram-v1", help_text)
        self.assertIn("suite_v2_heldout.json", help_text)
        self.assertIn("--expect-suite-sha256 6b9a2a48e7d2dec311e7064de4b2adbb717ab8a67372a7590f1595f450a2c823", help_text)
        self.assertIn("--expect-suite-sha256 5c64470c0773d9bb4b2d672b1a207c24959784f7cb5e7539497b02ef03dc5b85", help_text)

    def test_cli_defaults_keep_v3_gate_alias_and_embedding_settings(self) -> None:
        args = build_parser().parse_args([
            "--database", str(self.database), "--run-id", "q4-post-hoc-v3-defaults",
        ])
        self.assertEqual(args.suite, DEFAULT_SUITE)
        self.assertEqual(args.embedding, "bge")
        self.assertFalse(args.strict_untrusted)
        self.assertFalse(args.slot_aliases)
        self.assertEqual(
            (args.reader_max_tokens, args.classifier_max_tokens,
             args.reasoning_budget, args.provider_timeout),
            (2048, 1024, 512, 300.0),
        )

    def test_every_guard_runs_before_evaluation_provider_setup(self) -> None:
        args = build_parser().parse_args([
            "--database", str(self.root / "local-q4-posthoc-v2.sqlite3"),
            "--run-id", "q4-post-hoc-v3-guard-test",
        ])
        with mock.patch("scripts.run_posthoc_v3.run_evaluation") as runner:
            with self.assertRaisesRegex(ValueError, "protected database filename"):
                execute(args)
            runner.assert_not_called()


class PosthocV3ExecutionTests(unittest.TestCase):
    def test_smoke_selection_and_options_reach_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = build_parser().parse_args([
                "--database", str(root / "new-v3.sqlite3"),
                "--run-id", "q4-post-hoc-v3-smoke-test",
                "--preset", "smoke",
                "--embedding", "none",
                "--strict-untrusted",
                "--slot-aliases",
                "--results", str(root / "results.json"),
                "--progress-log", str(root / "progress.jsonl"),
            ])
            fake_summary = {
                "persisted_question_results": 3,
                "provider_failure_result_count": 0,
                "failures": [],
            }
            with mock.patch("scripts.run_posthoc_v3.run_evaluation", return_value=fake_summary) as runner:
                with redirect_stdout(io.StringIO()):
                    execute(args)
            kwargs = runner.call_args.kwargs
            self.assertEqual(kwargs["context_version"], "v3")
            self.assertEqual(kwargs["embedding_kind"], "none")
            self.assertTrue(kwargs["strict_untrusted"])
            self.assertTrue(kwargs["slot_aliases"])
            self.assertEqual(len(kwargs["question_ids"]), 3)
            self.assertIn("attack_ext_subtle_jacket", kwargs["question_ids"])

    def test_offline_fake_run_records_v3_flags_suite_hash_and_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database = root / "v3.sqlite3"
            suite = PROJECT_ROOT / "eval" / "suite_v2.json"
            digest = hashlib.sha256(suite.read_bytes()).hexdigest()
            args = build_parser().parse_args([
                "--database", str(database),
                "--run-id", "q4-post-hoc-v3-fake-test",
                "--suite", str(suite),
                "--only", "city_current",
                "--expect-suite-sha256", digest,
                "--embedding", "none",
                "--mode", "offline",
                "--results", str(root / "results.json"),
                "--progress-log", str(root / "progress.jsonl"),
            ])
            with redirect_stdout(io.StringIO()):
                summary = execute(args)
            self.assertEqual(summary["selected_question_ids"], ["city_current"])
            self.assertEqual(summary["provider_failure_result_count"], 0)
            with connection(database) as conn:
                row = conn.execute(
                    """SELECT run_config_json FROM eval_runs
                       WHERE run_id='q4-post-hoc-v3-fake-test'
                         AND system='engram' AND question_id='city_current'"""
                ).fetchone()
                vectors = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            config = json.loads(row["run_config_json"])
            self.assertEqual(config["context_version"], "v3")
            self.assertEqual(config["embedding"], "none")
            self.assertEqual(config["suite_sha256"], digest)
            self.assertFalse(config["strict_untrusted"])
            self.assertFalse(config["slot_aliases"])
            self.assertEqual(vectors, 0)


if __name__ == "__main__":
    unittest.main()
