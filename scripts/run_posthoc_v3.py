"""Run the explicitly labeled post-hoc v3 evaluation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from eval.loader import load_suite  # noqa: E402
from scripts.run_eval import SYSTEMS, run_evaluation  # noqa: E402
from scripts.run_posthoc_v2 import settings, write_summary  # noqa: E402

RUN_ID_PREFIX = "q4-post-hoc-v3-"
FORBIDDEN_RUN_IDS = frozenset({
    "q4-locked-classifier1024-resume300-v1",
    "q4-post-hoc-v2-engram-v1",
    "q4-post-hoc-v2-heldout-v1",
})
FORBIDDEN_DATABASE_NAMES = frozenset({
    "local-q4-canary-classifier1024.sqlite3",
    "local-q4-posthoc-v2.sqlite3",
})
DEFAULT_SUITE = PROJECT_ROOT / "eval" / "suite_v2.json"

HELP_COMMANDS = """Commands (run in separate terminals/shells as indicated):

1. Start the local server:
   powershell -ExecutionPolicy Bypass -File scripts/local_server.ps1

2. Run the smoke preset (two A-to-B-to-A questions plus attack_ext_subtle_jacket):
   .\\.venv\\Scripts\\python.exe scripts\\run_posthoc_v3.py --database eval\\local-q4-posthoc-v3.sqlite3 --run-id q4-post-hoc-v3-smoke --preset smoke --embedding bge

3. Run the same smoke preset without an embedding model, in its own database:
   .\\.venv\\Scripts\\python.exe scripts\\run_posthoc_v3.py --database eval\\local-q4-posthoc-v3-none.sqlite3 --run-id q4-post-hoc-v3-smoke-none --preset smoke --embedding none

4. Run the held-out suite across all four systems in a separate database:
   .\\.venv\\Scripts\\python.exe scripts\\run_posthoc_v3.py --database eval\\local-q4-posthoc-v3-heldout.sqlite3 --run-id q4-post-hoc-v3-heldout-v1 --suite eval\\suite_v2_heldout.json --systems recency hybrid_rag hybrid_rag_date_filter engram --embedding bge

5. Run the full 38-question Engram suite after the smoke results look sane:
   .\\.venv\\Scripts\\python.exe scripts\\run_posthoc_v3.py --database eval\\local-q4-posthoc-v3.sqlite3 --run-id q4-post-hoc-v3-engram-v1 --resume --embedding bge"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=HELP_COMMANDS,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--only", nargs="+", metavar="QUESTION_ID")
    selection.add_argument("--preset", choices=("smoke",))
    parser.add_argument("--expect-suite-sha256")
    parser.add_argument("--embedding", choices=("bge", "none"), default="bge")
    parser.add_argument("--strict-untrusted", action="store_true")
    parser.add_argument("--slot-aliases", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--progress-log", type=Path)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS)
    parser.add_argument("--mode", choices=("local", "offline"), default="local")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model-file", default="Muse-Glimmer-30B-UD-Q4_K_XL.gguf")
    parser.add_argument(
        "--model-sha256",
        default="82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38",
    )
    parser.add_argument("--reader-max-tokens", type=int, default=2048)
    parser.add_argument("--classifier-max-tokens", type=int, default=1024)
    parser.add_argument("--reasoning-budget", type=int, default=512)
    parser.add_argument("--provider-timeout", type=float, default=300.0)
    return parser


def validate_run_id(run_id: str) -> None:
    if run_id.startswith("q4-locked") or run_id in FORBIDDEN_RUN_IDS:
        raise ValueError(f"refusing protected run ID: {run_id!r}")
    suffix = run_id[len(RUN_ID_PREFIX):] if run_id.startswith(RUN_ID_PREFIX) else ""
    if not suffix or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", suffix):
        raise ValueError(f"run ID must start with {RUN_ID_PREFIX!r}")


def validate_run_guards(
    *,
    database: Path,
    run_id: str,
    results_path: Path,
    progress_path: Path,
    resume: bool,
) -> None:
    """Apply every write-protection guard before opening the database/provider."""
    validate_run_id(run_id)
    if database.name.casefold() in {name.casefold() for name in FORBIDDEN_DATABASE_NAMES}:
        raise ValueError(f"refusing protected database filename: {database.name!r}")
    if database.exists() and not resume:
        raise FileExistsError(f"database already exists; pass --resume to continue: {database}")
    for path, label in ((results_path, "results"), (progress_path, "progress")):
        if path.exists() and not resume:
            raise FileExistsError(f"{label} path already exists; pass --resume to continue: {path}")


def result_paths(
    run_id: str,
    *,
    results: Path | None = None,
    progress_log: Path | None = None,
) -> tuple[Path, Path]:
    validate_run_id(run_id)
    label = run_id[len(RUN_ID_PREFIX):]
    return (
        results or PROJECT_ROOT / "eval" / f"results_posthoc_v3_{label}.json",
        progress_log or PROJECT_ROOT / "logs" / f"posthoc_v3_{label}_progress.jsonl",
    )


def smoke_question_ids(suite: dict[str, Any]) -> list[str]:
    questions = suite.get("questions")
    if not isinstance(questions, list):
        raise ValueError("suite questions must be a list")
    axis_ids = [
        str(question["question_id"])
        for question in questions
        if isinstance(question, dict)
        and question.get("axis") in {"a_b_a_supersession", "a_to_b_to_a_supersession"}
    ]
    attack_ids = [
        str(question["question_id"])
        for question in questions
        if isinstance(question, dict)
        and question.get("question_id") == "attack_ext_subtle_jacket"
    ]
    selected = list(dict.fromkeys((*axis_ids, *attack_ids)))
    if len(axis_ids) != 2 or len(attack_ids) != 1 or len(selected) != 3:
        raise ValueError(
            "smoke preset requires exactly two A-to-B-to-A questions and the "
            "attack_ext_subtle_jacket question in the selected suite"
        )
    return selected


def execute(args: argparse.Namespace) -> dict[str, Any]:
    validate_run_id(args.run_id)
    results_path, progress_path = result_paths(
        args.run_id, results=args.results, progress_log=args.progress_log,
    )
    validate_run_guards(
        database=args.database,
        run_id=args.run_id,
        results_path=results_path,
        progress_path=progress_path,
        resume=args.resume,
    )

    suite_bytes = args.suite.read_bytes()
    suite_sha256 = hashlib.sha256(suite_bytes).hexdigest()
    if args.expect_suite_sha256 is not None and args.expect_suite_sha256.lower() != suite_sha256:
        raise ValueError(
            f"suite SHA-256 mismatch: expected {args.expect_suite_sha256!r}, got {suite_sha256}"
        )
    suite = load_suite(args.suite)
    if hashlib.sha256(args.suite.read_bytes()).hexdigest() != suite_sha256:
        raise ValueError("suite changed while it was being loaded")
    if args.preset == "smoke":
        selected_ids = smoke_question_ids(suite)
    elif args.only:
        selected_ids = list(dict.fromkeys(args.only))
    else:
        selected_ids = [str(question["question_id"]) for question in suite["questions"]]

    local_settings = None
    if args.mode == "local":
        local_settings = settings(
            reader_max_tokens=args.reader_max_tokens,
            classifier_max_tokens=args.classifier_max_tokens,
            reasoning_budget=args.reasoning_budget,
        )
        local_settings = replace(
            local_settings,
            base_url=args.base_url,
            model=args.model_file,
            model_sha256=args.model_sha256,
        )

    summary = run_evaluation(
        database=str(args.database),
        run_id=args.run_id,
        suite_path=args.suite,
        embedding_kind=args.embedding,
        mode=args.mode,
        local_settings=local_settings,
        question_ids=set(selected_ids),
        progress_log=progress_path,
        provider_timeout=args.provider_timeout,
        systems=tuple(args.systems) if args.systems else ("engram",),
        context_version="v3",
        strict_untrusted=args.strict_untrusted,
        slot_aliases=args.slot_aliases,
        expect_suite_sha256=args.expect_suite_sha256,
    )
    summary["selected_question_ids"] = selected_ids
    summary["preset"] = args.preset
    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_summary(results_path, summary)
    output = {
        "run_id": args.run_id,
        "results_path": str(results_path),
        "progress_path": str(progress_path),
        "selected_question_ids": selected_ids,
        "persisted_question_results": summary["persisted_question_results"],
        "provider_failure_result_count": summary["provider_failure_result_count"],
        "failures": summary["failures"],
    }
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = execute(args)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.error(str(exc))
        return 2
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
