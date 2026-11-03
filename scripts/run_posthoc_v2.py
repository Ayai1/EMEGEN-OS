"""Run the labeled post-hoc v2 evaluation without mutating any v1 artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from engramOS.db import initialize_database  # noqa: E402
from engramOS.local_llama import LocalModelSettings  # noqa: E402
from scripts.run_eval import run_evaluation, SYSTEMS, _rate  # noqa: E402

V1_DATABASE = PROJECT_ROOT / "eval" / "local-q4-canary-classifier1024.sqlite3"
V2_DATABASE = PROJECT_ROOT / "eval" / "local-q4-posthoc-v2.sqlite3"
V1_SUITE = PROJECT_ROOT / "eval" / "suite_v2.json"
HELDOUT_SUITE = PROJECT_ROOT / "eval" / "suite_v2_heldout.json"


def copy_database(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"refusing to overwrite v2 database: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as source_conn, sqlite3.connect(target) as target_conn:
        source_conn.backup(target_conn)
    initialize_database(target)
    with sqlite3.connect(target) as conn:
        v1_rows = conn.execute(
            "SELECT COUNT(*) FROM eval_question_results WHERE run_id=?",
            ("q4-locked-classifier1024-resume300-v1",),
        ).fetchone()[0]
    if v1_rows != 152:
        raise RuntimeError(f"expected to preserve 152 v1 rows in the copied DB, found {v1_rows}")


def settings(
    *,
    reader_max_tokens: int = 2048,
    classifier_max_tokens: int = 1024,
    reasoning_budget: int = 512,
) -> LocalModelSettings:
    return LocalModelSettings(
        base_url="http://127.0.0.1:8080",
        model="Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
        model_sha256="82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38",
        quantization="UD-Q4_K_XL",
        context_size=8192,
        gpu_layers=32,
        parallel_slots=1,
        kv_cache_type="q8_0",
        flash_attention=True,
        chat_template="gguf_builtin",
        reasoning_mode="low",
        reasoning_effort="low",
        reasoning_budget=reasoning_budget,
        reasoning_strength="low",
        temperature=0.0,
        top_p=0.95,
        top_k=40,
        seed=42,
        reader_max_tokens=reader_max_tokens,
        classifier_max_tokens=classifier_max_tokens,
    )


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def add_posthoc_attack_reporting(summary: dict[str, object]) -> None:
    """Separate the trusted override from untrusted attack denominators."""
    question_rows = summary.get("question_results", [])
    if not isinstance(question_rows, list):
        raise TypeError("evaluation summary question_results must be a list")
    by_system: dict[str, dict[str, object]] = {}
    for system in SYSTEMS:
        rows = [
            row for row in question_rows
            if isinstance(row, dict) and row.get("system") == system and row.get("attack_turn_id")
        ]
        trusted_channel = [row for row in rows if row.get("question_id") == "poison_adaptive_user"]
        untrusted = [row for row in rows if row.get("question_id") != "poison_adaptive_user"]

        def metrics(group: list[dict[str, object]]) -> dict[str, object]:
            exposed = sum(bool(row.get("attack_reached_reader")) for row in group)
            with_claim = [row for row in group if row.get("attack_claim")]
            adopted = sum(bool(row.get("attack_claim_adopted")) for row in with_claim)
            return {
                "reader_reached": _rate(exposed, len(group)),
                "claim_adopted": _rate(adopted, len(with_claim)),
            }

        by_system[system] = {
            "all_attacks_including_compromised_trusted_channel": metrics(rows),
            "untrusted_attacks_excluding_compromised_trusted_channel": metrics(untrusted),
            "compromised_trusted_channel_only": metrics(trusted_channel),
        }
    summary["posthoc_attack_reporting"] = {
        "separate_case": {
            "question_id": "poison_adaptive_user",
            "label": "compromised_trusted_channel",
        },
        "by_system": by_system,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=V2_DATABASE)
    parser.add_argument("--source-database", type=Path, default=V1_DATABASE)
    parser.add_argument("--progress-dir", type=Path, default=PROJECT_ROOT / "logs")
    parser.add_argument("--skip-copy", action="store_true")
    parser.add_argument("--context-version", choices=("v2", "v3"), default="v2")
    parser.add_argument("--strict-untrusted", action="store_true")
    parser.add_argument("--reader-max-tokens", type=int, default=2048)
    parser.add_argument("--classifier-max-tokens", type=int, default=1024)
    parser.add_argument("--reasoning-budget", type=int, default=512)
    args = parser.parse_args()

    if not args.skip_copy:
        copy_database(args.source_database, args.database)
    elif not args.database.exists():
        raise FileNotFoundError(args.database)

    cfg = settings(
        reader_max_tokens=args.reader_max_tokens,
        classifier_max_tokens=args.classifier_max_tokens,
        reasoning_budget=args.reasoning_budget,
    )
    evaluations = (
        (
            "q4-post-hoc-v2-engram-v1",
            V1_SUITE,
            ("engram",),
            "results_posthoc_v2_engram_v1.json",
            "posthoc-v2-engram-progress.jsonl",
        ),
        (
            "q4-post-hoc-v2-heldout-v1",
            HELDOUT_SUITE,
            SYSTEMS,
            "results_posthoc_v2_heldout_v1.json",
            "posthoc-v2-heldout-progress.jsonl",
        ),
    )
    for run_id, suite_path, systems, result_name, progress_name in evaluations:
        result_path = PROJECT_ROOT / "eval" / result_name
        progress_path = args.progress_dir / progress_name
        summary = run_evaluation(
            database=str(args.database),
            run_id=run_id,
            suite_path=suite_path,
            embedding_kind="bge",
            mode="local",
            local_settings=cfg,
            progress_log=progress_path,
            provider_timeout=300.0,
            systems=systems,
            context_version=args.context_version,
            strict_untrusted=args.strict_untrusted,
        )
        if run_id == "q4-post-hoc-v2-engram-v1":
            add_posthoc_attack_reporting(summary)
        write_summary(result_path, summary)
        print(json.dumps({
            "run_id": run_id,
            "summary_path": str(result_path),
            "persisted_question_results": summary["persisted_question_results"],
            "provider_failure_result_count": summary["provider_failure_result_count"],
            "failures": summary["failures"],
        }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
