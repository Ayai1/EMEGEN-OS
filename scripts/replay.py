"""Read-only SQLite replay for temporal, gate, and authored-eval evidence.

This script deliberately imports no retrieval or provider code. The database is
opened with SQLite's read-only URI mode, and all output is reconstructed from
the existing event and evaluation tables.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _stamp(value: object) -> str:
    if value is None:
        return "—"
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return str(value)


def _json_cell(value: object) -> str:
    return "—" if value is None else str(value)


def render_report(
    database: str | Path,
    *,
    history_id: str,
    entity: str,
    attribute: str,
    question_id: str,
    run_id: str,
) -> str:
    """Build the replay report using only SELECT statements on a read-only DB."""
    db_path = Path(database).resolve()
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    uri = db_path.as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        timeline = conn.execute(
            """SELECT f.fact_id, f.turn_id, f.value, f.value_norm, f.key_text,
                      f.valid_from, f.valid_until, f.superseded_by, f.recorded_at,
                      f.expired_at, f.trust, f.quarantined, f.quarantine_reason,
                      GROUP_CONCAT(o.observation_id, ',') AS observation_ids
               FROM facts AS f
               LEFT JOIN fact_observations AS o
                 ON o.canonical_fact_id = f.fact_id
               WHERE f.history_id = ? AND f.key_type = 'slot'
                 AND lower(f.entity) = lower(?) AND lower(f.attribute) = lower(?)
               GROUP BY f.fact_id
               ORDER BY f.valid_from, f.fact_id""",
            (history_id, entity, attribute),
        ).fetchall()
        gate_rows = conn.execute(
            """SELECT t.turn_id, t.event_time, t.source_type, t.raw_content,
                      g.signal_source, g.signal_policy, g.signal_bridge,
                      g.signals_fired, g.decision
               FROM turns AS t
               LEFT JOIN gate_events AS g ON g.turn_id = t.turn_id
               WHERE t.history_id = ?
               ORDER BY t.event_time, t.turn_idx, g.event_id""",
            (history_id,),
        ).fetchall()
        answers = conn.execute(
            """SELECT q.system, q.axis, q.answer_text, q.abstained,
                      q.cited_ids_json, q.retrieved_ids_json,
                      q.expected_answer, q.expected_label, q.correct,
                      q.gate_outcome_json, q.prompt_tokens, q.completion_tokens,
                      q.latency_ms, q.cost_credits, q.cache_request_hash,
                      r.reader_model, r.judge_model, r.reader_prompt_hash,
                      r.judge_prompt_hash, r.run_config_json, r.answer AS run_answer,
                      r.correct AS run_correct
               FROM eval_question_results AS q
               JOIN eval_runs AS r
                 ON r.run_id = q.run_id AND r.system = q.system
                AND r.question_id = q.question_id
               WHERE q.run_id = ? AND q.question_id = ?
               ORDER BY q.system""",
            (run_id, question_id),
        ).fetchall()
        cache_hashes = sorted(
            {str(row["cache_request_hash"]) for row in answers if row["cache_request_hash"]}
        )
        cache_rows: list[sqlite3.Row] = []
        for request_hash in cache_hashes:
            cache_rows.extend(
                conn.execute(
                    """SELECT model, prompt_hash, task_tag, phase_tag, run_id,
                              in_tokens, out_tokens, cost_credits, latency_ms
                       FROM llm_cache WHERE prompt_hash = ?
                       ORDER BY created_at, model""",
                    (request_hash,),
                ).fetchall()
            )
    finally:
        conn.close()

    lines = [
        f"Replay: history={history_id} slot={entity}.{attribute} "
        f"run={run_id} question={question_id}",
        "",
        "INTERVAL TIMELINE (facts projection; observations retain source IDs)",
        "fact_id | value | valid_from_utc | valid_until_utc | observation_ids | "
        "source_turn | trust | quarantined | reason",
    ]
    for row in timeline:
        lines.append(
            " | ".join(
                (
                    str(row["fact_id"]), str(row["value"]), _stamp(row["valid_from"]),
                    _stamp(row["valid_until"]), _json_cell(row["observation_ids"]),
                    str(row["turn_id"]), str(row["trust"]), str(row["quarantined"]),
                    _json_cell(row["quarantine_reason"]),
                )
            )
        )
    if not timeline:
        lines.append("(no matching slot facts)")

    lines.extend(
        [
            "",
            "GATE AUDIT (one row per source turn / gate event)",
            "turn_id | event_time_utc | source | source_signal | policy_signal | "
            "bridge_signal | fired | decision | turn_text",
        ]
    )
    for row in gate_rows:
        text = str(row["raw_content"]).replace("\r", " ").replace("\n", " ")
        lines.append(
            " | ".join(
                (
                    str(row["turn_id"]), _stamp(row["event_time"]), str(row["source_type"]),
                    _json_cell(row["signal_source"]), _json_cell(row["signal_policy"]),
                    _json_cell(row["signal_bridge"]), _json_cell(row["signals_fired"]),
                    _json_cell(row["decision"]), text,
                )
            )
        )
    if not gate_rows:
        lines.append("(no turns for history)")

    lines.extend(
        [
            "",
            "SIDE-BY-SIDE EVAL ANSWERS",
            "system | answer | abstained | correct | expected | label | cited_ids | "
            "retrieved_ids | gate_outcome | tokens(in/out) | latency_ms | credits | cache_hash",
        ]
    )
    for row in answers:
        lines.append(
            " | ".join(
                (
                    str(row["system"]), str(row["answer_text"]).replace("\n", " "),
                    str(row["abstained"]), str(row["correct"]),
                    _json_cell(row["expected_answer"]), str(row["expected_label"]),
                    str(row["cited_ids_json"]), str(row["retrieved_ids_json"]),
                    _json_cell(row["gate_outcome_json"]),
                    f"{row['prompt_tokens']}/{row['completion_tokens']}",
                    str(row["latency_ms"]), str(row["cost_credits"]),
                    _json_cell(row["cache_request_hash"]),
                )
            )
        )
        lines.append(
            f"  reader={row['reader_model']} judge={row['judge_model']} "
            f"reader_prompt={row['reader_prompt_hash']} "
            f"judge_prompt={row['judge_prompt_hash']} "
            f"run_config={row['run_config_json']}"
        )
    if not answers:
        lines.append("(no result rows for run/question)")

    lines.extend(["", "LLM CACHE REQUESTS FOR RESULT HASHES", "model | task | phase | run | in/out | credits | latency_ms | request_hash"])
    for row in cache_rows:
        lines.append(
            " | ".join(
                (
                    str(row["model"]), str(row["task_tag"]), str(row["phase_tag"]),
                    str(row["run_id"]), f"{row['in_tokens']}/{row['out_tokens']}",
                    str(row["cost_credits"]), _json_cell(row["latency_ms"]),
                    str(row["prompt_hash"]),
                )
            )
        )
    if not cache_rows:
        lines.append("(no matching cache request rows)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--history-id", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--attribute", required=True)
    parser.add_argument("--question-id", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    print(
        render_report(
            args.database,
            history_id=args.history_id,
            entity=args.entity,
            attribute=args.attribute,
            question_id=args.question_id,
            run_id=args.run_id,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
