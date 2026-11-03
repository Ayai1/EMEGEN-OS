"""Build a local replay gallery for authored attack and benign-control cases.

The builder reads the locked suite, a saved run summary, and gate events from a
read-only SQLite connection. It never opens providers or makes network calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any


SYSTEM_ORDER = ("recency", "hybrid_rag", "hybrid_rag_date_filter", "engram")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _turn_index(suite: dict[str, Any]) -> dict[str, dict[str, Any]]:
    turns: dict[str, dict[str, Any]] = {}
    for history in suite["histories"]:
        history_id = history["history_id"]
        for index, turn in enumerate(history["turns"]):
            session = turn["session"]
            turn_id = f"{history_id}:{session}:0"
            turns[turn_id] = {
                "history_id": history_id,
                "session_id": session,
                "turn_index": index,
                "date": turn["date"],
                "source_type": turn.get("source_type", "user"),
                "text": turn["text"],
                "facts": turn.get("facts", []),
            }
    return turns


def _gate_index(database: Path) -> dict[str, dict[str, Any]]:
    uri = database.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT turn_id, signal_source, signal_policy, signal_bridge,
                      signals_fired, decision
               FROM gate_events ORDER BY event_id"""
        ).fetchall()
    finally:
        conn.close()
    return {
        row["turn_id"]: {
            "signals": {
                "signal_source": bool(row["signal_source"]),
                "signal_policy": bool(row["signal_policy"]),
                "signal_bridge": bool(row["signal_bridge"]),
            },
            "signals_fired": row["signals_fired"],
            "decision": row["decision"],
        }
        for row in rows
    }


def build_gallery(
    *,
    suite_path: Path,
    results_path: Path,
    database_path: Path,
    run_id: str,
) -> dict[str, Any]:
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results.get("run_id") != run_id:
        raise ValueError(f"results run_id is {results.get('run_id')!r}, expected {run_id!r}")
    suite_hash = _sha256(suite_path)
    if results.get("suite", {}).get("questions") != len(suite.get("questions", [])):
        raise ValueError("suite question count does not match the saved run summary")

    turns = _turn_index(suite)
    gate = _gate_index(database_path)
    rows_by_question: dict[str, dict[str, dict[str, Any]]] = {}
    for row in results.get("question_results", []):
        rows_by_question.setdefault(row["question_id"], {})[row["system"]] = row

    attacks: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    seen_attack_turns: set[str] = set()
    seen_control_turns: set[str] = set()

    def case_outcomes(question_id: str, turn_id: str, *, attack: bool) -> dict[str, Any]:
        question_rows = rows_by_question.get(question_id, {})
        outcomes: dict[str, Any] = {}
        for system in SYSTEM_ORDER:
            row = question_rows.get(system)
            if row is None:
                continue
            reached = turn_id in row.get("retrieved_ids", []) or f"turn:{turn_id}" in row.get("retrieved_ids", [])
            outcomes[system] = {
                "answer": row.get("answer"),
                "abstained": row.get("abstained"),
                "correct": row.get("correct"),
                "provider_failure": row.get("provider_failure", False),
                "provider_failure_types": row.get("provider_failure_types", []),
                "target_turn_reached_reader": bool(reached),
                "claim_adopted": row.get("attack_claim_adopted") if attack else None,
                "quarantine_leak": row.get("quarantine_leak", False) if attack else None,
                "retrieved_ids": row.get("retrieved_ids", []),
            }
        return outcomes

    for question in suite["questions"]:
        question_id = question["question_id"]
        attack_turn_id = question.get("attack_turn_id")
        if attack_turn_id and attack_turn_id not in seen_attack_turns:
            turn = turns.get(attack_turn_id)
            gate_row = gate.get(attack_turn_id)
            if turn is None or gate_row is None:
                raise ValueError(f"missing attack turn or gate event for {attack_turn_id}")
            seen_attack_turns.add(attack_turn_id)
            attacks.append({
                "case_type": "attack",
                "question_id": question_id,
                "axis": question["axis"],
                "query": question["query"],
                "expected_answer": question["expected_answer"],
                "turn_id": attack_turn_id,
                "turn": turn,
                "attack_claim": suite.get("attack_claims", {}).get(question_id),
                "gate": gate_row,
                "outcomes": case_outcomes(question_id, attack_turn_id, attack=True),
            })

        control_turn_id = question.get("benign_turn_id")
        if control_turn_id and control_turn_id not in seen_control_turns:
            turn = turns.get(control_turn_id)
            gate_row = gate.get(control_turn_id)
            if turn is None or gate_row is None:
                raise ValueError(f"missing control turn or gate event for {control_turn_id}")
            seen_control_turns.add(control_turn_id)
            controls.append({
                "case_type": "benign_control",
                "question_id": question_id,
                "axis": question["axis"],
                "query": question["query"],
                "expected_answer": question["expected_answer"],
                "turn_id": control_turn_id,
                "turn": turn,
                "gate": gate_row,
                "outcomes": case_outcomes(question_id, control_turn_id, attack=False),
            })

    return {
        "format_version": 1,
        "run_id": run_id,
        "suite_sha256": suite_hash,
        "suite_questions": len(suite["questions"]),
        "systems": list(SYSTEM_ORDER),
        "attack_case_count": len(attacks),
        "control_case_count": len(controls),
        "attacks": attacks,
        "controls": controls,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    gallery = build_gallery(
        suite_path=args.suite,
        results_path=args.results,
        database_path=args.database,
        run_id=args.run_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gallery, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: gallery[key] for key in (
        "run_id", "suite_sha256", "suite_questions", "attack_case_count", "control_case_count"
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
