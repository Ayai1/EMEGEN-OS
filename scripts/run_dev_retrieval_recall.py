"""Run an offline LongMemEval-S dev retrieval-recall check with real BGE-M3."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engramOS.db import connection, initialize_database
from engramOS.embeddings import BgeM3EmbeddingProvider, index_history
from engramOS.retrieval import hybrid_rag, recency


def run(
    dataset_path: Path,
    split_path: Path,
    database_path: Path,
    output_path: Path,
) -> dict[str, object]:
    raw = dataset_path.read_bytes()
    dataset_hash = hashlib.sha256(raw).hexdigest()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("dataset_sha256") != dataset_hash:
        raise ValueError("dataset SHA-256 does not match the locked split")
    rows = json.loads(raw)
    selected_ids = set(split["dev"]["knowledge-update"] + split["dev"]["temporal-reasoning"])
    selected = [row for row in rows if row["question_id"] in selected_ids]
    if len(selected) != 22:
        raise ValueError(f"expected 22 dev questions, found {len(selected)}")
    if database_path.exists():
        raise FileExistsError(f"refusing to overwrite existing database: {database_path}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output_path}")

    database_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    initialize_database(database_path)
    total_sessions = 0
    total_turns = 0
    with connection(database_path) as conn:
        for question in selected:
            history_id = question["question_id"]
            session_rows: list[tuple[str, str, str, int, str, str, float, str]] = []
            for session_ordinal, (session_id, date_text, turns) in enumerate(zip(
                question["haystack_session_ids"],
                question["haystack_dates"],
                question["haystack_sessions"],
                strict=True,
            )):
                session_time = datetime.strptime(
                    date_text, "%Y/%m/%d (%a) %H:%M"
                ).replace(tzinfo=timezone.utc).timestamp()
                total_sessions += 1
                for turn_idx, turn in enumerate(turns):
                    role = turn["role"]
                    if role not in ("user", "assistant"):
                        raise ValueError(f"unsupported dataset role: {role!r}")
                    raw_content = turn["content"]
                    turn_id = f"{history_id}:{session_id}:instance-{session_ordinal}:{turn_idx}"
                    event_time = session_time + turn_idx
                    digest = hashlib.sha256(
                        f"{history_id}\0{session_id}\0{turn_idx}\0{raw_content}".encode("utf-8")
                    ).hexdigest()
                    session_rows.append((
                        turn_id, history_id, session_id, turn_idx, role,
                        raw_content, event_time, digest,
                    ))
                    total_turns += 1
            conn.executemany(
                """INSERT INTO turns (
                       turn_id, history_id, session_id, turn_idx, role, raw_content,
                       source_type, event_time, recorded_at, hash
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    (turn_id, history, session, turn_idx, role, content,
                     "user_turn" if role == "user" else "assistant_turn",
                     event_time, event_time + 1_000_000, digest)
                    for (turn_id, history, session, turn_idx, role, content,
                         event_time, digest) in session_rows
                ),
            )
            conn.executemany(
                """INSERT INTO gate_events (
                       turn_id, signal_bridge, signal_policy, signal_source,
                       signals_fired, decision, created_at
                   ) VALUES (?, 0, 0, 0, 0, 'store', ?)""",
                ((row[0], row[6] + 1_000_000) for row in session_rows),
            )

    provider = BgeM3EmbeddingProvider(device="cuda", batch_size=16)
    model_id = provider.model_id
    per_question: list[dict[str, object]] = []
    for question in selected:
        history_id = question["question_id"]
        index_history(database_path, history_id, provider)
        hybrid = hybrid_rag(
            database_path, history_id, question["question"], provider,
            top_k=10, candidate_limit=50,
        )
        newest = recency(
            database_path, history_id, question["question"], provider,
            top_k=10, candidate_limit=50,
        )
        gold_sessions = set(question["answer_session_ids"])
        hybrid_sessions = {turn.session_id for turn in hybrid.turns}
        recency_sessions = {turn.session_id for turn in newest.turns}
        hybrid_hits = len(gold_sessions & hybrid_sessions)
        recency_hits = len(gold_sessions & recency_sessions)
        per_question.append({
            "question_id": history_id,
            "question_type": question["question_type"],
            "gold_session_count": len(gold_sessions),
            "hybrid_hit": bool(hybrid_hits),
            "hybrid_recall": hybrid_hits / len(gold_sessions) if gold_sessions else None,
            "recency_hit": bool(recency_hits),
            "recency_recall": recency_hits / len(gold_sessions) if gold_sessions else None,
            "same_retrieved_candidate_set": {
                turn.turn_id for turn in hybrid.turns
            } == {turn.turn_id for turn in newest.turns},
        })

    def summarize(name: str) -> dict[str, object]:
        entries = [item for item in per_question if item["question_type"] == name]
        key = "hybrid" if name == "hybrid_rag" else "recency"
        recalls = [float(item[f"{key}_recall"]) for item in entries
                   if item[f"{key}_recall"] is not None]
        hits = sum(bool(item[f"{key}_hit"]) for item in entries)
        return {
            "n": len(entries),
            "hit_at_10": hits / len(entries) if entries else None,
            "mean_gold_session_recall_at_10": sum(recalls) / len(recalls) if recalls else None,
        }

    report = {
        "dataset": "LongMemEval-S cleaned",
        "dataset_sha256": dataset_hash,
        "split_seed": split["seed"],
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "embedding_model_id": model_id,
        "device": provider.device,
        "retrieval_only": True,
        "answer_accuracy_measured": False,
        "history_count": len(selected),
        "session_count": total_sessions,
        "turn_count": total_turns,
        "metrics": {
            "hybrid_rag": {
                "knowledge-update": summarize("knowledge-update"),
                "temporal-reasoning": summarize("temporal-reasoning"),
            },
            "recency": {
                "knowledge-update": summarize("knowledge-update"),
                "temporal-reasoning": summarize("temporal-reasoning"),
            },
        },
        "all_recency_candidates_match_hybrid": all(
            bool(item["same_retrieved_candidate_set"]) for item in per_question
        ),
        "per_question": per_question,
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/longmemeval/longmemeval_s_cleaned.json")
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "splits.json")
    parser.add_argument("--database", type=Path, default=PROJECT_ROOT / "data/eval/dev_retrieval.sqlite3")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/eval/dev_retrieval_recall.json")
    args = parser.parse_args()
    report = run(args.dataset, args.split, args.database, args.output)
    print(json.dumps({
        "output": str(args.output),
        "history_count": report["history_count"],
        "session_count": report["session_count"],
        "turn_count": report["turn_count"],
        "metrics": report["metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
