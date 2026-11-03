"""Compare Q4 Reader behavior on hand-made prompts, outside the locked suite."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engramOS.local_llama import (  # noqa: E402
    LocalModelSettings,
    LocalOpenAIClient,
    LocalReaderProvider,
)
from engramOS.pipeline import answer_turn_baseline_query  # noqa: E402
from engramOS.query_classifier import QueryClassification  # noqa: E402
from engramOS.retrieval import RetrievedTurn  # noqa: E402


MODEL_SHA256 = "82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38"
MODEL_FILE = "Muse-Glimmer-30B-UD-Q4_K_XL.gguf"
PROMPT_FILE = ROOT / "eval" / "dev_reader_prompts.json"
MAX_PROVIDER_TIMEOUT = 900.0


def _event_time(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("development evidence timestamps must include a timezone")
    return parsed.astimezone(timezone.utc).timestamp()


def _load_prompts() -> dict[str, Any]:
    return json.loads(PROMPT_FILE.read_text(encoding="utf-8"))


def _turns(history_id: str, evidence: list[dict[str, str]]) -> list[RetrievedTurn]:
    return [
        RetrievedTurn(
            turn_id=item["id"],
            history_id=history_id,
            session_id=f"session-{item['id']}",
            turn_idx=index,
            role="user",
            raw_content=item["text"],
            source_type="user_turn",
            event_time=_event_time(item["event_time"]),
            recorded_at=_event_time(item["event_time"]),
            dense_score=None,
            bm25_score=None,
            rrf_score=1.0 / (index + 1),
            channels=("dev_reader",),
        )
        for index, item in enumerate(evidence)
    ]


def _grade(answer: str, abstained: bool, citations: list[str], prompt: dict[str, Any]) -> tuple[bool, bool]:
    expected_abstain = bool(prompt["expected_abstain"])
    if expected_abstain:
        answer_ok = abstained and answer == "I don't know."
    else:
        terms = [str(term).casefold() for term in prompt["expected_answer_terms"]]
        answer_ok = not abstained and any(term in answer.casefold() for term in terms)
    citations_ok = set(citations) == set(prompt["expected_citations"])
    if expected_abstain:
        citations_ok = citations_ok and abstained
    return answer_ok, citations_ok


def run_mode(*, mode: str, base_url: str, output: Path) -> dict[str, Any]:
    prompts_payload = _load_prompts()
    settings = LocalModelSettings(
        base_url=base_url,
        model=MODEL_FILE,
        model_sha256=MODEL_SHA256,
        reasoning_mode=mode,
        gpu_layers=32,
    )
    reader = LocalReaderProvider(LocalOpenAIClient(settings))
    classification = QueryClassification(
        "classified", "current", date.fromisoformat(prompts_payload["question_date"]), "present_state"
    )
    rows: list[dict[str, Any]] = []
    for prompt_case in prompts_payload["prompts"]:
        reader.last_completion = None
        outcome = answer_turn_baseline_query(
            prompt_case["query"],
            prompts_payload["history_id"],
            _turns(prompts_payload["history_id"], prompt_case["evidence"]),
            classification,
            reader,
            ordering=prompts_payload["ordering"],
            timeout=MAX_PROVIDER_TIMEOUT,
        )
        completion = reader.last_completion
        citations = [str(candidate_id) for candidate_id in outcome.citations]
        answer_ok, citations_ok = _grade(outcome.answer, outcome.abstained, citations, prompt_case)
        parser_accepted = outcome.reason_code in {"answered", "provider_abstained"}
        row = {
            "prompt_id": prompt_case["prompt_id"],
            "axis": prompt_case["axis"],
            "query": prompt_case["query"],
            "finish_reason": completion.finish_reason if completion else None,
            "prompt_tokens": completion.prompt_tokens if completion else None,
            "completion_tokens": completion.completion_tokens if completion else None,
            "visible_tokens": completion.visible_tokens if completion else None,
            "nonvisible_tokens_estimate": (
                max(0, completion.completion_tokens - completion.visible_tokens)
                if completion else None
            ),
            "latency_ms": round(completion.latency_ms, 2) if completion else None,
            "request_hash": completion.request_hash if completion else None,
            "response_model": completion.response_model if completion else None,
            "parser_accepted": parser_accepted,
            "reader_reason_code": outcome.reason_code,
            "answer": outcome.answer,
            "abstained": outcome.abstained,
            "citations": citations,
            "answer_correct": answer_ok,
            "citations_correct": citations_ok,
            "finish_reason_length_failure": bool(completion and completion.finish_reason == "length"),
            "correct": answer_ok and citations_ok and parser_accepted and bool(
                completion and completion.finish_reason != "length"
            ),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))

    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "engram.q4-reader-dev-report.v1",
        "mode": mode,
        "prompt_file": str(PROMPT_FILE.relative_to(ROOT)),
        "prompt_file_sha256": hashlib.sha256(PROMPT_FILE.read_bytes()).hexdigest(),
        "settings": settings.server_identity(),
        "calls": rows,
        "summary": {
            "calls": len(rows),
            "parser_accepted": sum(bool(row["parser_accepted"]) for row in rows),
            "answer_correct": sum(bool(row["answer_correct"]) for row in rows),
            "citations_correct": sum(bool(row["citations_correct"]) for row in rows),
            "fully_correct": sum(bool(row["correct"]) for row in rows),
            "truncated": sum(bool(row["finish_reason_length_failure"]) for row in rows),
            "mean_latency_ms": round(
                sum(float(row["latency_ms"] or 0) for row in rows) / len(rows), 2
            ) if rows else None,
            "max_latency_ms": max((float(row["latency_ms"] or 0) for row in rows), default=0),
            "credits": 0,
        },
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "output": str(output)}, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("low", "off"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.base_url.startswith("http://127.0.0.1:"):
        raise SystemExit("Q4 development calls are restricted to a 127.0.0.1 server")
    report = run_mode(mode=args.mode, base_url=args.base_url, output=args.output)
    summary = report["summary"]
    return 0 if summary["parser_accepted"] == summary["calls"] and summary["truncated"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
