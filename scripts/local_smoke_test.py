"""Offline schema dry run and one-call-per-protocol local smoke test."""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import closing
from datetime import date
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Mapping
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engramOS.local_llama import (  # noqa: E402
    LocalModelSettings,
    LocalOpenAIClient,
    LocalQueryClassifierProvider,
    LocalReaderProvider,
    canned_classifier_response,
)
from engramOS.db import initialize_database, transaction  # noqa: E402
from engramOS.pipeline import answer_turn_baseline_query  # noqa: E402
from engramOS.query_classifier import (  # noqa: E402
    QueryClassificationOutputError,
    classify_query,
)
from engramOS.reader import READER_RESPONSE_SCHEMA_VERSION  # noqa: E402
from engramOS.retrieval import RetrievedTurn  # noqa: E402


MODEL_SHA256 = "82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38"
MODEL_FILE = "Muse-Glimmer-30B-UD-Q4_K_XL.gguf"
QUESTION_DATE = date(2026, 11, 3)
QUERY = "Which city is named in my personal notes?"


def _chat_response(text: str, *, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "model": MODEL_FILE,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text, "reasoning_content": ""},
            "finish_reason": finish_reason,
        }],
        "usage": {"prompt_tokens": 44, "completion_tokens": max(1, len(text.split()))},
    }


class CannedTransport:
    """In-memory responses; this object cannot open a socket."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = deque(responses)
        self.payloads: list[Mapping[str, object]] = []
        self.network_calls = 0

    def __call__(self, url: str, payload: Mapping[str, object] | None, timeout: float) -> Any:
        if url.endswith("/tokenize"):
            content = str((payload or {}).get("content", ""))
            return {"tokens": content.split()}
        self.payloads.append(dict(payload or {}))
        if not self.responses:
            raise AssertionError("canned transport has no response")
        return self.responses.popleft()


def _turn() -> RetrievedTurn:
    return RetrievedTurn(
        turn_id="smoke-pune",
        history_id="local-smoke",
        session_id="local-smoke-session",
        turn_idx=0,
        role="user",
        raw_content="My notes say that I live in Pune.",
        source_type="user_turn",
        event_time=1790000000.0,
        recorded_at=1790000000.0,
        dense_score=None,
        bm25_score=None,
        rrf_score=1.0,
        channels=("local_smoke",),
    )


def _settings(mode: str = "low", base_url: str = "http://127.0.0.1:8080") -> LocalModelSettings:
    return LocalModelSettings(
        base_url=base_url,
        model=MODEL_FILE,
        model_sha256=MODEL_SHA256,
        reasoning_mode=mode,
        gpu_layers=32,
    )


def dry_run(mode: str) -> int:
    reader_good = json.dumps({
        "schema_version": READER_RESPONSE_SCHEMA_VERSION,
        "answer": "The city in the note is Pune.",
        "citations": ["turn:smoke-pune"],
        "abstained": False,
    }, separators=(",", ":"))
    transport = CannedTransport([
        _chat_response(canned_classifier_response()),
        _chat_response(reader_good),
    ])
    client = LocalOpenAIClient(_settings(mode), json_request=transport)
    classifier = LocalQueryClassifierProvider(client)
    classification = classify_query(
        QUERY, question_date=QUESTION_DATE, classifier=classifier, timeout=10.0
    )
    reader = LocalReaderProvider(client)
    outcome = answer_turn_baseline_query(
        QUERY,
        "local-smoke",
        [_turn()],
        classification,
        reader,
        ordering="relevance",
        timeout=10.0,
    )
    if not classification.ready_for_retrieval or outcome.reason_code != "answered":
        raise AssertionError("canned successful outputs did not pass production parsers")

    print("dry_run_classifier_payload=" + json.dumps(transport.payloads[0], sort_keys=True))
    print("dry_run_reader_payload=" + json.dumps(transport.payloads[1], sort_keys=True))
    print("dry_run_classifier_request_hash=" + str(classifier.last_completion.request_hash))
    print("dry_run_reader_request_hash=" + str(reader.last_completion.request_hash))
    print(f"dry_run_reader_max_tokens={client.settings.reader_max_tokens}")
    print(f"dry_run_success_parsers=classifier:accepted,reader:{outcome.reason_code}")

    malformed_transport = CannedTransport([
        _chat_response("{malformed"),
        _chat_response("{malformed"),
    ])
    malformed_client = LocalOpenAIClient(_settings(mode), json_request=malformed_transport)
    malformed_classifier = LocalQueryClassifierProvider(malformed_client)
    try:
        classify_query(QUERY, question_date=QUESTION_DATE, classifier=malformed_classifier)
    except QueryClassificationOutputError:
        classifier_malformed = "rejected"
    else:
        classifier_malformed = "unexpectedly_accepted"
    malformed_reader = LocalReaderProvider(malformed_client)
    malformed_outcome = answer_turn_baseline_query(
        QUERY,
        "local-smoke",
        [_turn()],
        classification,
        malformed_reader,
        ordering="relevance",
        timeout=10.0,
    )
    print(
        "dry_run_malformed_parsers="
        f"classifier:{classifier_malformed},reader:{malformed_outcome.reason_code}"
    )
    print("dry_run_network_calls=0 (all completions came from injected in-memory transport)")
    if classifier_malformed != "rejected" or malformed_outcome.reason_code != "invalid_reader_output":
        return 1

    # A length finish is counted even if a provider happens to return parseable text.
    length_transport = CannedTransport([_chat_response(reader_good, finish_reason="length")])
    length_reader = LocalReaderProvider(
        LocalOpenAIClient(_settings(mode), json_request=length_transport)
    )
    length_outcome = answer_turn_baseline_query(
        QUERY,
        "local-smoke",
        [_turn()],
        classification,
        length_reader,
        ordering="relevance",
        timeout=10.0,
    )
    print(
        "dry_run_truncation="
        f"finish_reason={length_reader.last_completion.finish_reason},"
        f"parser={length_outcome.reason_code},counted_failures={length_reader.truncated_calls}"
    )
    return 0 if length_reader.truncated_calls == 1 else 1


def live_run(mode: str, base_url: str) -> int:
    database = ROOT / "eval" / "local-smoke.sqlite3"
    run_id = f"local-smoke-{uuid4().hex}"
    initialize_database(database)
    client = LocalOpenAIClient(
        _settings(mode, base_url),
        cache_database=str(database),
        phase_tag="local_smoke",
        run_id=run_id,
        cache_nonce=f"smoke:{run_id}",
    )
    classifier = LocalQueryClassifierProvider(client)
    classification_error: str | None = None
    classifier_ready = False
    try:
        classification = classify_query(
            QUERY, question_date=QUESTION_DATE, classifier=classifier, timeout=900.0
        )
        classifier_ready = classification.ready_for_retrieval
    except Exception as exc:
        classification_error = type(exc).__name__
        classification = None
    reader = LocalReaderProvider(client)
    if classification is None or not classification.ready_for_retrieval:
        # Keep the separately required Reader request even when classification
        # fails; the smoke test reports the classifier failure independently.
        from engramOS.query_classifier import QueryClassification
        classification = QueryClassification("classified", "current", QUESTION_DATE, "present_state")
    outcome = answer_turn_baseline_query(
        QUERY,
        "local-smoke",
        [_turn()],
        classification,
        reader,
        ordering="relevance",
        timeout=900.0,
    )
    classifier_call = classifier.last_completion
    reader_call = reader.last_completion
    if classifier_call is not None:
        _persist_smoke_call(
            database,
            run_id=run_id,
            question_id="classifier",
            settings=client.settings,
            completion=classifier_call,
            answer=classifier_call.text,
            correct=classification_error is None and classifier_ready,
        )
    if reader_call is not None:
        _persist_smoke_call(
            database,
            run_id=run_id,
            question_id="reader",
            settings=client.settings,
            completion=reader_call,
            answer=outcome.answer,
            correct=outcome.reason_code == "answered" and not reader_call.finish_reason == "length",
        )
    if classifier_call is None:
        print(f"classifier: failed before a completed response ({classification_error or 'unknown'})")
    else:
        print(json.dumps({
            "call": "classifier",
            "request_hash": classifier_call.request_hash,
            "cache_request_hash": classifier_call.cache_request_hash,
            "cache_hit": classifier_call.cache_hit,
            "response_model": classifier_call.response_model,
            "prompt_tokens": classifier_call.prompt_tokens,
            "completion_tokens": classifier_call.completion_tokens,
            "visible_tokens": classifier_call.visible_tokens,
            "latency_ms": round(classifier_call.latency_ms, 2),
            "finish_reason": classifier_call.finish_reason,
            "classification_error": classification_error,
            "credits": 0,
        }, sort_keys=True))
    if reader_call is None:
        print(json.dumps({"call": "reader", "parser": outcome.reason_code, "credits": 0}))
    else:
        print(json.dumps({
            "call": "reader",
            "request_hash": reader_call.request_hash,
            "cache_request_hash": reader_call.cache_request_hash,
            "cache_hit": reader_call.cache_hit,
            "response_model": reader_call.response_model,
            "prompt_tokens": reader_call.prompt_tokens,
            "completion_tokens": reader_call.completion_tokens,
            "visible_tokens": reader_call.visible_tokens,
            "latency_ms": round(reader_call.latency_ms, 2),
            "finish_reason": reader_call.finish_reason,
            "parser": outcome.reason_code,
            "answer": outcome.answer,
            "citations": [str(item) for item in outcome.citations],
            "truncation_count": reader.truncated_calls,
            "credits": 0,
        }, sort_keys=True))
    with closing(sqlite3.connect(database)) as conn:
        cache_rows, local_credits = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(cost_credits), 0) FROM llm_cache WHERE run_id=?",
            (run_id,),
        ).fetchone()
        eval_rows = conn.execute(
            "SELECT COUNT(*) FROM eval_runs WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        ledger_rows = conn.execute(
            "SELECT COUNT(*) FROM llm_call_ledger WHERE run_id=?", (run_id,)
        ).fetchone()[0]
    print(json.dumps({
        "call": "local_accounting",
        "run_id": run_id,
        "database": str(database),
        "llm_cache_rows": cache_rows,
        "eval_runs_rows": eval_rows,
        "ledger_rows": ledger_rows,
        "cost_credits": local_credits,
        "pricing_mode": "local",
    }, sort_keys=True))
    valid = bool(
        classifier_call and reader_call
        and classification_error is None
        and classifier_ready
        and outcome.reason_code == "answered"
        and reader_call.finish_reason != "length"
        and reader_call.response_model is not None
        and Path(reader_call.response_model).name == MODEL_FILE
    )
    return 0 if valid else 1


def _persist_smoke_call(
    database: Path,
    *,
    run_id: str,
    question_id: str,
    settings: LocalModelSettings,
    completion: Any,
    answer: str,
    correct: bool,
) -> None:
    judge_model = "none-local-smoke-parser-check"
    run_config = {
        "pricing_mode": "local",
        "cost_credits": 0,
        "cache_request_hash": completion.cache_request_hash,
        "cache_hit": completion.cache_hit,
        "server_settings": settings.server_identity(),
    }
    with transaction(str(database)) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO eval_runs (
               run_id, system, question_id, question_type, split, reader_model,
               judge_model, reader_prompt_hash, judge_prompt_hash, run_config_json,
               answer, correct, evidence_recall_at_5, prompt_tokens, out_tokens,
               latency_ms, created_at
               ) VALUES (?, 'local_smoke', ?, 'parser_smoke', 'dev', ?, ?, ?, ?, ?, ?, ?,
                         NULL, ?, ?, ?, strftime('%s','now'))""",
            (
                run_id, question_id, settings.model, judge_model,
                completion.request_hash,
                hashlib.sha256(judge_model.encode("utf-8")).hexdigest(),
                json.dumps(run_config, sort_keys=True, separators=(",", ":")),
                answer, int(correct), completion.prompt_tokens,
                completion.completion_tokens, int(completion.latency_ms),
            ),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--live", action="store_true")
    parser.add_argument("--mode", choices=("low", "off"), default="low")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    if args.live and not args.base_url.startswith("http://127.0.0.1:"):
        raise SystemExit("local live smoke is restricted to a 127.0.0.1 server")
    if args.dry_run:
        return dry_run(args.mode)
    return live_run(args.mode, args.base_url)


if __name__ == "__main__":
    raise SystemExit(main())
