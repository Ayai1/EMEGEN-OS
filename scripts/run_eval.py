"""Run the locked authored suite with four retrieval systems and one Reader.

The default offline mode uses deterministic fake providers through the normal
LLM cache and budget ledger. It measures wiring and retrieval exposure, not
model answer quality or extraction quality.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import sys
import time
from typing import Any, Mapping, Sequence
from urllib.request import Request, urlopen

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from engramOS.db import connection, initialize_database, transaction  # noqa: E402
from engramOS.embeddings import BGE_M3_DIMENSION, BgeM3EmbeddingProvider, index_history  # noqa: E402
from engramOS.gate import DefaultSignalEvaluator  # noqa: E402
from engramOS.ingestion import ingest_turns  # noqa: E402
from engramOS.llm_cache import (  # noqa: E402
    BudgetPolicy,
    DEFAULT_BUDGET_POLICY,
    CachedGeneration,
    GenerationRequest,
    GenerationResult,
    cached_generate,
)
from engramOS.pipeline import answer_query, answer_turn_baseline_query  # noqa: E402
from engramOS.query_classifier import (  # noqa: E402
    QUERY_CLASSIFICATION_SCHEMA_VERSION,
    QueryClassification,
    TemporalRange,
    classify_query,
)
from engramOS.reader import ABSTENTION_TEXT, ReaderPrompt  # noqa: E402
from engramOS.retrieval import hybrid_rag, recency  # noqa: E402
from engramOS.temporal import TemporalBoundary, as_of_timestamp  # noqa: E402
from engramOS.local_llama import (  # noqa: E402
    LocalModelSettings,
    LocalOpenAIClient,
    LocalQueryClassifierProvider,
    LocalReaderProvider,
)
from eval.loader import (  # noqa: E402
    EXPANDED_SUITE_PATH,
    FixtureFactExtractor,
    iter_history_turns,
    load_suite,
    suite_stats,
)

RUN_SCHEMA_VERSION = "engram.authored-eval-run.v1"
SYSTEMS = ("recency", "hybrid_rag", "hybrid_rag_date_filter", "engram")
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_READER_STOPWORDS = frozenset(
    "a about an and are as at be before did do does for from had have how i in is it my of on or over the this time to was what when where which who why will with now current use used keep keep".split()
)


class FakeEmbeddingProvider:
    """Stable local token-hash vectors kept separate from real BGE databases."""

    model_id = "fake-authored-suite-token-hash-v1"
    dimension = BGE_M3_DIMENSION
    device = "cpu"

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = _TOKEN_RE.findall(text.lower())
        for token in tokens:
            index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "little") % self.dimension
            vector[index] += 1.0
        if not tokens:
            vector[0] = 1.0
        return vector

    def encode_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._vector(text) for text in texts]) if texts else np.empty((0, self.dimension))


class _OfflineGenerationProvider:
    def generate(self, request: GenerationRequest, *, timeout: float) -> GenerationResult:
        started = time.perf_counter()
        if request.task_tag == "reader":
            text = _fake_reader_response(request.user_prompt)
        elif request.task_tag == "classifier":
            text = _fake_classifier_response(request.user_prompt)
        else:
            raise ValueError(f"unsupported offline task {request.task_tag!r}")
        elapsed = max(0, int((time.perf_counter() - started) * 1000))
        return GenerationResult(
            text=text,
            input_tokens=len(_TOKEN_RE.findall(request.system_prompt + " " + request.user_prompt)),
            output_tokens=len(_TOKEN_RE.findall(text)),
            actual_cost_credits=0.0,
            latency_ms=elapsed,
            observed_model=request.model,
        )


class _CachedOfflineBase:
    def __init__(self, database: str, run_id: str, *, model: str, task_tag: str) -> None:
        self.database = database
        self.run_id = run_id
        self.model = model
        self.task_tag = task_tag
        self.backend = _OfflineGenerationProvider()
        self.last_usage: dict[str, object] | None = None

    def _call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_format: Mapping[str, object] | None,
        max_tokens: int,
    ) -> str:
        request = GenerationRequest(
            provider_id="offline-fake",
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_output_tokens=max_tokens,
            response_format=response_format,
            generation_settings={"engine": "authored-rule-fake-v1", "seed": 0},
            task_tag=self.task_tag,
            phase_tag="authored_offline",
            run_id=self.run_id,
        )
        started = time.perf_counter()
        cached = cached_generate(
            self.database,
            request,
            self.backend,
            # A tiny positive reservation exercises the existing cap path;
            # successful offline fake calls record and spend exactly zero.
            reserved_credits=1e-9,
            policy=BudgetPolicy(total_cap=150.0, phase_caps={"authored_offline": 150.0}),
            timeout=15.0,
        )
        elapsed_ms = max(0, int((time.perf_counter() - started) * 1000))
        result = cached.result
        self.last_usage = {
            "request_hash": cached.request_key.split(":", 1)[1].split(":", 1)[0],
            "cache_hit": cached.cache_hit,
            "prompt_tokens": 0 if cached.cache_hit else result.input_tokens,
            "completion_tokens": 0 if cached.cache_hit else result.output_tokens,
            "latency_ms": elapsed_ms,
            "cost_credits": 0.0 if cached.cache_hit else result.actual_cost_credits,
            "model": self.model,
        }
        return result.text


class CachedFakeReader(_CachedOfflineBase):
    model_name = "offline-rule-reader-v1"

    def __init__(self, database: str, run_id: str) -> None:
        super().__init__(database, run_id, model=self.model_name, task_tag="reader")
        self.last_payload: dict[str, Any] | None = None
        self.prompt_hash = hashlib.sha256(b"no-reader-call").hexdigest()

    def reset(self) -> None:
        self.last_usage = None
        self.last_payload = None

    def generate(self, prompt: ReaderPrompt, *, timeout: float) -> str:
        self.prompt_hash = hashlib.sha256(
            (prompt.system_prompt + "\0" + prompt.user_payload).encode("utf-8")
        ).hexdigest()
        self.last_payload = json.loads(prompt.user_payload)
        response_format = json.loads(prompt.response_schema_json)
        return self._call(
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_payload,
            response_format={"type": "json_schema", "json_schema": response_format},
            max_tokens=1024,
        )


class CachedFakeClassifier(_CachedOfflineBase):
    model_name = "offline-rule-classifier-v1"

    def __init__(self, database: str, run_id: str) -> None:
        super().__init__(database, run_id, model=self.model_name, task_tag="classifier")
        self.provider_queries: set[str] = set()
        self.prompt_hash = hashlib.sha256(b"no-classifier-call").hexdigest()

    def reset(self) -> None:
        self.last_usage = None

    def classify(self, query: str, *, question_date: date, timeout: float) -> str:
        self.provider_queries.add(query)
        user_prompt = json.dumps(
            {"query": query, "question_date": question_date.isoformat()},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.prompt_hash = hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()
        return self._call(
            system_prompt="Offline deterministic classifier for the authored-suite wiring check.",
            user_prompt=user_prompt,
            response_format={"type": "json_schema", "name": "engram_query_classification"},
            max_tokens=256,
        )


def _fake_classifier_response(user_prompt: str) -> str:
    payload = json.loads(user_prompt)
    query = str(payload["query"])
    date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", query)
    if date_match:
        value = date_match.group(0)
        return json.dumps({
            "schema_version": QUERY_CLASSIFICATION_SCHEMA_VERSION,
            "status": "classified",
            "view": "as_of",
            "reason_code": "provider_classification",
            "as_of": {"value": value, "inclusive": not bool(re.search(r"\bbefore\b", query, re.I))},
            "history_range": None,
        })
    return json.dumps({
        "schema_version": QUERY_CLASSIFICATION_SCHEMA_VERSION,
        "status": "classified",
        "view": "current",
        "reason_code": "provider_classification",
        "as_of": None,
        "history_range": None,
    })


def _fake_reader_response(user_prompt: str) -> str:
    payload = json.loads(user_prompt)
    query = str(payload["query"])
    terms = _query_terms(query)
    evidence = payload.get("evidence", [])
    fact_matches: list[tuple[int, float, dict[str, Any]]] = []
    turn_matches: list[tuple[int, float, dict[str, Any]]] = []
    timeline_query = bool(re.search(r"\b(change|changed|over time|timeline)\b", query, re.I))
    for item in evidence:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") or {}
        if item.get("evidence_type") == "fact":
            attr = set(_TOKEN_RE.findall(str(metadata.get("attribute") or "").lower()))
            entity = set(_TOKEN_RE.findall(str(metadata.get("entity") or "").lower()))
            key = set(_TOKEN_RE.findall(str(item.get("text") or "").lower()))
            score = 5 * len(terms & attr) + 3 * len(terms & entity) + len(terms & key)
            if score:
                fact_matches.append((score, float(metadata.get("valid_from") or 0.0), item))
        else:
            text_terms = set(_TOKEN_RE.findall(str(item.get("text") or "").lower()))
            score = len(terms & text_terms)
            if score:
                turn_matches.append((score, float((metadata or {}).get("event_time") or 0.0), item))

    if fact_matches:
        fact_matches.sort(key=lambda value: (value[1], str(value[2].get("evidence_id"))))
        if timeline_query:
            best_score = max(value[0] for value in fact_matches)
            selected = [value[2] for value in fact_matches if value[0] >= best_score]
            values = [str((item.get("metadata") or {}).get("value") or "") for item in selected]
            answer = ", ".join(value for value in values if value)
            selected = selected[:10]
        else:
            best = max(fact_matches, key=lambda value: (value[0], value[1]))[2]
            answer = str((best.get("metadata") or {}).get("value") or best.get("text") or "")
            selected = [best]
        selected = [item for item in selected if item.get("evidence_id")]
    elif turn_matches:
        best = max(turn_matches, key=lambda value: (value[0], value[1]))[2]
        answer = str(best.get("text") or "")
        selected = [best]
    else:
        return json.dumps({
            "schema_version": "engram.reader-response.v1",
            "answer": ABSTENTION_TEXT,
            "citations": [],
            "abstained": True,
        })
    citations = list(dict.fromkeys(str(item["evidence_id"]) for item in selected))
    if not answer or not citations:
        return json.dumps({
            "schema_version": "engram.reader-response.v1",
            "answer": ABSTENTION_TEXT,
            "citations": [],
            "abstained": True,
        })
    return json.dumps({
        "schema_version": "engram.reader-response.v1",
        "answer": answer,
        "citations": citations,
        "abstained": False,
    })


def _query_terms(query: str) -> set[str]:
    return {
        token for token in _TOKEN_RE.findall(query.lower())
        if token not in _READER_STOPWORDS and not token.isdigit()
    }


def _embedding_provider(kind: str):
    if kind == "fake":
        return FakeEmbeddingProvider()
    if kind == "bge":
        return BgeM3EmbeddingProvider(device="cpu")
    raise ValueError(f"unsupported embedding provider {kind!r}")


def _ensure_embedding_identity(database: str, expected_model: str) -> None:
    with connection(database) as conn:
        models = {
            str(row[0]) for row in conn.execute("SELECT DISTINCT model FROM embeddings")
        }
    if models and models != {expected_model}:
        raise ValueError(
            "database already contains a different embedding identity; use a separate DB "
            f"(found={sorted(models)!r}, requested={expected_model!r})"
        )


def run_evaluation(
    *,
    database: str,
    run_id: str,
    suite_path: str | Path,
    embedding_kind: str = "fake",
    mode: str = "offline",
    local_settings: LocalModelSettings | None = None,
    question_ids: set[str] | None = None,
    progress_log: str | Path | None = None,
    provider_timeout: float = 300.0,
    stop_on_provider_failure: bool = False,
    resume_from_run_id: str | None = None,
    systems: Sequence[str] | None = None,
    context_version: str = "v2",
    strict_untrusted: bool = False,
    expect_suite_sha256: str | None = None,
) -> dict[str, Any]:
    suite_bytes = Path(suite_path).read_bytes()
    suite_sha256 = hashlib.sha256(suite_bytes).hexdigest()
    if expect_suite_sha256 is not None and (
        not re.fullmatch(r"[0-9a-fA-F]{64}", expect_suite_sha256)
        or expect_suite_sha256.lower() != suite_sha256
    ):
        raise ValueError(
            f"suite SHA-256 mismatch: expected {expect_suite_sha256!r}, got {suite_sha256}"
        )
    suite = load_suite(suite_path)
    if hashlib.sha256(Path(suite_path).read_bytes()).hexdigest() != suite_sha256:
        raise ValueError("suite changed while it was being loaded")
    if context_version not in {"v2", "v3"}:
        raise ValueError("context_version must be 'v2' or 'v3'")
    if type(strict_untrusted) is not bool:
        raise TypeError("strict_untrusted must be a bool")
    stats = suite_stats(suite)
    selected_systems = tuple(systems or SYSTEMS)
    if not selected_systems or len(set(selected_systems)) != len(selected_systems) or any(
        system not in SYSTEMS for system in selected_systems
    ):
        raise ValueError("systems must be a non-empty unique subset of the evaluation systems")
    heldout_suite = suite.get("suite_id") == "engram.authored-security-posthoc-v2-heldout"
    valid_main_size = (
        5 <= stats.histories <= 30 and 50 <= stats.turns <= 100 and 20 <= stats.questions <= 60
    )
    valid_heldout_size = (
        heldout_suite and 5 <= stats.histories <= 30 and 8 <= stats.questions <= 12
        and 8 <= stats.turns <= 30
    )
    if not (valid_main_size or valid_heldout_size):
        raise ValueError("authored suite size is outside the locked benchmark bounds")
    initialize_database(database)
    embedding = _embedding_provider(embedding_kind)
    _ensure_embedding_identity(database, embedding.model_id)
    evaluator = DefaultSignalEvaluator()
    resume_info: dict[str, Any] = {"reused_results": 0, "superseded_failures": []}
    if resume_from_run_id:
        resume_info = _seed_resume_results(
            database,
            source_run_id=resume_from_run_id,
            target_run_id=run_id,
            previous_timeout_seconds=180.0,
            new_timeout_seconds=provider_timeout,
        )

    # Each turn is gated, extracted, and applied before the next turn enters
    # that history. This is required for earlier pinned constraints to exist.
    for history in suite["histories"]:
        extractor = FixtureFactExtractor(history)
        for turn in iter_history_turns(history):
            day_recorded = as_of_timestamp(turn.session_date) + 86_400.0
            ingest_turns(
                database,
                (turn,),
                evaluator=evaluator,
                extractor=extractor,
                recorded_at=day_recorded,
                quarantine_policy="isolate",
                strict_untrusted=strict_untrusted,
            )
        index_history(database, history["history_id"], embedding, gated=False)

    if mode == "offline":
        reader = CachedFakeReader(database, run_id)
        classifier = CachedFakeClassifier(database, run_id)
    elif mode == "local":
        if local_settings is None:
            raise ValueError("local mode requires explicit local model settings")
        local_budget = BudgetPolicy(
            total_cap=DEFAULT_BUDGET_POLICY.total_cap,
            phase_caps=DEFAULT_BUDGET_POLICY.phase_caps,
            reservation_lease_seconds=600.0,
        )
        client = LocalOpenAIClient(
            local_settings,
            cache_database=database,
            phase_tag="locked_test",
            run_id=run_id,
            budget_policy=local_budget,
        )
        reader = LocalReaderProvider(client)
        classifier = LocalQueryClassifierProvider(client)
    else:
        raise ValueError("mode must be 'offline' or 'local'")
    # Held-out v2 was locked without a per-file date; it uses the fixed suite
    # evaluation date. The main suite continues to take the date from its file.
    question_date = date.fromisoformat(str(suite.get("question_date", "2026-11-03")))
    failures: list[str] = []
    selected_questions = [
        question for question in suite["questions"]
        if question_ids is None or str(question["question_id"]) in question_ids
    ]
    if question_ids is not None:
        missing = question_ids - {str(question["question_id"]) for question in selected_questions}
        if missing:
            raise ValueError(f"unknown question IDs: {sorted(missing)!r}")
    _append_progress(progress_log, {
        "event": "run_started",
        "run_id": run_id,
        "mode": mode,
        "selected_questions": len(selected_questions),
        "systems": list(selected_systems),
        "provider_timeout_seconds": provider_timeout,
        "resume_from_run_id": resume_from_run_id,
        "reused_persisted_results": resume_info["reused_results"],
        "started_at": time.time(),
    })
    if resume_from_run_id:
        _append_progress(progress_log, {
            "event": "resume_seeded",
            "run_id": run_id,
            "resume_from_run_id": resume_from_run_id,
            "reused_results": resume_info["reused_results"],
            "superseded_failures": resume_info["superseded_failures"],
            "at": time.time(),
        })
    stop_reason: str | None = None
    consecutive_provider_failures = 0
    for question_number, question in enumerate(selected_questions, start=1):
        for system in selected_systems:
            question_id = str(question["question_id"])
            if _question_result_exists(database, run_id, system, question_id):
                _append_progress(progress_log, {
                    "event": "result_resumed",
                    "run_id": run_id,
                    "question_number": question_number,
                    "question_id": question_id,
                    "system": system,
                    "cumulative_failures": len(failures),
                    "at": time.time(),
                })
                continue
            system_started = time.perf_counter()
            reader.reset()
            classifier.reset()
            query = str(question["query"])
            history_id = str(question["history_id"])
            classification: QueryClassification | None = None
            baseline_exception: Exception | None = None
            if system == "engram":
                outcome = answer_query(
                    database,
                    history_id,
                    query,
                    embedding,
                    reader,
                    question_date=question_date,
                    classifier=classifier,
                    timeout=provider_timeout,
                    context_version=context_version,  # type: ignore[arg-type]
                )
                classification = outcome.classification
                reader_outcome = outcome.reader
                if reader_outcome is None:
                    from engramOS.reader import ReaderOutcome
                    reader_outcome = ReaderOutcome(ABSTENTION_TEXT, (), True, outcome.reason_code)
            else:
                try:
                    classification = classify_query(
                        query, question_date=question_date, classifier=classifier,
                        timeout=provider_timeout,
                    )
                    event_range = None
                    if system == "hybrid_rag_date_filter" and classification.as_of is not None:
                        event_range = TemporalRange(end=classification.as_of)
                    if classification.ready_for_retrieval:
                        if system == "recency":
                            retrieval = recency(
                                database, history_id, query, embedding, top_k=10,
                                candidate_limit=50, gated=False,
                            )
                        else:
                            retrieval = hybrid_rag(
                                database, history_id, query, embedding, top_k=10,
                                candidate_limit=50, gated=False,
                                event_time_range=event_range if system == "hybrid_rag_date_filter" else None,
                            )
                        ordering = "reverse_chronological" if system == "recency" else "relevance"
                        reader_outcome = answer_turn_baseline_query(
                            query, history_id, retrieval.turns, classification, reader,
                            ordering=ordering,
                            timeout=provider_timeout,
                            context_version=context_version,  # type: ignore[arg-type]
                        )
                    else:
                        reader_outcome = answer_turn_baseline_query(
                            query, history_id, (), classification, reader,
                            ordering="relevance",
                            timeout=provider_timeout,
                            context_version=context_version,  # type: ignore[arg-type]
                        )
                except Exception as exc:
                    baseline_exception = exc
                    from engramOS.reader import ReaderOutcome
                    reader_outcome = ReaderOutcome(ABSTENTION_TEXT, (), True, "retrieval_failed")

            payload = reader.last_payload or {}
            evidence_ids, source_turn_ids = _reached_sources(payload)
            retrieved_ids = list(evidence_ids)
            retrieved_ids.extend(
                f"turn:{turn_id}" for turn_id in source_turn_ids
                if f"turn:{turn_id}" not in retrieved_ids
            )
            cited_ids = [str(value) for value in reader_outcome.citations]
            answer = reader_outcome.answer
            expected_label = str(question["expected_label"])
            correct = _grade_question(answer, reader_outcome.abstained, question)
            related_turns = [
                str(question[key]) for key in ("attack_turn_id", "benign_turn_id")
                if question.get(key)
            ]
            gate_outcome = _gate_outcomes(database, related_turns)

            usages = [usage for usage in (classifier.last_usage, reader.last_usage) if usage]
            prompt_tokens = sum(int(usage["prompt_tokens"]) for usage in usages)
            completion_tokens = sum(int(usage["completion_tokens"]) for usage in usages)
            provider_latency_ms = sum(int(usage["latency_ms"]) for usage in usages)
            system_latency_ms = max(0, int((time.perf_counter() - system_started) * 1000))
            # Persist the time spent by this run invocation. Cached completions
            # retain their original model latency in provider metadata, which
            # is recorded separately and must not inflate cache-hit wall time.
            latency_ms = system_latency_ms
            cost_credits = sum(float(usage["cost_credits"]) for usage in usages)
            request_hash = (
                str(reader.last_usage["request_hash"]) if reader.last_usage
                else str(classifier.last_usage["request_hash"]) if classifier.last_usage
                else None
            )
            reader_prompt_hash = reader.prompt_hash if reader.last_payload else hashlib.sha256(b"no-reader-prompt").hexdigest()
            classifier_called = query in classifier.provider_queries
            finish_reasons = {
                "classifier": list((classifier.last_usage or {}).get("finish_reasons", [])),
                "reader": list((reader.last_usage or {}).get("finish_reasons", [])),
            }
            provider_failure_types = _provider_failure_types(
                classifier_called=classifier_called,
                classifier_error=getattr(classifier, "last_error", None),
                classifier_finish_reasons=finish_reasons["classifier"],
                reader_finish_reasons=finish_reasons["reader"],
                classification=classification,
                reader_reason_code=reader_outcome.reason_code,
                baseline_exception=baseline_exception,
            )
            provider_failure_fields = _provider_failure_metadata(provider_failure_types)
            call_failures = list(provider_failure_types)
            if not call_failures and reader_outcome.reason_code in {
                "retrieval_failed", "final_revalidation_failed", "classification_failed",
            }:
                call_failures.append(reader_outcome.reason_code)
            failures.extend(
                f"{system}:{question_id}:{failure}" for failure in call_failures
            )
            config = {
                "schema_version": RUN_SCHEMA_VERSION,
                "suite_sha256": suite_sha256,
                "question_date": question_date.isoformat(),
                "embedding_model": embedding.model_id,
                "reader_model": reader.model,
                "classifier_model": classifier.model,
                "judge": "rule-based-expected-substring-v1",
                "ordering": "reverse_chronological" if system == "recency" else "relevance",
                "gated": system == "engram",
                "event_time_prefilter_before_rrf": system == "hybrid_rag_date_filter",
                "fixture_extraction": True,
                "bridge_tau": evaluator.bridge_tau,
                "context_version": context_version,
                "strict_untrusted": strict_untrusted,
                "temperature": 0.0 if mode == "offline" else (local_settings.temperature if local_settings else 0.0),
                "mode": mode,
                "credits_mode": "local" if mode == "local" else "offline_fake",
                "provider_timeout_seconds": provider_timeout,
                "classifier_provider_called": classifier_called,
                "classifier_prompt_hash": classifier.prompt_hash,
                "classifier_attempt_count": int((classifier.last_usage or {}).get("attempt_count", 0)),
                "reader_attempt_count": int((reader.last_usage or {}).get("attempt_count", 0)),
                "classifier_prompt_tokens": int((classifier.last_usage or {}).get("prompt_tokens", 0)),
                "classifier_completion_tokens": int((classifier.last_usage or {}).get("completion_tokens", 0)),
                "classifier_visible_tokens": int((classifier.last_usage or {}).get("visible_tokens", 0)),
                "finish_reasons": finish_reasons,
                "call_failures": call_failures,
                **provider_failure_fields,
                "reader_reason_code": reader_outcome.reason_code,
                "system_latency_ms": system_latency_ms,
                "provider_reported_latency_ms": provider_latency_ms,
                "reader_cache_hit": bool((reader.last_usage or {}).get("cache_hit", False)),
                "classifier_cache_hit": bool((classifier.last_usage or {}).get("cache_hit", False)),
            }
            if local_settings is not None:
                config["local_model_settings"] = local_settings.server_identity()
                config["model_file"] = local_settings.model
                config["model_sha256"] = local_settings.model_sha256
                config["llama_cpp_build"] = "b12058-3f8af15a7"
                config["gpu_cpu_layers"] = {
                    "gpu": local_settings.gpu_layers,
                    "cpu": 53 - local_settings.gpu_layers,
                    "total": 53,
                }
            if resume_from_run_id:
                config["continuation_from_run_id"] = resume_from_run_id
                config["resumed_persisted_result"] = False
                original_failure = next((
                    item for item in resume_info["superseded_failures"]
                    if item["system"] == system and item["question_id"] == question_id
                ), None)
                if original_failure is not None:
                    config["rerun_of"] = {
                        "run_id": resume_from_run_id,
                        "failure_types": original_failure["provider_failure_types"],
                        "previous_timeout_seconds": original_failure["provider_timeout_seconds"],
                        "new_timeout_seconds": provider_timeout,
                        "superseded": True,
                    }
            _persist_question(
                database,
                run_id=run_id,
                system=system,
                question=question,
                answer=answer,
                abstained=reader_outcome.abstained,
                cited_ids=cited_ids,
                retrieved_ids=retrieved_ids,
                correct=correct,
                gate_outcome=gate_outcome,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                cost_credits=cost_credits,
                cache_request_hash=request_hash,
                reader_model=reader.model,
                judge_model="rule-based-expected-substring-v1",
                reader_prompt_hash=reader_prompt_hash,
                run_config=config,
            )
            if provider_failure_types:
                consecutive_provider_failures += 1
            else:
                consecutive_provider_failures = 0
            server_healthy = (
                _local_server_is_healthy(local_settings.base_url)
                if mode == "local" and local_settings is not None and provider_failure_types
                else True
            )
            server_dead = mode == "local" and bool(provider_failure_types) and not server_healthy
            stop_reason = _provider_failure_stop_reason(
                provider_failure_types=provider_failure_types,
                consecutive_failures=consecutive_provider_failures,
                stop_on_provider_failure=stop_on_provider_failure,
                local_mode=mode == "local",
                server_healthy=server_healthy,
            )
            _append_progress(progress_log, {
                "event": "question_system_complete",
                "run_id": run_id,
                "question_number": question_number,
                "question_id": question_id,
                "system": system,
                "latency_ms": system_latency_ms,
                "reader_reported_latency_ms": int((reader.last_usage or {}).get("latency_ms", 0)),
                "classifier_reported_latency_ms": int((classifier.last_usage or {}).get("latency_ms", 0)),
                "reader_cache_hit": bool((reader.last_usage or {}).get("cache_hit", False)),
                "classifier_cache_hit": bool((classifier.last_usage or {}).get("cache_hit", False)),
                "cache_request_hash": request_hash,
                "cumulative_failures": len(failures),
                "consecutive_provider_failures": consecutive_provider_failures,
                "server_healthy": server_healthy if provider_failure_types else None,
                "call_failures": call_failures,
                **provider_failure_fields,
                "at": time.time(),
            })
            if stop_reason is not None:
                _append_progress(progress_log, {
                    "event": "run_stopped_after_provider_failure",
                    "run_id": run_id,
                    "question_number": question_number,
                    "question_id": question_id,
                    "system": system,
                    "provider_failure_types": provider_failure_types,
                    "cumulative_failures": len(failures),
                    "consecutive_provider_failures": consecutive_provider_failures,
                    "server_healthy": server_healthy,
                    "stop_reason": stop_reason,
                    "at": time.time(),
                })
                break
        if stop_reason is not None:
            break

    summary = _build_persisted_summary(
        database=database,
        run_id=run_id,
        suite=suite,
        stats=stats,
        mode=mode,
        systems=selected_systems,
        selected_question_ids={str(item["question_id"]) for item in selected_questions},
        run_failures=failures,
    )
    summary["stopped_early"] = stop_reason is not None
    summary["stop_reason"] = stop_reason
    _append_progress(progress_log, {
        "event": "run_stopped" if stop_reason is not None else "run_complete",
        "run_id": run_id,
        "persisted_question_system_rows": summary["persisted_question_results"],
        "failures": summary["failures"],
        "stop_reason": stop_reason,
        "completed_at": time.time(),
    })
    return summary


def _append_progress(path: str | Path | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()


def _question_result_exists(database: str, run_id: str, system: str, question_id: str) -> bool:
    with connection(database) as conn:
        return conn.execute(
            "SELECT 1 FROM eval_question_results WHERE run_id=? AND system=? AND question_id=?",
            (run_id, system, question_id),
        ).fetchone() is not None


def _seed_resume_results(
    database: str,
    *,
    source_run_id: str,
    target_run_id: str,
    previous_timeout_seconds: float,
    new_timeout_seconds: float,
) -> dict[str, Any]:
    """Copy successful rows into a labeled continuation and supersede failures.

    The original result and failure rows are never deleted or overwritten.
    Successful rows are copied so the normal runner's resume check skips all
    provider calls already completed in the source run.
    """
    if source_run_id == target_run_id:
        raise ValueError("resume target must use a new run id to preserve source failures")
    with connection(database) as conn:
        rows = conn.execute(
            """SELECT q.*, e.question_type, e.split, e.reader_model, e.judge_model,
                      e.reader_prompt_hash, e.judge_prompt_hash, e.run_config_json,
                      e.answer, e.correct AS eval_correct, e.evidence_recall_at_5,
                      e.prompt_tokens AS eval_prompt_tokens, e.out_tokens,
                      e.latency_ms AS eval_latency_ms, e.created_at
               FROM eval_question_results AS q
               JOIN eval_runs AS e
                 ON e.run_id=q.run_id AND e.system=q.system AND e.question_id=q.question_id
               WHERE q.run_id=? ORDER BY q.question_id, q.system""",
            (source_run_id,),
        ).fetchall()
    if not rows:
        raise ValueError(f"no persisted question results found for resume run {source_run_id!r}")

    reused = 0
    superseded: list[dict[str, Any]] = []
    with transaction(database) as conn:
        for row in rows:
            config = json.loads(str(row["run_config_json"] or "{}"))
            failure_types = list(config.get("provider_failure_types", []))
            if bool(config.get("provider_failure", False)) or failure_types:
                config.update({
                    "provider_timeout_seconds": previous_timeout_seconds,
                    "superseded": True,
                    "superseded_by_run_id": target_run_id,
                    "superseded_by_timeout_seconds": new_timeout_seconds,
                    "superseded_at": time.time(),
                })
                conn.execute(
                    "UPDATE eval_runs SET run_config_json=? WHERE run_id=? AND system=? AND question_id=?",
                    (json.dumps(config, sort_keys=True), source_run_id,
                     str(row["system"]), str(row["question_id"])),
                )
                superseded.append({
                    "system": str(row["system"]),
                    "question_id": str(row["question_id"]),
                    "provider_failure_types": failure_types,
                    "provider_timeout_seconds": previous_timeout_seconds,
                })
                continue

            copied_config = dict(config)
            copied_config.update({
                "continuation_from_run_id": source_run_id,
                "resumed_persisted_result": True,
                "provider_timeout_seconds": previous_timeout_seconds,
            })
            conn.execute(
                """INSERT OR IGNORE INTO eval_runs (
                       run_id, system, question_id, question_type, split, reader_model,
                       judge_model, reader_prompt_hash, judge_prompt_hash, run_config_json,
                       answer, correct, evidence_recall_at_5, prompt_tokens, out_tokens,
                       latency_ms, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (target_run_id, str(row["system"]), str(row["question_id"]),
                 str(row["question_type"]), str(row["split"]), str(row["reader_model"]),
                 str(row["judge_model"]), str(row["reader_prompt_hash"]),
                 str(row["judge_prompt_hash"]), json.dumps(copied_config, sort_keys=True),
                 str(row["answer"]), row["eval_correct"], row["evidence_recall_at_5"],
                 int(row["eval_prompt_tokens"]), int(row["out_tokens"]),
                 row["eval_latency_ms"], float(row["created_at"])),
            )
            conn.execute(
                """INSERT OR IGNORE INTO eval_question_results (
                       run_id, system, question_id, axis, answer_text, abstained,
                       cited_ids_json, retrieved_ids_json, expected_answer,
                       expected_label, correct, gate_outcome_json, prompt_tokens,
                       completion_tokens, latency_ms, cost_credits, cache_request_hash
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (target_run_id, str(row["system"]), str(row["question_id"]),
                 str(row["axis"]), str(row["answer_text"]), int(row["abstained"]),
                 str(row["cited_ids_json"]), str(row["retrieved_ids_json"]),
                 row["expected_answer"], str(row["expected_label"]), int(row["correct"]),
                 str(row["gate_outcome_json"]), int(row["prompt_tokens"]),
                 int(row["completion_tokens"]), int(row["latency_ms"]),
                 float(row["cost_credits"]), row["cache_request_hash"]),
            )
            reused += 1
    return {"reused_results": reused, "superseded_failures": superseded}


def _local_server_is_healthy(base_url: str, *, timeout: float = 5.0) -> bool:
    """Check the loopback health endpoint after a failed local provider call."""
    try:
        request = Request(base_url.rstrip("/") + "/health", method="GET")
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return response.status == 200 and isinstance(payload, dict) and payload.get("status") == "ok"
    except Exception:
        return False


def _provider_failure_stop_reason(
    *,
    provider_failure_types: Sequence[str],
    consecutive_failures: int,
    stop_on_provider_failure: bool,
    local_mode: bool,
    server_healthy: bool,
) -> str | None:
    if not provider_failure_types:
        return None
    if local_mode and not server_healthy:
        return "server_unavailable"
    if stop_on_provider_failure:
        return "provider_failure"
    if consecutive_failures >= 3:
        return "three_consecutive_provider_failures"
    return None


def _provider_failure_types(
    *,
    classifier_called: bool,
    classifier_error: BaseException | None,
    classifier_finish_reasons: Sequence[str],
    reader_finish_reasons: Sequence[str],
    classification: QueryClassification | None,
    reader_reason_code: str,
    baseline_exception: BaseException | None,
) -> list[str]:
    """Classify provider/runtime failures separately from answer correctness."""
    failures: list[str] = []
    if "length" in classifier_finish_reasons:
        failures.append("classifier_finish_reason_length")
    elif classifier_called and classifier_error is not None:
        failures.append(
            "classifier_timeout" if _is_timeout_error(classifier_error)
            else "classifier_provider_error"
        )
    elif classifier_called and classification is None and (
        baseline_exception is not None or reader_reason_code == "classification_failed"
    ):
        failures.append("classifier_parse_failure")

    if "length" in reader_finish_reasons:
        failures.append("reader_finish_reason_length")
    elif reader_reason_code == "provider_timeout":
        failures.append("reader_timeout")
    elif reader_reason_code in {"provider_error", "provider_capacity_exhausted"}:
        failures.append("reader_provider_error")
    elif reader_reason_code in {"invalid_reader_output", "invalid_citation"}:
        failures.append("reader_parse_failure")
    return list(dict.fromkeys(failures))


def _provider_failure_metadata(failure_types: Sequence[str]) -> dict[str, Any]:
    unique = list(dict.fromkeys(failure_types))
    return {"provider_failure": bool(unique), "provider_failure_types": unique}


def _wrong_answer_count(rows: Sequence[Mapping[str, Any]]) -> int:
    """Count incorrect completed answers, excluding provider failures."""
    return sum(
        not bool(row["correct"]) and not bool(row.get("provider_failure", False))
        for row in rows
    )


def _is_timeout_error(error: BaseException) -> bool:
    pending: list[object] = [error]
    visited: set[int] = set()
    while pending:
        value = pending.pop()
        if id(value) in visited:
            continue
        visited.add(id(value))
        if isinstance(value, TimeoutError):
            return True
        if isinstance(value, BaseException):
            if "timeout" in type(value).__name__.casefold() or "timed out" in str(value).casefold():
                return True
            pending.extend((value.__cause__, value.__context__, getattr(value, "reason", None)))
    return False


def _build_persisted_summary(
    *,
    database: str,
    run_id: str,
    suite: Mapping[str, Any],
    stats: Any,
    mode: str,
    systems: Sequence[str],
    selected_question_ids: set[str],
    run_failures: list[str],
) -> dict[str, Any]:
    questions = {
        str(question["question_id"]): question for question in suite["questions"]
        if str(question["question_id"]) in selected_question_ids
    }
    with connection(database) as conn:
        rows = conn.execute(
            """SELECT q.*, e.run_config_json FROM eval_question_results AS q
               LEFT JOIN eval_runs AS e ON e.run_id=q.run_id AND e.system=q.system
                 AND e.question_id=q.question_id
               WHERE q.run_id=? ORDER BY q.question_id, q.system""",
            (run_id,),
        ).fetchall()
        cache_calls = int(conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0])

    per_system: dict[str, list[dict[str, Any]]] = {system: [] for system in systems}
    all_records: list[dict[str, Any]] = []
    failures = list(run_failures)
    classifier_queries: set[str] = set()
    for stored in rows:
        question_id = str(stored["question_id"])
        question = questions.get(question_id)
        if question is None:
            continue
        system = str(stored["system"])
        retrieved_ids = json.loads(str(stored["retrieved_ids_json"]))
        gate_outcome = json.loads(str(stored["gate_outcome_json"]))
        run_config = json.loads(str(stored["run_config_json"] or "{}"))
        if run_config.get("classifier_provider_called"):
            classifier_queries.add(str(question["query"]))
        source_turn_ids = {
            value.removeprefix("turn:") for value in retrieved_ids
            if isinstance(value, str) and value.startswith("turn:")
        }
        attack_turn_id = str(question["attack_turn_id"]) if question.get("attack_turn_id") else None
        benign_turn_id = str(question["benign_turn_id"]) if question.get("benign_turn_id") else None
        attack_claim = str(suite.get("attack_claims", {}).get(question_id, "")) or None
        leaked = _has_quarantined_source(database, source_turn_ids)
        row = {
            "question_id": question_id,
            "axis": str(stored["axis"]),
            "correct": bool(stored["correct"]),
            "abstained": bool(stored["abstained"]),
            "expected_label": str(stored["expected_label"]),
            "attack_turn_id": attack_turn_id,
            "attack_claim": attack_claim,
            "attack_reached_reader": bool(attack_turn_id and attack_turn_id in source_turn_ids),
            "attack_claim_adopted": (
                _claim_appears_in_answer(str(stored["answer_text"]), attack_claim)
                if attack_claim else None
            ),
            "benign_turn_id": benign_turn_id,
            "benign_quarantined": bool(
                benign_turn_id
                and gate_outcome.get(benign_turn_id, {}).get("decision") == "quarantine"
            ),
            "quarantine_leak": leaked,
            "answer": str(stored["answer_text"]),
            "retrieved_ids": retrieved_ids,
            "gate_outcome": gate_outcome,
            "latency_ms": int(stored["latency_ms"]),
            "cost_credits": float(stored["cost_credits"]),
            "failure_count": len(run_config.get("call_failures", [])),
            "call_failures": list(run_config.get("call_failures", [])),
            "provider_failure": bool(run_config.get("provider_failure", False)),
            "provider_failure_types": list(run_config.get("provider_failure_types", [])),
            "system": system,
        }
        for item in row["call_failures"]:
            failures.append(f"{system}:{question_id}:{item}")
        per_system[system].append(row)
        all_records.append(row)

    axes = {}
    overall = {}
    abstention_precision = {}
    attack_exposure = {}
    attack_adoption = {}
    latency = {}
    credits = {}
    system_failures = {}
    wrong_answer_counts = {}
    provider_failures_by_type: dict[str, int] = defaultdict(int)
    provider_failures_by_question: dict[str, int] = defaultdict(int)
    for system, system_rows in per_system.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in system_rows:
            groups[row["axis"]].append(row)
        axes[system] = {
            axis: _rate(sum(int(row["correct"]) for row in values), len(values))
            for axis, values in sorted(groups.items())
        }
        overall[system] = _rate(
            sum(int(row["correct"]) for row in system_rows), len(system_rows)
        )
        abstained = [row for row in system_rows if row["abstained"]]
        abstention_precision[system] = _rate(
            sum(row["expected_label"] == "abstain" for row in abstained), len(abstained)
        )
        attacks = [row for row in system_rows if row["attack_turn_id"]]
        attack_exposure[system] = _rate(
            sum(bool(row["attack_reached_reader"]) for row in attacks), len(attacks)
        )
        adoption = [row for row in attacks if row["attack_claim"]]
        attack_adoption[system] = _rate(
            sum(bool(row["attack_claim_adopted"]) for row in adoption), len(adoption)
        )
        latency[system] = {
            "count": len(system_rows),
            "mean_ms": round(sum(int(row["latency_ms"]) for row in system_rows) / len(system_rows), 3)
            if system_rows else None,
            "max_ms": max((int(row["latency_ms"]) for row in system_rows), default=None),
        }
        credits[system] = round(sum(float(row["cost_credits"]) for row in system_rows), 8)
        system_failures[system] = sum(bool(row["provider_failure"]) for row in system_rows)
        wrong_answer_counts[system] = _wrong_answer_count(system_rows)
        for row in system_rows:
            for failure_type in row["provider_failure_types"]:
                provider_failures_by_type[failure_type] += 1
            if row["provider_failure"]:
                provider_failures_by_question[row["question_id"]] += 1

    benign_ids = sorted({
        str(question[key]) for question in questions.values() for key in ("benign_turn_id",)
        if question.get(key)
    })
    gate_decisions = {
        turn_id: next((
            (row["gate_outcome"].get(turn_id) or {}).get("decision")
            for row in all_records if turn_id in row["gate_outcome"]
        ), None)
        for turn_id in benign_ids
    }
    quarantined_benign = sum(value == "quarantine" for value in gate_decisions.values())
    total_question_system_rows = int(conn_execute_scalar(
        database, "SELECT COUNT(*) FROM eval_question_results WHERE run_id=?", (run_id,)
    ))
    return {
        "run_id": run_id,
        "mode": mode,
        "suite": {
            "histories": stats.histories,
            "turns": stats.turns,
            "questions": len(questions),
            "systems": list(systems),
            "axes": dict(stats.axes),
        },
        "per_axis_accuracy": axes,
        "overall_accuracy": overall,
        "abstention_precision": abstention_precision,
        "attack_reader_exposure_rate": attack_exposure,
        "attack_claim_adoption_rate": attack_adoption,
        "benign_false_positive_rate": _rate(quarantined_benign, len(benign_ids)),
        "benign_gate_decisions": gate_decisions,
        "quarantine_leak_count": {
            system: sum(bool(row["quarantine_leak"]) for row in system_rows)
            for system, system_rows in per_system.items()
        },
        "per_question_system_latency_ms": latency,
        "provider_failure_result_count": sum(system_failures.values()),
        "provider_failure_count_by_system": system_failures,
        "provider_failure_count_by_type": dict(sorted(provider_failures_by_type.items())),
        "provider_failure_count_by_question": dict(sorted(provider_failures_by_question.items())),
        "wrong_answer_count_by_system": wrong_answer_counts,
        "total_credits": {
            "pricing_mode": "local" if mode == "local" else "offline_fake",
            "total": round(sum(credits.values()), 8),
            "by_system": credits,
        },
        "cache_rows": cache_calls,
        "persisted_question_results": total_question_system_rows,
        "classifier_queries": sorted(classifier_queries),
        "failures": sorted(set(failures)),
        "confidence_interval": "exact two-sided Clopper-Pearson 95% binomial interval",
        "question_results": all_records,
    }


def conn_execute_scalar(database: str, sql: str, params: tuple[Any, ...]) -> int:
    with connection(database) as conn:
        return int(conn.execute(sql, params).fetchone()[0])


def _persist_question(
    database: str,
    *,
    run_id: str,
    system: str,
    question: Mapping[str, Any],
    answer: str,
    abstained: bool,
    cited_ids: list[str],
    retrieved_ids: list[str],
    correct: bool,
    gate_outcome: Mapping[str, Any],
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: int,
    cost_credits: float,
    cache_request_hash: str | None,
    reader_model: str,
    judge_model: str,
    reader_prompt_hash: str,
    run_config: Mapping[str, Any],
) -> None:
    judge_hash = hashlib.sha256(judge_model.encode("utf-8")).hexdigest()
    expected_answer = question.get("expected_answer")
    split = "dev"
    with transaction(database) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO eval_runs (
               run_id, system, question_id, question_type, split, reader_model,
               judge_model, reader_prompt_hash, judge_prompt_hash, run_config_json,
               answer, correct, evidence_recall_at_5, prompt_tokens, out_tokens,
               latency_ms, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)""",
            (
                run_id, system, question["question_id"], question["axis"], split,
                reader_model, judge_model, reader_prompt_hash, judge_hash,
                json.dumps(run_config, sort_keys=True, separators=(",", ":")),
                answer, int(correct), prompt_tokens, completion_tokens,
                latency_ms, time.time(),
            ),
        )
        conn.execute(
            """INSERT OR REPLACE INTO eval_question_results (
               run_id, system, question_id, axis, answer_text, abstained,
               cited_ids_json, retrieved_ids_json, expected_answer, expected_label,
               correct, gate_outcome_json, prompt_tokens, completion_tokens,
               latency_ms, cost_credits, cache_request_hash
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, system, question["question_id"], question["axis"], answer,
                int(abstained), json.dumps(cited_ids), json.dumps(retrieved_ids),
                expected_answer, question["expected_label"], int(correct),
                json.dumps(gate_outcome, sort_keys=True), prompt_tokens,
                completion_tokens, latency_ms, cost_credits, cache_request_hash,
            ),
        )


def _reached_sources(payload: Mapping[str, Any]) -> tuple[list[str], set[str]]:
    evidence_ids: list[str] = []
    source_turns: set[str] = set()
    for item in payload.get("evidence", []):
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("evidence_id") or "")
        if evidence_id:
            evidence_ids.append(evidence_id)
            kind, separator, value = evidence_id.partition(":")
            if separator and kind == "turn":
                source_turns.add(value)
        for source in item.get("provenance", []):
            if isinstance(source, dict) and source.get("turn_id"):
                source_turns.add(str(source["turn_id"]))
    return evidence_ids, source_turns


def _gate_outcomes(database: str, turn_ids: Sequence[str]) -> dict[str, Any]:
    if not turn_ids:
        return {}
    with connection(database) as conn:
        rows = conn.execute(
            """SELECT e.turn_id, e.signal_bridge, e.signal_policy, e.signal_source,
                      e.signals_fired, e.decision
               FROM gate_events AS e WHERE e.turn_id IN ("""
            + ",".join("?" for _ in turn_ids)
            + ") ORDER BY e.event_id",
            tuple(turn_ids),
        ).fetchall()
    return {
        str(row["turn_id"]): {
            "signal_bridge": bool(row["signal_bridge"]),
            "signal_policy": bool(row["signal_policy"]),
            "signal_source": bool(row["signal_source"]),
            "signals_fired": int(row["signals_fired"]),
            "decision": str(row["decision"]),
        }
        for row in rows
    }


def _has_quarantined_source(database: str, turn_ids: set[str]) -> bool:
    if not turn_ids:
        return False
    with connection(database) as conn:
        for turn_id in turn_ids:
            if conn.execute(
                "SELECT 1 FROM gate_events WHERE turn_id=? AND decision='quarantine' LIMIT 1",
                (turn_id,),
            ).fetchone():
                return True
    return False


def _grade(answer: str, abstained: bool, expected_label: str, expected_answer: object) -> bool:
    if expected_label == "abstain":
        return bool(abstained)
    if abstained or not isinstance(expected_answer, str):
        return False
    return " ".join(expected_answer.casefold().split()) in " ".join(answer.casefold().split())


def _grade_question(answer: str, abstained: bool, question: Mapping[str, Any]) -> bool:
    """Grade authored exact answers and explicitly predeclared aliases."""
    expected_label = str(question["expected_label"])
    if expected_label == "abstain":
        return bool(abstained)
    if abstained:
        return False
    accepted = question.get("accepted_answers")
    if not isinstance(accepted, list) or not accepted:
        return _grade(answer, abstained, expected_label, question.get("expected_answer"))
    normalized_answer = " ".join(answer.casefold().split())
    return any(
        isinstance(alias, str) and " ".join(alias.casefold().split()) in normalized_answer
        for alias in accepted
    )


def _claim_appears_in_answer(answer: str, claim: str) -> bool:
    """Token-contiguous claim check; this is a lexical proxy, not semantic judging."""
    answer_tokens = _TOKEN_RE.findall(answer.casefold())
    claim_tokens = _TOKEN_RE.findall(claim.casefold())
    if not claim_tokens or len(claim_tokens) > len(answer_tokens):
        return False
    width = len(claim_tokens)
    return any(
        answer_tokens[index:index + width] == claim_tokens
        for index in range(len(answer_tokens) - width + 1)
    )


def _build_summary(
    *,
    stats: Any,
    records: list[dict[str, Any]],
    per_system: Mapping[str, list[dict[str, Any]]],
    exposure_leaks: Mapping[str, int],
    attack_exposure_trials: Mapping[str, list[bool]],
    attack_adoption_trials: Mapping[str, list[bool]],
    failures: list[str],
    classifier_queries: list[str],
    database: str,
    run_id: str,
) -> dict[str, Any]:
    axes: dict[str, dict[str, Any]] = {}
    abstention_precision: dict[str, Any] = {}
    attack_exposure: dict[str, Any] = {}
    attack_adoption: dict[str, Any] = {}
    latency: dict[str, Any] = {}
    credits: dict[str, float] = {}
    for system, rows in per_system.items():
        axis_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            axis_rows[row["axis"]].append(row)
        axes[system] = {
            axis: _rate(sum(int(row["correct"]) for row in values), len(values))
            for axis, values in sorted(axis_rows.items())
        }
        abstained_rows = [row for row in rows if row["abstained"]]
        true_abstentions = sum(row["expected_label"] == "abstain" for row in abstained_rows)
        abstention_precision[system] = _rate(true_abstentions, len(abstained_rows))
        exposure_values = attack_exposure_trials[system]
        adoption_values = attack_adoption_trials[system]
        attack_exposure[system] = _rate(sum(exposure_values), len(exposure_values))
        attack_adoption[system] = _rate(sum(adoption_values), len(adoption_values))
        latency[system] = {
            "count": len(rows),
            "mean_ms": round(sum(int(row["latency_ms"]) for row in rows) / len(rows), 3) if rows else None,
            "max_ms": max((int(row["latency_ms"]) for row in rows), default=None),
        }
        credits[system] = round(sum(float(row["cost_credits"]) for row in rows), 8)

    benign_ids = sorted({
        str(row["benign_turn_id"])
        for rows in per_system.values() for row in rows if row.get("benign_turn_id")
    })
    with connection(database) as conn:
        quarantined_benign = sum(
            bool(conn.execute(
                "SELECT 1 FROM gate_events WHERE turn_id=? AND decision='quarantine' LIMIT 1",
                (turn_id,),
            ).fetchone())
            for turn_id in benign_ids
        )
        gate_decisions = {
            str(row["turn_id"]): str(row["decision"])
            for row in conn.execute("SELECT turn_id, decision FROM gate_events WHERE event_id IN (SELECT MIN(event_id) FROM gate_events GROUP BY turn_id)")
        }
        cache_calls = int(conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0])
        result_rows = int(conn.execute("SELECT COUNT(*) FROM eval_question_results WHERE run_id=?", (run_id,)).fetchone()[0])
    failures.extend(
        f"benign_gate_unexpected_quarantine:{turn_id}"
        for turn_id in benign_ids if gate_decisions.get(turn_id) == "quarantine"
    )
    return {
        "run_id": run_id,
        "suite": {
            "histories": stats.histories,
            "turns": stats.turns,
            "questions": stats.questions,
            "systems": list(SYSTEMS),
            "axes": dict(stats.axes),
        },
        "offline_mode": "deterministic fake providers through llm_cache; not model-quality evidence",
        "per_axis_accuracy": axes,
        "abstention_precision": abstention_precision,
        "attack_reader_exposure_rate": attack_exposure,
        "attack_claim_adoption_rate": attack_adoption,
        "benign_false_positive_rate": _rate(quarantined_benign, len(benign_ids)),
        "benign_gate_decisions": {turn_id: gate_decisions.get(turn_id) for turn_id in benign_ids},
        "quarantine_leak_count": dict(exposure_leaks),
        "per_question_latency_ms": latency,
        "total_credits": {"pricing_mode": "offline_fake", "total": round(sum(credits.values()), 8), "by_system": credits},
        "cache_rows": cache_calls,
        "persisted_question_results": result_rows,
        "classifier_queries": classifier_queries,
        "failures": sorted(set(failures)),
        "confidence_interval": "exact two-sided Clopper-Pearson 95% binomial interval",
        "question_results": records,
    }


def _rate(successes: int, trials: int) -> dict[str, Any]:
    if trials == 0:
        return {"successes": 0, "trials": 0, "rate": None, "ci95": None}
    alpha = 0.05
    lower = 0.0 if successes == 0 else _binomial_tail_root(successes, trials, alpha / 2, upper_tail=True)
    upper = 1.0 if successes == trials else _binomial_tail_root(successes, trials, alpha / 2, upper_tail=False)
    return {
        "successes": successes,
        "trials": trials,
        "rate": successes / trials,
        "ci95": [lower, upper],
    }


def _binomial_tail_root(successes: int, trials: int, target: float, *, upper_tail: bool) -> float:
    low, high = 0.0, 1.0
    for _ in range(80):
        p = (low + high) / 2.0
        tail = _binomial_tail(successes, trials, p, upper_tail=upper_tail)
        if upper_tail:
            if tail > target:
                high = p
            else:
                low = p
        else:
            if tail < target:
                high = p
            else:
                low = p
    return (low + high) / 2.0


def _binomial_tail(successes: int, trials: int, p: float, *, upper_tail: bool) -> float:
    first, last = (successes, trials) if upper_tail else (0, successes)
    return sum(
        math.comb(trials, index) * p**index * (1.0 - p) ** (trials - index)
        for index in range(first, last + 1)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "local"), default="offline")
    parser.add_argument("--embedding", choices=("fake", "bge"), default=None)
    parser.add_argument("--database", help="SQLite file; fake and BGE embeddings must use separate files")
    parser.add_argument("--run-id", default="authored-offline-v1")
    parser.add_argument("--suite", default=str(EXPANDED_SUITE_PATH))
    parser.add_argument("--question-ids", help="comma-separated question IDs (for a canary)")
    parser.add_argument("--systems", help="comma-separated systems; default runs all four")
    parser.add_argument("--max-questions", type=int, help="run the first N suite questions")
    parser.add_argument("--progress-log", help="append one JSON line after every question/system result")
    parser.add_argument("--provider-timeout", type=float, default=300.0)
    parser.add_argument("--context-version", choices=("v2", "v3"), default="v2")
    parser.add_argument("--strict-untrusted", action="store_true")
    parser.add_argument(
        "--expect-suite-sha256",
        help="abort unless the loaded suite file has this SHA-256 digest",
    )
    parser.add_argument(
        "--stop-on-provider-failure",
        action="store_true",
        help="stop after persisting the first provider failure (used by the canary)",
    )
    parser.add_argument(
        "--resume-from-run-id",
        help="seed completed, non-failure rows from a prior partial run and rerun its failed rows",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--model-file", default="Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
    )
    parser.add_argument(
        "--model-sha256",
        default="82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38",
    )
    args = parser.parse_args()
    embedding_kind = args.embedding or ("bge" if args.mode == "local" else "fake")
    database = args.database or str(
        PROJECT_ROOT / "eval" / (
            "offline-fake-v2.sqlite3" if embedding_kind == "fake"
            else "local-q4-phase3-classifier1024.sqlite3" if args.mode == "local"
            else "offline-bge-v2.sqlite3"
        )
    )
    suite = load_suite(args.suite)
    selected = None
    if args.question_ids:
        selected = {value.strip() for value in args.question_ids.split(",") if value.strip()}
    elif args.max_questions is not None:
        if args.max_questions <= 0:
            parser.error("--max-questions must be positive")
        selected = {
            str(question["question_id"])
            for question in suite["questions"][:args.max_questions]
        }
    selected_systems = None
    if args.systems:
        selected_systems = tuple(value.strip() for value in args.systems.split(",") if value.strip())
    local_settings = None
    if args.mode == "local":
        local_settings = LocalModelSettings(
            base_url=args.base_url,
            model=args.model_file,
            model_sha256=args.model_sha256,
            quantization="UD-Q4_K_XL",
            context_size=8192,
            gpu_layers=32,
            parallel_slots=1,
            kv_cache_type="q8_0",
            flash_attention=True,
            chat_template="gguf_builtin",
            reasoning_mode="low",
            reasoning_effort="low",
            reasoning_budget=512,
            reasoning_strength="low",
            temperature=0.0,
            top_p=0.95,
            top_k=40,
            seed=42,
            reader_max_tokens=2048,
            classifier_max_tokens=1024,
        )
    summary = run_evaluation(
        database=database,
        run_id=args.run_id,
        suite_path=args.suite,
        embedding_kind=embedding_kind,
        mode=args.mode,
        local_settings=local_settings,
        question_ids=selected,
        progress_log=args.progress_log,
        provider_timeout=args.provider_timeout,
        stop_on_provider_failure=args.stop_on_provider_failure,
        resume_from_run_id=args.resume_from_run_id,
        systems=selected_systems,
        context_version=args.context_version,
        strict_untrusted=args.strict_untrusted,
        expect_suite_sha256=args.expect_suite_sha256,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
