"""Count production Reader contexts for the locked suite without generation."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from engramOS.pipeline import answer_query, answer_turn_baseline_query  # noqa: E402
from engramOS.query_classifier import TemporalRange, classify_query  # noqa: E402
from engramOS.reader import READER_RESPONSE_SCHEMA_VERSION, ReaderPrompt  # noqa: E402
from engramOS.retrieval import hybrid_rag, recency  # noqa: E402
from eval.loader import EXPANDED_SUITE_PATH, load_suite  # noqa: E402
from run_eval import FakeEmbeddingProvider, SYSTEMS, _fake_classifier_response  # noqa: E402


class RuleClassifier:
    """Same deterministic fake response function used by the offline runner."""

    def classify(self, query: str, *, question_date: date, timeout: float) -> str:
        return _fake_classifier_response(json.dumps({
            "query": query,
            "question_date": question_date.isoformat(),
        }))


class CaptureReader:
    def __init__(self) -> None:
        self.prompt: ReaderPrompt | None = None

    def generate(self, prompt: ReaderPrompt, *, timeout: float) -> str:
        self.prompt = prompt
        payload = json.loads(prompt.user_payload)
        evidence = payload.get("evidence", [])
        citations = [evidence[0]["evidence_id"]] if evidence else []
        return json.dumps({
            "schema_version": READER_RESPONSE_SCHEMA_VERSION,
            "answer": "Captured for prompt token counting.",
            "citations": citations,
            "abstained": False,
        })


def _post_json(url: str, payload: Mapping[str, object]) -> Any:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=30.0) as response:
        return json.loads(response.read().decode("utf-8"))


def _token_count(base_url: str, prompt: ReaderPrompt) -> int:
    messages = [
        {"role": "system", "content": prompt.system_prompt},
        {"role": "user", "content": prompt.user_payload},
    ]
    rendered = _post_json(base_url.rstrip("/") + "/apply-template", {
        "messages": messages,
        "add_generation_prompt": True,
    })
    full_prompt = rendered.get("prompt") if isinstance(rendered, dict) else None
    if type(full_prompt) is not str:
        raise RuntimeError("llama-server /apply-template returned no prompt text")
    tokens = _post_json(base_url.rstrip("/") + "/tokenize", {
        "content": full_prompt,
        "add_special": False,
    })
    values = tokens.get("tokens") if isinstance(tokens, dict) else None
    if type(values) is not list:
        raise RuntimeError("llama-server /tokenize returned no token array")
    return len(values)


def _summarize(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values) if values else 0,
        "median": statistics.median(values) if values else 0,
        "mean": round(statistics.mean(values), 2) if values else 0,
        "max": max(values) if values else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="eval/offline-fake-v2.sqlite3")
    parser.add_argument("--suite", default=str(EXPANDED_SUITE_PATH))
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--output", default="logs/locked-suite-reader-prompt-tokens.json")
    args = parser.parse_args()
    if not args.base_url.startswith("http://127.0.0.1:"):
        raise SystemExit("prompt tokenization is restricted to a 127.0.0.1 llama-server")
    suite = load_suite(args.suite)
    embedding = FakeEmbeddingProvider()
    classifier = RuleClassifier()
    question_date = date.fromisoformat(suite["question_date"])
    measurements: dict[str, list[int]] = defaultdict(list)
    per_question: list[dict[str, object]] = []

    for question in suite["questions"]:
        query = str(question["query"])
        history_id = str(question["history_id"])
        for system in SYSTEMS:
            reader = CaptureReader()
            classification = None
            if system == "engram":
                answer_query(
                    args.database,
                    history_id,
                    query,
                    embedding,
                    reader,
                    question_date=question_date,
                    classifier=classifier,
                )
                classification = None
            else:
                classification = classify_query(
                    query, question_date=question_date, classifier=classifier
                )
                if classification.ready_for_retrieval:
                    event_range = None
                    if system == "hybrid_rag_date_filter" and classification.as_of is not None:
                        event_range = TemporalRange(end=classification.as_of)
                    if system == "recency":
                        retrieval = recency(
                            args.database, history_id, query, embedding, top_k=10,
                            candidate_limit=50, gated=False,
                        )
                    else:
                        retrieval = hybrid_rag(
                            args.database, history_id, query, embedding, top_k=10,
                            candidate_limit=50, gated=False,
                            event_time_range=event_range if system == "hybrid_rag_date_filter" else None,
                        )
                    ordering = "reverse_chronological" if system == "recency" else "relevance"
                    answer_turn_baseline_query(
                        query, history_id, retrieval.turns, classification, reader,
                        ordering=ordering,
                    )
            prompt = reader.prompt
            if prompt is None:
                per_question.append({
                    "question_id": question["question_id"], "system": system,
                    "reader_call": False, "prompt_tokens": None,
                })
                continue
            count = _token_count(args.base_url, prompt)
            measurements[system].append(count)
            per_question.append({
                "question_id": question["question_id"], "axis": question["axis"],
                "system": system, "reader_call": True, "prompt_tokens": count,
            })

    suite_bytes = Path(args.suite).read_bytes()
    report = {
        "suite_path": str(Path(args.suite)),
        "suite_sha256": hashlib.sha256(suite_bytes).hexdigest(),
        "suite_questions": len(suite["questions"]),
        "systems": list(SYSTEMS),
        "generation_calls": 0,
        "tokenization_mode": "local llama-server /apply-template + /tokenize; no completion endpoint",
        "summary_by_system": {name: _summarize(values) for name, values in measurements.items()},
        "rows": per_question,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "suite_sha256": report["suite_sha256"],
        "suite_questions": report["suite_questions"],
        "reader_contexts_by_system": report["summary_by_system"],
        "total_reader_contexts": sum(len(values) for values in measurements.values()),
        "generation_calls": 0,
        "output": str(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
