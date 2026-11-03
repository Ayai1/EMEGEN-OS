"""Measure local llama.cpp prompt processing and generation without hosted calls."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "logs"
MODEL_ID = "Muse-Glimmer-30B-UD-Q4_K_XL.gguf"
SEED = 42
TEMPERATURE = 0.0
TOP_P = 0.95
TOP_K = 40
MAX_TOKENS = 1024


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 30) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {} if data is None else {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else None


def token_count(base_url: str, text: str) -> int:
    response = request_json(base_url + "/tokenize", {"content": text, "add_special": False})
    tokens = response.get("tokens") if isinstance(response, dict) else None
    if isinstance(tokens, list):
        return len(tokens)
    raise RuntimeError(f"Unexpected /tokenize response: {response!r}")


def build_reader_prompt(base_url: str, target_tokens: int = 2800) -> tuple[list[dict[str, str]], int]:
    vocab = [
        "cedar", "maple", "birch", "juniper", "river", "harbor", "studio", "archive",
        "orchid", "compass", "notebook", "lantern", "marble", "canvas", "terrace", "copper",
        "weekday", "shipment", "schedule", "workshop", "draft", "catalog", "recipe", "garden",
        "conference", "station", "museum", "kitchen", "project", "restoration", "label", "sample",
    ]
    rows: list[str] = []
    index = 0
    while True:
        a, b, c, d = (vocab[(index * n + n * n) % len(vocab)] for n in (1, 3, 7, 11))
        rows.append(
            f"Evidence {index:04d}: On {2021 + index % 6}-0{1 + index % 9}-0{1 + index % 8}, "
            f"the {a} {b} record described a {c} {d} update. The note is historical context; "
            "prefer later dated evidence for current state and do not infer unsupported details."
        )
        body = "\n".join(rows)
        count = token_count(base_url, body)
        if count >= target_tokens:
            break
        index += 1
        if index > 1000:
            raise RuntimeError("Could not build target-size Reader prompt")

    messages = [
        {
            "role": "system",
            "content": "Answer from the supplied evidence. Return one JSON object with keys answer and cited_ids.",
        },
        {
            "role": "user",
            "content": body + '\n\nQuestion: Which dated evidence items mention a garden or restoration? '
            'Return a concise answer and an array of evidence IDs. If unsupported, abstain.',
        },
    ]
    return messages, count


def build_classifier_prompt() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Classify the query. Return JSON only."},
        {
            "role": "user",
            "content": 'Query: What was true about the project before 2024-03-01? '
            'Return {"intent":"...","as_of":"...","needs_retrieval":true}.',
        },
    ]


def timing_from_slots(base_url: str) -> dict[str, Any] | None:
    try:
        data = request_json(base_url + "/slots")
    except Exception:
        return None
    if isinstance(data, list):
        for slot in data:
            timings = slot.get("timings") if isinstance(slot, dict) else None
            if isinstance(timings, dict) and ("prompt_per_second" in timings or "predicted_per_second" in timings):
                return timings
    return None


def stream_completion(base_url: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_ms: float | None = None
    visible_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] = {}
    extra_timings: dict[str, Any] | None = None
    finish_reason: str | None = None
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            if isinstance(chunk.get("timings"), dict):
                extra_timings = chunk["timings"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            content = delta.get("content")
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if content:
                visible_parts.append(content)
            if reasoning:
                reasoning_parts.append(reasoning)
            if first_token_ms is None and (content or reasoning):
                first_token_ms = (time.perf_counter() - started) * 1000
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    elapsed_ms = (time.perf_counter() - started) * 1000
    visible_text = "".join(visible_parts)
    return {
        "elapsed_ms": elapsed_ms,
        "ttft_ms": first_token_ms,
        "visible_text": visible_text,
        "reasoning_text_chars": sum(map(len, reasoning_parts)),
        "usage": usage,
        "finish_reason": finish_reason,
        "timings": extra_timings or timing_from_slots(base_url),
    }


def run_case(base_url: str, case: str, messages: list[dict[str, str]], run_number: int) -> dict[str, Any]:
    result = stream_completion(base_url, messages)
    usage = result["usage"]
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    visible_tokens = token_count(base_url, result["visible_text"]) if result["visible_text"] else 0
    timings = result.get("timings") or {}
    return {
        "case": case,
        "run": run_number,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "visible_answer_tokens": visible_tokens,
        "hidden_or_nonvisible_completion_tokens_estimate": max(0, completion_tokens - visible_tokens),
        "ttft_ms": round(result["ttft_ms"], 2) if result["ttft_ms"] is not None else None,
        "prompt_processing_tokens_per_second": timings.get("prompt_per_second"),
        "generation_tokens_per_second": timings.get("predicted_per_second"),
        "server_prompt_ms": timings.get("prompt_ms"),
        "server_generation_ms": timings.get("predicted_ms"),
        "end_to_end_ms": round(result["elapsed_ms"], 2),
        "reasoning_text_chars": result["reasoning_text_chars"],
        "finish_reason": result["finish_reason"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", default="logs/local-bench-q4-20261103.json")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    health = request_json(base_url + "/health")
    print(f"health={health}")
    reader_messages, reader_body_tokens = build_reader_prompt(base_url)
    print(f"reader_evidence_tokens={reader_body_tokens} (target 2,000-4,000)")
    cases = [("reader", reader_messages), ("classifier", build_classifier_prompt())]
    results: list[dict[str, Any]] = []
    for case, messages in cases:
        for run in range(1, args.runs + 1):
            row = run_case(base_url, case, messages, run)
            results.append(row)
            print(json.dumps(row, sort_keys=True))
    REPORT_DIR.mkdir(exist_ok=True)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_file": MODEL_ID,
        "base_url": base_url,
        "sampling": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "seed": SEED, "max_tokens": MAX_TOKENS},
        "reasoning": {"strength": "low", "budget_tokens": 512},
        "reader_body_tokens": reader_body_tokens,
        "runs": results,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report={output}")
    for case, _ in cases:
        rows = [row for row in results if row["case"] == case]
        end_to_end = [row["end_to_end_ms"] for row in rows]
        ttft = [row["ttft_ms"] for row in rows if row["ttft_ms"] is not None]
        prompt_tps = [row["prompt_processing_tokens_per_second"] for row in rows if row["prompt_processing_tokens_per_second"] is not None]
        generation_tps = [row["generation_tokens_per_second"] for row in rows if row["generation_tokens_per_second"] is not None]
        print(json.dumps({
            "summary": case,
            "runs": len(rows),
            "mean_end_to_end_ms": round(statistics.mean(end_to_end), 2),
            "max_end_to_end_ms": round(max(end_to_end), 2),
            "mean_ttft_ms": round(statistics.mean(ttft), 2) if ttft else None,
            "mean_prompt_processing_tokens_per_second": round(statistics.mean(prompt_tps), 2) if prompt_tps else None,
            "mean_generation_tokens_per_second": round(statistics.mean(generation_tps), 2) if generation_tps else None,
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
