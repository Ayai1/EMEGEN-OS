"""Small OpenAI-compatible transport for the zero-credit local llama-server.

This module deliberately has no hosted-provider fallback. A server failure is
returned to the caller as an error, and request identity includes both the
complete chat payload and the output-affecting local server/model settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import time
from typing import Any, Callable, Mapping
from urllib.request import Request, urlopen

from .llm_cache import (
    BudgetPolicy,
    CachedGeneration,
    DEFAULT_BUDGET_POLICY,
    GenerationRequest,
    GenerationResult,
    cached_generate,
)
from .meta_adapters import (
    QUERY_CLASSIFIER_SYSTEM_PROMPT,
    _QUERY_CLASSIFICATION_RESPONSE_SCHEMA,
)
from .query_classifier import (
    QUERY_CLASSIFICATION_SCHEMA_VERSION,
    QueryClassificationOutputError,
    QueryClassificationProvider,
    classify_query,
)
from .reader import (
    ReaderPrompt,
    ReaderProvider,
    _parse_response as _parse_reader_response,
)
from .reranking import CandidateId


LOCAL_READER_MAX_TOKENS = 2048


@dataclass(frozen=True, slots=True)
class LocalModelSettings:
    """Output identity and sampling settings.

    Callers may set ``reader_max_tokens``, ``classifier_max_tokens``, and
    ``reasoning_budget`` per run. Their defaults remain 2048, 1024, and 512.
    """

    base_url: str
    model: str
    model_sha256: str
    quantization: str = "UD-Q4_K_XL"
    context_size: int = 8192
    gpu_layers: int = 32
    parallel_slots: int = 1
    kv_cache_type: str = "q8_0"
    flash_attention: bool = True
    chat_template: str = "gguf_builtin"
    reasoning_mode: str = "low"
    reasoning_effort: str = "low"
    reasoning_budget: int = 512
    reasoning_strength: str = "low"
    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int = 40
    seed: int = 42
    reader_max_tokens: int = LOCAL_READER_MAX_TOKENS
    classifier_max_tokens: int = 1024

    def __post_init__(self) -> None:
        if not self.base_url.startswith("http://127.0.0.1:"):
            raise ValueError("local llama-server must bind to an HTTP 127.0.0.1 URL")
        if self.reasoning_mode not in {"low", "off"}:
            raise ValueError("reasoning_mode must be 'low' or 'off'")
        if self.context_size < 1 or self.reader_max_tokens < 1 or self.classifier_max_tokens < 1:
            raise ValueError("context and output token limits must be positive")
        if len(self.model_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in self.model_sha256):
            raise ValueError("model_sha256 must be a lowercase SHA-256 digest")

    def server_identity(self) -> dict[str, object]:
        return {
            "base_url": self.base_url.rstrip("/"),
            "model": self.model,
            "model_sha256": self.model_sha256,
            "quantization": self.quantization,
            "context_size": self.context_size,
            "gpu_layers": self.gpu_layers,
            "parallel_slots": self.parallel_slots,
            "kv_cache_type": self.kv_cache_type,
            "flash_attention": self.flash_attention,
            "chat_template": self.chat_template,
            "reasoning_mode": self.reasoning_mode,
            "reasoning_effort": self.reasoning_effort,
            "reasoning_budget": self.reasoning_budget,
            "reasoning_strength": self.reasoning_strength,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "reader_max_tokens": self.reader_max_tokens,
            "classifier_max_tokens": self.classifier_max_tokens,
        }

    def request_hash(self, payload: Mapping[str, object]) -> str:
        identity = {"server": self.server_identity(), "request": payload}
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LocalCompletion:
    text: str
    reasoning_text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    visible_tokens: int
    latency_ms: float
    request_hash: str
    response_model: str | None
    cache_hit: bool = False
    cache_request_hash: str | None = None


JsonRequest = Callable[[str, Mapping[str, object] | None, float], Any]


def _http_json(url: str, payload: Mapping[str, object] | None, timeout: float) -> Any:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {} if data is None else {"Content-Type": "application/json"}
    request = Request(url, data=data, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else None


class LocalOpenAIClient:
    """Strict non-streaming client; network errors propagate without fallback."""

    def __init__(
        self,
        settings: LocalModelSettings,
        *,
        json_request: JsonRequest = _http_json,
        cache_database: str | None = None,
        phase_tag: str = "local",
        run_id: str = "unspecified",
        cache_nonce: str = "",
        budget_policy: BudgetPolicy = DEFAULT_BUDGET_POLICY,
    ) -> None:
        self.settings = settings
        self._json_request = json_request
        self.cache_database = cache_database
        self.phase_tag = phase_tag
        self.run_id = run_id
        self.cache_nonce = cache_nonce
        self.budget_policy = budget_policy
        self._backend_completion: LocalCompletion | None = None

    def complete(
        self,
        payload: Mapping[str, object],
        *,
        timeout: float,
        task_tag: str = "local",
        cache_nonce: str | None = None,
    ) -> LocalCompletion:
        request_hash = self.settings.request_hash(payload)
        if self.cache_database is None:
            return self._complete_uncached(payload, timeout=timeout, request_hash=request_hash)

        messages = payload.get("messages")
        if type(messages) is not list or len(messages) != 2:
            raise ValueError("local cached generation requires exactly system and user messages")
        if any(type(message) is not dict or type(message.get("content")) is not str for message in messages):
            raise ValueError("local cached generation requires string system and user message content")
        response_format = payload.get("response_format")
        if response_format is not None and type(response_format) is not dict:
            raise ValueError("local response_format must be a JSON object")
        generation_settings: dict[str, object] = {
            "local_server_identity": self.settings.server_identity(),
            "complete_request_payload": dict(payload),
        }
        request = GenerationRequest(
            provider_id="llama.cpp-local-openai-compatible",
            model=self.settings.model,
            system_prompt=str(messages[0]["content"]),
            user_prompt=str(messages[1]["content"]),
            temperature=self.settings.temperature,
            max_output_tokens=int(payload["max_tokens"]),
            response_format=response_format,
            generation_settings=generation_settings,
            task_tag=task_tag,
            phase_tag=self.phase_tag,
            run_id=self.run_id,
            cache_nonce=self.cache_nonce if cache_nonce is None else cache_nonce,
        )
        self._backend_completion = None
        policy_timeout = min(float(timeout), 300.0, self.budget_policy.reservation_lease_seconds / 2.0)
        if policy_timeout <= 0:
            raise ValueError("local cached request timeout must be positive")
        cached = cached_generate(
            self.cache_database,
            request,
            _LocalGenerationBackend(self),
            # Successful local generations record zero cost. A positive tiny
            # reservation keeps the existing total-credit cap and conservative
            # failed-call accounting path intact.
            reserved_credits=1e-9,
            policy=self.budget_policy,
            timeout=policy_timeout,
        )
        cache_request_hash = cached.request_key.split(":", 1)[1].split(":", 1)[0]
        if cached.cache_hit:
            completion = _decode_cached_completion(cached.result.text)
            return LocalCompletion(
                **{
                    **completion,
                    "request_hash": request_hash,
                    "cache_hit": True,
                    "cache_request_hash": cache_request_hash,
                }
            )
        if self._backend_completion is None:
            raise RuntimeError("local cache backend completed without response metadata")
        completion = self._backend_completion
        return LocalCompletion(
            text=completion.text,
            reasoning_text=completion.reasoning_text,
            finish_reason=completion.finish_reason,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            visible_tokens=completion.visible_tokens,
            latency_ms=completion.latency_ms,
            request_hash=request_hash,
            response_model=completion.response_model,
            cache_hit=False,
            cache_request_hash=cache_request_hash,
        )

    def _complete_uncached(
        self,
        payload: Mapping[str, object],
        *,
        timeout: float,
        request_hash: str,
    ) -> LocalCompletion:
        started = time.perf_counter()
        response = self._json_request(
            self.settings.base_url.rstrip("/") + "/v1/chat/completions",
            payload,
            timeout,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        try:
            choice = response["choices"][0]
            message = choice["message"]
            text = message["content"]
            finish_reason = choice["finish_reason"]
            usage = response["usage"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("local chat server returned an incomplete completion response") from exc
        if type(text) is not str or type(finish_reason) is not str or type(usage) is not dict:
            raise ValueError("local chat server returned invalid completion fields")
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if type(prompt_tokens) is not int or type(completion_tokens) is not int:
            raise ValueError("local chat server omitted token usage")
        token_result = self._json_request(
            self.settings.base_url.rstrip("/") + "/tokenize",
            {"content": text, "add_special": False},
            timeout,
        )
        tokens = token_result.get("tokens") if isinstance(token_result, dict) else None
        if type(tokens) is not list:
            raise ValueError("local chat server returned an invalid /tokenize response")
        reasoning = message.get("reasoning_content", message.get("reasoning", ""))
        if type(reasoning) is not str:
            reasoning = ""
        model = response.get("model")
        return LocalCompletion(
            text=text,
            reasoning_text=reasoning,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            visible_tokens=len(tokens),
            latency_ms=latency_ms,
            request_hash=request_hash,
            response_model=model if isinstance(model, str) else None,
        )


class _LocalGenerationBackend:
    def __init__(self, client: LocalOpenAIClient) -> None:
        self.client = client

    def generate(self, request: GenerationRequest, *, timeout: float) -> GenerationResult:
        settings = request.generation_settings or {}
        payload = settings.get("complete_request_payload")
        if type(payload) is not dict:
            raise ValueError("local cached request is missing its complete payload")
        completion = self.client._complete_uncached(
            payload,
            timeout=timeout,
            request_hash=self.client.settings.request_hash(payload),
        )
        self.client._backend_completion = completion
        return GenerationResult(
            text=_encode_cached_completion(completion),
            input_tokens=completion.prompt_tokens,
            output_tokens=completion.completion_tokens,
            actual_cost_credits=0.0,
            latency_ms=int(completion.latency_ms),
            observed_model=completion.response_model,
        )


def _schema_format(name: str, schema: Mapping[str, object]) -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": False, "schema": dict(schema)},
    }


class LocalReaderProvider(ReaderProvider):
    def __init__(self, client: LocalOpenAIClient) -> None:
        self.client = client
        self.model = client.settings.model
        self.last_completion: LocalCompletion | None = None
        self.last_usage: dict[str, object] | None = None
        self.last_payload: dict[str, object] | None = None
        self.prompt_hash = hashlib.sha256(b"no-reader-prompt").hexdigest()
        self.attempts: list[LocalCompletion] = []
        self.truncated_calls = 0

    def reset(self) -> None:
        self.last_completion = None
        self.last_usage = None
        self.last_payload = None
        self.prompt_hash = hashlib.sha256(b"no-reader-prompt").hexdigest()
        self.attempts.clear()

    def generate(self, prompt: ReaderPrompt, *, timeout: float) -> str:
        schema = json.loads(prompt.response_schema_json)
        payload: dict[str, object] = {
            "model": self.client.settings.model,
            "messages": [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_payload},
            ],
            "temperature": self.client.settings.temperature,
            "top_p": self.client.settings.top_p,
            "top_k": self.client.settings.top_k,
            "seed": self.client.settings.seed,
            "max_tokens": self.client.settings.reader_max_tokens,
            "response_format": _schema_format("engram_reader_response", schema),
        }
        self.last_payload = json.loads(prompt.user_payload)
        self.prompt_hash = hashlib.sha256(
            (prompt.system_prompt + "\0" + prompt.user_payload).encode("utf-8")
        ).hexdigest()
        evidence = self.last_payload.get("evidence", [])
        supplied_ids: set[CandidateId] = set()
        if isinstance(evidence, list):
            for item in evidence:
                if isinstance(item, dict) and isinstance(item.get("evidence_id"), str):
                    kind, separator, value = item["evidence_id"].partition(":")
                    if separator:
                        supplied_ids.add(CandidateId(kind, value))

        completion: LocalCompletion | None = None
        for attempt in range(2):
            nonce = None if attempt == 0 else f"{self.client.cache_nonce}:invalid-json-retry-1"
            completion = self.client.complete(
                payload, timeout=timeout, task_tag="reader", cache_nonce=nonce
            )
            self.attempts.append(completion)
            if completion.finish_reason == "length":
                self.truncated_calls += 1
                break
            try:
                _parse_reader_response(completion.text, supplied_ids)
                break
            except Exception:
                if attempt == 1:
                    break
        assert completion is not None
        self.last_completion = completion
        self.last_usage = _aggregate_usage(self.attempts, self.model)
        return completion.text


class LocalQueryClassifierProvider(QueryClassificationProvider):
    def __init__(self, client: LocalOpenAIClient) -> None:
        self.client = client
        self.model = client.settings.model
        self.last_completion: LocalCompletion | None = None
        self.last_usage: dict[str, object] | None = None
        self.last_error: BaseException | None = None
        self.provider_queries: set[str] = set()
        self.attempts: list[LocalCompletion] = []
        self.prompt_hash = hashlib.sha256(b"no-classifier-prompt").hexdigest()

    def reset(self) -> None:
        self.last_completion = None
        self.last_usage = None
        self.last_error = None
        self.attempts.clear()
        self.prompt_hash = hashlib.sha256(b"no-classifier-prompt").hexdigest()

    def classify(self, query: str, *, question_date: date, timeout: float) -> str:
        self.provider_queries.add(query)
        user_payload = json.dumps(
            {"query": query, "question_date": question_date.isoformat()},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        payload: dict[str, object] = {
            "model": self.client.settings.model,
            "messages": [
                {"role": "system", "content": QUERY_CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            ],
            "temperature": self.client.settings.temperature,
            "top_p": self.client.settings.top_p,
            "top_k": self.client.settings.top_k,
            "seed": self.client.settings.seed,
            "max_tokens": self.client.settings.classifier_max_tokens,
            "response_format": _schema_format(
                "engram_query_classification", _QUERY_CLASSIFICATION_RESPONSE_SCHEMA
            ),
        }
        self.prompt_hash = hashlib.sha256(
            (QUERY_CLASSIFIER_SYSTEM_PROMPT + "\0" + user_payload).encode("utf-8")
        ).hexdigest()
        self.last_error = None
        completion: LocalCompletion | None = None
        for attempt in range(2):
            nonce = None if attempt == 0 else f"{self.client.cache_nonce}:invalid-json-retry-1"
            try:
                completion = self.client.complete(
                    payload, timeout=timeout, task_tag="classifier", cache_nonce=nonce
                )
            except BaseException as exc:
                self.last_error = exc
                raise
            self.attempts.append(completion)
            if completion.finish_reason == "length":
                self.last_error = RuntimeError("local classifier response was truncated (finish_reason=length)")
                break
            try:
                # Reuse the public classifier route for production schema and
                # query-consistency validation without making another request.
                classify_query(
                    query,
                    question_date=question_date,
                    classifier=_FixedResponseClassifier(completion.text),
                    timeout=timeout,
                )
                break
            except QueryClassificationOutputError:
                if attempt == 1:
                    break
        assert completion is not None
        self.last_completion = completion
        self.last_usage = _aggregate_usage(self.attempts, self.model)
        return completion.text


def canned_classifier_response() -> str:
    return json.dumps({
        "schema_version": QUERY_CLASSIFICATION_SCHEMA_VERSION,
        "status": "classified",
        "view": "current",
        "as_of": None,
        "history_range": None,
        "reason_code": "provider_classification",
    }, separators=(",", ":"))


class _FixedResponseClassifier:
    def __init__(self, response: str) -> None:
        self.response = response

    def classify(self, query: str, *, question_date: date, timeout: float) -> str:
        del query, question_date, timeout
        return self.response


def _aggregate_usage(attempts: list[LocalCompletion], model: str) -> dict[str, object]:
    return {
        "request_hash": attempts[-1].cache_request_hash or attempts[-1].request_hash,
        "cache_hit": all(item.cache_hit for item in attempts),
        "prompt_tokens": sum(item.prompt_tokens for item in attempts),
        "completion_tokens": sum(item.completion_tokens for item in attempts),
        "visible_tokens": sum(item.visible_tokens for item in attempts),
        "latency_ms": int(sum(item.latency_ms for item in attempts)),
        "cost_credits": 0.0,
        "model": model,
        "attempt_count": len(attempts),
        "finish_reasons": [item.finish_reason for item in attempts],
    }


def _encode_cached_completion(completion: LocalCompletion) -> str:
    return json.dumps({
        "local_cached_completion_version": 1,
        "text": completion.text,
        "reasoning_text": completion.reasoning_text,
        "finish_reason": completion.finish_reason,
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
        "visible_tokens": completion.visible_tokens,
        "latency_ms": completion.latency_ms,
        "request_hash": completion.request_hash,
        "response_model": completion.response_model,
    }, ensure_ascii=False, separators=(",", ":"))


def _decode_cached_completion(value: str) -> dict[str, object]:
    try:
        payload = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        # Older local smoke cache rows contain the raw model text. Preserve
        # compatibility while marking unavailable metadata conservatively.
        return {
            "text": value,
            "reasoning_text": "",
            "finish_reason": "cache_metadata_unavailable",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "visible_tokens": 0,
            "latency_ms": 0.0,
            "request_hash": "",
            "response_model": None,
        }
    if not isinstance(payload, dict) or payload.get("local_cached_completion_version") != 1:
        return {
            "text": value,
            "reasoning_text": "",
            "finish_reason": "cache_metadata_unavailable",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "visible_tokens": 0,
            "latency_ms": 0.0,
            "request_hash": "",
            "response_model": None,
        }
    return {
        key: payload[key]
        for key in (
            "text", "reasoning_text", "finish_reason", "prompt_tokens",
            "completion_tokens", "visible_tokens", "latency_ms", "request_hash",
            "response_model",
        )
    }
