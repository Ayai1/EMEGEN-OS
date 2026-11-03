"""One-shot Meta Model API Chat Completions adapter.

Paid calls are never made by constructing this adapter. Use it only through
``llm_cache.cached_generate`` so budget reservation and response caching occur
before the HTTP request.
"""

from __future__ import annotations

import math
import os
import time
import httpx
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from .llm_cache import GenerationRequest, GenerationResult


class MetaModelAPIError(RuntimeError):
    """The provider response could not be used safely."""

    def __init__(
        self, error_type: str, *, retryable: bool = False,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(error_type)
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds


class MetaModelChatProvider:
    """Minimal OpenAI-compatible client for Meta Model API chat completions."""

    provider_id = "meta-model-api"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.meta.ai/v1",
        input_cost_per_million: float,
        output_cost_per_million: float,
        cached_input_cost_per_million: float | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("MODEL_API_KEY")
        if type(key) is not str or not key.strip():
            raise ValueError("Set MODEL_API_KEY before using the Meta Model API")
        if type(base_url) is not str or not base_url.startswith("https://"):
            raise ValueError("base_url must be an HTTPS URL")
        for value, label in (
            (input_cost_per_million, "input_cost_per_million"),
            (output_cost_per_million, "output_cost_per_million"),
            (input_cost_per_million if cached_input_cost_per_million is None
             else cached_input_cost_per_million, "cached_input_cost_per_million"),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        self._api_key = key
        self._base_url = base_url.rstrip("/")
        self._input_rate = float(input_cost_per_million)
        self._output_rate = float(output_cost_per_million)
        self._cached_input_rate = float(
            input_cost_per_million if cached_input_cost_per_million is None
            else cached_input_cost_per_million
        )
        self._client = client

    def estimate_reserved_credits(
        self, request: GenerationRequest, *, input_token_upper_bound: int | None = None
    ) -> float:
        """Estimate a conservative reservation from a caller token bound.

        The caller supplies a tokenizer-derived bound where available. The
        fallback uses UTF-8 byte count plus message overhead, which is
        conservative for byte-level tokenizers but is still an estimate.
        """
        if input_token_upper_bound is None:
            input_token_upper_bound = (
                len(request.system_prompt.encode("utf-8"))
                + len(request.user_prompt.encode("utf-8")) + 16
            )
        if (isinstance(input_token_upper_bound, bool)
                or not isinstance(input_token_upper_bound, int)
                or input_token_upper_bound < 0):
            raise ValueError("input_token_upper_bound must be a non-negative integer")
        return (
            input_token_upper_bound * self._input_rate
            + request.max_output_tokens * self._output_rate
        ) / 1_000_000

    def generate(self, request: GenerationRequest, *, timeout: float) -> GenerationResult:
        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be GenerationRequest")
        if request.provider_id != self.provider_id:
            raise ValueError("request provider_id does not match Meta Model API")
        if request.generation_settings is not None and {
            "model", "messages", "temperature", "max_tokens", "response_format"
        }.intersection(request.generation_settings):
            raise ValueError("generation_settings cannot override reserved request fields")
        body: dict[str, object] = {
            "model": request.model,
            "messages": [
                {"role": "developer", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        if request.response_format is not None:
            body["response_format"] = dict(request.response_format)
        if request.generation_settings is not None:
            body.update(dict(request.generation_settings))

        owns_client = self._client is None
        client = self._client or httpx.Client(base_url=self._base_url, timeout=timeout)
        started = time.perf_counter()
        try:
            response = client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=body,
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            retry_after = _retry_after_seconds(exc.response.headers.get("retry-after"))
            raise MetaModelAPIError(
                type(exc).__name__,
                retryable=status in (408, 429) or status >= 500,
                retry_after_seconds=retry_after,
            ) from exc
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise MetaModelAPIError(type(exc).__name__, retryable=True) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise MetaModelAPIError(type(exc).__name__) from exc
        finally:
            if owns_client:
                client.close()

        try:
            choices = payload["choices"]
            message = choices[0]["message"]
            text = message["content"]
            usage = payload["usage"]
            input_tokens = usage["prompt_tokens"]
            output_tokens = usage["completion_tokens"]
            details = usage.get("prompt_tokens_details") or {}
            cached_input_tokens = details.get("cached_tokens", 0)
            observed_model = payload.get("model")
        except (KeyError, IndexError, TypeError) as exc:
            raise MetaModelAPIError("response schema is incomplete") from exc
        if type(text) is not str:
            raise MetaModelAPIError("response content is not text")
        for value in (input_tokens, output_tokens, cached_input_tokens):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MetaModelAPIError("response usage is invalid")
        if cached_input_tokens > input_tokens:
            raise MetaModelAPIError("response cached-token usage is invalid")
        if observed_model is not None and type(observed_model) is not str:
            raise MetaModelAPIError("response model identifier is invalid")
        cost = (
            (input_tokens - cached_input_tokens) * self._input_rate
            + cached_input_tokens * self._cached_input_rate
            + output_tokens * self._output_rate
        ) / 1_000_000
        return GenerationResult(
            text, input_tokens, output_tokens, cost,
            max(0, round((time.perf_counter() - started) * 1000)),
            observed_model,
        )


def _retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        try:
            deadline = parsedate_to_datetime(value)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            parsed = deadline.timestamp() - datetime.now(timezone.utc).timestamp()
        except (TypeError, ValueError, OverflowError):
            return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None
