"""Offline request/response tests for the Meta Model API adapter."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engramOS.db import initialize_database
from engramOS.llm_cache import GenerationRequest, cached_generate
from engramOS.meta_provider import MetaModelAPIError, MetaModelChatProvider


class MetaModelProviderTests(unittest.TestCase):
    def request(self, **kwargs: object) -> GenerationRequest:
        values: dict[str, object] = {
            "provider_id": "meta-model-api",
            "model": "muse-spark-1.3",
            "system_prompt": "system rule",
            "user_prompt": "hello",
            "temperature": 0,
            "max_output_tokens": 64,
            "response_format": {"type": "json_object"},
        }
        values.update(kwargs)
        return GenerationRequest(**values)  # type: ignore[arg-type]

    def test_openai_compatible_chat_request_and_cost_accounting(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "ready"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            })

        client = httpx.Client(transport=httpx.MockTransport(handler))
        provider = MetaModelChatProvider(
            api_key="test-secret",
            input_cost_per_million=1.25,
            output_cost_per_million=4.25,
            client=client,
        )
        result = provider.generate(self.request(), timeout=1)

        body = json.loads(seen[0].content)
        self.assertEqual(str(seen[0].url), "https://api.meta.ai/v1/chat/completions")
        self.assertEqual(seen[0].headers["authorization"], "Bearer test-secret")
        self.assertEqual(body["model"], "muse-spark-1.3")
        self.assertEqual(body["max_tokens"], 64)
        self.assertEqual(body["messages"][0]["role"], "developer")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(result.text, "ready")
        self.assertEqual(result.input_tokens, 10)
        self.assertEqual(result.output_tokens, 2)
        self.assertAlmostEqual(result.actual_cost_credits, (10 * 1.25 + 2 * 4.25) / 1_000_000)
        client.close()

    def test_reservation_estimate_uses_input_bound_and_output_limit(self) -> None:
        provider = MetaModelChatProvider(
            api_key="test-secret",
            input_cost_per_million=1.25,
            output_cost_per_million=4.25,
        )
        estimate = provider.estimate_reserved_credits(
            self.request(max_output_tokens=100), input_token_upper_bound=200
        )
        self.assertAlmostEqual(estimate, (200 * 1.25 + 100 * 4.25) / 1_000_000)

    def test_error_does_not_include_provider_body_or_api_key(self) -> None:
        client = httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(401, text="secret diagnostic")
        ))
        provider = MetaModelChatProvider(
            api_key="test-secret",
            input_cost_per_million=1.25,
            output_cost_per_million=4.25,
            client=client,
        )
        with self.assertRaises(MetaModelAPIError) as caught:
            provider.generate(self.request(), timeout=1)
        self.assertNotIn("test-secret", str(caught.exception))
        self.assertNotIn("secret diagnostic", str(caught.exception))
        client.close()

    def test_generation_settings_cannot_raise_reserved_output_limit(self) -> None:
        provider = MetaModelChatProvider(
            api_key="test-secret",
            input_cost_per_million=1.25,
            output_cost_per_million=4.25,
        )
        with self.assertRaises(ValueError):
            provider.generate(
                self.request(generation_settings={"max_tokens": 500_000}), timeout=1
            )

    def test_cached_provider_flow_reserves_then_reuses_without_second_http_call(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "ready"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            })

        with tempfile.TemporaryDirectory() as folder:
            database = str(Path(folder) / "cache.sqlite3")
            initialize_database(database)
            client = httpx.Client(transport=httpx.MockTransport(handler))
            provider = MetaModelChatProvider(
                api_key="test-secret",
                input_cost_per_million=1.25,
                output_cost_per_million=4.25,
                client=client,
            )
            request = self.request(
                task_tag="reader", phase_tag="dev_baselines", run_id="smoke"
            )
            reservation = provider.estimate_reserved_credits(request)
            first = cached_generate(
                database, request, provider, reserved_credits=reservation
            )
            second = cached_generate(
                database, request, provider, reserved_credits=reservation
            )
            client.close()

        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
