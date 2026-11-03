"""Complete-request LLM caching and pre-call credit reservations.

This module intentionally contains no live provider client. Adapters must call
``cached_generate`` so the cache lookup and budget reservation happen before a
request can reach a paid endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import sqlite3
import time
from types import MappingProxyType
from typing import Mapping, Protocol
from uuid import uuid4

from .db import connection, transaction


class BudgetExceededError(RuntimeError):
    """The request's maximum estimated cost exceeds a configured credit cap."""


class GenerationInFlightTimeout(TimeoutError):
    """Another worker did not finish an identical request before the wait limit."""


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    provider_id: str
    model: str
    system_prompt: str
    user_prompt: str
    temperature: float = 0.0
    max_output_tokens: int = 1024
    response_format: Mapping[str, object] | None = None
    generation_settings: Mapping[str, object] | None = None
    task_tag: str = "unspecified"
    phase_tag: str = "unspecified"
    run_id: str = "unspecified"
    cache_nonce: str = ""


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    actual_cost_credits: float
    latency_ms: int | None = None
    observed_model: str | None = None


class GenerationProvider(Protocol):
    def generate(
        self, request: GenerationRequest, *, timeout: float
    ) -> GenerationResult:
        """Perform exactly one provider request and return measured usage."""


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    total_cap: float = 150.0
    phase_caps: Mapping[str, float] | None = None
    reservation_lease_seconds: float = 300.0


DEFAULT_BUDGET_POLICY = BudgetPolicy(
    total_cap=150.0,
    phase_caps=MappingProxyType({
        "dev_baselines": 19.0,
        "dev_extraction": 32.0,
        "dev_poison": 18.0,
        "locked_test": 67.0,
        "buffer": 14.0,
    }),
)


@dataclass(frozen=True, slots=True)
class CachedGeneration:
    result: GenerationResult
    cache_hit: bool
    request_key: str


def cached_generate(
    database: str,
    request: GenerationRequest,
    provider: GenerationProvider,
    *,
    reserved_credits: float,
    policy: BudgetPolicy = DEFAULT_BUDGET_POLICY,
    timeout: float = 30.0,
    wait_timeout: float = 60.0,
    bypass_cache: bool = False,
) -> CachedGeneration:
    """Return a cached output or reserve the budget, call, and record actual cost.

    ``reserved_credits`` must cover the worst-case cost for the given input and
    ``max_output_tokens`` under the configured provider rate card. If a provider
    call fails without usage details, the full reservation is charged
    conservatively. Cache identity hashes every generation setting and prompt;
    task/phase/run labels are accounting metadata and do not fragment reuse.
    """
    _validate_request(request)
    _finite_nonnegative(reserved_credits, "reserved_credits")
    if reserved_credits <= 0:
        raise ValueError("reserved_credits must be greater than zero")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    if not math.isfinite(wait_timeout) or wait_timeout < 0:
        raise ValueError("wait_timeout must be non-negative and finite")
    _validate_policy(policy)
    if timeout >= policy.reservation_lease_seconds:
        raise ValueError("timeout must be shorter than the reservation lease")
    if bypass_cache:
        request = GenerationRequest(
            **{
                field_name: getattr(request, field_name)
                for field_name in request.__dataclass_fields__
                if field_name != "cache_nonce"
            },
            cache_nonce=f"bypass:{uuid4().hex}",
        )
    request_hash = _request_hash(request)
    request_key = f"{request.model}:{request_hash}:{request.temperature}:{request.cache_nonce}"
    deadline = time.monotonic() + wait_timeout

    while True:
        _, cached, call_id = _reserve_or_read(
            database, request, request_hash, request_key,
            reserved_credits, policy,
        )
        if cached is not None:
            return CachedGeneration(cached, True, request_key)
        if call_id is not None:
            break
        if time.monotonic() >= deadline:
            raise GenerationInFlightTimeout("identical provider request is still in flight")
        time.sleep(0.05)

    try:
        result = provider.generate(request, timeout=timeout)
        _validate_result(result)
    except BaseException as exc:
        _finish_failure(database, call_id, reserved_credits, type(exc).__name__)
        raise

    over_reservation = result.actual_cost_credits > reserved_credits
    total_overrun, phase_overrun = _finish_success(
        database, call_id, request, request_hash, result, policy
    )
    if over_reservation or total_overrun or phase_overrun:
        raise BudgetExceededError(
            "provider actual cost exceeded its reservation or a credit cap; "
            "the call is recorded and its output is not returned"
        )
    return CachedGeneration(result, False, request_key)


def _reserve_or_read(
    database: str,
    request: GenerationRequest,
    request_hash: str,
    request_key: str,
    reserved: float,
    policy: BudgetPolicy,
) -> tuple[str, GenerationResult | None, str | None]:
    with transaction(database) as conn:
        row = conn.execute(
            """SELECT response, in_tokens, out_tokens, cost_credits, latency_ms
               FROM llm_cache WHERE model = ? AND prompt_hash = ? AND temperature = ?
                 AND cache_nonce = ?""",
            (request.model, request_hash, request.temperature, request.cache_nonce),
        ).fetchone()
        if row is not None:
            cached = GenerationResult(
                str(row["response"]), int(row["in_tokens"]), int(row["out_tokens"]),
                float(row["cost_credits"]), row["latency_ms"],
            )
            return "cached", cached, None

        now = time.time()
        stale = conn.execute(
            """SELECT call_id, reserved_credits FROM llm_call_ledger
               WHERE request_key = ? AND status = 'reserved' AND started_at <= ?""",
            (request_key, now - policy.reservation_lease_seconds),
        ).fetchone()
        if stale is not None:
            conn.execute(
                """UPDATE llm_call_ledger SET status='failed', actual_credits=reserved_credits,
                   finished_at=?, error_type='ReservationLeaseExpired' WHERE call_id=?""",
                (now, stale["call_id"]),
            )
        inflight = conn.execute(
            "SELECT 1 FROM llm_call_ledger WHERE request_key=? AND status='reserved'",
            (request_key,),
        ).fetchone()
        if inflight is not None:
            return "wait", None, None

        total_spent, phase_spent = _spend_totals(conn, request.phase_tag)
        if total_spent + reserved > policy.total_cap:
            raise BudgetExceededError("request reservation exceeds the total credit cap")
        phase_cap = (policy.phase_caps or {}).get(request.phase_tag)
        if phase_cap is not None and phase_spent + reserved > phase_cap:
            raise BudgetExceededError(
                f"request reservation exceeds the {request.phase_tag!r} phase cap"
            )
        call_id = uuid4().hex
        conn.execute(
            """INSERT INTO llm_call_ledger (
               call_id, request_key, model, task_tag, phase_tag, run_id,
                   reserved_credits, actual_credits, status, started_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 'reserved', ?)""",
            (call_id, request_key, request.model, request.task_tag, request.phase_tag,
             request.run_id, reserved, now),
        )
        return "reserved", None, call_id


def _finish_success(
    database: str,
    call_id: str,
    request: GenerationRequest,
    request_hash: str,
    result: GenerationResult,
    policy: BudgetPolicy,
) -> tuple[bool, bool]:
    with transaction(database) as conn:
        call = conn.execute(
            "SELECT reserved_credits FROM llm_call_ledger WHERE call_id=? AND status='reserved'",
            (call_id,),
        ).fetchone()
        if call is None:
            raise RuntimeError("provider reservation disappeared before completion")
        conn.execute(
            """INSERT INTO llm_cache (
                   model, prompt_hash, temperature, cache_nonce, task_tag, response,
                   in_tokens, out_tokens, cost_credits, latency_ms, phase_tag, run_id, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (request.model, request_hash, request.temperature, request.cache_nonce,
             request.task_tag, result.text, result.input_tokens, result.output_tokens,
             result.actual_cost_credits, result.latency_ms, request.phase_tag,
             request.run_id, time.time()),
        )
        conn.execute(
            """UPDATE llm_call_ledger SET status='succeeded', actual_credits=?, finished_at=?
               WHERE call_id=?""",
            (result.actual_cost_credits, time.time(), call_id),
        )
        total_now, phase_now = _spend_totals(conn, request.phase_tag)
        phase_cap = (policy.phase_caps or {}).get(request.phase_tag)
        return total_now > policy.total_cap, phase_cap is not None and phase_now > phase_cap


def _finish_failure(database: str, call_id: str, reserved: float, error_type: str) -> None:
    with transaction(database) as conn:
        conn.execute(
            """UPDATE llm_call_ledger SET status='failed', actual_credits=?, finished_at=?,
               error_type=? WHERE call_id=? AND status='reserved'""",
            (reserved, time.time(), error_type, call_id),
        )


def _spend_totals(conn: sqlite3.Connection, phase_tag: str) -> tuple[float, float]:
    cache_total = float(conn.execute(
        "SELECT COALESCE(SUM(cost_credits), 0) FROM llm_cache"
    ).fetchone()[0])
    ledger_total = float(conn.execute(
        """SELECT COALESCE(SUM(actual_credits), 0) FROM llm_call_ledger
           WHERE status='failed'"""
    ).fetchone()[0])
    reserved_total = float(conn.execute(
        """SELECT COALESCE(SUM(reserved_credits), 0) FROM llm_call_ledger
           WHERE status='reserved'"""
    ).fetchone()[0])
    phase_cache = float(conn.execute(
        "SELECT COALESCE(SUM(cost_credits), 0) FROM llm_cache WHERE phase_tag=?",
        (phase_tag,),
    ).fetchone()[0])
    phase_ledger = float(conn.execute(
        """SELECT COALESCE(SUM(actual_credits), 0) FROM llm_call_ledger
           WHERE phase_tag=? AND status='failed'""",
        (phase_tag,),
    ).fetchone()[0])
    phase_reserved = float(conn.execute(
        """SELECT COALESCE(SUM(reserved_credits), 0) FROM llm_call_ledger
           WHERE phase_tag=? AND status='reserved'""",
        (phase_tag,),
    ).fetchone()[0])
    return cache_total + ledger_total + reserved_total, phase_cache + phase_ledger + phase_reserved


def _request_hash(request: GenerationRequest) -> str:
    payload = {
        "provider_id": request.provider_id,
        "model": request.model,
        "system_prompt": request.system_prompt,
        "user_prompt": request.user_prompt,
        "temperature": request.temperature,
        "max_output_tokens": request.max_output_tokens,
        "response_format": dict(request.response_format) if request.response_format is not None else None,
        "generation_settings": dict(request.generation_settings) if request.generation_settings is not None else None,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_request(request: GenerationRequest) -> None:
    if not isinstance(request, GenerationRequest):
        raise TypeError("request must be GenerationRequest")
    for field_name in ("provider_id", "model", "system_prompt", "user_prompt", "task_tag", "phase_tag", "run_id", "cache_nonce"):
        if type(getattr(request, field_name)) is not str:
            raise TypeError(f"{field_name} must be a string")
    if not request.provider_id or not request.model or not request.phase_tag:
        raise ValueError("provider_id, model, and phase_tag must be non-empty")
    if isinstance(request.temperature, bool) or not isinstance(request.temperature, (int, float)) or not math.isfinite(request.temperature):
        raise ValueError("temperature must be finite")
    if isinstance(request.max_output_tokens, bool) or not isinstance(request.max_output_tokens, int) or request.max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be a positive integer")
    _canonical_json(request.response_format)
    _canonical_json(request.generation_settings)


def _validate_result(result: GenerationResult) -> None:
    if not isinstance(result, GenerationResult) or type(result.text) is not str:
        raise TypeError("provider must return GenerationResult with text")
    for field_name in ("input_tokens", "output_tokens"):
        value = getattr(result, field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field_name} must be a non-negative integer")
    _finite_nonnegative(result.actual_cost_credits, "actual_cost_credits")
    if result.latency_ms is not None and (isinstance(result.latency_ms, bool) or not isinstance(result.latency_ms, int) or result.latency_ms < 0):
        raise ValueError("latency_ms must be a non-negative integer or None")


def _validate_policy(policy: BudgetPolicy) -> None:
    _finite_nonnegative(policy.total_cap, "total_cap")
    _finite_nonnegative(policy.reservation_lease_seconds, "reservation_lease_seconds")
    if policy.total_cap <= 0 or policy.reservation_lease_seconds <= 0:
        raise ValueError("budget caps and reservation lease must be positive")
    for name, cap in (policy.phase_caps or {}).items():
        if type(name) is not str or not name:
            raise ValueError("phase cap names must be non-empty strings")
        _finite_nonnegative(cap, f"phase cap for {name}")


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("generation settings must be JSON-serializable") from exc


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)
