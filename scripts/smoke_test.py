"""One explicitly approved live classifier request and one Reader request."""

from __future__ import annotations

import json
import os
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4
from datetime import date

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engramOS.db import initialize_database
from engramOS.meta_adapters import (
    MetaModelConfig,
    MetaQueryClassifier,
    MetaReaderProvider,
    pricing_mode_from_model,
    smoke_nonce,
    smoke_reader_prompt,
)


def main() -> int:
    if not os.environ.get("MODEL_API_KEY"):
        raise SystemExit("Set MODEL_API_KEY in the environment before the approved smoke test")
    config = MetaModelConfig.from_environment()
    run_id = f"smoke-{uuid4().hex}"
    errors: list[tuple[str, str]] = []

    with TemporaryDirectory(prefix="engramos-smoke-") as folder:
        database = str(Path(folder) / "smoke.sqlite3")
        initialize_database(database)
        classifier = MetaQueryClassifier(
            database,
            config=config,
            phase_tag="dev_baselines",
            run_id=run_id,
            max_attempts=1,
            cache_nonce=smoke_nonce(),
        )
        reader = MetaReaderProvider(
            database,
            api=classifier.api,
            config=config,
            phase_tag="dev_baselines",
            run_id=run_id,
            max_attempts=1,
            cache_nonce=smoke_nonce(),
        )

        # Keep the two independent smoke calls even if one returns an error.
        for name, action in (
            ("classifier", lambda: classifier.classify(
                "What city is in my current project notes?",
                question_date=date.today(), timeout=30.0,
            )),
            ("reader", lambda: reader.generate(smoke_reader_prompt(), timeout=30.0)),
        ):
            try:
                action()
            except Exception as exc:
                errors.append((name, type(exc).__name__))

        observed_models = []
        for name, adapter in (("classifier", classifier), ("reader", reader)):
            call = adapter.last_call
            if call is None:
                print(f"{name}: no completed cache record")
                continue
            result = call.result
            observed = result.observed_model
            print(
                f"{name}: model={observed or 'not returned'} "
                f"prompt_tokens={result.input_tokens} completion_tokens={result.output_tokens} "
                f"cost_credits={result.actual_cost_credits:.8f} cache_hit={call.cache_hit}"
            )
            if observed:
                observed_models.append(observed)

        with closing(sqlite3.connect(database)) as conn:
            ledger_calls = conn.execute(
                "SELECT COUNT(*) FROM llm_call_ledger WHERE run_id=?", (run_id,)
            ).fetchone()[0]
            cache_calls = conn.execute(
                "SELECT COUNT(*) FROM llm_cache WHERE run_id=?", (run_id,)
            ).fetchone()[0]
            total_credits = conn.execute(
                "SELECT COALESCE(SUM(cost_credits), 0) FROM llm_cache WHERE run_id=?",
                (run_id,),
            ).fetchone()[0]

    input_usd, output_usd, cached_usd = config.usd_rates_per_million
    input_credits, output_credits, cached_credits = config.credit_rates_per_million
    print(
        "published_rate_card="
        f"{config.pricing_mode} USD_per_1M(input={input_usd}, cached_input={cached_usd}, output={output_usd})"
    )
    print(
        f"campaign_conversion={config.credits_per_usd} credits/USD; "
        f"configured_credits_per_1M(input={input_credits}, "
        f"cached_input={cached_credits}, output={output_credits})"
    )
    if observed_models:
        observed_modes = sorted({pricing_mode_from_model(model) for model in observed_models})
        print(
            f"observed_response_models={','.join(observed_models)} "
            f"observed_pricing_modes={','.join(observed_modes)}"
        )
    else:
        print("observed_pricing_mode=unknown (response did not include a model identifier)")
    print(
        f"run_id={run_id} api_attempts={ledger_calls} cache_rows={cache_calls} "
        f"total_credits={total_credits:.8f}"
    )
    for name, error in errors:
        print(f"{name}: failed ({error})")
    if ledger_calls != 2:
        print("Expected exactly two reserved API attempts; inspect the temporary-run output.")
        return 2
    if errors or len(observed_models) != 2:
        return 1
    if len({pricing_mode_from_model(model) for model in observed_models}) != 1:
        print("The classifier and Reader returned different pricing-mode model IDs.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
