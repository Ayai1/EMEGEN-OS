"""Record the neutral scripted scenario from a running local demo server."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "demo" / "recordings" / "session.json"


def request_json(base_url: str, method: str, path: str, body: dict[str, object] | None = None) -> tuple[dict[str, object], float]:
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = Request(
        base_url.rstrip("/") + path,
        data=payload,
        method=method,
        headers={"Content-Type": "application/json"} if payload is not None else {},
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=300.0) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"demo server request failed: {exc}") from exc
    elapsed = time.perf_counter() - started
    if not isinstance(result, dict):
        raise RuntimeError(f"demo server returned a non-object response for {path}")
    return result, elapsed


def record(base_url: str, output: Path, *, force: bool = False) -> dict[str, object]:
    output = output.expanduser().resolve()
    if output.exists() and not force:
        raise FileExistsError(f"recording already exists; pass --force to overwrite: {output}")
    scenario = json.loads((ROOT / "demo" / "scenario.json").read_text(encoding="utf-8"))
    state, state_latency = request_json(base_url, "GET", "/api/state")
    if state.get("mode") != "live":
        raise ValueError("recording requires a live-mode demo server")
    events: list[dict[str, object]] = [{
        "label": "Initial server state",
        "kind": "state",
        "request": {"method": "GET", "path": "/api/state"},
        "response": state,
        "latency_seconds": state_latency,
    }]
    reset_response, reset_latency = request_json(base_url, "POST", "/api/reset", {})
    events.append({
        "label": "Reset session",
        "kind": "reset",
        "request": {"method": "POST", "path": "/api/reset", "body": {}},
        "response": reset_response,
        "latency_seconds": reset_latency,
    })
    for step in scenario["steps"]:
        kind = step["kind"]
        if kind == "ingest":
            body = {key: step[key] for key in ("text", "role", "source_type", "timestamp")}
            route = "/api/ingest"
        elif kind == "ask":
            body = {"question": step["question"]}
            route = "/api/ask"
        else:
            raise ValueError(f"unsupported scenario step kind: {kind!r}")
        response, latency = request_json(base_url, "POST", route, body)
        events.append({
            "label": step["label"],
            "kind": kind,
            "request": {"method": "POST", "path": route, "body": body},
            "response": response,
            "latency_seconds": latency,
        })
    final_state, latency = request_json(base_url, "GET", "/api/state")
    events.append({
        "label": "Final state",
        "kind": "state",
        "request": {"method": "GET", "path": "/api/state"},
        "response": final_state,
        "latency_seconds": latency,
    })
    recording: dict[str, object] = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "mode": "live",
        "embedding": state.get("embedding"),
        "models": state.get("models", {}),
        "events": events,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(recording, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return recording


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        recording = record(args.base_url, args.output, force=args.force)
    except (FileExistsError, ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    print(f"Recorded {len(recording['events'])} requests to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
