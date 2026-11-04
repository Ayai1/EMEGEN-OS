"""Offline-capable local web/API host for the EngramOS hackathon demo."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse
from uuid import uuid4
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from engramOS.db import connection, initialize_database  # noqa: E402
from engramOS.gate import DefaultSignalEvaluator  # noqa: E402
from engramOS.ingestion import QuarantinedTurnError, TurnInput, ingest_turn  # noqa: E402
from engramOS.pipeline import answer_query  # noqa: E402
from engramOS.temporal import get_current_facts, get_fact_history  # noqa: E402

MODEL_FILE = "Muse-Glimmer-30B-UD-Q4_K_XL.gguf"
MODEL_SHA256 = "82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38"
START_COMMAND = "powershell -ExecutionPolicy Bypass -File scripts/local_server.ps1 -GpuLayers 32"
HISTORY_ID = "demo-session"


class LocalFactExtractor:
    """Extraction adapter using the existing local Muse Glimmer transport."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def extract(self, turn: Any, prompt: Any) -> str:
        settings = self.client.settings
        payload = {
            "model": settings.model,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "top_k": settings.top_k,
            "seed": settings.seed,
            "max_tokens": 2048,
        }
        try:
            return self.client.complete(payload, timeout=300.0, task_tag="extractor").text
        except Exception as exc:
            raise RuntimeError(
                f"Local model server request failed ({type(exc).__name__}). Start it with: {START_COMMAND}"
            ) from exc


def parse_fixed_timestamp(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        timestamp = float(value)
    except ValueError:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        timestamp = parsed.timestamp()
    if not math.isfinite(timestamp):
        raise ValueError("--fixed-now must be a finite timestamp or ISO date/time")
    return timestamp


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _safe_db_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    for forbidden in (PROJECT_ROOT / "eval", PROJECT_ROOT / "logs"):
        if _is_within(resolved, forbidden.resolve()):
            raise ValueError(f"database directory cannot be under {forbidden.name}/")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


class DemoService:
    """Session state and application routes, with injectable fake providers for tests."""

    def __init__(
        self,
        *,
        mode: str = "live",
        embedding: str = "bge",
        db_dir: Path | None = None,
        fixed_now: float | None = None,
        providers: Mapping[str, Any] | None = None,
    ) -> None:
        if mode not in {"live", "replay"}:
            raise ValueError("mode must be live or replay")
        if embedding not in {"bge", "none"}:
            raise ValueError("embedding must be bge or none")
        self.mode = mode
        self.embedding_mode = embedding
        self.fixed_now = fixed_now
        default_dir = Path(tempfile.gettempdir()) / "engramos-demo"
        self.db_dir = _safe_db_dir(db_dir or default_dir)
        self.lock = threading.RLock()
        self.evaluator = DefaultSignalEvaluator()
        self.db_path: Path | None = None
        self.session_id = ""
        self.embedding_provider: Any | None = None
        self.classifier: Any | None = None
        self.reader: Any | None = None
        self.extractor: Any | None = None
        self.client: Any | None = None
        self.injected_providers = dict(providers or {})
        self.reset()

    @property
    def models(self) -> dict[str, str]:
        return {
            "classifier": "Muse Glimmer (local llama-server)",
            "reader": "Muse Glimmer (local llama-server)",
            "extractor": "Muse Glimmer (local llama-server)",
            "embeddings": "BGE-M3 (non-Meta)" if self.embedding_mode == "bge" else "none",
        }

    def reset(self) -> dict[str, Any]:
        with self.lock:
            path = (self.db_dir / f"session-{uuid4().hex}.sqlite3").resolve()
            self._guard_database_path(path)
            initialize_database(path)
            if self.embedding_mode == "none":
                with connection(path) as conn:
                    conn.execute(
                        "INSERT INTO database_metadata(key, value) VALUES ('embedding_identity', 'none')"
                    )
            self.db_path = path
            self.session_id = uuid4().hex
            self.embedding_provider = None
            self.classifier = None
            self.reader = None
            self.extractor = None
            self.client = None
            if self.mode == "live":
                self._build_live_providers()
            return {"database": path.name, "mode": self.mode, "embedding": self.embedding_mode}

    def _guard_database_path(self, path: Path) -> None:
        resolved = path.resolve()
        for forbidden in (PROJECT_ROOT / "eval", PROJECT_ROOT / "logs"):
            if _is_within(resolved, forbidden.resolve()):
                raise ValueError(f"refusing database path under {forbidden.name}/")

    def _build_live_providers(self) -> None:
        if self.injected_providers:
            self.reader = self.injected_providers.get("reader")
            self.classifier = self.injected_providers.get("classifier")
            self.extractor = self.injected_providers.get("extractor")
            self.embedding_provider = self.injected_providers.get("embedding")
            return
        from engramOS.local_llama import (
            LocalModelSettings,
            LocalOpenAIClient,
            LocalQueryClassifierProvider,
            LocalReaderProvider,
        )

        settings = LocalModelSettings(
            base_url="http://127.0.0.1:8080",
            model=MODEL_FILE,
            model_sha256=MODEL_SHA256,
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
        self.client = LocalOpenAIClient(
            settings, cache_database=str(self.db_path), phase_tag="demo", run_id="local-demo"
        )
        self.reader = LocalReaderProvider(self.client)
        self.classifier = LocalQueryClassifierProvider(self.client)
        self.extractor = LocalFactExtractor(self.client)
        if self.embedding_mode == "bge":
            from engramOS.embeddings import BgeM3EmbeddingProvider
            self.embedding_provider = BgeM3EmbeddingProvider()

    def _ensure_db(self) -> Path:
        if self.db_path is None:
            self.reset()
        assert self.db_path is not None
        return self.db_path

    def scenario(self) -> dict[str, Any]:
        return json.loads((PROJECT_ROOT / "demo" / "scenario.json").read_text(encoding="utf-8"))

    def recording(self) -> dict[str, Any]:
        path = PROJECT_ROOT / "demo" / "recordings" / "session.json"
        if not path.is_file():
            return {
                "available": False,
                "message": "No recording exists yet. Start the server in live mode, then run:",
                "command": ".\\.venv\\Scripts\\python.exe demo\\record_session.py",
            }
        return {"available": True, "recording": json.loads(path.read_text(encoding="utf-8"))}

    def claims(self) -> dict[str, Any]:
        path = PROJECT_ROOT / "demo" / "claims.json"
        if not path.is_file():
            return {"figures": [], "limitations": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def ingest(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.mode != "live":
                raise ValueError("Recorded replay mode is read-only; select live mode to ingest turns.")
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be non-empty")
            role = str(payload.get("role", "user"))
            source_type = str(payload.get("source_type", "user_turn"))
            timestamp = _timestamp(payload.get("timestamp"))
            db = self._ensure_db()
            with connection(db) as conn:
                next_idx = int(conn.execute(
                    "SELECT COUNT(*) FROM turns WHERE history_id=?",
                    (HISTORY_ID,),
                ).fetchone()[0])
            turn = TurnInput(
                history_id=HISTORY_ID,
                # The ingestion contract requires a stable timestamp for each
                # session. Each standalone demo turn is its own session so an
                # ISO timestamp can retain exact event ordering.
                session_id=f"{self.session_id}-{next_idx:06d}",
                turn_idx=0,
                role=role,
                raw_content=text.strip(),
                session_unix_ts=timestamp,
                source_type=source_type,
                session_date=datetime.fromtimestamp(timestamp, timezone.utc).date().isoformat(),
            )
            try:
                result = ingest_turn(
                    str(db), turn, evaluator=self.evaluator, extractor=self.extractor,
                    recorded_at=timestamp, timeout=300.0, strict_untrusted=False,
                    slot_aliases=False,
                )
                state = result.state
                turn_id = result.turn_id
            except QuarantinedTurnError as exc:
                state = "quarantined"
                turn_id = exc.turn_id
            except (ConnectionError, TimeoutError, OSError) as exc:
                raise RuntimeError(f"Local model server unavailable: {type(exc).__name__}. Start it with: {START_COMMAND}") from exc
            with connection(db) as conn:
                gate = conn.execute(
                    "SELECT event_id, signal_bridge, signal_policy, signal_source, signals_fired, decision "
                    "FROM gate_events WHERE turn_id=? ORDER BY event_id DESC LIMIT 1", (turn_id,),
                ).fetchone()
                facts = conn.execute(
                    "SELECT fact_id, entity, attribute, value, kind, valid_from, valid_until "
                    "FROM facts WHERE turn_id=? AND key_type='slot' ORDER BY fact_seq", (turn_id,),
                ).fetchall()
                supersessions = conn.execute(
                    "SELECT old.entity, old.attribute, old.value AS old_value, new.value AS new_value, "
                    "old.valid_from AS old_valid_from, old.valid_until AS old_valid_until, "
                    "new.valid_from AS new_valid_from FROM facts old JOIN facts new "
                    "ON old.superseded_by=new.fact_id WHERE new.turn_id=? ORDER BY new.fact_seq",
                    (turn_id,),
                ).fetchall()
            if gate is None:
                raise RuntimeError("ingestion completed without a persisted gate event")
            fired = int(gate["signals_fired"])
            reason = (
                f"{fired}/3 signals fired; quarantined by the two-signal gate."
                if gate["decision"] == "quarantine"
                else f"{fired}/3 signals fired; stored with its recorded provenance."
            )
            if state == "completed" and self.embedding_provider is not None:
                from engramOS.embeddings import index_history
                index_history(str(db), HISTORY_ID, self.embedding_provider, gated=False)
            return {
                "turn_id": turn_id,
                "state": state,
                "decision": str(gate["decision"]),
                "signals_fired": fired,
                "signals": {
                    name: {"value": float(gate[column] or 0), "fired": bool(gate[column])}
                    for name, column in (("bridge", "signal_bridge"), ("policy", "signal_policy"), ("source", "signal_source"))
                },
                "reason": reason,
                "facts": [dict(row) for row in facts],
                "supersession_events": [dict(row) for row in supersessions],
                "mode": self.mode,
                "embedding": self.embedding_mode,
            }

    def ask(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.mode != "live":
                raise ValueError("Recorded replay mode is read-only; use the replay controls.")
            question = payload.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("question must be non-empty")
            db = self._ensure_db()
            as_of = payload.get("as_of")
            query = question.strip() + (f" as of {as_of}" if as_of else "")
            started = time.perf_counter()
            try:
                outcome = answer_query(
                    db, HISTORY_ID, query, self.embedding_provider, self.reader,
                    question_date=date.today(), classifier=self.classifier,
                    timeout=300.0, context_version="v3",
                    authorization_timestamp=self.fixed_now,
                )
            except (ConnectionError, TimeoutError, OSError) as exc:
                raise RuntimeError(f"Local model server unavailable: {type(exc).__name__}. Start it with: {START_COMMAND}") from exc
            if outcome.reason_code == "classification_failed":
                raise RuntimeError(f"Local model server classification failed. Start it with: {START_COMMAND}")
            if outcome.reader is not None and outcome.reader.reason_code in {"provider_error", "provider_timeout"}:
                raise RuntimeError(f"Local model server Reader request failed. Start it with: {START_COMMAND}")
            latency = time.perf_counter() - started
            classification = outcome.classification
            retrieval = outcome.retrieval
            evidence: list[dict[str, Any]] = []
            if retrieval is not None:
                for turn in retrieval.turns:
                    evidence.append({
                        "id": f"turn:{turn.turn_id}", "text": turn.raw_content,
                        "source": turn.source_type, "trust": "unverified" if turn.source_type in {"tool", "external_doc"} else "trusted",
                        "quarantined": False, "provenance": [{"turn_id": turn.turn_id, "source": turn.source_type}],
                    })
                for fact in retrieval.facts:
                    source = fact.source_turn
                    evidence.append({
                        "id": f"fact:{fact.fact_id}",
                        "text": f"{fact.fact.get('entity')} / {fact.fact.get('attribute')}: {fact.fact.get('value')}",
                        "source": source.source_type,
                        "trust": "unverified" if source.source_type in {"tool", "external_doc"} else "trusted",
                        "quarantined": bool(fact.fact.get("quarantined", False)),
                        "provenance": [{"turn_id": source.turn_id, "source": source.source_type, "role": source.role}],
                    })
            reader = outcome.reader
            return {
                "status": outcome.status,
                "classification": None if classification is None else {
                    "status": classification.status, "view": classification.view,
                    "route": classification.reason_code,
                    "as_of": None if classification.as_of is None else classification.as_of.timestamp,
                },
                "evidence": evidence,
                "answer": "I don't know." if reader is None else reader.answer,
                "citations": [] if reader is None else [item.value for item in reader.citations],
                "latency_seconds": latency,
                "reader_reason": None if reader is None else reader.reason_code,
                "embedding": self.embedding_mode,
            }

    def timeline(self, slot: str | None) -> dict[str, Any]:
        with self.lock:
            db = self._ensure_db()
            entity, attribute = _parse_slot(slot)
            rows = get_fact_history(str(db), HISTORY_ID, entity=entity, attribute=attribute)
            return {"slot": slot, "items": rows}

    def state(self) -> dict[str, Any]:
        with self.lock:
            db = self._ensure_db()
            current = get_current_facts(str(db), HISTORY_ID)
            with connection(db) as conn:
                quarantined = conn.execute(
                    "SELECT t.turn_id, t.raw_content, t.source_type, t.event_time FROM turns t "
                    "JOIN gate_events e USING(turn_id) WHERE t.history_id=? AND e.decision='quarantine' "
                    "ORDER BY t.event_time, t.turn_idx", (HISTORY_ID,),
                ).fetchall()
                gates = conn.execute(
                    "SELECT e.event_id, e.turn_id, e.signal_bridge, e.signal_policy, e.signal_source, "
                    "e.signals_fired, e.decision, t.raw_content FROM gate_events e JOIN turns t USING(turn_id) "
                    "WHERE t.history_id=? ORDER BY e.event_id", (HISTORY_ID,),
                ).fetchall()
            return {
                "mode": self.mode, "embedding": self.embedding_mode, "models": self.models,
                "current_slots": current,
                "quarantined_turns": [dict(row) for row in quarantined],
                "gate_events": [dict(row) for row in gates],
                "recording_date": None,
            }


def _timestamp(value: Any) -> float:
    if value is None:
        return time.time()
    if type(value) in (int, float):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value)
        except ValueError:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            result = parsed.timestamp()
    else:
        raise ValueError("timestamp must be a number, ISO date/time, or null")
    if not math.isfinite(result):
        raise ValueError("timestamp must be finite")
    return result


def _parse_slot(value: str | None) -> tuple[str | None, str | None]:
    if value is None or not value.strip():
        return None, None
    if "." in value:
        entity, attribute = value.split(".", 1)
        return entity, attribute
    return "user", value


class DemoRequestHandler(BaseHTTPRequestHandler):
    service: DemoService

    def log_message(self, format: str, *args: object) -> None:
        print("[demo] " + format % args)

    def _json(self, status: int, value: Any) -> None:
        data = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path)
        try:
            with self.service.lock:
                if route.path in {"/", "/index.html"}:
                    data = (PROJECT_ROOT / "demo" / "index.html").read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                elif route.path == "/api/state":
                    self._json(200, self.service.state())
                elif route.path == "/api/claims":
                    self._json(200, self.service.claims())
                elif route.path == "/api/recording":
                    self._json(200, self.service.recording())
                elif route.path == "/api/scenario":
                    self._json(200, self.service.scenario())
                elif route.path in {"/README.md", "/LICENSES.md"}:
                    filename = route.path.removeprefix("/")
                    data = (PROJECT_ROOT / filename).read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/markdown; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                elif route.path == "/api/timeline":
                    query = parse_qs(route.query)
                    self._json(200, self.service.timeline(query.get("slot", [None])[0]))
                else:
                    self._json(404, {"error": "not found"})
        except Exception as exc:
            self._json(500, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            with self.service.lock:
                if self.path == "/api/reset":
                    result = self.service.reset()
                elif self.path == "/api/ingest":
                    result = self.service.ingest(payload)
                elif self.path == "/api/ask":
                    result = self.service.ask(payload)
                else:
                    self._json(404, {"error": "not found"})
                    return
            self._json(200, result)
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except RuntimeError as exc:
            self._json(503, {"error": str(exc), "start_command": START_COMMAND})
        except Exception as exc:
            self._json(500, {"error": str(exc)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", choices=("bge", "none"), default="bge")
    parser.add_argument("--mode", choices=("live", "replay"), default="live")
    parser.add_argument("--db-dir", type=Path, default=Path(tempfile.gettempdir()) / "engramos-demo")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fixed-now", help="optional Reader authorization timestamp (epoch or ISO date/time)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        fixed_now = parse_fixed_timestamp(args.fixed_now)
        service = DemoService(mode=args.mode, embedding=args.embedding, db_dir=args.db_dir, fixed_now=fixed_now)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    handler = type("ConfiguredDemoHandler", (DemoRequestHandler,), {"service": service})
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    server.daemon_threads = True
    print(f"EngramOS demo: http://127.0.0.1:{args.port}/ ({args.mode}; embedding={args.embedding})")
    print("Models by stage: " + json.dumps(service.models, sort_keys=True))
    print("Non-Meta model: " + ("BGE-M3 embeddings" if args.embedding == "bge" else "none"))
    if args.fixed_now is not None:
        print(f"Fixed Reader authorization timestamp: {fixed_now}")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
