"""Strict loader for the locked, authored memory evaluation suite."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any, Mapping

from engramOS.extraction import ExtractionPrompt, ExtractionTurn
from engramOS.ingestion import TurnInput
from engramOS.temporal import as_of_timestamp

SCHEMA_VERSION = "engram.authored-memory-suite.v1"
DEFAULT_SUITE_PATH = Path(__file__).with_name("suite.json")
EXPANDED_SUITE_PATH = Path(__file__).with_name("suite_v2.json")
_FACT_FIELDS = frozenset(
    {"entity", "attribute", "value", "cardinality", "kind", "as_of", "replaces", "confidence"}
)


@dataclass(frozen=True, slots=True)
class SuiteStats:
    histories: int
    turns: int
    questions: int
    axes: Mapping[str, int]


def load_suite(path: str | Path = DEFAULT_SUITE_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or malformed authored suite")
    histories = payload.get("histories")
    questions = payload.get("questions")
    if not isinstance(histories, list) or not isinstance(questions, list):
        raise ValueError("suite histories and questions must be arrays")
    history_ids: set[str] = set()
    turn_ids: set[str] = set()
    for history in histories:
        history_id = _required_text(history, "history_id")
        if history_id in history_ids:
            raise ValueError(f"duplicate history ID {history_id!r}")
        history_ids.add(history_id)
        turns = history.get("turns")
        if not isinstance(turns, list):
            raise ValueError("each history must contain a turns array")
        for turn in turns:
            session = _required_text(turn, "session")
            session_date = date.fromisoformat(_required_text(turn, "date"))
            _required_text(turn, "text")
            turn_id = _turn_id(history_id, session)
            if turn_id in turn_ids:
                raise ValueError(f"duplicate turn ID {turn_id!r}")
            turn_ids.add(turn_id)
            if turn.get("source_type", "user_turn") not in (
                "user_turn", "assistant_turn", "system_policy", "tool", "external_doc"
            ):
                raise ValueError(f"unsupported source type in {turn_id!r}")
            facts = turn.get("facts", [])
            if not isinstance(facts, list):
                raise ValueError(f"facts for {turn_id!r} must be an array")
            for fact in facts:
                if not isinstance(fact, dict) or not _FACT_FIELDS.issuperset(fact):
                    raise ValueError(f"fixture fact for {turn_id!r} has unknown fields")
                for field in ("entity", "attribute", "value", "cardinality", "kind"):
                    _required_text(fact, field)
                if fact.get("as_of") is not None:
                    as_of = date.fromisoformat(str(fact["as_of"]))
                    if as_of > session_date:
                        raise ValueError(f"fixture fact for {turn_id!r} is future-dated")
    question_ids: set[str] = set()
    attack_question_ids: set[str] = set()
    for question in questions:
        question_id = _required_text(question, "question_id")
        if question_id in question_ids:
            raise ValueError(f"duplicate question ID {question_id!r}")
        question_ids.add(question_id)
        if question.get("history_id") not in history_ids:
            raise ValueError(f"question {question_id!r} references an unknown history")
        _required_text(question, "axis")
        _required_text(question, "query")
        if question.get("expected_label") not in ("answer_contains", "abstain"):
            raise ValueError(f"question {question_id!r} has unsupported expected label")
        if question["expected_label"] == "answer_contains":
            _required_text(question, "expected_answer")
        for field in ("attack_turn_id", "benign_turn_id"):
            turn_id = question.get(field)
            if turn_id is not None and (
                not isinstance(turn_id, str) or turn_id not in turn_ids
            ):
                raise ValueError(f"question {question_id!r} references an unknown {field}")
        if question.get("attack_turn_id"):
            attack_question_ids.add(question_id)
    attack_claims = payload.get("attack_claims", {})
    if not isinstance(attack_claims, dict):
        raise ValueError("attack_claims must be an object keyed by question ID")
    if not set(attack_claims).issubset(attack_question_ids):
        raise ValueError("attack_claims may only reference attack questions")
    for question_id, claim in attack_claims.items():
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError(f"attack claim for {question_id!r} must be non-empty text")
    return payload


def suite_stats(suite: Mapping[str, Any]) -> SuiteStats:
    turns = [turn for history in suite["histories"] for turn in history["turns"]]
    axes: dict[str, int] = {}
    for question in suite["questions"]:
        axis = str(question["axis"])
        axes[axis] = axes.get(axis, 0) + 1
    return SuiteStats(len(suite["histories"]), len(turns), len(suite["questions"]), axes)


def iter_history_turns(history: Mapping[str, Any]):
    """Yield one immutable turn input at a time in authored chronological order."""
    history_id = str(history["history_id"])
    for turn in history["turns"]:
        session = str(turn["session"])
        session_date = str(turn["date"])
        yield TurnInput(
            history_id=history_id,
            session_id=session,
            turn_idx=0,
            role="user",
            source_type=str(turn.get("source_type", "user_turn")),
            raw_content=str(turn["text"]),
            session_unix_ts=as_of_timestamp(session_date),
            session_date=session_date,
            turn_id=_turn_id(history_id, session),
        )


class FixtureFactExtractor:
    """Return fixed Section A payloads; it does not infer facts from text."""

    def __init__(self, history: Mapping[str, Any]) -> None:
        history_id = str(history["history_id"])
        self._payloads: dict[str, list[dict[str, object]]] = {}
        for turn in history["turns"]:
            turn_id = _turn_id(history_id, str(turn["session"]))
            self._payloads[turn_id] = [
                {
                    "session_id": str(turn["session"]),
                    "turn_idx": 0,
                    "entity": fact["entity"],
                    "attribute": fact["attribute"],
                    "value": fact["value"],
                    "cardinality": fact["cardinality"],
                    "kind": fact["kind"],
                    "as_of": fact.get("as_of"),
                    "replaces": fact.get("replaces"),
                    "confidence": fact.get("confidence", 1.0),
                }
                for fact in turn.get("facts", [])
            ]

    def extract(self, turn: ExtractionTurn, prompt: ExtractionPrompt) -> str:
        del prompt
        return json.dumps(self._payloads.get(turn.turn_id, []), ensure_ascii=False)


def _turn_id(history_id: str, session: str) -> str:
    return f"{history_id}:{session}:0"


def _required_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value
