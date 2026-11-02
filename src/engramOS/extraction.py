"""Provider-independent extraction prompts, validation, and fact types.

No model client is implemented here. A future adapter must use the project's
cache and budget manager before making any live generation request.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import logging
import math
import re
from typing import Protocol, Sequence

from .normalize import normalize_slot_name, normalize_value

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CARDINALITIES = frozenset({"single", "multi"})
_KINDS = frozenset({"state", "preference", "event", "constraint", "other"})
_FACT_FIELDS = frozenset(
    {
        "session_id",
        "turn_idx",
        "entity",
        "attribute",
        "value",
        "cardinality",
        "kind",
        "as_of",
        "replaces",
        "confidence",
    }
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExtractionTurn:
    """Immutable source-turn context supplied to prompt builders and extractors."""

    turn_id: str
    history_id: str
    session_id: str
    session_date: str
    session_unix_ts: float
    turn_idx: int
    role: str
    raw_content: str
    source_type: str
    event_time: float


@dataclass(frozen=True, slots=True)
class ExtractionPrompt:
    system: str
    user: str


@dataclass(frozen=True, slots=True)
class ExtractedFact:
    """One validated fact using the exact fields from ``prompts.md``.

    ``fact_seq`` is assigned from the response array position; it is not a
    model-output field in the specified extraction schema.
    """

    session_id: str
    turn_idx: int
    entity: str
    attribute: str
    value: str
    cardinality: str
    kind: str
    as_of: str | None
    replaces: str | None
    confidence: float
    fact_seq: int

    def to_prompt_object(self) -> dict[str, object]:
        """Return the original response schema without the internal sequence."""
        return {
            "session_id": self.session_id,
            "turn_idx": self.turn_idx,
            "entity": self.entity,
            "attribute": self.attribute,
            "value": self.value,
            "cardinality": self.cardinality,
            "kind": self.kind,
            "as_of": self.as_of,
            "replaces": self.replaces,
            "confidence": self.confidence,
        }


class FactExtractor(Protocol):
    """Injectable extraction interface; implementations return raw JSON text."""

    def extract(self, turn: ExtractionTurn, prompt: ExtractionPrompt) -> str:
        """Return a raw response for the provided turn and exact prompt."""


class ExtractionResponseError(ValueError):
    """Raised when an extraction response does not match the specified schema."""


class TransientExtractionError(RuntimeError):
    """Extractor error explicitly marked safe to retry by its adapter."""


class ExtractionExecutionError(RuntimeError):
    """Raised when an extractor fails after the configured bounded attempts."""


_SYSTEM_TEMPLATE = """You extract durable memory facts from chat sessions between a user and an assistant.
Return a JSON array only, no prose.

For each fact about the USER or the user's project/life, emit an object:
{
  "session_id": "<session_id from the header>",
  "turn_idx": <integer index of the turn the fact comes from>,
  "entity": "<lowercase snake_case noun, e.g. user, database, auth_service, sister>",
  "attribute": "<lowercase snake_case, e.g. job_title, port, token_ttl, home_city>",
  "value": "<short value, keep numbers/names/units exactly as written>",
  "cardinality": "single" | "multi",
  "kind": "state" | "preference" | "event" | "constraint" | "other",
  "as_of": "<ISO date YYYY-MM-DD if the turn explicitly states when the fact became true in the past, else null>",
  "replaces": "<old value if the turn explicitly says it changed, else null>",
  "confidence": <0.0-1.0>
}

Rules:
- "single" = only one value can be true at a time (job, city, port, current plan, diet). "multi" = values accumulate (hobbies, books read, pets, visited cities).
- Emit facts in the exact chronological order they appear in the turns.
- Only extract facts the text actually states. Never infer or guess.
- If the user says something changed ("switched from X to Y", "moved to Z", "I no longer..."), emit the NEW value and set "replaces".
- Reuse an existing slot name (entity, attribute) when it refers to the same concept. Existing slots for this history:
  {existing_slots}
- Skip chitchat, greetings, questions, and generic assistant explanations.
- Keep numbers, names, and dates exactly as written."""


def build_extraction_prompt(
    turn: ExtractionTurn,
    existing_slots: Sequence[tuple[str, str]],
) -> ExtractionPrompt:
    """Build the Section A prompt without changing its extraction instructions."""
    if not isinstance(turn, ExtractionTurn):
        raise TypeError("turn must be an ExtractionTurn")
    slots = sorted(set(existing_slots))
    slot_text = "\n".join(f"  - {entity}.{attribute}" for entity, attribute in slots)
    if not slot_text:
        slot_text = "  (none)"
    system = _SYSTEM_TEMPLATE.replace("{existing_slots}", slot_text)
    user = (
        f"=== Session {turn.session_id} | Date: {turn.session_date} ===\n"
        f"[{turn.turn_idx}] {turn.role}: {turn.raw_content}"
    )
    return ExtractionPrompt(system=system, user=user)


def validate_extraction_response(
    response: str,
    turn: ExtractionTurn,
) -> tuple[ExtractedFact, ...]:
    """Parse and validate the complete response before any temporal writes."""
    if not isinstance(turn, ExtractionTurn):
        raise TypeError("turn must be an ExtractionTurn")
    if not isinstance(response, str):
        raise ExtractionResponseError("extraction response must be JSON text")
    try:
        parsed = json.loads(response, object_pairs_hook=_reject_duplicate_keys)
    except ExtractionResponseError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError, ValueError) as exc:
        raise ExtractionResponseError("extraction response is malformed JSON") from exc
    if not isinstance(parsed, list):
        raise ExtractionResponseError("extraction response must be a JSON array")

    facts: list[ExtractedFact] = []
    source_session_date: date | None = None
    for sequence, item in enumerate(parsed):
        prefix = f"fact at array index {sequence}"
        if not isinstance(item, dict):
            raise ExtractionResponseError(f"{prefix} must be an object")
        keys = frozenset(item)
        if keys != _FACT_FIELDS:
            missing = sorted(_FACT_FIELDS - keys)
            extra = sorted(keys - _FACT_FIELDS)
            raise ExtractionResponseError(
                f"{prefix} has invalid fields (missing={missing}, extra={extra})"
            )

        session_id = item["session_id"]
        if not isinstance(session_id, str) or session_id != turn.session_id:
            raise ExtractionResponseError(f"{prefix} references an unrelated session")
        turn_idx = item["turn_idx"]
        if isinstance(turn_idx, bool) or not isinstance(turn_idx, int):
            raise ExtractionResponseError(f"{prefix}.turn_idx must be an integer")
        if turn_idx != turn.turn_idx:
            raise ExtractionResponseError(f"{prefix} references an unrelated source turn")

        entity = _validate_identifier(item["entity"], f"{prefix}.entity")
        attribute = _validate_identifier(item["attribute"], f"{prefix}.attribute")
        normalized_entity, normalized_attribute = normalize_slot_name(entity, attribute)
        if (normalized_entity, normalized_attribute) != (entity, attribute):
            raise ExtractionResponseError(
                f"{prefix} entity and attribute must already be normalized snake_case"
            )

        value = item["value"]
        if not isinstance(value, str) or not value.strip() or not normalize_value(value):
            raise ExtractionResponseError(f"{prefix}.value must be a non-empty fact value")
        cardinality = item["cardinality"]
        if not isinstance(cardinality, str) or cardinality not in _CARDINALITIES:
            raise ExtractionResponseError(f"{prefix}.cardinality is invalid")
        kind = item["kind"]
        if not isinstance(kind, str) or kind not in _KINDS:
            raise ExtractionResponseError(f"{prefix}.kind is invalid")

        as_of = item["as_of"]
        if as_of is not None:
            if not isinstance(as_of, str) or not _ISO_DATE.fullmatch(as_of):
                raise ExtractionResponseError(f"{prefix}.as_of must be null or YYYY-MM-DD")
            try:
                as_of_date = date.fromisoformat(as_of)
            except ValueError as exc:
                raise ExtractionResponseError(f"{prefix}.as_of is not a valid date") from exc
            if source_session_date is None:
                try:
                    source_session_date = date.fromisoformat(turn.session_date)
                except (TypeError, ValueError) as exc:
                    raise ValueError("source turn session_date must be a valid ISO date") from exc
                if source_session_date.isoformat() != turn.session_date:
                    raise ValueError("source turn session_date must use YYYY-MM-DD format")
            if as_of_date > source_session_date:
                raise ExtractionResponseError(
                    f"{prefix}.as_of {as_of} is later than source session date "
                    f"{turn.session_date}"
                )

        replaces = item["replaces"]
        if replaces is not None and (
            not isinstance(replaces, str) or not replaces.strip()
        ):
            raise ExtractionResponseError(f"{prefix}.replaces must be null or non-empty text")

        confidence = item["confidence"]
        try:
            confidence_value = float(confidence)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ExtractionResponseError(
                f"{prefix}.confidence must be a finite number in [0, 1]"
            ) from exc
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence_value)
            or not 0.0 <= confidence_value <= 1.0
        ):
            raise ExtractionResponseError(f"{prefix}.confidence must be a finite number in [0, 1]")

        facts.append(
            ExtractedFact(
                session_id=session_id,
                turn_idx=turn_idx,
                entity=entity,
                attribute=attribute,
                value=value,
                cardinality=cardinality,
                kind=kind,
                as_of=as_of,
                replaces=replaces,
                confidence=confidence_value,
                fact_seq=sequence,
            )
        )
    return tuple(facts)


def serialize_extraction(facts: Sequence[ExtractedFact]) -> str:
    """Serialize a fully validated extraction deterministically for recovery."""
    return json.dumps(
        [fact.to_prompt_object() for fact in facts],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def extract_with_bounded_retries(
    extractor: FactExtractor,
    turn: ExtractionTurn,
    prompt: ExtractionPrompt,
    *,
    max_attempts: int = 1,
) -> str:
    """Retry only explicitly transient extractor errors, never malformed data."""
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    for attempt in range(1, max_attempts + 1):
        try:
            response = extractor.extract(turn, prompt)
        except TransientExtractionError as exc:
            if attempt == max_attempts:
                raise ExtractionExecutionError(
                    f"extractor failed after {max_attempts} attempts"
                ) from exc
            continue
        except Exception as exc:
            raise ExtractionExecutionError("extractor failed with a non-retryable error") from exc
        if not isinstance(response, str):
            raise ExtractionResponseError("extractor must return raw JSON text")
        return response
    raise AssertionError("bounded retry loop exited unexpectedly")


def extract_and_validate_with_bounded_retries(
    extractor: FactExtractor,
    turn: ExtractionTurn,
    prompt: ExtractionPrompt,
    *,
    max_attempts: int = 2,
) -> tuple[ExtractedFact, ...]:
    """Retry transient failures and invalid model responses within a fixed bound.

    Malformed JSON or schema violations may be corrected by a fresh generation,
    so they are retried up to ``max_attempts``. Invalid adapter return types and
    non-transient execution errors fail immediately. No database write occurs
    until a complete response passes validation.
    """
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    last_transient: TransientExtractionError | None = None
    last_invalid_response: ExtractionResponseError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = extractor.extract(turn, prompt)
        except TransientExtractionError as exc:
            last_transient = exc
            logger.warning(
                "Transient extraction failure for turn %s on attempt %d/%d (%s)",
                turn.turn_id,
                attempt,
                max_attempts,
                type(exc).__name__,
            )
            if attempt == max_attempts:
                raise ExtractionExecutionError(
                    f"extractor failed after {max_attempts} attempts"
                ) from exc
            continue
        except Exception as exc:
            raise ExtractionExecutionError("extractor failed with a non-retryable error") from exc

        if not isinstance(response, str):
            raise ExtractionResponseError("extractor must return raw JSON text")
        try:
            return validate_extraction_response(response, turn)
        except ExtractionResponseError as exc:
            last_invalid_response = exc
            logger.warning(
                "Invalid extraction response for turn %s on attempt %d/%d (%s)",
                turn.turn_id,
                attempt,
                max_attempts,
                type(exc).__name__,
            )
            if attempt == max_attempts:
                raise ExtractionResponseError(
                    f"extraction response remained invalid after {max_attempts} attempts: {exc}"
                ) from exc

    if last_invalid_response is not None:
        raise last_invalid_response
    if last_transient is not None:
        raise ExtractionExecutionError(
            f"extractor failed after {max_attempts} attempts"
        ) from last_transient
    raise AssertionError("bounded extraction and validation loop exited unexpectedly")


def _validate_identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ExtractionResponseError(f"{field} must be a lowercase snake_case identifier")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ExtractionResponseError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result
