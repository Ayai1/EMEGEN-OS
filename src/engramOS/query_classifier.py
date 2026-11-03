"""Provider-independent query classification and temporal input validation.

No model client is provided. An optional classifier adapter can be injected,
but its JSON response is treated as untrusted and validated before it is
returned to retrieval code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
import logging
import math
import re
from typing import Literal, Protocol

from .temporal import FactView, TemporalBoundary, as_of_timestamp

QUERY_CLASSIFICATION_SCHEMA_VERSION = "engram.query-classification.v1"
logger = logging.getLogger(__name__)

ClassificationStatus = Literal["classified", "clarification_required", "unsupported"]
ReasonCode = Literal[
    "present_state",
    "explicit_as_of",
    "before_boundary",
    "relative_as_of",
    "historical_changes",
    "explicit_history_range",
    "provider_classification",
    "ambiguous_month",
    "ambiguous_temporal_intent",
    "invalid_temporal_expression",
    "classifier_unavailable",
    "knowledge_time_unsupported",
    "future_as_of_unsupported",
    "unsupported_temporal_expression",
]


class QueryClassificationError(RuntimeError):
    """Raised when classification cannot safely produce a retrieval route."""


class QueryClassificationOutputError(QueryClassificationError):
    """Raised when an injected classifier returns malformed or unsafe output."""


class QueryClassificationProvider(Protocol):
    """Injectable provider contract; implementations must enforce ``timeout``."""

    def classify(self, query: str, *, question_date: date, timeout: float) -> str:
        """Return JSON matching the versioned contract in ``prompts.md``."""


@dataclass(frozen=True, slots=True)
class TemporalRange:
    """Optional bounded interval for a history query."""

    start: TemporalBoundary | None = None
    end: TemporalBoundary | None = None

    def __post_init__(self) -> None:
        if self.start is None and self.end is None:
            raise ValueError("a temporal range must have at least one endpoint")
        if self.start is not None and not isinstance(self.start, TemporalBoundary):
            raise TypeError("range start must be a TemporalBoundary or None")
        if self.end is not None and not isinstance(self.end, TemporalBoundary):
            raise TypeError("range end must be a TemporalBoundary or None")
        if self.start is not None and self.end is not None:
            if self.start.timestamp > self.end.timestamp:
                raise ValueError("history range start is after its end")
            if self.start.timestamp == self.end.timestamp and not (
                self.start.inclusive and self.end.inclusive
            ):
                raise ValueError("history range is empty or contradictory")


@dataclass(frozen=True, slots=True)
class QueryClassification:
    """Validated routing decision; non-classified results cannot reach retrieval."""

    status: ClassificationStatus
    view: FactView | None
    question_date: date
    reason_code: ReasonCode
    as_of: TemporalBoundary | None = None
    history_range: TemporalRange | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.question_date, date) or isinstance(self.question_date, datetime):
            raise TypeError("question_date must be a normalized date")
        if not isinstance(self.status, str) or self.status not in _REASONS_BY_STATUS:
            raise ValueError("classification status is invalid")
        if not isinstance(self.reason_code, str) or self.reason_code not in _REASONS_BY_STATUS[self.status]:
            raise ValueError("reason_code is invalid for the classification status")
        if self.status != "classified":
            if self.view is not None or self.as_of is not None or self.history_range is not None:
                raise ValueError("non-classified results cannot contain retrieval boundaries")
            return
        if self.view not in ("current", "as_of", "history"):
            raise ValueError("classified result requires a supported view")
        reason_by_view = {
            "current": frozenset({"present_state", "provider_classification"}),
            "as_of": frozenset(
                {"explicit_as_of", "before_boundary", "relative_as_of", "provider_classification"}
            ),
            "history": frozenset(
                {"historical_changes", "explicit_history_range", "provider_classification"}
            ),
        }
        if self.reason_code not in reason_by_view[self.view]:
            raise ValueError("reason_code is inconsistent with the classified view")
        if self.as_of is not None and not isinstance(self.as_of, TemporalBoundary):
            raise TypeError("as_of must be a TemporalBoundary or None")
        if self.history_range is not None and not isinstance(self.history_range, TemporalRange):
            raise TypeError("history_range must be a TemporalRange or None")
        if self.view == "current" and (self.as_of is not None or self.history_range is not None):
            raise ValueError("current view cannot include temporal boundaries")
        if self.view == "as_of" and (self.as_of is None or self.history_range is not None):
            raise ValueError("as_of view requires exactly one as_of boundary")
        if self.view == "history" and self.as_of is not None:
            raise ValueError("history view cannot include an as_of boundary")

    @property
    def ready_for_retrieval(self) -> bool:
        return self.status == "classified" and self.view is not None


@dataclass(frozen=True, slots=True)
class _Mention:
    start: int
    end: int
    boundary: TemporalBoundary | None
    kind: Literal["date", "month", "relative", "relative_period"]
    month_start: date | None = None


_ISO_DATETIME = re.compile(
    r"(?<!\d)\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}"
    r"(?::\d{2}(?:\.\d{1,6})?)?(?:[Zz]|[+-]\d{2}:\d{2})?(?!\w)"
)
_ISO_DATE = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?![\dTt])")
_DAY_MONTH_YEAR = re.compile(
    r"(?<!\w)(?P<day>\d{1,2})\s+(?P<month>"
    r"January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    r"\s+(?P<year>\d{4})(?!\d)",
    re.IGNORECASE,
)
_MONTH_DAY_YEAR = re.compile(
    r"(?<!\w)(?P<month>"
    r"January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    r"\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})(?!\d)",
    re.IGNORECASE,
)
_MONTH_NAME = (
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)
_MONTH_YEAR = re.compile(
    rf"(?<!\w)(?P<month>{_MONTH_NAME})\s+(?P<year>\d{{4}})(?!\d)",
    re.IGNORECASE,
)
_RELATIVE_DAY = re.compile(r"\b(today|yesterday|tomorrow)\b", re.IGNORECASE)
_DAYS_AGO = re.compile(r"\b(?P<count>\d+)\s+days?\s+ago\b", re.IGNORECASE)
_RELATIVE_PERIOD = re.compile(r"\b(last|this|past)\s+(week|month|year)\b", re.IGNORECASE)
_MONTHS = {
    name.casefold(): number
    for number, name in enumerate(
        ("January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"),
        start=1,
    )
}
_MONTHS.update({name[:3].casefold(): number for name, number in _MONTHS.items()})
_BEFORE_WORD = re.compile(r"\b(?:before|prior\s+to)\s*$", re.IGNORECASE)
_AFTER_WORD = re.compile(r"\b(?:after|since)\s*$", re.IGNORECASE)
_THROUGH_WORD = re.compile(r"\bthrough\s*$", re.IGNORECASE)
_KNOWLEDGE_TIME = re.compile(
    r"\b(?:what\s+(?:did|could)\s+(?:i|we|the\s+system)\s+know|"
    r"what\s+was\s+known\s+to\s+(?:me|us|the\s+system))\b.*\b(?:then|at\s+the\s+time|on|in)\b",
    re.IGNORECASE,
)
_CHANGE_INTENT = re.compile(
    r"\bhow\s+(?:did|has|have)\s+(?:my|your|the\s+user's)\b.{0,120}"
    r"\b(?:change|changed|evolve|evolved)\b"
    r"|\bhow\b.{0,120}\b(?:changed|evolved)\b"
    r"|\bwhat\s+(?:has|have|did)\b.{0,160}\bchanged\b"
    r"|\b(?:show|list|summarize|summarise|describe|trace)\b.{0,120}"
    r"\b(?:the\s+)?(?:history|changes?|timeline|evolution)\b"
    r"|\b(?:history|timeline|evolution)\s+(?:of|for)\b",
    re.IGNORECASE,
)
_CURRENT_EXPLICIT = re.compile(
    r"\b(?:current|currently|right\s+now|latest|presently)\b", re.IGNORECASE
)
_CURRENT_QUESTION = re.compile(
    r"^\s*(?:what|who|where|which|when|how\s+many|how\s+much)\s+"
    r"(?:is|are|do|does|has|have|can)\b",
    re.IGNORECASE,
)
_TEMPORAL_WORDS = re.compile(
    r"\b(?:before|after|previously|prior|used\s+to|formerly|historical|"
    r"past|earlier|changed|change|timeline|ever)\b",
    re.IGNORECASE,
)
_JSON_FIELDS = frozenset(
    {"schema_version", "status", "view", "as_of", "history_range", "reason_code"}
)
_ENDPOINT_FIELDS = frozenset({"value", "inclusive"})
_RANGE_FIELDS = frozenset({"start", "end"})
_REASONS_BY_STATUS: dict[str, frozenset[str]] = {
    "classified": frozenset(
        {
            "present_state", "explicit_as_of", "before_boundary", "relative_as_of",
            "historical_changes", "explicit_history_range", "provider_classification",
        }
    ),
    "clarification_required": frozenset(
        {
            "ambiguous_month", "ambiguous_temporal_intent", "invalid_temporal_expression",
            "classifier_unavailable",
        }
    ),
    "unsupported": frozenset(
        {
            "knowledge_time_unsupported", "future_as_of_unsupported",
            "unsupported_temporal_expression",
        }
    ),
}
_NON_ROUTING_REASONS = frozenset({
    "ambiguous_month",
    "ambiguous_temporal_intent",
    "invalid_temporal_expression",
    "classifier_unavailable",
    "knowledge_time_unsupported",
    "future_as_of_unsupported",
    "unsupported_temporal_expression",
})


def classify_query(
    query: str,
    *,
    question_date: date | datetime | str,
    classifier: QueryClassificationProvider | None = None,
    timeout: float = 5.0,
) -> QueryClassification:
    """Classify a query without allowing uncertain or invalid routes to default current.

    Recognized deterministic patterns are handled locally. Other queries can
    use an injected provider; without one, the result requests clarification.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    qdate = _normalize_question_date(question_date)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("timeout must be a positive finite number")
    if not math.isfinite(float(timeout)):
        raise ValueError("timeout must be a positive finite number")

    query_text = query
    if _KNOWLEDGE_TIME.search(query_text):
        return _non_routed(qdate, "unsupported", "knowledge_time_unsupported")

    try:
        mentions = _extract_mentions(query_text, qdate)
    except (TypeError, ValueError, OverflowError):
        return _non_routed(qdate, "clarification_required", "invalid_temporal_expression")

    if any(
        mention.kind == "relative"
        and mention.boundary is not None
        and datetime.fromtimestamp(mention.boundary.timestamp, timezone.utc).date() > qdate
        for mention in mentions
    ):
        return _non_routed(qdate, "unsupported", "future_as_of_unsupported")

    if any(mention.kind == "relative_period" for mention in mentions):
        return _non_routed(qdate, "clarification_required", "ambiguous_temporal_intent")

    month_mentions = [mention for mention in mentions if mention.kind == "month"]
    if month_mentions:
        if _CHANGE_INTENT.search(query_text):
            classification = _classify_history_month(query_text, qdate, month_mentions)
            if classification is not None:
                return classification
        if len(month_mentions) == 1 and _BEFORE_WORD.search(
            query_text[: month_mentions[0].start]
        ) and _is_point_question(query_text):
            month_start = month_mentions[0].month_start
            assert month_start is not None
            return _as_of(
                qdate,
                TemporalBoundary(as_of_timestamp(month_start)),
                inclusive=False,
                reason="before_boundary",
            )
        return _non_routed(qdate, "clarification_required", "ambiguous_month")

    history_intent = _CHANGE_INTENT.search(query_text) is not None
    if history_intent:
        history = _classify_history(query_text, qdate, mentions)
        if history is not None:
            return history
        return _non_routed(qdate, "clarification_required", "ambiguous_temporal_intent")

    if mentions:
        if _CURRENT_EXPLICIT.search(query_text):
            return _non_routed(qdate, "clarification_required", "ambiguous_temporal_intent")
        before = _before_mention(query_text, mentions)
        if before is not None and _is_point_question(query_text):
            assert before.boundary is not None
            return _as_of(qdate, before.boundary, inclusive=False, reason="before_boundary")
        if len(mentions) == 1 and _is_point_question(query_text):
            mention = mentions[0]
            if mention.boundary is not None:
                return _as_of(
                    qdate,
                    mention.boundary,
                    inclusive=True,
                    reason="relative_as_of" if mention.kind == "relative" else "explicit_as_of",
                )
        if _CURRENT_EXPLICIT.search(query_text) or _CURRENT_QUESTION.search(query_text):
            return _non_routed(qdate, "clarification_required", "ambiguous_temporal_intent")

    if not mentions and not _TEMPORAL_WORDS.search(query_text):
        if _CURRENT_EXPLICIT.search(query_text) or _CURRENT_QUESTION.search(query_text):
            return _classified(qdate, "current", "present_state")

    if classifier is None:
        reason: ReasonCode = (
            "ambiguous_temporal_intent"
            if _TEMPORAL_WORDS.search(query_text)
            else "classifier_unavailable"
        )
        logger.info("query classification requires clarification (%s)", reason)
        return _non_routed(qdate, "clarification_required", reason)

    try:
        raw = classifier.classify(query_text, question_date=qdate, timeout=float(timeout))
    except Exception as exc:
        logger.warning("query classification provider failed (%s)", type(exc).__name__)
        raise QueryClassificationError("query classification provider failed") from exc
    try:
        result = _parse_provider_result(raw, qdate)
        if (
            result.status != "classified"
            and result.reason_code in _NON_ROUTING_REASONS
            and not mentions
            and not _TEMPORAL_WORDS.search(query_text)
        ):
            # A provider cannot turn an unqualified present-state question into an
            # unsupported/ambiguous temporal query. Explicit dates, relative dates,
            # knowledge-time questions, and temporal words are handled above or
            # remain subject to provider-output validation below.
            return _classified(qdate, "current", "present_state")
        _validate_against_query(result, query_text, qdate, mentions)
    except QueryClassificationOutputError as exc:
        logger.warning("query classification output rejected (%s)", type(exc).__name__)
        raise
    return result


def _normalize_question_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        try:
            return datetime.fromtimestamp(as_of_timestamp(value), timezone.utc).date()
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("question_date must be timezone-aware when it is a datetime") from exc
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            normalized = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("question_date must use YYYY-MM-DD format") from exc
        if normalized.isoformat() != value:
            raise ValueError("question_date must use YYYY-MM-DD format")
        return normalized
    raise TypeError("question_date must be a date, timezone-aware datetime, or ISO date")


def _extract_mentions(query: str, question_date: date) -> list[_Mention]:
    candidates: list[_Mention] = []
    patterns = (
        (_ISO_DATETIME, "datetime"),
        (_ISO_DATE, "date"),
        (_DAY_MONTH_YEAR, "natural_date"),
        (_MONTH_DAY_YEAR, "natural_date"),
        (_MONTH_YEAR, "month"),
    )
    for pattern, kind in patterns:
        for match in pattern.finditer(query):
            raw = match.group(0)
            if kind == "month":
                month = _month_number(match.group("month"))
                year = int(match.group("year"))
                candidates.append(
                    _Mention(match.start(), match.end(), None, "month", date(year, month, 1))
                )
                continue
            if kind == "natural_date":
                month = _month_number(match.group("month"))
                day = int(match.group("day"))
                year = int(match.group("year"))
                raw = date(year, month, day).isoformat()
            try:
                boundary = TemporalBoundary(as_of_timestamp(raw))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("query contains an invalid date or timestamp") from exc
            candidates.append(
                _Mention(match.start(), match.end(), boundary, "date")
            )

    for match in _RELATIVE_DAY.finditer(query):
        relative = match.group(1).casefold()
        offset = {"today": 0, "yesterday": -1, "tomorrow": 1}[relative]
        day = question_date + timedelta(days=offset)
        candidates.append(
            _Mention(match.start(), match.end(), TemporalBoundary(as_of_timestamp(day)), "relative")
        )
    for match in _DAYS_AGO.finditer(query):
        count = int(match.group("count"))
        day = question_date - timedelta(days=count)
        candidates.append(
            _Mention(match.start(), match.end(), TemporalBoundary(as_of_timestamp(day)), "relative")
        )
    for match in _RELATIVE_PERIOD.finditer(query):
        candidates.append(_Mention(match.start(), match.end(), None, "relative_period"))

    candidates.sort(key=lambda item: (item.start, -(item.end - item.start)))
    selected: list[_Mention] = []
    for candidate in candidates:
        if selected and candidate.start < selected[-1].end:
            if candidate.end > selected[-1].end:
                selected[-1] = candidate
            continue
        selected.append(candidate)
    return selected


def _month_number(value: str) -> int:
    month = _MONTHS.get(value.casefold())
    if month is None:
        raise ValueError("unrecognized month name")
    return month


def _classify_history_month(
    query: str, qdate: date, mentions: list[_Mention]
) -> QueryClassification | None:
    if len(mentions) != 1 or mentions[0].month_start is None:
        return None
    month_start = mentions[0].month_start
    year, month = month_start.year, month_start.month
    if _BEFORE_WORD.search(query[: mentions[0].start]):
        end = TemporalBoundary(as_of_timestamp(month_start), inclusive=False)
        return QueryClassification(
            "classified", "history", qdate, "explicit_history_range",
            history_range=TemporalRange(end=end),
        )
    next_month = date(year + (month == 12), 1 if month == 12 else month + 1, 1)
    start = TemporalBoundary(as_of_timestamp(month_start), inclusive=True)
    end = TemporalBoundary(as_of_timestamp(next_month), inclusive=False)
    return QueryClassification(
        "classified", "history", qdate, "explicit_history_range",
        history_range=TemporalRange(start=start, end=end),
    )


def _classify_history(
    query: str, qdate: date, mentions: list[_Mention]
) -> QueryClassification | None:
    if not mentions:
        return _classified(qdate, "history", "historical_changes")
    if any(mention.boundary is None for mention in mentions):
        return None
    if len(mentions) > 2:
        return None
    if len(mentions) == 2:
        first = mentions[0]
        second = mentions[1]
        start_boundary = first.boundary
        end_boundary = second.boundary
        if (
            start_boundary is None
            or end_boundary is None
            or start_boundary.timestamp > end_boundary.timestamp
        ):
            return _non_routed(qdate, "clarification_required", "invalid_temporal_expression")
        connector = query[first.end:second.start]
        prefix = query[:first.start]
        if re.search(r"\bfrom\s*$", prefix, re.IGNORECASE) and re.search(
            r"\b(?:through|until)\b", connector, re.IGNORECASE
        ):
            start = TemporalBoundary(start_boundary.timestamp, inclusive=True)
            through = re.search(r"\bthrough\b", connector, re.IGNORECASE)
            end = TemporalBoundary(end_boundary.timestamp, inclusive=through is not None)
        elif re.search(r"\bafter\s*$", prefix, re.IGNORECASE) and re.search(
            r"\bbefore\b", connector, re.IGNORECASE
        ):
            start = TemporalBoundary(start_boundary.timestamp, inclusive=False)
            end = TemporalBoundary(end_boundary.timestamp, inclusive=False)
        else:
            return _non_routed(qdate, "clarification_required", "ambiguous_temporal_intent")
        return QueryClassification(
            "classified", "history", qdate, "explicit_history_range",
            history_range=TemporalRange(start=start, end=end),
        )
    mention = mentions[0]
    boundary = mention.boundary
    if boundary is None:
        return None
    before = _before_mention(query, mentions)
    if before is not None:
        end = TemporalBoundary(boundary.timestamp, inclusive=False)
        return QueryClassification(
            "classified", "history", qdate, "explicit_history_range",
            history_range=TemporalRange(end=end),
        )
    after_match = _AFTER_WORD.search(query[: mention.start])
    if after_match:
        start = TemporalBoundary(
            boundary.timestamp,
            inclusive=not after_match.group(0).casefold().startswith("after"),
        )
        return QueryClassification(
            "classified", "history", qdate, "explicit_history_range",
            history_range=TemporalRange(start=start),
        )
    through_match = _THROUGH_WORD.search(query[: mention.start])
    if through_match:
        end = TemporalBoundary(boundary.timestamp, inclusive=True)
        return QueryClassification(
            "classified", "history", qdate, "explicit_history_range",
            history_range=TemporalRange(end=end),
        )
    if mention.kind == "relative":
        next_day = datetime.fromtimestamp(boundary.timestamp, timezone.utc).date() + timedelta(days=1)
        start = TemporalBoundary(boundary.timestamp, inclusive=True)
        end = TemporalBoundary(as_of_timestamp(next_day), inclusive=False)
        return QueryClassification(
            "classified", "history", qdate, "explicit_history_range",
            history_range=TemporalRange(start=start, end=end),
        )
    return None


def _before_mention(query: str, mentions: list[_Mention]) -> _Mention | None:
    if len(mentions) != 1:
        return None
    mention = mentions[0]
    prefix = query[: mention.start]
    return mention if _BEFORE_WORD.search(prefix) else None


def _is_point_question(query: str) -> bool:
    return bool(
        re.match(r"\s*(?:what|where|who|which|when|how\s+(?:many|much))\b", query, re.I)
        and re.search(r"\b(?:is|are|was|were|do|does|did|has|have|had|will)\b", query, re.I)
    )


def _as_of(
    qdate: date,
    boundary: TemporalBoundary,
    *,
    inclusive: bool,
    reason: ReasonCode,
) -> QueryClassification:
    endpoint = TemporalBoundary(boundary.timestamp, inclusive=inclusive)
    query_day = datetime.fromtimestamp(endpoint.timestamp, timezone.utc).date()
    if query_day > qdate:
        return _non_routed(qdate, "unsupported", "future_as_of_unsupported")
    return QueryClassification("classified", "as_of", qdate, reason, as_of=endpoint)


def _classified(qdate: date, view: FactView, reason: ReasonCode) -> QueryClassification:
    return QueryClassification("classified", view, qdate, reason)


def _non_routed(
    qdate: date, status: Literal["clarification_required", "unsupported"], reason: ReasonCode
) -> QueryClassification:
    return QueryClassification(status, None, qdate, reason)


def _parse_provider_result(raw: object, qdate: date) -> QueryClassification:
    if not isinstance(raw, str):
        raise QueryClassificationOutputError("classifier output must be JSON text")
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except QueryClassificationOutputError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError, ValueError) as exc:
        raise QueryClassificationOutputError("classifier output is malformed JSON") from exc
    if not isinstance(payload, dict):
        raise QueryClassificationOutputError("classifier output must be a JSON object")
    if frozenset(payload) != _JSON_FIELDS:
        raise QueryClassificationOutputError("classifier output has missing or unexpected fields")
    if payload["schema_version"] != QUERY_CLASSIFICATION_SCHEMA_VERSION:
        raise QueryClassificationOutputError("classifier output uses an unsupported schema version")
    status = payload["status"]
    view = payload["view"]
    reason = payload["reason_code"]
    if not isinstance(status, str) or status not in _REASONS_BY_STATUS:
        raise QueryClassificationOutputError("classifier status is unsupported")
    if not isinstance(reason, str) or reason not in _REASONS_BY_STATUS[status]:
        raise QueryClassificationOutputError("classifier reason_code is invalid for its status")

    as_of = _parse_boundary(payload["as_of"], field="as_of")
    history_range = _parse_range(payload["history_range"])
    if status != "classified":
        if view is not None or as_of is not None or history_range is not None:
            raise QueryClassificationOutputError(
                "non-classified output cannot contain a retrieval view or temporal boundary"
            )
        return _non_routed(qdate, status, reason)  # type: ignore[arg-type]

    if view not in ("current", "as_of", "history"):
        raise QueryClassificationOutputError("classifier view is unsupported")
    if view == "current" and (as_of is not None or history_range is not None):
        raise QueryClassificationOutputError("current view cannot contain temporal boundaries")
    if view == "as_of" and (as_of is None or history_range is not None):
        raise QueryClassificationOutputError("as_of view requires exactly one as_of boundary")
    if view == "history" and as_of is not None:
        raise QueryClassificationOutputError("history view cannot contain an as_of boundary")
    try:
        result = QueryClassification(
            "classified", view, qdate, reason, as_of=as_of, history_range=history_range
        )
    except (TypeError, ValueError) as exc:
        raise QueryClassificationOutputError(
            "classifier reason or temporal fields are inconsistent with its view"
        ) from exc
    if result.view == "as_of" and result.as_of is not None:
        query_day = datetime.fromtimestamp(result.as_of.timestamp, timezone.utc).date()
        if query_day > qdate:
            raise QueryClassificationOutputError("classifier proposed an as_of date in the future")
    return result


def _parse_boundary(value: object, *, field: str) -> TemporalBoundary | None:
    if value is None:
        return None
    if not isinstance(value, dict) or frozenset(value) != _ENDPOINT_FIELDS:
        raise QueryClassificationOutputError(f"{field} boundary has invalid fields")
    raw = value["value"]
    inclusive = value["inclusive"]
    if not isinstance(raw, str):
        raise QueryClassificationOutputError(f"{field} boundary value must be a date or timestamp string")
    if not isinstance(inclusive, bool):
        raise QueryClassificationOutputError(f"{field} boundary inclusive must be a boolean")
    try:
        timestamp = as_of_timestamp(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QueryClassificationOutputError(f"{field} boundary date or timestamp is invalid") from exc
    return TemporalBoundary(timestamp, inclusive=inclusive)


def _parse_range(value: object) -> TemporalRange | None:
    if value is None:
        return None
    if not isinstance(value, dict) or frozenset(value) != _RANGE_FIELDS:
        raise QueryClassificationOutputError("history_range has invalid fields")
    start = _parse_boundary(value["start"], field="history_range.start")
    end = _parse_boundary(value["end"], field="history_range.end")
    try:
        return TemporalRange(start=start, end=end)
    except (TypeError, ValueError) as exc:
        raise QueryClassificationOutputError("history_range boundaries are contradictory") from exc


def _validate_against_query(
    result: QueryClassification,
    query: str,
    qdate: date,
    mentions: list[_Mention],
) -> None:
    if not result.ready_for_retrieval:
        return
    if result.view == "current" and mentions:
        raise QueryClassificationOutputError(
            "classifier routed an explicitly dated query to current"
        )
    recognized = {
        mention.boundary.timestamp
        for mention in mentions
        if mention.boundary is not None
    }
    if result.view == "as_of" and result.as_of is not None and not recognized:
        raise QueryClassificationOutputError(
            "classifier proposed an as_of date not expressed in the query"
        )
    if result.view == "as_of" and result.as_of is not None and recognized:
        if len(recognized) > 1:
            raise QueryClassificationOutputError(
                "multiple query dates cannot be reduced to one as_of boundary"
            )
        if result.as_of.timestamp not in recognized:
            raise QueryClassificationOutputError(
                "classifier as_of boundary does not match a date in the query"
            )
    if result.view == "as_of" and result.as_of is not None:
        before = _before_mention(query, mentions)
        if before is not None and result.as_of.inclusive:
            raise QueryClassificationOutputError(
                "a before-date query requires an exclusive temporal boundary"
            )
        query_day = datetime.fromtimestamp(result.as_of.timestamp, timezone.utc).date()
        if query_day > qdate:
            raise QueryClassificationOutputError("classifier proposed an as_of date in the future")
    if result.view == "history" and result.history_range is None and len(recognized) > 1:
        raise QueryClassificationOutputError(
            "classifier omitted the requested multi-date history range"
        )
    if result.view == "history" and result.history_range is not None:
        endpoints = {
            endpoint.timestamp
            for endpoint in (result.history_range.start, result.history_range.end)
            if endpoint is not None
        }
        if not recognized or endpoints != recognized:
            raise QueryClassificationOutputError(
                "classifier history range does not match the dates in the query"
            )
        before = _before_mention(query, mentions)
        if before is not None and (
            result.history_range.end is None or result.history_range.end.inclusive
        ):
            raise QueryClassificationOutputError(
                "a before-date history range requires an exclusive end"
            )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise QueryClassificationOutputError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise QueryClassificationOutputError(f"invalid JSON constant: {value}")

