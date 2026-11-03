# Batch 4 — Stage 1: Query classification

**Status:** The Stage 1 classifier module is provider-independent; an optional cache-backed Meta adapter exists in `src/engramOS/meta_adapters.py`. The revised offline benchmark injects a fake classifier.  
**Scope:** Query classification contract, strict output validation, deterministic
date parsing, and explicit `as_of` boundary semantics. Reranking, Reader
integration, and end-to-end orchestration remain open.

## Interface

`engramOS.query_classifier.classify_query(query, *, question_date,
classifier=None, timeout=5.0) -> QueryClassification` accepts a non-empty
query and an explicit date context. `question_date` may be a `date`, an ISO
`YYYY-MM-DD` string, or an offset-aware `datetime` (converted to its UTC
calendar date). There is no machine-clock fallback.

The frozen `QueryClassification` has:

| Field | Type | Meaning |
|---|---|---|
| `status` | `classified`, `clarification_required`, or `unsupported` | Only `classified` may proceed to retrieval. |
| `view` | `current`, `as_of`, `history`, or `None` | The existing Phase 4A temporal view. |
| `question_date` | `datetime.date` | The explicit normalized question date. |
| `reason_code` | Versioned string enum | Non-prose diagnostic category. |
| `as_of` | `TemporalBoundary` or `None` | A point boundary for the `as_of` view. |
| `history_range` | `TemporalRange` or `None` | Optional bounded history endpoints. |
| `ready_for_retrieval` | `bool` property | True only for a validated `classified` result with a view. |

`TemporalBoundary(timestamp: float, inclusive: bool = True)` is defined in
`engramOS.temporal`. `timestamp` is Unix seconds normalized to UTC. A date-only
value continues to mean 00:00:00 UTC at the start of the date, as implemented
by `as_of_timestamp()`.

For a fact `as_of` lookup, an inclusive boundary selects the state at the
instant. An exclusive boundary selects the left limit immediately before the
instant. Thus a fact interval `[valid_from, valid_until)` is selected at an
exclusive boundary `T` when `valid_from < T <= valid_until`. This is represented
explicitly and uses no epsilon arithmetic. `get_facts_as_of()` and
`retrieve_facts(view="as_of", ...)` accept this boundary type additively; all
existing scalar/date inputs keep their previous inclusive behavior.

`TemporalRange(start, end)` is an immutable pair of optional `TemporalBoundary`
values. It must contain at least one endpoint and rejects reversed or empty
ranges. For range endpoints, `inclusive` has its ordinary endpoint meaning.

## Provider and failure contract

`QueryClassificationProvider.classify(query, *, question_date, timeout) ->
str` is the injection point. An adapter must enforce the supplied positive,
finite timeout and return raw JSON matching the v1 contract in `prompts.md`.
Stage 1 implements no network or model provider. Tests use local deterministic
fakes.

The response parser rejects duplicate JSON keys, invalid JSON constants,
missing or extra keys, unsupported schema/view/status/reason values, invalid
dates, naive timestamps, wrong field types, contradictory boundaries, and
date boundaries that do not match those in the question. A provider exception
or timeout raises `QueryClassificationError`; malformed or unsafe output raises
`QueryClassificationOutputError`. No such failure can produce a `current`
route. The caller must stop before retrieval on either exception or on a result
whose `ready_for_retrieval` is false.

Diagnostic logs identify the query-classification stage and failure category
or exception class. They do not log raw query text, provider output, prompts,
or evidence content.

Deterministic high-confidence routes do not call a provider:

- Direct present-state forms such as “What is my role?” route to `current`.
- Explicit date point questions route to `as_of`.
- `before DATE` point-state questions route to `as_of` with an exclusive
  boundary. For an explicit month-year form such as “before March 2023,” the
  boundary is the start of that month in UTC; it is not an arbitrary point in
  the month.
- Explicit change/timeline requests route to `history`; a clear `before DATE`
  history request uses that date as an exclusive end.
- A clearly requested calendar-month history uses `[month start, next month
  start)` rather than selecting an arbitrary point in the month.
- `today`, `yesterday`, and integer `N day(s) ago` resolve from the supplied
  `question_date`.
- `tomorrow` resolves from `question_date` as an unsupported future request;
  it never routes to current.

The rules evaluate query shape and intent, not single words. A `before` query
with a point-state question is not sent to history. Unclear uses of `before`,
`after`, `previously`, or `history` do not determine a view by themselves.
Month-only point questions, relative periods such as “last week,” invalid
dates, mixed current/date requests, and underspecified multi-date ranges return
`clarification_required`. A known as-of date later than `question_date` returns
`unsupported`. Knowledge-time requests return `unsupported`; this repository
does not reconstruct what was known at the time.

For a two-date history range, deterministic routing accepts only endpoint
wording with clear inclusivity (`from DATE through DATE`, `from DATE until
DATE`, or `after DATE and before DATE`). Phrases such as `between DATE and
DATE` request clarification because their endpoint inclusivity is not explicit
in the approved policy.

When no deterministic route applies and no provider is injected, classification
returns `clarification_required` with `classifier_unavailable` or
`ambiguous_temporal_intent`; it never defaults to current. Unsupported views
and partial classifications cannot pass `ready_for_retrieval`.

## Prompt preservation and versioning

The original Section C text in `prompts.md` remains byte-for-byte intact. Its
keyword rule that routed every `before` query to history conflicts with the
approved Batch 4 policy. A clearly marked `engram.query-classification.v1`
extension follows Section C and governs Batch 4 behavior without rewriting the
historical text or its hash semantics.

The v1 prompt hash basis is SHA-256 over the exact UTF-8 instruction-template
bytes in the extension, with literal `{question_date}` and `{question}`
placeholders before substitution. The schema hash basis is SHA-256 over the
complete versioned extension subsection in `prompts.md`, from its heading
through the final hash-basis paragraph and excluding the following horizontal
rule. Both hash inputs use UTF-8 with CRLF normalized to LF. Stage 1 does not
write hashes to a run table; run/evaluation metadata is outside this stage.

## Integration boundary and known limitations

Stage 1 produces a validated route only; it does not invoke Phase 4A retrieval
or connect a Reader. The new exclusive boundary behavior is implemented in
fact temporal retrieval. Before Batch 4 orchestration is complete, the later
pipeline stage must also apply the same inclusivity to standalone turn event
times and any history range. Existing `retrieve_memory()` does not itself
apply an as-of cutoff to standalone turn results, so that behavior must not be
assumed from the new classifier contract. Later pipeline work must also
prevent a backdated fact's later source or confirmation turn from being shown
as contemporaneous: the projected fact's validity interval and its source or
confirmation `event_time` are distinct. In particular, a later confirmation
turn must be omitted from an as-of context when its `event_time` exceeds the
requested cutoff.

The classifier recognizes ISO dates, common English month/day/year forms,
offset-aware ISO timestamps, and the relative day forms listed above. Other
relative expressions, unsupported date formats, unknown temporal semantics,
and broad natural-language questions require a provider or clarification.
There is no historical knowledge-state reconstruction, no live provider, and
no schema/index change in this stage.

## Stage 1 tests

`tests/test_query_classifier.py` covers current/as-of/history routes, full
intent over isolated keywords, strict structured output validation, date-only
UTC normalization, offset timestamps, question-date-based relative dates,
month ambiguity, future and knowledge-time rejection, provider failure, and
fail-closed behavior. `tests/test_temporal.py` adds
`test_exclusive_as_of_boundary_selects_left_limit_without_epsilon`, covering
the representable timestamp immediately before a cutoff, the exact cutoff
under inclusive and exclusive semantics, and the representable timestamp
immediately after it.

## Stage 1 provider-call audit — 2026-11-03

The classifier itself is deterministic through its local query-shape checks.
If a provider is injected, it is called only after those branches fail to
produce a classified, clarification, or unsupported result. Without a
provider, an unresolved query returns `clarification_required`; it does not
default to current.

The exact deterministic routes read from `query_classifier.py` are:

- **Unsupported before provider:** recognized knowledge-time phrases matching
  `_KNOWLEDGE_TIME`; a parsed relative `tomorrow`; or an as-of point boundary
  later than the supplied `question_date` when `_as_of()` validates that route.
- **Clarification before provider:** malformed date expressions; relative
  periods matching `last|this|past week|month|year`; unresolved month-only
  mentions; history intent with unclear date endpoints; explicit current intent
  combined with a date; current-shaped temporal requests that cannot be made
  safe; and other locally detected ambiguity.
- **Present state:** with no recognized temporal mention and no `_TEMPORAL_WORDS`
  match, either `_CURRENT_EXPLICIT` (`current`, `currently`, `right now`,
  `latest`, `presently`) or `_CURRENT_QUESTION` (question begins with `what`,
  `who`, `where`, `which`, `when`, `how many`, or `how much`, followed by
  `is`, `are`, `do`, `does`, `has`, `have`, or `can`) produces `current`.
- **Explicit as-of:** one supported date or offset-aware timestamp in a point
  question; recognized `before`/`prior to` plus a point question yields an
  exclusive boundary. The parser recognizes ISO dates, ISO date-times,
  `D Month YYYY`, `Month D YYYY`, and `Month YYYY`; an explicit month point
  question is only deterministic when phrased `before Month YYYY`.
- **Relative as-of:** `today`, `yesterday`, and integer `N day(s) ago` use the
  supplied question date. `tomorrow` is unsupported rather than current.
- **History:** an explicit `_CHANGE_INTENT` without dates routes to the full
  history view. A single date can form an exclusive `before` end, a start
  after/since a date, or an inclusive through-date end. Two dates route only
  with explicit endpoint wording: `from DATE through DATE`, `from DATE until
  DATE`, or `after DATE and before DATE`. A change request for one clear month
  routes to that calendar month's `[start, next-start)` range; “before Month
  YYYY” uses an exclusive month-start boundary for a point query and for a
  history range.

The authored-suite audit test ran all 20 question strings through the actual
classifier with a counting provider. Only these two queries called the
provider fallback:

- `Name the astronomy project I keep in my notes.`
- `What market feature was reported?`

The representative query `What is my current city?` returned
`present_state` with zero provider calls. `tests/test_stage1_eval_audit.py`
contains both assertions. In the offline evaluation these two provider
fallbacks were served by the cached fake classifier, not a live model.
