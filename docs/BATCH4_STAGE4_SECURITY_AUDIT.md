# EngramOS Batch 4 Stage 4 — Security Acceptance Audit

**Review type:** Read-only security and integration acceptance review  
**Scope:** Working-tree implementation of Batch 4 Stages 1–4  
**Verdict:** **REJECT**  
**Reason:** An eligible as-of trigger candidate can expose its trigger cue even when that text comes from after the requested event-time boundary. The integrated pipeline also has an independently callable legacy Reader entry point that does not enforce Stage 4 revalidation or as-of redaction.

## Execution order

```text
answer_query(...)
  ├─ classify_query(query, question_date)
  │    └─ clarification / unsupported / failure → return before retrieval
  ├─ retrieve_memory(classified view + exact boundaries)
  │    ├─ turn FTS event-time filter before SQL LIMIT
  │    ├─ dense turn allowed_ids filter before scoring / top-k
  │    ├─ temporal fact view + shared eligibility and provenance checks
  │    └─ initial retrieval snapshot closes
  ├─ optional rerank (current/as_of only; history skips reranking)
  │    └─ accepted output must be a permutation of the supplied typed IDs
  ├─ rehydrate_reader_candidates(original IDs only)
  │    ├─ new SQLite read transaction and fresh snapshot
  │    ├─ eligibility, quarantine, event-time, fact-view, and provenance checks
  │    ├─ fresh rows become candidates in the selected ordering
  │    └─ authorization snapshot closes
  ├─ build ReaderContext from rehydrated candidates; redact later provenance text
  └─ answer_from_context → provider → strict response/citation validation
       └─ answer remains an outcome; pipeline does not persist it
```

The code order is visible in `src/engramOS/pipeline.py:67-153`. The original
retrieval snapshot is not the Reader authorization point; the subsequent
rehydration snapshot is.

## Pipeline and component findings

### Verified behavior

- Classification precedes retrieval. `answer_query()` returns for any
  non-ready result, and classifier errors return without a retrieval result
  (`pipeline.py:67-79`). Integration tests cover unsupported and failed
  classification short-circuits (`tests/test_pipeline.py:93-122`). The
  classifier validates provider output and does not default uncertain input to
  current (`query_classifier.py:237-347`).
- Retrieval receives `classification.view`, `as_of`, and `history_range`
  directly (`pipeline.py:82-92`). Turn event-time bounds are included in FTS
  SQL before `LIMIT` and as dense `allowed_ids` before scoring; they are also
  checked after fusion (`retrieval.py:167-190, 492-514, 557-559`). Tests cover
  inclusive and exclusive cutoffs and filtering before dense top-k
  (`tests/test_pipeline.py:174-194`).
- Reranking receives immutable candidate snapshots and validates an exact
  permutation; it cannot add IDs or change candidate contents. Failures return
  the original retrieval result, which still goes through final revalidation.
  History skips reranking (`reranking.py:116-211`). Tests cover fabricated or
  missing IDs, fallback, and history order (`tests/test_reranking.py:181-255,
  427-445`).
- Revalidation runs after reranking (`pipeline.py:104-132`). The Reader context
  is built only from `materialized.candidates`, never from the initial
  retrieval result. Empty final evidence causes Reader abstention without a
  provider call (`reader.py:282-354`). The context path validates citation
  membership against its top-level evidence IDs (`reader.py:350-352,
  717-769`).
- The pipeline and Reader context path contain no memory-write operation.
  Generated answers are returned in `QueryAnswerOutcome` and are not persisted.

### Finding F-01 — Post-cutoff trigger cue can enter an as-of context

**Severity: P1 — acceptance blocker.**

For a trigger fact, `_snapshot_fact()` sets the top-level claim to the supplied
parent fact text but also copies the trigger row's own `key_text` into
`metadata.retrieval_cue` (`reader.py:460-529`, especially line 501). The
pipeline's as-of handling redacts only nested provenance `text` fields
(`pipeline.py:168-204`); it does not redact or omit `retrieval_cue`. Trigger
retrieval expands provenance from the parent fact (`retrieval.py:872-941`), so
the trigger row's own source turn is not present among the provenance turns
whose event times the pipeline checks. The shared fact eligibility predicate
checks source-turn gate/quarantine state and parent eligibility, but not the
trigger source turn's event-time boundary (`eligibility.py:56-75`).

Consequently, an eligible persisted trigger with `valid_from` in the as-of
view and a cue derived from a later turn can expose that cue in the Reader JSON
metadata even when its source turn is after the cutoff. Trigger generation is
not implemented, but persisted triggers are supported by retrieval and the
Reader; the path is therefore reachable from eligible database state. The
Reader prompt treats evidence as untrusted but that does not satisfy the
approved temporal omission policy.

Tests verify trigger cue preservation in the Reader and trigger temporal
retrieval separately (`tests/test_reader.py:563-594`,
`tests/test_retrieval.py:568-635`), but no integration test places a
post-cutoff trigger cue in an as-of Reader context.

### Finding F-02 — Alternate Reader APIs do not enforce the pipeline boundary

**Severity: P2 — caller-dependent bypass.**

`answer_from_evidence()` is a public callable that snapshots any caller-supplied
retrieval candidates and sends them to the legacy Reader prompt; it performs no
fresh database revalidation or as-of event-time filtering/redaction
(`reader.py:209-279`). `answer_from_context()` likewise accepts any
constructible `ReaderContext`; it validates shape and history consistency, but
cannot establish that the context was created by `answer_query()` or authorized
by SQLite (`reader.py:282-358`).

The Stage 4 `answer_query()` path calls only `answer_from_context()` with its
own freshly built context, and repository call-site search found no production
call to `answer_from_evidence()` beyond its tests. The bypass is therefore not
inside the current pipeline path, but consumers can bypass the guarantees if
they call either Reader API directly with stale, fabricated, or unfiltered
evidence. The API boundary relies on caller discipline; it is not an
unforgeable authorization capability.

### Finding F-03 — Retrospective marker does not inspect fact recording time

**Severity: P2 — temporal interpretation gap.**

For as-of facts, `_reader_evidence()` marks `retrospective` only when a
provenance turn's `event_time` or `recorded_at` is later than the fact's
`valid_from` (`pipeline.py:181-198`). It separately copies the fact row's own
`recorded_at` into metadata but does not use it in that marker decision. The
temporal write API accepts a `FactInput.recorded_at` independently of the
source turn's recording time (`temporal.py:27-52, 150-153`). If a backdated
fact was recorded after its valid time while its source provenance timestamps
are not later than that valid time, the context can show a later fact
`recorded_at` with `retrospective: false`. This could let the Reader imply
contemporaneous knowledge contrary to the prompt's retrospective rule.

Existing tests cover a later source and a later confirmation, not a fact whose
own recording time is the only later timestamp (`tests/test_pipeline.py:124-172`).

## Fresh-snapshot boundary and race

`rehydrate_reader_candidates()` executes `BEGIN` at `retrieval.py:668`. This is
a deferred SQLite transaction; the read snapshot is established by its first
read, `SELECT 1 FROM turns WHERE history_id = ? LIMIT 1` at line 670. The
authorization timestamp is taken immediately afterward at line 673. All
subsequent checks and row reconstruction use this same connection and
transaction:

- Requested standalone turns are checked by `_eligible_turn_ids()` using the
  shared store-required/no-quarantine predicate, then restricted by the
  as-of/history event-time range and selected again with the eligibility
  predicate (`retrieval.py:674-702`, `eligibility.py:17-35`).
- Facts are rebuilt from the fresh temporal view, which applies shared fact,
  source-turn, parent-fact, and quarantine eligibility. History candidates
  are additionally filtered by the requested `valid_from` range
  (`retrieval.py:704-710, 821-870`, `eligibility.py:56-75`).
- Fact source/confirmation turns are selected and checked inside the same
  transaction during evidence expansion (`retrieval.py:872-941`). For as-of
  answers, later provenance text is blanked after rehydration and before the
  Reader prompt is built (`pipeline.py:168-204`).
- Rehydration only emits requested IDs from the original retrieval result;
  fresh rows do not add candidates (`retrieval.py:656-665, 712-731`).

The `connection()` context manager commits and closes after rehydration returns
(`db.py:50-66`). No database reads occur between that close and construction of
`ReaderContext`; the pipeline only copies/materializes the returned values.
Reader evidence is frozen dataclass data with tuple provenance and
read-only metadata (`reader.py:107-156, 282-344`). It does not receive the
database handle or retrieval result object. Ranking diagnostics carried through
from the original retrieval objects are not exposed as Reader text/evidence.

**Race guarantee:** A quarantine committed before the first read of the final
snapshot is visible to revalidation and excludes the affected candidate. A
quarantine committed after that snapshot begins is not visible to that
snapshot. Once the snapshot closes and the immutable context is built, there
is no mechanism to revoke or invalidate it before/during the provider call.
The Reader may therefore answer from evidence that became quarantined after
the authorization snapshot. The guarantee is authorization against the
database state visible at the final snapshot, not continuous authorization
through answer generation. This matches the documented snapshot policy; it is
not a claim of post-snapshot revocation. The integration test exercises a
quarantine committed between initial retrieval and final revalidation
(`tests/test_pipeline.py:196-219`), but not a writer committing after the
final snapshot begins.

## As-of temporal isolation

### Verified behavior

- Standalone turn FTS and dense candidates are filtered on `event_time` before
  candidate limits/ranking, then checked again during final rehydration
  (`retrieval.py:167-190, 492-514, 674-702`). Inclusive/exclusive tests cover
  the cutoff, including the exact event-time boundary
  (`tests/test_pipeline.py:174-194`).
- Facts use valid-time intervals. Inclusive as-of selects
  `valid_from <= T < valid_until`; an exclusive boundary selects the left
  limit `valid_from < T <= valid_until`, without epsilon arithmetic
  (`temporal.py:863-957`). The Stage 1 boundary test covers immediately before,
  at, and after the cutoff (`tests/test_temporal.py:409-429`).
- Ordinary fact source, confirmation, and supporting provenance records are
  copied into nested provenance. `_reader_evidence()` checks every nested
  provenance event time and blanks text after the cutoff, including equality
  for an exclusive boundary (`reader.py:551-572`, `pipeline.py:181-194`). Tests
  cover later source and confirmation text (`tests/test_pipeline.py:124-172`).
- Nested provenance IDs are not top-level evidence IDs and cannot be cited on
  their own; response citation validation checks exact membership in the
  supplied top-level ID set (`reader.py:717-769`).
- Backdated facts remain eligible by valid time; the Reader prompt instructs
  that marked retrospective facts do not establish contemporaneous knowledge
  (`reader.py:42-45`). The marker gap is recorded as F-03 above.

### Indirect post-cutoff text path

Finding F-01 is an indirect text path: trigger `key_text` is retained as
`retrieval_cue` metadata and is not subject to the provenance text redaction.
Also, `answer_from_evidence()` does not perform the Stage 4 as-of filtering or
redaction; direct callers can pass later source text into its legacy prompt
(F-02). Within the intended ordinary-fact `answer_query()` route, top-level
standalone turns are filtered and nested fact provenance text is withheld.

## History-range semantics

`retrieve_memory(view="history")` obtains the chronological temporal fact view,
then selects changes by `valid_from` using `_in_range()`. A change exactly at
an inclusive start/end is selected; a change at an exclusive endpoint is not
(`retrieval.py:1204-1217, 1234-1287`). Baselines require
`valid_from < start.timestamp`, independent of start inclusivity, and must be
active at the opening boundary. Thus a change exactly at an exclusive start
cannot be reintroduced as a baseline (`retrieval.py:1234-1255, 1289-1296`).
For each slot, active prior facts are sorted by descending `valid_from` and
the latest is retained; final order is deterministic by `valid_from`, entity,
attribute, and fact ID.

Integration tests cover inclusive and exclusive exact endpoints, exclusion of
an exclusive-start change as a baseline, and a labeled pre-range baseline
(`tests/test_pipeline.py:221-294`). The baseline fixture has one prior state
per slot, so it does not directly test choosing among multiple prior changes
in the same slot. The implementation's descending `valid_from` selection
provides that behavior, but this particular edge has no direct integration
assertion.

History context labels the fact-change timeline as complete while setting
`turn_evidence_complete=False`; its ordering label says turns are supplemental
query-ranked evidence (`pipeline.py:114-145`). The Reader prompt also says
ranked supplemental turns are not complete history (`reader.py:42-45`). No
complete raw conversation history claim is made.

## Stage isolation and scope

- Stage 1 strict classification remains before retrieval and requires explicit
  question date; ambiguous, malformed, unsupported, or failed classification
  stops. Relevant tests include `tests/test_query_classifier.py` and the
  pipeline short-circuit tests.
- Stage 2 reranking remains candidate-only and exact-permutation validated;
  no trust or quarantine decision is exposed as ranking metadata. Tests are in
  `tests/test_reranking.py`.
- Stage 3 Reader response parsing rejects malformed schema, invalid/fabricated
  citations, duplicate citations, and prompt-injection attempts remain data.
  Tests are in `tests/test_reader.py`. The separate direct-call limitation is
  F-02.
- Gate and eligibility code remains shared and fail-closed: eligible turns
  require a store event and no quarantine; eligible facts require a clean
  source and, for triggers, an eligible parent (`eligibility.py:17-75`). The
  audit found no diff to gate or eligibility code and no change weakening
  trust/quarantine checks.
- No `schema.sql` or persistent index definition changed. No Batch 3 decision
  was silently altered; the decisions diff adds a Stage 4 section only.
- No live provider adapter or external network/model dependency is connected
  by these modules. The executed suites use deterministic fake providers and
  local vectors. No live provider call, paid API, network call, or model
  download occurred during this audit.

## Verification

Focused command:

```powershell
python -m unittest tests.test_pipeline tests.test_retrieval -v
```

Result: **43 tests passed, 0 failures; exit code 0.**

Full command:

```powershell
python -m unittest discover -s tests -v
```

Result: **290 tests passed, 0 failures; exit code 0.**

## Remaining limitations

- F-01 permits a conditional post-cutoff trigger-cue disclosure in as-of
  context and is the reason for the reject verdict.
- F-02 means Stage 4 guarantees apply to `answer_query()` callers; direct
  Reader API consumers must supply already-authorized evidence, but this is
  not enforced by an unforgeable context capability.
- F-03 can under-label facts whose own recording time is later than valid time
  when provenance turn timestamps do not independently reveal that fact.
- A quarantine after the final snapshot starts cannot revoke already
  materialized evidence.
- No production classifier, reranker, or Reader provider has been audited;
  provider implementations are injected Python code and are not sandboxed by
  this repository. The Reader module itself receives a prompt and does not
  access SQLite or retrieve more evidence.
- Citation membership does not prove semantic entailment. Supplemental turns
  are not a complete raw conversation timeline.

**Final verdict: REJECT for security acceptance pending resolution of F-01;
F-02 and F-03 should also be addressed or explicitly accepted by the security
review owner.**

## Remediation and revalidation — 2026-11-03

The three findings are addressed in the working tree:

- Trigger evidence no longer includes the trigger's `key_text` as Reader metadata.
- The only supported public query entry point is `pipeline.answer_query()`. The
  supplied-evidence and context Reader helpers are module-private; the context
  authorization marker is an accidental-misuse guard, not an unforgeable
  capability against code running in the same Python process.
- As-of retrospective marking also checks the fact row's own `recorded_at`
  against `valid_from`.

Regression coverage now includes a post-cutoff trigger cue through
`answer_query()`, a fact whose own recording time is the only late timestamp,
and the absence of the former public Reader function names. Focused verification
passed **53 tests, 0 failures**. Full verification passed **293 tests, 0
failures** using `python -m unittest discover -s tests -v`.

**Disposition:** F-01, F-02, and F-03 are closed for the documented application
boundary. Existing scope limits still apply: direct SQL and arbitrary code
running in-process are outside the authorization boundary.
