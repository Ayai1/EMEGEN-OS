# Technical decisions

## Temporal storage and interval projection

`facts` is the query-facing projection of normalized slot intervals. A coalesced
interval can represent multiple source observations: for example, A@10 followed
by A@30 becomes one A interval. If B@20 later arrives, the A@30 boundary must
be restored. Therefore the schema adds `fact_observations`, a lossless ledger of
slot events and their source turns. The temporal writer rebuilds one history and
slot's interval projection from those observations inside one SQLite
`BEGIN IMMEDIATE` transaction. The ledger also preserves source provenance and
lets retrieval recalculate eligibility when a turn has a quarantine event.

The addition is backward-compatible at the schema level. Database
initialization copies existing slot facts into the ledger and preserves their
available confirmation count and latest confirmation provenance. Historical
observations that were already coalesced or discarded before this migration
cannot be reconstructed; the migration treats each old interval row as one
available source observation.

Single-valued slot intervals are half-open: `[valid_from, valid_until)`. Equal
normalized values in consecutive event order are one interval, with confirmation
count and latest confirmation provenance aggregated. A value equal to its
successor continues that successor's logical interval. Backdated insertion
rebuilds both neighbors and all affected supersession links atomically.

The project specifies deterministic timestamp formulas but does not guarantee
that two separate sessions cannot produce the same `valid_from`. For exact
single-valued timestamp collisions, ordering is stable by event time, session
ID, turn index, fact sequence, then observation ID. The implementation advances
later exact ties by one representable floating-point step (`math.nextafter`),
which preserves positive-duration intervals without a fixed epsilon that could
collide at large Unix timestamps. Multi-valued observations at the same time
remain simultaneous and are not shifted.

## Date-only `as_of`

`BUILD_PLAN.md`, `prompts.md`, and `schema.sql` specify date parsing but do not
say whether a date-only query means the beginning or end of that date, or which
timezone applies. This milestone chooses **00:00:00 UTC at the start of the
date**. The convention is implemented for `date` values and `YYYY-MM-DD`
strings. Timestamp strings and `datetime` values require an explicit UTC offset;
naive values are rejected instead of depending on the machine timezone.

## Current-state view

The current view means the latest eligible event-time interval in a history for
each single-valued slot, not “the fact whose timestamp is before the wall-clock
now.” This makes replay deterministic and independent of when the query runs.
An `as_of` query uses the half-open interval boundary; history view returns all
eligible normalized intervals in event order.

## Quarantine eligibility

`gate_events` records decisions with `event_id`, `turn_id`, and a `store` or
`quarantine` decision. It has no separate current-status column, and the source
specifications do not define a quarantine-revocation transition. This temporal
layer reads the event history without updating or deleting it.
The fail-closed eligibility contract requires at least one `store` event and
no `quarantine` event. Any quarantine decision makes a turn ineligible, even
if a later `store` event exists. A later store can make a previously pending
turn eligible, but cannot silently reopen quarantined content.

| Gate events for the turn, in insertion order | Eligible |
|---|---|
| none | no |
| store | yes |
| quarantine | no |
| store, quarantine | no |
| quarantine, store | no |
| store, store | yes |

`turn_eligibility_predicate(..., gated=True)` implements this rule with an
`EXISTS store` check and a `NOT EXISTS quarantine` check. `gated=False` remains
exactly `1 = 1` for turn-only baseline paths. Fact retrieval and
`retrieve_memory` have no ungated option and always apply the new contract.

The SQL eligibility predicates live in `src/engramOS/eligibility.py`. Raw-turn
FTS and dense retrieval, fact FTS and dense retrieval, persisted-trigger
retrieval, `hybrid_rag`, and `recency` are implemented and covered by tests.
These retrieval paths use the shared turn, fact, and observation eligibility
predicates where applicable. Trigger generation, the query classifier,
reranking, Reader-context construction, and answer generation are not
implemented.

## Phase 3 trust gate

The source specification names these three signals:

- `signal_bridge`: a bridging-similarity spike across shortening query prefixes.
- `signal_policy`: contradiction against a pinned system/user policy fact of
  kind `constraint`.
- `signal_source`: source type is exactly `tool` or `external_doc`.

The source rule is evaluated locally and deterministically. The specification
does not define a bridge similarity algorithm or threshold, nor a policy
contradiction algorithm. `SignalEvaluator` therefore injects those two
evaluators; this milestone adds no heuristic or paid model implementation.
Both evaluators return explicit `fired` or `clear` results. An exception,
missing return, or explicit incomplete result aborts gate evaluation. It cannot
be persisted as `store`, counted as a fired signal, or followed by memory
ingestion. The caller must stop processing that turn and surface the error.
There is no gate-failure event in the current schema, so failed evaluations are
logged but do not create a `gate_events` row.

The pure decision function counts fired results and quarantines at two or more.
Every complete signal combination is:

| Bridge | Policy | Source | Fired count | Decision |
|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | store |
| 0 | 0 | 1 | 1 | store |
| 0 | 1 | 0 | 1 | store |
| 0 | 1 | 1 | 2 | quarantine |
| 1 | 0 | 0 | 1 | store |
| 1 | 0 | 1 | 2 | quarantine |
| 1 | 1 | 0 | 2 | quarantine |
| 1 | 1 | 1 | 3 | quarantine |

The existing `gate_events` columns store each signal outcome as `0.0` or `1.0`,
the fired count, decision, creation timestamp, and SQLite-generated `event_id`.
Each complete evaluation appends a row; no previous decision is updated or
deleted. Schema triggers reject direct updates and deletes as well. Gate-event
insertion, trust change, fact/observation quarantine, and temporal interval
rebuilding share one transaction.

The canonical application-facing operation is
`evaluate_and_record_turn(database, turn_id, evaluator, ...)`. It loads the
source turn, evaluates bridge and policy signals through the injected evaluator,
computes source provenance locally, applies `decide_gate`, and then persists.
Callers cannot pass signal results or a prebuilt `GateEvaluation` to this API.
`evaluate_turn` remains diagnostic only; `_record_gate_decision` is an internal
helper used by the canonical operation.

Evaluators implement `fingerprint_inputs(conn, turn)` as a cheap, deterministic
local fingerprint covering every mutable database input used by their bridge
and policy checks. The gate checks the fingerprint before and after evaluation,
then checks it again inside the persistence transaction. Signal evaluation
therefore runs without a SQLite write lock, while `BEGIN IMMEDIATE` serializes
the final fingerprint check and event/trust/quarantine writes. The source turn's
identity, history, and source type are also revalidated; source content and
provenance columns are immutable. Current detector implementations are absent,
so future evaluators must honor the fingerprint contract for their actual input
sets. This contract assumes gate input fingerprints are local and do not call
external services.

The build plan says quarantine “lowers” `base_trust` but does not give a
decrement or formula. This implementation sets quarantined turns to `0.0`, the
lowest possible trust, without adding another scoring model. A `store` decision
leaves the current trust unchanged. A later `store` event never raises trust or
restores eligibility after any quarantine event.

On quarantine, facts and observations whose `turn_id` is the quarantined source
are marked `quarantined = 1`; their source rows are retained. Single-valued
slot projections are rebuilt from the remaining eligible observations so
quarantined confirmations cannot affect counts, dates, or interval boundaries.
The common predicates also check source-turn/history relationships and parent
facts, so a trigger derived from a quarantined parent is ineligible.

Raw-turn FTS is implemented and its candidate query uses the shared turn
predicate. Raw-turn dense retrieval, fact FTS and dense retrieval, persisted
trigger retrieval, `hybrid_rag`, and `recency` are also implemented and tested;
they use the shared eligibility predicates where applicable. Trigger
generation, the query classifier, reranking, Reader-context construction, and
answer generation are not implemented.

## Scope and dependencies

Temporal operations use the existing synchronous SQLite layer and deterministic
Python only. Phase 2B adds one small `ingestion_jobs` recovery table; it does not
add an LLM/API client, embedding dependency, or other runtime dependency. Paid
API requests and evaluation-credit use remain outside this milestone.

## Phase 2B turn ingestion and extraction

### Turn identity and source preservation

The canonical turn ID follows the schema comment exactly:
`{history_id}:{session_id}:{turn_idx}`. An omitted ID is generated from those
fields; a supplied nonmatching ID is rejected. The source payload fingerprint
covers turn/history/session IDs, turn index, role, exact raw content, source
type, token count, session timestamp, and session date. Replaying the same ID
with a different fingerprint raises `TurnConflictError`. `recorded_at` is
transaction metadata and does not change source identity on replay.

`event_time` is derived as `session_unix_ts + turn_idx`. The per-history hash
chain uses SHA-256 of the previous hash followed by the exact UTF-8 source
content. `session_date` may be supplied from the source dataset; if omitted it
is derived from `session_unix_ts` as a UTC calendar date. The source documents
do not define a complete role-to-source-type mapping; this repository applies
the asymmetric ingestion contract described in the Batch 2B Part B decision
below. Turns are append-only in
`(event_time, session_id, turn_idx, turn_id)` order within one history; callers
should serialize ingestion for a history and may process different histories
independently.

### Extraction schema and prompt

`src/engramOS/extraction.py` uses the original Section A system instructions
and the specified JSON fields: `session_id`, `turn_idx`, `entity`,
`attribute`, `value`, `cardinality`, `kind`, `as_of`, `replaces`, and
`confidence`. It does not add `history_id` or `fact_seq` to model output because
neither appears in the original response schema. Fact sequence is assigned by
the candidate's array position; returned session and turn index must match the
single source turn in the prompt, and the application attaches the validated
fact to that turn's persisted history. Empty arrays are successful results.

The full response is parsed and validated before staging: required/extra keys,
types, normalized identifiers, nonempty values, enumerations, ISO dates,
confidence bounds, and turn provenance are checked. `replaces` is retained in
staged extraction data for audit/replay; the existing `FactInput` schema has no
replacement field, so temporal supersession remains determined by the temporal
engine's slot/value interval rules. The local interface returns raw JSON text
to preserve strict validation at the trust boundary. `FactExtractor` is only a
protocol; no live model provider or substitute heuristic was added. Only
explicitly transient adapter errors and malformed JSON/schema responses may be
retried with a bounded attempt count (two by default in ingestion). Invalid
adapter return types and non-transient execution errors fail immediately.
Malformed responses never reach staging or temporal writes unless a subsequent
bounded attempt validates successfully.

### Persistent ingestion state and recovery

The additive `ingestion_jobs` table is a checkpoint for extraction recovery;
`turns`, `gate_events`, and `fact_observations` remain the source/audit records.
Its normal transitions are:

```text
pending_gate --completed store event--> approved
approved --validated extraction staged--> staged
staged --all temporal observations applied--> completed
pending_gate | approved | staged | completed --any quarantine event--> quarantined
```

A missing/failed gate evaluation creates no store event and leaves the job
pending. A retry first inspects `gate_events`: any quarantine event blocks
processing regardless of later store events; an existing store event is reused
instead of calling the gate again. This preserves sticky quarantine without
inventing revocation. Turn plus job insertion is one transaction. The canonical
gate call persists the event/trust/quarantine atomically; job-state
reconciliation follows in a separate transaction so an interruption after the
gate event is recovered from the authoritative event history. Evaluator and
extractor operations run outside SQLite write transactions.

All extraction candidates in a one-history batch are collected and validated
before any are staged or written. Each accepted JSON payload and SHA-256 hash
are staged before temporal writes. Temporal candidates are processed in
session/event/turn/fact-sequence order; the eligible slot registry is loaded
once for the history and extended after each successful write. `insert_fact()` remains the only path to facts and
observations. IDs `{turn_id}:fact:{fact_seq}` make retries idempotent. The
temporal insert and ingestion progress checkpoint are separate transactions;
if interruption occurs between them, replay submits the same observation ID
and the temporal layer verifies that its source fact matches before returning.
The batch can recover from a partially applied set of turns because each turn's
accepted extraction is staged independently before the first write.

The schema change is additive and `CREATE TABLE IF NOT EXISTS` based. Existing
turn/fact/gate tables are not rewritten. There is no cross-process lease for
same-history ingestion; callers must serialize a history's turns, consistent
with the build plan's sequential per-history extraction. The prompt's
multi-session batching is not yet connected to a live provider: this local
interface handles per-turn responses, while `ingest_turns` applies a supplied
batch in deterministic order. A crash between extractor return and the staging
transaction may require a fresh extraction on retry, but no fact has been
applied at that point.

### Phase 2B boundaries

The original Section A prompt remains unchanged in its instructions; the
builder places one exact source turn in its session block. Exact slot-name
matches are normalized and reused. The specified embedding and local reranker
canonicalization are not available in this milestone, so new names remain new
slots instead of being merged through an invented heuristic. No production
extraction provider is connected because the cache and budget manager are not
implemented. No paid requests were made; API credits consumed: **0**.

## Phase 4A hybrid retrieval

### Embedding contract and indexing

The existing `embeddings` table remains the vector store. `BAAI/bge-m3` is the
base repository identity; persisted BGE-M3 vectors use a revision-specific
provider ID. Vectors have 1024 dense dimensions, normalized cosine values, and little-endian
float16 BLOBs. Vector validation rejects wrong dimensions, non-finite and zero
vectors, malformed BLOB lengths, and model mismatches. NumPy is already a
project runtime dependency; no new dependency or vector database was added.

`embeddings.content_hash` stores SHA-256 of the exact indexed source text. This
small additive column prevents a reused fact reference ID from keeping an old
vector after a temporal projection rewrite. Database initialization adds the
nullable column to existing databases. Legacy embeddings with no digest remain
stored but are ignored until `index_history()` refreshes them. Indexing uses
the centralized database layer, enumerates only eligible records in the
requested history, skips unchanged model/text pairs, encodes bounded batches
outside write transactions, then rechecks current text and eligibility before
an atomic upsert. A changed source between encode and write raises
`EmbeddingSourceChanged`; a retry recomputes it.

`BgeM3EmbeddingProvider` implements the provider protocol through the project's
optional Sentence Transformers package. It is lazy and resolves either a
caller-supplied local directory or an existing Hugging Face cache snapshot. It
does not download weights automatically. Runtime device selection uses CUDA
only when the installed Torch reports CUDA availability; otherwise NumPy
performs vectorized cosine search on CPU. Local dense similarity also uses
CUDA tensors only under the same condition. The inspected environment has Torch
`2.14.1+cpu`, so CUDA is not available to this Python runtime even though
`nvidia-smi` detects an NVIDIA GeForce RTX 5070 Ti Laptop GPU. The BGE-M3
package and weights were not present; the adapter and real-model path have not
been end-to-end tested. Tests use deterministic local 1024-dimensional
providers.

### Lexical retrieval and query tokenization

`turns_fts` and `facts_fts` are queried with FTS5 `bm25()`. Lower BM25 values
rank first. Raw query syntax is never interpolated: retrieval extracts Unicode
word tokens, deduplicates them, quotes each token, and joins them with `OR`;
punctuation-only and empty token sets return no lexical candidates. SQL values,
history IDs, candidate limits, and temporal fact IDs are parameterized. The
turn route uses `turn_eligibility_predicate`; the fact route uses
`fact_eligibility_predicate` and intersects with the IDs returned by the
temporal engine for the explicitly selected view.

### Dense candidates and security

An in-memory index is rebuilt for a single history, reference type, and model.
It joins embeddings to source records, ignores orphaned, stale-digest, and
other-model rows, and checks shared eligibility before scoring. Fact dense
matches are further restricted to IDs in the temporal view. Results are
deterministically ordered by cosine descending, then reference ID. A corrupt
vector for the selected model raises a validation error instead of producing a
plausible but wrong score.

Turn and fact paths both check the source history and fail-closed gate policy.
Fact eligibility also checks its source turn, direct quarantine flag, and
parent fact for trigger rows. FTS candidates are filtered in SQL before they
leave the retrieval function. Fact expansion rechecks both the original source
turn and latest confirmation turn; if either required provenance row is
missing or ineligible, that fact result is omitted. Confirmation evidence is
deduplicated, and source turns are not filtered by the query's as-of time:
backdated facts may legitimately be reported by a later source turn.
Trigger rows are not generated in this phase. Existing eligible trigger rows
can be retrieval candidates only while their parent fact is in the selected
temporal view; evidence expands to the parent fact's original source turns.

These protections are tested independently for turn FTS, turn dense, fact FTS,
fact dense, trigger candidates, and source/confirmation evidence, including a
quarantine event added after indexing and a later `store` event. The query
classifier, reranker, Reader context builder, and answer generation do not yet
exist; no claim is made about those future routes until implemented and tested.

### Fusion and temporal views

Lexical and dense channels remain separate. RRF uses one-based ranks and the
standard formula

```text
RRF(d) = sum_c 1 / (k + rank_c(d))
```

with default `k = 60`; candidate pool sizes and output sizes are configurable.
Ties use reference ID. BM25 and cosine scores are retained separately and are
never added to each other. Turn and fact results are fused independently.

`hybrid_rag()` fuses `turn_lexical` and `turn_dense`, defaults to top 10, and
raises a clear dense-unavailable error if no eligible turn vectors exist; it
never labels sparse-only output as hybrid. `recency()` takes that same fused
top-k set and sorts it by `event_time DESC`, with stable session/turn/ID ties.
The implementation uses the same `event_time` field populated from the
project's deterministic session timestamp plus turn-index offset.

`retrieve_memory()` accepts an explicit `current`, `as_of`, or `history` view;
it does not classify the query. Temporal eligibility and intervals come from
`temporal.retrieve_facts()` and its existing `as_of_timestamp()` conversion.
Date-only values continue to mean 00:00 UTC at the start of that day. For
callers that need the complete temporal timeline rather than query-ranked
fact candidates, `retrieve_temporal_facts()` returns all eligible facts in
`valid_from` order for the requested history/view/slot and attaches provenance.
`retrieve_memory(view="history")` also returns the complete eligible timeline;
its fact candidate limit affects diagnostics but does not truncate that view.
Closed intervals carry `[valid: {from} -> {until}]` metadata. A backdated
fact's original source turn may have a later `event_time`; both dates are
preserved.

Retrieval diagnostics expose candidate counts per channel, model ID, vector
dimension, similarity device, RRF constant, candidate limits, and final result
counts only. Diagnostics do not contain candidate text or quarantined content.

### Unverified limits

- Real BGE-M3 output quality and actual model loading have not been verified;
  local weights and required optional packages are absent.
- CUDA preference/fallback logic is implemented but could not be exercised
  because the installed Torch build is CPU-only.
- Index matrices are history-scoped but currently loaded into memory on demand;
  large-history latency and memory use have not been benchmarked.
- Query classification, reranking, Reader integration, cache/budget, and
  benchmark execution remain later phases.

### Phase 4A hardening decisions

#### Dense-index freshness

`check_index_freshness(database, history_id, provider_or_model_id)` is a
read-only operation. It enumerates records that pass the shared turn/fact
eligibility predicates and compares their `(ref_type, ref_id)`, provider model
identity, non-NULL `content_hash`, and digest of current source text with
persisted embeddings. It includes eligible trigger rows because they are facts.
Quarantined or otherwise ineligible records are excluded from the required set.
It reports missing vectors, model identity mismatches, legacy NULL digests, and
changed source text without calling an encoder or changing database state.

Dense retrieval raises `DenseIndexStaleError` with an instruction to run
`index_history(database, history_id, provider)` when the required eligible
index is incomplete or incompatible. Indexing remains a separate idempotent
operation that regenerates only stale/missing vectors. This intentionally
rejects sparse-only degradation under a hybrid API. A legacy row with model
`BAAI/bge-m3` is not assumed to match a newly resolved BGE checkpoint and must
be indexed under the revision-specific identity.

#### Revision-aware local model identity

The BGE adapter resolves a Hugging Face cache snapshot from the configured
`refs/<revision>` metadata and uses the referenced commit hash in its identity
(`BAAI/bge-m3@<commit>`). A direct commit is accepted only when that exact
snapshot directory exists. Snapshot directory names are never sorted to guess
which checkpoint is intended; missing revision metadata fails clearly without
network access. A caller-supplied local directory receives a path-independent
SHA-256 fingerprint over sorted relative file names and file bytes. The
fingerprint is cached per provider instance and rechecked before local model
loading. This reads checkpoint files to establish identity but does not load
weights into memory; local checkpoint hashing can therefore add startup I/O.
Injected providers keep their own `model_id` unchanged. Different resolved
identities use separate embedding rows and are checked consistently for
query/document use, indexing, index loading, retrieval, and diagnostics.

#### Similarity candidate filtering

`InMemoryVectorIndex.search(allowed_ids=...)` resolves the allowed row indexes
before calculating dot products. CPU NumPy and available CUDA tensor paths
score only that selected matrix while retaining cosine scores, one-based rank,
descending score order, and reference-ID tie-breaks. Empty or unknown allowed
sets return no hits. CUDA is attempted when available; a runtime CUDA failure
is logged before the existing CPU fallback, rather than silently disabling
CUDA.

#### Trigger coverage

Trigger generation remains outside Phase 4A. Integration tests insert
schema-valid persisted trigger rows and verify FTS and dense retrieval,
parent-fact linkage, source and confirmation evidence, current/as-of behavior,
parent supersession, and sticky quarantine. Eligibility continues to be
centralized in `eligibility.py`; no trigger-specific security predicate or
generation lifecycle was added.

#### Retrieval snapshot and concurrency contract

`retrieve_memory()` and `retrieve_temporal_facts()` compute query vectors
before opening a database transaction, then use one deferred `BEGIN` read
transaction for index freshness, FTS candidates, dense source rows, temporal
projection, parent expansion, source/confirmation turns, and final eligibility
validation. WAL writers can commit while the reader continues to see the
snapshot established by its first read. Therefore an in-flight call may return
the fully consistent pre-commit state if quarantine commits after that
snapshot began; the next call sees the quarantine and fails closed. This is
SQLite snapshot semantics, not a process-local lock or a promise that an
already-running read sees future commits. Standalone FTS/dense search APIs use
one connection for their own operation.

No schema change was needed. Tests use temporary file-backed SQLite databases
and explicit events to coordinate reader/writer interleavings. BGE-M3 weights,
real-model retrieval quality, and CUDA execution remain unverified in the
available environment.

## Batch 2A groundwork

### Extraction `as_of` bound

An extracted fact's non-null `as_of` date must be on or before the source
turn's `session_date`; equality is allowed. A later date is rejected as an
invalid extraction response so bounded validation retries can request a
corrected response. This prevents a source turn from asserting a fact became
effective after the session that supplied it. Source session dates are parsed
lazily when a response contains a non-null `as_of`; an invalid source date is a
plain `ValueError` and is not retried. The extraction prompt and
`calculate_valid_from()` remain unchanged.

### Explicitly ungated turn baselines

Turn-only indexing and retrieval paths accept `gated=True` by default. Passing
`gated=False` is reserved for baseline experiments that intentionally use
unvetted turns. It affects turn sources only; fact sources, fact retrieval,
evidence turns, and `retrieve_memory()` remain gated. Production eligibility
continues to exclude turns with any quarantine event and does not require a
`store` event.

### Turns-only ingestion

`ingest_turns_only()` persists validated turns and their hash-chain provenance
without running a gate or extractor. It creates `pending_gate` jobs and writes
no gate events or facts. This path is for turn-only baselines; calling
`ingest_turns()` later upgrades the same turns through the normal gate and
extraction flow using replay detection.

### Store-event test fixture helper

`tests.helpers.record_store_event()` inserts a schema-valid clean `store`
event for direct-turn fixtures that model ordinary gated data. Tests that
exercise absent/pending/failed gate states, gate-event counts, database schema
immutability, or ingestion recovery continue to create their intended event
state directly.

## Batch 2B — gate and ingestion correctness

### Gate decision persistence

`evaluate_and_record_turn()` evaluates signals before opening the write
transaction, then `_record_gate_decision()` uses a fresh connection and
`BEGIN IMMEDIATE` for the event insert, quarantine flags on facts and
observations, the source turn's `base_trust`, and each affected single-valued
slot rebuild. The rebuild receives the same active connection. Triggered
failures at each propagation stage, including after a complete projection
rebuild, leave the persisted gate and projection tables equal to their
pre-call rows. No production change was needed for this atomicity audit.

The API accepts a database path and does not accept a caller-owned connection.
Its transaction is atomic per call; a caller cannot enlist it in an outer
transaction. Ingestion reconciles the job state after the gate call in a
separate transaction. On retry, `gate_events` is authoritative, so an event
committed before a job-state interruption is reused.

### Quarantine isolation

`ingest_turns()` accepts `quarantine_policy="raise"` by default or opt-in
`"isolate"`. The default retains the existing batch-abort behavior. Isolation
returns one result per deduplicated turn in deterministic order; a quarantined
turn has state `quarantined`, its gate event ID, zero extracted/applied counts,
and no observation IDs. Only a `QuarantinedTurnError` naming the turn being
processed is isolated. Other errors continue to propagate. All turns are
structurally validated before persistence, and all extraction responses are
validated and staged before temporal fact writes begin.

### Recorded-time propagation

When supplied, `recorded_at` is used for turn rows, gate events, job updates,
observations, projected fact `recorded_at`, and derived `expired_at`; the
ingestion and gate clocks are not read for those writes. When omitted, turn,
gate-event, and job timestamps retain their wall-clock behavior. Fact and
observation timestamps use the source turn's persisted `turns.recorded_at`
instead of a separate apply-time clock read. On staged recovery, an explicit
replay value takes precedence; otherwise the persisted turn timestamp is used.
The temporal replay identity check continues to ignore `recorded_at`.

### Role/source-type consistency (Batch 2B Part B)

`role` is the conversational speaker and `source_type` is the turn's
provenance category. Ingestion validates these dimensions together using an
asymmetric rule: trusted conversational provenance must match its speaker, but
any valid speaker may be conservatively labeled `tool` or `external_doc` so
the existing untrusted-source signal cannot be cleared by an inconsistent
trusted label.

The validation rules are: `user_turn` requires role `user`;
`assistant_turn` requires role `assistant`; `system_policy` requires role
`system`; role `tool` requires source type `tool` or `external_doc`; and source
types `tool` and `external_doc` accept any valid role. This allows a
conservative over-label as untrusted while rejecting a trusted provenance
claim that contradicts the declared speaker.

| role / source_type | `user_turn` | `assistant_turn` | `system_policy` | `tool` | `external_doc` |
|---|---|---|---|---|---|
| `user` | accept | reject | reject | accept | accept |
| `assistant` | reject | accept | reject | accept | accept |
| `system` | reject | reject | accept | accept | accept |
| `tool` | reject | reject | reject | accept | accept |

The matrix accepts 11 of the 20 valid role/source pairs and rejects 9. An
omitted `TurnInput.source_type` is derived during shared turn validation:
`user` → `user_turn`, `assistant` → `assistant_turn`, `system` →
`system_policy`, and `tool` → `tool`. The resolved string is used for
validation, payload hashing, persistence, validated-turn data, and extraction
context. `ingest_turn`, `ingest_turns`, and `ingest_turns_only` share this
pre-write validation.

`(role="user", source_type="external_doc")` remains valid. The role records
who submitted the chat turn; the source type records that its content came
from external material such as pasted API documentation or meeting notes.
Labeling such content `external_doc` fires the existing untrusted-source
signal. The gate still does not read `role`, and no positive user-origin signal
or predicate was added.

**Known limits:** both fields remain caller-supplied and unauthenticated. The
consistency rule rejects mismatched metadata; it cannot stop a caller from
labeling tool output as `(role="user", source_type="user_turn")`. The code
constructing turns must set `role` from the real channel. A positive
user-origin predicate is deferred trust-architecture work and is out of scope.
Direct SQL bypasses ingestion validation; schema-level constraints would be
needed to reject inconsistent direct inserts. No schema change or `CHECK`
constraint was added.

**Replay/migration consideration:** `source_type` participates in the payload
hash. A pre-existing `(assistant, user_turn)` or `(system, user_turn)` row is
invalid under the new rule. Replaying it without an explicit source type now
derives `assistant_turn` or `system_policy`, respectively, and its changed
payload hash raises `TurnConflictError`. Historical rows are not rewritten.

## Batch 3 — fail-closed eligibility and provenance enforcement

The eligibility rule and its consequences are fixed as follows:

- **D1:** A turn is eligible only when it has at least one `store` event and
  has no `quarantine` event. The six event-history cases are listed above.
- **D2:** Only the gated branch of the shared turn predicate requires a store
  event. Its ungated branch remains exactly `1 = 1`; fact paths and
  `retrieve_memory` always use gated eligibility.
- **D3:** A missing gate decision means “not yet evaluated”; a gate decision
  without a store event is ineligible; an actual quarantine event means
  “quarantined.” Missing store history alone never raises
  `QuarantinedTurnError`, sets quarantine flags/reasons, zeroes trust, or marks
  an ingestion job quarantined. Ingestion leaves a missing decision pending
  or reports a recovery error. Quarantine handling requires an actual
  quarantine event or an existing explicit `quarantined=True` input.
- **D4:** `temporal.insert_fact()` rejects a source turn that has neither a
  store nor quarantine decision with `UngatedSourceTurnError`, a
  `TemporalInvariantError` subclass that names the source turn. A fact must
  never be written before its source turn has a gate decision. Otherwise, an
  observation excluded by eligibility can be marked quarantined during slot
  reconciliation and may never become eligible later. A real quarantine
  decision still permits quarantined fact insertion, preserving the existing
  behavior.
- **Reconciliation:** The obsolete-row branch in `_reconcile_slot` now sets
  quarantine flags and reasons only for explicitly quarantined observations or
  observations whose source turn has an actual quarantine event. A pending
  observation remains stored and unquarantined, allowing it to rejoin the
  projection after a store event. The branch does not clear historical
  quarantine flags or delete a row merely because it is ineligible.
- **D5:** `src/engramOS/provenance.py` owns the shared role/source validation.
  `evaluate_turn()` validates persisted provenance before any evaluator call,
  including `fingerprint_inputs`; `_record_gate_decision()` validates it again
  immediately before inserting an event. Inconsistent or unknown provenance
  cannot receive a gate event and therefore cannot become eligible through the
  gate.
- **D6:** No migration or backfill was added. Existing turns without a store
  event, and facts sourced from those turns, are excluded from gated retrieval
  until the normal gate flow records a store event. They remain available to
  audit and to the explicitly ungated turn-only paths where requested.
- **D7:** The 2-of-3 rule, `evaluate_source_signal`, fact `trust`, `base_trust`
  semantics, and retrieval ranking are unchanged. The gate still does not
  treat `role` as a positive user-origin signal.

The schema was not changed. Direct SQL can bypass both ingestion provenance
validation and the D4 guard; schema constraints would be needed to block such
inserts. No schema constraints or indexes were added. There is no index on
`gate_events(turn_id)`, so the correlated eligibility checks may scan gate
history as the table grows.

**Trust limit:** `role` and `source_type` remain caller-supplied and
unauthenticated. Gate validation enforces consistency, not authenticity. A
caller can still label tool output as a consistent user turn. A positive
user-origin predicate is deferred trust-architecture work and was not added.

### Open decisions (not implemented)

1. Whether a positive user-origin signal counts toward the 2-of-3 or only weights trust, and how assistant/system turns are treated.
2. Trust tiers per source type and how fact `trust` is derived, and whether anything consumes it in ranking.
3. Whether `as_of` queries must filter raw turns by `event_time`.
4. The policy when a slot appears with both `single` and `multi` cardinality.
5. What `replaces` should do, especially retracting multi-valued facts.
6. The policy when `session_ts + turn_idx` event times of different sessions interleave.
7. Number-word and Unicode normalization behavior.
