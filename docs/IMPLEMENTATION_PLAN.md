# EngramOS Implementation Plan

**Reviewed:** 2026-10-31  
**Status:** Database foundation, temporal memory, trust/quarantine, local Phase 2B ingestion/extraction, and Phase 4A retrieval hardening are implemented and covered by SQLite integration tests. Real BGE-M3 execution, query classification, reranking, Reader integration, cache/budget, evaluation, and replay remain open.

This plan is based on the repository and the three requested specifications: `BUILD_PLAN.md`, `prompts.md`, and `schema.sql`. Other repository documents were not treated as specifications.

## 1. Current implementation

### Repository architecture

- `schema.sql` defines the SQLite tables for turns, facts, embeddings, gate events, ingestion recovery, LLM cache entries, evaluation runs, and FTS5 indexes. It also documents the intended temporal supersession algorithm.
- `src/engramOS/normalize.py` implements value and slot-name normalization.
- `src/engramOS/temporal.py` implements deterministic fact writes, interval reconciliation, and historical retrieval.
- `src/engramOS/eligibility.py` centralizes fail-closed turn, fact, and observation eligibility rules.
- `src/engramOS/gate.py` evaluates injected bridge/policy signals, applies the 2-of-3 rule, appends gate events, and atomically propagates quarantine and trust updates.
- `src/engramOS/extraction.py` preserves the Section A prompt, validates untrusted JSON against its exact extraction schema, and exposes a provider-independent injectable extractor protocol.
- `src/engramOS/ingestion.py` validates and stores turns, reuses gate-event history on recovery, stages validated extraction payloads, and applies facts sequentially through the temporal API.
- `src/engramOS/embeddings.py` defines the embedding provider contract, lazy local BGE-M3 adapter, 1024-dimensional float16 storage validation, content digests, and idempotent history indexing.
- `src/engramOS/vector_index.py` rebuilds a history-scoped in-memory cosine index from persisted vectors and filters stale, orphaned, cross-history, and ineligible source rows.
- `src/engramOS/retrieval.py` provides safe FTS5/BM25 search, dense search, RRF, explicit temporal views, source-evidence expansion, `hybrid_rag`, and `recency`.
- `src/engramOS/__init__.py` declares package version `0.1.0`.
- `pyproject.toml` declares Python 3.12+, runtime and optional ML/UI dependencies, and pytest settings.
- `tests/` contains SQLite integration coverage for the database foundation, temporal memory, trust/quarantine, extraction, ingestion/recovery, embeddings, vector indexing, and retrieval. Query interpretation, reranking, Reader integration, evaluation, and replay UI are not implemented.

The executable architecture is a small Python package with SQLite connection/schema management, deterministic temporal memory, shared eligibility rules, local embedding adapters, and provider-independent retrieval. No paid or live LLM integration exists.

### Status by component

| Component | Current status | Responsibility / gap |
|---|---|---|
| Schema | Foundation plus temporal ledger and embedding digest | Tables, FTS synchronization, turn immutability, observation ledger, and embeddings exist. `embeddings.content_hash` is additive; initialization adds it to older databases and treats legacy NULL digests as stale. |
| Normalization | Implemented and exercised | Normalizes value strings and slot names; temporal integration tests cover canonical values and normalized names. |
| Database access | Foundation implemented | `src/engramOS/db.py` centralizes SQLite connections, enables/verifies foreign keys, initializes the schema, backfills legacy facts, and provides rollback-safe transaction contexts. |
| Ingestion and extraction | Local infrastructure implemented | `ingestion.py` validates turns, creates deterministic IDs and per-history hash chains, deduplicates replays, integrates the canonical gate, stages validated responses, and resumes partial temporal writes. `extraction.py` implements the exact Section A output schema, original prompt instructions, strict validation, and bounded retries for transient adapter errors or malformed responses. No production model provider is connected. |
| Temporal memory | Core implemented and used by ingestion | `src/engramOS/temporal.py` validates and inserts/replays facts, rebuilds single-valued slot intervals, confirms and supersedes facts, handles backdating, and retrieves current/as-of/history views. Ingestion uses `insert_fact()` only; fuzzy slot matching remains unavailable until the specified local embedding/reranker components exist. |
| Trust and retrieval security | Gate and retrieval eligibility implemented; detector algorithms missing | `gate.py` supplies the canonical 2-of-3 gate and transactional quarantine. Retrieval FTS, dense turns/facts, trigger candidates, temporal filtering, evidence expansion, and RRF use shared `eligibility.py` predicates. Bridge/policy detector algorithms, reranker, and Reader remain unimplemented. |
| Retrieval and query handling | Phase 4A implemented and hardened; Phase 4B open | `embeddings.py`, `vector_index.py`, and `retrieval.py` implement freshness-enforced indexing, revision-aware local BGE-M3 identity, prefiltered cosine search, FTS5 BM25, RRF, explicit temporal views, evidence expansion, `hybrid_rag`, and `recency`. Composite retrieval uses one SQLite read snapshot. Real model inference is unverified because package/weights are absent. Classifier, reranker, and Reader are not implemented. |
| Cache and budget | Schema only | Cache table and aggregate view exist; no cache wrapper, complete request identity, atomic reservation, phase attribution, or cost calculation. |
| Evaluation and replay | Schema/specification only | No dataset split, evaluation runner, metrics, replay UI, or reproducibility records produced by code. |
| Tests and operator docs | SQLite retrieval tests added | Tests cover schema migration, vector serialization/indexing, FTS5/BM25, dense cosine search, RRF, temporal views, evidence, and quarantine across implemented channels. Exact current results are recorded in `docs/TESTING.md`. |

## 2. Environment and dependencies

- Project metadata requires Python 3.12 or newer. This milestone ran with the available Python 3.12.10 runtime and SQLite 3.49.1 with FTS5 support. The repository `.venv` was not used.
- `pyproject.toml` declares NumPy as a runtime dependency and Sentence Transformers/Torch in optional ML dependencies. NumPy 2.5.3 and Torch 2.14.1+cpu are installed; Sentence Transformers, FlagEmbedding, Transformers, and FAISS are absent. The local runtime reports no CUDA-capable Torch build despite an NVIDIA GeForce RTX 5070 Ti Laptop GPU and driver being visible to `nvidia-smi`. `pytest` is available in the current Python environment but was not run; the prescribed `unittest` runner was used.
- No `BAAI/bge-m3` weights were found in the standard Hugging Face cache. The adapter loads an explicit existing local directory or cached snapshot only; it never downloads weights automatically. Retrieval tests use deterministic fake 1024-dimensional vectors.
- No API client or environment/configuration handling is implemented. No paid API requests are needed for the first milestone.
- Git status could not be read because this workspace has no `.git` directory. Changes must therefore be reviewed by inspecting the files directly.

## 3. Important gaps, inconsistencies, and risks

| Finding | Practical consequence | Recommended correction |
|---|---|---|
| `turns` is described as immutable, but the schema has no update/delete protection. The gate must still be able to adjust `base_trust`. | Source text and provenance metadata can be accidentally rewritten; a blanket update ban would also block legitimate trust updates. | Block updates to every turn column except `base_trust`; block turn deletion. Test both blocked and allowed operations. |
| `PRAGMA foreign_keys=ON` in `schema.sql` applies only to the connection executing it. | Later connections can silently accept dangling references. | Make every connection through one factory that enables and verifies foreign keys. |
| Turn and fact FTS tables have insert triggers only. | Fact key changes/deletions can desynchronize FTS; future edits may produce stale search results. | Add FTS update/delete synchronization where the parent rows are mutable; keep turn content immutable. |
| Turn FTS indexes raw source text, while the schema has no materialized security status. | Quarantined rows remain present in the FTS and vector stores for audit, so every candidate query must apply sticky gate eligibility before returning content. | Implemented for turn/fact FTS, dense candidates, trigger candidates, and fact evidence expansion using shared predicates; future classifier/reranker/Reader routes must retain the same policy. |
| `facts` has no database exclusion constraint for interval overlaps. | Direct SQL can bypass the temporal engine and create invalid intervals. | Keep writes through `temporal.py`; its transaction rebuilds the slot projection and integration tests check positive duration, adjacency, and value normalization. A database overlap constraint is deferred to avoid a larger SQLite trigger design. |
| `prompts.md` specifies cache identity `(model, prompt_hash, temperature, cache_nonce)`, and the schema primary key matches it. | Provider, max output tokens, response format, tools, and other generation settings can differ while reusing one response. | Hash a canonical representation of the complete generation request and use it as the cache identity; keep nonce as an explicit bypass dimension. |
| `prompts.md` sends every query containing `before` to `history`. | Questions such as “What was my role before March 2023?” can be routed to a history listing instead of an as-of lookup. | Keep only unambiguous deterministic fast paths; send ambiguous temporal questions to the specified classifier and document any prompt/rule adjustment. |
| Date-only `as_of` semantics are not specified in the three source documents. | Boundary-day answers may differ depending on timezone and whether a date means its start or end. | Resolved for this implementation: interpret date-only values as midnight UTC at the start of the date; document and test the convention in `docs/DECISIONS.md` and `tests/test_temporal.py`. |
| `llm_cache` stores cost and aggregate totals, but there is no code for caps/reservations or per-run/phase attribution. | Aggregate 150-credit enforcement cannot protect the 67-credit test reserve and 14-credit contingency; failed/time-out calls may be undercounted. | Route every paid call through a budget manager, enforce the pre-evaluation ceiling of 69 credits and total cap of 150, and record estimates/actuals plus phase/run. Do not make paid calls without explicit approval. |
| Real BGE-M3 inference has not been run locally. | The adapter, indexing, and retrieval pipeline compile and pass fake-vector tests, but model weights, CUDA inference, and real semantic quality are unverified. | Install reviewed optional ML dependencies and provide local BGE-M3 weights before validating real dense retrieval; no automatic download is performed. |

## 4. Proposed phases and dependencies

| Phase | Scope | Depends on | Required verification |
|---|---|---|---|
| 1. Database foundation | Connection factory, schema initialization, foreign keys, transactions, turn immutability, FTS maintenance, normalization tests. | None | SQLite integration tests for initialization/idempotency, FK enforcement on every connection, rollback, allowed `base_trust` update, blocked source edits/deletes, FTS insert/update/delete, and normalization. |
| 2. Ingestion and temporal memory | Deterministic temporal facts and local turn-ingestion/extraction infrastructure are implemented. Production extraction provider and fuzzy canonicalization remain unavailable. | Phase 1 | Tests cover confirmations, supersession, backdating, same-day ordering, interval invariants, history isolation, malformed responses, gate/recovery behavior, deterministic batches, and partial-write replay. |
| 3. Trust and quarantine | Gate decision, persistence, trust update, fact/observation propagation, fail-closed shared eligibility, and gate provenance validation are implemented. Bridge/policy detection algorithms remain injected because source thresholds/algorithms are unspecified. | Phases 1–2 | Tests cover all eight signal truth-table rows, incomplete evaluators, event history, atomic rollback, trust, facts/observations, store-required eligibility, gate provenance, pending/quarantine exclusion across retrieval, sticky quarantine, and history isolation. Trigger generation, reranker, and Reader remain unimplemented. |
| 4A. Retrieval engine | FTS/dense retrieval, embedding indexing, RRF, temporal filters, evidence expansion, `hybrid_rag`, and `recency` are implemented and freshness/snapshot hardened. | Phases 1–3 | Fake-vector SQLite tests cover freshness, revision identity, filtered scoring, positive/negative triggers, snapshot concurrency, history/view isolation, evidence, and quarantine. Real BGE-M3 execution remains unverified. |
| 4B. Query interpretation and answering | Query view classification, reranking, Reader integration using the original versioned prompts, and answer generation. | Phase 4A | Classifier ambiguity/temporal-routing tests, reranking tests, prompt preservation/hash checks, and end-to-end quarantine-leak tests. |
| 5. Cache, budget, and cost ledger | Complete-request cache key, concurrent miss coordination, budget reservation, run/phase cost records and error accounting. | Phase 1; before paid requests | Cache identity variation tests (provider/model/settings/max tokens), concurrent cache/budget tests, failed/time-out charging behavior, phase cap and evaluation reserve tests. No live API test without explicit approval. |
| 6. Evaluation | Locked seeded split, baselines and Engram variants, security metrics, costs by phase, statistical reporting. | Phases 1–5 and dataset/config lock | Split hash/seed checks, baseline parity, security metric separation, deterministic local reproduction, attribution checks; paid evaluation only with explicit approval. |
| 7. Replay and reporting | Read-only replay UI, timeline, gate audit, comparative metrics, reproducibility records and docs. | Phases 1–6 | UI/data integration tests proving replay makes no API calls; verify recorded inputs/configuration reproduce displayed outputs. |

### Dependencies between work streams

Phase 1 is the foundation for all database work. The deterministic temporal core and local turn-ingestion/extraction infrastructure in Phase 2 are implemented. Phase 3 is a prerequisite for any Reader context built from retrieved memory. Phase 5 must be in place before paid API integration or evaluation. Phase 6 must use a locked test split and preserve the 22/78 dev/test split, seed 42, and 67-credit evaluation reserve described in `BUILD_PLAN.md`. Phase 7 consumes stored data and must not issue fresh model requests.

## 5. Milestone 1A — database foundation

**Milestone 1A — SQLite connection and integrity foundation (completed, no API calls).** Implemented a synchronous stdlib `sqlite3` connection layer to avoid adding a dependency for basic persistence. It enables/verifies foreign keys on each connection, initializes the schema, offers transaction contexts, protects immutable turn data while allowing `base_trust` updates, and keeps fact FTS synchronized. Five integration tests exercise real SQLite behavior.

Implementation remains staged: deterministic temporal memory is now layered on the tested foundation, while extraction, models, and user-facing features remain subsequent milestones.

## 6. Current milestone progress

- [x] Inspect repository structure, existing source, schema, package metadata, dependencies, tests, and Git availability.
- [x] Read `BUILD_PLAN.md`, `prompts.md`, and `schema.sql` in full.
- [x] Record current architecture, implementation status, gaps, risks, conflicts, dependencies, and test requirements above.
- [x] Implement Milestone 1A and run its local SQLite integration tests.
- [x] Inspect the implementation and record exact test results and remaining limitations.

**Milestone 2A — Temporal memory core (completed, no API calls).** Added deterministic fact writes and replay, a lossless observation ledger, atomic single-slot interval projection, confirmation and supersession metadata, date/timestamp retrieval, and shared quarantine eligibility for fact reads. Initialization backfills existing slot facts. Extraction, turn ingestion, and non-temporal retrieval routes remain out of scope.

- [x] Implement transactional insertion, confirmation, supersession, and historical backdating.
- [x] Normalize adjacent equal logical values and resolve exact timestamp ties deterministically.
- [x] Implement current, timestamp/date `as_of`, and history-scoped retrieval.
- [x] Exclude quarantined observations and source turns from temporal retrieval.
- [x] Add SQLite coverage for rollback, legacy backfill, randomized insertion order, history isolation, and quarantine.
- [x] Update decisions and testing documentation.

### Milestone 2A result

- Added `src/engramOS/temporal.py` and `src/engramOS/eligibility.py`; modified `src/engramOS/db.py` and `schema.sql` for legacy backfill and the additive observation ledger.
- Added `tests/test_temporal.py`. The full suite passed: **25 tests**, including legacy backfill, raw-turn FTS candidate filtering through the shared eligibility predicate, and randomized insertion order, using `python -m unittest discover -s tests -v` with the available Python 3.12.14 runtime and SQLite 3.53.1.
- No paid API requests were made; **0 credits consumed**. `pytest` was not installed or run.
- Known limits: historical provenance already coalesced before the ledger cannot be reconstructed; quarantine is sticky because `gate_events` has no explicit revocation state. Raw-turn and fact FTS/dense retrieval, persisted-trigger retrieval, `hybrid_rag`, and `recency` are implemented and tested with shared eligibility predicates. Trigger generation and query interpretation/answering remain future work. See `docs/DECISIONS.md` and `docs/TESTING.md`.

**Phase 3 — Trust and quarantine (completed for the currently specified gate core; no API calls).** Added a pure 2-of-3 decision function and canonical `evaluate_and_record_turn` API, injected bridge/policy evaluators with mutable-input fingerprint validation, deterministic source-provenance evaluation, append-only event writes, `base_trust` lowering, fact/observation quarantine, and transactional slot-projection rebuilding. The common eligibility predicates now verify history/source relationships; tests apply them to temporal facts and real raw-turn FTS candidates.

- [x] Test all eight signal combinations and reject missing or failed evaluations.
- [x] Persist each complete decision without overwriting prior events.
- [x] Reject caller-supplied evaluations through the public API and revalidate evaluator inputs before persistence.
- [x] Apply trust, fact, observation, and temporal projection changes in one transaction.
- [x] Verify sticky quarantine, later-store behavior, history isolation, immutability, and raw-turn FTS filtering.
- [x] Update `docs/DECISIONS.md` and `docs/TESTING.md` with policy and test limits.

### Phase 3 result

- Added `src/engramOS/gate.py`; updated `src/engramOS/eligibility.py` and `src/engramOS/temporal.py` for source/history checks and transactional slot rebuilding.
- Added `tests/test_gate.py`. Gate API hardening adds canonical orchestration, fabricated-evaluation rejection, invalid-result handling, source-result validation, and input-fingerprint race checks. Full suite: **42 tests passed** using `python -m unittest discover -s tests -v` with Python 3.12.14 and SQLite 3.53.1.
- Existing `gate_events` columns store binary signal outcomes, fired count, decision, event identity, and timestamp; schema triggers now reject event updates/deletes to preserve the audit history. No table migration was needed.
- No paid API requests; **0 credits consumed**. Bridge/policy detector implementations remain unimplemented because their algorithms and thresholds are absent from the source specifications. No dense, trigger, reranker, Reader, or evaluation routes were added.

### Milestone 1A result

- Added `src/engramOS/db.py` with a central connection factory, per-connection foreign-key enforcement, transactional schema initialization, and commit/rollback context managers.
- Updated `schema.sql` to block turn deletion and edits to all turn fields except `base_trust`; added fact FTS synchronization for key updates and deletion.
- Added `tests/test_db.py` covering schema initialization/idempotency, foreign keys on multiple connections, rollback, turn immutability with mutable trust, and FTS insert/update/delete behavior.
- Test command: `python -m unittest discover -s tests -v` using the available Codex Python 3.12.14 runtime. Result: **5 tests passed**.
- No paid APIs or credits were used. `pytest` and the project `.venv` are unavailable in this environment; tests ran with Python's built-in `unittest` and SQLite 3.53.1.
- Remaining after Milestone 1A: temporal core, ingestion, gate, retrieval, cache, budget manager, API client, and evaluation implementation. Temporal core is now delivered in Milestone 2A; the other items remain open.

## Phase 2B — Turn ingestion and extraction infrastructure

**Status: implemented locally; no live provider/API calls.** `src/engramOS/ingestion.py` validates immutable source turns, deterministic identity and ordering, invokes only the canonical gate API for a new decision, and uses the gate event log for recovery. `src/engramOS/extraction.py` retains the original Section A extraction instructions and exact output fields, validates responses before staging, and exposes an injectable provider-independent extractor protocol. The additive `ingestion_jobs` table stores progress and the validated extraction payload before any temporal writes.

- [x] Preserve source content and hash-chain provenance; reject conflicting turn-ID replays.
- [x] Reuse prior `store` events, stop on any quarantine event, and reconcile interruptions after gate persistence.
- [x] Validate every extraction candidate for a history batch before staging or fact writes.
- [x] Apply candidates in chronological order through `temporal.insert_fact()` using stable turn/fact-sequence observation IDs.
- [x] Resume partial fact batches and empty extractions without duplicate logical memory.
- [x] Test source roles/types, validation, gate behavior, ordering, temporal writes, and interruption points with file-backed SQLite.

The `ingestion_jobs` state path is `pending_gate → approved → staged → completed`; any state with a persisted quarantine event becomes `quarantined` and cannot be reopened. Gate failure leaves `pending_gate` with no gate event. Turn/job insertion and each state checkpoint are transactional; evaluation and extraction run outside SQLite write transactions. Gate history is authoritative and is reconciled on retry. Fact insertion and progress updates are separate transactions, so replay uses the same observation ID if a crash occurs between them.

The full command `python -m unittest discover -s tests -v` completed with **73 tests passing** using Python 3.12.10 and SQLite 3.49.1. Remaining work includes a production extraction provider after cache/budget support, bridge/policy detector algorithms, fuzzy slot canonicalization, Phase 4 retrieval, Phase 5 cache/budget, evaluation, and replay. A crash after extraction returns but before its validated response is staged may require extraction to run again; no fact writes have occurred at that point. Calls for a single history must be serialized; separate histories can be processed independently.

## Phase 4A — Hybrid retrieval engine

**Status: implemented and hardened with deterministic fake embeddings; real BGE-M3 inference unverified.** The retrieval API accepts an explicit `current`, `as_of`, or `history` view and does not classify queries. `embeddings.py` provides a lazy local-only BGE-M3 Sentence Transformers adapter, a provider protocol, 1024-dimensional vector validation, float16 BLOB serialization, history-scoped idempotent indexing, and a read-only freshness check. `content_hash` on embedding rows records the SHA-256 of the exact source text; the existing additive initialization migration remains unchanged, and legacy NULL rows are skipped until reindexed. Dense retrieval fails closed with an actionable stale-index error when any required eligible turn/fact (including a trigger) is missing, uses a different model identity, has a NULL digest, or has changed text.

`vector_index.py` rebuilds one history/reference-type NumPy matrix from matching model rows, validates vectors, checks digests, joins back to source records, and applies shared eligibility before scoring. Temporal `allowed_ids` are selected before CPU or CUDA similarity computation. CUDA is selected when the installed Torch reports availability; otherwise NumPy computes cosine similarity on CPU. `retrieval.py` quotes tokenized FTS5 terms, orders `bm25()` scores ascending, filters facts against the temporal projection, fuses lexical and dense ranks with `1/(60 + rank)` by default, expands fact hits to eligible source/confirmation turns, and provides hybrid and same-candidate recency baselines. `retrieve_memory()` and `retrieve_temporal_facts()` use one deferred SQLite read transaction for candidate selection, temporal metadata, evidence expansion, and final eligibility validation.

- [x] Add provider-independent BGE-M3 adapter and strict 1024-dimensional float16 vector validation.
- [x] Add content-digest indexing and compatibility migration for preexisting embedding tables.
- [x] Add history-scoped dense index rebuild and deterministic cosine ranking.
- [x] Add safe FTS5/BM25 turn/fact search and shared eligibility checks.
- [x] Add separate turn/fact RRF results, current/as-of/history filtering, and source evidence expansion.
- [x] Add strict dense-plus-lexical `hybrid_rag` top-10 and recency ordering over the identical candidate set.
- [x] Test sticky quarantine on turn FTS/dense, fact FTS/dense, trigger candidates, and source/confirmation evidence.
- [x] Validate that every eligible turn and fact/trigger has a matching model identity and current content digest before dense retrieval; stale indexes require `index_history()`.
- [x] Resolve BGE-M3 Hugging Face snapshots through refs metadata and identify local checkpoints by content fingerprint; never sort snapshots or reuse generic legacy BGE vectors.
- [x] Filter temporal allowed IDs before vector scoring and hold composite retrieval in one SQLite WAL-compatible read snapshot.
- [x] Add positive trigger retrieval/evidence tests and synchronized reader/writer snapshot tests for both composite retrieval APIs.
- [x] Keep classifier, reranker, Reader, evaluation, cache/budget, and paid calls outside this milestone.

### Phase 4A result

- Added `src/engramOS/embeddings.py`, `src/engramOS/vector_index.py`, and `src/engramOS/retrieval.py`; updated `schema.sql` and `src/engramOS/db.py` for the additive embedding source digest; added `tests/test_embeddings.py`, `tests/test_vector_index.py`, and `tests/test_retrieval.py`.
- The full command `python -m unittest discover -s tests -v` passed **99 tests** using Python 3.12.10 and SQLite 3.49.1 with FTS5. This includes all 73 tests from the Phase 2B baseline plus 26 new retrieval/embedding tests.
- Local environment: NumPy 2.5.3; Torch 2.14.1+cpu reports CUDA unavailable; `nvidia-smi` reports an NVIDIA GeForce RTX 5070 Ti Laptop GPU and driver 621.01; no BGE-M3 cache snapshot or Sentence Transformers/FlagEmbedding install was found. Real-model embedding and CUDA retrieval therefore remain unverified. No weights were downloaded and no API credits were consumed.
- Remaining: Phase 4B classifier/reranker/Reader; validation against locally supplied BGE-M3 weights and a CUDA-enabled ML environment; later cache/budget, evaluation, and replay phases.

### Phase 4A hardening progress

- Implemented read-only freshness reports and fail-closed dense retrieval for new eligible records, missing/legacy digests, changed fact text, and incompatible provider identities. `index_history()` remains the only repair operation and remains idempotent.
- Replaced snapshot directory sorting with Hugging Face `refs/<revision>` resolution. Local directories receive a path-independent SHA-256 fingerprint over relative file names and file contents. Existing `BAAI/bge-m3` vector rows do not match revision-specific identities and must be reindexed.
- Applied `allowed_ids` before CPU/CUDA dot products, and refactored composite retrieval to use a single deferred read transaction. New tests coordinate WAL writers with retrieval to verify a consistent snapshot; the next call observes a committed quarantine.
- Trigger generation remains outside Phase 4A. Positive integration coverage uses manually persisted, schema-valid trigger rows.
- No schema migration or new dependency was required. This milestone used deterministic fake vectors only; BGE-M3 model quality and CUDA execution/performance remain unverified.
- Baseline before these changes: **99 tests passed** with `python -m unittest discover -s tests -v`. The post-hardening full run passed **120 tests**; exact environment and coverage are recorded in `docs/TESTING.md`.

## Batch 2A — groundwork before trust-architecture changes

**Status: complete and verified locally.** Extraction validation rejects a
non-null `as_of` later than the source session date. Turn-only embedding and
retrieval APIs expose an explicit `gated` option that defaults to the existing
quarantine-aware behavior; `gated=False` is restricted to turn sources. The
new `ingest_turns_only()` path persists validated, hash-chained turns as
`pending_gate` jobs without gate events or facts, and normal ingestion can
upgrade those turns through the existing replay path. Direct-turn fixtures
that represent ordinary clean data use a shared genuine `store` event helper.

- [x] Reject extracted future `as_of` dates and verify bounded retry and ingest behavior.
- [x] Add and adopt the shared clean store-event fixture helper in ordinary direct-turn fixtures.
- [x] Add gated defaults and explicit ungated turn-only baseline paths without changing fact eligibility.
- [x] Add turns-only ingestion, pending-gate persistence, replay/conflict/order checks, and upgrade coverage.
- [x] Run and revert the temporary store-event readiness probe; restore the eligibility source byte-for-byte.
- [x] Verify the focused temporal/retrieval suite (**57 passed**) and full suite (**147 passed**).

At the Batch 2A checkpoint, this batch did not require a store event for
production eligibility and did not change the schema or temporal/ranking
behavior. Batch 3 later changed the current gated contract to require a store
event. The temporary readiness probe was reverted and its source hash matched
the pre-probe backup. No later batch had started at that checkpoint.

## Batch 2B — gate and ingestion correctness

**Status: complete.** Gate persistence rollback characterization, batch
quarantine isolation, explicit timestamp propagation, and the corrected
role/source consistency contract are implemented. `role` is the conversational
speaker; `source_type` is provenance. The asymmetric rule and its full 20-pair
matrix are documented in `DECISIONS.md` and validated before writes. The
derived default is role-specific. `(user, external_doc)` remains valid for a
user submitting external material; the existing gate source signal still
fires from `source_type` alone. The schema and gate implementation are
unchanged.

- [x] Audit gate transaction ownership and verify rollback at each propagation stage.
- [x] Enforce the five-part role/source rule and derive an omitted source type
  through shared validation.
- [x] Add matrix, single-turn, batch-atomicity, default/replay, shared-entry,
  and unchanged-gate tests without editing existing tests.
- [x] Add omitted-`recorded_at` assertions for projected fact timestamps,
  expiration, gate events, final job updates, source observations, and replay.
- [x] Preserve opt-in `quarantine_policy="isolate"` and the default `"raise"` behavior.
- [x] Preserve explicit `recorded_at` propagation and persisted source-turn
  timestamps for facts and observations when omitted.
- [x] Document caller-authentication and direct-SQL limits, plus the replay
  migration consideration.

The full suite passed with the baseline 158 tests plus 8 new test methods.
No schema or eligibility change was made. No Batch 3 work was started.

## Batch 3 — fail-closed eligibility and provenance enforcement

**Status: complete and verified locally.** Gated turn eligibility now requires
at least one `store` event and no `quarantine` event; ungated turn-only paths
still use exactly `1 = 1`. Missing gate decisions remain distinct from actual
quarantine. `temporal.insert_fact()` rejects a source turn with no gate
decision, and slot reconciliation does not mark a pending observation
quarantined merely because it is ineligible. The shared role/source rule is
validated before gate evaluators run and again before gate persistence.

- [x] Add hard-coded eligibility cases and pending/quarantine exclusion tests
  across gated retrieval surfaces, including turn-only ungated behavior.
- [x] Require a gate decision before fact insertion while preserving actual
  quarantine behavior and observation replay.
- [x] Keep missing decisions pending or recoverable at ingestion checkpoints;
  verify they are never isolated or persisted as quarantine.
- [x] Validate role/source provenance before evaluator calls and recheck it
  immediately before writing a gate event.
- [x] Audit SQL reads across retrieval, embeddings, vector indexing, and
  temporal paths; retrieval-facing reads use shared eligibility predicates.
- [x] Preserve the schema, gate signal semantics, trust semantics, ranking,
  timestamps, and quarantine policy; add no migration or index.
- [x] Run the prescribed focused and full test commands and record measured
  counts in `docs/TESTING.md`.
- [x] Record the legacy no-store consequence, caller-supplied provenance
  limit, missing gate-event index risk, and seven open decisions in
  `docs/DECISIONS.md`.

The Batch 3 baseline was 166 passing tests. Sixteen test methods were added;
seven existing gate test fixtures received P1 store-event setup updates.
Three event-count assertions now compare against a captured pre-operation
baseline, preserving the original count and append-only intent without
weakening behavior. Final verification passed **70
gate/ingestion tests**, **65 temporal/retrieval tests**, and **182 tests** in
the full suite. No schema change, migration, backfill, new dependency, paid
call, model download, or Batch 4 work was performed.
