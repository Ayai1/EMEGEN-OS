# EngramOS Implementation Plan

**Reviewed:** 2026-11-03
**Current status (2026-11-03, post-hoc v2 complete):** Phase 1A–1D and the local provider integration are complete. The v1 suite remains locked at 23 histories/80 turns/38 questions; the separate held-out suite is locked at 8 histories/12 turns/8 questions. The user confirmed 361/361 tests before live evaluation. Post-hoc v2 persisted 38 Engram rows on v1 and 32 rows on the held-out suite, with 0 provider failures and 0 local credits. The running Q4 server was verified at 32/53 GPU layers, context 8192, Flash Attention enabled, q8_0 K/V cache, and low reasoning. Q3 remains hash-verified and not loaded. V1 outputs remain unchanged. Detailed results and caveats are in `docs/DECISIONS.md` and `docs/TESTING.md`.

**Plan supersession:** The earlier Meta API smoke, rate-card, and hosted-credit steps remain documented below as history, but are obsolete under the zero-credit local-backend plan. Do not run `scripts/smoke_test.py` for this task.

This plan is based on the repository and the three requested specifications: `BUILD_PLAN.md`, `prompts.md`, and `schema.sql`. Other repository documents were not treated as specifications.

## 1. Current implementation

### Repository architecture

- `schema.sql` defines the SQLite tables for turns, facts, embeddings, gate events, ingestion recovery, LLM cache entries, evaluation runs, additive per-question results, and FTS5 indexes. It also documents the intended temporal supersession algorithm.
- `src/engramOS/normalize.py` implements value and slot-name normalization.
- `src/engramOS/temporal.py` implements deterministic fact writes, interval reconciliation, and historical retrieval.
- `src/engramOS/eligibility.py` centralizes fail-closed turn, fact, and observation eligibility rules.
- `src/engramOS/gate.py` provides deterministic source, policy, and bridge signals for the authored suite, applies the unchanged 2-of-3 rule, appends gate events, and atomically propagates quarantine and trust updates.
- `src/engramOS/extraction.py` preserves the Section A prompt, validates untrusted JSON against its exact extraction schema, and exposes a provider-independent injectable extractor protocol.
- `src/engramOS/ingestion.py` validates and stores turns, reuses gate-event history on recovery, stages validated extraction payloads, and applies facts sequentially through the temporal API.
- `src/engramOS/embeddings.py` defines the embedding provider contract, lazy local BGE-M3 adapter, 1024-dimensional float16 storage validation, content digests, and idempotent history indexing.
- `src/engramOS/vector_index.py` rebuilds a history-scoped in-memory cosine index from persisted vectors and filters stale, orphaned, cross-history, and ineligible source rows.
- `src/engramOS/retrieval.py` provides safe FTS5/BM25 search, dense search, RRF, explicit temporal views, source-evidence expansion, `hybrid_rag`, and `recency`.
- `src/engramOS/query_classifier.py`, `reranking.py`, and `reader.py` define strict provider-independent query, candidate ordering, and evidence-grounded answer contracts.
- `src/engramOS/pipeline.py` connects classification, constrained retrieval, optional reranking, fresh-snapshot revalidation, and immutable Reader context construction without persisting generated answers; its baseline helper is evaluation-only.
- `src/engramOS/__init__.py` declares package version `0.1.0`.
- `pyproject.toml` declares Python 3.12+, runtime and optional ML/UI dependencies, and pytest settings.
- `tests/` contains SQLite integration coverage for the database foundation, temporal memory, concrete gate scenarios, extraction, ingestion/recovery, embeddings, vector indexing, retrieval, Batch 4 Stages 1–4, offline evaluation persistence, attack metrics, and the expanded suite. Classifier, reranker, and Reader providers remain injected; replay is read-only and offline.

The executable architecture includes cache-backed `MetaQueryClassifier`, `MetaReaderProvider`, and `MetaFactExtractor` adapters, retained for compatibility and offline tests. Phase 1 used cached fake providers. Phase 2A used the verified local OpenAI-compatible server for six measurements after explicit user approval; Phase 2B local provider adapters and the full model-backed evaluation remain pending. The old hosted smoke script remains only as historical work; no Meta API request is authorized by the revised plan.

### Status by component

| Component | Current status | Responsibility / gap |
|---|---|---|
| Schema | Foundation plus temporal ledger and embedding digest | Tables, FTS synchronization, turn immutability, observation ledger, and embeddings exist. `embeddings.content_hash` is additive; initialization adds it to older databases and treats legacy NULL digests as stale. |
| Normalization | Implemented and exercised | Normalizes value strings and slot names; temporal integration tests cover canonical values and normalized names. |
| Database access | Foundation implemented | `src/engramOS/db.py` centralizes SQLite connections, enables/verifies foreign keys, initializes the schema, backfills legacy facts, and provides rollback-safe transaction contexts. |
| Ingestion and extraction | Local infrastructure plus optional cached adapter | `ingestion.py` validates turns, creates deterministic IDs and per-history hash chains, deduplicates replays, integrates the canonical gate, stages validated responses, and resumes partial temporal writes. `extraction.py` implements the exact Section A schema and validator. `MetaFactExtractor` is available for an optional secondary check; ingestion remains injectable and offline by default. |
| Temporal memory | Core implemented and used by ingestion | `src/engramOS/temporal.py` validates and inserts/replays facts, rebuilds single-valued slot intervals, confirms and supersedes facts, handles backdating, and retrieves current/as-of/history views. Ingestion uses `insert_fact()` only; fuzzy slot matching remains unavailable until the specified local embedding/reranker components exist. |
| Trust and retrieval security | Deterministic authored-suite detectors implemented | `gate.py` retains the canonical 2-of-3 gate and transactional quarantine. `DefaultSignalEvaluator` adds a source-provenance signal, an eligible active-constraint contradiction check, a deterministic prefix-overlap bridge check, and a fingerprint over all detector inputs. See `docs/DECISIONS.md`; detector thresholds are benchmark-specific. |
| Retrieval and query handling | Phase 4A and Batch 4 Stages 1–4 implemented; Meta adapters available | Phase 4A remains in `embeddings.py`, `vector_index.py`, and `retrieval.py`. Batch 4 adds strict query classification, exclusive `as_of` boundaries, candidate-only reranking, the evidence-grounded Reader, and the guarded `pipeline.answer_query()` flow with fresh-snapshot revalidation. `MetaQueryClassifier` and `MetaReaderProvider` provide cached production adapters; reranking remains injected. |
| Cache and budget | Core manager plus offline-tested adapters | Complete request identity, concurrent miss coalescing, atomic total/phase reservations, failure charging, actual-cost accounting, and run/phase summaries remain unchanged. Hosted Meta adapters are retained but obsolete for this run; the local endpoint adapter is a later phase. |
| Evaluation and replay | V1 preserved; v2 locked and replay implemented | `eval/suite_v2.json` extends the original 20 questions with 10 benign cases and 8 attacks. `scripts/run_eval.py` reports separate attack exposure/adoption rates with exact intervals and persists per-question outcomes. `scripts/replay.py` is SQLite read-only. Offline outputs validate wiring only. |
| Tests and operator docs | Deterministic SQLite and Batch 4 integration tests | Tests cover schema migration, vector serialization/indexing, FTS5/BM25, dense cosine search, RRF, temporal views, evidence, quarantine across implemented channels, and the Stage 4 query pipeline. Exact current results are recorded in `docs/TESTING.md`. |

## 2. Environment and dependencies

- Project metadata requires Python 3.12 or newer. In this run, `python` resolved to the Windows Store alias and could not start; `.venv\Scripts\python.exe` ran the suite with SQLite FTS5 support.
- `pyproject.toml` declares NumPy as a runtime dependency and Sentence Transformers/Torch in optional ML dependencies. The project `.venv` has Sentence Transformers and CPU Torch installed. The cached BGE-M3 checkpoint encoded a normalized 1024-dimensional query on CPU. Retrieval tests still use deterministic fake vectors; this does not establish retrieval quality.
- `BgeM3EmbeddingProvider` resolves an existing local directory or Hugging Face snapshot and does not download weights automatically. The checkpoint used for the Step 0 smoke check is `BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181`.
- The Meta Model API adapters read `MODEL_API_KEY` from the environment in normal construction and use the official Standard or Contributor rate card selected by model ID. No live API request has been made. `ENGRAMOS_CREDITS_PER_USD` is required because the campaign's conversion is not recorded in the repository.
- Git metadata is available in this workspace. The tree already contained unrelated modified and untracked files before this step; inspect the working tree before attributing those changes to this plan.

## 3. Important gaps, inconsistencies, and risks

| Finding | Practical consequence | Recommended correction |
|---|---|---|
| `turns` is described as immutable, but the schema has no update/delete protection. The gate must still be able to adjust `base_trust`. | Source text and provenance metadata can be accidentally rewritten; a blanket update ban would also block legitimate trust updates. | Block updates to every turn column except `base_trust`; block turn deletion. Test both blocked and allowed operations. |
| `PRAGMA foreign_keys=ON` in `schema.sql` applies only to the connection executing it. | Later connections can silently accept dangling references. | Make every connection through one factory that enables and verifies foreign keys. |
| Turn and fact FTS tables have insert triggers only. | Fact key changes/deletions can desynchronize FTS; future edits may produce stale search results. | Add FTS update/delete synchronization where the parent rows are mutable; keep turn content immutable. |
| Turn FTS indexes raw source text, while the schema has no materialized security status. | Quarantined rows remain present in the FTS and vector stores for audit, so every candidate query must apply sticky gate eligibility before returning content. | Implemented for turn/fact FTS, dense candidates, trigger candidates, and fact evidence expansion using shared predicates; future classifier/reranker/Reader routes must retain the same policy. |
| `facts` has no database exclusion constraint for interval overlaps. | Direct SQL can bypass the temporal engine and create invalid intervals. | Keep writes through `temporal.py`; its transaction rebuilds the slot projection and integration tests check positive duration, adjacency, and value normalization. A database overlap constraint is deferred to avoid a larger SQLite trigger design. |
| Cache identity must include all generation settings. | A prompt-only hash can reuse a response across different providers, output caps, or structured-output settings. | Resolved in `llm_cache.py`: hash canonical provider/model/prompts/temperature/output cap/response format/settings and retain nonce for explicit bypass. |
| `prompts.md` sends every query containing `before` to `history`. | Questions such as “What was my role before March 2023?” can be routed to a history listing instead of an as-of lookup. | Keep only unambiguous deterministic fast paths; send ambiguous temporal questions to the specified classifier and document any prompt/rule adjustment. |
| Date-only `as_of` semantics are not specified in the three source documents. | Boundary-day answers may differ depending on timezone and whether a date means its start or end. | Resolved for this implementation: interpret date-only values as midnight UTC at the start of the date; document and test the convention in `docs/DECISIONS.md` and `tests/test_temporal.py`. |
| Earlier Meta live smoke call has not been made. | The revised task has no hosted access or Meta credits; the former pricing/card/spend step no longer applies. | Leave the historical Meta smoke decision in this document, do not call Meta, and continue with the verified local backend only after its checkpoint approval. |
| Real BGE-M3 inference has only passed a local encoding smoke check. | The adapter returns finite 1024-dimensional vectors on CPU, but corpus indexing, retrieval quality, throughput, and benchmark quality have not been evaluated with real weights. CUDA is unavailable in the installed Torch build. | Use the local checkpoint and CPU adapter for a measured dev run; do not treat the smoke check as retrieval-quality evidence. |

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

**Status: implemented and hardened with deterministic fake embeddings; local model smoke check passed.** The retrieval API accepts an explicit `current`, `as_of`, or `history` view and does not classify queries. `embeddings.py` provides a lazy local-only BGE-M3 Sentence Transformers adapter, a provider protocol, 1024-dimensional vector validation, float16 BLOB serialization, history-scoped idempotent indexing, and a read-only freshness check. `content_hash` on embedding rows records the SHA-256 of the exact source text; the existing additive initialization migration remains unchanged, and legacy NULL rows are skipped until reindexed. Dense retrieval fails closed with an actionable stale-index error when any required eligible turn/fact (including a trigger) is missing, uses a different model identity, has a NULL digest, or has changed text. A local CPU smoke check loaded the cached checkpoint and produced finite 1024-dimensional query/document vectors; semantic retrieval quality remains unmeasured.

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
- Local environment at Phase 4A completion: NumPy 2.5.3; Torch 2.14.1+cpu reported CUDA unavailable; `nvidia-smi` reported an NVIDIA GeForce RTX 5070 Ti Laptop GPU and driver 621.01; no BGE-M3 cache snapshot or Sentence Transformers install was found at that time. See the 2026-11-03 progress update for the subsequent CPU model smoke check. No API credits were consumed.
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

## Batch 4 — Stage 1: Query classification

Stage 1 adds `engramOS.query_classifier.classify_query()` with immutable,
strictly validated results, deterministic high-confidence routing, an
injectable provider protocol, and clarification/unsupported outcomes that do
not reach retrieval. The original Section C prompt is preserved; a versioned
structured-output extension documents the classifier schema and hash basis.
`temporal.TemporalBoundary` adds exact exclusive `as_of` left-limit semantics
for facts without epsilon arithmetic. Stage 4 now applies temporal constraints
to standalone turns and history ranges and connects reranking, Reader-safe
context construction, and end-to-end orchestration. Detailed Stage 1 interface
and limitations are in
`docs/BATCH4_STAGE1.md`; measured Stage 1 test results are in
`docs/TESTING.md`.

## Batch 4 Stage 4 — query pipeline integration

Stage 4 connects classification, constrained retrieval, optional candidate-only
reranking, fresh-snapshot candidate revalidation/rehydration, and the Reader in
`engramOS.pipeline.answer_query()`. History ranges select changes by
`valid_from`; a preceding active state is included as a labeled baseline when
it explains the opening state. Standalone turn event-time filters are applied
in FTS and dense retrieval before top-k selection and checked again during
rehydration. The Reader gets an immutable database-free context. Later
as-of source/confirmation text is withheld while retrospective recording
metadata remains. The snapshot used to materialize that context is the
authorization point; later quarantine changes do not retroactively change an
already materialized context.

Detailed contracts, prompt hashes, test counts, and remaining limitations are
recorded in `docs/BATCH4_STAGE4.md` and `docs/TESTING.md`.

## Revised local-only plan status — Phase 1 complete — 2026-11-03

- **Phase 1A complete:** `DefaultSignalEvaluator` is deterministic and local.
  Its exact policy matcher, bridge overlap, `tau=0.60`, source behavior,
  pre-extraction timing, and fingerprint inputs are recorded in
  `docs/DECISIONS.md`. The full suite grew from 310 to 318 passing tests.
- **Phase 1B complete:** the locked authored suite has 5 histories, 50 turns,
  and 20 queries. Fixture extraction is fed sequentially through
  `ingest_turns()`. All four systems share the same cached fake Reader and
  grader; fake vectors use `eval/offline-fake.sqlite3`. The only schema
  addition is `eval_question_results`; current `eval_runs` columns remain
  unchanged. The baseline helper is intentionally ungated and unrevalidated,
  and `answer_query()` remains the production path. The post-edit full suite
  passed 326 tests (8 above the Phase 1A count).
- **Phase 1C complete:** `query_classifier.py` routes deterministic current,
  as-of, before-boundary, history, relative-date, unsupported, and
  clarification shapes before the provider fallback. The authored suite
  fallback set has two questions; a new regression test proves a representative
  present-state query makes zero provider calls. The full suite now passes 328
  tests.
- **Phase 1D complete:** `scripts/replay.py` reads the specified SQLite tables
  through a read-only connection and renders slot intervals, gate signals,
  saved per-system answers, and matching cache metadata. Its test patches both
  network socket creation and `cached_generate` to fail. The full suite passes
  329 tests, one above the 328-test Phase 1C result.
- **Local backend verification and Phase 2A complete:** the Q4 GGUF, b12058
  CUDA server/runtime archives, and pre-approved Q3 fallback match their
  published SHA-256 digests. The server version/help and exact flags were
  recorded. Q4 was measured with `35/53` layers on GPU, `18/53` on CPU, one
  server slot, an 8,192 context, Flash Attention, and q8_0 K/V cache. The
  three Reader-sized calls took 77.4–99.8 seconds, above the ~60-second
  target. The verified Q3 rung is proposed but has not been loaded; wait for
  the user's approval before changing quant. Full details and commands are in
  `docs/DECISIONS.md` and `docs/TESTING.md`.
- **Hosted smoke is obsolete:** the previous Meta smoke/rate-card instructions
  remain in the historical sections above, but no hosted call or API key is
  part of the revised task. Offline fake run credits: 0.

## Phase 3 locked local evaluation — complete — 2026-11-03

- The locked suite hash remains
  `6b9a2a48e7d2dec311e7064de4b2adbb717ab8a67372a7590f1595f450a2c823`.
- The 3-question canary (`q4-canary-classifier1024-v1`) completed all 12
  question/system rows, graded 12/12 correct, and recorded 0 provider failures.
- The approved infrastructure-only change raised classifier `max_tokens`
  from 256 to 1,024 after a hidden-reasoning `finish_reason=length`; the Reader
  remains at 2,048. No other model, prompt, detector, grading, or suite setting
  changed.
- Initial full run `q4-locked-classifier1024-v1` stopped at
  `city_asof_boston` / `hybrid_rag_date_filter` after a **180,078 ms** Reader
  timeout. The runner persisted **7/152** question/system rows (six completed,
  one timeout), recorded 0 credits, and stopped without retrying. The server
  health endpoint remained `ok`.
- The timeout/cache-identity and resume regressions pass; the full suite after
  those changes passed **349 tests**. Run
  `q4-locked-classifier1024-resume300-v1` has now seeded the six successful
  rows, marked the original timeout row superseded without changing it, and
  resumed at the failed question in the same database. It uses 300-second
  calls and logs progress to
  `logs/q4-full-classifier1024-resume300-progress.jsonl`. The original run and
  its timeout failure remain available for audit.
- The continuation finished with **152/152** question/system rows across all
  38 locked questions and four systems. It recorded **0 provider failures**,
  **0 local credits**, and no stop reason. The prior timeout row remains in
  the original run and is marked superseded; its labeled 300-second rerun
  completed. Exact accuracy intervals, security metrics, latency, classifier
  fallback queries, and attack cases are recorded in `docs/TESTING.md` and
  `eval/results_q4-locked-classifier1024-resume300-v1.json`.
- The attack gallery was generated from the read-only evaluation database by
  `scripts/build_attack_gallery.py` and saved as
  `eval/attack_gallery_q4-locked-classifier1024-resume300-v1.json`. A results-
  first README draft is present at the repository root. No suite, prompt,
  detector, sampling, or grading changes followed the live run.
- Cached-only failure triage and a separate `RE-GRADE-1 manual-semantic-v1`
  sensitivity artifact are complete. The original results and grade remain
  unchanged. A gallery-builder regression was added, and the full suite now
  passes **350 tests**. Trust-aware supersession, classifier routing, prompt,
  fixture-label, and grading proposals remain unimplemented pending a
  separately labeled run.

## Post-hoc v2 — preflight pending test verification — 2026-11-03

- The new `eval/suite_v2_heldout.json` was created before v2 inference and is
  locked at SHA-256
  `5c64470c0773d9bb4b2d672b1a207c24959784f7cb5e7539497b02ef03dc5b85`;
  the original v1 suite hash remains unchanged.
- Implemented provenance-aware projection: `user_turn` source authority
  outranks `external_doc` and `tool` regardless of equal numeric trust; blocked
  observations stay in the ledger; current Reader rehydration suppresses the
  raw conflicting turn. Implemented date-free classifier failure override
  while preserving deterministic explicit-date/knowledge-time routes.
- Added 11 focused v2 tests; all passed during the full-suite run. The suite
  discovered 361 tests, with 360 passing and one assertion failure before its
  authorized edit. The prior inequality condition in
  `tests/test_expanded_authored_suite.py` now checks adoption <= exposure and
  that only `poison_adaptive_user` is both reached and adopted. The fake run
  measured 1/11 reached and 1/11 adopted for Engram.
- The post-hoc runner records attack metrics both including and excluding
  `poison_adaptive_user`, and reports that compromised trusted channel by
  itself for each system.
- The held-out offline wiring run completed 32 rows with zero provider
  failures and zero credits. No v2 DB copy, live local model call, or live v2
  result was started. The full suite was not rerun after the authorized
  assertion edit. The user will run the official full-suite command and
  provide its result before the v2 preflight checklist.

## Post-hoc v2 live evaluation — complete — 2026-11-03

- The user supplied the result of the prescribed full-suite command:
  **361/361 passing**. The suite was not rerun during the live run.
- Preflight hashes matched for the original v1 suite and the locked held-out
  suite. The server's metadata, startup log, and running process settings
  matched the approved Q4 configuration. Local mode selected only local
  classifier and Reader providers; no fake fallback was available.
- Runs `q4-post-hoc-v2-engram-v1` and `q4-post-hoc-v2-heldout-v1` completed in
  `eval/local-q4-posthoc-v2.sqlite3`, with 70 v2 result rows, no provider
  failures, and 0 credits. Machine-readable outputs and metrics are listed in
  `docs/TESTING.md`.
- The six named primary v1 files have modification times before the v2 DB
  creation time. The locked suite hashes have pre-run lock records; no
  independent pre-run digests for the other v1 artifacts were found in docs
  or logs. The v1 `-shm` sidecar timestamp is 0.0009981 seconds later than
  v2 DB creation. No code, test, suite, prompt, detector, or grading edits
  were made after live v2 inference began.

## Hackathon safety and first performance options — complete — 2026-11-04

- Phase A added scope/security disclosures, a third-party inventory, placeholder-only `.env.example`, ignore rules, and post-hoc artifact hashes. The tracked-path scan found no tracked LongMemEval data, SQLite files, logs, `.env`, or GGUF assets; the secret-pattern scan found no matches.
- Phase B added opt-in v3 Reader evidence labeling, opt-in strict handling for untrusted turns, and human-readable gate reasons without a schema change. Unknown source types were already rejected; a focused regression test records that behavior.
- Phase C added v3-only date/history prompt formatting, opt-in exact slot aliasing, configurable post-hoc token/reasoning caps with unchanged defaults, and optional suite SHA-256 validation before provider setup.
- Full suite after Phase A: 361 tests passed. After Phase B: 366 passed. After Phase C: 370 passed. Commands used: `\.venv\Scripts\python.exe -m unittest discover -s tests -v`.
- No model calls, server start, network access, or embedding loads were performed. Locked evaluation artifacts were not edited.
