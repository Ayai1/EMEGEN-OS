# EMEGEN-OS Architecture

## Contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Repository map](#2-repository-map)
3. [Architecture and data flow](#3-architecture-and-data-flow)
4. [SQLite data model](#4-sqlite-data-model)
5. [Ingestion, provenance, and gate](#5-ingestion-provenance-and-gate)
6. [Temporal facts and projection](#6-temporal-facts-and-projection)
7. [Retrieval](#7-retrieval)
8. [Query classification](#8-query-classification)
9. [Reader and answer synthesis](#9-reader-and-answer-synthesis)
10. [LLM provider layer](#10-llm-provider-layer)
11. [Evaluation harness](#11-evaluation-harness)
12. [Configuration and run instructions](#12-configuration-and-run-instructions)
13. [Tests and design assertions](#13-tests-and-design-assertions)
14. [Recorded results](#14-recorded-results)
15. [Limitations and known issues](#15-limitations-and-known-issues)
16. [Open questions](#16-open-questions)

## 1. Purpose and scope

EMEGEN-OS (the Python package is named `engramOS`) stores conversation turns, extracts normalized slot facts, records temporal versions and provenance, screens incoming turns with a deterministic three-signal gate, retrieves eligible evidence, and asks an injected Reader to produce a cited answer or abstain. The persisted turn log and fact projection are scoped by `history_id`. (`schema.sql:16-30`; `src/engramOS/ingestion.py:164-218`; `src/engramOS/gate.py:158-177`; `src/engramOS/pipeline.py:52-68`.)

The system does not authenticate that a caller-supplied `source_type` truthfully identifies the origin of a message. It does not validate semantic entailment of Reader answers: the Reader validates response shape and citation membership, while the evaluation grader applies declared string rules. (`src/engramOS/ingestion.py:58-70,780-792`; `src/engramOS/reader.py:1-6,720-770`; `scripts/run_eval.py:1265-1287`.) The locked evaluation uses fixture facts, so its scores describe memory, temporal, retrieval, and gate behavior rather than fact-extraction quality. (`eval/loader.py:135-161`; `README.md:66-72`.)

## 2. Repository map

| Path | Contents |
|---|---|
| `src/engramOS/` | SQLite access, ingestion, extraction contracts, provenance/eligibility, gate, temporal projection, embeddings, retrieval, classifier, Reader, provider adapters, and pipeline. Representative entry points are `ingestion.py`, `gate.py`, `temporal.py`, `retrieval.py`, and `pipeline.py`. (`src/engramOS/ingestion.py:1`; `src/engramOS/pipeline.py:1`.) |
| `tests/` | `unittest` tests for database, temporal logic, gate, ingestion, embeddings, retrieval, providers, pipeline, evaluation, and replay. (`docs/TESTING.md:93-118`; `tests/test_temporal.py:1`; `tests/test_local_llama.py:1`.) |
| `scripts/` | Evaluation runners, server launcher, local benchmarks/smoke checks, replay, resource monitor, and gallery builder. (`scripts/run_eval.py:1-6`; `scripts/replay.py:1-5`.) |
| `eval/` | Suite loader and authored suite JSON, locked result summaries, gallery and re-grade artifacts, plus local SQLite files. The result and suite artifacts are run records, not package code. (`eval/loader.py:1-17`; `docs/TESTING.md:906-915,1241-1246`.) |
| `docs/` | Technical decisions, implementation status, test records, phase notes, and this architecture description. (`docs/DECISIONS.md:1`; `docs/TESTING.md:1`; `docs/IMPLEMENTATION_PLAN.md:1`.) |
| `logs/` | Generated server traces, progress JSONL, and benchmark/resource output; recorded examples are identified in `docs/TESTING.md`. (`docs/TESTING.md:879-886,1203-1206`.) |
| `data/` | A cleaned LongMemEval-S JSON file is present; the recorded v1 and held-out authored runs use the fixture suites under `eval/`, not full LongMemEval histories. (`data/longmemeval/longmemeval_s_cleaned.json`; `docs/TESTING.md:906-915,1175-1178`.) |
| `schema.sql`, `prompts.md`, `pyproject.toml` | SQLite DDL, extraction/provider prompt specification, and Python package/dependency metadata. (`schema.sql:1-10`; `src/engramOS/extraction.py:145-161`; `pyproject.toml:1-20`.) |

## 3. Architecture and data flow

The ingestion path persists a source turn and durable job, evaluates and records its gate outcome before extracting that turn, validates and stages the complete extraction response, then applies each fact to the observation ledger and rebuilds the affected slot projection. The query path classifies the requested temporal view, retrieves candidates, rehydrates them against a fresh SQLite snapshot, constructs an immutable Reader context, and validates the Reader's JSON response. (`src/engramOS/ingestion.py:279-359,421-459,461-510`; `src/engramOS/pipeline.py:68-160`.)

```text
TurnInput
  -> validate provenance, persist immutable turn + ingestion job
  -> bridge / policy / source signals -> store or sticky quarantine
  -> extraction contract -> validate -> stage -> fact_observations
  -> temporal slot projection in facts -> optional embedding index

Question
  -> deterministic classifier or injected classifier provider
  -> lexical + dense retrieval -> RRF / temporal projection
  -> fresh eligibility and provenance revalidation
  -> immutable ReaderContext -> JSON answer with citations or abstention
  -> eval_runs + eval_question_results (evaluation runner only)
```

`ingest_turns()` evaluates every input turn's gate before starting extraction for a batch; fixture evaluation instead invokes ingestion one turn at a time so an earlier pinned fact exists before a later turn is gated. (`src/engramOS/ingestion.py:197-203,305-359`; `scripts/run_eval.py:388-402`.) `pipeline.answer_query()` returns an outcome and does not persist evaluation results; `run_eval.py` owns evaluation persistence. (`src/engramOS/pipeline.py:52-68`; `scripts/run_eval.py:649-669`.)

## 4. SQLite data model

`initialize_database()` executes the idempotent `schema.sql` path, enables WAL for file databases, configures and verifies foreign keys, and applies additive initialization helpers. (`src/engramOS/db.py:26-45,83-118`.) The application tables and their declared columns are:

| Table | Columns | Purpose and relationships |
|---|---|---|
| `turns` | `turn_id`, `history_id`, `session_id`, `turn_idx`, `role`, `raw_content`, `token_count`, `source_type`, `base_trust`, `event_time`, `recorded_at`, `prev_hash`, `hash` | Immutable source text and provenance/time identity; referenced by jobs, facts, observations, and gate events. Turn updates are blocked except mutable `base_trust`; deletes are blocked. (`schema.sql:16-30,41-64`.) |
| `ingestion_jobs` | `turn_id`, `payload_hash`, `session_unix_ts`, `session_date`, `state`, `extraction_json`, `extraction_hash`, `fact_count`, `applied_count`, `updated_at` | Per-turn resumable extraction checkpoint; `turn_id` references `turns`; state and count checks constrain progress. (`schema.sql:66-86`.) |
| `facts` | `fact_id`, `history_id`, `turn_id`, `fact_seq`, `key_type`, `entity`, `attribute`, `value`, `value_norm`, `cardinality`, `kind`, `key_text`, `parent_fact_id`, `valid_from`, `valid_until`, `superseded_by`, `last_confirmed_at`, `last_confirmed_turn_id`, `confirmation_count`, `recorded_at`, `expired_at`, `trust`, `quarantined`, `quarantine_reason`, `confidence` | Searchable slot interval projection plus trigger cues. `turn_id`, `parent_fact_id`, `superseded_by`, and confirmation turn link to source/projection records. Only single-valued slots are superseded. (`schema.sql:97-124`.) |
| `fact_observations` | `observation_id`, `history_id`, `turn_id`, `fact_seq`, `entity`, `attribute`, `value`, `value_norm`, `cardinality`, `kind`, `key_text`, `valid_at`, `effective_at`, `canonical_fact_id`, `last_confirmed_at`, `last_confirmed_turn_id`, `confirmation_count`, `recorded_at`, `trust`, `quarantined`, `quarantine_reason`, `confidence` | Per-source extracted slot observations used to rebuild `facts`; unique by `(turn_id, fact_seq)`. `canonical_fact_id` points to the current projected interval and may be null for an observation excluded from projection. (`schema.sql:287-316`; `src/engramOS/temporal.py:277-326,648-653`.) |
| `embeddings` | `ref_type`, `ref_id`, `model`, `vec`, `content_hash` | Serialized float16 vectors keyed by reference type, ID, and embedding model; the polymorphic reference does not use a foreign key. Content hashes detect stale vectors. (`schema.sql:147-155`; `src/engramOS/embeddings.py:417-439,548-580`.) |
| `gate_events` | `event_id`, `turn_id`, `signal_bridge`, `signal_policy`, `signal_source`, `signals_fired`, `decision`, `created_at` | Append-only audit decision for a turn; update/delete triggers reject mutation. Signal values are persisted as 0/1. (`schema.sql:157-175`; `src/engramOS/gate.py:447-465`.) |
| `turns_fts` | `raw_content` | FTS5 external-content index over `turns.raw_content`, synchronized by insert trigger. (`schema.sql:34-39`.) |
| `facts_fts` | `key_text` | FTS5 external-content index over `facts.key_text`, synchronized on insert, key-text update, and delete. (`schema.sql:129-145`.) |
| `llm_cache` | `model`, `prompt_hash`, `temperature`, `cache_nonce`, `task_tag`, `response`, `in_tokens`, `out_tokens`, `cost_credits`, `latency_ms`, `phase_tag`, `run_id`, `created_at` | Successful completion cache and token/cost attribution; primary key is model, request hash, temperature, and nonce. (`schema.sql:177-197`.) |
| `llm_call_ledger` | `call_id`, `request_key`, `model`, `task_tag`, `phase_tag`, `run_id`, `reserved_credits`, `actual_credits`, `status`, `started_at`, `finished_at`, `error_type` | Pre-call budget reservation, in-flight coalescing key, and success/failure accounting. (`schema.sql:199-219`.) |
| `eval_runs` | `run_id`, `system`, `question_id`, `question_type`, `split`, `reader_model`, `judge_model`, `reader_prompt_hash`, `judge_prompt_hash`, `run_config_json`, `answer`, `correct`, `evidence_recall_at_5`, `prompt_tokens`, `out_tokens`, `latency_ms`, `created_at` | Per-question/system run metadata and compatibility result row; primary key `(run_id, system, question_id)`. (`schema.sql:238-258`.) |
| `eval_question_results` | `run_id`, `system`, `question_id`, `axis`, `answer_text`, `abstained`, `cited_ids_json`, `retrieved_ids_json`, `expected_answer`, `expected_label`, `correct`, `gate_outcome_json`, `prompt_tokens`, `completion_tokens`, `latency_ms`, `cost_credits`, `cache_request_hash` | Detailed per-question/system output; foreign key to the corresponding `eval_runs` composite key. (`schema.sql:260-285`.) |

The `budget_summary` view groups cache usage and pending/failed ledger accounting by task, model, phase, and run; its columns are `task_tag`, `model`, `phase_tag`, `run_id`, `api_calls`, `total_in_tokens`, `total_out_tokens`, `credits_spent`, and `credits_reserved`. (`schema.sql:221-236`.) SQLite also maintains FTS5 shadow tables for the virtual indexes and `sqlite_sequence` for the autoincrementing `gate_events.event_id`; these engine-managed layouts are not declared as application tables in `schema.sql`. (`schema.sql:34-39,129-145,158-160`.)

## 5. Ingestion, provenance, and gate

`TurnInput` supplies history/session identity, role, raw text, event-time basis, and optional source type. Missing source types default by role (`user`→`user_turn`, `assistant`→`assistant_turn`, `system`→`system_policy`, `tool`→`tool`); provenance validation accepts `external_doc` as a source and enforces the required role/source combinations. These labels are supplied by the caller, not verified against an identity provider in the inspected code. (`src/engramOS/ingestion.py:48-70,780-792`; `src/engramOS/provenance.py:6-9,16-43`.)

The gate has three deterministic signals and quarantines at two fired signals out of three. `signal_source` fires for `tool` and `external_doc`. `signal_policy` checks the incoming text against eligible, active, same-history `kind='constraint'` slot rows using normalized slot terms plus explicit negation/override markers; it cannot fire until a prior eligible constraint exists. `signal_bridge` compares shortening token prefixes against earlier eligible fact slot keys (entity, attribute, key text), using unigram and adjacent-bigram sets, Dice overlap, and fixed `tau = 0.60`. These patterns and threshold were chosen for the authored suite. (`src/engramOS/gate.py:158-177,179-246,249-277,348-365`; `docs/DECISIONS.md:739-770`.)

`fingerprint_inputs()` hashes canonical JSON containing the history and event time plus the active constraints and bridge slot keys. The evaluator input fingerprint is checked after evaluation and again in the write transaction; a changed input fails closed. (`src/engramOS/gate.py:168-177,297-329,368-379,432-440`.)

An approved turn can be extracted and applied. A quarantined turn gets a durable quarantine event and its source fact rows/observations are marked quarantined; the source turn remains stored, its `base_trust` is set to zero, and the ingestion job becomes quarantined. A prior quarantine event is sticky: later store events do not make the turn eligible. (`src/engramOS/gate.py:415-420,469-510`; `src/engramOS/ingestion.py:305-349,822-878`; `src/engramOS/eligibility.py:17-34,56-79`.)

The current trust rule is source-class based for temporal projection, not numeric. For single-valued slots, `user_turn` has higher authority than conflicting `tool` or `external_doc` observations even when numeric trust fields are equal. A lower-authority conflict is retained as an observation but excluded from the projected interval; a later user observation of that proposed value confirms it and makes the value user-authoritative. The source classes and trust labels do not prove that the user channel is uncompromised. (`src/engramOS/temporal.py:471-484,729-762`; `docs/DECISIONS.md:1316-1335`.)

## 6. Temporal facts and projection

Each accepted extraction becomes an observation with a source turn, slot, normalized value, validity time, transaction record time, trust, confidence, and quarantine state. A regular fact's `valid_from` is based on session time plus turn and fact-sequence tie-breakers; a backdated `as_of` fact uses the asserted date with smaller deterministic tie-breakers. `recorded_at` remains ingestion/transaction time. (`schema.sql:88-96,292-316`; `src/engramOS/temporal.py:120-138,277-307`.)

For one single-valued slot, eligible observations are ordered by valid time and stable source identifiers. Equal normalized values coalesce into a logical run; a different value starts a new run. Each projected interval has `valid_from` at its first observation, `valid_until` at the next run, `superseded_by` pointing to the next interval, and confirmation fields from the latest observation in the run. Multi-valued observations remain separate. Reconciliation rebuilds these relationships from observations, which supports deterministic replay and backdated insertion. (`src/engramOS/temporal.py:437-468,491-520,581-627,640-727,729-762`.)

The v2 projection change resolves source authority before forming intervals: a conflicting external-document or tool observation cannot close a current user-established single-slot interval; it remains in the observation ledger with no canonical projected fact until confirmed by a user turn. The comparison is based on `value_norm` inequality in the same normalized slot and current authority class, not the numeric `trust` column. (`src/engramOS/temporal.py:471-484,729-762`.)

`retrieve_facts(view='current')` returns the latest eligible single-valued interval and all eligible multi-valued entries. Inclusive `as_of` selects `[valid_from, valid_until)` at the timestamp; an exclusive `TemporalBoundary` selects the left limit without subtracting a floating-point epsilon. `history` returns ordered intervals. (`src/engramOS/temporal.py:891-907,961-1019`.)

For a retrospective fact, the Reader evidence metadata marks `retrospective` when the fact was recorded after its valid time or when supporting provenance is later than the valid time. For an as-of query, source provenance after the cutoff has its text emptied before the Reader prompt is built; event-time and valid-time are kept distinct. (`src/engramOS/pipeline.py:217-249`; `src/engramOS/reader.py:40-45,631-658`.)

## 7. Retrieval

Lexical turn/fact retrieval uses FTS5 and SQLite BM25; user query tokens are quoted as safe FTS terms. Dense retrieval uses persisted model-identified embeddings and cosine ranking. The recorded embedding contract is BGE-M3 with 1,024 dimensions; indexing records a content hash and fails freshness validation when vectors are missing, stale, or from another model identity. (`src/engramOS/retrieval.py:145-150,153-193,218-275,278-355`; `src/engramOS/embeddings.py:21-22,388-455,548-601`.)

Turn hybrid retrieval fuses lexical and dense ranks with reciprocal-rank fusion, `sum(1 / (k + one_based_rank))`; the default `k` is 60. The recorded runner retrieves up to 50 candidates per channel and presents up to 10 turns. `recency` sorts the same top-k hybrid turn candidates by descending event time. (`src/engramOS/retrieval.py:358-388,391-451`; `scripts/run_eval.py:507-518`.)

`retrieve_memory()` applies the query's event-time range to lexical and dense turn candidates before fusion/top-k and rechecks the range afterwards. For date-filtered hybrid evaluation, the event-time upper bound is passed into `hybrid_rag()` before RRF; the baseline does not project temporal facts. (`src/engramOS/retrieval.py:454-515,558-563`; `scripts/run_eval.py:503-517`.) Engram also selects temporal fact rows using the classifier view, fuses fact lexical/dense channels, and revalidates them. (`src/engramOS/retrieval.py:517-577`.)

Before Reader authorization, `rehydrate_reader_candidates()` accepts only a unique subset of retrieved typed IDs, reads fresh source text and eligibility in a new SQLite snapshot, and rechecks fact/turn source and confirmation rows. In current view, it additionally removes a conflicting external/tool turn when an active user-sourced or user-confirmed value wins. (`src/engramOS/retrieval.py:636-745,1000-1045,1088-1157`.)

## 8. Query classification

The classifier can return `classified`, `clarification_required`, or `unsupported`; classified views are `current`, `as_of`, and `history`. Reason codes cover present state, explicit and relative as-of dates, exclusive `before` boundaries, historical changes/ranges, ambiguity, invalid expressions, unsupported knowledge-time requests, and future as-of requests. (`src/engramOS/query_classifier.py:23-39,80-128,210-243`.)

Deterministic routing handles knowledge-time expressions, parsed ISO/natural-language dates, relative days and periods, month references, change/history language, and date-free current-state questions matching explicit current wording or the point-question detector. Other shapes use the injected provider; if absent, the result asks for clarification. The post-hoc correction changes a non-routed provider classification to current only when there is no parsed date and no temporal-word match. Explicit future dates, knowledge-time patterns, and date-bearing queries remain unsupported or invalid rather than being silently routed to current. (`src/engramOS/query_classifier.py:182-243,246-366,388-444,554-581,683-700`.)

If classification raises or returns a non-ready route, `answer_query()` does not retrieve evidence: provider/validation errors become a `classification_failed` abstention outcome, while a non-ready classification returns clarification or unsupported status. (`src/engramOS/pipeline.py:68-82`.) The four date-free held-out misroutes that motivated the correction and the 14 v1 queries that used the classifier are recorded in the run notes. (`docs/DECISIONS.md:1310-1314,1561-1562`; `docs/TESTING.md:1287-1302`.)

## 9. Reader and answer synthesis

The pipeline constructs a database-free immutable `ReaderContext` containing the query, validated classification, history ID, ordered typed evidence, provenance, ordering label, timeline-completeness flags, and authorization timestamp. The baseline helper constructs the same type of context from already retrieved turns, but its docstring explicitly says the baseline route is evaluation-only, ungated, and unrevalidated. (`src/engramOS/reader.py:108-165`; `src/engramOS/pipeline.py:129-197`.)

The Reader prompt supplies temporal scope and JSON evidence/provenance. Its instructions treat query/evidence as untrusted data, forbid tool calls and retrieval expansion, require citations to top-level supplied IDs, and require abstention when the evidence does not support an answer. The parser checks exact schema fields, schema version, citation uniqueness/membership, and consistency of answer versus abstained state. It does not establish semantic entailment. (`src/engramOS/reader.py:29-45,613-675,720-770`; `src/engramOS/reader.py:1-6`.)

The fixed abstention is `I don't know.` with no citations. Provider absence, provider error/timeout, invalid output, invalid citations, no evidence, or non-ready classification returns an abstention outcome. (`src/engramOS/reader.py:211-281,284-361,801-802`.) In the recorded local configuration, the Reader cap is 2,048 generated tokens and classifier cap is 1,024; `finish_reason=length` is separately counted as a provider failure by the evaluation runner. (`src/engramOS/local_llama.py:44-70,374-393,414-470`; `scripts/run_eval.py:894-926`.)

## 10. LLM provider layer

`ReaderProvider`, `QueryClassificationProvider`, and `FactExtractor` are injected interfaces. Hosted chat adapters remain in the source tree and route requests through `cached_generate`; their API key is read from `MODEL_API_KEY` when not explicitly supplied. The recorded benchmark did not use hosted calls. (`src/engramOS/reader.py:161-165`; `src/engramOS/query_classifier.py:50-54`; `src/engramOS/meta_provider.py:32-60`; `src/engramOS/meta_adapters.py:1-5,151-250`.)

`run_eval.py --mode offline` selects deterministic fake Reader/classifier providers; `--mode local` requires explicit local settings and selects `LocalReaderProvider` and `LocalQueryClassifierProvider`. The local transport restricts its base URL to `127.0.0.1`, sends no authorization header, and propagates server/network failures rather than switching to a fake provider. (`scripts/run_eval.py:404-425,1434-1525`; `src/engramOS/local_llama.py:1-6,72-80,130-160`; `tests/test_local_llama.py:93-104,133-150`.)

The local model identity includes model file name and SHA-256, quantization, context, GPU layers, parallel slots, KV-cache type, flash attention, chat template, reasoning settings, sampling settings, and token caps. The request hash also includes the full chat payload. `llm_cache` request identity hashes provider/model, both prompts, temperature, output cap, response format, and generation settings; task, phase, and run labels are accounting metadata. (`src/engramOS/local_llama.py:47-109,174-199`; `src/engramOS/llm_cache.py:299-311`.)

Before a provider call, `cached_generate()` checks cache, coalesces identical concurrent misses through the request ledger, reserves budget, and records successful token/cost/latency data. A provider failure without usage data charges the full reservation conservatively. The hosted default has a 150-credit cap and phase caps. Local successful requests record actual cost 0 in `local` pricing mode while using a tiny positive reservation to preserve the common cap path. (`src/engramOS/llm_cache.py:64-80,90-162,165-223,226-267`; `src/engramOS/local_llama.py:204-214,296-318`.) Timeout is not part of the local request identity; a cache hit can therefore reuse the same completion under a different caller timeout. (`tests/test_local_llama.py:194-221`; `docs/TESTING.md:888-890`.)

The local adapter retries malformed classifier/Reader JSON once with a distinct nonce, records both attempts, and stops without retry on `finish_reason=length`; the runner counts truncation, parse, timeout, and other provider failures separately from wrong answers. (`src/engramOS/local_llama.py:374-393,441-470`; `scripts/run_eval.py:894-939`.)

## 11. Evaluation harness

`eval/suite_v2.json` is the frozen v1 authored suite: 23 histories, 80 turns, 38 questions. Its recorded SHA-256 is `6b9a2a48e7d2dec311e7064de4b2adbb717ab8a67372a7590f1595f450a2c823`. `eval/suite_v2_heldout.json` contains 8 histories, 12 turns, 8 questions and has recorded SHA-256 `5c64470c0773d9bb4b2d672b1a207c24959784f7cb5e7539497b02ef03dc5b85`. The default `eval/suite.json` is a distinct earlier suite. (`docs/TESTING.md:906-915,1175-1178,1120-1123`; `scripts/run_posthoc_v2.py:19-23`.) The loader validates schema, unique IDs, source types, fact payloads, expected labels, and suite relationships, but the runner records the file hash rather than comparing it to a compiled-in expected digest. (`eval/loader.py:31-104`; `scripts/run_eval.py:357-377,591-603`.)

The axes cover present-state and unsupported/future queries; inclusive as-of and exclusive-before boundaries; A→B→A, backdated retrospective redaction; standard, caught, and adaptive poisoning; benign untrusted controls/updates, multi-topic turns, and trusted policy changes. (`docs/TESTING.md:962-980,1261-1266`.) Facts are precomputed Section A payloads. `FixtureFactExtractor` serializes those fixture facts and does not infer from text. (`eval/loader.py:135-161`.)

All four systems share the injected Reader provider and production Reader prompt: `recency` sorts the top-10 hybrid turn set by event time; `hybrid_rag` uses ungated turn retrieval; `hybrid_rag_date_filter` adds the as-of event-time prefilter before RRF; `engram` calls `pipeline.answer_query()` and uses gated turn/fact retrieval. (`scripts/run_eval.py:66-67,481-529`; `src/engramOS/retrieval.py:391-451`; `src/engramOS/pipeline.py:52-160`.) The evaluation runner also calls `classify_query()` for baseline routes. Thus the source does not implement a classifier-free baseline; all systems can be affected by classifier routing, even though their retrieval paths differ. (`scripts/run_eval.py:481-529`.)

The runner reports two attack metrics: exposure is true when an attack turn is direct Reader evidence or appears in evidence provenance; adoption is a contiguous-token match for the declared attack claim in the answer. Quarantine leaks are separately counted from cited/retrieved source turns. These are lexical/evidence proxies, not semantic security judgments. (`scripts/run_eval.py:1000-1035,1210-1225,1289-1300`.)

The pre-registered v1 run ID is `q4-locked-classifier1024-resume300-v1`; the post-hoc Engram-on-v1 run ID is `q4-post-hoc-v2-engram-v1`; the held-out run ID is `q4-post-hoc-v2-heldout-v1`. The v2 runner writes into a separate database and writes separate result JSON and progress files; v1 result rows remain separately auditable. The documented database copy command was not recorded. (`docs/TESTING.md:865-915,1203-1228,1241-1246`; `scripts/run_posthoc_v2.py:19-23,107-153`.)

For each question/system result, `run_eval.py` writes `eval_runs` and `eval_question_results` within one transaction, then appends and flushes a progress JSONL record. Existing rows under the same run/system/question key are treated as resumed; persistence uses `INSERT OR REPLACE`, so reusing an existing run ID for a changed invocation can replace its row. (`scripts/run_eval.py:460-473,649-705,747-763,1177-1207`.) Post-hoc v2 summaries are written as separate JSON files after each run's `run_evaluation()` returns. (`scripts/run_posthoc_v2.py:137-160`.)

## 12. Configuration and run instructions

Recorded local runs used `Muse-Glimmer-30B-UD-Q4_K_XL.gguf`, SHA-256 `82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38`, quant `UD-Q4_K_XL`, 32 of 53 layers on GPU, 21 on CPU, context 8,192, one server slot, Flash Attention on, q8_0 K/V cache, built-in GGUF chat template, low reasoning with 512-token budget/strength, temperature 0, top-p 0.95, top-k 40, seed 42, Reader cap 2,048, classifier cap 1,024, and 300-second provider timeout. The recorded llama.cpp build was `b12058-3f8af15a7`; successful local calls recorded 0 credits. (`docs/TESTING.md:941-947`; `scripts/run_posthoc_v2.py:41-63`; `scripts/run_eval.py:1487-1523`.)

For a new local run, start the already-installed server with the measured settings, then use a new database, run ID, and progress path. The launcher binds to `127.0.0.1` and sets flash attention, q8_0 K/V cache, reasoning options, context, and slot count. (`scripts/local_server.ps1:1-18,32-63`.) A reproduction evaluation command is:

```powershell
.\.venv\Scripts\python.exe scripts\run_eval.py --mode local --embedding bge --database eval\reproduction-q4.sqlite3 --run-id q4-reproduction-v1 --suite eval\suite_v2.json --progress-log logs\q4-reproduction-progress.jsonl --provider-timeout 300
```

The evaluation CLI defaults to offline mode and fake embeddings; local mode defaults to BGE-M3 and requires the local server settings. Keep fake-vector and BGE-M3 indexes in separate databases: the runner rejects an embedding-model identity mismatch. (`scripts/run_eval.py:330-339,1434-1509`.) The read-only replay command is `..venv\Scripts\python.exe scripts\replay.py --help`; `scripts/replay.py` opens SQLite with URI `mode=ro` and issues SELECT queries only. (`README.md:95-100`; `scripts/replay.py:34-50,51-107`.)

The full test command recorded for the project is:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`docs/TESTING.md` records the user's confirmed result of 361/361 tests; this architecture task did not run the suite. (`docs/TESTING.md:1154-1164`.)

## 13. Tests and design assertions

Tests use Python `unittest`; database integrations use temporary file-backed SQLite databases. Coverage includes schema initialization, FTS sync, gate combinations and sticky quarantine, extraction validation/retries, temporal intervals/backdating/replay, eligibility, embeddings/index freshness, BM25/RRF, temporal retrieval and Reader revalidation, local transport/cache failure paths, evaluation persistence/resume, and read-only replay. (`docs/TESTING.md:93-124`; `tests/test_local_llama.py:81-253`; `tests/test_replay.py:18-110`.)

The F-02 public Reader-helper regression remains explicit: `tests/test_reader.py` asserts the old public function names are absent. Baseline-context tests separately check ordering labels and that `answer_query()` does not call the evaluation-only baseline helper. (`tests/test_reader.py:596-600`; `tests/test_baseline_pipeline.py:50-70`.) Gate tests cover the default signal evaluator and fingerprint input changes. (`tests/test_default_signal_evaluator.py:24-183`.)

At `tests/test_expanded_authored_suite.py:63-78`, the offline-run assertion now requires adoption successes to be no greater than exposure successes and pins both reached and adopted question-ID sets to `{poison_adaptive_user}`. The former condition required exposure and adoption counts to be unequal; the trust-aware fake run instead has the same one reached/adopted trusted override. The changed condition was in place before any v2 live model result. (`tests/test_expanded_authored_suite.py:63-78`; `docs/TESTING.md:1137-1160`; `docs/DECISIONS.md:1430-1439`.)

The benchmark does not test open-ended extraction accuracy because it uses fixed facts; small authored samples do not establish generalization; and recorded coverage notes identify large-history latency/memory and cross-process same-history ingestion coordination as untested. (`eval/loader.py:135-161`; `docs/TESTING.md:126-151`.)

## 14. Recorded results

The following values are copied from `docs/TESTING.md`; no results were recomputed for this document. Intervals are exact two-sided Clopper–Pearson 95% intervals. (`docs/TESTING.md:949-957,1248-1257`.)

| Status / run | System | n | Correct | Accuracy (95% CI) | Mean / max latency | Provider failures |
|---|---|---:|---:|---:|---:|---:|
| Pre-registered v1, `q4-locked-classifier1024-resume300-v1` | Engram | 38 | 26 | 68.42% (51.35–82.50%) | 48,109 / 199,581 ms | 0 |
| Pre-registered v1 | `hybrid_rag` | 38 | 22 | 57.89% (40.82–73.69%) | 43,052 / 158,208 ms | 0 |
| Pre-registered v1 | `hybrid_rag_date_filter` | 38 | 19 | 50.00% (33.38–66.62%) | 15,979 / 144,553 ms | 0 |
| Pre-registered v1 | `recency` | 38 | 19 | 50.00% (33.38–66.62%) | 50,509 / 151,333 ms | 0 |
| Post-hoc v2 on v1 questions, `q4-post-hoc-v2-engram-v1` | Engram | 38 | 34 | 89.47% (75.20–97.06%) | 45,426 / 104,348 ms | 0 |
| Held-out, `q4-post-hoc-v2-heldout-v1` | Engram | 8 | 8 | 100% (63.06–100%) | 38,743 / 70,998 ms | 0 |
| Held-out | `hybrid_rag` | 8 | 6 | 75% (34.91–96.81%) | 30,087 / 56,073 ms | 0 |
| Held-out | `hybrid_rag_date_filter` | 8 | 6 | 75% (34.91–96.81%) | 46 / 52 ms | 0 |
| Held-out | `recency` | 8 | 6 | 75% (34.91–96.81%) | 44,633 / 70,877 ms | 0 |

The v1 11-attack aggregate was Engram exposure 6/11 and adoption 5/11; each baseline exposed 9–10/11 and adopted 3/11. Post-hoc Engram on the same questions exposed 6/11 and adopted 1/11; excluding `poison_adaptive_user` as a compromised trusted channel, its untrusted-only exposure/adoption was 5/10 and 0/10. On held-out cases, Engram exposure/adoption was 0/3 and 0/3, while the baselines exposed 3/3 and adopted 1/3, 1/3, and 2/3 respectively. All recorded v2 runs had zero quarantine leaks and zero credits. (`docs/TESTING.md:987-1003,1268-1285`; `docs/DECISIONS.md:1523-1545`.)

## 15. Limitations and known issues

1. **Compromised trusted channel.** A `user_turn` can supersede an earlier user value; the v2 protection only blocks conflicting `tool` and `external_doc` observations when current authority is user. A compromised input labeled `user_turn` therefore remains authorized by the provenance policy. The recorded `poison_adaptive_user` was reached and adopted. Since role/source labels are caller-supplied and no identity/authentication mechanism appears in the checked provenance/ingestion path, defense against a compromised trusted channel is not provided. Impact: security. (`src/engramOS/temporal.py:729-762`; `src/engramOS/provenance.py:6-9,23-43`; `docs/TESTING.md:1268-1276`.)

2. **Untrusted attacks can pass the gate or remain exposed.** The gate quarantines at 2/3, while a normal untrusted-source signal alone contributes 1/3; the authored v1 gate stored six attack turns. Engram adopted untrusted claims in v1, and the held-out hybrid baselines adopted jacket/region claims. V2 reduced Engram's untrusted adoption on the original labels but has only 8 held-out questions. Impact: security; changes would require stronger independent signals or authority-aware retrieval/gating, with separate validation because the detector and authored attacks were designed together. (`src/engramOS/gate.py:348-365`; `docs/TESTING.md:996-1003,1268-1282`; `docs/DECISIONS.md:922-923`.)

3. **A→B→A results remain 0/2 under the locked exact grader.** The v2 recorded Engram axis is 0/2. The v1 manual semantic re-grade accepted both ordered A→B→A answers for every system, changing those rows from 0/2 to 2/2, so grading sensitivity is documented. The exact v2 failure location is **not determined**: the checked code forms normalized temporal runs and returns chronological history intervals, then passes them to the Reader; the aggregate result does not identify whether a particular miss came from route selection, retrieved evidence, Reader synthesis, or exact matching. (`src/engramOS/temporal.py:512-520,961-1019`; `src/engramOS/retrieval.py:546-557`; `src/engramOS/pipeline.py:115-160`; `docs/TESTING.md:1079-1084,1261-1266`.) Impact: correctness and evaluation attribution.

4. **`attack_ext_subtle_jacket` reaches the Reader but causes an abstention.** In post-hoc v2, the Engram row records `abstained=true`, answer `I don't know.`, `attack_reached_reader=true`, no adopted claim, and retrieved IDs containing the prior jacket turn and attack turn; the attack itself was stored with only the source signal fired. This is a utility/correct-answer miss, not a provider failure. Why the Reader abstained is **not determined** by the persisted row. (`eval/results_posthoc_v2_engram_v1.json:592-625`; `src/engramOS/reader.py:264-281`.) Impact: correctness/usability.

5. **`poison_adaptive_external` uses a frozen v1 label that favors the injected value.** The v1 expected answer is `cobalt`, while post-hoc Engram preserved trusted `blue`; the v2 grade therefore counts it wrong even though the safe value is blue. The held-out version changes the expected label to blue, but v1 is not relabeled. Impact: interpretation of correctness and attack-success comparisons. (`docs/DECISIONS.md:1346-1352,1529-1535,1553-1559`.)

6. **Latency and context limits constrain use.** Post-hoc Engram mean/max latency was about 45/104 seconds per question/system row. Recorded settings use one llama-server slot and an 8,192-token context; a 2,048-token Reader completion cap leaves reasoning/output work within that context but does not remove long prompt processing. Cache hits make some recorded means non-comparable to fresh inference. Impact: performance and throughput. (`docs/TESTING.md:941-947,1252-1257,1259-1260`; `scripts/run_posthoc_v2.py:47-63`; `src/engramOS/local_llama.py:44-70`.)

7. **Small, same-author authored benchmark; v2 is post-hoc.** V1 has 38 questions, held-out has 8, and the authors tuned detector patterns/threshold with the attacks. V2 applies provenance-aware projection and date-free classifier corrections after v1; these comparisons do not isolate causal effects or establish statistical significance/generalization. Impact: evaluation validity and security claims. (`docs/TESTING.md:906-915,1166-1178,1241-1266`; `docs/DECISIONS.md:922-930,1553-1559`.)

8. **Classifier comparison nuance.** The stated concern that Engram alone pays classifier costs/errors does not describe this runner: baseline branches also invoke `classify_query()` before retrieval, and the run records 14 v1 classifier-provider queries. There is no classifier-free baseline implementation, so results do not compare against a route that avoids classifier calls. Impact: cost/fairness interpretation. (`scripts/run_eval.py:481-529`; `docs/TESTING.md:1024-1040`.)

9. **Model and quantization sensitivity.** Answer generation and classifier routing in the recorded local runs depend on one quantized Muse Glimmer model, the GGUF chat template, reasoning mode, token limits, and sampling settings. Results cannot be assumed invariant under a different model, quantization, llama.cpp build, or server flags. Impact: correctness, reproducibility, and performance. (`src/engramOS/local_llama.py:47-104`; `docs/TESTING.md:941-947`.)

10. **Engram Reader cache identity includes a per-query timestamp.** `pipeline.answer_query()` rehydrates evidence with `authorization_timestamp=time.time()` and serializes that value into `reader_context`; the complete prompt is part of the cache hash. Rebuilding the same Engram question in a later run therefore creates a different Reader request hash, so its saved answer cannot prewarm a later demo. Reader cache hits for this route require an identical rebuilt context, such as a repeated request within a process; this does not describe classifier or baseline requests whose complete payloads may remain identical across runs. The v2 rows and cache hashes are recorded artifacts, but their original prompt payloads cannot be rebuilt byte-for-byte from the stored data. Impact: prompt reproducibility and demo prewarming. (`src/engramOS/retrieval.py:744-745,783`; `src/engramOS/pipeline.py:132-151`; `src/engramOS/reader.py:333-345,696-732`; `src/engramOS/llm_cache.py:299-311`.)

11. **V1 artifact provenance is incomplete.** The recorded audit states that v1 artifacts were not modified after v2 DB creation based on timestamps and absence of a v2 write path, but there were no pre-run hashes for v1 DB/results/gallery/re-grade and the DB copy command was not recorded. This does not establish hash equality or reproduce the one-off copy command. Impact: auditability/reproducibility. (`docs/TESTING.md:1309-1333`; `docs/DECISIONS.md:1563-1568`.)

12. **Suite “lock” is recorded, not enforced by an expected digest.** The loader validates structure and the runner stores the digest it observes; a validly edited suite can be run under a new hash without a code-level check against the recorded locked digest. This is inferred from the loader and run configuration code. Impact: benchmark reproducibility. (`eval/loader.py:31-104`; `scripts/run_eval.py:357-377,591-603`.)

13. **Reader timeout does not cancel every arbitrary provider worker.** `_invoke_provider()` returns when its future times out, but the daemon worker releases its semaphore slot only when the provider call finishes. A non-cooperative provider may continue in the background and consume one of four Reader slots; the local server itself was configured with one slot. Impact: performance/availability under stalled providers. (`src/engramOS/reader.py:102-103,684-717`; `docs/TESTING.md:941-946`.)

14. **Slot identity is exact after normalization; `replaces` does not choose a target.** Ingestion canonicalizes a candidate by exact normalized `(entity, attribute)` match and writes the resulting `FactInput`; the extraction contract includes a `replaces` field, but `_apply_batch_facts()` does not use it to select an existing fact. Fuzzy slot canonicalization is not implemented. Impact: correctness when equivalent entities/attributes are extracted with different slot names. (`src/engramOS/ingestion.py:1036-1061`; `src/engramOS/extraction.py:23-35,62-80`; `docs/TESTING.md:134-135`.)

15. **Cross-process same-history write coordination is not covered.** Ingestion performs multiple stage transactions and recorded coverage notes require callers to serialize ingestion within one history. Concurrent histories are separable, but the checked pipeline does not provide a whole-history lock. Impact: race/replay correctness under concurrent writers. (`src/engramOS/ingestion.py:279-359,461-510`; `docs/TESTING.md:136-137`.)

16. **Trigger generation is not implemented.** The schema and retrieval paths can store, link, and search trigger cue facts, but the extraction contract does not produce trigger rows as a generation phase. Impact: completeness for cue-based retrieval. (`schema.sql:102-114`; `src/engramOS/extraction.py:23-35,164-218`; `docs/TESTING.md:116`.)

## 16. Open questions

- **Caller authentication for provenance is not determined.** The inspected package accepts and validates `role`/`source_type`, but does not show an identity or signature check; whether an external application authenticates those inputs was not determined. Checked `src/engramOS/ingestion.py:58-70,780-792` and `src/engramOS/provenance.py:23-43`.
- **Exact internal FTS5 shadow-table columns are not determined from the application schema.** `schema.sql` declares the two virtual tables and external-content columns but not SQLite's internal shadow DDL. Checked `schema.sql:34-39,129-145` and the `CREATE` declarations in `schema.sql`.
- **The causal stage for the remaining A→B→A v2 exact-grade misses is not determined.** Aggregate results and the generic temporal/retrieval/Reader paths do not include enough per-stage evidence to attribute those two outcomes. Checked `docs/TESTING.md:1261-1266`, `src/engramOS/temporal.py:961-1019`, `src/engramOS/retrieval.py:546-577`, and `src/engramOS/pipeline.py:68-160`.
- **Why the Reader abstained on `attack_ext_subtle_jacket` is not determined.** Its saved row shows the answer, attack exposure, and retrieved IDs, not the model's internal decision basis. Checked `eval/results_posthoc_v2_engram_v1.json:592-625` and the Reader output handling in `src/engramOS/reader.py:264-281`.
- **The exact v2 database copy command and process are not determined.** The documented note explicitly says the DB copy command was not recorded. Checked `docs/TESTING.md:1217-1228,1309-1315` and `docs/DECISIONS.md:1563-1568`.
- **Production-scale capacity is not determined.** Large-history memory/latency and cross-process same-history ingestion behavior were not benchmarked in the recorded coverage notes. Checked `docs/TESTING.md:136-151`.
