# Testing

## Run the suite

From the project root, run:

```powershell
python -m unittest discover -s tests -v
```

Historical Phase 4A hardening completed with 120 tests. For Batch 2A, the
baseline before edits was **132 passing tests**. The final full command passed
**147 tests** (15 added), and the focused temporal/retrieval command passed
**57 tests**. Both runs used Python's built-in `unittest`; no paid model/API
calls were made and API credits consumed remain **0**.

For the initial Batch 2B implementation, the pre-edit baseline was **147
passing, 0 failing**; 11 tests were added for gate propagation rollback/success,
opt-in quarantine isolation, and deterministic timestamp propagation. That
verification passed 54 gate/ingestion tests, 57 temporal/retrieval tests, and
158 tests in the full suite.

For Batch 2B completion, the pre-edit baseline was **158 passing, 0 failing**.
Eight test methods were added for the complete role/source matrix, shared
entry-point validation, single-turn and batch rejection, derived defaults and
replay identity, unchanged gate behavior, and omitted-`recorded_at` timestamp
coverage. Final verification passed **62 gate/ingestion tests**, **57
temporal/retrieval tests**, and **166 tests** in the full suite, with 0
failures. Existing `(user, external_doc)` expectations remain unchanged.

Batch 3 started from a separate historical baseline of **166 passing, 0
failing**. It added 16 test methods across gate, ingestion, temporal, and
retrieval coverage. Seven existing gate test methods received P1 fixture-only
updates to add genuine store events for turns intended to be eligible. Three
gate-event count assertions now compare against a captured pre-operation
baseline, preserving their original event-count and append-only intent. No
behavioral assertion was weakened. The final measured count is **182 passing,
0 failing**. Earlier batch totals above are historical and have not been
rewritten to include Batch 3 tests.

The Batch 2B verification commands were:

```powershell
python -m unittest tests.test_gate tests.test_ingestion -v
python -m unittest tests.test_temporal tests.test_retrieval -v
python -m unittest discover -s tests -v
```

The modified ingestion test module was also run directly:

```powershell
python -m unittest tests.test_ingestion -v
```

It passed all 43 tests, including the 8 new methods.

Batch 3 verification used these commands:

```powershell
python -m unittest tests.test_gate tests.test_ingestion -v
python -m unittest tests.test_temporal tests.test_retrieval -v
python -m unittest discover -s tests -v
```

The final measured counts were **70 gate/ingestion tests**, **65
temporal/retrieval tests**, and **182 tests** in the full suite; all three
commands exited successfully.

## Coverage

All tests use Python's built-in `unittest`. Database, temporal, gate,
ingestion, embedding, vector-index, and retrieval integration tests use
temporary file-backed SQLite databases with the project's real schema.

| Area | Tested behavior |
|---|---|
| Database foundation | Schema initialization/idempotency, foreign keys on separate connections, rollback, turn-field immutability with mutable `base_trust`, and FTS synchronization. |
| Temporal writes and reads | New facts, confirmation/provenance, supersession, backdating, A→B→A, equal-value normalization, matching-successor continuation, deterministic same-time ordering, replay, current/as-of/history views, history isolation, and multi-valued observations projected once in each view with exact inclusive `as_of` boundaries and deterministic ties. |
| Gate and quarantine | All eight 2-of-3 combinations, incomplete evaluator handling, canonical gate API, append-only gate events, trust changes, exact rollback snapshots after fact/observation/trust failures and after slot rebuilding, source/history checks, sticky quarantine, temporal projection rebuilding, provenance matrix and persistence recheck before gate writes, and real FTS candidate filtering. |
| Extraction contract | Original Section A instructions, exact JSON fields, empty arrays, sequence from response order, normalized identifiers, cardinality/kind/date/confidence validation, wrong session/turn provenance, duplicate JSON keys, malformed JSON, bounded transient/validation retries, and rejection/retry of `as_of` later than the source session date. |
| Turn ingestion | User/assistant/tool/external-document fixtures; all 20 role/source pairs against the asymmetric 11-accept/9-reject table; role-derived default source types and replay identity; pre-write single/batch rejection with no persistence or gate/extractor calls; unchanged untrusted-source gate behavior; exact source-content persistence, event/hash-chain values, ordering, duplicate replay, and turns-only persistence/replay/upgrade. |
| Gate integration and recovery | Store-event reuse, missing/failed gate stop, default quarantine stop before extraction, opt-in batch quarantine isolation and replay, isolation of quarantine detected before apply, slot exclusion from later prompts, propagation of unrelated exceptions, later-store non-revocation, extraction failure recovery, and no duplicate gate event on sequential retry. |
| Temporal ingestion and recovery | Empty extraction, multi-fact same-slot supersession, backdated facts, whole-batch validation before writes, staging before temporal writes, partial fact-batch recovery after a SQLite trigger failure, supplied and omitted `recorded_at` propagation through observations, projected facts and `expired_at`, gate events, final job writes, and replay stability. |
| Fail-closed eligibility and recovery | Six hard-coded store/quarantine history cases; pending and quarantined exclusion across turn/fact FTS and dense search, hybrid, recency, temporal views, memory evidence, and freshness; gated versus `gated=False` turn-only behavior; missing-gate checkpoint recovery and isolation; D4 no-write guard; pending-observation reconciliation recovery; sticky quarantine and history scope. |
| Provenance at the gate | All 20 role/source pairs plus unknown values; parity with ingestion; evaluator call count zero for invalid provenance; persistence-time recheck; isolate-policy propagation; no event, trust, quarantine, or projection side effects on failure. |
| Embeddings and migration | 1024-dimensional validation, finite/nonzero checks, float16 BLOB round-trip, additive `content_hash` migration, idempotent indexing, stale content refresh, read-only freshness checks for new turns/facts, changed text, NULL hashes, model mismatch, quarantine exclusion, history isolation, gated/ungated turn-source selection with facts always gated, and lazy no-download BGE adapter behavior. |
| BGE-M3 identity | Hugging Face `refs/<revision>` resolution without snapshot sorting, ambiguous/missing revision errors, stable distinct revision IDs, path-independent local checkpoint fingerprints, content-change detection, query/document identity consistency, and legacy generic-vector incompatibility. |
| Dense index | Rebuild from persisted vectors, fail-closed stale index loading, history isolation, orphan ignoring, cosine order, stable ties, pre-scoring `allowed_ids` filtering, CPU execution, and CUDA execution only if available. |
| FTS5 and BM25 | Exact and multi-token matching, lower-score-first BM25 ordering, punctuation/empty/missing queries, operator-injection tokenization, and history isolation. |
| RRF and hybrid baselines | One-based reciprocal-rank formula, shared and single-channel candidates, duplicate elimination, deterministic ties, dense-plus-lexical top-10, explicit failure when dense turn embeddings are unavailable, gated defaults, explicitly ungated turn-only paths, and recency sorting over the same top-k IDs. |
| Temporal retrieval and evidence | Current/as-of/date-only/history filters, closed interval metadata, complete chronological temporal history, backdated source turns whose event time follows validity time, deduplicated source/confirmation evidence, and single-snapshot projection/evidence retrieval. |
| Trigger retrieval | Positive persisted-trigger retrieval through FTS and dense search; parent fact linkage; parent source and confirmation evidence; current/as-of filtering; exclusion after supersession/quarantine. Trigger generation remains unimplemented. |
| Retrieval quarantine and concurrency | Quarantined turn FTS/dense exclusion, quarantined fact FTS/dense exclusion, parent-fact trigger exclusion, source/confirmation evidence checks, quarantine after indexing, sticky later-store behavior, cross-history isolation, and explicit WAL writer coordination during both composite read APIs. |

The multi-turn batch tests verify that a malformed later response prevents any
fact writes from the batch, and that accepted candidates are applied in
chronological turn/fact-sequence order through `temporal.insert_fact()`. The
quarantine-isolation tests retain whole-batch structural validation and
whole-extraction validation before temporal fact writes.

## Not covered or not implemented

- Live extraction provider integration. `FactExtractor` is injectable only;
  no model API client is connected, and no cache/budget manager exists.
- Bridge-similarity and policy-contradiction detector algorithms and
  thresholds. Gate tests use deterministic local evaluators.
- Fuzzy slot canonicalization through the specified embedding/reranker
  components. Current ingestion normalizes and reuses exact slot names.
- Cross-process same-history ingestion coordination. Callers must serialize
  ingestion within a history; independent histories can be handled separately.
- Real BGE-M3 model loading, model-generated vectors, and retrieval quality
  have not been run. Sentence Transformers and local BGE-M3 weights are absent;
  the adapter is local-only and pipeline tests use deterministic fake
  1024-dimensional embeddings.
- CUDA execution could not be tested: `nvidia-smi` detects an NVIDIA GeForce
  RTX 5070 Ti Laptop GPU, but installed Torch `2.14.1+cpu` reports
  `torch.cuda.is_available() == False`. CPU vector search is tested.
- Query classifier, reranker, Reader-context construction, answer generation,
  and end-to-end model answers are not implemented.
- Large-history memory/latency benchmarking and concurrent indexing behavior
  have not been tested. Concurrent retrieval against WAL writes is covered
  with explicit event synchronization and is consistent with the snapshot
  visible when each read transaction begins.
- Cache identity/concurrency, budget reservations, API failure cost
  accounting, evaluation, and replay.
- Property-based testing with Hypothesis; it is not installed. Deterministic
  integration tests cover the specified ingestion interruption points.

The **182 passing tests** establish local SQLite and deterministic fake-vector
behavior only; they do not establish real BGE-M3 quality, CUDA execution, or
CUDA performance.
