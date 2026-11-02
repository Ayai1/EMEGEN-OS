# EngramOS — Updated Engineering Reassessment

**Purpose:** Reconcile the original *EngramOS Engineering Audit* with the independent reassessment and its SQLite test results. This document is intended to replace the original audit's severity ranking and remediation plan as the implementation guide.

**Review basis**
- `BUILD_PLAN.md`
- `prompts.md`
- `schema.sql`
- Original `EngramOS_Engineering_Audit.md`
- Independent reassessment supplied on 30 October 2026

**Important scope limit:** The independent reviewer reports running SQLite 3.45 tests against the latest `schema_cee1c1.sql`. Those tests support the specific database findings below. No Python application source was supplied, so runtime behavior, locking, retry handling, retrieval filters, and API accounting still require implementation-level verification. Do not interpret a schema-level test as proof that the complete application is correct.

---

## 1. Executive conclusion

The original audit identified real design risks, but it overstated the severity of several findings and proposed some fixes that are either unnecessary for the current benchmark or unsafe as written.

The updated conclusion is:

1. **Quarantine leakage through raw-turn retrieval is the highest-priority correctness and security risk.** Filtering extracted facts is not sufficient if FTS or dense turn retrieval can still surface a quarantined source turn.
2. **Temporal interval normalization has a reproduced edge case.** A backdated fact can leave adjacent intervals with the same value (`B, A, A`). The insertion algorithm must reconcile both neighbors.
3. **Supersession must be atomic.** Multi-step interval updates should run in one transaction.
4. **Turn immutability must be column-scoped.** A blanket update-blocking trigger prevents the gate from updating `base_trust`.
5. **Cache identity must include the full generation request.** In particular, a cached truncated response must not be reused after changing `max_tokens` or other generation settings.
6. **SQLite foreign-key enforcement must be enabled on every connection.**
7. **Batch extraction can remain, provided the application sorts, validates, canonicalizes, and inserts facts sequentially.**
8. **The broad regex query-classification fast path should be removed or reduced to unambiguous present-state phrases.** Its cost savings are negligible and it can misroute historical questions.
9. **The project's token-efficiency claim must distinguish query-time prompt tokens from total lifecycle cost.** On LongMemEval, ingestion may cost more than the query-time savings.
10. **The pre-Monday budget cap is 69 credits, not 136, if 67 credits are reserved for Monday's evaluation and 14 credits are held as contingency.**

The architecture does **not** need a wholesale rewrite. Keep the existing design and focus on these bounded fixes. Defer lower-value schema hardening and prompt redesign until the core pipeline and tests work.

---

## 2. Corrected severity and disposition matrix

The severity labels below are implementation priorities, not claims that every issue is already a runtime defect. “Confirmed” means the independent reassessment reports a reproducible schema/test result; “design gap” means the documents leave behavior underspecified; “runtime verification” means Python code is needed.

| ID | Original finding | Updated verdict | Priority / disposition |
|---|---|---|---|
| C1 | Backdated timestamp ordering | Partly valid; original severity overstated. The reviewer found no collisions across 4,000 realistic turn/fact pairs. The meaningful unresolved issue is date-only `as_of` semantics; collisions reportedly begin only at very high `fact_seq` values (around 1,000). | **P1:** Define date-only semantics and test bounds. Do not redesign timestamps solely on the original collision argument. |
| C2 | Missing individual confirmation provenance | Valid, but low value for the current KU benchmark. | **Defer:** Do not add a confirmation table now. Existing latest-confirmation/count metadata is adequate for current scope unless full evidence replay becomes a requirement. |
| C3 | Quarantined content can leak through raw turns | Valid and most important. Facts are filtered, but FTS and dense turn retrieval reportedly lack equivalent filtering. Quarantine state is in `gate_events`, not on `turns`. | **P0:** Centralize eligibility and filter every retrieval route. |
| C4 | Concurrent budget race / duplicate cache misses | Valid in a multi-process or unlocked design, but low risk under a documented single-process asyncio architecture with one shared lock around check-and-reserve. | **P1:** Use a shared asyncio lock if and only if one process owns all requests and every request uses it. Add a DB reservation system only if that assumption changes. |
| C5 | Batched extraction state drift | Partly valid. The plan already canonicalizes slots in code/design. | **P1:** Keep batching; explicitly sort, validate, canonicalize, and insert sequentially against the live slot registry. |
| C6 | Adjacent equal-value intervals | Valid and reproduced. Backdated insertion can leave `B, A, A`. | **P0:** Normalize against both predecessor and successor; same-valued successor should be treated as confirmation/continuation, not a duplicate interval. |
| C7 | Non-atomic supersession | Valid. | **P0:** Wrap the complete supersession/interval-stitching operation in one transaction. |
| C8 | Cross-history references | Valid and reproduced at schema level, but low practical risk if application code always scopes operations by `history_id`. | **Defer:** Add defensive composite constraints only if simple and non-disruptive; ensure every query and write is history-scoped. |
| C9 | Turn immutability not enforced | Valid, but the original blanket-trigger fix breaks the gate's `base_trust` update. | **P0:** Use a column-scoped trigger that blocks changes to immutable source fields while allowing the gate-managed `base_trust` field. |
| C10 | FTS synchronization | Valid but minor. The reviewer reports the latest FTS trigger fix works; immutability reduces the remaining risk. | **P2:** Verify insert/update/delete synchronization in tests. Avoid a larger redesign if source text is immutable. |
| C11 | Missing CHECK constraints | Valid but low priority. The only immediately useful check identified is `valid_until > valid_from` where both values are non-null. | **P2:** Add only the useful interval check if compatible with open-ended intervals. Defer broad constraint expansion. |
| C12 | Foreign-key pragma per connection | Valid and reproduced: a fresh connection had `foreign_keys = 0`. | **P1:** Centralize SQLite connection creation and execute `PRAGMA foreign_keys=ON` for every connection. |
| C13 | Ambiguous 2-of-3 gate semantics | Valid as a documentation/test gap. | **P1:** Add a compact truth table for the three signals and expected decisions. Defer a complex authority hierarchy. |
| C14 | Trigger lifecycle propagation | Partly valid. Triggers are reportedly generated only from non-quarantined facts and are already first on the cut list. | **Defer / cut:** Keep trigger generation minimal; ensure quarantined facts cannot produce eligible triggers. Do not build complex lifecycle propagation now. |
| C15 | Security benchmark too limited | Valid. | **P1:** Report both whether poisoned content entered memory and whether it changed the final answer. Keep claims bounded to the tested attack set. |
| C16 | Regex precedence misclassifies historical questions | Concern valid, but the original example was wrong: “previous job in March 2023” does not match the regex according to the reviewer. A real failure is “How many days before the trip did I buy the tent?” being routed to `history`. | **P1:** Drop the broad regex fast path, or keep only obvious patterns such as “currently” and “right now”. |
| C17 | Month/year as-of semantics | Valid and moderately important. | **P1:** Define period-level interpretation for month/year questions and test boundary behavior. |
| C18 | Reader evidence policy underspecified | Partly valid, but the proposed prompt change conflicts with KU reward behavior and risks invalidating tuned baselines. | **Defer:** Preserve the current Reader prompt for the locked experiment. Revisit only in a separately versioned experiment. |
| C19 | Trigger-generation batch size conflict | Valid but trivial. | **P1:** Chunk trigger generation into batches of at most 15 facts. |
| C20 | Cache key omits generation configuration | Valid. A truncated response could be reused after `max_tokens` is raised. | **P0:** Hash the full request body, including model/provider and all generation parameters. |
| C21 | Cost attribution cannot identify API execution | Valid and medium importance. | **P1:** Add `run_id` and phase attribution to the cost ledger. |
| C22 | Judge determinism mistaken for correctness | Valid but minor. | **P2:** Hand-check approximately 20 verdicts. State that scores are not directly comparable with published results if judge model/prompt differs (the reviewer believes the official judge is GPT-4o; verify before stating as fact). |
| C23 | Statistical assumptions | Partly valid. The reassessment's exact calculation estimates roughly 68% power for an 11-vs-2 split over 13 discordant pairs, making the original 55–60% estimate slightly conservative. Cluster bootstrap is likely unnecessary for this short project. | **P2:** State the test and assumptions clearly; avoid overengineering the analysis. |
| C24 | Benchmark leakage controls | Valid and inexpensive to address. | **P1:** Hash and record `splits.json`; keep the test split locked and do not tune against it. |
| C25 | Token-compression claim | Valid and understated in the original audit. Query-time savings are not the same as total lifecycle savings. | **P0:** Report ingestion and query costs separately. Headline only query-time prompt-token reduction unless total-cost evidence supports a broader claim. |
| C26 | Budget wording contradiction | Valid. The listed allocation totals 150 (`19 + 32 + 18 + 67 + 14`). Reserving 67 for Monday and 14 as buffer leaves 69 for all pre-Monday work. | **P0:** Set the pre-Monday spend cap to 69 credits; retain 67 for evaluation and 14 as contingency. |

### Additional findings A1–A15

The original audit's additional architecture list was too broad for the current time and budget. The independent reassessment considers most of it generic rather than an immediate blocker.

| ID | Original concern | Updated disposition |
|---|---|---|
| A1 | Schema migration/version mechanism | Defer. Useful for a maintained product, not a blocker for the bounded experiment. |
| A2 | Embedding referential integrity | Defer; validate references in application code where practical. |
| A3 | Embedding model/revision/dimension metadata | Defer unless multiple embedding models or migrations are introduced. |
| A4 | Embedding invalidation by source hash | Defer; not a current blocker if source text is immutable. |
| A5 | `trust` / `base_trust` divergence | Document the intended distinction and propagation. Avoid a larger trust-model rewrite. |
| A6 | Slot uniqueness/cardinality rules | Define only the invariants needed by the current temporal memory implementation. |
| A7 | `facts.kind` CHECK | Defer. |
| A8 | Gate-event decision identity | Document whether multiple gate evaluations per turn are expected; defer schema expansion. |
| A9 | Trigger-generation model metadata | Defer with trigger generation. |
| A10 | Retrieval-score breakdown | Defer unless debugging shows it is necessary. |
| A11 | Malformed extraction JSON retries | Keep a bounded validation/retry path; avoid unbounded retries. |
| A12 | Interrupted ingestion recovery | Make ingestion idempotent if feasible; this is more valuable than several broad schema changes. |
| A13 | Duplicate-turn handling | Add deterministic ingestion identity/deduplication if duplicate imports are plausible. |
| A14 | User assertions vs assistant inferences | Preserve speaker/source role and distinguish user assertions from assistant-generated inferences. |
| A15 | Backdated correction semantics | Define whether an event is a correction, a historical assertion, or a new superseding value. |

---

## 3. Corrections to the original audit's key arguments

### 3.1 C1 — Do not overstate timestamp collision risk

The original audit recommended replacing the fractional-offset approach with a more extensive event-time/valid-time/recorded-time model and integer microsecond ordering.

The reassessment tested 4,000 realistic `(turn, fact)` pairs and reports zero collisions. It observed collisions only at unusually high `fact_seq` values (approximately 1,000 or more). That does not prove the scheme is universally collision-free, but it means the original “Critical” label and broad redesign recommendation are not justified by the available evidence.

**Corrected action**
- Keep the current timestamp scheme for the first implementation if realistic bounds are enforced and tested.
- Document maximum expected `turn_idx` and `fact_seq`.
- Define date-only `as_of` semantics precisely (for example, whether a date means start-of-day or end-of-day in the history's timezone).
- Test multiple sessions on the same date and same-day historical queries.
- Revisit integer timestamps only if tests expose an actual ordering failure or the product needs higher-resolution temporal guarantees.

### 3.2 C2 — Do not add a confirmation table for this benchmark

The original audit treated missing per-restatement provenance as Critical and proposed `fact_confirmations`.

For the current KU-oriented benchmark, the independent reviewer considers intermediate restatement links unnecessary. A latest confirmation pointer and count are sufficient for the current questions.

**Corrected action:** Defer the table. Preserve the option to add it later if the product needs complete evidence lineage, replay, or explainability for every confirmation.

### 3.3 C3 — Treat quarantine as a retrieval eligibility rule, not a fact-table flag

The critical issue is not simply that a quarantine column is missing from `turns`. Quarantine status currently lives in `gate_events`, while retrieval can access turns through multiple routes.

A turn may be absent from fact retrieval but still appear in:
- Full-text search (FTS).
- Dense turn retrieval.
- A candidate set passed to a reranker.
- Trigger or cue retrieval derived from that turn.

**Corrected action**
- Create one centralized eligibility function/policy used by all retrieval paths.
- A small `turn_security_status` table or equivalent materialized status may simplify filtering. Keep `gate_events` as the audit/event record; do not overload immutable turn content.
- Exclude quarantined turns from every candidate-generation path, not merely from final display.
- Ensure facts and triggers derived from a quarantined turn are also ineligible.
- Keep embeddings for audit if desired, but never retrieve them while their source is quarantined.
- Add tests that attempt retrieval through FTS, dense, facts, triggers, and reranking.

**Acceptance criterion:** A known quarantined attack turn must not be available to the Reader through any retrieval path, and must not influence the final answer through hidden candidate context.

### 3.4 C4 — A single-process lock is acceptable under an explicit assumption

The original audit proposed atomic database reservations and in-flight request records. The reassessment argues that the project can safely use one asyncio process and one shared lock around “check + reserve”, with each API call costing only around $0.15.

That is a reasonable scope reduction, but it is conditional.

**Corrected action**
- Document: exactly one application process owns the budget and all API calls pass through the same lock.
- Hold the lock across the budget check and reservation update.
- Prevent any direct call path from bypassing the budget manager.
- Reconcile reservation and actual cost after success/failure/timeout.
- If multiple processes, workers, or machines are introduced, replace this with database-backed atomic reservations.

Do not claim the lock protects a multi-process deployment.

### 3.5 C5 — Keep batching; make deterministic processing explicit

The issue is not batching by itself. The risk is applying a batch as if all extracted facts were based on an unchanged slot registry.

**Corrected sequence**
1. Extract candidate facts for a batch.
2. Validate the response structure and fields.
3. Sort by session order, turn index, and fact sequence.
4. Canonicalize each fact against the current live slot registry.
5. Insert/update one fact at a time.
6. Update the in-memory slot registry after each write.
7. Continue to the next candidate.

This preserves batch efficiency while making state transitions deterministic.

### 3.6 C6 — Normalize against both neighbors

The reassessment reproduced a case where inserting a backdated `A` between an earlier `B` and a later `A` leaves two adjacent `A` intervals.

**Corrected action:** After inserting a historical value, inspect both predecessor and successor intervals. If an adjacent interval has the same normalized value, merge/continue the logical interval or record a confirmation according to the project's temporal semantics. Do not create duplicate adjacent intervals simply because the new assertion arrived later.

### 3.7 C7 — Make supersession atomic

All writes that close the previous interval, insert the new interval, link supersession, and normalize neighboring intervals must occur in one transaction. If any step fails, roll back the entire operation.

### 3.8 C9 — Use column-scoped immutability

The original recommendation for a blanket `BEFORE UPDATE` trigger is unsafe because the gate needs to update `base_trust`.

**Corrected action:** Block updates only to immutable source fields (for example, raw content and original event/source metadata), while allowing explicitly mutable gate-managed fields such as `base_trust`. Verify the exact column list against the current schema and gate code.

The reviewer reports that a column-scoped `BEFORE UPDATE OF ...` trigger blocks source mutation while allowing the trust update.

### 3.9 C16 — Correct the regex example and remove the broad fast path

The original example “previous job in March 2023” was not reproduced. The reassessment's actual failing example is:

> “How many days before the trip did I buy the tent?”

The word “before” can describe a relative temporal relation rather than a historical query mode.

The regex classifier saves very little because the LLM classifier costs only a fraction of a cent per question.

**Corrected action:** Remove the broad regex shortcut. If retaining a fast path, restrict it to obvious current-state phrases such as “currently” and “right now”, and ensure all ambiguous queries go through the normal classifier.

### 3.10 C18 — Preserve the tuned Reader prompt for the locked experiment

The original audit proposed stronger evidence-authority instructions, including “Never equate recency with authority.” The reassessment points out that this conflicts with what KU questions reward and could invalidate tuned baselines.

**Corrected action:** Do not change the Reader prompt in the locked experiment. If a revised evidence policy is worth testing, make it a separate prompt/version and report it as a separate experiment.

### 3.11 C20 — Hash the full generation request

The most concrete cache failure is not merely provider identity. A response truncated under a lower `max_tokens` value must not be returned after the token limit is increased.

**Corrected action:** Construct cache identity from the complete normalized request body, including:
- Provider and model identifier.
- System/developer/user messages.
- Temperature and sampling settings.
- `max_tokens` / output-token limit.
- Tool or response-format settings.
- Any other parameter that can affect the generated response.

Do not reuse a cache entry when any generation-affecting field changes.

### 3.12 C23 — Keep statistical analysis proportionate

The reassessment's exact calculation estimates approximately 68% power for an 11-versus-2 split among 13 discordant pairs. This is somewhat higher than the original audit's 55–60% estimate. For the short project, elaborate cluster-bootstrap machinery is likely unnecessary.

**Corrected action:** State the test, sample size, discordant-pair counts, assumptions, and confidence interval method. Avoid claiming high statistical certainty from a small benchmark.

---

## 4. Two material issues missed by the original audit

### 4.1 Ingestion cost changes the total-cost claim

The original audit discussed query-time token accounting but missed the effect of ingesting each history.

The independent reassessment estimates:
- Approximately **$0.21** to ingest a history.
- Approximately **$0.14** for a full-context query.
- A break-even point of roughly **1.5 queries per history**.

These are estimates based on the pricing assumptions used by the reviewer, not independently re-verified live prices. They should be recalculated using the actual provider, model, and observed token counts before publication.

LongMemEval has roughly one question per history, so the benchmark may demonstrate query-time prompt-token savings without demonstrating total lifecycle cost savings.

**Corrected reporting**
- Report ingestion tokens and cost separately.
- Report query-time prompt tokens and cost separately.
- Report total per-history lifecycle cost.
- Use the phrase **“query-time prompt-token reduction”** unless the measured total-cost result supports a broader claim.
- Include the amortization assumption: how many queries are expected per ingested history.

### 4.2 The regex fast path is not economically worthwhile

The LLM query classifier is estimated to cost only a fraction of a cent per question. The broad regex path introduces misclassification risk while saving very little.

**Corrected reporting/action:** Remove it, or retain only a tiny set of unambiguous present-state patterns. Do not build a large rule system to save a negligible amount.

---

## 5. Updated implementation order

This order supersedes the original audit's P0/P1/P2 plan.

### P0 — Fix before meaningful evaluation

1. **C3 — Quarantine retrieval leakage**
   - Centralize eligibility.
   - Filter facts, FTS, dense turns, triggers, and reranker inputs.
   - Test that quarantined content cannot reach the Reader.
2. **C6 — Adjacent interval normalization**
   - Check predecessor and successor after backdated insertion.
   - Merge/confirm matching adjacent values.
3. **C7 — Transactional supersession**
   - Make interval closure, insertion, linking, and normalization atomic.
4. **C9 — Column-scoped immutability**
   - Protect source fields while allowing gate updates to `base_trust`.
5. **C20 — Complete cache identity**
   - Hash the full generation request and configuration.
6. **C25 — Correct cost/token accounting**
   - Separate ingestion from query-time metrics.
   - Avoid total-cost claims unless measured.
7. **C26 — Correct budget cap**
   - Pre-Monday maximum: 69 credits.
   - Monday evaluation reserve: 67 credits.
   - Contingency: 14 credits.

### P1 — Complete before the locked benchmark

1. **C12:** SQLite connection factory that enables foreign keys on every connection.
2. **C5:** Deterministic sort/validate/canonicalize/sequential insert for batched extraction.
3. **C4:** One-process asyncio lock around budget check-and-reserve; document the single-process assumption.
4. **C13:** Gate truth table and explicit decision semantics.
5. **C15:** Separate “poison entered memory” from “poison changed answer” metrics.
6. **C16:** Remove broad regex classification; retain only obvious present-state shortcuts if necessary.
7. **C17:** Define month/year as-of semantics.
8. **C19:** Chunk trigger generation into batches of no more than 15 facts.
9. **C21:** Add `run_id` and phase to cost records.
10. **C24:** Hash `splits.json` and keep the test split locked.
11. **A11/A12/A14:** Bounded malformed-JSON handling, idempotent ingestion, and user-vs-assistant assertion distinction.

### P2 — Only if time remains

- **C1:** Add realistic timestamp-bound tests and date-only semantics; do not perform a broad timestamp redesign without evidence.
- **C10:** FTS insert/update/delete synchronization tests.
- **C11:** Add the interval validity CHECK if compatible with open-ended intervals.
- **C22:** Hand-review around 20 judge decisions.
- **C23:** Report statistical assumptions clearly without complex resampling.
- **C8:** Add stronger cross-history constraints only if straightforward.
- **A1–A10, A13, A15:** Defer broader migrations, embedding lifecycle, score diagnostics, duplicate detection, and correction semantics unless implementation reveals a concrete blocker.

### Explicitly defer

- C2 confirmation/evidence table.
- Complex cross-history schema hardening.
- Complex trigger lifecycle propagation.
- Reader prompt redesign (C18) during the locked experiment.
- Broad trust/authority hierarchy redesign.
- Cluster bootstrap unless the evaluation design or sample size makes it necessary.

---

## 6. Updated acceptance tests

These are the tests that matter most for the revised scope.

| Area | Test | Expected result |
|---|---|---|
| Quarantine | Quarantined turn queried through FTS | No candidate returned |
| Quarantine | Quarantined turn queried through dense retrieval | No candidate returned |
| Quarantine | Fact derived from quarantined turn | Ineligible |
| Quarantine | Trigger derived from quarantined fact | Ineligible |
| Quarantine | Reranker receives candidate list | Quarantined source absent before reranking |
| Temporal | Insert `A → B → A` chronologically | Three valid periods, no overlap |
| Temporal | Insert a backdated `A` between `B` and later `A` | No adjacent duplicate `A` periods |
| Temporal | Backdated insertion matches successor | Confirmation/continuation behavior, not duplicate interval |
| Temporal | Same-day sessions | Deterministic ordering under documented bounds |
| Transactions | Force failure midway through supersession | All writes roll back |
| Immutability | Attempt to edit raw source content | Rejected |
| Immutability | Gate updates `base_trust` | Allowed |
| Cache | Change `max_tokens` | Cache miss |
| Cache | Change model/provider or other generation parameter | Cache miss |
| SQLite | Open a fresh connection | `PRAGMA foreign_keys` returns `1` |
| Extraction | Batch contains multiple updates to one slot | Sequential state is respected |
| Extraction | Malformed JSON | Bounded retry/failure; no partial writes |
| Budget | Concurrent requests in one process | Shared lock prevents overspend |
| Budget | Attempt API call outside budget manager | Prohibited by architecture/tests |
| Classifier | “How many days before the trip did I buy the tent?” | Correctly classified by normal classifier |
| Evaluation | Ingestion cost vs query cost | Recorded separately |
| Evaluation | Security attack | Reports memory-entry rate and answer-manipulation rate separately |
| Evaluation | Benchmark split | `splits.json` hash recorded and test set unchanged |

For interval logic, add randomized/property-based sequences if time permits. The essential invariants are:
- No overlapping intervals for a single-valued slot.
- No adjacent duplicate logical values after normalization.
- No negative-duration intervals.
- Historical queries return the expected state.
- Replaying the same ordered events produces the same logical memory.

---

## 7. Budget and experiment accounting

The allocation arithmetic is:

| Phase | Credits |
|---|---:|
| Existing planned work allocations | 19 |
| Existing planned work allocations | 32 |
| Existing planned work allocations | 18 |
| Monday evaluation reserve | 67 |
| Contingency reserve | 14 |
| **Total** | **150** |

The three pre-evaluation allocations total 69. Therefore:

- **Maximum pre-Monday spend: 69 credits.**
- **Do not spend the 67-credit Monday reserve during development.**
- **Keep 14 credits untouched as contingency unless explicitly approved.**
- Track reserved, estimated, and actual cost separately.
- A failed or timed-out request may still incur provider cost; reconcile it rather than assuming zero.

The phase names for the 19/32/18 allocations should be copied from the build plan rather than invented in this reassessment.

---

## 8. Claims and reporting language

Use these distinctions in the final report:

| Avoid saying | Prefer saying |
|---|---|
| “EngramOS reduces total cost” | “EngramOS reduces query-time prompt tokens” unless lifecycle cost is measured and lower |
| “The system is robust against memory poisoning” | “The system resisted the tested poisoning scenarios under the specified benchmark” |
| “Quarantined content is blocked” | Only claim this after all retrieval paths, including FTS and dense retrieval, pass tests |
| “The judge is reliable” | “The judge was deterministic under the tested setup; a sample of verdicts was manually reviewed” |
| “The timestamp model guarantees ordering” | State the tested bounds and date-only semantics |
| “The budget cannot be exceeded” | State the single-process lock assumption and measured enforcement behavior |

Record model identifiers, prompt versions, generation parameters, dataset split hash, and evaluation run ID with every benchmark result.

---

## 9. Final recommendation

**Proceed with EngramOS using the existing architecture. Do not restart the design.**

The implementation should concentrate on the security boundary around quarantine, deterministic temporal updates, transaction safety, cache correctness, and honest cost accounting. The original audit's broad timestamp redesign, confirmation table, blanket immutability trigger, and Reader prompt rewrite should not be implemented as originally proposed.

The most important end-to-end proof is this:

> A quarantined source turn must not enter any Reader-visible retrieval context, while legitimate temporal facts continue to update and answer historical queries correctly.

Once that passes alongside interval, transaction, cache, and budget tests, move to the locked benchmark. Keep the Reader prompt and test split stable during that experiment.

---

## 10. Evidence and limitations

- The independent reassessment reports SQLite 3.45 tests for the latest `schema_cee1c1.sql`, including clean schema application, FTS trigger behavior, cross-history reference acceptance, foreign-key pragma defaults, interval edge cases, and the immutability-trigger interaction with `base_trust`.
- The reassessment did not inspect the Python implementation because it was not supplied.
- Cost figures are estimates from the reassessment's stated pricing assumptions and should be recomputed against actual provider usage.
- The claim about the official judge being GPT-4o was presented as the reviewer's understanding, not verified evidence.
- The updated priorities are scoped to the current project timeline and benchmark, not a general production-readiness checklist.
