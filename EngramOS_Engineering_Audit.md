# EngramOS Engineering Audit: Problems and Recommendations

**Audit scope:** `BUILD_PLAN.md`, `prompts.md`, and `schema.sql`  
**Review type:** Static design review and SQLite schema initialization test  
**Overall assessment:** Promising architecture, but several correctness, security, and reproducibility issues should be resolved before implementing the complete pipeline.

> **Scope limitation:** This audit covers the three supplied specification/schema files. The application’s Python implementation was not supplied, so findings about missing runtime safeguards are specification gaps or risks to verify—not confirmed defects in unseen application code.

---

## 1. Executive summary

The most consequential issues are:

1. Backdated timestamp rules do not guarantee the ordering properties claimed in the documents.
2. Restatements retain only aggregate/last-confirmation metadata, not every supporting turn relationship.
3. Quarantined facts may still influence retrieval through raw turns, FTS, or embeddings unless every retrieval path applies the same eligibility policy.
4. A read-then-check API budget circuit breaker is vulnerable to concurrent overspend.
5. Batched extraction needs deterministic sequential post-processing to maintain evolving slot state.
6. Supersession needs transactional updates and explicit handling of both neighboring intervals.
7. SQLite foreign keys and immutability rules do not currently enforce all documented invariants.
8. Evaluation, cache identity, and cost attribution need additional precision.

### Severity overview

| ID | Severity | Area | Main issue |
|---|---|---|---|
| C1 | Critical | Temporal model | Backdated timestamp ordering is not guaranteed |
| C2 | Critical | Provenance | Individual restatement evidence is not retained as fact relationships |
| C3 | Critical | Security | Quarantined content may leak through turn retrieval |
| C4 | Critical | Budget | Concurrent requests can overspend or duplicate API calls |
| C5 | Critical | Extraction | Batched extraction can conflict with evolving chronological slot state |
| C6 | High | Supersession | Adjacent equal-value intervals are not consistently normalized |
| C7 | High | Transactions | Multi-step supersession needs an atomic transaction |
| C8 | High | Database integrity | Cross-history references are possible |
| C9 | High | Immutability | Turn immutability is documented but not enforced |
| C10 | High | FTS | FTS synchronization is only partially protected |
| C11 | High | Constraints | Several columns lack domain and consistency checks |
| C12 | High | SQLite setup | Foreign-key enforcement must be enabled on every connection |
| C13 | High | Security gate | The 2-of-3 gate has ambiguous authority and threat semantics |
| C14 | High | Trigger lifecycle | Trigger eligibility and parent lifecycle propagation are incomplete |
| C15 | High | Security evaluation | Attack benchmark does not establish broad security robustness |
| C16 | High | Query classification | Regex precedence can misclassify dated historical questions |
| C17 | High | Temporal retrieval | Month/year as-of semantics are undefined |
| C18 | High | Reader prompt | Evidence authority and conflict-resolution rules are underspecified |
| C19 | Medium/High | Trigger generation | Per-history batching instructions conflict with the 15-fact limit |
| C20 | High | Cache | Cache identity omits generation/provider configuration |
| C21 | High | Cost attribution | Evaluation rows cannot be tied reliably to individual API executions |
| C22 | Medium/High | Judge | Determinism does not establish judge correctness |
| C23 | High | Statistics | Confidence intervals and resampling assumptions need clarification |
| C24 | High | Benchmark | Split and leakage controls need stronger verification |
| C25 | High | Claims | Token-compression claim needs a consistent accounting definition |
| C26 | High | Build plan | Budget wording and phase accounting are ambiguous |
| A1–A15 | Medium | Architecture | Additional design and operational gaps |

---

# 2. Critical findings

## C1. Backdated timestamp ordering is not guaranteed

**Affected locations:**
- `BUILD_PLAN.md`: lines 46, 69–70
- `prompts.md`: lines 57–65
- `schema.sql`: lines 42–49, 174–176

The documents use two timestamp formulas.

Normal facts:

```text
session_unix_ts + turn_idx + fact_seq * 1e-3
```

Backdated facts:

```text
as_of_unix_ts + turn_idx * 1e-3 + fact_seq * 1e-6
```

The documentation claims that the backdated offsets remain strictly below one second and do not interleave with actual session turn offsets. That is not guaranteed unless explicit bounds are enforced.

Problems:

- `turn_idx` has no documented upper bound.
- `fact_seq` has no documented upper bound.
- Multiple sessions on the same date may share or overlap timestamp ranges, depending on how `session_unix_ts` is constructed.
- Conversation event time and the time a fact became valid are different concepts.
- A date-only `as_of` value does not specify an exact time of day.

**Recommendations:**

- Separate `event_time`, `valid_from`, and `recorded_at`.
- Prefer integer timestamps (for example, microseconds) over floating-point timestamp offsets.
- Store explicit deterministic tie-breakers such as session order, turn index, and fact sequence.
- Define date-only semantics: beginning of date, end of date, or date-level interval boundary.
- Document how same-date sessions are ordered.

Suggested fields:

```sql
event_time_us INTEGER NOT NULL,
valid_from_us INTEGER NOT NULL,
recorded_at_us INTEGER NOT NULL,
source_session_order INTEGER NOT NULL,
source_turn_idx INTEGER NOT NULL,
fact_seq INTEGER NOT NULL
```

A deterministic ordering tuple could be:

```text
(valid_from_us, source_session_order, source_turn_idx, fact_seq)
```

Do not implement interval stitching until the temporal model is unambiguous.

## C2. Restatement handling loses individual provenance

**Affected locations:**
- `prompts.md`: line 65
- `schema.sql`: lines 68–70, 188–199, 232–234

When a value is repeated, the proposed algorithm updates:

- `last_confirmed_at`
- `last_confirmed_turn_id`
- `confirmation_count`

It then returns the original fact ID.

This retains the latest confirmation and a count, but not a direct record of every confirmation turn.

Example:

| Turn | Statement |
|---|---|
| T1 | My preferred database is PostgreSQL. |
| T2 | I generally use PostgreSQL. |
| T3 | PostgreSQL is still my preferred database. |

The fact can retain T1, T3, and a count of 3, but the fact table does not preserve a separate relationship to T2.

**Recommendation:** Add a confirmation/evidence table.

```sql
CREATE TABLE fact_confirmations (
    confirmation_id TEXT PRIMARY KEY,
    fact_id TEXT NOT NULL REFERENCES facts(fact_id),
    turn_id TEXT NOT NULL REFERENCES turns(turn_id),
    confirmed_at_us INTEGER NOT NULL,
    recorded_at_us INTEGER NOT NULL
);
```

Keep `confirmation_count` only as an optional cached aggregate. Preserve every supporting turn relationship for evidence retrieval, audit, and replay.

## C3. Quarantined content may still be retrievable through raw turns

**Affected locations:**
- `BUILD_PLAN.md`: lines 92–103
- `prompts.md`: lines 136–139, 151–156
- `schema.sql`: lines 24–29, 73–75, 100–109

The design quarantines extracted facts and lowers `turns.base_trust`, while retaining the original turn. Retrieval combines facts, FTS turns, and dense turn embeddings.

The documents specify quarantine filtering for facts, but do not fully specify equivalent filtering for raw turns and embeddings.

A poisoned turn could therefore remain available to the Reader through a raw-turn retrieval path even if its extracted fact is quarantined.

**Recommendations:**

- Centralize retrieval eligibility in one shared policy.
- Apply it to fact retrieval, FTS, dense turn retrieval, trigger retrieval, and any reranking candidates.
- Define whether a quarantined turn quarantines all facts extracted from it.
- Define whether embeddings remain stored for audit but are excluded from retrieval.
- Define restoration/unquarantine behavior.
- Decide whether low-trust content is excluded or merely downweighted.

Illustrative filtering pattern:

```sql
SELECT
    t.turn_id,
    t.raw_content,
    t.event_time,
    t.base_trust
FROM turns t
WHERE t.history_id = ?
  AND NOT EXISTS (
      SELECT 1
      FROM gate_events g
      WHERE g.turn_id = t.turn_id
        AND g.decision = 'quarantine'
  );
```

This is illustrative only; the implementation must define how multiple gate decisions for a turn are resolved.

A security filter applied to only one retrieval path is not a complete security boundary.

## C4. Credit circuit breaker is vulnerable to concurrency races

**Affected locations:**
- `BUILD_PLAN.md`: lines 14–24, 44–47
- `prompts.md`: lines 3–11
- `schema.sql`: lines 112–129

The plan proposes checking:

```sql
SELECT SUM(cost_credits) FROM llm_cache;
```

A check-then-call approach is not safe when asynchronous requests run concurrently. Multiple requests can read the same balance and all pass before any request records its cost.

The cache primary key also does not prevent two workers from making the same external API request simultaneously after the same cache miss.

**Recommendations:**

- Implement atomic budget reservations that account for in-flight requests.
- Reserve estimated maximum cost in a transaction before calling the API.
- Reconcile the reservation against actual cost after the response.
- Release unused reservation.
- Define handling for timeouts, retries, provider charges on failed requests, and process crashes.
- Add per-cache-key locking or a database-backed in-flight request state.
- Keep a unique cache key, but do not treat uniqueness as API-call deduplication.

Suggested flow:

```text
Request
  -> Estimate maximum cost
  -> Begin write transaction
  -> Check cache
  -> Reserve budget atomically
  -> Commit reservation
  -> Call API
  -> Record actual cost
  -> Release unused reservation
```

## C5. Batched extraction needs deterministic sequential post-processing

**Affected locations:**
- `BUILD_PLAN.md`: lines 63–70
- `prompts.md`: lines 6–18, 39–47

The prompts recommend batching 3–4 chronological sessions per extraction call, while also requiring the existing slot registry to evolve after each session.

A model receiving several sessions and only the pre-batch `{existing_slots}` may produce facts without reliably simulating the database state after every earlier session in that batch.

Potential failures include duplicate slots, inconsistent attributes, incorrect `replaces` values, and missed confirmations.

**Recommendation: Retain batching but make post-processing deterministic.**

1. Extract candidate facts for the batch.
2. Validate the returned JSON.
3. Sort facts by session order, turn index, and fact sequence.
4. Canonicalize each fact against the live slot registry.
5. Insert each fact sequentially.
6. Update the slot registry after every insertion.

The LLM should identify candidate facts. Deterministic application code should own canonicalization, temporal ordering, and supersession.

---

# 3. Supersession and temporal database findings

## C6. Adjacent identical intervals are not consistently normalized

**Affected location:** `schema.sql`, lines 178–228

The algorithm checks whether the immediate predecessor has the same normalized value. It does not explicitly define normalization against both predecessor and successor.

Backdated insertion can leave adjacent intervals with the same value, for example:

| Fact | Value | Interval |
|---|---|---|
| F1 | A | 100–200 |
| F2 | A | 200–300 |

These intervals represent the same continuous state but remain separate logical facts.

**Recommendation:**

- Inspect predecessor, new fact, and successor after insertion.
- Define whether adjacent equal values should be merged into one logical interval.
- Preserve all original fact and turn provenance when merging.
- Do not delete audit records merely to simplify the logical interval view.

## C7. Supersession requires a transaction boundary

**Affected location:** `schema.sql`, lines 181–228

A logical insertion requires multiple operations:

1. Find predecessor.
2. Find successor.
3. Calculate the new interval.
4. Insert the new fact.
5. Update predecessor.
6. Propagate changes to child triggers.

A crash between operations could leave overlapping active facts or gaps.

**Recommendation:** Wrap the entire operation in one transaction, for example:

```sql
BEGIN IMMEDIATE;

-- Find predecessor and successor.
-- Insert new fact.
-- Update predecessor.
-- Update child trigger intervals.
-- Insert confirmation/audit records.

COMMIT;
```

Roll back the complete operation on any error. Serialize writes within a `history_id`.

---

# 4. Database schema and integrity findings

## C8. Cross-history references are possible

**Affected location:** `schema.sql`, lines 16–30 and 50–76

`facts.history_id` and `facts.turn_id` are independently constrained. The schema does not guarantee that the referenced turn belongs to the same history as the fact.

**Recommendation:**

Add a composite uniqueness constraint:

```sql
UNIQUE(history_id, turn_id)
```

Then use a composite foreign key:

```sql
FOREIGN KEY(history_id, turn_id)
REFERENCES turns(history_id, turn_id)
```

Apply equivalent history-isolation checks to other relationships where appropriate.

## C9. Turn immutability is not enforced

**Affected location:** `schema.sql`, lines 12–38

The schema comments state that turns must never be updated or deleted, but no triggers enforce this.

**Recommendation:**

```sql
CREATE TRIGGER prevent_turn_update
BEFORE UPDATE ON turns
BEGIN
    SELECT RAISE(ABORT, 'Turns are immutable');
END;

CREATE TRIGGER prevent_turn_delete
BEFORE DELETE ON turns
BEGIN
    SELECT RAISE(ABORT, 'Turns are immutable');
END;
```

Use an explicit controlled repair or migration mechanism if administrative corrections are required.

## C10. FTS synchronization is only partially protected

**Affected location:** `schema.sql`, lines 34–38 and 82–89

The FTS tables use external-content mode and currently have insert triggers. They do not have corresponding delete or update synchronization triggers.

The design assumes that underlying text never changes, but this is not enforced for every indexed table.

**Recommendations:**

- Enforce turn immutability.
- Enforce immutability of `facts.key_text`, or implement complete FTS update/delete triggers.
- Add an integrity/rebuild procedure for FTS indexes.
- Test index consistency after migrations and recovery.

## C11. Important columns lack CHECK constraints

**Affected location:** `schema.sql`

Examples of missing validation:

| Column | Recommended protection |
|---|---|
| `turns.role` | Restrict allowed role values |
| `turns.source_type` | Restrict allowed source values |
| `facts.key_type` | Validate slot/trigger field consistency |
| `facts.valid_until` | Require NULL or greater than `valid_from` |
| `facts.quarantined` | Require 0 or 1 |
| `facts.trust` | Require 0–1 |
| `turns.base_trust` | Require 0–1 |
| `eval_runs.correct` | Require NULL, 0, or 1 |
| `gate_events.signals_fired` | Require 0–3 |
| `llm_cache.in_tokens` | Require nonnegative value |
| `llm_cache.cost_credits` | Require nonnegative value |

Example:

```sql
CHECK (quarantined IN (0, 1))
```

```sql
CHECK (trust >= 0 AND trust <= 1)
```

```sql
CHECK (
    valid_until IS NULL
    OR valid_until > valid_from
)
```

Also define constraints ensuring that slot rows have entity, attribute, and value fields, while trigger rows have a valid parent and the appropriate trigger fields.

## C12. Foreign-key enforcement is connection-dependent

**Affected location:** `schema.sql`, line 10

`PRAGMA foreign_keys = ON` applies to the connection that executes it. Every application connection must enable it.

**Recommendation:**

```python
def connect_db(path):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn
```

Put this in the single database connection factory.

---

# 5. Provenance gate and security findings

## C13. The 2-of-3 gate has ambiguous semantics

**Affected locations:**
- `BUILD_PLAN.md`: lines 89–103
- `schema.sql`: lines 100–109

The gate uses:

1. Bridging similarity.
2. Policy contradiction.
3. Untrusted source.

A 2-of-3 decision rule means a user-originated malicious memory that triggers only the bridging signal may still be stored. This may be acceptable under the intended threat model, but that limitation must be explicit.

There is also an ambiguity around legitimate user policy updates. A new preference may contradict an old preference without being malicious.

**Recommendations:**

- Define protected system policies separately from user preferences.
- Establish authority levels, such as `system_policy`, `user_policy`, `user_preference`, `external_claim`, and `tool_output`.
- Define which authority levels may supersede which others.
- Distinguish legitimate policy updates from attempts to override protected invariants.
- Document the exact behavior of every signal combination.

## C14. Trigger cues have incomplete lifecycle protection

**Affected locations:**
- `prompts.md`: lines 86–113
- `schema.sql`: lines 50–76

Issues:

- Parent quarantine propagation is not fully defined.
- Supersession propagation is described but not transactionally guaranteed.
- Trigger regeneration after parent changes is unspecified.
- Embedding generation for trigger rows is not clearly specified.

**Recommendations:**

Define trigger eligibility through its parent:

```text
Trigger eligible =
    Parent exists
    AND Parent is not quarantined
    AND Parent is temporally valid
    AND Trigger is temporally valid
```

Also:

- Version trigger generation.
- Record the model and prompt version.
- Define regeneration and invalidation.
- Generate and store embeddings if trigger cues participate in dense retrieval.
- Make parent lifecycle propagation transactional.

## C15. Security benchmark is too limited for broad robustness claims

**Affected location:** `BUILD_PLAN.md`, lines 90–103 and 128–132

Fifteen attack chains and 50 benign turns are useful as an initial synthetic benchmark, but they cannot establish broad robustness.

The adaptive attack coverage is particularly limited.

**Recommendations:**

Report two distinct outcomes:

1. Memory ingestion success: whether malicious content entered retrievable memory.
2. Downstream answer manipulation success: whether the Reader followed malicious instructions or treated poisoned content as trusted truth.

Define attack success precisely. Report exact binomial confidence intervals and disclose the synthetic nature and limitations of the benchmark.

---

# 6. Retrieval and prompt findings

## C16. Query-view regex precedence can misclassify dated historical questions

**Affected location:** `prompts.md`, lines 117–139

The history regex runs before current-state classification. A query such as:

```text
What was my previous job in March 2023?
```

may be classified as `history`, even though it specifies a date.

**Recommendation:**

Use a clearly defined priority:

1. Explicit date/period query → `as_of`.
2. Explicit change-history query → `history`.
3. Explicit present-state query → `current`.
4. Otherwise → cached LLM classifier.

Define how combined date and history requests should behave.

## C17. Month/year as-of semantics are undefined

**Affected location:** `prompts.md`, lines 127–139

For:

```text
What was my job in March 2023?
```

The prompt does not establish whether March means:

- State on March 1.
- State on March 31.
- Any job held during March.
- Job held for the majority of March.

These produce different results.

**Recommendation:**

Define period semantics and support interval-overlap queries where appropriate.

Store the original query period separately from the resolved timestamp.

## C18. Reader prompt lacks a sufficiently explicit evidence policy

**Affected location:** `prompts.md`, lines 143–162

The Reader is told to use memory items and prefer the most recent valid fact unless the question is historical.

This does not fully address:

- Conflicting evidence.
- User statements versus assistant suggestions.
- External/tool claims.
- Quarantined or low-trust evidence.
- Overlapping validity intervals.
- Missing dates.
- Multiple supporting facts.
- Unresolved contradictions.

**Recommendation:** Explicitly instruct the Reader to:

1. Use only eligible memory items.
2. Distinguish user assertions, assistant suggestions, external claims, and tool outputs.
3. Resolve temporal questions using validity intervals.
4. Never equate recency with authority.
5. Acknowledge unresolved conflicts.
6. Avoid substituting current values for historical answers.
7. Say "I don't know" when evidence is insufficient.

## C19. Trigger-generation batching instructions conflict

**Affected locations:**
- `BUILD_PLAN.md`: line 105
- `prompts.md`: lines 86–103

The build plan says one call per history, while the prompt specification limits a call to 15 facts.

Histories with more than 15 eligible facts require multiple calls.

**Recommendation:**

Use deterministic batches:

```python
for batch in chunks(eligible_facts, 15):
    generate_triggers(batch)
```

Validate that every input fact has exactly one output entry and four valid cues. Check word count, duplicate cues, fact IDs, and malformed responses.

Measure trigger precision as well as recall@5.

---

# 7. Cache, API accounting, and reproducibility

## C20. Cache identity omits generation configuration

**Affected locations:**
- `prompts.md`: lines 3–11
- `schema.sql`: lines 112–129

Current cache key:

```sql
PRIMARY KEY (
    model,
    prompt_hash,
    temperature,
    cache_nonce
)
```

It does not capture all generation parameters, such as:

- `max_tokens`
- `top_p`
- Frequency/presence penalties
- Response-format configuration
- Tool configuration
- Provider/model revision
- API version

Two different calls could therefore collide.

**Recommendation:**

Add fields such as:

```sql
generation_config_json TEXT NOT NULL,
generation_config_hash TEXT NOT NULL
```

Include the generation configuration hash in cache identity. Record provider, model revision, API version, response format, and seed where applicable.

## C21. Evaluation rows cannot be tied reliably to API execution costs

**Affected location:** `schema.sql`, lines 112–161

`eval_runs` records evaluation metrics, while `llm_cache` records cached responses and aggregate cost. There is no direct per-execution relationship.

This makes it difficult to calculate cost by question, system, phase, or history.

**Recommendation:**

Introduce an API execution ledger separate from the response cache.

Illustrative schema:

```sql
CREATE TABLE llm_call_events (
    call_id TEXT PRIMARY KEY,
    cache_model TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    cache_nonce TEXT NOT NULL DEFAULT '',
    run_id TEXT,
    question_id TEXT,
    history_id TEXT,
    phase TEXT NOT NULL,
    task_tag TEXT NOT NULL,
    actual_cost REAL NOT NULL,
    created_at REAL NOT NULL
);
```

The cache stores reusable responses; the execution ledger records actual API usage.

## C22. Judge determinism does not establish judge correctness

**Affected location:** `BUILD_PLAN.md`, line 53

Two identical outputs at temperature zero establish repeatability for those calls, not correctness.

**Recommendations:**

- Compare outputs.
- Record and inspect disagreements.
- Distinguish parsing failures from model variation.
- Manually verify a subset of judgments.
- Freeze the judge prompt only after diagnostics.

---

# 8. Evaluation and experimental-design findings

## C23. Statistical reporting needs clearer assumptions

**Affected location:** `BUILD_PLAN.md`, lines 31–36, 66, 80, 103, 117

Issues:

- The assumed 20% discordance rate for McNemar power is not guaranteed.
- Planned power is not achieved power.
- Question-level bootstrap may underestimate uncertainty if questions share histories.
- Small attack and benign samples produce wide confidence intervals.
- Canonicalization precision intervals need the correct positive-class denominator.

**Recommendations:**

- Report observed discordant pairs and exact McNemar results.
- Preserve system pairing in bootstrap samples.
- Investigate whether questions share histories; use cluster-level resampling if necessary.
- Report exact binomial confidence intervals for security metrics.
- Report canonicalization positive/negative counts and metric denominators.
- Avoid claiming a 90% precision guarantee based solely on a point estimate.

## C24. Benchmark split needs leakage controls

**Affected location:** `BUILD_PLAN.md`, lines 29–36 and 48

The seeded split is reproducible in principle, but the plan does not fully specify checks for:

- Shared histories.
- Duplicate or near-duplicate questions.
- Synthetic attacks derived from test examples.
- Calibration examples derived from test histories.
- Prompt changes after inspecting test results.

**Recommendations:**

- Hash `splits.json`.
- Record dataset version and source checksum.
- Verify split integrity before test execution.
- Ensure test histories are not used for calibration or prompt tuning.
- Document the unit of independence for statistical analysis.

## C25. Headline token-compression claim needs a consistent accounting definition

**Affected location:** `BUILD_PLAN.md`, lines 6–8

Token comparisons can be misleading if different components are counted for different systems.

**Recommendation:**

Define:

```text
Token Compression =
1 - (Engram prompt tokens / Full-context prompt tokens)
```

Use consistent accounting for:

- System prompt.
- Memory items.
- Retrieved turns.
- Extracted facts.
- Trigger cues.
- Query tokens.

Report average and total tokens across the test set, plus the percentage of questions where Engram uses more tokens than the baseline.

---

# 9. Build-plan and budget findings

## C26. Credit-budget instructions are ambiguous

**Affected location:** `BUILD_PLAN.md`, lines 18–24, 58, 85, 108, 113

The phase budget adds up:

| Phase | Credits |
|---|---:|
| Phase 1 | 19 |
| Phase 2 | 32 |
| Phase 3 | 18 |
| Phase 4 | 67 |
| Buffer | 14 |
| **Total** | **150** |

However, the phrase "Hard circuit breaker at 136 credits before Monday" is ambiguous.

If 136 credits can be spent before Monday, only 14 remain for a 67-credit test phase.

The apparent intended plan is:

```text
Friday–Sunday maximum: 69 credits
Monday test evaluation: 67 credits
Emergency buffer: 14 credits
Total: 150 credits
```

**Recommendations:**

- Clarify the pre-Monday maximum as 69 credits.
- Verify that at least 67 credits remain before the locked test phase.
- Implement phase-specific reservations/limits.
- Clarify the "four ablation reader/judge runs" description against the five-row ablation ladder.

---

# 10. Additional architectural issues

These are additional risks and missing specifications, rather than all being confirmed runtime bugs.

| ID | Problem | Recommendation |
|---|---|---|
| A1 | No schema migration/version mechanism | Add `schema_version` and migrations |
| A2 | Embeddings lack referential integrity | Validate referenced turns/facts |
| A3 | Embedding model identification is incomplete | Store model revision and dimension |
| A4 | No embedding invalidation policy | Version embeddings by source content hash |
| A5 | `trust` and `base_trust` can diverge | Define a single trust propagation policy |
| A6 | No explicit slot uniqueness/invariant rules | Define canonical slot identity and cardinality rules |
| A7 | `facts.kind` has no CHECK constraint | Restrict to documented values |
| A8 | Gate events have no explicit decision identity policy | Define whether multiple evaluations per turn are allowed |
| A9 | Trigger generation has no model/version metadata | Store trigger-generation provenance |
| A10 | Retrieval scores lack an explicit breakdown | Log dense, lexical, reranker, and trust contributions |
| A11 | Malformed extraction JSON failure policy is missing | Add validation and bounded retries |
| A12 | Interrupted extraction recovery is unspecified | Make ingestion idempotent |
| A13 | Duplicate turn handling is unspecified | Add deterministic ingestion identity and duplicate detection |
| A14 | User assertions and assistant inferences are not distinguished | Preserve source role and assertion status |
| A15 | Backdated corrections lack explicit semantics | Define correction versus supersession |

Additional embedding-table concern:

`embeddings` uses a polymorphic `(ref_type, ref_id)` relationship without foreign keys. The application must verify that every reference exists and that the vector dimension matches the expected model.

---

# 11. Recommended architecture separation

Avoid making `db.py` responsible for all business logic.

Suggested responsibilities:

| Component | Responsibility |
|---|---|
| Ingestion layer | Session parsing, deduplication, immutable turn storage |
| Extraction engine | LLM extraction, JSON validation, batching |
| Provenance engine | Trust classification, quarantine, audit |
| Temporal memory engine | Slot canonicalization, interval stitching, confirmations, as-of views |
| Retrieval engine | Dense retrieval, FTS, RRF, trust filtering |
| Query classifier | Current, as-of, history |
| Reader | Answer generation |
| Evaluation | Judge, metrics, statistics, replay |
| Shared infrastructure | SQLite transactions, LLM cache, budget ledger, logging, configuration |

The LLM should identify candidate facts. Deterministic application code should enforce database invariants.

---

# 12. Recommended test suite

Implement deterministic unit tests before spending the full API budget.

| Test | Expected behavior |
|---|---|
| Same value repeated three times | One logical fact, three confirmation records |
| A → B → C | Correctly ordered validity intervals |
| A → B → A | A returns as a new valid interval |
| Backdated A inserted between B and C | Existing intervals stitched correctly |
| Backdated fact matches predecessor | Confirmation recorded without incorrect supersession |
| Backdated fact matches successor | No duplicate adjacent logical interval |
| Same-day sessions | Deterministic ordering without collisions |
| Two histories with identical slots | No cross-history contamination |
| Quarantined fact | Never appears in eligible retrieval |
| Quarantined parent trigger | Trigger cannot be retrieved |
| Benign external document | Not automatically quarantined |
| Legitimate policy update | Correct authority/update semantics |
| Concurrent budget requests | Never exceed 150 credits |
| Concurrent identical cache misses | Only one live API request |
| API timeout after reservation | Reservation reconciled |
| Interrupted supersession transaction | Database remains consistent |
| FTS versus source records | Results remain synchronized |
| Month-level as-of query | Correct period semantics |
| Multiple confirmations | Correct evidence and chronology |

### Property-based interval tests

Generate random sequences such as:

```text
A, B, C, A, D, B, A
```

Insert them in chronological and randomized arrival order.

Verify:

1. No overlapping active intervals.
2. No more than one current value for a single-valued slot.
3. Every supersession relationship is valid.
4. No interval has negative duration.
5. Historical queries return the expected state.
6. Replaying the same events produces the same logical memory.

---

# 13. Remediation priority

## P0 — Resolve before full implementation

1. Replace fractional timestamp assumptions with an explicit temporal ordering model.
2. Define date-level as-of semantics.
3. Add individual fact confirmation records.
4. Centralize retrieval eligibility and prevent quarantined-turn leakage.
5. Implement atomic credit reservations and cache-miss deduplication.
6. Implement deterministic sequential post-processing for batched extraction.

## P1 — Resolve before benchmark evaluation

1. Make supersession transactional.
2. Handle both predecessor and successor interval normalization.
3. Add composite foreign keys and database constraints.
4. Enforce turn and indexed-text immutability.
5. Define policy authority and gate semantics.
6. Complete trigger lifecycle handling.
7. Fix query-view classification and period semantics.
8. Improve Reader evidence instructions.
9. Add generation configuration to cache identity.
10. Add per-execution API cost attribution.
11. Strengthen benchmark split and statistical procedures.
12. Clarify phase budgets.

## P2 — Resolve before final packaging

1. Add migrations and schema versioning.
2. Add embedding versioning and validation.
3. Add retrieval score diagnostics.
4. Add robust extraction recovery.
5. Build the Replay UI.
6. Produce Pareto plots.
7. Finalize README, threat model, and demo video.

---

# 14. Final assessment

**Recommendation: Continue with the architecture, but revise the implementation specification before building the full pipeline.**

The main concern is the interaction between:

- Temporal validity.
- Provenance.
- Supersession.
- Quarantine.
- Retrieval.

These components determine whether EngramOS can make trustworthy claims about historical memory and resistance to memory poisoning.

The project does not need a complete architectural rewrite. It needs stronger invariants, transactional operations, centralized security enforcement, and more precise experimental accounting.

### Audit limitations

This document covers the supplied `BUILD_PLAN.md`, `prompts.md`, and `schema.sql`. The SQLite schema was reported to initialize successfully in the initial review. No application source code was provided, so Python runtime behavior, actual API integration, concurrency implementation, and retrieval correctness remain to be verified.

### Suggested implementation order

**Correct the temporal model → revise the schema → revise the prompts → revise the build plan → implement unit tests → run the benchmark.**
