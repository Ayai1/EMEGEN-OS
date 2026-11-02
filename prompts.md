# EngramOS prompts (150-Credit Budget Edition)

Run everything at `temperature = 0` through the cached LLM wrapper (`llm_cache`).
Version these prompts: hash each prompt template (`sha256`) and log `reader_prompt_hash` and `judge_prompt_hash` in `eval_runs`.

## Budget-Preserving Execution Rules (150 Credits Total)
1. **Batch 3–4 chronological sessions per extraction call** (up to ~6,000 input tokens) within the same `history_id` instead of 1 session per call. This cuts repeated system-prompt overhead by $70\%$ and reduces API calls from ~50/history to ~12/history.
2. **Zero-credit slot canonicalization in the grey zone**: Use local GPU `bge-reranker-v2-m3` instead of an LLM call for slot pairs with cosine similarity in $[\tau_{\text{low}}, \tau_{\text{high}})$.
3. **Batch trigger generation per history**: Generate trigger cues for all extracted `constraint` and `preference` facts in a history in a **single** LLM call after session extraction finishes.
4. **Regex fast-path for Query View Classifier**: Only call the LLM view classifier (Section C) when deterministic temporal patterns do not match.
5. **Uncached nonce for determinism checks**: `llm_cache` includes `cache_nonce` in its primary key `(model, prompt_hash, temperature, cache_nonce)`. Normal calls use `cache_nonce = ''`. The Friday judge determinism check passes `cache_nonce = 'det_run_2'` (`bypass_cache = True`) on the second pass so the API is genuinely called twice and both calls are billed to the 150-credit ledger.

---

## A. Slot extraction (chronological session batch, via muse-spark)

Ingest sessions sequentially in chronological `(session_date, session_id)` order within each `history_id`. Group up to 4 consecutive sessions into a single prompt.

**System:**

```
You extract durable memory facts from chat sessions between a user and an assistant.
Return a JSON array only, no prose.

For each fact about the USER or the user's project/life, emit an object:
{
  "session_id": "<session_id from the header>",
  "turn_idx": <integer index of the turn the fact comes from>,
  "entity": "<lowercase snake_case noun, e.g. user, database, auth_service, sister>",
  "attribute": "<lowercase snake_case, e.g. job_title, port, token_ttl, home_city>",
  "value": "<short value, keep numbers/names/units exactly as written>",
  "cardinality": "single" | "multi",
  "kind": "state" | "preference" | "event" | "constraint" | "other",
  "as_of": "<ISO date YYYY-MM-DD if the turn explicitly states when the fact became true in the past, else null>",
  "replaces": "<old value if the turn explicitly says it changed, else null>",
  "confidence": <0.0-1.0>
}

Rules:
- "single" = only one value can be true at a time (job, city, port, current plan, diet). "multi" = values accumulate (hobbies, books read, pets, visited cities).
- Emit facts in the exact chronological order they appear in the turns.
- Only extract facts the text actually states. Never infer or guess.
- If the user says something changed ("switched from X to Y", "moved to Z", "I no longer..."), emit the NEW value and set "replaces".
- Reuse an existing slot name (entity, attribute) when it refers to the same concept. Existing slots for this history:
  {existing_slots}
- Skip chitchat, greetings, questions, and generic assistant explanations.
- Keep numbers, names, and dates exactly as written.
```

**User:**

```
{chronological_session_blocks}
```
*(Format each block as `=== Session {session_id} | Date: {session_date} ===` followed by `[turn_idx] role: text`.)*

### Deterministic Timestamp, Tie-Breaking & Value Normalization
For the $k$-th extracted fact (`fact_seq` $= k \in \{0, 1, 2, \dots\}$) in a session:
- Every turn is stored with `turn.event_time = session_unix_ts + (turn_idx * 1.0)`.
- If `as_of` is `null`, set:
  $$\text{valid\_from} = \text{session\_unix\_ts} + (\text{turn\_idx} \times 1.0) + (k \times 10^{-3})$$
- If `as_of` is an explicit ISO date (`YYYY-MM-DD`), parse `as_of_unix_ts` and set:
  $$\text{valid\_from} = \text{as\_of\_unix\_ts} + (\text{turn\_idx} \times 10^{-3}) + (k \times 10^{-6})$$
  *Note on `as_of` tie-breaking:* Because `turn_idx` belongs to the reporting session rather than the historical `as_of` date, scaling `turn_idx` by $10^{-3}$ keeps the offset strictly sub-second ($< 1\text{ s}$). This guarantees deterministic ordering among multiple backdated facts without ever interleaving with integer-second turn offsets (`+1.0`, `+2.0`, ...) from an actual session that occurred on `as_of_unix_ts`.
- **Value Normalization (`value_norm`)**: Before checking whether a new fact supersedes or confirms an existing fact (`G_prev`), normalize `value` via `normalize_value(value)` (lowercase, trim punctuation/articles, collapse whitespace, convert English number words `zero`–`twenty` to digits so `"three bedrooms"` and `"3 bedrooms"` match). When `G_prev.value_norm == F.value_norm`, update `G_prev.last_confirmed_at`, `G_prev.last_confirmed_turn_id`, and `G_prev.confirmation_count` rather than closing `G_prev` or dropping `F`'s turn provenance.

### Slot Canonicalization & Conservative Calibration
1. **Exact match** on `(entity, attribute)` against existing slots in the same history: reuse immediately.
2. **Dense similarity** via `bge-m3` embedding of `"entity attribute"` against existing slots in the same history:
   - If $\text{cosine} \ge \tau_{\text{high}}$: reuse existing `(entity, attribute)`.
   - If $\tau_{\text{low}} \le \text{cosine} < \tau_{\text{high}}$: score with local `bge-reranker-v2-m3` on GPU (zero API credits). Because `bge-reranker-v2-m3` is trained on query-passage relevance rather than bare slot-name equivalence, format the pair with explicit slot context:
     - Query: `"Does the user attribute '{entity}.{attribute}' (example value: '{value}') refer to the exact same property as '{existing_entity}.{existing_attribute}'?"`
     - Passage: `"Slot: {existing_entity}.{existing_attribute} (known value: '{existing_value}')"`
     Merge if reranker score $\ge \tau_{\text{rerank}}$.
   - If $\text{cosine} < \tau_{\text{low}}$: create a new slot.
3. **Dev Calibration Protocol (Avoiding 3-parameter overfitting on 50 pairs)**:
   - Fix conservative cosine bounds $(\tau_{\text{high}}, \tau_{\text{low}}) = (0.88, 0.75)$ so auto-merges only occur at very high lexical/semantic overlap.
   - Extract 3 dev histories, collect 50 candidate slot pairs in the grey zone $[0.65, 0.95]$, and hand-label them (`1 = same slot`, `0 = distinct slot`).
   - Calibrate $\tau_{\text{rerank}}$ (subject to a conservative precision floor $\text{Precision} \ge 0.90$) to maximize $F_{0.5}$:
     $$F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{(0.5^2 \cdot \text{Precision}) + \text{Recall}}$$
     weighting Precision $2\times$ over Recall because a false slot merge wrongly closes an active fact.
   - Report dev Precision, Recall, and $F_{0.5}$ together with exact $95\%$ Clopper-Pearson binomial confidence intervals to reflect the uncertainty of a 50-pair sample. Freeze thresholds in `run_config_json` before extracting the remaining histories.

---

## B. Trigger cue generation (batched per history for `kind IN ('constraint', 'preference')`)

To save credits, pass all non-quarantined `constraint` and `preference` facts for a history in a single call (up to 15 facts per call).

**System:**

```
You will be given a list of remembered user constraints and preferences.
For each item, write 4 short, realistic future user requests that would silently depend on this fact WITHOUT mentioning its distinctive keywords.
Return JSON only:
{
  "triggers": [
    {"fact_id": "<fact_id>", "cues": ["...", "...", "...", "..."]}
  ]
}
Rules:
- Each cue must be under 20 words, phrased as a natural user request or task prompt.
- Do NOT copy distinctive nouns, brands, medical terms, or numbers from the original fact.
```

**User:**

```
Facts:
{json_list_of_constraint_facts_with_raw_turn}
```

Store each cue as a `facts` row with `key_type='trigger'`, `parent_fact_id = fact_id`, `valid_from = parent.valid_from`, `valid_until = parent.valid_until`, and `key_text = cue`. If the parent fact is later superseded, propagate `valid_until` to its child triggers.

---

## C. Query view classifier (Regex fast-path + cached LLM fallback)

**Step 1: Zero-credit deterministic fast-path (code)**
- If query matches past-change patterns (`/\b(before|previously|used to|prior to|history of|how many times|changed from)\b/i`): return `{"view": "history", "as_of_date": null}`.
- If query matches current-state patterns (`/\b(currently|right now|latest|presently|what is my current)\b/i`) and contains no explicit month/year/date: return `{"view": "current", "as_of_date": null}`.
- Otherwise, call the cached LLM classifier below (~35% of queries).

**Step 2: Cached LLM classifier prompt**

```
Classify the question's temporal view. Return JSON only:
{"view": "current" | "as_of" | "history", "as_of_date": "<YYYY-MM-DD or null>"}
- "current": asks about the present/latest state ("what is...", "where do I...")
- "as_of": asks about the state at a specific past date or month ("what was my job in March 2023?")
- "history": asks how something changed over time or what it used to be before a change ("what did I use before switching?")
Question date: {question_date}
Question: {question}
```

Retrieval filters:
- `current`: `valid_until IS NULL AND quarantined = 0`
- `as_of`: `valid_from <= t_star AND (valid_until IS NULL OR valid_until > t_star) AND quarantined = 0`
- `history`: `quarantined = 0`, ordered by `valid_from ASC` (includes superseded facts annotated with `[valid: {from} -> {until}]`).

---

## D. Reader (parameterized ordering label across systems)

Parameterize `{ordering_label}` per system so the prompt never contradicts the sort order of `{memory_items}`, and log `"memory_order"` in `eval_runs.run_config_json`:
- `full_context`, `engram_no_gate`, `engram`: `ordering_label = "ordered chronologically from oldest to newest"` (`memory_order = "chronological_asc"`)
- `hybrid_rag`: `ordering_label = "ordered by relevance score"` (`memory_order = "relevance_desc"`)
- `recency`: `ordering_label = "ordered reverse-chronologically from newest to oldest"` (`memory_order = "chronological_desc"`)

```
Answer the question concisely using only the memory items below.
If the memory contains a changed or updated fact, use the most recent valid one unless the question asks about the past.
If the answer is not in the memory, say "I don't know."

Memory ({ordering_label}, each item has a date and session ID):
{memory_items}

Question date: {question_date}
Question: {question}
```

**Judge**: Use LongMemEval's official evaluation prompt at `temperature = 0`. Compute `sha256` of the exact rendered judge template and log it in `eval_runs.judge_prompt_hash`.

---

## E. Recency baseline (run Friday night before building supersession)

Retrieve top-$k$ ($k=10$) turns using hybrid RAG (`bge-m3` dense + `turns_fts` BM25 fused via RRF), sort retrieved turns strictly by `event_time DESC` (`session_date` + `turn_idx` offset so later turns inside the same session rank above earlier turns), set `ordering_label = "ordered reverse-chronologically from newest to oldest"`, and prepend this single line to the Reader system prompt:

```
If the memory items below contain conflicting facts, always trust the item with the latest timestamp/turn index (shown first).
```

No LLM extraction, no slot table, zero extraction credits.