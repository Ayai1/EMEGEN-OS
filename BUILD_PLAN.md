# EngramOS: scoped build plan (150-Credit Budget Edition)

Deadline: **Nov 4, 9:00 PM ET = Nov 5, ~7:30 AM IST** (US daylight saving ends Nov 1; confirm on the portal).
Plan to submit by the **evening of Nov 4 IST**. Working time starts Fri Oct 30 evening.

## The claim (one sentence, fill in measured numbers at the end)
> Verbatim storage plus slot-tracked supersession and as-of retrieval reaches near full-context accuracy
> on knowledge-update questions at ~$X\%$ of the tokens, and a provenance gate cuts memory-poisoning success from $A\%$ to $B\%$.

Everything below exists to measure that sentence within **150 free API credits**. If a task doesn't, cut it.

---

## 150-Credit Budget Ledger & Hard Circuit Breakers

Every LLM call goes through `llm_cache` (`temperature = 0`), which computes `cost_credits` on insert and raises a `BudgetExceededError` if cumulative spend crosses the phase ceiling.

| Phase | Scope | Cost-Saving Mechanism | Credit Cap |
|---|---|---|---:|
| **Phase 1 (Fri): Smoke + Dev Baselines** | 22 dev questions $\times$ 3 baselines (`full_context`, `hybrid_rag`, `recency`) + Judge (incl. uncached determinism check) | Cache all reader/judge calls; `hybrid_rag` & `recency` send only top-10 turns (~1.5k tok) | **19 cr** |
| **Phase 2 (Sat): Dev Extraction + Ablations** | Extract 22 dev histories + 4 ablation reader/judge runs | Batch 3–4 sessions/call (~12 calls/history); local GPU `bge-reranker` for grey-zone slots (0 cr); regex view fast-path | **32 cr** |
| **Phase 3 (Sun): Poisoning + Trigger Dev** | Seeded policy facts + 15 attack chains + 50 benign stress turns + 30 trigger scenarios | Reuse already-extracted dev histories; batch trigger generation into 1 call/history | **18 cr** |
| **Phase 4 (Mon): Locked Test Evaluation** | **78 test questions** (64 `knowledge-update` + 14 `temporal-reasoning`) | One-shot test extraction (batched) + single cached evaluation pass per system | **67 cr** |
| **Buffer / Re-runs** | Hard circuit breaker at 136 credits before Monday; 14 credits emergency buffer | Never re-run `full_context` or extraction once cached in `llm_cache` | **14 cr** |
| **TOTAL** | **100 questions total (22 dev + 78 test)** covering all ~78 KU questions in LongMemEval-S | | **150 cr** |

---

## Rules for the whole build
1. **Identical Reader & Judge**: Same `reader_model`, `judge_model`, and `judge_prompt_hash` for every system in the headline table. The Reader prompt's `{ordering_label}` is parameterized per system (`chronological_asc`, `relevance_desc`, `chronological_desc`) so `recency` never contradicts its own prompt header.
2. **Power-aware split & realistic statistical claims**:
   - Use all ~78 `knowledge-update` (KU) questions and 22 `temporal-reasoning` (TR) questions from LongMemEval-S (**100 questions total**).
   - **Dev split ($N = 22$)**: 14 KU + 8 TR. Used for threshold calibration, prompt tuning, and failure-mode diagnostics.
   - **Test split ($N = 78$)**: 64 KU + 14 TR. Locked in `splits.json` on Friday night and run **once** on Monday.
   - **Statistical reporting**: Under an exact two-sided McNemar test at $\alpha = 0.05$ on $N = 64$ KU questions assuming $\sim 20\%$ discordant pairs ($\sim 13$ discordant pairs), detecting an 8-question net advantage ($\sim 13$ percentage points) has $\sim 55\text{--}60\%$ power, while $\ge 80\%$ power requires a $\sim 15\text{--}16$-percentage-point gap ($\sim 10$ net questions). Always report $95\%$ paired bootstrap CIs ($10{,}000$ resamples) and exact McNemar $p$-values rather than relying on binary significance thresholds, and explicitly label the 14-question TR test slice as **directional** due to sample size.
3. **Full Reproducibility Logging**: Every row in `eval_runs` logs `reader_model`, `judge_model`, `reader_prompt_hash`, `judge_prompt_hash`, `out_tokens`, and `run_config_json` (seed, `top_k`, `memory_order`, calibrated `tau_high`, `tau_low`, `tau_rerank`, gate weights).
4. **Zero Placeholders**: Placeholder numbers in slides, README, or UI are forbidden.

---

## Fri Oct 30 (tonight): foundations, baselines, and the recency gate
- [ ] Read hackathon rules: required Meta model usage, video length, open-source/license requirements, and Contributor-ID pricing/rate limits.
- [ ] `nvidia-smi` (confirm local VRAM for `bge-m3` and `bge-reranker-v2-m3`). `uv init`, repo skeleton, `.gitignore`, MIT license.
- [ ] Apply `schema.sql`. Write `db.py` including:
  - `llm_cache` wrapper with token counting, `cache_nonce` support (`bypass_cache=True` sets a fresh `cache_nonce` so uncached verification calls still log tokens and credits), and the **150-credit circuit breaker** (`SELECT SUM(cost_credits) FROM llm_cache`).
  - Deterministic turn timestamping (`event_time = session_unix_ts + turn_idx`) and per-history `prev_hash` chain.
- [ ] API smoke test: 30-request async burst against `muse-spark` (measure rate limit headers and exact token pricing).
- [ ] Download LongMemEval-S. Sample the **22 dev** (14 KU, 8 TR) and **78 test** (64 KU, 14 TR) question IDs with `seed = 42`. Write to `splits.json` and commit.
- [ ] Implement and run the 3 zero-extraction baselines on **dev ($N = 22$)**:
  - **Baseline 1 (`full_context`)**: Reader sees the full history (`ordering_label = "ordered chronologically from oldest to newest"`, cached so you never pay for these input tokens again).
  - **Baseline 2 (`hybrid_rag`)**: `bge-m3` dense + `turns_fts` BM25, RRF fusion, top-10 turns (`ordering_label = "ordered by relevance score"`).
  - **Baseline 3 (`recency`)**: Same top-10 hybrid RAG turns sorted strictly by `event_time DESC` (`session_date + turn_idx`), `ordering_label = "ordered reverse-chronologically from newest to oldest"`, plus the one-line recency rule in `prompts.md` Section E.
- [ ] **Judge determinism check (with `bypass_cache=True` / `cache_nonce='det_run_2'`)**: Run the judge twice at `temperature = 0` on the 22 dev answers using distinct `cache_nonce` values so the second pass genuinely calls the API and counts toward the credit budget; verify $100\%$ agreement and commit `judge_prompt_hash`.
- [ ] **Fri Night Headroom Check**: Compare `full_context` vs `recency` on the 14 dev KU questions tonight *before* writing a single line of supersession code. Inspect every dev question where `recency` fails:
  - Does `recency` fail because the older turn had higher lexical/dense similarity and pushed the newer turn out of top-10, or because both appeared in top-10 and the reader got confused, or because the question was an `as-of` / `history` query?
  - Document the exact breakdown of failure modes to guide Saturday's extraction and supersession build.

**Done when:** dev table of accuracy / prompt_tokens / out_tokens / latency / credits spent exists for `full_context`, `hybrid_rag`, and `recency`, judge determinism is verified across two live API passes, and failure modes of `recency` are categorized.

---

## Sat Oct 31: calibrated extraction, interval-stitching supersession, and as-of views
- [ ] **Step 1 — Calibrate slot canonicalization on 3 dev histories first**:
  - Extract 3 dev histories (batching 3–4 chronological sessions per call).
  - Hand-label 50 candidate slot pairs across cosine $[0.65, 0.95]$.
  - Fix conservative cosine bounds $(\tau_{\text{high}}, \tau_{\text{low}}) = (0.88, 0.75)$ and calibrate $\tau_{\text{rerank}}$ using contextualized `bge-reranker-v2-m3` scores to maximize $F_{0.5}$ (subject to $\text{Precision} \ge 0.90$) on those 50 pairs. Report Precision and $F_{0.5}$ with $95\%$ exact Clopper-Pearson binomial confidence intervals. Freeze thresholds in `config.json` before extracting the remaining 19 dev histories.
- [ ] **Step 2 — Causal extraction across the 22 dev histories**:
  - Parallelize workers **across distinct `history_id`s only**; within each `history_id`, process session batches **strictly sequentially in chronological `(session_date, session_id)` order** so `{existing_slots}` and `prev_hash` are deterministic.
  - Assign `valid_from = session_unix_ts + turn_idx + (fact_seq * 1e-3)` for normal facts, and `valid_from = as_of_unix_ts + (turn_idx * 1e-3) + (fact_seq * 1e-6)` for backdated `as_of` facts so backdated tie-breakers remain strictly sub-second.
  - Normalize extracted values (`value_norm`) before comparing. Implement the **interval-stitching supersession rule** in `db.py`: update `last_confirmed_at`, `last_confirmed_turn_id`, and `confirmation_count` when `G_prev.value_norm == F.value_norm` (preserving confirmation provenance), and stitch intervals when `G_prev.value_norm != F.value_norm`.
- [ ] **Step 3 — Query view classifier & hybrid fact+turn retrieval**:
  - Implement the regex fast-path + cached LLM fallback (`prompts.md` C).
  - Retrieve over `facts` (`valid_until` filtered by view) + `turns_fts` + `bge-m3` embeddings, returning the parent verbatim turns (plus `last_confirmed_turn_id` when applicable) with dates.
- [ ] **Step 4 — Dev ablation ladder ($N = 22$)**:
  1. `hybrid_rag`
  2. `recency`
  3. `+ slot keys (no supersession)`
  4. `+ interval-stitching supersession`
  5. `+ as-of / history view routing` (`engram_no_gate`)
- [ ] Hand-audit 20 supersession chains in SQLite: report slot-invalidation precision and recall with $95\%$ binomial CIs.
- [ ] **Diagnostic Go/No-Go (Sat night — failure-mode read, not a noisy threshold test)**: Because a 1–2 question swing on $N = 14$ dev KU questions is binomial noise, evaluate Saturday's go/no-go by inspecting the per-question transitions against Friday's failure-mode breakdown:
  - Did interval-stitching and `as-of` routing resolve the specific stale-fact intrusion and temporal-window failures identified on Friday without introducing false-supersession regressions?
  - If `recency` and `engram_no_gate` remain within 1–2 questions on dev KU, report that parity transparently and frame the headline around token compression vs `full_context` ($>90\%$ fewer tokens), `as-of`/history capability, and the Sunday provenance quarantine gate.

**Done when:** 5-row ablation table on dev + 50-pair slot canonicalization Precision/$F_{0.5}$ with $95\%$ CIs + 20-chain supersession audit exist, with total cumulative spend $\le 51$ credits.

---

## Sun Nov 1: provenance gate, seeded policy facts, stratified benign stress test, and trigger keys
- [ ] **Seed baseline policy facts for gate evaluation**:
  - Because LongMemEval histories contain natural chat rather than pre-pinned security/system policies, seed each evaluation history in the gate benchmark with a small set of pinned policy/constraint facts (`source_type = 'system_policy'`, `kind = 'constraint'`, `trust = 1.0`, e.g., project security invariants, pinned communication/scheduling rules, and credential/endpoint constraints) so `signal_policy` has explicit ground-truth policies to check against.
- [ ] **Quarantine gate (`2 of 3` signals)**:
  1. `signal_bridge`: bridging-similarity spike across shortening query prefixes.
  2. `signal_policy`: contradiction against a pinned system/user policy fact (`kind = 'constraint'`).
  3. `signal_source`: untrusted provenance (`source_type IN ('tool', 'external_doc')`).
  - Gate sets `quarantined = 1` on extracted facts and lowers `turns.base_trust`; it **never deletes turns**.
- [ ] **Poisoning & Stratified False-Positive Evaluation**:
  - **Attack set (15 synthetic chains injected onto dev histories)**: 10 MINJA-style bridging chains + 5 held-out variants (including an adaptive variant engineered to stay just below the `signal_bridge` threshold).
  - **Stratified 50-turn Benign Stress Set (synthetic injections to prevent a flattered false-positive rate)**:
    - **20 benign untrusted-source turns** (`source_type IN ('tool', 'external_doc')`, e.g., benign tool outputs, pasted API docs, meeting notes) where `signal_source = 1` fires automatically—so *any* single false trigger of `signal_bridge` or `signal_policy` causes a real false quarantine.
    - **20 benign user policy/constraint turns** (e.g., *"never schedule meetings before 10 AM"*, *"always use port 5432 instead of 3306"*) that legitimately set or update rules.
    - **10 benign multi-step topic-shift user turns** that rapidly pivot entities across sessions.
  - Report **Attack Success Rate (ASR)** and **Benign False-Positive Rate (FPR)** (with $95\%$ binomial CIs) broken down by the 3 benign strata across `full_context`, `hybrid_rag`, `recency`, and `engram`.
- [ ] **Batched Trigger Cues (`prompts.md` B)**:
  - Run batched trigger generation (1 call per history) for `constraint` and `preference` facts.
  - Evaluate on 30 self-made implicit-constraint scenarios (zero lexical overlap between constraint and task query). Report `recall@5` with vs without trigger keys. If gain is $<10$ points, cut triggers from the Monday test run to save credits.

**Done when:** Poison ASR + stratified 50-benign FPR table and 30-scenario trigger recall@5 table exist, with cumulative spend $\le 69$ credits.

---

## Mon Nov 2: one-shot test evaluation ($N = 78$) and replay UI
- [ ] Verify frozen config: confirm `tau_high`, `tau_low`, `tau_rerank`, `memory_order`, and prompt hashes match Saturday/Sunday dev locks. Check `SELECT SUM(cost_credits) FROM llm_cache` (must have $\ge 67$ credits remaining).
- [ ] **Batch-extract the 78 test histories** (sequentially within each `history_id`, parallelized across histories).
- [ ] **Run all systems once on `test` ($N = 78$: 64 KU + 14 TR)**:
  - `full_context`, `hybrid_rag`, `recency`, `engram_no_gate`, `engram`.
  - Compute $95\%$ paired bootstrap CIs ($10{,}000$ resamples) and exact paired McNemar $p$-values (`engram` vs `hybrid_rag` and `engram` vs `recency`).
- [ ] Generate Pareto plots: Accuracy vs Prompt Tokens, Latency, and Poison ASR.
- [ ] Build the single-page **Replay UI** (reads directly from SQLite `turns`, `facts`, `gate_events`, and `eval_runs`—zero live API calls):
  - Timeline showing interval-stitching supersession (including a backdated `as_of` fact, a restatement confirmation via `last_confirmed_turn_id`, and a same-session turn tie-break).
  - Quarantine audit inspector showing the 3 gate signals on a blocked poison chain vs a passed benign `external_doc` policy turn.
  - Head-to-head metrics & reproducibility hash table.

---

## Tue Nov 3: freeze, package, and record
- [ ] Freeze code. No new LLM calls.
- [ ] Write `README.md`:
  - Filled-in one-sentence claim with $95\%$ CIs.
  - One-command SQLite reproduction from cached runs + `budget_summary` table showing total run cost under 150 credits.
  - **Explicit Threat Model & Synthetic Benchmark Disclosure**: State clearly that LongMemEval does not natively contain pinned security policies or adversarial turns, so the pinned policy facts (`source_type = 'system_policy'`), the 15 poisoning attack chains, and the 50 stratified benign stress turns are synthetic overlays injected onto LongMemEval histories.
  - **Statistical Power & Scope Note**: Document the exact McNemar power characteristics ($N = 64$ KU has $\sim 55\text{--}60\%$ power for a $\sim 13$-pt delta and $\ge 80\%$ power for a $\sim 15\text{--}16$-pt delta at $\le 20\%$ discordant pairs; $N = 14$ TR is directional) and include an honest **"Where `recency` and `full_context` win"** section.
- [ ] Record the demo video using the deterministic Replay UI.

## Wed Nov 4: buffer and submit
- [ ] Test fresh clone in a clean `uv` environment. Submit by **Nov 4 evening IST**.