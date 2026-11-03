-- EngramOS (scoped build) -- SQLite schema
-- Design rules:
--   * turns are IMMUTABLE verbatim values. Never update or delete them.
--   * facts are the keys: they route to turns and carry validity.
--   * validity times use deterministic EVENT time with sub-turn tie-breakers, never wall-clock "now".
--   * every query is scoped by history_id so one DB can hold many LongMemEval histories.
--   * llm_cache tracks token & credit spend to enforce the 150-credit hard cap.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- 1. Immutable verbatim values ------------------------------------------------
-- Point 2 fix: event_time MUST be stored as:
--   session_unix_ts + (turn_idx * 1.0)
-- so two turns in the same session never share an identical timestamp.
CREATE TABLE IF NOT EXISTS turns (
  turn_id       TEXT PRIMARY KEY,            -- '{history_id}:{session_id}:{turn_idx}'
  history_id    TEXT NOT NULL,
  session_id    TEXT NOT NULL,
  turn_idx      INTEGER NOT NULL,
  role          TEXT NOT NULL,               -- 'user' | 'assistant' | 'tool' | 'system'
  raw_content   TEXT NOT NULL,
  token_count   INTEGER,
  source_type   TEXT NOT NULL DEFAULT 'user_turn',
                                             -- 'user_turn' | 'assistant_turn' | 'tool' | 'system_policy' | 'external_doc'
  base_trust    REAL NOT NULL DEFAULT 0.6,   -- gate adjusts trust; it does NOT delete turns
  event_time    REAL NOT NULL,               -- session_unix_ts + (turn_idx * 1.0)
  recorded_at   REAL NOT NULL,               -- transaction time (when ingested)
  prev_hash     TEXT,                        -- provenance chain within history_id: sha256(prev_hash || raw_content)
  hash          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_hist_time ON turns(history_id, event_time, turn_idx);

CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
  raw_content, content='turns', content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
  INSERT INTO turns_fts(rowid, raw_content) VALUES (new.rowid, new.raw_content);
END;

-- Preserve verbatim source turns and provenance. The gate may only change
-- base_trust; content and event metadata remain fixed after ingestion.
CREATE TRIGGER IF NOT EXISTS turns_immutable_update
BEFORE UPDATE ON turns
WHEN new.turn_id      IS NOT old.turn_id
  OR new.history_id   IS NOT old.history_id
  OR new.session_id   IS NOT old.session_id
  OR new.turn_idx     IS NOT old.turn_idx
  OR new.role         IS NOT old.role
  OR new.raw_content  IS NOT old.raw_content
  OR new.token_count  IS NOT old.token_count
  OR new.source_type  IS NOT old.source_type
  OR new.event_time   IS NOT old.event_time
  OR new.recorded_at  IS NOT old.recorded_at
  OR new.prev_hash    IS NOT old.prev_hash
  OR new.hash         IS NOT old.hash
BEGIN
  SELECT RAISE(ABORT, 'turn content and event metadata are immutable');
END;

CREATE TRIGGER IF NOT EXISTS turns_immutable_delete
BEFORE DELETE ON turns BEGIN
  SELECT RAISE(ABORT, 'source turns cannot be deleted');
END;

-- Durable ingestion checkpoints. Turn rows and gate_events remain authoritative
-- source/audit records; this table only tracks safe recovery of extraction.
CREATE TABLE IF NOT EXISTS ingestion_jobs (
  turn_id           TEXT PRIMARY KEY REFERENCES turns(turn_id),
  payload_hash      TEXT NOT NULL,            -- canonical caller payload identity
  session_unix_ts   REAL NOT NULL,
  session_date      TEXT NOT NULL,
  state             TEXT NOT NULL CHECK (state IN (
                      'pending_gate','approved','staged','completed','quarantined'
                    )),
  extraction_json   TEXT,                     -- validated original-schema response
  extraction_hash   TEXT,
  fact_count        INTEGER CHECK (fact_count IS NULL OR fact_count >= 0),
  applied_count     INTEGER NOT NULL DEFAULT 0 CHECK (applied_count >= 0),
  updated_at        REAL NOT NULL,
  CHECK (extraction_json IS NULL OR extraction_hash IS NOT NULL),
  CHECK (state NOT IN ('staged','completed') OR extraction_json IS NOT NULL),
  CHECK (state NOT IN ('staged','completed') OR fact_count IS NOT NULL),
  CHECK (fact_count IS NULL OR applied_count <= fact_count),
  CHECK (state != 'completed' OR applied_count = fact_count)
);

-- 2. Keys: slot facts and trigger cues, many-to-one onto turns ----------------
-- Tie-breaker rules:
--   * Normal fact (as_of IS NULL):
--       valid_from = session_unix_ts + (turn_idx * 1.0) + (fact_seq * 1e-3)
--   * Backdated fact (explicit as_of YYYY-MM-DD):
--       valid_from = as_of_unix_ts + (turn_idx * 1e-3) + (fact_seq * 1e-6)
--     Note: For backdated facts, turn_idx comes from the reporting session and is
--     scaled to 1e-3 seconds purely as a deterministic tie-breaker so it never
--     interleaves with integer-second turn offsets of an actual session on as_of_unix_ts.
CREATE TABLE IF NOT EXISTS facts (
  fact_id                TEXT PRIMARY KEY,
  history_id             TEXT NOT NULL,
  turn_id                TEXT NOT NULL REFERENCES turns(turn_id),
  fact_seq               INTEGER NOT NULL DEFAULT 0, -- 0-based extraction index within the turn/session
  key_type               TEXT NOT NULL CHECK (key_type IN ('slot','trigger')),
  entity                 TEXT,                    -- normalized snake_case, NULL for triggers
  attribute              TEXT,                    -- normalized snake_case, NULL for triggers
  value                  TEXT,                    -- raw extracted value, NULL for triggers
  value_norm             TEXT,                    -- normalized value (lowercased, number-words->digits, trimmed)
  cardinality            TEXT CHECK (cardinality IN ('single','multi')),
                                                  -- only 'single' slots can be superseded
  kind                   TEXT,                    -- 'state'|'preference'|'event'|'constraint'|'other'
  key_text               TEXT NOT NULL,           -- what gets embedded + FTS-indexed
  parent_fact_id         TEXT REFERENCES facts(fact_id),  -- for triggers: the constraint they point to
  valid_from             REAL NOT NULL,           -- monotonic event time with deterministic tie-breaker
  valid_until            REAL,                    -- NULL = currently active; strictly > valid_from when closed
  superseded_by          TEXT REFERENCES facts(fact_id),
  last_confirmed_at      REAL,                    -- updated when a later turn restates the same value_norm
  last_confirmed_turn_id TEXT REFERENCES turns(turn_id),
  confirmation_count     INTEGER NOT NULL DEFAULT 1,
  recorded_at            REAL NOT NULL,           -- transaction time created
  expired_at             REAL,                    -- transaction time invalidated
  trust                  REAL NOT NULL DEFAULT 0.6,
  quarantined            INTEGER NOT NULL DEFAULT 0,
  quarantine_reason      TEXT,
  confidence             REAL
);
CREATE INDEX IF NOT EXISTS idx_facts_slot   ON facts(history_id, entity, attribute, valid_from);
CREATE INDEX IF NOT EXISTS idx_facts_active ON facts(history_id, valid_until, quarantined);
CREATE INDEX IF NOT EXISTS idx_facts_turn   ON facts(turn_id);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
  key_text, content='facts', content_rowid='rowid'
);
-- Bug 1 fix: qualify new.key_text so SQLite FTS5 trigger succeeds on insert
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
  INSERT INTO facts_fts(rowid, key_text) VALUES (new.rowid, new.key_text);
END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE OF key_text ON facts
WHEN old.key_text IS NOT new.key_text BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, key_text)
    VALUES ('delete', old.rowid, old.key_text);
  INSERT INTO facts_fts(rowid, key_text) VALUES (new.rowid, new.key_text);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, key_text)
    VALUES ('delete', old.rowid, old.key_text);
END;

-- 3. Persisted vectors rebuild a history-scoped in-memory CPU/GPU index ------
CREATE TABLE IF NOT EXISTS embeddings (
  ref_type  TEXT NOT NULL CHECK (ref_type IN ('turn','fact')),
  ref_id    TEXT NOT NULL,
  model     TEXT NOT NULL,                   -- e.g. 'bge-m3'
  vec       BLOB NOT NULL,                   -- float16, 1024 dims
  content_hash TEXT,                         -- SHA-256 of source text; NULL legacy rows are stale
  PRIMARY KEY (ref_type, ref_id, model)
);

-- 4. Quarantine / gate decisions (audit trail, also powers the demo) ----------
CREATE TABLE IF NOT EXISTS gate_events (
  event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  turn_id        TEXT NOT NULL REFERENCES turns(turn_id),
  signal_bridge  REAL,                       -- bridging-similarity signal (0/1 or score)
  signal_policy  REAL,                       -- conflicts with pinned policy fact
  signal_source  REAL,                       -- untrusted source signal (1.0 when source_type IN ('tool','external_doc'))
  signals_fired  INTEGER NOT NULL,           -- quarantine when >= 2
  decision       TEXT NOT NULL CHECK (decision IN ('store','quarantine')),
  created_at     REAL NOT NULL
);
CREATE TRIGGER IF NOT EXISTS gate_events_immutable_update
BEFORE UPDATE ON gate_events BEGIN
  SELECT RAISE(ABORT, 'gate event history is immutable');
END;
CREATE TRIGGER IF NOT EXISTS gate_events_immutable_delete
BEFORE DELETE ON gate_events BEGIN
  SELECT RAISE(ABORT, 'gate event history is immutable');
END;

-- 5. LLM response cache + 150-credit budget ledger ----------------------------
-- Bug 2 fix: cache_nonce (default '') allows intentional uncached verification passes
-- (such as the second run of the judge determinism check with cache_nonce='det_run_2')
-- to hit the live API while still recording token and credit spend in llm_cache.
CREATE TABLE IF NOT EXISTS llm_cache (
  model         TEXT NOT NULL,
  prompt_hash   TEXT NOT NULL,               -- sha256 of complete canonical generation request
  temperature   REAL NOT NULL,
  cache_nonce   TEXT NOT NULL DEFAULT '',    -- '' for normal cached calls; 'det_run_2' when bypass_cache=True
  task_tag      TEXT NOT NULL DEFAULT 'unspecified',
                                             -- 'extract' | 'trigger' | 'classify' | 'reader' | 'judge' | 'judge_det_check'
  response      TEXT NOT NULL,
  in_tokens     INTEGER NOT NULL DEFAULT 0,
  out_tokens    INTEGER NOT NULL DEFAULT 0,
  cost_credits  REAL NOT NULL DEFAULT 0.0,   -- computed at insert from model rate card
  latency_ms    INTEGER,
  phase_tag     TEXT NOT NULL DEFAULT 'unspecified',
  run_id        TEXT NOT NULL DEFAULT 'unspecified',
  created_at    REAL NOT NULL,
  PRIMARY KEY (model, prompt_hash, temperature, cache_nonce)
);

-- Reservations make concurrent paid requests count against caps before the
-- provider call begins. Failed calls are charged conservatively at their
-- reservation when provider usage is unavailable.
CREATE TABLE IF NOT EXISTS llm_call_ledger (
  call_id          TEXT PRIMARY KEY,
  request_key      TEXT NOT NULL,
  model            TEXT NOT NULL DEFAULT 'unspecified',
  task_tag         TEXT NOT NULL,
  phase_tag        TEXT NOT NULL,
  run_id           TEXT NOT NULL,
  reserved_credits REAL NOT NULL CHECK (reserved_credits >= 0),
  actual_credits   REAL NOT NULL DEFAULT 0 CHECK (actual_credits >= 0),
  status           TEXT NOT NULL CHECK (status IN ('reserved','succeeded','failed')),
  started_at       REAL NOT NULL,
  finished_at      REAL,
  error_type       TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_llm_call_inflight_request
  ON llm_call_ledger(request_key) WHERE status = 'reserved';
CREATE INDEX IF NOT EXISTS idx_llm_call_ledger_phase_status
  ON llm_call_ledger(phase_tag, status);

CREATE VIEW IF NOT EXISTS budget_summary AS
SELECT task_tag, model, phase_tag, run_id, COUNT(*) AS api_calls,
       SUM(in_tokens) AS total_in_tokens, SUM(out_tokens) AS total_out_tokens,
       ROUND(SUM(credits_spent), 4) AS credits_spent,
       ROUND(SUM(credits_reserved), 4) AS credits_reserved
FROM (
  SELECT task_tag, model, phase_tag, run_id, in_tokens, out_tokens,
         cost_credits AS credits_spent, 0.0 AS credits_reserved
  FROM llm_cache
  UNION ALL
  SELECT task_tag, model, phase_tag, run_id, 0 AS in_tokens, 0 AS out_tokens,
         CASE WHEN status='failed' THEN actual_credits ELSE 0.0 END AS credits_spent,
         CASE WHEN status='reserved' THEN reserved_credits ELSE 0.0 END AS credits_reserved
  FROM llm_call_ledger WHERE status IN ('reserved','failed')
)
GROUP BY task_tag, model, phase_tag, run_id;

-- 6. Evaluation results (full reproducibility columns) ------------------------
CREATE TABLE IF NOT EXISTS eval_runs (
  run_id               TEXT NOT NULL,
  system               TEXT NOT NULL,        -- 'full_context' | 'hybrid_rag' | 'recency' | 'engram_no_gate' | 'engram'
  question_id          TEXT NOT NULL,
  question_type        TEXT,
  split                TEXT NOT NULL CHECK (split IN ('dev','test')),
  reader_model         TEXT NOT NULL,
  judge_model          TEXT NOT NULL,
  reader_prompt_hash   TEXT NOT NULL,
  judge_prompt_hash    TEXT NOT NULL,
  run_config_json      TEXT NOT NULL,        -- JSON: {"seed":42,"top_k":10,"memory_order":"...","tau_high":...,"tau_low":...,"tau_rerank":...,"gate_signals":...}
  answer               TEXT,
  correct              INTEGER,              -- judge verdict (0 or 1)
  evidence_recall_at_5 REAL,
  prompt_tokens        INTEGER,
  out_tokens           INTEGER,
  latency_ms           INTEGER,
  created_at           REAL NOT NULL,
  PRIMARY KEY (run_id, system, question_id)
);

-- Per-system question detail for the authored security-led memory evaluation.
-- Kept separate so the existing eval_runs contract remains unchanged.
CREATE TABLE IF NOT EXISTS eval_question_results (
  run_id              TEXT NOT NULL,
  system              TEXT NOT NULL,
  question_id         TEXT NOT NULL,
  axis                TEXT NOT NULL,
  answer_text         TEXT NOT NULL,
  abstained           INTEGER NOT NULL CHECK (abstained IN (0,1)),
  cited_ids_json      TEXT NOT NULL,
  retrieved_ids_json  TEXT NOT NULL,
  expected_answer     TEXT,
  expected_label      TEXT NOT NULL,
  correct             INTEGER NOT NULL CHECK (correct IN (0,1)),
  gate_outcome_json   TEXT,
  prompt_tokens       INTEGER NOT NULL DEFAULT 0 CHECK (prompt_tokens >= 0),
  completion_tokens   INTEGER NOT NULL DEFAULT 0 CHECK (completion_tokens >= 0),
  latency_ms          INTEGER NOT NULL DEFAULT 0 CHECK (latency_ms >= 0),
  cost_credits        REAL NOT NULL DEFAULT 0.0 CHECK (cost_credits >= 0),
  cache_request_hash  TEXT,
  PRIMARY KEY (run_id, system, question_id),
  FOREIGN KEY (run_id, system, question_id)
    REFERENCES eval_runs(run_id, system, question_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_eval_question_axis
  ON eval_question_results(run_id, axis, system);

-- 2b. Lossless slot observations used to rebuild normalized intervals --------
-- `facts` is the interval projection. Coalescing adjacent equal values there
-- must not discard a source observation: a later backdated fact can split that
-- logical interval again. This additive ledger preserves each extracted slot
-- observation and its source provenance for deterministic projection/replay.
CREATE TABLE IF NOT EXISTS fact_observations (
  observation_id    TEXT PRIMARY KEY,       -- stable extracted fact ID
  history_id        TEXT NOT NULL,
  turn_id           TEXT NOT NULL REFERENCES turns(turn_id),
  fact_seq          INTEGER NOT NULL,
  entity            TEXT NOT NULL,
  attribute         TEXT NOT NULL,
  value             TEXT NOT NULL,
  value_norm        TEXT NOT NULL,
  cardinality       TEXT NOT NULL CHECK (cardinality IN ('single','multi')),
  kind              TEXT,
  key_text          TEXT NOT NULL,
  valid_at          REAL NOT NULL,           -- timestamp from the prompt formula
  effective_at      REAL NOT NULL,           -- deterministic tie-resolved timestamp
  canonical_fact_id TEXT REFERENCES facts(fact_id) ON DELETE SET NULL,
  last_confirmed_at REAL NOT NULL,
  last_confirmed_turn_id TEXT NOT NULL REFERENCES turns(turn_id),
  confirmation_count INTEGER NOT NULL DEFAULT 1 CHECK (confirmation_count > 0),
  recorded_at       REAL NOT NULL,
  trust             REAL NOT NULL DEFAULT 0.6,
  quarantined       INTEGER NOT NULL DEFAULT 0 CHECK (quarantined IN (0,1)),
  quarantine_reason TEXT,
  confidence        REAL,
  UNIQUE(turn_id, fact_seq)
);
CREATE INDEX IF NOT EXISTS idx_fact_observations_slot
  ON fact_observations(history_id, entity, attribute, valid_at, fact_seq);
CREATE INDEX IF NOT EXISTS idx_fact_observations_canonical
  ON fact_observations(canonical_fact_id);

-- =============================================================================
-- Interval-Stitching Supersession Rule (implemented in temporal.py)
-- =============================================================================
-- Value Normalization helper normalize_value(val):
--   1. Lowercase, strip leading/trailing whitespace and surrounding punctuation.
--   2. Collapse internal whitespace to single spaces.
--   3. Convert standalone English number words ('zero'..'twenty', 'thirty', etc.)
--      to digits so "three bedrooms" and "3 bedrooms" normalize identically to "3 bedrooms".
--
-- Every slot fact F has a strictly unique valid_from within its slot:
--   * Normal:    valid_from = session_unix_ts + (turn_idx * 1.0) + (fact_seq * 1e-3)
--   * Backdated: valid_from = as_of_unix_ts   + (turn_idx * 1e-3) + (fact_seq * 1e-6)
--
-- When inserting a new 'slot' fact F with cardinality='single' and quarantined=0
-- for (history_id, entity, attribute):
--
-- 1. Find immediate predecessor G_prev:
--      SELECT * FROM facts
--      WHERE history_id = F.history_id AND key_type = 'slot'
--        AND entity = F.entity AND attribute = F.attribute
--        AND quarantined = 0 AND valid_from <= F.valid_from
--      ORDER BY valid_from DESC LIMIT 1;
--
--    * If G_prev exists AND G_prev.value_norm == F.value_norm:
--        -- Restated/confirming value: do NOT close G_prev, and do NOT lose F's provenance!
--        UPDATE facts
--        SET last_confirmed_at      = MAX(COALESCE(last_confirmed_at, valid_from), F.valid_from),
--            last_confirmed_turn_id = CASE
--                                       WHEN F.valid_from >= COALESCE(last_confirmed_at, valid_from)
--                                       THEN F.turn_id
--                                       ELSE last_confirmed_turn_id
--                                     END,
--            confirmation_count     = confirmation_count + 1
--        WHERE fact_id = G_prev.fact_id;
--        Return G_prev.fact_id.
--
-- 2. Find immediate successor G_next (handles out-of-order / backdated arrival):
--      SELECT * FROM facts
--      WHERE history_id = F.history_id AND key_type = 'slot'
--        AND entity = F.entity AND attribute = F.attribute
--        AND quarantined = 0 AND valid_from > F.valid_from
--      ORDER BY valid_from ASC LIMIT 1;
--
-- 3. Set F's closing interval and confirmation metadata before INSERT:
--    * Set F.last_confirmed_at      = F.valid_from
--    * Set F.last_confirmed_turn_id = F.turn_id
--    * Set F.confirmation_count     = 1
--    * If G_next exists:
--        F.valid_until   = G_next.valid_from
--        F.superseded_by = G_next.fact_id
--        F.expired_at    = <now>
--    * Else:
--        F.valid_until   = NULL
--        F.superseded_by = NULL
--        F.expired_at    = NULL
--    INSERT F into facts.
--
-- 4. Stitch predecessor G_prev (if G_prev exists AND G_prev.value_norm != F.value_norm):
--    * If G_prev.valid_until IS NULL OR G_prev.valid_until > F.valid_from:
--        UPDATE facts
--        SET valid_until   = F.valid_from,
--            superseded_by = F.fact_id,
--            expired_at    = <now>
--        WHERE fact_id = G_prev.fact_id;
--        -- Also propagate valid_until = F.valid_from to any child trigger rows
--        -- where parent_fact_id = G_prev.fact_id.
--
-- Note: During retrieval, if a slot fact has last_confirmed_turn_id != turn_id,
-- the reader context includes both the originating turn and the latest confirming turn
-- (or annotates "[confirmed again on {last_confirmed_date}]") so restatement recency is preserved.
--
-- AS-OF retrieval filter for time t*:
--   valid_from <= t* AND (valid_until IS NULL OR valid_until > t*) AND quarantined = 0
