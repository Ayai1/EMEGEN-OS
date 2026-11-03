# Batch 4 Stage 4 — Query pipeline integration

**Status:** Implemented. The integration is provider-independent; all external
classifier, reranker, embedding, and Reader behavior remains injected.

## Entry point and flow

`engramOS.pipeline.answer_query()` takes an explicit `question_date`, classifies
the query, and stops on clarification, unsupported routes, malformed
classification output, or classifier failure. Non-ready routes do not call
retrieval, reranking, or the Reader. Ready routes pass the validated view and
exact temporal boundary to `retrieve_memory()`.

For current and as-of views, the optional reranker receives only the eligible
retrieval candidate set. Its exact validated combined ID order is preserved on
success; fallback keeps the original turn-then-fact collection order. History
facts retain chronological order and query-ranked turns are supplemental.

Turn event-time limits are applied to FTS SQL and dense-index allowed IDs before
candidate limits and RRF. They are checked again during final revalidation.
As-of turn limits preserve inclusive/exclusive boundaries. History turns use
the history range but remain a query-ranked top-k collection.

After reranking, `rehydrate_reader_candidates()` opens a fresh read snapshot,
restricts checks to the original typed IDs and requested history, rechecks
shared turn/fact/provenance eligibility, and reconstructs text and provenance
from current rows. It does not add candidates found only in that snapshot. The
transaction closes before the Reader runs. Any failure in this phase abstains
without using the initial retrieval objects. The final snapshot is the Reader
authorization point.

## History-range policy

Facts are selected as changes whose `valid_from` falls within the explicitly
inclusive/exclusive endpoints. One active pre-range fact per slot can be added
as a separately labeled baseline to explain the opening state. History facts
are not truncated by semantic top-k. Supplemental standalone turns are
query-ranked and explicitly marked incomplete; they are not presented as a
complete raw conversation timeline.

## Reader context and as-of provenance

The additive `ReaderContext` is immutable and database-free. It contains the
query, validated classification, history ID, typed evidence IDs, fresh text and
metadata, ordering/completeness labels, and the final authorization timestamp.
Reader citation validation remains exact membership against the supplied
context IDs.

For as-of evidence, turns must meet the requested event-time boundary. A
backdated fact may remain valid at that time, while source or confirmation text
after the boundary is replaced with empty text. The fact carries separate
valid-time and recording-time metadata and a `retrospective` marker. The Reader
prompt extension explains that later provenance text was withheld and that
retrospective facts do not establish what was known contemporaneously.

The legacy Reader path and its prompt hash remain unchanged. The context path
uses the following versioned prompt extension; the response schema remains
unchanged:

- Prompt: `engram.reader-context-prompt.v1`
- Prompt SHA-256: `8a84b5926dfbb9478e695a42283f380d98f87231ec06f7589a0e98cb79c9cace`
- Response schema: `engram.reader-response.v1`
- Schema SHA-256: `97fb89ae46443e6bc394bb23060bee3107833b5a6655fc1a7b746957281ad6be`

No generated answer is persisted as memory. Empty final context skips the
provider and returns the fixed `I don't know.` abstention.

## Verification and limits

`tests/test_pipeline.py` covers unsupported and failed classification
short-circuiting, inclusive and exclusive as-of turn cutoffs, event-time
filtering before top-k, inclusive and exclusive history `valid_from`
endpoints, labeled baselines, later source/confirmation-text redaction, a quarantine
committed between retrieval and final revalidation, and preservation of the
validated combined reranker order. The full-suite count and command are
recorded in `docs/TESTING.md`.

The authorization snapshot cannot be revoked by a quarantine committed after
it begins. Citation membership does not prove semantic entailment. Real model
quality, a live provider integration, and complete raw-turn history retrieval
remain outside this stage.
