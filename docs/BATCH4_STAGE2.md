# Batch 4, Stage 2 — Candidate-Only Reranking

## Interface

`engramOS.reranking.rerank_retrieval_result(query, result, reranker, *, timeout=5.0)`
adapts a Phase 4A `RetrievalResult` to an injected `CandidateReranker`. It returns
a `RerankingOutcome` containing the resulting retrieval object, a status, a
content-free reason code, and the ordered typed candidate IDs.

Candidate identifiers are immutable `CandidateId(kind, value)` values. The
`kind` is exactly `turn` or `fact`; their display forms are `turn:<id>` and
`fact:<id>`. The reranker receives a tuple of immutable `RerankCandidate`
snapshots with candidate text and a read-only allowlist of scalar ranking
metadata. It does not receive mutable Phase 4A evidence objects, trust values,
or a database connection. `original_rank` is zero-based within the turn or fact
collection.

The provider returns a built-in list or tuple containing one
`CandidateScore(candidate_id, score)` per candidate.
Scores must be finite Python integers or floats; booleans and other numeric-like
types are rejected. The result must be an exact permutation of the submitted
typed IDs. Scores are sorted descending; ties use the candidate's original
rank, then its typed identifier for a stable secondary key.

## Failure and timeline behavior

Invalid timeout arguments raise before the provider is called. A provider
exception, timeout, malformed response, unknown/duplicate/missing candidate,
invalid identifier, or invalid score rejects the entire reranking result. The
exact original `RetrievalResult` object and candidate ordering are returned;
there is no partial acceptance. Duplicate IDs in the original candidate pool
skip the provider. Empty candidate pools also skip it.

The adapter enforces its wait deadline using daemon worker threads, with at
most four provider calls active across the process. A provider that does not
stop after the deadline can continue running in its worker and retains one of
those four slots until it exits. Calls made after all slots are occupied fail
closed immediately with `provider_capacity_exhausted`; they do not enqueue work
or create more threads. Timed-out workers cannot block Python interpreter
shutdown because they are daemon threads. Their late output is ignored.
Providers should still honor the supplied timeout to avoid holding capacity.

History results are returned unchanged without invoking the provider. This
preserves every eligible fact in the complete chronological timeline. This
module does not select Reader context, present a history answer, or perform
eligibility checks. Eligibility remains Phase 4A's responsibility; the
reranker cannot add candidates or alter original candidate objects, provenance,
trust, temporal fields, or database state.

Fallback logging contains only a reason code and, for provider exceptions, the
exception class. It does not log queries, prompts, evidence text, candidate IDs,
or exception messages.
