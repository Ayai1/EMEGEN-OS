# EngramOS — Batch 4, Stage 3: Evidence-Grounded Reader

**Status:** Implemented and verified locally; no production Reader provider is connected.
**Repository:** `C:\Projects\EMEGEN-OS`
**Branch / HEAD:** `main` / `cedd9602c72208b2a35ada65ee5961c258007bb1`

## Implementation

Stage 3 adds `src/engramOS/reader.py`. Its public entry point is:

```python
answer_from_evidence(
    query: str,
    candidates: Sequence[RetrievedTurn | RetrievedFact],
    classification: QueryClassification,
    provider: ReaderProvider | None,
    *,
    timeout: float = 5.0,
) -> ReaderOutcome
```

`ReaderProvider.generate(prompt: ReaderPrompt, *, timeout: float) -> str` is the
injection point. `ReaderPrompt` is a frozen value containing the exact system
prompt, a JSON user payload, the response schema, and prompt/schema versions and
hashes. A provider receives no database handle or retrieval object. Only the
candidate sequence explicitly supplied by the caller is snapshotted.

Input accepts built-in tuples or lists of Phase 4A `RetrievedTurn` and
`RetrievedFact` values. The Reader makes frozen `ReaderEvidence` snapshots with
Stage 2 `CandidateId` values (`turn:<id>` or `fact:<id>`), allowlisted scalar
metadata, and nested `ReaderProvenance` records for fact source,
confirmation, and supporting turns. Trust and quarantine values are omitted.
Fact validity timestamps and source-turn event timestamps remain separate.
Trigger candidates retain their trigger ID and cue; supplied parent fact text
is used as the claim text, with the parent relation included in metadata.
Nested provenance does not create independently citable candidates.

The caller's validated `QueryClassification` is copied into the prompt scope
without rerouting or date interpretation. A non-ready classification abstains.
The Reader performs no retrieval, eligibility decision, final revalidation,
temporal filtering, or database write. It does not persist generated answers.

## Response and citation contract

Prompt version: `engram.reader-prompt.v1`
Response schema version: `engram.reader-response.v1`

The provider returns raw JSON with exactly:

```json
{
  "schema_version": "engram.reader-response.v1",
  "answer": "string",
  "citations": ["turn:<id>"],
  "abstained": false
}
```

The runtime also supplies the strict JSON Schema in `ReaderPrompt`. The parser
rejects malformed JSON, duplicate keys, non-standard constants, missing or
extra fields, incorrect types, empty answers, and contradictory states. A
non-abstaining answer requires at least one unique citation. An abstention
requires exactly `answer: "I don't know."`, an empty citation list, and
`abstained: true`.

Citation IDs are parsed as typed IDs and checked against the exact candidate-ID
set for that invocation. Correct formatting alone is insufficient. Duplicate,
fabricated, stale, wrong-kind, nested-only, or mixed valid/invalid citations
invalidate the whole response; no partial citation repair occurs. Citation
membership proves only that the ID was supplied. It does **not** prove that the
cited record semantically entails the answer.

## Abstention and failures

All failure outcomes use the exact user-facing text `I don't know.` with no
citations. Empty evidence returns `no_evidence` without a provider call. A
candidate with unusable/blank text or malformed provenance returns
`invalid_evidence` without a provider call. Invalid timeout configuration,
non-ready classification, missing provider, provider exception/timeout,
capacity exhaustion, invalid response, and invalid citation also fail closed.

The Reader uses at most four daemon provider workers. Timed-out workers retain
their slot until they exit; calls at capacity abstain immediately without
queueing. A non-cooperative provider can retain a slot indefinitely, but the
daemon worker cannot prevent interpreter shutdown. Its late output is ignored.
Logs contain only provider exception classes, never query, evidence, prompt,
output, or exception messages.

## Prompt version and hashes

The original Section D Reader prompt in `prompts.md` is preserved. The new
versioned extension is appended after the historical sections, and the Stage 1
query-classification text and hash semantics are unchanged.

- Prompt hash: `bcc577d351a597ac3c86a51e0777102cc5d8d25a17f69ff24e35333a9cd01726`
- Schema hash: `97fb89ae46443e6bc394bb23060bee3107833b5a6655fc1a7b746957281ad6be`
- Prompt hashing basis: SHA-256 of the exact UTF-8 system-template bytes, CRLF
  normalized to LF, with no trailing newline. Dynamic query/scope/evidence
  payload is excluded.
- Schema hashing basis: SHA-256 of the canonical compact schema JSON, sorted
  keys, UTF-8, with CRLF normalized to LF.

Both hashes and versions are included in the immutable provider request. Stage
3 does not create evaluation-run records or alter the schema.

## Files created or modified for Stage 3

- Created `src/engramOS/reader.py`.
- Created `tests/test_reader.py`.
- Created `BATCH4_STAGE3.md` (this report).
- Appended the versioned Reader prompt and schema contract to `prompts.md`.

No Stage 1 classifier/temporal or Stage 2 reranking source/test files were
changed. `prompts.md` was extended only after the historical prompt sections;
the existing Stage 1 classifier prompt text and hash semantics are unchanged.
All other Stage 1 and Stage 2 working-tree changes were preserved.

## Verification

Pre-implementation baseline:

```text
python -m unittest discover -s tests -v
Ran 240 tests in 24.070s
OK
Exit code: 0
```

Focused Stage 3 tests:

```text
python -m unittest tests.test_reader -v
Ran 40 tests in 0.161s
OK
Exit code: 0
```

Full regression suite after implementation:

```text
python -m unittest discover -s tests -v
Ran 280 tests in 21.079s
OK
Exit code: 0
```

Tests use deterministic fake Readers and local Phase 4A dataclasses. No live
provider calls, paid APIs, network calls, model downloads, or dependency
changes were made. No schema or index changes were made.

## Limitations and boundary

The Reader prompt instructs the provider to use only evidence, ignore
instructions embedded in evidence, qualify incomplete support, and abstain
when unsupported. Tests verify prompt construction and deterministic response
validation, not the behavior of a real language model. Citation membership
does not establish factual entailment or prevent a model from generating
unsupported prose accompanied by a supplied citation.

Stage 4 now performs final eligibility revalidation and standalone-turn
as-of filtering before calling this Reader. The Reader remains a consumer of
the immutable, freshly rehydrated evidence set supplied by the pipeline; it
does not make eligibility decisions or retrieve additional evidence. Stage 4
implementation details and current limits are documented in
`docs/BATCH4_STAGE4.md`.
