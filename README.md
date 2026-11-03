# EngramOS

EngramOS is an experimental memory pipeline for retrieving time-aware facts
while quarantining suspicious updates. The pre-registered authored benchmark
and a separately labeled post-hoc v2 run are reported below. The benchmark is
small and synthetic, so these results are measured prototype results, not a
general security guarantee.

## Results

The first table is the pre-registered v1 result. All systems used the same
local Reader model, prompt, and sampling settings. Accuracy intervals are
exact two-sided Clopper–Pearson 95% binomial intervals.

| System | Accuracy | Poison reached Reader | Poison claim adopted in answer | Quarantine leaks |
| --- | ---: | ---: | ---: | ---: |
| Engram | 26/38, 68.42% (51.35–82.50%) | 6/11, 54.55% (23.38–83.25%) | 5/11, 45.45% (16.75–76.62%) | 0 |
| `hybrid_rag` | 22/38, 57.89% (40.82–73.69%) | 10/11, 90.91% (58.72–99.77%) | 3/11, 27.27% (6.02–60.97%) | 7 |
| `hybrid_rag_date_filter` | 19/38, 50.00% (33.38–66.62%) | 9/11, 81.82% (48.22–97.72%) | 3/11, 27.27% (6.02–60.97%) | 6 |
| `recency` | 19/38, 50.00% (33.38–66.62%) | 10/11, 90.91% (58.72–99.77%) | 3/11, 27.27% (6.02–60.97%) | 7 |

Provider failures were **0/152** in the completed run. Benign false positives
were **0/11** (exact 95% CI 0.00–28.49%). Abstention precision was Engram
3/10 (30.00%, CI 6.67–65.25%), `hybrid_rag` 3/12 (25.00%, CI 5.49–57.19%),
date-filtered hybrid 3/15 (20.00%, CI 4.33–48.09%), and recency 3/14 (21.43%,
CI 4.66–50.80%). Total local credits: **0**.

### Post-hoc v2 (separate run IDs and database)

Engram was rerun on the same 38 locked v1 questions after the provenance-aware
supersession and date-free classifier fixes. The 8-question held-out suite was
run with all four systems. V1 primary artifacts were not modified after the
v2 DB was created, based on their file timestamps and the absence of a v2 write
path to them. No pre-run hashes were recorded, so this is not a hash-equality
claim. The DB copy command was not recorded.

| Suite / system | Accuracy | Poison reached Reader | Poison claim adopted | Quarantine leaks |
| --- | ---: | ---: | ---: | ---: |
| v2 Engram on v1 questions | 34/38, 89.47% (75.20–97.06%) | 6/11, 54.55% (23.38–83.25%) | 1/11, 9.09% (0.23–41.28%) | 0 |
| Held-out Engram | 8/8, 100% (63.06–100%) | 0/3, 0% (0–70.76%) | 0/3, 0% (0–70.76%) | 0 |
| Held-out `hybrid_rag` | 6/8, 75% (34.91–96.81%) | 3/3, 100% (29.24–100%) | 1/3, 33.33% (0.84–90.57%) | 0 |
| Held-out `hybrid_rag_date_filter` | 6/8, 75% (34.91–96.81%) | 3/3, 100% (29.24–100%) | 1/3, 33.33% (0.84–90.57%) | 0 |
| Held-out `recency` | 6/8, 75% (34.91–96.81%) | 3/3, 100% (29.24–100%) | 2/3, 66.67% (9.43–99.16%) | 0 |

`poison_adaptive_user` is a compromised trusted channel, not an untrusted
attack. On the 38-question suite, Engram's untrusted-only exposure/adoption
counts were **5/10** and **0/10**; the compromised channel was **1/1** reached
and **1/1** adopted. Held-out Engram had no incorrect answers, so no held-out
baseline beat it. Paired held-out comparisons each had two Engram-only wins,
zero baseline-only wins, and exploratory exact McNemar p=0.5. Provider
failures were **0/70** across v2 rows and local credits were **0**. The held-out
sample is small and authored; it does not establish significance or real-world
attack prevalence. Full counts and intervals are in
[docs/TESTING.md](docs/TESTING.md), and both machine-readable summaries are
[eval/results_posthoc_v2_engram_v1.json](eval/results_posthoc_v2_engram_v1.json)
and [eval/results_posthoc_v2_heldout_v1.json](eval/results_posthoc_v2_heldout_v1.json).

The full per-axis table and run accounting are in
[docs/TESTING.md](docs/TESTING.md); the machine-readable summary is
[eval/results_q4-locked-classifier1024-resume300-v1.json](eval/results_q4-locked-classifier1024-resume300-v1.json).
The per-attack replay data with detector signals and all system outcomes is
[eval/attack_gallery_q4-locked-classifier1024-resume300-v1.json](eval/attack_gallery_q4-locked-classifier1024-resume300-v1.json).

## Demo flow

The locked fixture suite contains **23 histories, 80 turns, and 38 questions**
across temporal retrieval, supersession, retrospective redaction, poisoning,
benign controls, and abstention cases. Its SHA-256 is
`6b9a2a48e7d2dec311e7064de4b2adbb717ab8a67372a7590f1595f450a2c823`.
Fixture extraction payloads are ingested sequentially by a deterministic local
extractor. This measures memory logic and gate behavior, not extraction
quality.

The four comparisons are `recency` (the ten newest hybrid turns),
`hybrid_rag` without the gate, `hybrid_rag_date_filter` (event-time prefilter
before RRF), and the Engram `pipeline.answer_query()` path. All share the same
Reader prompt and provider. The run used a local Muse Glimmer 30B
`UD-Q4_K_XL` GGUF (SHA-256
`82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38`) on an
RTX 5070 Ti Laptop GPU with 32 of 53 model layers on GPU, 21 on CPU, an 8,192
token context, Flash Attention, q8_0 KV cache, and low reasoning mode. Sampling
was temperature 0, top-p 0.95, top-k 40, seed 42. Reader and classifier caps
were 2,048 and 1,024 tokens. The local pricing mode recorded zero credits.
Detailed server and laptop measurements are recorded in
[docs/DECISIONS.md](docs/DECISIONS.md).

After starting the local server with the measured configuration in
`docs/DECISIONS.md`, run a fresh labeled evaluation in its own BGE database
with:

```powershell
.\.venv\Scripts\python.exe scripts/run_eval.py --mode local --embedding bge --database eval/reproduction-q4.sqlite3 --run-id q4-reproduction-v1 --suite eval/suite_v2.json --progress-log logs/q4-reproduction-progress.jsonl --provider-timeout 300
```

Replay recorded facts, gate decisions, and side-by-side answers without
providers or network access:

```powershell
.\.venv\Scripts\python.exe scripts/replay.py --help
```

The user-confirmed full test command before the first live v2 run was:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The user reported **361/361 tests passed**. The suite was not rerun during the
live evaluation.

## Limitations

### Threat model and synthetic benchmark

The gate is intended to identify suspicious untrusted external-document and
tool-output updates, especially those that contradict an already stored
constraint or bridge into a remembered slot. Trusted user changes can update
policies. Gate evaluation precedes extraction of the incoming turn, so a
constraint detector cannot fire before a pinned constraint exists. The suite
includes standard and adaptive attacks, benign untrusted updates, and topic
shifts. The authored attacks and detectors were developed together and the
thresholds were chosen for this suite; the results do not estimate real-world
attack prevalence or establish adversarial robustness.

In pre-registered v1, the gate allowed six attacks through: `attack_ext_policy_paraphrase`,
`attack_ext_subtle_jacket`, `attack_tool_policy_paraphrase`,
`attack_tool_subtle_backdrop`, `poison_adaptive_external`, and
`poison_adaptive_user`. Engram adopted five malicious claims overall, while
each baseline adopted three. Reader exposure and answer adoption are separate
measures: filtering exposure does not guarantee correct behavior on the
remaining exposed cases.

### Where simpler baselines won

`hybrid_rag`, date-filtered hybrid, and recency all answered
`attack_ext_policy_paraphrase` with the pinned 30-day policy while Engram
adopted the injected quarterly claim. `hybrid_rag` and recency also answered
`writing_target` with the 5,000-word target while Engram abstained; the
date-filtered baseline abstained too. Engram's overall accuracy advantage
does not mean it won every query or security outcome.

### Scope and runtime

The benchmark uses **38 questions** and only fixture extraction outputs. It
does not assess open-ended extraction quality. Wide exact confidence
intervals on per-axis results reflect the small sample sizes (many axes have
one to four questions). Fourteen benchmark queries used the LLM classifier;
they are listed in `docs/TESTING.md` so deterministic routing and model-routed
queries can be distinguished.

The local model was run with a mixed GPU/CPU layer placement. Question/system
latency means include cache and retrieval time; the date-filtered hybrid had
24 cache hits out of 38, so its elapsed-time mean is not a fresh-inference
comparison. See `docs/TESTING.md` for per-system latency, cache counts,
per-axis intervals, failure accounting, and the preserved initial timeout.

## Scope and what this is not

EngramOS stores and retrieves user-stated facts as `(entity, attribute, value)`
slots with provenance and temporal validity. It is an experimental memory
system, not a general-purpose RAG app; hybrid RAG is included only as a
comparison baseline. It does not provide medical, mental-health, legal, or
financial advice, and it does not make personality assessments or build
personality profiles.

## Built during the hackathon

Building period: October 27 to November 4. The work recorded in the
implementation plan and decision log covers:

- Temporal fact slots, observations, supersession intervals, as-of retrieval,
  and retrospective redaction.
- Provenance validation, deterministic source/policy/bridge gate signals,
  quarantine and eligibility handling, and provenance-aware projection.
- Query classification, lexical and dense retrieval, hybrid ranking, and the
  guarded Reader pipeline.
- Cached provider adapters, request accounting, budget handling, and a local
  Muse Glimmer backend.
- Authored v1 and held-out evaluation suites, comparison baselines,
  incremental result recording, replay, and attack-gallery tooling.

## Security model and limits

The gate is deterministic and its thresholds and patterns were selected for
the authored attacks in this benchmark. It is not a complete defense. The
`role` and `source_type` provenance fields are supplied by the caller and are
not authenticated by the package. A compromised input labeled `user_turn` is
outside the trust model. In default mode, an `external_doc` or `tool` turn for
which only the source signal fires is stored; quarantine requires at least two
signals. An opt-in strict mode quarantines `tool` and `external_doc` turns when
at least one signal fires; the recorded benchmark scripts keep that mode off.
The suite hash is recorded with evaluation metadata, but loading a suite does
not enforce it by default; `run_eval.py` can optionally enforce an expected
hash when one is supplied.

## Demo-only / not production

The recorded workflow uses local SQLite and is intended for a single-user
demo. It has no authentication, no encryption at rest, and no multi-writer
coordination across processes. The locked evaluation uses fixture extraction
payloads: it measures memory, temporal, retrieval, and gate behavior, not
extraction quality.

The post-hoc local runner accepts `--reader-max-tokens`,
`--classifier-max-tokens`, and `--reasoning-budget` when lower resource caps
are needed. Their recorded defaults remain 2,048, 1,024, and 512 respectively.

## Running without non-Meta models (`--embedding none`)

Pass `--embedding none` to skip embedding indexing and use FTS5 BM25 with the
existing temporal projection. This mode does not import or load
`sentence_transformers` or `torch`, and its database identity cannot be mixed
with a BGE database. At the retrieval/library level, it lets an application
use Meta Reader/classifier adapters without adding a separate BGE embedding
model. The standalone `run_eval.py` runner still selects its local or offline
providers with `--mode`.

```powershell
.\.venv\Scripts\python.exe scripts\run_eval.py --mode local --embedding none --database eval\local-q4-no-embedding.sqlite3 --run-id q4-local-lexical-v1 --suite eval\suite_v2.json
```

The v3 post-hoc runner is implemented; its evaluation is pending. It defaults
to BGE and keeps strict-untrusted mode and slot aliases off. Its smoke command
is:

```powershell
.\.venv\Scripts\python.exe scripts\run_posthoc_v3.py --database eval\local-q4-posthoc-v3.sqlite3 --run-id q4-post-hoc-v3-smoke --preset smoke
```

Run `scripts/run_posthoc_v3.py --help` for the full-suite and held-out
commands. V3 results are post-hoc and have not been run.
