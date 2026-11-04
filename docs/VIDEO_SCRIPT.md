# Demo video script (3:00 maximum)

| Time | Segment | Recording notes |
| --- | --- | --- |
| 0:00–0:30 | Problem | Show the Problem section. Explain stale facts and conflicting document claims in neutral terms. |
| 0:30–1:00 | Architecture | Show the diagram and name the deterministic gate, Muse Glimmer stages, temporal store, retrieval, and cited Reader answer. Mention optional non-Meta BGE-M3 embeddings and `--embedding none`. |
| 1:00–2:15 | Live demo | Follow the scenario in order: set city A, update to B, set editor X, change to Y then back to X, ask current/prior/history questions, show the one-signal external document, then show the fresh two-signal poisoning attempt quarantined. Keep the LIVE badge visible. |
| 2:15–2:45 | Results and limits | Show the Results section and artifact labels. Mention fixture-based evaluation does not measure extraction quality, the gate was tuned with authored attacks, caller-supplied source labels, and small samples. |
| 2:45–3:00 | Close | Show Scope, safety, licenses, and the local run commands. |

Record the live segment only after the local model server is available and the
scenario has been checked. If using a saved segment, keep the visible
`RECORDED RUN, not live` badge on screen throughout; do not cut to a frame that
could make recorded output look live.

## Text-only live run sheet (3:00 maximum)

Keep the browser in live mode. The raw capture can run longer if model requests
take extra time; trim only idle waiting in the edit so the finished video stays
within three minutes and each click is still followed by its actual result.

Before recording:

1. Keep the local Muse server and demo server running. Open
   `http://127.0.0.1:8765/#live-demo` and confirm the green `LIVE` badge,
   `Embedding: none`, and Muse Glimmer model labels.
2. Enable `Capture cursor` or `Show mouse pointer` in the screen recorder.
   Move to each control, pause briefly, click once, and move away from result
   text before it appears. Keep the cursor visible without covering the badge
   or gate result.
3. Click `Reset session`; this clears the previous demo test state.
4. Use short overlay captions rather than narration. Keep each caption to one
   or two lines and leave the UI output readable.

| Target time | Mouse and screen action | Overlay caption |
| --- | --- | --- |
| 0:00–0:10 | Start on the page title. Move the pointer to the EngramOS heading. | `EngramOS — time-aware memory with provenance` |
| 0:10–0:25 | Scroll to **Problem** and pause on its text. | `Old facts can outlive their truth.` |
| 0:25–0:43 | Scroll to **Architecture**. Point along the diagram from Gate to Reader. | `Every turn is checked before extraction.`<br>`Facts keep source and time.` |
| 0:43–0:52 | Scroll to **Run or replay the scenario**. Point at `LIVE` and `Embedding: none`. | `Local run · Muse Glimmer · FTS5 retrieval` |
| 0:52–1:22 | Click **1. Set home city A**, then **Ingest turn**. Wait for the fact panel to show Brookhaven. | `Trusted memory stores: Brookhaven.` |
| 1:22–1:38 | Click **7. Fresh poisoning attempt**, then **Ingest turn**. Move the pointer to the Gate decision panel as the result arrives. | `An untrusted document proposes a replacement.` |
| 1:38–1:55 | Hold on the 2/3 result and the fired Bridge and Source signals. Show that no facts were extracted. | `Bridge + Source fired. Quarantined before extraction.` |
| 1:55–2:12 | Click **5a. Current home city**, then **Ask**. Point to the answer and citation after it appears. | `The quarantined turn did not replace eligible memory.` |
| 2:12–2:32 | Scroll to **Results**. Keep the artifact labels in view. | `Saved evaluation artifacts; small authored samples.` |
| 2:32–2:52 | Scroll to **Scope & safety**. | `Prototype scope: source labels are caller-supplied.` |
| 2:52–3:00 | Return to the architecture diagram or title for the end frame. | `Time-aware memory with a visible trust boundary.` |

For this short cut, use the already verified sequence: reset, preset 1, then
preset 7. The poisoning preset should show two fired signals and quarantine.
If adding the temporal update (`2. Update home city to B`) and earlier-city
question, capture that as a separate segment; reset and repeat presets 1 and 7
before the poisoning shot so its starting state matches the verified run.

The `demo/record_session.py` command records the scenario's API events to JSON;
it does not record screen video. Replay mode displays a `RECORDED RUN, not live`
badge, which must remain visible in any replay footage.
