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
