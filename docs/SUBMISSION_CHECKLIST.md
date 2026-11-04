# Submission checklist

## Automated checks

- [x] Run `scripts/preflight.py` from a clean clone and resolve every FAIL.
- [x] Verify the clean clone setup and tests with the exact commands in README:
  `py -3.12 -m venv .venv`, `.\.venv\Scripts\python.exe -m pip install -e ".[dev]"`, then
  `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`.
- [x] Confirm the latest commit has no `.sqlite3`, log, `.gguf`, `.env`, or
  `data/` files staged or tracked.

## Manual submission checks

- [ ] The repository is public and opens in a logged-out browser.
- [ ] The fresh clone passes setup and the complete test suite.
- [ ] The video is 3:00 or less, public, in English, and shows only work built
  during the hackathon.
- [ ] Add the video link to the submission form and README.
- [ ] List every contributor.
- [ ] Select the Indie Developer or Startups theme/track and use the same
  wording in the README.
- [ ] Record the organizer answer about BGE-M3 (save the link or screenshot
  filename) and confirm README explains the Meta-only `--embedding none` mode.
- [x] Confirm no keys were committed (clean-clone preflight secret scan).
- [ ] Submit at least two hours before the deadline: Wednesday, November 4,
  9:00 PM ET (7:30 AM IST, November 5).

The page displays existing saved post-hoc v3 artifacts. The demo's own v3
evaluation has not been run.
