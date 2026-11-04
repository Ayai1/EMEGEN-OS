"""Offline repository and submission preflight; exits nonzero on any FAIL."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo.build_claims import build_claims  # noqa: E402

_MANIFEST_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|", re.I)
_SECRET_PATTERNS = (
    ("API-style key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("bearer token", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}", re.I)),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("private-key header", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("non-empty model API key assignment", re.compile(
        r"(?im)^[ \t]*(?:MODEL_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|GEMINI_API_KEY)[ \t]*=[ \t]*(?!#)(?=\S)[^\r\n]*$"
    )),
)


class Checks:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def emit(self, status: str, name: str, detail: str = "") -> None:
        print(f"{status:<4} {name}" + (f" — {detail}" if detail else ""))
        if status == "FAIL":
            self.failures += 1
        elif status == "WARN":
            self.warnings += 1


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False,
        encoding="utf-8", errors="replace",
    )


def _tracked_paths(checks: Checks) -> list[str]:
    result = _git("ls-files", "-z")
    if result.returncode != 0:
        checks.emit("FAIL", "git tracked-file listing", result.stderr.strip())
        return []
    return [part for part in result.stdout.split("\0") if part]


def check_git(checks: Checks, tracked: list[str]) -> None:
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status.returncode != 0:
        checks.emit("FAIL", "git working tree", status.stderr.strip())
    elif status.stdout.strip():
        checks.emit("FAIL", "git working tree clean", status.stdout.strip())
    else:
        checks.emit("PASS", "git working tree clean")
    untracked = _git("ls-files", "--others", "--exclude-standard", "-z")
    source_untracked = [
        path for path in untracked.stdout.split("\0") if path
        and PurePosixPath(path.replace("\\", "/")).parts
        and PurePosixPath(path.replace("\\", "/")).parts[0] in {"src", "scripts", "tests", "demo"}
    ]
    if source_untracked:
        checks.emit("FAIL", "untracked source files", ", ".join(source_untracked))
    else:
        checks.emit("PASS", "no untracked files under src/, scripts/, tests/, or demo/")


def check_tracked_files(checks: Checks, tracked: list[str]) -> None:
    forbidden: list[str] = []
    large: list[str] = []
    for raw in tracked:
        path = raw.replace("\\", "/")
        name = PurePosixPath(path).name.casefold()
        if (name.endswith((".sqlite3", ".sqlite3-wal", ".sqlite3-shm", ".gguf"))
                or name == ".env" or path.startswith("logs/")
                or path.startswith("data/longmemeval/")):
            forbidden.append(raw)
        file_path = ROOT / Path(raw)
        try:
            if file_path.is_file() and file_path.stat().st_size > 20 * 1024 * 1024:
                large.append(f"{raw} ({file_path.stat().st_size / (1024 * 1024):.1f} MB)")
        except OSError:
            pass
    if forbidden:
        checks.emit("FAIL", "tracked forbidden artifacts", ", ".join(forbidden))
    else:
        checks.emit("PASS", "tracked artifact scan (sqlite, gguf, .env, logs, LongMemEval)")
    checks.emit("WARN" if large else "PASS", "tracked files over 20 MB", ", ".join(large) if large else "none")


def check_secrets(checks: Checks, tracked: list[str]) -> None:
    hits: list[str] = []
    for raw in tracked:
        path = ROOT / Path(raw)
        try:
            if not path.is_file() or path.stat().st_size > 20 * 1024 * 1024:
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        for label, pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                hits.append(f"{raw}: {label}")
    if hits:
        checks.emit("FAIL", "secret scan", "; ".join(hits))
    else:
        checks.emit("PASS", "secret scan on tracked files")


def check_readme(checks: Checks) -> None:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    headings = "\n".join(line.casefold() for line in text.splitlines() if line.lstrip().startswith("#"))
    required = (
        "scope", "built during the hackathon", "security model and limits",
        "demo-only", "running without non-meta models", "run instructions for the demo",
    )
    missing = [heading for heading in required if heading not in headings]
    checks.emit("FAIL" if missing else "PASS", "README required headings", ", ".join(missing) if missing else "all present")
    forbidden_phrase = bool(re.search(r"(?i)RAG application", text))
    checks.emit("FAIL" if forbidden_phrase else "PASS", "README wording", "contains forbidden phrase" if forbidden_phrase else "no forbidden application label")
    advice_lines: list[str] = []
    current_paragraph: list[tuple[int, str]] = []

    def check_advice_paragraph(paragraph_lines: list[tuple[int, str]]) -> None:
        if not paragraph_lines:
            return
        joined = " ".join(line for _, line in paragraph_lines)
        if not re.search(r"\badvice\b", joined, re.I):
            return
        disclaimer = re.search(
            r"\bdoes not provide\b.{0,180}\b(?:medical|health|mental-health|legal|financial)\b.{0,80}\badvice\b",
            joined,
            re.I,
        )
        if disclaimer is None:
            advice_lines.append(f"line {paragraph_lines[0][0]}: {joined.strip()}")

    for idx, line in enumerate(text.splitlines(), 1):
        if line.strip():
            current_paragraph.append((idx, line))
        else:
            check_advice_paragraph(current_paragraph)
            current_paragraph = []
    check_advice_paragraph(current_paragraph)
    checks.emit("WARN" if advice_lines else "PASS", "README advice wording", "; ".join(advice_lines) if advice_lines else "advice appears only in the negated scope statement")
    positive_health = bool(re.search(r"(?i)(?:provides?|offers?|gives?)\s+(?:personalized\s+)?(?:medical|health|mental-health)\s+advice", text))
    checks.emit("FAIL" if positive_health else "PASS", "README health-advice claims", "positive advice wording found" if positive_health else "none")


def check_licenses_env(checks: Checks) -> None:
    licenses = ROOT / "LICENSES.md"
    if not licenses.is_file():
        checks.emit("FAIL", "LICENSES.md present")
    else:
        rows = [f"line {idx}: {line.strip()}" for idx, line in enumerate(licenses.read_text(encoding="utf-8").splitlines(), 1) if "TODO: verify license" in line]
        checks.emit("WARN" if rows else "PASS", "license TODOs", "; ".join(rows) if rows else "none")
    env_example = ROOT / ".env.example"
    if not env_example.is_file():
        checks.emit("FAIL", ".env.example present")
    else:
        nonempty = [f"line {idx}" for idx, line in enumerate(env_example.read_text(encoding="utf-8").splitlines(), 1) if "=" in line and line.split("=", 1)[1].strip()]
        checks.emit("FAIL" if nonempty else "PASS", ".env.example values empty", ", ".join(nonempty) if nonempty else "all assignments empty")


def check_artifact_hashes(checks: Checks, tracked: list[str]) -> None:
    manifest_path = ROOT / "docs" / "ARTIFACT_HASHES.md"
    if not manifest_path.is_file():
        checks.emit("FAIL", "artifact hash manifest present")
        return
    entries: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        match = _MANIFEST_ROW.match(line)
        if match:
            entries[match.group(1)] = match.group(2).lower()
    tracked_json = {
        path.replace("\\", "/") for path in tracked
        if path.replace("\\", "/").startswith("eval/")
        and path.casefold().endswith(".json")
    }
    missing = sorted(tracked_json - set(entries))
    mismatches: list[str] = []
    for relative, expected in entries.items():
        normalized = relative.replace("\\", "/")
        if normalized not in tracked:
            continue  # Local SQLite copies are documented but deliberately not tracked.
        path = ROOT / Path(relative)
        if not path.is_file():
            mismatches.append(f"{relative} missing")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        raw = path.read_bytes()
        lf_bytes = raw.replace(b"\r\n", b"\n")
        crlf_bytes = lf_bytes.replace(b"\n", b"\r\n")
        equivalent_hashes = {
            actual,
            hashlib.sha256(lf_bytes).hexdigest(),
            hashlib.sha256(crlf_bytes).hexdigest(),
        }
        if expected not in equivalent_hashes:
            mismatches.append(f"{relative} expected {expected}, found {actual}")
    problems = missing + mismatches
    checks.emit("FAIL" if problems else "PASS", "eval artifact hashes", "; ".join(problems) if problems else f"{len(entries)} manifest rows checked; all tracked eval JSON files covered")


def check_demo(checks: Checks) -> None:
    page = ROOT / "demo" / "index.html"
    claims_file = ROOT / "demo" / "claims.json"
    checks.emit("FAIL" if not page.is_file() else "PASS", "demo page present")
    if not claims_file.is_file():
        checks.emit("FAIL", "demo claims present")
    else:
        try:
            actual = json.loads(claims_file.read_text(encoding="utf-8"))
            expected = build_claims()
            consistent = actual == expected
        except Exception as exc:
            consistent = False
            detail = type(exc).__name__
        else:
            detail = "matches current eval result artifacts"
        checks.emit("FAIL" if not consistent else "PASS", "demo claims consistent", detail if consistent else detail)
    recording = ROOT / "demo" / "recordings" / "session.json"
    if recording.is_file():
        checks.emit("PASS", "recorded demo session present")
    else:
        checks.emit("WARN", "recorded demo session absent", "run demo/record_session.py")


def check_tests(checks: Checks) -> None:
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False, encoding="utf-8", errors="replace")
    output = completed.stdout + "\n" + completed.stderr
    match = re.search(r"Ran\s+(\d+)\s+tests?", output)
    count = match.group(1) if match else "unknown"
    failures = len(re.findall(r"^FAIL:\s", output, re.M))
    errors = len(re.findall(r"^ERROR:\s", output, re.M))
    if completed.returncode == 0:
        checks.emit("PASS", "unittest suite", f"{count} tests passed")
    else:
        checks.emit("FAIL", "unittest suite", f"{count} tests; failures={failures}, errors={errors}; exit={completed.returncode}")
        tail = output[-4000:].strip()
        if tail:
            print(tail)


def main() -> int:
    checks = Checks()
    tracked = _tracked_paths(checks)
    check_git(checks, tracked)
    check_tracked_files(checks, tracked)
    check_secrets(checks, tracked)
    check_readme(checks)
    check_licenses_env(checks)
    check_artifact_hashes(checks, tracked)
    check_demo(checks)
    check_tests(checks)
    print(f"\nPreflight: {checks.failures} FAIL, {checks.warnings} WARN")
    return 1 if checks.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
