"""Build the page's read-only result claims from the checked-in artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _figure(name: str, successes: int, trials: int, source: str, label: str, rate: float | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "successes": successes,
        "trials": trials,
        "value": f"{successes}/{trials}" + (f" ({rate * 100:.2f}%)" if rate is not None else ""),
        "source": source,
        "label": label,
    }


def build_claims() -> dict[str, Any]:
    v1_source = "eval/results_q4-locked-classifier1024-resume300-v1.json"
    v2_source = "eval/results_posthoc_v2_engram_v1.json"
    heldout_source = "eval/results_posthoc_v2_heldout_v1.json"
    v1 = _read(v1_source)
    v2 = _read(v2_source)
    heldout = _read(heldout_source)
    v3_path = ROOT / "eval" / "results_posthoc_v3_engram-resume-v1.json"
    figures: list[dict[str, Any]] = []

    def add_accuracy(data: dict[str, Any], source: str, label: str, system: str, name: str) -> None:
        metric = data["overall_accuracy"][system]
        figures.append(_figure(name, int(metric["successes"]), int(metric["trials"]), source, label, float(metric["rate"])))

    def add_adoption(data: dict[str, Any], source: str, label: str, system: str, name: str) -> None:
        metric = data["attack_claim_adoption_rate"][system]
        figures.append(_figure(name, int(metric["successes"]), int(metric["trials"]), source, label, float(metric["rate"])))

    add_accuracy(v1, v1_source, "pre-registered v1", "engram", "Engram accuracy")
    add_accuracy(v1, v1_source, "pre-registered v1", "hybrid_rag", "hybrid_rag baseline accuracy")
    add_adoption(v1, v1_source, "pre-registered v1", "engram", "Engram attack claim adoption")
    add_accuracy(v2, v2_source, "post-hoc v2", "engram", "Engram accuracy")
    add_adoption(v2, v2_source, "post-hoc v2", "engram", "Engram attack claim adoption")
    add_accuracy(heldout, heldout_source, "post-hoc v2", "engram", "Held-out Engram accuracy")
    add_accuracy(heldout, heldout_source, "post-hoc v2", "hybrid_rag", "Held-out hybrid_rag baseline accuracy")
    if v3_path.is_file():
        v3_source = "eval/results_posthoc_v3_engram-resume-v1.json"
        v3 = json.loads(v3_path.read_text(encoding="utf-8"))
        add_accuracy(v3, v3_source, "post-hoc v3", "engram", "Engram accuracy")
        add_adoption(v3, v3_source, "post-hoc v3", "engram", "Engram attack claim adoption")

    figures.extend((
        _figure("Main suite questions", int(v1["suite"]["questions"]), int(v1["suite"]["questions"]), v1_source, "pre-registered v1"),
        _figure("Authored attack cases", int(v1["attack_claim_adoption_rate"]["engram"]["trials"]), int(v1["attack_claim_adoption_rate"]["engram"]["trials"]), v1_source, "pre-registered v1"),
        _figure("Held-out questions", int(heldout["suite"]["questions"]), int(heldout["suite"]["questions"]), heldout_source, "post-hoc v2"),
    ))

    architecture = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    limitations = [
        "Fixture-based evaluation does not measure open-ended extraction quality.",
        "The gate patterns and thresholds were tuned with the authored attacks.",
        "The caller supplies role and source_type; the system does not authenticate them.",
        "The authored samples are small and do not establish generalization or real-world attack prevalence.",
    ]
    required_terms = ("extraction quality", "authored", "caller-supplied", "small")
    if not all(term in architecture.casefold() for term in required_terms):
        raise ValueError("docs/ARCHITECTURE.md no longer contains the expected limitations source text")
    return {"figures": figures, "limitations": limitations, "limitations_source": "docs/ARCHITECTURE.md §15"}


def main() -> int:
    output = ROOT / "demo" / "data" / "claims.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_claims(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
