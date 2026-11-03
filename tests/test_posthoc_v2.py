"""Regression coverage for post-hoc v2 trust-aware projection and routing."""

from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engramOS.db import connection, initialize_database, transaction  # noqa: E402
from engramOS.embeddings import BGE_M3_DIMENSION, index_history  # noqa: E402
from engramOS.pipeline import answer_query  # noqa: E402
from engramOS.query_classifier import (  # noqa: E402
    QUERY_CLASSIFICATION_SCHEMA_VERSION,
    classify_query,
)
from engramOS.reader import ReaderPrompt  # noqa: E402
from engramOS.temporal import (  # noqa: E402
    FactInput,
    get_current_facts,
    get_fact_history,
    insert_fact,
    rebuild_slot_projection,
)
from tests.helpers import record_store_event  # noqa: E402
from eval.loader import load_suite, suite_stats  # noqa: E402
from scripts.run_eval import _grade_question  # noqa: E402


def classifier_result(status: str, reason: str) -> str:
    return json.dumps({
        "schema_version": QUERY_CLASSIFICATION_SCHEMA_VERSION,
        "status": status,
        "view": None if status != "classified" else "current",
        "as_of": None,
        "history_range": None,
        "reason_code": reason,
    })


class StubClassifier:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def classify(self, query: str, *, question_date: date, timeout: float) -> str:
        del query, question_date, timeout
        self.calls += 1
        return self.response


class PosthocClassifierTests(unittest.TestCase):
    def test_four_date_free_v1_misroutes_are_overridden_to_current(self) -> None:
        fixtures = (
            ("Which gallery contains the main hall?", "unsupported", "unsupported_temporal_expression"),
            ("How can I pay for the ferry?", "unsupported", "knowledge_time_unsupported"),
            ("Which weekday do I keep clear of meetings?", "clarification_required", "ambiguous_temporal_intent"),
            ("Which region does the cloud service use?", "unsupported", "future_as_of_unsupported"),
        )
        for query, status, reason in fixtures:
            with self.subTest(query=query):
                provider = StubClassifier(classifier_result(status, reason))
                result = classify_query(
                    query, question_date="2026-11-03", classifier=provider
                )
                self.assertEqual((result.status, result.view, result.reason_code),
                                 ("classified", "current", "present_state"))
                self.assertEqual(provider.calls, 1)

    def test_explicit_future_date_and_knowledge_time_still_abstain(self) -> None:
        provider = StubClassifier(classifier_result("unsupported", "future_as_of_unsupported"))
        future = classify_query(
            "What region will the service use on 2099-01-01?",
            question_date="2026-11-03", classifier=provider,
        )
        knowledge = classify_query(
            "What did I know about the service on 2026-02-12?",
            question_date="2026-11-03", classifier=provider,
        )
        self.assertEqual((future.status, future.reason_code),
                         ("unsupported", "future_as_of_unsupported"))
        self.assertEqual((knowledge.status, knowledge.reason_code),
                         ("unsupported", "knowledge_time_unsupported"))
        self.assertEqual(provider.calls, 0)


class HeldoutSuiteLockTests(unittest.TestCase):
    def test_heldout_suite_stats_and_locked_hash(self) -> None:
        suite_path = PROJECT_ROOT / "eval" / "suite_v2_heldout.json"
        self.assertEqual(
            hashlib.sha256(suite_path.read_bytes()).hexdigest(),
            "5c64470c0773d9bb4b2d672b1a207c24959784f7cb5e7539497b02ef03dc5b85",
        )
        suite = load_suite(suite_path)
        self.assertEqual(
            (suite_stats(suite).histories, suite_stats(suite).turns, suite_stats(suite).questions),
            (8, 12, 8),
        )
        self.assertEqual(suite["reporting_labels"]["poison_adaptive_user"], "compromised_trusted_channel")

    def test_predeclared_accepted_answer_aliases_are_used(self) -> None:
        question = {
            "expected_label": "answer_contains",
            "expected_answer": "every 30 days",
            "accepted_answers": ["30 days", "every 30 days"],
        }
        self.assertTrue(_grade_question("The rule is 30 days.", False, question))
        self.assertFalse(_grade_question("The rule is quarterly.", False, question))

    def test_provider_cannot_override_explicit_date_route(self) -> None:
        provider = StubClassifier(classifier_result("unsupported", "knowledge_time_unsupported"))
        result = classify_query(
            "What was my editor on 2026-02-01?", question_date="2026-11-03",
            classifier=provider,
        )
        self.assertEqual((result.status, result.view, result.reason_code),
                         ("classified", "as_of", "explicit_as_of"))
        self.assertEqual(provider.calls, 0)


class FakeEmbeddingProvider:
    model_id = "posthoc-v2-test-embedding"
    dimension = BGE_M3_DIMENSION
    device = "cpu"

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        vector = np.zeros(BGE_M3_DIMENSION, dtype=np.float32)
        vector[int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little") % BGE_M3_DIMENSION] = 1
        return vector

    def encode_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def encode_documents(self, texts: list[str] | tuple[str, ...]) -> np.ndarray:
        return np.vstack([self._vector(text) for text in texts]) if texts else np.empty((0, BGE_M3_DIMENSION))


class CapturingReader:
    model = "fake-posthoc-reader"

    def __init__(self) -> None:
        self.prompts: list[ReaderPrompt] = []

    def generate(self, prompt: ReaderPrompt, *, timeout: float) -> str:
        del timeout
        self.prompts.append(prompt)
        payload = json.loads(prompt.user_payload)
        evidence_id = payload["evidence"][0]["evidence_id"]
        return json.dumps({
            "schema_version": "engram.reader-response.v1",
            "answer": "The jacket color is blue.",
            "citations": [evidence_id],
            "abstained": False,
        })


class ProvenanceSupersessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.database = str(Path(self.tmp.name) / "v2.sqlite3")
        initialize_database(self.database)
        self.serial = 0

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def add_fact(
        self, value: str, source_type: str, when: float, *, fact_id: str | None = None,
        attribute: str = "jacket_color", entity: str = "user", raw: str | None = None,
    ) -> tuple[str, str]:
        self.serial += 1
        session = f"s{self.serial}"
        turn_id = f"h1:{session}:0"
        fact_id = fact_id or f"f{self.serial}"
        with connection(self.database) as conn:
            conn.execute(
                """INSERT INTO turns (
                    turn_id, history_id, session_id, turn_idx, role, raw_content,
                    source_type, event_time, recorded_at, hash
                ) VALUES (?, 'h1', ?, 0, 'user', ?, ?, ?, ?, ?)""",
                (turn_id, session, raw or f"The {attribute.replace('_', ' ')} is {value}.",
                 source_type, when, when + 1000, f"hash-{fact_id}"),
            )
            record_store_event(conn, turn_id, created_at=when + 1000)
        insert_fact(self.database, FactInput(
            fact_id, "h1", turn_id, 0, entity, attribute, value, "single", when,
            when + 1000, kind="constraint" if attribute == "policy" else "preference",
            key_text=f"{entity}.{attribute}: {value}", trust=0.5,
        ))
        return turn_id, fact_id

    def test_untrusted_external_fact_cannot_supersede_trusted_and_is_auditable(self) -> None:
        self.add_fact("blue", "user_turn", 100.0, fact_id="trusted-blue")
        external_turn, _ = self.add_fact(
            "cobalt", "external_doc", 200.0, fact_id="untrusted-cobalt",
            raw="A wardrobe registry says the jacket is cobalt.",
        )
        current = get_current_facts(self.database, "h1", entity="user", attribute="jacket_color")
        history = get_fact_history(self.database, "h1", entity="user", attribute="jacket_color")
        with connection(self.database) as conn:
            observation = conn.execute(
                "SELECT canonical_fact_id FROM fact_observations WHERE turn_id=?", (external_turn,)
            ).fetchone()
            projected = conn.execute(
                "SELECT 1 FROM facts WHERE fact_id='untrusted-cobalt'"
            ).fetchone()
        self.assertEqual([row["value"] for row in current], ["blue"])
        self.assertEqual([(row["value"], row["valid_from"], row["valid_until"]) for row in history],
                         [("blue", 100.0, None)])
        self.assertIsNone(observation["canonical_fact_id"])
        self.assertIsNone(projected)

    def test_trusted_user_can_confirm_and_promote_untrusted_proposal(self) -> None:
        self.add_fact("blue", "user_turn", 100.0, fact_id="trusted-blue")
        self.add_fact("cobalt", "external_doc", 200.0, fact_id="external-cobalt")
        self.add_fact("cobalt", "user_turn", 300.0, fact_id="user-cobalt")
        rows = get_fact_history(self.database, "h1", entity="user", attribute="jacket_color")
        self.assertEqual([(row["value"], row["valid_from"], row["valid_until"]) for row in rows],
                         [("blue", 100.0, 300.0), ("cobalt", 300.0, None)])
        self.assertEqual(rows[-1]["last_confirmed_turn_id"], "h1:s3:0")

    def test_untrusted_source_can_update_an_untrusted_slot(self) -> None:
        self.add_fact("East", "external_doc", 100.0, fact_id="external-east", attribute="region")
        self.add_fact("West", "tool", 200.0, fact_id="tool-west", attribute="region")
        current = get_current_facts(self.database, "h1", entity="user", attribute="region")
        self.assertEqual([(row["value"], row["fact_id"]) for row in current], [("West", "tool-west")])

    def test_trusted_a_b_a_interval_semantics_are_unchanged(self) -> None:
        self.add_fact("A", "user_turn", 100.0, fact_id="a1", attribute="mode")
        self.add_fact("B", "user_turn", 200.0, fact_id="b", attribute="mode")
        self.add_fact("A", "user_turn", 300.0, fact_id="a2", attribute="mode")
        rows = get_fact_history(self.database, "h1", entity="user", attribute="mode")
        self.assertEqual([(row["fact_id"], row["value"], row["valid_from"], row["valid_until"]) for row in rows], [
            ("a1", "A", 100.0, 200.0), ("b", "B", 200.0, 300.0), ("a2", "A", 300.0, None),
        ])

    def test_projection_replay_is_stable(self) -> None:
        self.add_fact("blue", "user_turn", 100.0, fact_id="trusted-blue")
        self.add_fact("cobalt", "external_doc", 200.0, fact_id="external-cobalt")
        before_facts = get_fact_history(self.database, "h1", entity="user", attribute="jacket_color")
        with transaction(self.database) as conn:
            rebuild_slot_projection(conn, "h1", "user", "jacket_color", 3000.0)
        after_facts = get_fact_history(self.database, "h1", entity="user", attribute="jacket_color")
        with connection(self.database) as conn:
            observations = [tuple(row) for row in conn.execute(
                "SELECT observation_id, canonical_fact_id, effective_at FROM fact_observations ORDER BY observation_id"
            )]
        with transaction(self.database) as conn:
            rebuild_slot_projection(conn, "h1", "user", "jacket_color", 3000.0)
        with connection(self.database) as conn:
            observations_again = [tuple(row) for row in conn.execute(
                "SELECT observation_id, canonical_fact_id, effective_at FROM fact_observations ORDER BY observation_id"
            )]
        self.assertEqual(before_facts, after_facts)
        self.assertEqual(observations, observations_again)

    def test_conflicting_raw_untrusted_turn_is_not_in_current_reader_context(self) -> None:
        self.add_fact("blue", "user_turn", 1_767_225_600.0, fact_id="trusted-blue",
                      raw="My jacket color is blue.")
        self.add_fact("cobalt", "external_doc", 1_769_817_600.0, fact_id="external-cobalt",
                      raw="A wardrobe registry says the jacket is cobalt.")
        embedding = FakeEmbeddingProvider()
        index_history(self.database, "h1", embedding)
        reader = CapturingReader()
        outcome = answer_query(
            self.database, "h1", "What is my jacket color?", embedding, reader,
            question_date="2026-11-03",
        )
        self.assertEqual(outcome.status, "answered")
        self.assertEqual(len(reader.prompts), 1)
        payload = reader.prompts[0].user_payload
        self.assertIn("blue", payload.casefold())
        self.assertNotIn("cobalt", payload.casefold())

