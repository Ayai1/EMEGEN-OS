"""Strict local tests for the provider-independent extraction boundary."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engramOS.extraction import (  # noqa: E402
    ExtractionResponseError,
    ExtractionTurn,
    TransientExtractionError,
    build_extraction_prompt,
    extract_and_validate_with_bounded_retries,
    extract_with_bounded_retries,
    serialize_extraction,
    validate_extraction_response,
)


def sample_fact(**updates: object) -> dict[str, object]:
    fact: dict[str, object] = {
        "session_id": "session-1",
        "turn_idx": 2,
        "entity": "user",
        "attribute": "home_city",
        "value": "New York",
        "cardinality": "single",
        "kind": "state",
        "as_of": None,
        "replaces": None,
        "confidence": 0.9,
    }
    fact.update(updates)
    return fact


class FakeTurnExtractor:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls = 0
        self.last_turn: ExtractionTurn | None = None
        self.last_prompt = None

    def extract(self, turn: ExtractionTurn, prompt: object) -> str:
        self.calls += 1
        self.last_turn = turn
        self.last_prompt = prompt
        return self.response  # type: ignore[return-value]


class ExtractionContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.turn = ExtractionTurn(
            turn_id="history-1:session-1:2",
            history_id="history-1",
            session_id="session-1",
            session_date="2024-03-04",
            session_unix_ts=1709510400.0,
            turn_idx=2,
            role="user",
            raw_content="I moved to New York.",
            source_type="user_turn",
            event_time=1709510402.0,
        )

    def test_prompt_preserves_original_section_a_instructions_and_source_text(self) -> None:
        prompt = build_extraction_prompt(self.turn, (("user", "home_city"),))
        self.assertIn("Return a JSON array only, no prose.", prompt.system)
        self.assertIn('"as_of":', prompt.system)
        self.assertIn('"replaces":', prompt.system)
        self.assertIn("Only extract facts the text actually states. Never infer or guess.", prompt.system)
        self.assertIn("- user.home_city", prompt.system)
        self.assertEqual(
            prompt.user,
            "=== Session session-1 | Date: 2024-03-04 ===\n"
            "[2] user: I moved to New York.",
        )

    def test_valid_response_preserves_schema_and_assigns_sequence_by_array_order(self) -> None:
        response = json.dumps(
            [
                sample_fact(value="Three bedrooms", replaces="Boston"),
                sample_fact(attribute="room_count", value="3", fact_seq=None),
            ]
        )
        # `fact_seq` is not in the specified model schema and must be rejected.
        with self.assertRaisesRegex(ExtractionResponseError, "invalid fields"):
            validate_extraction_response(response, self.turn)

        response = json.dumps(
            [
                sample_fact(value="Three bedrooms", replaces="Boston"),
                sample_fact(attribute="room_count", value="3"),
            ]
        )
        facts = validate_extraction_response(response, self.turn)
        self.assertEqual([fact.fact_seq for fact in facts], [0, 1])
        self.assertEqual(facts[0].value, "Three bedrooms")
        self.assertEqual(facts[0].replaces, "Boston")
        self.assertEqual(facts[1].attribute, "room_count")
        reparsed = validate_extraction_response(serialize_extraction(facts), self.turn)
        self.assertEqual(facts, reparsed)

    def test_empty_array_is_a_valid_successful_extraction(self) -> None:
        self.assertEqual(validate_extraction_response("[]", self.turn), ())

    def test_as_of_equal_to_session_date_is_accepted_and_preserved(self) -> None:
        response = json.dumps([sample_fact(as_of="2024-03-04")])

        facts = validate_extraction_response(response, self.turn)

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].as_of, "2024-03-04")
        self.assertEqual(facts[0].value, "New York")

    def test_as_of_before_session_date_is_accepted_and_preserved(self) -> None:
        response = json.dumps([sample_fact(as_of="2024-03-03")])

        facts = validate_extraction_response(response, self.turn)

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].as_of, "2024-03-03")
        self.assertEqual(facts[0].value, "New York")

    def test_as_of_after_session_date_is_rejected(self) -> None:
        response = json.dumps([sample_fact(as_of="2024-03-05")])

        with self.assertRaisesRegex(ExtractionResponseError, r"index 0.*2024-03-05"):
            validate_extraction_response(response, self.turn)

    def test_far_future_as_of_is_rejected(self) -> None:
        response = json.dumps([sample_fact(as_of="2099-01-01")])

        with self.assertRaisesRegex(ExtractionResponseError, r"index 0.*2099-01-01"):
            validate_extraction_response(response, self.turn)

    def test_invalid_source_session_date_is_checked_only_for_non_null_as_of(self) -> None:
        invalid_turn = ExtractionTurn(
            turn_id=self.turn.turn_id,
            history_id=self.turn.history_id,
            session_id=self.turn.session_id,
            session_date="not-a-date",
            session_unix_ts=self.turn.session_unix_ts,
            turn_idx=self.turn.turn_idx,
            role=self.turn.role,
            raw_content=self.turn.raw_content,
            source_type=self.turn.source_type,
            event_time=self.turn.event_time,
        )
        self.assertEqual(
            validate_extraction_response(json.dumps([sample_fact()]), invalid_turn)[0].as_of,
            None,
        )
        with self.assertRaises(ValueError) as caught:
            validate_extraction_response(
                json.dumps([sample_fact(as_of="2024-03-04")]), invalid_turn
            )
        self.assertIs(type(caught.exception), ValueError)

    def test_future_as_of_responses_retry_within_the_configured_bound(self) -> None:
        class SequencedExtractor:
            def __init__(self, *responses: str) -> None:
                self.responses = list(responses)
                self.calls = 0

            def extract(self, turn: ExtractionTurn, prompt: object) -> str:
                self.calls += 1
                return self.responses.pop(0)

        future = json.dumps([sample_fact(as_of="2099-01-01")])
        valid = json.dumps([sample_fact(as_of="2024-03-04")])
        extractor = SequencedExtractor(future, valid)
        facts = extract_and_validate_with_bounded_retries(
            extractor, self.turn, build_extraction_prompt(self.turn, ()), max_attempts=2
        )
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].as_of, "2024-03-04")
        self.assertEqual(extractor.calls, 2)

        always_future = SequencedExtractor(future, future, future)
        with self.assertRaisesRegex(ExtractionResponseError, "after 3 attempts"):
            extract_and_validate_with_bounded_retries(
                always_future,
                self.turn,
                build_extraction_prompt(self.turn, ()),
                max_attempts=3,
            )
        self.assertEqual(always_future.calls, 3)

    def test_malformed_json_and_non_array_responses_are_rejected(self) -> None:
        for response in ("not json", json.dumps(sample_fact())):
            with self.subTest(response=response):
                with self.assertRaises(ExtractionResponseError):
                    validate_extraction_response(response, self.turn)
        with self.assertRaisesRegex(ExtractionResponseError, "duplicate JSON object key"):
            validate_extraction_response(
                '[{"session_id":"session-1","session_id":"session-1"}]', self.turn
            )

    def test_missing_and_extra_fields_are_rejected(self) -> None:
        malformed = sample_fact()
        malformed.pop("confidence")
        with self.assertRaisesRegex(ExtractionResponseError, "missing"):
            validate_extraction_response(json.dumps([malformed]), self.turn)
        with self.assertRaisesRegex(ExtractionResponseError, "extra"):
            validate_extraction_response(
                json.dumps([sample_fact(history_id="other-history")]), self.turn
            )

    def test_wrong_session_or_turn_provenance_is_rejected(self) -> None:
        for updates in ({"session_id": "other"}, {"turn_idx": 3}):
            with self.subTest(updates=updates):
                with self.assertRaises(ExtractionResponseError):
                    validate_extraction_response(json.dumps([sample_fact(**updates)]), self.turn)

    def test_invalid_identifiers_values_categories_dates_and_confidence_are_rejected(self) -> None:
        cases = (
            {"entity": "User"},
            {"attribute": "home-city"},
            {"value": "   "},
            {"cardinality": "many"},
            {"kind": "belief"},
            {"as_of": "2024-02-30"},
            {"as_of": "yesterday"},
            {"replaces": ""},
            {"confidence": True},
            {"confidence": 1.1},
            {"confidence": float("nan")},
            {"confidence": 10**1000},
            {"turn_idx": True},
        )
        for updates in cases:
            with self.subTest(updates=updates):
                with self.assertRaises(ExtractionResponseError):
                    validate_extraction_response(json.dumps([sample_fact(**updates)]), self.turn)

    def test_retry_is_bounded_and_only_retries_explicit_transient_errors(self) -> None:
        class TransientThenSuccess:
            calls = 0

            def extract(self, turn: ExtractionTurn, prompt: object) -> str:
                self.calls += 1
                if self.calls < 3:
                    raise TransientExtractionError("temporary local adapter failure")
                return "[]"

        extractor = TransientThenSuccess()
        response = extract_with_bounded_retries(
            extractor, self.turn, build_extraction_prompt(self.turn, ()), max_attempts=3
        )
        self.assertEqual(response, "[]")
        self.assertEqual(extractor.calls, 3)

        exhausted = TransientThenSuccess()
        exhausted.calls = 0
        with self.assertRaisesRegex(RuntimeError, "after 2 attempts"):
            extract_with_bounded_retries(
                exhausted, self.turn, build_extraction_prompt(self.turn, ()), max_attempts=2
            )
        self.assertEqual(exhausted.calls, 2)

    def test_malformed_responses_receive_bounded_validation_retries(self) -> None:
        class MalformedThenValid:
            calls = 0

            def extract(self, turn: ExtractionTurn, prompt: object) -> str:
                self.calls += 1
                return "not json" if self.calls == 1 else "[]"

        extractor = MalformedThenValid()
        facts = extract_and_validate_with_bounded_retries(
            extractor, self.turn, build_extraction_prompt(self.turn, ()), max_attempts=2
        )
        self.assertEqual(facts, ())
        self.assertEqual(extractor.calls, 2)

        always_malformed = FakeTurnExtractor("not json")
        with self.assertRaisesRegex(ExtractionResponseError, "after 2 attempts"):
            extract_and_validate_with_bounded_retries(
                always_malformed,
                self.turn,
                build_extraction_prompt(self.turn, ()),
                max_attempts=2,
            )
        self.assertEqual(always_malformed.calls, 2)


if __name__ == "__main__":
    unittest.main()
