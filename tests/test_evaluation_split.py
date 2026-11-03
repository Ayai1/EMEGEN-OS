"""Checks the locked LongMemEval dev/test question ID split."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EvaluationSplitTests(unittest.TestCase):
    def test_split_has_expected_counts_and_no_overlap(self) -> None:
        split = json.loads((PROJECT_ROOT / "splits.json").read_text(encoding="utf-8"))
        self.assertEqual(split["seed"], 42)
        self.assertEqual(len(split["dev"]["knowledge-update"]), 14)
        self.assertEqual(len(split["dev"]["temporal-reasoning"]), 8)
        self.assertEqual(len(split["test"]["knowledge-update"]), 64)
        self.assertEqual(len(split["test"]["temporal-reasoning"]), 14)
        dev_ids = {qid for ids in split["dev"].values() for qid in ids}
        test_ids = {qid for ids in split["test"].values() for qid in ids}
        self.assertEqual(len(dev_ids), 22)
        self.assertEqual(len(test_ids), 78)
        self.assertFalse(dev_ids & test_ids)
        self.assertEqual(split["unused_temporal_reasoning_count"], 111)


if __name__ == "__main__":
    unittest.main()
