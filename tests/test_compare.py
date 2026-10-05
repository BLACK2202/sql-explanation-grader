import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

from compare import outcome, sha256_text


class CompareTests(unittest.TestCase):
    def test_pair_outcomes(self):
        self.assertEqual(outcome(1, 1), "both_correct")
        self.assertEqual(outcome(1, 0), "only_a_correct")
        self.assertEqual(outcome(0, 1), "only_b_correct")
        self.assertEqual(outcome(0, 0), "both_wrong")
        self.assertEqual(outcome(None, 1), "evaluation_error")

    def test_hash_is_deterministic(self):
        self.assertEqual(sha256_text("sql", "prompt"), sha256_text("sql", "prompt"))
        self.assertNotEqual(sha256_text("sql", "prompt"), sha256_text("sql2", "prompt"))


if __name__ == "__main__":
    unittest.main()
