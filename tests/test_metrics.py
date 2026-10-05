import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

from metrics import mcnemar_exact_p, paired_bootstrap_delta, summarize


def row(a, b, category="JOIN"):
    dims = {"correctness": True, "completeness": True, "hallucination_free": True, "clarity": True}
    return {
        "id": f"q{a}{b}",
        "categories": [category],
        "outcome": "both_correct" if a and b else "only_a_correct" if a else "only_b_correct" if b else "both_wrong",
        "disagreement": a != b,
        "model_a_name": "A",
        "model_b_name": "B",
        "model_a": {"score": a, "dimensions": dims, "generation_cost": {"latency_s": 1.0, "in_tok": 10, "out_tok": 20, "tok_per_s": 20}},
        "model_b": {"score": b, "dimensions": dims, "generation_cost": {"latency_s": 2.0, "in_tok": 12, "out_tok": 24, "tok_per_s": 12}},
    }


class MetricTests(unittest.TestCase):
    def test_pair_stats(self):
        rows = [row(1, 1), row(1, 0), row(0, 1), row(0, 0)]
        s = summarize(rows)
        self.assertEqual(s["n_valid"], 4)
        self.assertEqual(s["pair_counts"]["both_correct"], 1)
        self.assertEqual(s["pair_counts"]["only_a_correct"], 1)
        self.assertEqual(s["pair_counts"]["only_b_correct"], 1)
        self.assertEqual(s["pair_counts"]["both_wrong"], 1)
        self.assertAlmostEqual(s["a_pass_rate_pct"], 50.0)
        self.assertAlmostEqual(s["b_pass_rate_pct"], 50.0)

    def test_bootstrap_is_deterministic(self):
        rows = [row(1, 1), row(1, 0), row(0, 1), row(0, 0)]
        self.assertEqual(
            paired_bootstrap_delta(rows, iterations=500, seed=7),
            paired_bootstrap_delta(rows, iterations=500, seed=7),
        )

    def test_mcnemar_no_disagreement(self):
        self.assertIsNone(mcnemar_exact_p([row(1, 1), row(0, 0)]))


if __name__ == "__main__":
    unittest.main()
