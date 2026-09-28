import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

from sql_features import detect_categories
from compare import outcome


class FeatureTests(unittest.TestCase):
    def test_no_phantom_join(self):
        sql = "SELECT * FROM doctors WHERE name IN (SELECT name FROM patients WHERE birth_year > 1990);"
        cats = detect_categories(sql)
        self.assertNotIn("JOIN", cats)
        self.assertIn("WHERE", cats)
        self.assertIn("subquery", cats)

    def test_complex_categories(self):
        sql = "SELECT city, COUNT(*) FROM customers WHERE active=1 GROUP BY city HAVING COUNT(*)>2 ORDER BY COUNT(*) DESC LIMIT 5"
        cats = detect_categories(sql)
        for c in ["WHERE", "GROUP BY", "HAVING", "ORDER BY", "aggregation", "LIMIT"]:
            self.assertIn(c, cats)

    def test_outcomes(self):
        self.assertEqual(outcome(1, 1), "both_correct")
        self.assertEqual(outcome(1, 0), "only_a_correct")
        self.assertEqual(outcome(0, 1), "only_b_correct")
        self.assertEqual(outcome(0, 0), "both_wrong")


if __name__ == "__main__":
    unittest.main()
