import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

from sql_features import detect_categories, operation_inventory


class FeatureTests(unittest.TestCase):
    def test_no_phantom_join_or_where_from_strings(self):
        sql = "SELECT 'JOIN WHERE GROUP BY' AS note FROM doctors WHERE name IN (SELECT name FROM patients WHERE birth_year > 1990);"
        cats = detect_categories(sql)
        self.assertNotIn("JOIN", cats)
        self.assertIn("WHERE", cats)
        self.assertIn("subquery", cats)
        self.assertIn("IN", cats)

    def test_complex_categories(self):
        sql = "SELECT city, COUNT(*) FROM customers WHERE active=1 GROUP BY city HAVING COUNT(*)>2 ORDER BY COUNT(*) DESC LIMIT 5 OFFSET 2"
        cats = detect_categories(sql)
        for c in ["WHERE", "GROUP BY", "HAVING", "ORDER BY", "aggregation", "LIMIT", "OFFSET"]:
            self.assertIn(c, cats)

    def test_window_case_and_cte(self):
        sql = "WITH ranked AS (SELECT name, ROW_NUMBER() OVER (ORDER BY score) AS rn, CASE WHEN score > 90 THEN 1 ELSE 0 END ok FROM students) SELECT * FROM ranked WHERE rn <= 3;"
        cats = detect_categories(sql)
        for c in ["CTE", "window", "ORDER BY", "CASE", "WHERE"]:
            self.assertIn(c, cats)

    def test_set_ops_and_filters(self):
        sql = "SELECT id FROM a WHERE x BETWEEN 1 AND 5 UNION SELECT id FROM b WHERE name LIKE 'JOIN';"
        cats = detect_categories(sql)
        for c in ["WHERE", "BETWEEN", "UNION", "LIKE"]:
            self.assertIn(c, cats)
        self.assertNotIn("JOIN", cats)

    def test_inventory_has_absent_operation_guardrail(self):
        inventory = operation_inventory("SELECT name FROM users;")
        self.assertIn("basic SELECT/FROM", inventory)
        self.assertIn("Absent-operation rule", inventory)


if __name__ == "__main__":
    unittest.main()
