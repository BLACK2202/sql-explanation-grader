"""Deterministic SQL feature detection used by the judge and comparison reports.

This is intentionally conservative: it only reports operations that are visibly
present in the SQL text. The judge is told to evaluate completeness against this
inventory instead of inventing requirements for absent clauses.
"""
from __future__ import annotations

import re


CATEGORY_PATTERNS: list[tuple[str, str]] = [
    ("JOIN", r"\b(?:INNER\s+|LEFT\s+(?:OUTER\s+)?|RIGHT\s+(?:OUTER\s+)?|FULL\s+(?:OUTER\s+)?|CROSS\s+)?JOIN\b"),
    ("WHERE", r"\bWHERE\b"),
    ("GROUP BY", r"\bGROUP\s+BY\b"),
    ("HAVING", r"\bHAVING\b"),
    ("ORDER BY", r"\bORDER\s+BY\b"),
    ("aggregation", r"\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\("),
    ("subquery", r"\(\s*SELECT\b"),
    ("LIMIT", r"\bLIMIT\b"),
    ("DISTINCT", r"\bDISTINCT\b"),
]


def detect_categories(sql: str) -> list[str]:
    """Return canonical categories in a stable order."""
    normalized = re.sub(r"--.*?$", " ", sql, flags=re.MULTILINE)
    return [name for name, pattern in CATEGORY_PATTERNS if re.search(pattern, normalized, flags=re.IGNORECASE)]


def operation_inventory(sql: str) -> str:
    cats = detect_categories(sql)
    if not cats:
        return "No special clauses detected beyond SELECT/FROM."
    return ", ".join(cats)
