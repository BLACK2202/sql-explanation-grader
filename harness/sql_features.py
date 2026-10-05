"""Conservative SQL feature detection for judge guardrails and analysis."""
from __future__ import annotations

import re

CATEGORY_PATTERNS: list[tuple[str, str]] = [
    ("JOIN", r"\b(?:INNER\s+|LEFT\s+(?:OUTER\s+)?|RIGHT\s+(?:OUTER\s+)?|FULL\s+(?:OUTER\s+)?|CROSS\s+)?JOIN\b"),
    ("WHERE", r"\bWHERE\b"),
    ("GROUP BY", r"\bGROUP\s+BY\b"),
    ("HAVING", r"\bHAVING\b"),
    ("ORDER BY", r"\bORDER\s+BY\b"),
    ("aggregation", r"\b(?:COUNT|SUM|AVG|MIN|MAX|TOTAL|GROUP_CONCAT)\s*\("),
    ("window", r"\bOVER\s*\("),
    ("subquery", r"\(\s*(?:WITH\b|SELECT\b)"),
    ("CTE", r"\bWITH\s+(?:RECURSIVE\s+)?[A-Za-z_]"),
    ("UNION", r"\bUNION(?:\s+ALL)?\b"),
    ("INTERSECT", r"\bINTERSECT\b"),
    ("EXCEPT", r"\bEXCEPT\b"),
    ("LIMIT", r"\bLIMIT\b"),
    ("OFFSET", r"\bOFFSET\b"),
    ("DISTINCT", r"\bDISTINCT\b"),
    ("CASE", r"\bCASE\b"),
    ("EXISTS", r"\b(?:NOT\s+)?EXISTS\b"),
    ("IN", r"\b(?:NOT\s+)?IN\s*\("),
    ("BETWEEN", r"\b(?:NOT\s+)?BETWEEN\b"),
    ("LIKE", r"\b(?:NOT\s+)?(?:LIKE|GLOB|REGEXP)\b"),
    ("NULL filter", r"\bIS\s+(?:NOT\s+)?NULL\b"),
]


def mask_sql(sql: str) -> str:
    chars = list(sql)
    n = len(chars)
    i = 0
    while i < n:
        if i + 1 < n and chars[i:i + 2] == ["-", "-"]:
            j = i
            while j < n and chars[j] not in "\r\n":
                chars[j] = " "
                j += 1
            i = j
            continue
        if i + 1 < n and chars[i:i + 2] == ["/", "*"]:
            j = i
            while j < n:
                if j + 1 < n and chars[j:j + 2] == ["*", "/"]:
                    chars[j] = chars[j + 1] = " "
                    j += 2
                    break
                if chars[j] not in "\r\n":
                    chars[j] = " "
                j += 1
            i = j
            continue
        if chars[i] in ("'", '"') or chars[i] == chr(96):
            quote = chars[i]
            chars[i] = " "
            i += 1
            while i < n:
                if chars[i] == quote:
                    chars[i] = " "
                    if i + 1 < n and chars[i + 1] == quote:
                        chars[i + 1] = " "
                        i += 2
                        continue
                    i += 1
                    break
                if chars[i] not in "\r\n":
                    chars[i] = " "
                i += 1
            continue
        if chars[i] == "[":
            chars[i] = " "
            i += 1
            while i < n:
                if chars[i] == "]":
                    chars[i] = " "
                    i += 1
                    break
                if chars[i] not in "\r\n":
                    chars[i] = " "
                i += 1
            continue
        i += 1
    return "".join(chars)


def detect_categories(sql: str) -> list[str]:
    masked = mask_sql(sql)
    return [name for name, pattern in CATEGORY_PATTERNS if re.search(pattern, masked, flags=re.IGNORECASE)]


def operation_inventory(sql: str) -> str:
    categories = detect_categories(sql)
    if not categories:
        return "Present operations: basic SELECT/FROM only.\nAbsent-operation rule: do not penalize the explanation for omitted optional SQL operations."
    return (
        "Present operations detected from the SQL: " + ", ".join(categories) + ".\n"
        "Absent-operation rule: do not require or penalize for operations not present in the SQL."
    )
