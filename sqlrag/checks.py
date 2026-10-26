"""Deterministic (model-free) checks on an explanation.

These checks use only the SQL structure (from SQLGlot) and the explanation
text. They are heuristics with known blind spots (synonyms, paraphrase), but
they never change between runs and do not depend on any LLM, which makes them
a useful second signal next to the LLM judge.

Checks (each is pass/fail and only applies when the SQL has the feature):
  limit_value        the LIMIT number appears (digits or words)
  order_direction    a word matching the final sort direction appears
  aggregates         each aggregate is named (count/number of, average, ...)
  null_filter        LEFT JOIN + IS NULL is described as "no/without/never ..."
  distinct           duplicate removal is mentioned
  grouping           grouping is mentioned ("each", "per", "group", ...)
  window_ranking     ranking/numbering is mentioned for ROW_NUMBER/RANK
  tables             every table (or its singular form) is mentioned
Signals (counts, not pass/fail):
  invented_numbers   numbers in the explanation that appear nowhere in the SQL
  sql_keyword_leaks  raw SQL keywords in the text (the v2 prompt forbids them)
"""
from __future__ import annotations

import re

from .sql_analysis import SQLAnalysis, mask_sql

NUM_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 15: "fifteen", 20: "twenty", 25: "twenty-five", 50: "fifty", 100: "hundred",
}
DESC_WORDS = ("descend", "highest", "largest", "most", "greatest", "biggest", "latest", "newest", "top", "decreas",
              "high to low", "largest to smallest", "most recent", "reverse", "from the best", "best", "longest", "oldest first")
ASC_WORDS = ("ascend", "alphabetic", "lowest", "smallest", "earliest", "oldest", "increas", "a to z", "low to high",
             "least", "fewest", "first to last", "chronolog", "shortest", "a-z")
AGG_WORDS = {
    "count": ("count", "number of", "how many", "tally"),
    "sum": ("sum", "total", "add"),
    "total": ("sum", "total", "add"),
    "avg": ("average", "mean"),
    "min": ("minimum", "lowest", "smallest", "earliest", "least", "fewest", "min"),
    "max": ("maximum", "highest", "largest", "latest", "most", "greatest", "biggest", "max"),
    "group_concat": ("concatenat", "combined", "joined together", "list of", "comma"),
}
NEGATION_WORDS = ("no ", "not ", "without", "never", "missing", "n't", "lack", "none", "absent", "unmatched", "no matching")
DISTINCT_WORDS = ("unique", "distinct", "duplicate", "different", "once")
GROUP_WORDS = ("each", "per ", "group", "for every", "by ", "category", "categories")
RANK_WORDS = ("rank", "number", "position", "order", "numbered", "1st", "first")
SQL_KEYWORDS = ("SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY", "HAVING", "LIMIT", "OFFSET", "DISTINCT",
                "PARTITION BY", "OVER", "UNION", "INNER", "LEFT JOIN", "IS NULL", "WITH", "CASE", "WHEN")


def _has(text: str, words) -> bool:
    return any(w in text for w in words)


def _singular(name: str) -> str:
    if name.endswith("ies"):
        return name[:-3] + "y"
    if name.endswith("ses"):
        return name[:-2]
    if name.endswith("s"):
        return name[:-1]
    return name


def run_checks(explanation: str, analysis: SQLAnalysis) -> dict:
    text = (explanation or "").lower()
    feats = set(analysis.features)
    checks: dict[str, bool] = {}

    if analysis.limit and analysis.limit.isdigit():
        n = int(analysis.limit)
        words = [str(n)] + ([NUM_WORDS[n]] if n in NUM_WORDS else [])
        checks["limit_value"] = any(re.search(rf"(?<![\d.]){re.escape(w)}(?![\d])", text) for w in words)
    elif "limit" in feats:
        checks["limit_value"] = _has(text, ("limit", "at most", "only the first", "top"))

    if analysis.order_by:
        first = analysis.order_by[0]
        checks["order_direction"] = _has(text, DESC_WORDS if first["desc"] else ASC_WORDS) or (
            not first["desc"] and _has(text, ("sorted", "ordered", "in order of", "arranged"))
        )

    for agg in ("count", "sum", "total", "avg", "min", "max", "group_concat"):
        if agg in feats:
            checks[f"aggregate_{agg}"] = _has(text, AGG_WORDS[agg])

    if {"left_join", "is_null"} <= feats:
        checks["null_filter"] = _has(text, NEGATION_WORDS)
    if "distinct" in feats or "count_distinct" in feats:
        checks["distinct"] = _has(text, DISTINCT_WORDS)
    if "group_by" in feats:
        checks["grouping"] = _has(text, GROUP_WORDS)
    if feats & {"row_number", "rank", "dense_rank", "ntile"}:
        checks["window_ranking"] = _has(text, RANK_WORDS)
    if analysis.tables:
        mentioned = [t for t in analysis.tables if t.lower() in text or _singular(t.lower()) in text]
        checks["tables"] = len(mentioned) == len(analysis.tables)

    sql_numbers = set(re.findall(r"\d+(?:\.\d+)?", analysis.sql))
    # Numbers inside string literals count as present in the SQL too.
    sql_numbers |= set(re.findall(r"\d+(?:\.\d+)?", mask_sql(analysis.sql) + " " + analysis.sql))
    expl_numbers = re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", explanation or "")
    invented = [n for n in expl_numbers if n not in sql_numbers and n not in {"1"}]
    leaks = [k for k in SQL_KEYWORDS if re.search(rf"\b{re.escape(k)}\b", explanation or "")]

    applicable = len(checks)
    passed = sum(checks.values())
    return {
        "checks": checks,
        "n_applicable": applicable,
        "n_passed": passed,
        "coverage_score": round(passed / applicable, 4) if applicable else None,
        "invented_numbers": invented,
        "sql_keyword_leaks": leaks,
    }
