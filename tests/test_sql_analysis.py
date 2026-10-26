"""SQLGlot structure analysis, retrieval-query construction and safe validation."""
from __future__ import annotations

import json

import pytest

from conftest import ROOT
from sqlrag.sql_analysis import SQLValidationError, analyze_sql, build_retrieval_queries, mask_sql, validate_sql

CASES = [
    ("SELECT * FROM t WHERE a > 1", {"select_star", "where"}),
    ("SELECT c.name FROM c LEFT JOIN o ON o.cid = c.id WHERE o.id IS NULL", {"left_join", "is_null", "where"}),
    ("SELECT a FROM t JOIN u ON u.x = t.x JOIN v ON v.y = u.y", {"inner_join", "multi_join"}),
    ("SELECT city, COUNT(*) FROM c GROUP BY city HAVING COUNT(*) > 2", {"group_by", "having", "count"}),
    ("SELECT name, RANK() OVER (PARTITION BY d ORDER BY s DESC) FROM e", {"window", "partition_by", "window_order", "rank"}),
    ("SELECT name, ROW_NUMBER() OVER (ORDER BY s) FROM e", {"window", "row_number", "window_order"}),
    ("SELECT SUM(x) OVER (PARTITION BY g) FROM t", {"window", "window_aggregate", "partition_by"}),
    ("WITH a AS (SELECT 1 AS x) SELECT x FROM a", {"cte"}),
    ("SELECT CASE WHEN a > 1 THEN 'hi' ELSE 'lo' END FROM t", {"case"}),
    ("SELECT * FROM t WHERE a IN (SELECT a FROM u)", {"in_subquery", "where", "select_star"}),
    ("SELECT * FROM t WHERE a NOT IN (1, 2)", {"in_list", "not_in"}),
    ("SELECT * FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.a = t.a)", {"exists"}),
    ("SELECT * FROM t WHERE a = (SELECT MAX(a) FROM t)", {"scalar_subquery", "max"}),
    ("SELECT DISTINCT a FROM t ORDER BY a DESC LIMIT 3 OFFSET 1", {"distinct", "order_by", "limit", "offset"}),
    ("SELECT COUNT(DISTINCT a) FROM t", {"count", "count_distinct"}),
    ("SELECT a FROM t WHERE b BETWEEN 1 AND 5 AND c LIKE 'x%'", {"between", "like"}),
    ("SELECT a FROM t UNION SELECT a FROM u", {"union"}),
    ("SELECT upper(name), strftime('%Y', d) FROM t", {"fn:upper", "fn:strftime"}),
]


@pytest.mark.parametrize("sql,expected", CASES)
def test_feature_detection(sql, expected):
    a = analyze_sql(sql)
    assert a.parsed
    assert expected <= set(a.features), (sql, a.features)


def test_details_order_limit_tables():
    a = analyze_sql("SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id ORDER BY c.name DESC LIMIT 10")
    assert a.tables == ["customers", "orders"]
    assert a.limit == "10"
    assert a.order_by == [{"expr": "c.name", "desc": True}]
    assert a.joins[0]["type"] == "INNER JOIN"
    inv = a.inventory()
    assert "LIMIT 10" in inv and "DESC" in inv and "Tables read: customers, orders" in inv


def test_strings_and_comments_are_masked():
    masked = mask_sql("SELECT 'count(' , \"x(\" -- sum(\nFROM t /* avg( */")
    assert "count(" not in masked and "sum(" not in masked and "avg(" not in masked
    assert len(masked) == len("SELECT 'count(' , \"x(\" -- sum(\nFROM t /* avg( */")
    a = analyze_sql("SELECT 'count(*) here' FROM t")
    assert "count" not in a.features


def test_unparseable_sql_falls_back_to_regex():
    a = analyze_sql("SELECT a FROM t LEFT JOIN u ON WHERE b IS NULL ORDER BY")
    assert not a.parsed or a.features  # either SQLGlot copes or the regex fallback runs
    if not a.parsed:
        assert "left_join" in a.features and "is_null" in a.features


def test_retrieval_queries_target_feature_interactions():
    a = analyze_sql("SELECT c.name FROM c LEFT JOIN o ON o.cid = c.id WHERE o.id IS NULL")
    qs = build_retrieval_queries(a)
    left = [q for q in qs if q.feature == "left_join"]
    assert left and "unmatched" in left[0].text and "IS NULL" in left[0].text
    assert qs[-1].feature is None  # whole-query probe last


def test_every_dataset_item_parses_and_validates():
    for line in (ROOT / "data" / "all.jsonl").open(encoding="utf-8"):
        it = json.loads(line)
        assert analyze_sql(it["sql"]).parsed, it["id"]
        validate_sql(it["sql"], it["schema"])


@pytest.mark.parametrize("sql,schema,msg", [
    ("DELETE FROM t", "CREATE TABLE t(a)", "read-only"),
    ("SELECT 1; SELECT 2", "CREATE TABLE t(a)", "exactly one"),
    ("SELECT * FROM missing", "CREATE TABLE t(a)", "no such table"),
    ("SELECT 1", "ATTACH DATABASE 'evil.db' AS e", "schema may only"),
    ("SELECT 1", "CREATE TABLE t(a); DROP TABLE t", "schema may only"),
    ("PRAGMA table_info(t)", "CREATE TABLE t(a)", "read-only"),
    ("SELECT * FROM t" + " " * 20_000, "CREATE TABLE t(a)", "longer"),
])
def test_validation_rejects_unsafe_or_invalid_input(sql, schema, msg):
    with pytest.raises(SQLValidationError, match=msg):
        validate_sql(sql, schema)


def test_validation_accepts_semicolons_inside_strings():
    validate_sql("SELECT a FROM t WHERE b = 'x;y';", "CREATE TABLE t(a, b);")


def test_validation_never_creates_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT 1", "CREATE TABLE t(a); ATTACH 'x.db' AS x;")
    assert list(tmp_path.iterdir()) == []
