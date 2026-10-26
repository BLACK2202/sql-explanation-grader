"""Structural analysis of SQL queries (SQLGlot) and retrieval-query construction.

The analysis has three consumers:
  1. retrieval: each detected feature becomes a focused retrieval query
     (e.g. LEFT JOIN + IS NULL -> "outer join keeps unmatched rows, NULL ...");
  2. grounding: each feature lists the evidence terms a retrieved passage must
     contain to count as covering that feature;
  3. the judge and the deterministic checks: a factual inventory of the
     operations in the query (tables, join types, sort direction, LIMIT value).

SQLGlot parses the structure. Function names are read from the original text
(after masking strings and comments) because SQLGlot normalises some names,
e.g. strftime -> TIME_TO_STR.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Feature catalogue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    label: str
    query: str                      # retrieval query text
    evidence_terms: tuple[str, ...]  # a covering passage contains one of these (case-insensitive)
    requires_evidence: bool = True


CATALOGUE: dict[str, FeatureSpec] = {f.key: f for f in [
    FeatureSpec("select_star", "SELECT * (all columns)", "SELECT * result columns: an asterisk expands to all columns of the tables in the FROM clause",
                ("*", "all columns", "asterisk"), requires_evidence=False),
    FeatureSpec("where", "WHERE filter", "WHERE clause filtering: rows for which the WHERE expression is false or NULL are excluded",
                ("WHERE clause", "WHERE expression", "WHERE"), requires_evidence=False),
    FeatureSpec("inner_join", "INNER JOIN", "inner join of two tables: only row pairs that satisfy the ON join constraint are kept",
                ("INNER JOIN", "inner join", "join constraint", "ON clause")),
    FeatureSpec("left_join", "LEFT (OUTER) JOIN", "LEFT JOIN left outer join: every row of the left table is kept; when there is no matching row in the right table the right-hand columns are NULL",
                ("LEFT JOIN", "LEFT OUTER JOIN", "left outer join", "outer join")),
    FeatureSpec("right_join", "RIGHT (OUTER) JOIN", "RIGHT OUTER JOIN keeps every row of the right table, NULL for unmatched left-hand columns",
                ("RIGHT JOIN", "RIGHT OUTER JOIN", "right outer join")),
    FeatureSpec("full_join", "FULL OUTER JOIN", "FULL OUTER JOIN keeps unmatched rows from both tables, filling the missing side with NULL",
                ("FULL JOIN", "FULL OUTER JOIN", "full outer join")),
    FeatureSpec("cross_join", "CROSS JOIN / comma join", "CROSS JOIN cartesian product: every row of the left table combined with every row of the right table",
                ("CROSS JOIN", "cartesian product", "cross join")),
    FeatureSpec("multi_join", "join of three or more tables", "joining more than two tables: the join operator is applied left to right to combine several tables",
                ("join", "JOIN"), requires_evidence=False),
    FeatureSpec("group_by", "GROUP BY", "GROUP BY clause: rows with the same values in the grouping expressions are combined into a single result row per group",
                ("GROUP BY",)),
    FeatureSpec("having", "HAVING", "HAVING clause filters groups after aggregation; groups for which the HAVING expression is false are discarded",
                ("HAVING",)),
    FeatureSpec("count", "COUNT aggregate", "count(*) returns the number of rows in the group; count(X) counts rows where X is not NULL",
                ("count(",)),
    FeatureSpec("count_distinct", "COUNT(DISTINCT ...)", "count(DISTINCT X) aggregate counts only distinct non-NULL values, duplicate elements are filtered",
                ("DISTINCT",)),
    FeatureSpec("sum", "SUM aggregate", "sum(X) total(X) aggregate returns the sum of all non-NULL values in the group",
                ("sum(",)),
    FeatureSpec("avg", "AVG aggregate", "avg(X) aggregate returns the average value of all non-NULL X within a group",
                ("avg(",)),
    FeatureSpec("min", "MIN aggregate", "min(X) aggregate returns the minimum value of all values in the group",
                ("min(",)),
    FeatureSpec("max", "MAX aggregate", "max(X) aggregate returns the maximum value of all values in the group",
                ("max(",)),
    FeatureSpec("group_concat", "GROUP_CONCAT aggregate", "group_concat(X) string_agg aggregate returns a string that is the concatenation of all non-NULL values",
                ("group_concat", "string_agg")),
    FeatureSpec("total", "TOTAL aggregate", "total(X) aggregate returns the floating point sum, 0.0 for no non-NULL rows",
                ("total(",)),
    FeatureSpec("distinct", "SELECT DISTINCT", "SELECT DISTINCT removes duplicate rows from the result set",
                ("DISTINCT",)),
    FeatureSpec("order_by", "ORDER BY", "ORDER BY clause sorts the returned rows; ASC ascending is the default, DESC sorts in descending order",
                ("ORDER BY",)),
    FeatureSpec("limit", "LIMIT", "LIMIT clause places an upper bound on the number of rows returned",
                ("LIMIT",)),
    FeatureSpec("offset", "OFFSET", "LIMIT OFFSET: the first M rows are skipped and the following rows are returned",
                ("OFFSET",)),
    FeatureSpec("scalar_subquery", "scalar subquery", "scalar subquery: a SELECT in parentheses used as an expression returns the value of the first row of its result",
                ("scalar subquer", "subquery", "sub-query")),
    FeatureSpec("in_subquery", "IN (subquery)", "IN operator with a subquery: true if the left operand equals any value returned by the subquery",
                ("IN operator", " IN (", "IN and NOT IN", "IN (SELECT", "subquery")),
    FeatureSpec("not_in", "NOT IN", "NOT IN operator: true if the value is not in the list or subquery result; NULL values make NOT IN behave unexpectedly",
                ("NOT IN",)),
    FeatureSpec("in_list", "IN (list)", "IN operator with a list of values: true if the left operand equals one of the values",
                ("IN operator", "IN and NOT IN", " IN (")),
    FeatureSpec("exists", "EXISTS", "EXISTS operator: true if the subquery returns one or more rows",
                ("EXISTS",)),
    FeatureSpec("derived_table", "subquery in FROM", "subquery in the FROM clause: a SELECT in parentheses acts as a table (derived table)",
                ("subquery", "sub-query", "derived table")),
    FeatureSpec("cte", "WITH (common table expression)", "WITH clause common table expression: a named temporary result set defined before the main SELECT and used like a table",
                ("common table expression", "WITH clause", "common-table-expression")),
    FeatureSpec("recursive_cte", "WITH RECURSIVE", "recursive common table expression WITH RECURSIVE: an initial select and a recursive select that refers to the CTE itself",
                ("RECURSIVE", "recursive common table expression")),
    FeatureSpec("window", "window function (OVER)", "window function OVER clause: computes a value for each row using a window of related rows; rows are not collapsed like GROUP BY",
                ("window function", "OVER clause", "OVER")),
    FeatureSpec("partition_by", "PARTITION BY", "PARTITION BY divides the rows into partitions; the window function is computed separately within each partition",
                ("PARTITION BY", "partition")),
    FeatureSpec("window_order", "window ORDER BY", "ORDER BY inside the OVER clause defines the order of rows within each partition for ranking and running totals",
                ("ORDER BY",)),
    FeatureSpec("window_frame", "window frame (ROWS/RANGE)", "window frame specification ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW, running totals",
                ("frame", "PRECEDING", "UNBOUNDED")),
    FeatureSpec("row_number", "ROW_NUMBER()", "row_number() window function: number of the row within its partition, starting from 1, ties get different numbers",
                ("row_number",)),
    FeatureSpec("rank", "RANK()", "rank() window function: rank with gaps; peers (ties) share the same rank and the next rank is skipped",
                ("rank()", "rank")),
    FeatureSpec("dense_rank", "DENSE_RANK()", "dense_rank() window function: rank without gaps; peers share a rank and the next rank is consecutive",
                ("dense_rank",)),
    FeatureSpec("ntile", "NTILE()", "ntile(N) window function divides the partition into N groups",
                ("ntile",)),
    FeatureSpec("lag_lead", "LAG()/LEAD()", "lag() lead() window functions return the value from a previous or following row in the partition",
                ("lag(", "lead(", "lag", "lead")),
    FeatureSpec("first_last_value", "FIRST_VALUE()/LAST_VALUE()/NTH_VALUE()", "first_value() last_value() nth_value() window functions return a value from the window frame",
                ("first_value", "last_value", "nth_value")),
    FeatureSpec("percent_rank", "PERCENT_RANK()/CUME_DIST()", "percent_rank() cume_dist() window functions return relative rank between 0 and 1",
                ("percent_rank", "cume_dist")),
    FeatureSpec("window_aggregate", "aggregate used as a window function", "aggregate window function: sum() count() avg() with an OVER clause computes a running or partitioned total without collapsing rows",
                ("aggregate window function", "aggregate window functions", "OVER clause")),
    FeatureSpec("case", "CASE expression", "CASE expression: WHEN conditions are evaluated in order and the THEN result of the first true condition is returned, otherwise ELSE or NULL",
                ("CASE",)),
    FeatureSpec("is_null", "IS NULL", "IS NULL operator tests whether a value is NULL (missing); comparisons with = NULL are never true",
                ("IS NULL", "ISNULL", "NULL")),
    FeatureSpec("is_not_null", "IS NOT NULL", "IS NOT NULL operator is true when the value is not NULL",
                ("IS NOT NULL", "IS NOT", "NOTNULL", "NOT NULL")),
    FeatureSpec("between", "BETWEEN", "BETWEEN operator: x BETWEEN y AND z is equivalent to x>=y AND x<=z, both ends inclusive",
                ("BETWEEN",)),
    FeatureSpec("like", "LIKE pattern", "LIKE operator pattern matching: percent sign matches any sequence of characters, underscore matches one character; case-insensitive for ASCII",
                ("LIKE",)),
    FeatureSpec("glob", "GLOB pattern", "GLOB operator Unix file globbing syntax wildcards, case sensitive",
                ("GLOB",)),
    FeatureSpec("union", "UNION / UNION ALL", "compound select UNION combines the rows of two SELECT statements, removing duplicates; UNION ALL keeps duplicates",
                ("UNION",)),
    FeatureSpec("intersect", "INTERSECT", "compound select INTERSECT returns rows that appear in both SELECT results",
                ("INTERSECT",)),
    FeatureSpec("except", "EXCEPT", "compound select EXCEPT returns rows of the left SELECT that are not in the right SELECT",
                ("EXCEPT",)),
    FeatureSpec("cast", "CAST", "CAST expression converts a value to a different storage class or type affinity",
                ("CAST",)),
    FeatureSpec("arithmetic", "arithmetic expression", "arithmetic operators + - * / % on numeric values; integer division truncates",
                ("operator",), requires_evidence=False),
    FeatureSpec("string_concat", "|| concatenation", "the || operator concatenates two strings",
                ("||", "concatenat")),
    FeatureSpec("self_join", "self join", "self join: the same table appears twice in the FROM clause with different aliases",
                ("alias", "join"), requires_evidence=False),
]}

AGGREGATE_FUNCS = {"count", "sum", "avg", "min", "max", "total", "group_concat", "string_agg"}
WINDOW_FUNCS = {
    "row_number": "row_number", "rank": "rank", "dense_rank": "dense_rank", "ntile": "ntile",
    "lag": "lag_lead", "lead": "lag_lead", "first_value": "first_last_value", "last_value": "first_last_value",
    "nth_value": "first_last_value", "percent_rank": "percent_rank", "cume_dist": "percent_rank",
}
DATE_FUNCS = {"date", "time", "datetime", "julianday", "unixepoch", "strftime", "timediff", "current_date", "current_time", "current_timestamp"}
NOT_FUNCTIONS = {
    "in", "exists", "over", "values", "as", "from", "join", "on", "and", "or", "not", "where", "select", "using",
    "when", "then", "else", "case", "cast", "with", "by", "is", "filter", "like", "glob", "between", "union",
    "intersect", "except", "all", "distinct", "having", "limit", "offset", "into", "table", "recursive",
}


def function_feature(name: str) -> FeatureSpec:
    """Feature for a scalar function that has no dedicated catalogue entry."""
    lname = name.lower()
    if lname in DATE_FUNCS:
        return FeatureSpec(f"fn:{lname}", f"date/time function {lname}()",
                           f"{lname}() date and time function in SQLite: what it returns and its format string",
                           (lname + "(", lname))
    if lname.startswith("json"):
        return FeatureSpec(f"fn:{lname}", f"JSON function {lname}()",
                           f"{lname}() JSON function: extracts or builds JSON values", (lname,))
    return FeatureSpec(f"fn:{lname}", f"function {lname}()",
                       f"{lname}() built-in scalar SQL function: what it returns", (lname + "(", lname))


def feature_spec(key: str) -> FeatureSpec:
    if key in CATALOGUE:
        return CATALOGUE[key]
    if key.startswith("fn:"):
        return function_feature(key[3:])
    raise KeyError(key)


# ---------------------------------------------------------------------------
# Masking (strings, quoted identifiers, comments)
# ---------------------------------------------------------------------------
def mask_sql(sql: str) -> str:
    """Blank out string literals, quoted identifiers and comments (same length, newlines kept).

    Keeps keyword/function detection from matching text such as 'count(' that
    only appears inside a string, and keeps semicolons in strings from
    splitting statements.
    """
    out = list(sql)
    i, n = 0, len(sql)

    def blank(a: int, b: int) -> None:
        for k in range(a, b):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        if sql.startswith("--", i):
            j = sql.find("\n", i)
            j = n if j == -1 else j
            blank(i, j)
            i = j
        elif sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            j = n if j == -1 else j + 2
            blank(i, j)
            i = j
        elif sql[i] in "'\"`[":
            close = "]" if sql[i] == "[" else sql[i]
            j = i + 1
            while j < n:
                if sql[j] == close:
                    if close != "]" and j + 1 < n and sql[j + 1] == close:  # doubled quote escape
                        j += 2
                        continue
                    break
                j += 1
            j = min(j + 1, n)
            blank(i, j)
            i = j
        else:
            i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
@dataclass
class SQLAnalysis:
    sql: str
    parsed: bool
    parse_error: str | None = None
    features: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    joins: list[dict] = field(default_factory=list)
    order_by: list[dict] = field(default_factory=list)
    limit: str | None = None
    offset: str | None = None
    group_by: list[str] = field(default_factory=list)
    ctes: list[str] = field(default_factory=list)
    window_specs: list[dict] = field(default_factory=list)
    statement_type: str = "select"

    def specs(self) -> list[FeatureSpec]:
        return [feature_spec(k) for k in self.features]

    def as_dict(self) -> dict:
        return {
            "parsed": self.parsed,
            "parse_error": self.parse_error,
            "features": [{"key": s.key, "label": s.label, "requires_evidence": s.requires_evidence} for s in self.specs()],
            "tables": self.tables,
            "columns": self.columns,
            "functions": self.functions,
            "joins": self.joins,
            "order_by": self.order_by,
            "limit": self.limit,
            "offset": self.offset,
            "group_by": self.group_by,
            "ctes": self.ctes,
            "window_specs": self.window_specs,
        }

    def inventory(self) -> str:
        """Short factual list of operations, used in the judge prompt."""
        lines = []
        if self.tables:
            lines.append("Tables read: " + ", ".join(self.tables))
        for j in self.joins:
            cond = f" on {j['on']}" if j.get("on") else ""
            lines.append(f"{j['type']} with {j['table']}{cond}")
        for k in self.features:
            if k in {"inner_join", "left_join", "right_join", "full_join", "cross_join", "order_by", "limit", "offset", "group_by"}:
                continue
            lines.append(feature_spec(k).label)
        if self.group_by:
            lines.append("GROUP BY " + ", ".join(self.group_by))
        if self.order_by:
            lines.append("Final ORDER BY " + ", ".join(f"{o['expr']} {'DESC' if o['desc'] else 'ASC'}" for o in self.order_by))
        if self.limit:
            lines.append(f"LIMIT {self.limit}" + (f" OFFSET {self.offset}" if self.offset else ""))
        return "\n".join(f"- {line}" for line in lines) if lines else "- simple SELECT ... FROM"


_FUNC_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _regex_features(masked: str) -> set[str]:
    """Fallback detector used when SQLGlot cannot parse the query."""
    up = " ".join(masked.upper().split())
    feats = set()
    checks = [
        ("left_join", r"\bLEFT\s+(OUTER\s+)?JOIN\b"), ("right_join", r"\bRIGHT\s+(OUTER\s+)?JOIN\b"),
        ("full_join", r"\bFULL\s+(OUTER\s+)?JOIN\b"), ("cross_join", r"\bCROSS\s+JOIN\b"),
        ("group_by", r"\bGROUP\s+BY\b"), ("having", r"\bHAVING\b"), ("order_by", r"\bORDER\s+BY\b"),
        ("limit", r"\bLIMIT\b"), ("offset", r"\bOFFSET\b"), ("distinct", r"\bSELECT\s+DISTINCT\b"),
        ("window", r"\bOVER\s*\("), ("partition_by", r"\bPARTITION\s+BY\b"), ("cte", r"^\s*WITH\b"),
        ("recursive_cte", r"\bWITH\s+RECURSIVE\b"), ("case", r"\bCASE\b"), ("is_not_null", r"\bIS\s+NOT\s+NULL\b"),
        ("between", r"\bBETWEEN\b"), ("like", r"\bLIKE\b"), ("glob", r"\bGLOB\b"), ("exists", r"\bEXISTS\s*\("),
        ("union", r"\bUNION\b"), ("intersect", r"\bINTERSECT\b"), ("except", r"\bEXCEPT\b"), ("where", r"\bWHERE\b"),
        ("not_in", r"\bNOT\s+IN\s*\("), ("cast", r"\bCAST\s*\("),
    ]
    for key, pat in checks:
        if re.search(pat, up):
            feats.add(key)
    if re.search(r"\bIS\s+NULL\b", up):
        feats.add("is_null")
    if re.search(r"\b(INNER\s+)?JOIN\b", up) and not feats & {"left_join", "right_join", "full_join", "cross_join"}:
        feats.add("inner_join")
    if re.search(r"\bIN\s*\(\s*(SELECT|WITH)\b", up):
        feats.add("in_subquery")
    elif re.search(r"\bIN\s*\(", up):
        feats.add("in_list")
    if re.search(r"\(\s*SELECT\b", up) and not feats & {"in_subquery", "exists"}:
        feats.add("scalar_subquery")
    return feats


def analyze_sql(sql: str) -> SQLAnalysis:
    import sqlglot
    from sqlglot import exp

    masked = mask_sql(sql)
    funcs: list[str] = []
    for name in _FUNC_CALL.findall(masked):
        lname = name.lower()
        if lname not in NOT_FUNCTIONS and lname not in funcs:
            funcs.append(lname)

    analysis = SQLAnalysis(sql=sql, parsed=False)
    feats: set[str] = set()
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        analysis.parsed = True
    except Exception as exc:  # sqlglot.errors.ParseError and friends
        analysis.parse_error = str(exc).split("\n")[0][:300]
        tree = None

    if tree is not None:
        analysis.statement_type = tree.key
        cte_names = [c.alias for c in tree.find_all(exp.CTE)]
        analysis.ctes = cte_names
        funcs = [f for f in funcs if f not in {c.lower() for c in cte_names}]
        if cte_names:
            feats.add("cte")
            w = tree.find(exp.With)
            if w is not None and w.args.get("recursive"):
                feats.add("recursive_cte")
        tables = []
        for t in tree.find_all(exp.Table):
            if t.name and t.name not in tables and t.name not in cte_names:
                tables.append(t.name)
        analysis.tables = tables
        analysis.columns = sorted({c.name for c in tree.find_all(exp.Column) if c.name})
        if any(isinstance(s, exp.Star) for s in tree.find_all(exp.Star)) and not any(
            isinstance(s.parent, exp.Count) for s in tree.find_all(exp.Star)
        ):
            feats.add("select_star")
        joins = list(tree.find_all(exp.Join))
        table_counts: dict[str, int] = {}
        for t in tree.find_all(exp.Table):
            table_counts[t.name] = table_counts.get(t.name, 0) + 1
        for j in joins:
            side = (j.side or "").upper()
            kind = (j.kind or "").upper()
            if side == "LEFT":
                jt, key = "LEFT JOIN", "left_join"
            elif side == "RIGHT":
                jt, key = "RIGHT JOIN", "right_join"
            elif side == "FULL":
                jt, key = "FULL OUTER JOIN", "full_join"
            elif kind == "CROSS" or (not j.args.get("on") and not j.args.get("using")):
                jt, key = "CROSS JOIN", "cross_join"
            else:
                jt, key = "INNER JOIN", "inner_join"
            feats.add(key)
            on = j.args.get("on")
            analysis.joins.append({"type": jt, "table": j.this.sql(dialect="sqlite"), "on": on.sql(dialect="sqlite") if on else None})
        if len(joins) >= 2:
            feats.add("multi_join")
        if any(v > 1 for v in table_counts.values()) and joins:
            feats.add("self_join")
        if tree.find(exp.Where) is not None:
            feats.add("where")
        group = tree.find(exp.Group)
        if group is not None:
            feats.add("group_by")
            analysis.group_by = [e.sql(dialect="sqlite") for e in group.expressions]
        if tree.find(exp.Having) is not None:
            feats.add("having")
        # DISTINCT: on SELECT vs inside COUNT(DISTINCT x)
        for d in tree.find_all(exp.Distinct):
            if isinstance(d.parent, exp.Select):
                feats.add("distinct")
            elif isinstance(d.parent, exp.AggFunc) and isinstance(d.parent, exp.Count):
                feats.add("count_distinct")
        # Windows
        windowed_aggs: set[int] = set()
        for w in tree.find_all(exp.Window):
            feats.add("window")
            fn = w.this
            fname = (fn.sql_name() if isinstance(fn, exp.Func) else fn.sql()).lower()
            windowed_aggs.add(id(fn))
            partition = [p.sql(dialect="sqlite") for p in (w.args.get("partition_by") or [])]
            order = w.args.get("order")
            if partition:
                feats.add("partition_by")
            if order is not None:
                feats.add("window_order")
            if w.args.get("spec") is not None:
                feats.add("window_frame")
            if fname in WINDOW_FUNCS:
                feats.add(WINDOW_FUNCS[fname])
            elif isinstance(fn, exp.AggFunc):
                feats.add("window_aggregate")
            analysis.window_specs.append({
                "function": fn.sql(dialect="sqlite"),
                "partition_by": partition,
                "order_by": order.sql(dialect="sqlite") if order is not None else None,
            })
        # Aggregates (not used as window functions)
        for agg in tree.find_all(exp.AggFunc):
            if id(agg) in windowed_aggs:
                continue
            name = agg.sql_name().lower()
            if name in {"count", "sum", "avg", "min", "max", "group_concat", "total"}:
                feats.add(name)
            elif name in {"string_agg", "groupconcat"}:
                feats.add("group_concat")
        # Final ORDER BY / LIMIT / OFFSET belong to the outermost query.
        order = tree.args.get("order")
        if order is not None:
            feats.add("order_by")
            analysis.order_by = [{"expr": o.this.sql(dialect="sqlite"), "desc": bool(o.args.get("desc"))} for o in order.expressions]
        elif any(isinstance(o.parent, exp.Select) for o in tree.find_all(exp.Order)):
            feats.add("order_by")
        limit = tree.args.get("limit")
        if limit is not None:
            feats.add("limit")
            expr = limit.args.get("expression")
            analysis.limit = expr.sql(dialect="sqlite") if expr is not None else limit.sql(dialect="sqlite")
        elif tree.find(exp.Limit) is not None:
            feats.add("limit")
        off = tree.args.get("offset")
        if off is not None:
            feats.add("offset")
            expr = off.args.get("expression")
            analysis.offset = expr.sql(dialect="sqlite") if expr is not None else off.sql(dialect="sqlite")
        # Predicates and expressions
        for node in tree.find_all(exp.Is):
            if isinstance(node.expression, exp.Null):
                feats.add("is_not_null" if isinstance(node.parent, exp.Not) else "is_null")
        for node in tree.find_all(exp.In):
            negated = isinstance(node.parent, exp.Not)
            if node.args.get("query") is not None:
                feats.add("in_subquery")
            else:
                feats.add("in_list")
            if negated:
                feats.add("not_in")
        if tree.find(exp.Exists) is not None:
            feats.add("exists")
        if tree.find(exp.Between) is not None:
            feats.add("between")
        if tree.find(exp.Like) is not None or tree.find(exp.ILike) is not None:
            feats.add("like")
        if hasattr(exp, "Glob") and tree.find(exp.Glob) is not None:
            feats.add("glob")
        if tree.find(exp.Case) is not None or tree.find(exp.If) is not None:
            feats.add("case")
        if tree.find(exp.Cast) is not None or tree.find(exp.TryCast) is not None:
            feats.add("cast")
        if tree.find(exp.DPipe) is not None:
            feats.add("string_concat")
        if any(tree.find(t) is not None for t in (exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Mod)):
            feats.add("arithmetic")
        for u, key in ((exp.Union, "union"), (exp.Intersect, "intersect"), (exp.Except, "except")):
            if tree.find(u) is not None or isinstance(tree, u):
                feats.add(key)
        # Subqueries: in FROM (derived table) vs scalar in expressions
        for sq in tree.find_all(exp.Subquery):
            if isinstance(sq.parent, (exp.From, exp.Join)):
                feats.add("derived_table")
            elif not isinstance(sq.parent, (exp.In, exp.Exists)) and not isinstance(sq.parent, exp.CTE):
                feats.add("scalar_subquery")
        if "in_subquery" not in feats:
            for node in tree.find_all(exp.In):
                if node.args.get("query") is not None:
                    feats.add("in_subquery")
    else:
        feats = _regex_features(masked)
        analysis.tables = sorted(set(re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", masked, re.IGNORECASE)))

    # Scalar functions that are not covered by a dedicated feature.
    covered_names = AGGREGATE_FUNCS | set(WINDOW_FUNCS) | {"cast"}
    analysis.functions = funcs
    for f in funcs:
        if f in covered_names:
            continue
        feats.add(f"fn:{f}")

    order_index = {k: i for i, k in enumerate(CATALOGUE)}
    analysis.features = sorted(feats, key=lambda k: (order_index.get(k, len(order_index)), k))
    return analysis


# ---------------------------------------------------------------------------
# Retrieval queries
# ---------------------------------------------------------------------------
@dataclass
class RetrievalQuery:
    feature: str | None   # None for the whole-query probe
    text: str


def build_retrieval_queries(analysis: SQLAnalysis, *, max_queries: int = 10) -> list[RetrievalQuery]:
    """One focused query per evidence-requiring feature, plus a whole-query probe.

    Interactions get a combined query because their meaning is more than the
    sum of the parts, e.g. LEFT JOIN + IS NULL is the "rows without a match"
    pattern, and NOT IN + subquery has NULL pitfalls.
    """
    feats = set(analysis.features)
    queries: list[RetrievalQuery] = []
    if {"left_join", "is_null"} <= feats:
        queries.append(RetrievalQuery("left_join", "LEFT OUTER JOIN unmatched rows: right table columns are NULL when no row matches; WHERE right.column IS NULL keeps only rows without a match"))
    if {"group_by", "having"} <= feats:
        queries.append(RetrievalQuery("having", "GROUP BY with HAVING: the HAVING expression is evaluated once per group after aggregate functions and groups that fail it are removed"))
    if "window" in feats and "partition_by" in feats:
        queries.append(RetrievalQuery("partition_by", "window function PARTITION BY and ORDER BY in the OVER clause: the function is computed for each row within its partition in the given order"))
    have = {q.feature for q in queries}
    for spec in analysis.specs():
        if not spec.requires_evidence or spec.key in have:
            continue
        queries.append(RetrievalQuery(spec.key, spec.query))
    labels = [s.label for s in analysis.specs()]
    probe = "SQLite SELECT query using " + ", ".join(labels) if labels else "SQLite SELECT statement result rows"
    queries.append(RetrievalQuery(None, probe))
    return queries[:max_queries] if len(queries) <= max_queries else queries[: max_queries - 1] + [queries[-1]]


# ---------------------------------------------------------------------------
# Safe validation of user-supplied SQL and schema
# ---------------------------------------------------------------------------
class SQLValidationError(ValueError):
    pass


_ALLOWED_SCHEMA = re.compile(r"^\s*CREATE\s+(TEMP\s+|TEMPORARY\s+)?(TABLE|VIEW|INDEX|UNIQUE\s+INDEX|VIRTUAL\s+TABLE)\b", re.IGNORECASE)


def _split_statements(script: str) -> list[str]:
    """Split on semicolons that are outside strings and comments."""
    masked = mask_sql(script)  # same length as script, so offsets line up
    stmts, start = [], 0
    for pos, ch in enumerate(masked):
        if ch == ";":
            if masked[start:pos].strip():
                stmts.append(script[start:pos].strip())
            start = pos + 1
    if masked[start:].strip():
        stmts.append(script[start:].strip())
    return stmts


def validate_sql(sql: str, schema: str, *, max_sql_chars: int = 10_000, max_schema_chars: int = 20_000) -> None:
    """Check that `sql` is a single read-only query that compiles against `schema`.

    Nothing is executed against real data: the schema is created in a private
    in-memory database, the query is only compiled (EXPLAIN), and an
    authorizer denies anything except reading (no ATTACH, PRAGMA, writes).
    """
    if len(sql) > max_sql_chars:
        raise SQLValidationError(f"SQL is longer than {max_sql_chars} characters")
    if len(schema) > max_schema_chars:
        raise SQLValidationError(f"schema is longer than {max_schema_chars} characters")
    statements = _split_statements(sql)
    if len(statements) != 1:
        raise SQLValidationError("submit exactly one SQL statement")
    head = mask_sql(statements[0]).lstrip("( \n\t").upper()
    if not (head.startswith("SELECT") or head.startswith("WITH") or head.startswith("VALUES")):
        raise SQLValidationError("only read-only SELECT / WITH queries can be explained")
    schema_statements = _split_statements(schema)
    for st in schema_statements:
        masked = mask_sql(st)
        if not _ALLOWED_SCHEMA.match(masked):
            raise SQLValidationError("the schema may only contain CREATE TABLE / VIEW / INDEX statements")
    con = sqlite3.connect(":memory:")
    try:
        con.execute("PRAGMA trusted_schema = OFF")
        denied = {sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH, sqlite3.SQLITE_PRAGMA}

        def schema_auth(action, *_args):
            return sqlite3.SQLITE_DENY if action in denied else sqlite3.SQLITE_OK

        con.set_authorizer(schema_auth)
        try:
            for st in schema_statements:
                con.execute(st)
        except sqlite3.Error as exc:
            raise SQLValidationError(f"schema error: {exc}") from exc
        read_only = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, getattr(sqlite3, "SQLITE_RECURSIVE", 33)}

        def query_auth(action, *_args):
            return sqlite3.SQLITE_OK if action in read_only else sqlite3.SQLITE_DENY

        con.set_authorizer(query_auth)
        try:
            con.execute("EXPLAIN " + statements[0])
        except sqlite3.Error as exc:
            raise SQLValidationError(f"SQL error: {exc}") from exc
    finally:
        con.close()
