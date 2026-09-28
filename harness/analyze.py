"""Analyze side-by-side benchmark results.

Usage:
  python harness/analyze.py results/qwen_vs_llama_dev.jsonl

Important terminology: without human/golden validation, this tool reports
"judge pass rate", not true accuracy.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter, defaultdict
from statistics import mean


def load(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pct(n: int, d: int) -> str:
    return f"{100*n/d:.1f}%" if d else "—"


def avg(values: list[float]) -> str:
    return f"{mean(values):.2f}" if values else "—"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def summarize(rows: list[dict]) -> dict:
    valid = [r for r in rows if not r.get("error") and r.get("model_a") and r.get("model_b")]
    if not valid:
        return {"valid": []}
    a_name = valid[0].get("model_a_name", "Model A")
    b_name = valid[0].get("model_b_name", "Model B")
    dims = ["correctness", "completeness", "hallucination_free", "clarity"]
    summary = {
        "valid": valid,
        "a_name": a_name,
        "b_name": b_name,
        "a_pass": sum(r["model_a"]["score"] for r in valid),
        "b_pass": sum(r["model_b"]["score"] for r in valid),
        "outcomes": Counter(r["outcome"] for r in valid),
        "a_dims": {d: sum(bool(r["model_a"]["dimensions"].get(d)) for r in valid) for d in dims},
        "b_dims": {d: sum(bool(r["model_b"]["dimensions"].get(d)) for r in valid) for d in dims},
        "a_lat": [r["model_a"]["generation_cost"]["latency"] for r in valid if r["model_a"].get("generation_cost")],
        "b_lat": [r["model_b"]["generation_cost"]["latency"] for r in valid if r["model_b"].get("generation_cost")],
        "a_in": [r["model_a"]["generation_cost"]["in_tok"] for r in valid if r["model_a"].get("generation_cost")],
        "b_in": [r["model_b"]["generation_cost"]["in_tok"] for r in valid if r["model_b"].get("generation_cost")],
        "a_out": [r["model_a"]["generation_cost"]["out_tok"] for r in valid if r["model_a"].get("generation_cost")],
        "b_out": [r["model_b"]["generation_cost"]["out_tok"] for r in valid if r["model_b"].get("generation_cost")],
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze a two-model SQL explanation benchmark.")
    ap.add_argument("results", type=pathlib.Path)
    args = ap.parse_args()
    rows = load(args.results)
    s = summarize(rows)
    valid = s["valid"]
    if not valid:
        print("No valid comparison rows found.")
        return

    n = len(valid)
    print(f"\nComparison: {s['a_name']} vs {s['b_name']}")
    print(f"Valid items: {n} | Errors: {len(rows)-n}")
    print("Metric note: pass percentages below are judge pass rates, not true accuracy unless validated against human/golden labels.\n")

    print_table(
        ["Metric", s["a_name"], s["b_name"]],
        [
            ["Judge pass rate", pct(s["a_pass"], n), pct(s["b_pass"], n)],
            ["Correctness", pct(s["a_dims"]["correctness"], n), pct(s["b_dims"]["correctness"], n)],
            ["Completeness", pct(s["a_dims"]["completeness"], n), pct(s["b_dims"]["completeness"], n)],
            ["Hallucination-free", pct(s["a_dims"]["hallucination_free"], n), pct(s["b_dims"]["hallucination_free"], n)],
            ["Clarity", pct(s["a_dims"]["clarity"], n), pct(s["b_dims"]["clarity"], n)],
            ["Avg generation latency (s)", avg(s["a_lat"]), avg(s["b_lat"])],
            ["Avg input tokens", avg(s["a_in"]), avg(s["b_in"])],
            ["Avg output tokens", avg(s["a_out"]), avg(s["b_out"])],
        ],
    )

    print("\nPair outcomes")
    for key in ("both_correct", "only_a_correct", "only_b_correct", "both_wrong"):
        print(f"  {key:18s} {s['outcomes'].get(key,0):4d}  ({pct(s['outcomes'].get(key,0), n)})")
    print(f"  {'disagreement':18s} {sum(r['disagreement'] for r in valid):4d}  ({pct(sum(r['disagreement'] for r in valid), n)})")

    categories: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        for c in r.get("categories", []):
            categories[c].append(r)
    print("\nBy SQL category")
    table = []
    for category in ("JOIN", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "aggregation", "subquery", "LIMIT", "DISTINCT"):
        subset = categories.get(category, [])
        if not subset:
            continue
        a = sum(r["model_a"]["score"] for r in subset)
        b = sum(r["model_b"]["score"] for r in subset)
        dis = sum(r["disagreement"] for r in subset)
        table.append([category, str(len(subset)), pct(a, len(subset)), pct(b, len(subset)), pct(dis, len(subset))])
    print_table(["Category", "N", f"{s['a_name']} pass", f"{s['b_name']} pass", "Disagree"], table)

    disagreements = [r for r in valid if r["disagreement"]]
    if disagreements:
        print("\nDisagreement cases")
        for r in disagreements[:20]:
            print(f"  {r['id']} | {r['outcome']} | {', '.join(r.get('categories', [])) or 'basic SELECT'}")
            print(f"    A: {r['model_a']['judge_reason']}")
            print(f"    B: {r['model_b']['judge_reason']}")


if __name__ == "__main__":
    main()
