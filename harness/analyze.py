"""Statistical analysis for benchmark-v3 two-model results."""
from __future__ import annotations

import argparse
import json
import pathlib

from metrics import DIMENSIONS, summarize


def load_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def pct(value):
    return "—" if value is None else f"{value:.1f}%"


def table(headers, rows):
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def gold_agreement(rows, gold_path):
    gold = {r["id"]: r for r in load_jsonl(gold_path)}
    valid = [r for r in rows if r.get("id") in gold and r.get("model_a", {}).get("score") is not None and r.get("model_b", {}).get("score") is not None]
    if not valid:
        return None
    out = {}
    for label, key in (("A", "model_a_score"), ("B", "model_b_score")):
        comparable = [r for r in valid if key in gold[r["id"]]]
        matches = sum(bool(r["model_" + label.lower()]["score"]) == bool(gold[r["id"]][key]) for r in comparable)
        out[label] = {
            "matches": matches,
            "n": len(comparable),
            "agreement_pct": 100 * matches / len(comparable) if comparable else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="Analyze paired SQL explanation benchmark results.")
    ap.add_argument("results", type=pathlib.Path)
    ap.add_argument("--gold", type=pathlib.Path, help="Optional JSONL with id, model_a_score, model_b_score (0/1).")
    ap.add_argument("--json", action="store_true", help="Print machine-readable summary.")
    args = ap.parse_args()

    rows = load_jsonl(args.results)
    summary = summarize(rows)
    if args.gold:
        summary["gold_agreement"] = gold_agreement(rows, args.gold)

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if not summary["n_valid"]:
        print("No valid A/B comparison rows found.")
        return

    print(f"\n{summary['a_name']} vs {summary['b_name']}")
    print(f"Valid comparisons: {summary['n_valid']} | evaluation errors: {summary['n_errors']}")
    print("Metric note: percentages below are judge pass rates. They are not true accuracy unless externally validated against human/golden labels.\n")

    metric_rows = [
        ["Judge pass rate", pct(summary["a_pass_rate_pct"]), pct(summary["b_pass_rate_pct"])],
        *[[d.replace("_", " ").title(), pct(summary["a_dimensions_pct"][d]), pct(summary["b_dimensions_pct"][d])] for d in DIMENSIONS],
        ["Avg generation latency (s)", f"{summary['a_latency_avg_s']:.3f}", f"{summary['b_latency_avg_s']:.3f}"],
        ["Median generation latency (s)", f"{summary['a_latency_median_s']:.3f}", f"{summary['b_latency_median_s']:.3f}"],
        ["Avg input tokens*", f"{summary['a_in_tok_avg']:.1f}", f"{summary['b_in_tok_avg']:.1f}"],
        ["Avg output tokens*", f"{summary['a_out_tok_avg']:.1f}", f"{summary['b_out_tok_avg']:.1f}"],
        ["Avg output tok/s", f"{summary['a_tok_per_s_avg']:.1f}", f"{summary['b_tok_per_s_avg']:.1f}"],
    ]
    table(["Metric", summary["a_name"], summary["b_name"]], metric_rows)
    print("* Token counts are tokenizer/model-specific; compare them as observed resource metrics, not identical units across architectures.")

    boot = summary["bootstrap"]
    print(f"\nPaired difference (B − A): {boot['delta_pct']:+.1f} percentage points; 95% bootstrap CI [{boot['ci_low_pct']:+.1f}, {boot['ci_high_pct']:+.1f}]")
    print(f"Exact McNemar p-value on disagreements: {summary['mcnemar_exact_p'] if summary['mcnemar_exact_p'] is not None else '—'}")

    print("\nPair outcomes")
    for key in ("both_correct", "only_a_correct", "only_b_correct", "both_wrong", "evaluation_error"):
        n = summary["pair_counts"].get(key, 0)
        if n or key != "evaluation_error":
            print(f"  {key:18s} {n:4d} ({100*n/summary['n_valid']:.1f}% of valid)")

    print("\nBy SQL category")
    cat_rows = []
    for category, data in summary["category"].items():
        cat_rows.append([category, data["n"], pct(data["a_pass_rate_pct"]), pct(data["b_pass_rate_pct"]), pct(data["disagreement_pct"])])
    table(["Category", "N", summary["a_name"] + " pass", summary["b_name"] + " pass", "Disagree"], cat_rows)

    fairness = {r.get("benchmark_fairness_sha256") for r in rows if r.get("benchmark_fairness_sha256")}
    if len(fairness) > 1:
        print(f"\nWARNING: {len(fairness)} benchmark fairness hashes are present. Do not pool these rows into one comparison.")

    if args.gold:
        print("\nExternal gold-label agreement")
        print(json.dumps(summary["gold_agreement"], indent=2))

    disagreements = [r for r in rows if r.get("disagreement") and not r.get("error")]
    if disagreements:
        print("\nDisagreement cases")
        for r in disagreements[:25]:
            print(f"  {r['id']} | {r['outcome']} | {', '.join(r.get('categories', [])) or 'basic SELECT'}")
            print(f"    A: {r['model_a'].get('judge_reason')}")
            print(f"    B: {r['model_b'].get('judge_reason')}")


if __name__ == "__main__":
    main()
