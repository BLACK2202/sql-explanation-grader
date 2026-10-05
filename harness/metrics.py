from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

DIMENSIONS = ("correctness", "completeness", "hallucination_free", "clarity")
CATEGORIES = (
    "JOIN", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "aggregation", "window",
    "subquery", "CTE", "UNION", "INTERSECT", "EXCEPT", "LIMIT", "OFFSET", "DISTINCT",
    "CASE", "EXISTS", "IN", "BETWEEN", "LIKE", "NULL filter",
)


def valid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if not r.get("error") and r.get("model_a") and r.get("model_b")
            and r["model_a"].get("score") is not None and r["model_b"].get("score") is not None]


def percent(n: int, d: int) -> float | None:
    return 100.0 * n / d if d else None


def pair_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(r.get("outcome", "evaluation_error") for r in rows)


def paired_bootstrap_delta(rows: list[dict[str, Any]], *, iterations: int = 3000, seed: int = 42) -> dict[str, float | None]:
    """95% paired bootstrap CI for B pass-rate minus A pass-rate."""
    valid = valid_rows(rows)
    if not valid:
        return {"delta_pct": None, "ci_low_pct": None, "ci_high_pct": None}
    pairs = [(int(r["model_a"]["score"]), int(r["model_b"]["score"])) for r in valid]
    observed = 100.0 * mean(b - a for a, b in pairs)
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        samples.append(100.0 * mean(b - a for a, b in sample))
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return {"delta_pct": observed, "ci_low_pct": lo, "ci_high_pct": hi}


def mcnemar_exact_p(rows: list[dict[str, Any]]) -> float | None:
    """Two-sided exact McNemar p-value using discordant pairs only."""
    valid = valid_rows(rows)
    b = sum(r["outcome"] == "only_a_correct" for r in valid)
    c = sum(r["outcome"] == "only_b_correct" for r in valid)
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = valid_rows(rows)
    result: dict[str, Any] = {
        "n_rows": len(rows),
        "n_valid": len(valid),
        "n_errors": len(rows) - len(valid),
        "pair_counts": dict(pair_counts(valid)),
        "disagreements": sum(bool(r.get("disagreement")) for r in valid),
    }
    if not valid:
        return result
    for key, model_key in (("a", "model_a"), ("b", "model_b")):
        result[f"{key}_name"] = valid[0].get(f"model_{key}_name", f"Model {key.upper()}")
        result[f"{key}_pass_rate_pct"] = percent(sum(r[model_key]["score"] for r in valid), len(valid))
        result[f"{key}_dimensions_pct"] = {
            d: percent(sum(bool(r[model_key]["dimensions"].get(d)) for r in valid), len(valid))
            for d in DIMENSIONS
        }
        costs = [r[model_key].get("generation_cost") for r in valid if r[model_key].get("generation_cost")]
        result[f"{key}_latency_avg_s"] = mean(c["latency_s"] for c in costs) if costs else None
        result[f"{key}_latency_median_s"] = median(c["latency_s"] for c in costs) if costs else None
        result[f"{key}_out_tok_avg"] = mean(c["out_tok"] for c in costs) if costs else None
        result[f"{key}_in_tok_avg"] = mean(c["in_tok"] for c in costs) if costs else None
        result[f"{key}_tok_per_s_avg"] = mean(c["tok_per_s"] for c in costs) if costs else None
    result["bootstrap"] = paired_bootstrap_delta(valid)
    result["mcnemar_exact_p"] = mcnemar_exact_p(valid)
    result["category"] = category_summary(valid)
    return result


def category_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for category in row.get("categories", []):
            buckets[category].append(row)
    output = {}
    for category in CATEGORIES:
        subset = buckets.get(category, [])
        if not subset:
            continue
        output[category] = {
            "n": len(subset),
            "a_pass_rate_pct": percent(sum(r["model_a"]["score"] for r in subset), len(subset)),
            "b_pass_rate_pct": percent(sum(r["model_b"]["score"] for r in subset), len(subset)),
            "disagreement_pct": percent(sum(bool(r.get("disagreement")) for r in subset), len(subset)),
        }
    return output
