"""Small, dependency-free statistics used by the evaluation reports."""
from __future__ import annotations

import math
import random
from statistics import mean, median


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float | None:
    """Two-sided exact McNemar test on discordant pairs (b = only A passes, c = only B passes)."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_bootstrap(diffs: list[float], *, iterations: int = 5000, seed: int = 42) -> dict:
    """Mean of paired differences with a 95% percentile bootstrap interval."""
    if not diffs:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = random.Random(seed)
    n = len(diffs)
    samples = sorted(mean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(iterations))
    return {
        "mean": mean(diffs),
        "ci_low": samples[int(0.025 * (iterations - 1))],
        "ci_high": samples[int(0.975 * (iterations - 1))],
        "n": n,
    }


def describe(values: list[float]) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"n": len(vals), "mean": mean(vals), "median": median(vals), "min": min(vals), "max": max(vals)}


def binary_classification(y_true: list[int], y_pred: list[int]) -> dict:
    """Metrics for predicting the positive class (1 = good). Also Cohen's kappa."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else None
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is None or recall is None:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    # Precision/recall for detecting BAD explanations (the judge's job as a filter).
    bad_precision = tn / (tn + fn) if (tn + fn) else None
    bad_recall = tn / (tn + fp) if (tn + fp) else None
    p_yes = ((tp + fp) / n) * ((tp + fn) / n) if n else 0
    p_no = ((tn + fn) / n) * ((tn + fp) / n) if n else 0
    pe = p_yes + p_no
    kappa = (acc - pe) / (1 - pe) if n and pe < 1 else None
    return {
        "n": n,
        "accuracy": acc,
        "precision_good": precision,
        "recall_good": recall,
        "f1_good": f1,
        "precision_bad": bad_precision,
        "recall_bad": bad_recall,
        "cohen_kappa": kappa,
        "confusion_matrix": {"labels": ["good", "bad"], "rows_true_cols_pred": [[tp, fn], [fp, tn]]},
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
