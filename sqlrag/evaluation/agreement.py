"""Agreement between the LLM judge and reference labels.

Reference labels live in data/labels/*.jsonl, one JSON object per line:

  {"exp_id": "...", "item_id": "q141", "condition": "baseline" | "rag",
   "explanation_sha256_12": "<hash of the exact text that was labelled>",
   "label": "good" | "bad",
   "label_source": "human" | "ai_reference",
   "annotator": "...", "labelled_at": "...", "notes": "..."}

`label_source` is never mixed: metrics are reported separately for genuine
human labels and for AI-assisted reference labels. A label only counts when
its explanation hash matches the stored explanation, so a label can never be
silently applied to different text.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from collections import defaultdict

from ..config import ROOT
from .experiment import read_jsonl
from .stats import binary_classification

LABEL_SOURCES = ("human", "ai_reference")


def text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]


def load_labels(labels_dir: pathlib.Path) -> list[dict]:
    rows = []
    for path in sorted(pathlib.Path(labels_dir).glob("*.jsonl")):
        for r in read_jsonl(path):
            r["_file"] = path.name
            rows.append(r)
    return rows


def reference_comparison(labels: list[dict], results_root: pathlib.Path) -> dict:
    """Baseline vs RAG measured with reference labels instead of the LLM judge.

    Only labels whose text hash matches the stored explanation are used. Results
    are kept separate per label source and per experiment.
    """
    from .stats import mcnemar_exact, paired_bootstrap, wilson_interval

    out: dict = {}
    for src in LABEL_SOURCES:
        by_exp: dict[str, dict] = {}
        for exp_id in sorted({lab["exp_id"] for lab in labels if lab.get("label_source") == src}):
            gens = {(g["id"], g["condition"]): g for g in read_jsonl(results_root / "rag" / exp_id / "generations.jsonl") if not g.get("error")}
            valid: dict[tuple[str, str], int] = {}
            for lab in labels:
                if lab.get("label_source") != src or lab["exp_id"] != exp_id:
                    continue
                key = (lab["item_id"], lab["condition"])
                g = gens.get(key)
                if g and text_hash(g.get("explanation_clean") or "") == lab.get("explanation_sha256_12"):
                    valid[key] = int(lab["label"] == "good")
            conds = {}
            for cond in ("baseline", "rag"):
                vals = [v for (i, c), v in valid.items() if c == cond]
                lo, hi = wilson_interval(sum(vals), len(vals))
                conds[cond] = {"good": sum(vals), "n": len(vals), "rate": (sum(vals) / len(vals)) if vals else None, "ci95": [lo, hi]}
            items = sorted({i for (i, c) in valid if (i, "baseline") in valid and (i, "rag") in valid})
            b = sum(1 for i in items if valid[(i, "baseline")] == 1 and valid[(i, "rag")] == 0)
            c = sum(1 for i in items if valid[(i, "baseline")] == 0 and valid[(i, "rag")] == 1)
            by_exp[exp_id] = {
                "conditions": conds,
                "n_pairs": len(items),
                "only_baseline_good": b,
                "only_rag_good": c,
                "both_good": sum(1 for i in items if valid[(i, "baseline")] == 1 and valid[(i, "rag")] == 1),
                "both_bad": sum(1 for i in items if valid[(i, "baseline")] == 0 and valid[(i, "rag")] == 0),
                "delta_rag_minus_baseline": paired_bootstrap([float(valid[(i, "rag")] - valid[(i, "baseline")]) for i in items]) if items else None,
                "mcnemar_exact_p": mcnemar_exact(b, c),
            }
        out[src] = by_exp
    return out


def compute_agreement(labels_dir: pathlib.Path, results_root: pathlib.Path) -> dict:
    labels = load_labels(labels_dir)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for lab in labels:
        src = lab.get("label_source")
        if src not in LABEL_SOURCES:
            continue
        by_source[src].append(lab)

    report = {
        "description": "LLM-judge agreement with reference labels. Positive class = good.",
        "label_counts": {s: len(by_source.get(s, [])) for s in LABEL_SOURCES},
        "sources": {},
    }
    for src in LABEL_SOURCES:
        labs = by_source.get(src, [])
        src_report = {"n_labels": len(labs), "label_distribution": {}, "judges": {}, "hash_mismatches": 0}
        if not labs:
            src_report["note"] = "no labels of this kind yet"
            report["sources"][src] = src_report
            continue
        dist = defaultdict(int)
        for lab in labs:
            dist[lab["label"]] += 1
        src_report["label_distribution"] = dict(dist)
        exp_ids = sorted({lab["exp_id"] for lab in labs})
        per_judge_pairs: dict[str, dict[str, list]] = defaultdict(lambda: {"true": [], "pred": [], "by_condition": defaultdict(lambda: {"true": [], "pred": []})})
        for exp_id in exp_ids:
            exp_dir = results_root / "rag" / exp_id
            gens = {(g["id"], g["condition"]): g for g in read_jsonl(exp_dir / "generations.jsonl") if not g.get("error")}
            for jpath in sorted(exp_dir.glob("judgments__*.jsonl")):
                tag = jpath.stem.split("__", 1)[1]
                judg = {(j["id"], j["condition"]): j for j in read_jsonl(jpath) if j.get("score") is not None}
                for lab in labs:
                    if lab["exp_id"] != exp_id:
                        continue
                    key = (lab["item_id"], lab["condition"])
                    g = gens.get(key)
                    if g is None or key not in judg:
                        continue
                    if text_hash(g.get("explanation_clean") or "") != lab.get("explanation_sha256_12"):
                        src_report["hash_mismatches"] += 1
                        continue
                    t = int(lab["label"] == "good")
                    p = judg[key]["score"]
                    bucket = per_judge_pairs[f"{exp_id}/{tag}"]
                    bucket["true"].append(t)
                    bucket["pred"].append(p)
                    bucket["by_condition"][lab["condition"]]["true"].append(t)
                    bucket["by_condition"][lab["condition"]]["pred"].append(p)
        for tag, d in per_judge_pairs.items():
            entry = binary_classification(d["true"], d["pred"])
            entry["by_condition"] = {c: binary_classification(v["true"], v["pred"]) for c, v in d["by_condition"].items()}
            entry["reference_good_rate"] = sum(d["true"]) / len(d["true"]) if d["true"] else None
            entry["judge_good_rate"] = sum(d["pred"]) / len(d["pred"]) if d["pred"] else None
            src_report["judges"][tag] = entry
        report["sources"][src] = src_report
    report["reference_comparison"] = reference_comparison(labels, results_root)
    return report


def main(args) -> int:
    report = compute_agreement(pathlib.Path(args.labels), ROOT / "results")
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({s: {t: {k: v for k, v in m.items() if k in ("n", "accuracy", "cohen_kappa", "f1_good", "recall_bad")}
                          for t, m in r.get("judges", {}).items()} for s, r in report["sources"].items()}, indent=2))
    print(f"wrote {out}")
    return 0
