"""Fair side-by-side benchmark for two local Ollama models.

Example:
  python harness/compare.py dev prompts/v2.md qwen_vs_llama \
    --model-a qwen2.5:7b --model-b llama3.1:8b --judge-model qwen2.5:7b

Each output JSONL row contains both explanations and both judge verdicts for the
same dataset item. "Judge pass rate" is deliberately used instead of "accuracy"
unless external human/golden validation is added.
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

from llm import call_json
from schema import Explanation, Verdict
from sql_features import detect_categories, operation_inventory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two Ollama models fairly on SQL explanations.")
    parser.add_argument("split", choices=("tiny", "dev", "test", "all"))
    parser.add_argument("prompt_file", type=pathlib.Path)
    parser.add_argument("run_id")
    parser.add_argument("--model-a", default="qwen2.5:7b", help="First generator model.")
    parser.add_argument("--model-b", default="llama3.1:8b", help="Second generator model.")
    parser.add_argument("--judge-model", default="qwen2.5:7b", help="Single shared judge model.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Items processed concurrently. Default 1 avoids GPU contention during latency comparison.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-predict", type=int, default=3000)
    parser.add_argument("--judge-prompt", type=pathlib.Path, default=pathlib.Path("prompts/judge.md"))
    args = parser.parse_args()
    for p, label in ((args.prompt_file, "prompt file"), (args.judge_prompt, "judge prompt")):
        if not p.exists():
            parser.error(f"{label} does not exist: {p}")
    return args


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def outcome(score_a: int, score_b: int) -> str:
    if score_a and score_b:
        return "both_correct"
    if score_a:
        return "only_a_correct"
    if score_b:
        return "only_b_correct"
    return "both_wrong"


def dimension_dict(v: Verdict) -> dict[str, bool]:
    return {
        "correctness": v.correctness,
        "completeness": v.completeness,
        "hallucination_free": v.hallucination_free,
        "clarity": v.clarity,
    }


def main() -> int:
    args = parse_args()
    prompt = args.prompt_file.read_text(encoding="utf-8")
    judge_prompt = args.judge_prompt.read_text(encoding="utf-8")
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    judge_prompt_sha256 = hashlib.sha256(judge_prompt.encode("utf-8")).hexdigest()
    input_path = pathlib.Path(f"data/{args.split}.jsonl")
    if not input_path.exists():
        raise SystemExit(f"dataset does not exist: {input_path}")
    items = load_jsonl(input_path)

    out_path = pathlib.Path(f"results/{args.run_id}_{args.split}.jsonl")
    out_path.parent.mkdir(exist_ok=True)
    completed: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        for row in load_jsonl(out_path):
            if row.get("benchmark_version") == 2:
                completed[row["id"]] = row
        if completed:
            logger.info("Resuming comparison — %d/%d items already complete.", len(completed), len(items))

    remaining = [item for item in items if item["id"] not in completed]
    logger.info(
        "Comparing %s vs %s | judge=%s | split=%s | prompt=%s | temp=%s seed=%s",
        args.model_a, args.model_b, args.judge_model, args.split, args.prompt_file, args.temperature, args.seed,
    )

    def generate(model: str, item: dict[str, Any]) -> tuple[Explanation, dict]:
        user = f"Schema:\n{item['schema']}\n\nSQL:\n{item['sql']}"
        return call_json(
            model, prompt, user, Explanation,
            temperature=args.temperature,
            seed=args.seed,
            num_predict=args.num_predict,
        )

    def judge(item: dict[str, Any], explanation: str) -> tuple[Verdict, dict]:
        inventory = operation_inventory(item["sql"])
        user = (
            f"Schema:\n{item['schema']}\n\n"
            f"SQL:\n{item['sql']}\n\n"
            f"Operations actually present:\n{inventory}\n\n"
            f"Explanation to evaluate:\n{explanation}"
        )
        return call_json(
            args.judge_model, judge_prompt, user, Verdict,
            temperature=0.0,
            seed=args.seed,
            num_predict=1200,
        )

    def process_item(item: dict[str, Any]) -> dict[str, Any]:
        base: dict[str, Any] = {
            "benchmark_version": 2,
            "id": item["id"],
            "domain": item.get("domain", "unknown"),
            "dataset_feature": item.get("feature", "unknown"),
            "categories": detect_categories(item["sql"]),
            "schema": item["schema"],
            "sql": item["sql"],
            "prompt_file": str(args.prompt_file),
            "prompt_sha256": prompt_sha256,
            "judge_prompt_file": str(args.judge_prompt),
            "judge_prompt_sha256": judge_prompt_sha256,
            "settings": {
                "temperature": args.temperature,
                "seed": args.seed,
                "num_predict": args.num_predict,
            },
            "model_a_name": args.model_a,
            "model_b_name": args.model_b,
            "judge_model": args.judge_model,
            "error": None,
        }
        try:
            # Alternate generation order to reduce systematic first/second effects.
            # Both models still receive identical prompt/context/settings.
            if sum(ord(c) for c in item["id"]) % 2 == 0:
                a_out, a_cost = generate(args.model_a, item)
                b_out, b_cost = generate(args.model_b, item)
            else:
                b_out, b_cost = generate(args.model_b, item)
                a_out, a_cost = generate(args.model_a, item)

            a_verdict, a_judge_cost = judge(item, a_out.explanation)
            b_verdict, b_judge_cost = judge(item, b_out.explanation)
            a_dims = dimension_dict(a_verdict)
            b_dims = dimension_dict(b_verdict)
            # Enforce the rubric deterministically: overall pass requires all four dimensions.
            # This prevents an internally inconsistent judge JSON from changing the benchmark result.
            score_a = int(all(a_dims.values()))
            score_b = int(all(b_dims.values()))
            base.update({
                "model_a": {
                    "explanation": a_out.explanation,
                    "score": score_a,
                    "dimensions": a_dims,
                    "judge_reported_grade": a_verdict.grade,
                    "judge_grade_consistent": (a_verdict.grade == "good") == bool(score_a),
                    "judge_reason": a_verdict.reason,
                    "generation_cost": a_cost,
                    "judge_cost": a_judge_cost,
                },
                "model_b": {
                    "explanation": b_out.explanation,
                    "score": score_b,
                    "dimensions": b_dims,
                    "judge_reported_grade": b_verdict.grade,
                    "judge_grade_consistent": (b_verdict.grade == "good") == bool(score_b),
                    "judge_reason": b_verdict.reason,
                    "generation_cost": b_cost,
                    "judge_cost": b_judge_cost,
                },
                "outcome": outcome(score_a, score_b),
                "disagreement": score_a != score_b,
            })
        except Exception as exc:
            logger.exception("Item %s failed", item["id"])
            base["error"] = str(exc)
        return base

    all_rows = list(completed.values())
    with out_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(process_item, item): item for item in remaining}
            with tqdm(total=len(remaining), unit="item", desc=f"{args.run_id}/{args.split}") as bar:
                for future in as_completed(futures):
                    row = future.result()
                    all_rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    bar.set_postfix(id=row["id"], outcome=row.get("outcome", "error"))
                    bar.update(1)

    valid = [r for r in all_rows if r.get("error") is None and r.get("model_a") and r.get("model_b")]
    if valid:
        a_pass = sum(r["model_a"]["score"] for r in valid)
        b_pass = sum(r["model_b"]["score"] for r in valid)
        outcomes: dict[str, int] = {}
        for row in valid:
            outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
        logger.info("Judge pass rate — A %d/%d (%.1f%%), B %d/%d (%.1f%%)",
                    a_pass, len(valid), 100*a_pass/len(valid), b_pass, len(valid), 100*b_pass/len(valid))
        logger.info("Pair outcomes: %s", outcomes)
    errors = [r for r in all_rows if r.get("error")]
    if errors:
        logger.warning("%d items had errors: %s", len(errors), [r["id"] for r in errors])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
