"""
run.py — Optimised evaluation runner for the SQL Explanation Grader.

Improvements over v1:
  - Parallel execution via ThreadPoolExecutor (--workers flag)
  - Checkpoint/resume: skips already-completed item IDs on restart
  - Per-item error isolation: failures are recorded, not fatal
  - tqdm progress bar
  - Versioned judge prompt loaded from prompts/judge.md
  - Structured logging
"""

import argparse
import json
import logging
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from llm import call_json
from schema import Explanation, Verdict
from sql_features import operation_inventory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Grade SQL explanations with two local Ollama models."
)
parser.add_argument("split", choices=("tiny", "dev", "test", "all"))
parser.add_argument("prompt_file", type=pathlib.Path)
parser.add_argument("run_id")
parser.add_argument("--system-model", default="qwen2.5:7b")
parser.add_argument("--judge-model", default="qwen2.5:3b")
parser.add_argument(
    "--workers",
    type=int,
    default=4,
    help="Number of parallel worker threads (default: 4).",
)
parser.add_argument(
    "--judge-prompt",
    type=pathlib.Path,
    default=pathlib.Path("prompts/judge.md"),
    help="Path to the judge system prompt file.",
)
parser.add_argument(
    "--no-pdf",
    action="store_true",
    help="Disable automatic PDF report generation on completion.",
)
args = parser.parse_args()

for p, label in ((args.prompt_file, "prompt file"), (args.judge_prompt, "judge prompt")):
    if not p.exists():
        parser.error(f"{label} does not exist: {p}")

SYS_MODEL = args.system_model
JUDGE_MODEL = args.judge_model
split = args.split
run_id = args.run_id

prompt = args.prompt_file.read_text(encoding="utf-8")
guide = args.judge_prompt.read_text(encoding="utf-8")
JUDGE = f"{guide}\n\nGive a grade (good or bad) and a short reason."

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
input_path = pathlib.Path(f"data/{split}.jsonl")
if not input_path.exists():
    parser.error(f"dataset does not exist: {input_path}")

items = [json.loads(line) for line in input_path.open(encoding="utf-8")]

# ---------------------------------------------------------------------------
# Checkpoint: load already-completed IDs so we can resume interrupted runs
# ---------------------------------------------------------------------------
out_path = pathlib.Path(f"results/{run_id}_{split}.jsonl")
pathlib.Path("results").mkdir(exist_ok=True)

completed: dict[str, dict] = {}
if out_path.exists():
    for line in out_path.open(encoding="utf-8"):
        row = json.loads(line)
        completed[row["id"]] = row
    logger.info("Resuming — %d/%d items already done.", len(completed), len(items))

remaining = [it for it in items if it["id"] not in completed]
logger.info(
    "Processing %d items with %d workers | system=%s judge=%s",
    len(remaining),
    args.workers,
    SYS_MODEL,
    JUDGE_MODEL,
)

# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------
def process_item(item: dict) -> dict:
    """Generate an explanation and grade it. Returns a result row."""
    try:
        out, m1 = call_json(
            SYS_MODEL,
            prompt,
            f"Schema:\n{item['schema']}\nSQL:\n{item['sql']}",
            Explanation,
        )
        v, m2 = call_json(
            JUDGE_MODEL,
            JUDGE,
            f"Schema:\n{item['schema']}\n\nSQL:\n{item['sql']}\n\nOperations actually present:\n{operation_inventory(item['sql'])}\n\nExplanation to evaluate:\n{out.explanation}",
            Verdict,
        )
        dims = {"correctness": v.correctness, "completeness": v.completeness, "hallucination_free": v.hallucination_free, "clarity": v.clarity}
        score = int(all(dims.values()))
        return {
            "id": item["id"],
            "input": item["sql"],
            "output": out.explanation,
            "score": score,
            "dimensions": dims,
            "judge_reported_grade": v.grade,
            "judge_grade_consistent": (v.grade == "good") == bool(score),
            "reason": v.reason,
            "sys_cost": m1,
            "judge_cost": m2,
            "error": None,
        }
    except Exception as exc:
        logger.error("Item %s failed: %s", item["id"], exc)
        return {
            "id": item["id"],
            "input": item["sql"],
            "output": None,
            "score": 0,
            "reason": None,
            "sys_cost": None,
            "judge_cost": None,
            "error": str(exc),
        }

# ---------------------------------------------------------------------------
# Parallel execution with live checkpoint writes
# ---------------------------------------------------------------------------
all_rows: list[dict] = list(completed.values())

with open(out_path, "a", encoding="utf-8") as out_file:
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_item = {executor.submit(process_item, it): it for it in remaining}

        with tqdm(total=len(remaining), unit="item", desc=f"{run_id}/{split}") as pbar:
            for future in as_completed(future_to_item):
                row = future.result()
                all_rows.append(row)
                # Write immediately so progress survives crashes
                out_file.write(json.dumps(row) + "\n")
                out_file.flush()
                status = "✓" if row["score"] else "✗"
                pbar.set_postfix(id=row["id"], status=status)
                pbar.update(1)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
scored = [r for r in all_rows if r["error"] is None]
errors = [r for r in all_rows if r["error"] is not None]
n_good = sum(r["score"] for r in scored)
pct = 100 * n_good / len(scored) if scored else 0.0

logger.info(
    "%s/%s: %d/%d good (%.1f%%) | %d errors",
    run_id,
    split,
    n_good,
    len(scored),
    pct,
    len(errors),
)

if not args.no_pdf:
    try:
        from export_pdf import generate_pdf_report
        pdf_out = out_path.parent / f"{out_path.stem}_report.pdf"
        generate_pdf_report(out_path, pdf_out, data_dir=pathlib.Path("data"))
        logger.info("PDF report generated: %s", pdf_out)
    except Exception as exc:
        logger.warning("Could not auto-generate PDF report: %s", exc)
if errors:
    logger.warning("Failed items: %s", [e["id"] for e in errors])
    sys.exit(1)