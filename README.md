# SQL Explanation Grader — Two-Model Benchmark

A local Ollama benchmark for comparing **two LLMs fairly** on plain-language explanations of SQLite `SELECT` queries.

Both generator models receive the **same SQL query, schema/context, prompt, temperature, seed, and token limit**. Their explanations are graded independently by the **same judge model and judge prompt** and saved side by side in one JSONL row per query.

> **Metric terminology:** the benchmark reports **judge pass rate**, not “true accuracy”. It should only be called accuracy after the judge/output has been validated against human or golden labels.

## What is compared

For every SQL query the benchmark records:

- Model A explanation
- Model B explanation
- shared judge verdict for each model
- correctness
- completeness
- hallucination-free rate
- clarity
- generator latency
- input/output token usage
- pair outcome: `both_correct`, `only_a_correct`, `only_b_correct`, `both_wrong`
- disagreement flag
- SQL categories: JOIN, WHERE, GROUP BY, HAVING, ORDER BY, aggregation, subquery, LIMIT, DISTINCT

The judge receives the SQL **and the schema**, plus a deterministic inventory of operations actually present. The judge prompt explicitly forbids demanding SQL operations that do not occur in the query.

## Requirements

- Python 3.10+
- Ollama running locally
- two generator models pulled
- one judge model pulled (it may be the same as one of the generators)

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Example model pulls:

```bash
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
```

## Validate the dataset

```bash
python harness/validate.py
```

## Run a quick comparison

```bash
python harness/compare.py tiny prompts/v2.md qwen_vs_llama \
  --model-a qwen2.5:7b \
  --model-b llama3.1:8b \
  --judge-model qwen2.5:7b
```

Development split:

```bash
python harness/compare.py dev prompts/v2.md qwen_vs_llama \
  --model-a qwen2.5:7b \
  --model-b llama3.1:8b \
  --judge-model qwen2.5:7b
```

Any Ollama model names can be supplied through `--model-a`, `--model-b`, and `--judge-model`.

### Fairness-related options

```text
--temperature 0.0   Same generation temperature for A and B
--seed 42           Same seed for A and B where supported by Ollama/model
--num-predict 3000  Same output-token ceiling for A and B
--workers 1         Default: avoids concurrent GPU contention skewing latency
```

Generation order alternates between A-first and B-first across items to reduce systematic first/second effects.

## Analyze results

```bash
python harness/analyze.py results/qwen_vs_llama_dev.jsonl
```

The report includes dimension scores, pair outcomes, disagreement cases, generation latency/token usage, and performance by SQL category.

## Dashboard

Start the server from the project root:

```bash
python dashboard/server.py
```

Open `http://localhost:7860`.

The dashboard shows Model A and Model B side by side, pair outcomes, category performance, disagreement rates, and per-query explanations/judge reasons.

## Output format

Results are stored at:

```text
results/<run_id>_<split>.jsonl
```

Each benchmark-v2 row contains the common SQL/schema/settings plus nested `model_a` and `model_b` objects. This prevents accidental comparisons across mismatched inputs or settings.

## PDF reports

PDF export remains available for both legacy single-model runs and the new two-model comparisons.

```bash
# Legacy single-model report
python harness/export_pdf.py results/v2_dev.jsonl

# Two-model comparison report
python harness/export_compare_pdf.py results/qwen_vs_llama_dev.jsonl
```

The comparison dashboard also includes an **Export PDF** button for the selected A/B run.

## Tests

Tests that do not require a running Ollama instance:

```bash
python -m unittest discover -s tests -v
```

## Legacy files

`harness/run.py` remains available for the older single-generator workflow. New A/B comparisons should use `harness/compare.py`.
