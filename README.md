# SQL Explanation Grader — Fair Two-Model Benchmark

A local Ollama benchmark for comparing two LLMs fairly on plain-language explanations of SQLite SELECT queries.

Model A and Model B receive the same SQL query, schema/context, generator prompt, generation settings, seed and output limit. Both explanations are evaluated independently by the same judge model and exact same judge prompt.

> Important: the benchmark reports **judge pass rate**, not “true accuracy”. Use the word accuracy only after validating the judge/output against human or trusted golden labels.

## What is compared

For every SQL query the benchmark records:

- Model A and Model B explanations side by side
- correctness
- completeness
- hallucination-free status
- clarity
- generation latency and output throughput
- input/output token usage
- both-correct / only-A / only-B / both-wrong outcomes
- disagreement cases
- SQL-category performance
- prompt, case and benchmark-fairness hashes
- A-first / B-first generation order
- per-model errors without treating infrastructure failures as wrong answers

## SQL categories

The deterministic detector covers:

JOIN, WHERE, GROUP BY, HAVING, ORDER BY, aggregation, window functions, subqueries, CTEs, UNION/INTERSECT/EXCEPT, LIMIT/OFFSET, DISTINCT, CASE, EXISTS, IN, BETWEEN, LIKE/GLOB/REGEXP and NULL filtering.

Comments and quoted strings/identifiers are masked before matching, so a word such as JOIN inside a string does not create a fake JOIN requirement.

## Why the judge was upgraded

The judge receives:

1. the database schema,
2. the exact SQL,
3. a deterministic inventory of operations actually present,
4. the candidate explanation.

The judge prompt explicitly says never penalize an explanation for an operation that is absent from the SQL.

Completeness is therefore evaluated against the actual query rather than an imagined checklist.

Overall pass is derived from four dimensions:

- correctness
- completeness
- hallucination_free
- clarity

The judge’s grade field is retained as an audit field, but it cannot silently override the deterministic four-dimension score.

## Requirements

- Python 3.10+
- Ollama running locally
- two generator models pulled
- one judge model pulled

Install:

~~~bash
python -m pip install -r requirements.txt
~~~

Example:

~~~bash
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
~~~

## Run a fair comparison

~~~bash
python harness/compare.py tiny prompts/v2.md qwen_vs_llama \
  --model-a qwen2.5:7b \
  --model-b llama3.1:8b \
  --judge-model qwen2.5:7b
~~~

Development split:

~~~bash
python harness/compare.py dev prompts/v2.md qwen_vs_llama \
  --model-a qwen2.5:7b \
  --model-b llama3.1:8b \
  --judge-model qwen2.5:7b
~~~

Any Ollama model name can be supplied through the three model flags.

### Fairness controls

Defaults are intentionally conservative:

~~~text
temperature=0
seed=42
num-predict=3000
num-ctx=8192
top-p=1
top-k=0
repeat-penalty=1
workers=1
warmup=true
~~~

Both generators receive identical settings.

Both generators and the judge are warmed up before measured calls, reducing cold model-load bias in latency.

Generation order alternates A-first and B-first across dataset items.

For clean latency comparisons, keep workers at 1. Parallel workers are available when throughput matters more than latency fairness.

Optional remote Ollama:

~~~bash
python harness/compare.py dev prompts/v2.md run \
  --model-a qwen2.5:7b \
  --model-b llama3.1:8b \
  --judge-model qwen2.5:7b \
  --ollama-host http://127.0.0.1:11434
~~~

## Results

Results are written to:

~~~text
results/<run_id>_<split>.jsonl
~~~

The runner resumes interrupted work and compacts checkpoints to one canonical row per query.

Infrastructure/model failures are stored as evaluation_error instead of being silently counted as wrong.

Each row stores a case hash and a common benchmark-fairness hash. This makes accidental mixing of runs with different prompts/settings detectable.

## Statistical analysis

~~~bash
python harness/analyze.py results/qwen_vs_llama_dev.jsonl
~~~

The analyzer reports:

- judge pass rate
- correctness, completeness, hallucination-free and clarity
- average and median generation latency
- input/output tokens and output tok/s
- paired B-minus-A pass-rate difference
- paired bootstrap 95% confidence interval
- exact McNemar p-value on discordant pairs
- both-correct / only-A / only-B / both-wrong counts
- disagreement rate
- performance by SQL category
- disagreement cases and both judge reasons

Token counts are useful resource metrics, but raw token units are tokenizer/model-specific and should not be treated as identical units across architectures.

## Optional human/golden validation

Create a JSONL file with:

~~~json
{"id":"q001","model_a_score":1,"model_b_score":0}
{"id":"q002","model_a_score":1,"model_b_score":1}
~~~

Then run:

~~~bash
python harness/analyze.py results/qwen_vs_llama_dev.jsonl --gold gold_labels.jsonl
~~~

The result is reported as agreement with the supplied gold labels. The project still avoids calling an unvalidated judge pass rate true accuracy.

A template is provided in gold_labels.example.jsonl.

## Dashboard

~~~bash
python dashboard/server.py
~~~

Open http://localhost:7860.

The upgraded dashboard provides:

- A/B pass-rate cards
- paired difference
- disagreement count
- latency and throughput
- dimension-by-dimension comparison
- pair-outcome table
- SQL-category comparison
- searchable/filterable per-query cases
- side-by-side explanations and judge reasons
- PDF export
- fairness-hash mismatch warning

## PDF report

~~~bash
python harness/export_compare_pdf.py results/qwen_vs_llama_dev.jsonl
~~~

The PDF includes the executive comparison, paired statistics, SQL-category performance, pair outcomes and disagreement cases.

## Tests

No Ollama server is required for the unit tests:

~~~bash
python -m unittest discover -s tests -v
python -m py_compile harness/*.py dashboard/server.py
~~~

## Legacy workflow

harness/run.py remains available for the original single-generator workflow. New A/B comparisons should use harness/compare.py.
