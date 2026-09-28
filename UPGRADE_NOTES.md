# Two-Model Benchmark Upgrade

## Added
- `harness/compare.py`: configurable two-model Ollama benchmark.
- `harness/sql_features.py`: deterministic SQL operation/category detection.
- Structured judge dimensions: correctness, completeness, hallucination-free, clarity.
- Side-by-side JSONL records with pair outcomes and disagreement flags.
- Category analysis for JOIN, WHERE, GROUP BY, HAVING, ORDER BY, aggregation, subquery, LIMIT, DISTINCT.
- Generation latency and token comparison.
- Prompt SHA-256 hashes and identical generation settings stored for reproducibility.
- New comparison dashboard.
- Regression tests for SQL feature detection and pair outcomes.

## Evaluation fixes
- Judge now receives the database schema.
- Judge receives a deterministic inventory of operations present in the SQL.
- Judge prompt explicitly forbids demanding absent operations.
- Overall pass is derived from all four judge dimensions, so an internally inconsistent judge `grade` cannot silently alter the benchmark result.
- Reports use `judge pass rate`, not `accuracy`, unless results are externally validated against human/golden labels.

## Fairness controls
- Same query, schema, prompt and generation settings for both models.
- Same judge model and judge prompt for both outputs.
- Same seed and temperature.
- Default one worker to avoid GPU contention skewing latency.
- A/B generation order alternates across items to reduce systematic first/second effects.
