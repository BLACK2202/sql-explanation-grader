# Prompt Changelog

## v1

- Initial instruction: explain each SQL query in plain language for a non-technical reader.
- The runner supplies the query schema and SQL as separate inputs.
- The judge checks correctness, completeness, and clarity using the labelling guide.
- Added a clause-by-clause checklist and prohibited unsupported inferences, headings, and meta-intros after the full baseline scored 15/50 on dev and 36/116 on test.

## v2

- Expanded the checklist to 10 explicit steps covering columns, filters, joins, subqueries, aggregates, grouping, window functions, CTEs, CASE, and ordering/limiting.
- Added hard rule against leaking SQL keywords (SELECT, WHERE, etc.) into the output.
- Clarified distinction between "rows returned" and "a single calculated value".
- Separated the judge prompt into its own versioned file (`prompts/judge.md`) with richer examples.

## v3_rag (RAG extension, first design)

- `prompts/v3_rag.md` = `v2.md` verbatim + a "Reference documentation" section: sources are untrusted reference text, the SQL and schema are the authority, cite `[S#]` markers, report insufficient evidence.
- Used with up to 6 retrieved passages (≤ 1,800 tokens).
- Result on dev (AI-assisted reference labels): 10/50 good vs 20/50 for v2; answers copied general documentation into explanations. See `docs/EVALUATION.md`.

## v4_rag (RAG extension, focused; current default)

- `prompts/v4_rag.md` = `v2.md` verbatim + stricter source-usage rules only (use a source only for features the query uses, do not copy general statements, do not lengthen the answer to cite). No SQL facts were added, so the difference from v2 stays retrieval-related.
- Used with one covering passage per SQL feature (≤ 4 passages, ≤ 1,200 tokens).
- Result on dev: 16/50 good vs 20/50 for v2 (designed after inspecting v3's dev errors, so optimistic).

## judge_v2_rubric (schema-aware rubric judge)

- `prompts/judge_v2_rubric.md`: the judge receives the schema, the SQL, an automatic operation inventory and the explanation; returns 1–5 correctness, completeness, clarity, groundedness, unsupported claims, and the legacy good/bad grade.
- With `qwen2.5:3b` it approved 99/100 dev explanations (Cohen's κ 0.01 against the reference labels); the original `judge.md` prompt agreed better (κ 0.20). The original `judge.md` is unchanged and remains available (`--judge-mode legacy`, or `legacy_schema` to add only the schema).
