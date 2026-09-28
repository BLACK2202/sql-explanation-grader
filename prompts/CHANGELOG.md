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
