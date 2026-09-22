# Prompt Changelog

## v1

- Initial instruction: explain each SQL query in plain language for a non-technical reader.
- The runner supplies the query schema and SQL as separate inputs.
- The judge checks correctness, completeness, and clarity using the labelling guide.
- Added a clause-by-clause checklist and prohibited unsupported inferences, headings, and meta-intros after the full baseline scored 15/50 on dev and 36/116 on test.
