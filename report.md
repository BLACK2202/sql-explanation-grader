# Evaluation Report

## Scope

The grader generates a plain-language SQL explanation with `llama3.2:3b` and evaluates it with `qwen2.5:3b`. The judge follows `data/LABELLING_GUIDE.md`, where an explanation must be correct, complete, and clear.

## Dataset

- `all.jsonl`: 166 items
- `dev.jsonl`: 50 items
- `test.jsonl`: 116 items
- Rejected generations recorded in `data/rejected.jsonl`: 70

`harness/validate.py` completed with `invalid: 0`, so every item in `all.jsonl` parses as valid SQLite and passes `EXPLAIN`.

## Smoke Evaluation

Command:

```powershell
c:/Users/louay/grader/.venv/Scripts/python.exe harness/run.py tiny prompts/v1.md smoke
```

Result: `1/1 good (100.0%)` for item `t1`.

The smoke output is stored in `results/smoke_tiny.jsonl`. This is a connectivity and schema-contract check, not a statistically meaningful quality estimate. Run the `dev` and `test` splits for broader evaluation.

## Full Evaluation

Commands:

```powershell
c:/Users/louay/grader/.venv/Scripts/python.exe harness/run.py dev prompts/v1.md full
c:/Users/louay/grader/.venv/Scripts/python.exe harness/run.py test prompts/v1.md full
```

Results:

- Development: `15/50 good (30.0%)`, written to `results/full_dev.jsonl`
- Test: `36/116 good (31.0%)`, written to `results/full_test.jsonl`

The two splits give a consistent baseline of roughly 30% good explanations. This indicates that the pipeline is functioning, but the v1 prompt/model combination needs improvement before the grader can be considered reliable.

## Limitations

- Scores are produced by an LLM judge and may vary with model versions and prompts.
- The current cost fields report local token counts and set USD cost to `0.0`; they are not billing estimates.
- The smoke split contains only one deliberately simple query.
