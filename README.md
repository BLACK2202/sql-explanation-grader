# SQL Explanation Grader

This project generates or grades plain-language explanations of SQLite `SELECT` queries. It uses one local Ollama model to write an explanation and a second model to judge it against the labelling guide.

## Requirements

- Python 3.10 or newer
- Ollama running locally
- `llama3.2:3b` and `qwen2.5:3b` installed, or equivalent models passed on the command line

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Validate the generated SQL:

```powershell
python harness/validate.py
```

Run the one-item smoke test:

```powershell
python harness/run.py tiny prompts/v1.md smoke
```

Run the development or test split:

```powershell
python harness/run.py dev prompts/v1.md v1
python harness/run.py test prompts/v1.md v1
```

Results are written to `results/<run_id>_<split>.jsonl`. Each row contains the generated explanation, binary score, judge reason, latency, and token counts.

To use different local models:

```powershell
python harness/run.py tiny prompts/v1.md smoke --system-model llama3.2:3b --judge-model qwen2.5:3b
```
