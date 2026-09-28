# SQL Explanation Grader

This project generates or grades plain-language explanations of SQLite `SELECT` queries. It uses one local Ollama model to write an explanation and a second model to judge it against the labelling guide.

## Requirements

- Python 3.10 or newer
- Ollama running locally
- At least one model pulled (e.g. `qwen2.5:7b` for generation, `qwen2.5:3b` for judging)

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Running

### Validate the dataset

```powershell
python harness/validate.py
```

### Smoke test (1 item)

```powershell
python harness/run.py tiny prompts/v2.md smoke
```

### Development split

```powershell
python harness/run.py dev prompts/v2.md v2 --system-model qwen2.5:7b --judge-model qwen2.5:3b
```

### Test split

```powershell
python harness/run.py test prompts/v2.md v2 --system-model qwen2.5:7b --judge-model qwen2.5:3b
```

Results are written to `results/<run_id>_<split>.jsonl`. Each row contains the generated explanation, binary score, judge reason, latency, and token counts. Runs that crash mid-way automatically resume from where they left off.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--system-model` | `qwen2.5:7b` | Model used to generate explanations |
| `--judge-model` | `qwen2.5:3b` | Model used to grade explanations |
| `--workers` | `4` | Number of parallel threads |
| `--judge-prompt` | `prompts/judge.md` | Path to the versioned judge system prompt |

## Prompts

| File | Description |
|------|-------------|
| `prompts/v1.md` | Initial baseline prompt (30% accuracy) |
| `prompts/v2.md` | Improved 10-step checklist prompt |
| `prompts/judge.md` | Versioned judge system prompt |

See `prompts/CHANGELOG.md` for the full prompt evolution history.
