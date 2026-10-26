"""Regression tests: the original pipeline, data and results must keep working."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys

import pytest

from conftest import ROOT

PY = sys.executable

# Approval counts recorded before the RAG extension. These files must never change.
HISTORICAL = {"full_dev.jsonl": (15, 50), "full_test.jsonl": (36, 116), "v2_dev.jsonl": (30, 50), "improved_dev.jsonl": (34, 50)}


@pytest.mark.parametrize("name,expected", HISTORICAL.items())
def test_historical_results_are_preserved(name, expected):
    rows = [json.loads(line) for line in (ROOT / "results" / name).open(encoding="utf-8")]
    scored = [r for r in rows if r.get("error") is None]
    assert (sum(r["score"] for r in scored), len(scored)) == expected


def test_dataset_sizes_and_validation_script():
    for split, n in (("all", 166), ("dev", 50), ("test", 116)):
        assert sum(1 for _ in (ROOT / "data" / f"{split}.jsonl").open(encoding="utf-8")) == n
    out = subprocess.run([PY, "harness/validate.py"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert out.returncode == 0 and "invalid: 0" in out.stdout


def test_original_prompts_unchanged():
    # The v1/v2/judge prompts are the baseline of every comparison; editing them would break it.
    # Hashes are of the newline-normalised text (git may check files out with CRLF on Windows).
    from sqlrag.prompting import load_prompt

    expected = {"v1.md": "c1fc4e1daf53", "v2.md": "0bd6b691a535", "judge.md": "d64efb3b7c7a"}
    assert {p: load_prompt(ROOT / "prompts" / p)[1] for p in expected} == expected
    raw = (ROOT / "prompts" / "v2.md").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest()[:12] == expected["v2.md"]


def test_run_py_cli_still_works_and_has_schema_flag():
    out = subprocess.run([PY, "harness/run.py", "--help"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert out.returncode == 0
    for flag in ("--system-model", "--judge-model", "--workers", "--judge-prompt", "--no-pdf", "--legacy-judge-input"):
        assert flag in out.stdout


def test_run_py_judge_input_includes_schema():
    src = (ROOT / "harness" / "run.py").read_text(encoding="utf-8")
    assert "Schema:\\n{item['schema']}\\nSQL:\\n{item['sql']}\\nExplanation:" in src
    assert "if args.legacy_judge_input:" in src


def test_harness_schema_contract_unchanged():
    spec = importlib.util.spec_from_file_location("legacy_schema", ROOT / "harness" / "schema.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert set(mod.Explanation.model_fields) == {"explanation"}
    assert set(mod.Verdict.model_fields) == {"grade", "reason"}
    from sqlrag.schemas import Explanation, Verdict

    for new, old in ((Explanation, mod.Explanation), (Verdict, mod.Verdict)):
        assert new.model_json_schema()["properties"] == old.model_json_schema()["properties"]
        assert new.model_json_schema()["required"] == old.model_json_schema()["required"]


def test_legacy_dashboard_reads_results_and_imports_pdf_exporter():
    spec = importlib.util.spec_from_file_location("legacy_server", ROOT / "dashboard" / "server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.RESULTS_DIR = ROOT / "results"
    mod.DATA_DIR = ROOT / "data"
    status = mod.read_results("full", "dev")
    assert status["good"] == 15 and status["done"] == 50
    # Same sys.path as `python dashboard/server.py`: the script folder, not the repo root.
    code = ("import sys, importlib.util; sys.path[0] = 'dashboard'; "
            "spec = importlib.util.spec_from_file_location('srv', 'dashboard/server.py'); "
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
            "from harness.export_pdf import generate_pdf_report; print('ok')")
    out = subprocess.run([PY, "-c", code], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert out.stdout.strip().endswith("ok"), out.stderr


def test_analyze_and_pdf_export_on_historical_results(tmp_path):
    out = subprocess.run([PY, "harness/analyze.py", "results/v2_dev.jsonl", "--no-color"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert out.returncode == 0 and "Good        : 30" in out.stdout
    pdf = tmp_path / "r.pdf"
    out = subprocess.run([PY, "harness/export_pdf.py", "results/v2_dev.jsonl", "-o", str(pdf)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert out.returncode == 0 and pdf.read_bytes()[:4] == b"%PDF"


def test_run_py_checkpoint_resume(tmp_path):
    """harness/run.py skips item ids already present in the results file."""
    src = (ROOT / "harness" / "run.py").read_text(encoding="utf-8")
    assert "remaining = [it for it in items if it[\"id\"] not in completed]" in src
