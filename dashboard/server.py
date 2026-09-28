"""
dashboard/server.py — Live evaluation dashboard server.

Serves the dashboard HTML and exposes a /api/status endpoint that reads
the results JSONL files in real time.

Usage:
    python dashboard/server.py
    # Then open http://localhost:7860 in your browser
"""

import json
import os
import pathlib
from http.server import BaseHTTPRequestHandler, HTTPServer

RESULTS_DIR = pathlib.Path("results")
DATA_DIR = pathlib.Path("data")
PORT = 7860


def read_results(run_id: str, split: str) -> dict:
    """Read a results JSONL file and compute live stats."""
    path = RESULTS_DIR / f"{run_id}_{split}.jsonl"
    if not path.exists():
        return {"exists": False, "rows": [], "total": 0, "done": 0, "good": 0, "bad": 0, "errors": 0}

    # Count total items in the split
    split_path = DATA_DIR / f"{split}.jsonl"
    total = sum(1 for _ in split_path.open(encoding="utf-8")) if split_path.exists() else 0

    rows = []
    for line in path.open(encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    good = sum(1 for r in rows if r.get("score") == 1 and r.get("error") is None)
    bad = sum(1 for r in rows if r.get("score") == 0 and r.get("error") is None)
    errors = sum(1 for r in rows if r.get("error") is not None)

    latencies = [
        r["sys_cost"]["latency"] + r["judge_cost"]["latency"]
        for r in rows
        if r.get("sys_cost") and r.get("judge_cost")
    ]
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0

    scored = good + bad
    pct = round(100 * good / scored, 1) if scored > 0 else 0

    return {
        "exists": True,
        "total": total,
        "done": len(rows),
        "good": good,
        "bad": bad,
        "errors": errors,
        "pct": pct,
        "avg_latency": avg_latency,
        "rows": list(reversed(rows[-30:])),  # last 30 items, newest first
    }


def list_runs() -> list[dict]:
    """Scan results/ for all run files and return their metadata."""
    if not RESULTS_DIR.exists():
        return []
    runs = {}
    for f in sorted(RESULTS_DIR.iterdir()):
        if not f.name.endswith(".jsonl"):
            continue
        parts = f.stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        run_id, split = parts
        key = f"{run_id}_{split}"
        runs[key] = {"run_id": run_id, "split": split}
    return list(runs.values())


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress request logs

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_file(pathlib.Path(__file__).parent / "index.html", "text/html")
        elif self.path == "/api/runs":
            self._json(list_runs())
        elif self.path.startswith("/api/status"):
            params = dict(p.split("=") for p in self.path.split("?", 1)[-1].split("&") if "=" in p)
            run_id = params.get("run_id", "v2")
            split = params.get("split", "dev")
            self._json(read_results(run_id, split))
        else:
            self.send_response(404)
            self.end_headers()

    def _json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: pathlib.Path, mime: str):
        if not path.exists():
            self.send_response(404)
            self.end_headers()
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Dashboard running at http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
