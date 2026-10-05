"""Zero-dependency dashboard server for benchmark-v3 result files."""
from __future__ import annotations

import json
import pathlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
INDEX = pathlib.Path(__file__).with_name("index.html")
PORT = 7860


def load_runs():
    return sorted(p.name for p in RESULTS.glob("*.jsonl") if p.is_file())


def load_rows(name: str):
    path = RESULTS / pathlib.Path(unquote(name)).name
    if not path.exists() or not path.is_file():
        return []
    rows = []
    for line in path.open(encoding="utf-8"):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = INDEX.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/health":
            self._json({"ok": True, "result_files": len(load_runs())})
            return
        if parsed.path == "/api/runs":
            self._json(load_runs())
            return
        if parsed.path.startswith("/api/results/"):
            self._json(load_rows(parsed.path[len("/api/results/"):]))
            return
        if parsed.path.startswith("/api/export-pdf/"):
            name = pathlib.Path(unquote(parsed.path[len("/api/export-pdf/"):])).name
            source = RESULTS / name
            if not source.exists() or source.suffix != ".jsonl":
                self._json({"error": "result file not found"}, 404)
                return
            try:
                import sys
                sys.path.insert(0, str(ROOT / "harness"))
                from export_compare_pdf import generate_pdf_report
                REPORTS.mkdir(exist_ok=True)
                target = REPORTS / (source.stem + "_comparison.pdf")
                generate_pdf_report(source, target)
                body = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/pdf")
                self.send_header("Content-Disposition", f'attachment; filename="{target.name}"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                self._json({"error": str(exc)}, 500)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt, *args):
        return


if __name__ == "__main__":
    print(f"SQL Explanation Grader dashboard: http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
