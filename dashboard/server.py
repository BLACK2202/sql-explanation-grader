"""Simple dashboard API for two-model comparison results."""
import json
import pathlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
INDEX = pathlib.Path(__file__).with_name("index.html")
PORT = 7860

def load_runs():
    return sorted([p.name for p in RESULTS.glob("*.jsonl")], reverse=True)

def load_rows(name):
    path = RESULTS / pathlib.Path(name).name
    if not path.exists():
        return []
    rows = []
    for line in path.open(encoding="utf-8"):
        if line.strip():
            try:
                row = json.loads(line)
                if row.get("benchmark_version") == 2:
                    rows.append(row)
            except json.JSONDecodeError:
                pass
    return rows

class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
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
        elif parsed.path == "/api/runs":
            self._json(load_runs())
        elif parsed.path.startswith("/api/results/"):
            self._json(load_rows(parsed.path.split("/")[-1]))
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    print(f"Dashboard: http://localhost:{PORT}")
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
