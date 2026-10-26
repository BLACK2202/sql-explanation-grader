"""Download the documents listed in corpus/manifest.json.

The script is polite and auditable:
  * it checks robots.txt for every URL before fetching it,
  * it waits between requests and identifies itself with a User-Agent,
  * it only downloads URLs that are explicitly listed in the manifest,
  * it records the retrieval date and a SHA-256 hash of every file
    in corpus/manifest.lock.json so the corpus is reproducible.

Usage:
    python scripts/fetch_corpus.py            # fetch missing files only
    python scripts/fetch_corpus.py --refresh  # re-download everything
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import re
import sys
import time
import urllib.parse
import urllib.request
import urllib.robotparser

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.json"
LOCK = ROOT / "corpus" / "manifest.lock.json"
USER_AGENT = "sql-explanation-grader-corpus-fetcher/1.0 (university project; polite, robots-aware)"
DELAY_S = 2.0


def allowed_by_robots(url: str, cache: dict[str, urllib.robotparser.RobotFileParser]) -> bool:
    parts = urllib.parse.urlsplit(url)
    base = f"{parts.scheme}://{parts.netloc}"
    if base not in cache:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(base + "/robots.txt")
        try:
            rp.read()
        except Exception as exc:  # network failure: refuse rather than guess
            print(f"[robots] could not read {base}/robots.txt: {exc}", file=sys.stderr)
            return False
        cache[base] = rp
    return cache[base].can_fetch(USER_AGENT, url)


_STRIP_BLOCKS = re.compile(rb"<(svg|script|style|noscript)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)


def normalize(body: bytes) -> bytes:
    """Remove inline SVG syntax diagrams, scripts and styles.

    They hold no retrievable prose (the SQLite pages embed megabytes of
    railroad diagrams), so the stored snapshot keeps only the text-bearing
    HTML. The original hash is still recorded in the lock file.
    """
    return _STRIP_BLOCKS.sub(b"", body)


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="Re-download files that already exist.")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {}
    robots: dict[str, urllib.robotparser.RobotFileParser] = {}
    failures = 0

    for doc in manifest["documents"]:
        dest = ROOT / doc["path"]
        if dest.exists() and not args.refresh:
            print(f"[skip] {doc['doc_id']} (already present)")
            continue
        if not allowed_by_robots(doc["url"], robots):
            print(f"[robots] {doc['url']} is disallowed; not fetching")
            failures += 1
            continue
        try:
            body = fetch(doc["url"])
        except Exception as exc:
            print(f"[error] {doc['doc_id']}: {exc}")
            failures += 1
            continue
        stored = normalize(body)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(stored)
        lock[doc["doc_id"]] = {
            "url": doc["url"],
            "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "sha256_original": hashlib.sha256(body).hexdigest(),
            "bytes_original": len(body),
            "sha256_stored": hashlib.sha256(stored).hexdigest(),
            "bytes_stored": len(stored),
        }
        print(f"[ok] {doc['doc_id']}: {len(body):,} bytes -> {len(stored):,} stored")
        time.sleep(DELAY_S)

    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
