import json
import sqlite3
import sys

bad = 0
for line in open("data/all.jsonl", encoding="utf-8"):
    r = json.loads(line)
    try:
        con = sqlite3.connect(":memory:")
        con.executescript(r["schema"])
        con.execute("EXPLAIN " + r["sql"])
    except Exception as e:
        bad += 1
        print(f"[INVALID] {r['id']}: {e}")

print(f"invalid: {bad}")
sys.exit(bad)  # Non-zero exit so CI/scripts can detect failures