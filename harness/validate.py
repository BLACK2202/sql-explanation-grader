import json
import sqlite3
import sys

bad = 0
seen_ids = set()

for line in open("data/all.jsonl", encoding="utf-8"):
    r = json.loads(line)

    if r["id"] in seen_ids:
        bad += 1
        print(f"[DUPLICATE] {r['id']}")
        continue

    seen_ids.add(r["id"])

    try:
        con = sqlite3.connect(":memory:")
        con.executescript(r["schema"])
        con.execute("EXPLAIN " + r["sql"])
    except Exception as e:
        bad += 1
        print(f"[INVALID] {r['id']}: {e}")

print(f"invalid: {bad}")
sys.exit(bad)