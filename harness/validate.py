import json, sqlite3
bad = 0
for l in open("data/all.jsonl", encoding="utf-8"):
    r = json.loads(l)
    try:
        con = sqlite3.connect(":memory:")
        con.executescript(r["schema"])
        con.execute("EXPLAIN " + r["sql"])
    except Exception as e:
        bad += 1
        print(r["id"], e)
print("invalid:", bad)