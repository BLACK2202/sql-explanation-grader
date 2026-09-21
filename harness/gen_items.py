import json, random
from pydantic import BaseModel
from llm import call_json

class Item(BaseModel):
    schema_ddl: str
    sql: str
class Batch(BaseModel):
    items: list[Item]

DOMAINS = ["retail", "school", "hospital", "bank", "airline", "library", "hotel", "logistics"]
FEATURES = ["simple SELECT/WHERE", "JOIN", "GROUP BY/HAVING", "subquery",
            "window function", "CTE", "CASE expression", "LEFT JOIN with NULL filter"]
SYS = "You write realistic SQL exercises. Vary difficulty and style. No duplicates."

random.seed(0)
seen, out, tries = set(), [], 0
while len(out) < 160 and tries < 100:
    tries += 1
    d, f = random.choice(DOMAINS), random.choice(FEATURES)
    b, _ = call_json("qwen2.5:7b", SYS,
        f"Domain: {d}. Feature: {f}. Write 5 different exercises. Each has CREATE TABLE statements (schema_ddl) and one query (sql).", Batch)
    for i in b.items:
        if i.sql not in seen:
            seen.add(i.sql)
            out.append({"id": f"q{len(out):03d}", "schema": i.schema_ddl, "sql": i.sql})
    print(len(out), "items", flush=True)

random.shuffle(out)
for name, part in (("dev", out[:50]), ("test", out[50:])):
    with open(f"data/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(r) + "\n")
print("done:", len(out))
