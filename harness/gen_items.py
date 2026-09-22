import json, random, os, sqlite3
from pydantic import BaseModel
from llm import call_json

class Queries(BaseModel):
    queries: list[str]

SCHEMAS = {
 "retail": """CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL, category TEXT);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER, quantity INTEGER, order_date TEXT);""",
 "school": """CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, year INTEGER);
CREATE TABLE courses (id INTEGER PRIMARY KEY, title TEXT, credits INTEGER);
CREATE TABLE enrollments (student_id INTEGER, course_id INTEGER, grade REAL);""",
 "hospital": """CREATE TABLE patients (id INTEGER PRIMARY KEY, name TEXT, birth_year INTEGER);
CREATE TABLE doctors (id INTEGER PRIMARY KEY, name TEXT, specialty TEXT);
CREATE TABLE appointments (id INTEGER PRIMARY KEY, patient_id INTEGER, doctor_id INTEGER, appt_date TEXT, fee REAL);""",
 "bank": """CREATE TABLE clients (id INTEGER PRIMARY KEY, name TEXT, branch TEXT);
CREATE TABLE accounts (id INTEGER PRIMARY KEY, client_id INTEGER, type TEXT, balance REAL);
CREATE TABLE transactions (id INTEGER PRIMARY KEY, account_id INTEGER, amount REAL, tx_date TEXT);""",
 "library": """CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT, author TEXT, year INTEGER);
CREATE TABLE members (id INTEGER PRIMARY KEY, name TEXT, joined TEXT);
CREATE TABLE loans (id INTEGER PRIMARY KEY, book_id INTEGER, member_id INTEGER, loan_date TEXT, returned INTEGER);""",
 "hotel": """CREATE TABLE guests (id INTEGER PRIMARY KEY, name TEXT, country TEXT);
CREATE TABLE rooms (id INTEGER PRIMARY KEY, room_type TEXT, price_per_night REAL);
CREATE TABLE bookings (id INTEGER PRIMARY KEY, guest_id INTEGER, room_id INTEGER, nights INTEGER, check_in TEXT);""",
}
FEATURES = ["simple SELECT with WHERE", "INNER JOIN of two tables", "GROUP BY with HAVING",
            "subquery in WHERE", "window function such as ROW_NUMBER or RANK", "CTE (WITH clause)",
            "CASE expression", "LEFT JOIN with IS NULL filter", "JOIN of three tables with ORDER BY and LIMIT"]
SYS = "You write SQLite SELECT queries for exercises. Use ONLY the tables and columns given. Reply with the queries only."
KEYWORD = {
 "simple SELECT with WHERE": "where",
 "INNER JOIN of two tables": "join",
 "GROUP BY with HAVING": "having",
 "subquery in WHERE": "(select",
 "window function such as ROW_NUMBER or RANK": " over (",
 "CTE (WITH clause)": "with ",
 "CASE expression": "case when",
 "LEFT JOIN with IS NULL filter": "left join",
 "JOIN of three tables with ORDER BY and LIMIT": "limit",
}
def check(ddl, sql):
    s = sql.strip().rstrip(";").strip()
    if ";" in s or not s.lower().startswith(("select", "with")):
        return None, "not a single SELECT"
    try:
        con = sqlite3.connect(":memory:")
        con.executescript(ddl)
        con.execute("EXPLAIN " + s)
        return s + ";", None
    except Exception as e:
        return None, str(e)[:80]

path, rej_path = "data/all.jsonl", "data/rejected.jsonl"
out = [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []
seen = {o["sql"] for o in out}
rejected = sum(1 for _ in open(rej_path, encoding="utf-8")) if os.path.exists(rej_path) else 0
random.seed(len(out))

call_no = 0
while len(out) < 165 and call_no < 250:
    call_no += 1
    dom = random.choice(list(SCHEMAS)); feat = random.choice(FEATURES)
    ddl = SCHEMAS[dom]
    try:
        b, _ = call_json("llama3.2:3b", SYS,
            f"Schema:\n{ddl}\n\nWrite 5 different queries. Each must use this feature: {feat}. "
            f"Use only these tables and columns. One statement each.",
            Queries, temperature=1.0, seed=call_no * 7919 + len(out))
    except ValueError:
        continue
    new = bad = 0
    for q in b.queries:
        sql, err = check(ddl, q)
        if not err and KEYWORD[feat] not in " ".join(sql.lower().split()):
            sql, err = None, "requested feature missing"
        if err:
            bad += 1; rejected += 1
            with open(rej_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"sql": q, "domain": dom, "reason": err}) + "\n")
        elif sql not in seen:
            seen.add(sql)
            row = {"id": f"q{len(out):03d}", "domain": dom, "feature": feat, "schema": ddl, "sql": sql}
            out.append(row); new += 1
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
    print(f"call {call_no}: +{new} valid, {bad} rejected | total {len(out)}, rejected overall {rejected}", flush=True)

random.seed(0); random.shuffle(out)
for name, part in (("dev", out[:50]), ("test", out[50:])):
    with open(f"data/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(r) + "\n")
print("done:", len(out), "valid items,", rejected, "rejected")