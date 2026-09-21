import json, sys, pathlib
from llm import call_json
from schema import Explanation, Verdict

SYS_MODEL, JUDGE_MODEL = "llama3.2:3b", "qwen2.5:7b"
split, prompt_file, run_id = sys.argv[1:4]
prompt = pathlib.Path(prompt_file).read_text(encoding="utf-8")
guide = pathlib.Path("data/LABELLING_GUIDE.md").read_text(encoding="utf-8")
JUDGE = f"You grade SQL explanations using this guide:\n{guide}\nGive a grade (good or bad) and a short reason."

rows = []
for line in open(f"data/{split}.jsonl", encoding="utf-8"):
    item = json.loads(line)
    out, m1 = call_json(SYS_MODEL, prompt, f"Schema:\n{item['schema']}\nSQL:\n{item['sql']}", Explanation)
    v, m2 = call_json(JUDGE_MODEL, JUDGE, f"SQL:\n{item['sql']}\nExplanation:\n{out.explanation}", Verdict)
    rows.append({"id": item["id"], "input": item["sql"], "output": out.explanation,
                 "score": int(v.grade == "good"), "reason": v.reason,
                 "sys_cost": m1, "judge_cost": m2})
    print(len(rows), item["id"], v.grade, flush=True)

pathlib.Path("results").mkdir(exist_ok=True)
with open(f"results/{run_id}_{split}.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
n = sum(r["score"] for r in rows)
print(f"{run_id}/{split}: {n}/{len(rows)} good ({100*n/len(rows):.1f}%)")