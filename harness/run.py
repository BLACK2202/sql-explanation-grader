import argparse
import json
import pathlib

from llm import call_json
from schema import Explanation, Verdict

parser = argparse.ArgumentParser(description="Grade SQL explanations with two local Ollama models.")
parser.add_argument("split", choices=("tiny", "dev", "test", "all"))
parser.add_argument("prompt_file", type=pathlib.Path)
parser.add_argument("run_id")
parser.add_argument("--system-model", default="llama3.2:3b")
parser.add_argument("--judge-model", default="qwen2.5:3b")
args = parser.parse_args()

if not args.prompt_file.exists():
    parser.error(f"prompt file does not exist: {args.prompt_file}")

SYS_MODEL, JUDGE_MODEL = args.system_model, args.judge_model
split, prompt_file, run_id = args.split, args.prompt_file, args.run_id
prompt = pathlib.Path(prompt_file).read_text(encoding="utf-8")
guide = pathlib.Path("data/LABELLING_GUIDE.md").read_text(encoding="utf-8")
JUDGE = f"You grade SQL explanations using this guide:\n{guide}\nGive a grade (good or bad) and a short reason."

rows = []
input_path = pathlib.Path(f"data/{split}.jsonl")
if not input_path.exists():
    parser.error(f"dataset does not exist: {input_path}")

for line in input_path.open(encoding="utf-8"):
    item = json.loads(line)
    try:
        out, m1 = call_json(SYS_MODEL, prompt, f"Schema:\n{item['schema']}\nSQL:\n{item['sql']}", Explanation)
        v, m2 = call_json(JUDGE_MODEL, JUDGE, f"SQL:\n{item['sql']}\nExplanation:\n{out.explanation}", Verdict)
    except Exception as error:
        raise RuntimeError(f"failed while processing {item['id']}") from error
    rows.append({"id": item["id"], "input": item["sql"], "output": out.explanation,
                 "score": int(v.grade == "good"), "reason": v.reason,
                 "sys_cost": m1, "judge_cost": m2})
    print(len(rows), item["id"], v.grade, flush=True)

pathlib.Path("results").mkdir(exist_ok=True)
with open(f"results/{run_id}_{split}.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
n = sum(r["score"] for r in rows)
percentage = 100 * n / len(rows) if rows else 0
print(f"{run_id}/{split}: {n}/{len(rows)} good ({percentage:.1f}%)")