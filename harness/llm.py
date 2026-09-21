import time, ollama
from pydantic import BaseModel

def call_json(model, system, user, schema: type[BaseModel]):
    t = time.time()
    for _ in range(3):
        r = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            format=schema.model_json_schema(),
            options={"temperature": 0, "num_predict": 3000},
        )
        try:
            return schema.model_validate_json(r.message.content), {
                "latency": round(time.time() - t, 2), "usd": 0.0,
                "in_tok": r.prompt_eval_count or 0, "out_tok": r.eval_count or 0}
        except Exception:
            continue
    raise ValueError("bad JSON after 3 tries")