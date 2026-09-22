import time

import ollama
from pydantic import BaseModel

def call_json(model, system, user, schema: type[BaseModel], temperature=0, seed=None):
    t = time.time()
    last_error = None
    for attempt in range(3):
        opts = {"temperature": temperature, "num_predict": 3000}
        if seed is not None:
            opts["seed"] = seed
        try:
            r = ollama.chat(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                format=schema.model_json_schema(),
                options=opts,
            )
            return schema.model_validate_json(r.message.content), {
                "latency": round(time.time() - t, 2), "usd": 0.0,
                "in_tok": r.prompt_eval_count or 0, "out_tok": r.eval_count or 0}
        except Exception as error:
            last_error = error
            if attempt < 2:
                continue
    raise RuntimeError(f"{model} failed to return valid {schema.__name__} JSON after 3 tries") from last_error