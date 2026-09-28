import time
import logging

import ollama
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def call_json(
    model: str,
    system: str,
    user: str,
    schema: type[BaseModel],
    temperature: float = 0,
    seed: int | None = None,
    max_retries: int = 3,
) -> tuple[BaseModel, dict]:
    """
    Call an Ollama model and parse structured JSON output.

    Returns a (parsed_model, metrics) tuple where metrics contains
    latency, usd (always 0.0 for local), in_tok, out_tok.
    Retries up to max_retries times with exponential backoff.
    """
    t = time.time()
    last_error: Exception | None = None

    opts: dict = {"temperature": temperature, "num_predict": 3000}
    if seed is not None:
        opts["seed"] = seed

    for attempt in range(max_retries):
        try:
            r = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                format=schema.model_json_schema(),
                options=opts,
            )
            parsed = schema.model_validate_json(r.message.content)
            metrics = {
                "latency": round(time.time() - t, 2),
                "usd": 0.0,
                "in_tok": r.prompt_eval_count or 0,
                "out_tok": r.eval_count or 0,
            }
            return parsed, metrics
        except Exception as error:
            last_error = error
            if attempt < max_retries - 1:
                backoff = 2 ** attempt
                logger.warning(
                    "Attempt %d/%d failed for model %s: %s — retrying in %ds",
                    attempt + 1,
                    max_retries,
                    model,
                    error,
                    backoff,
                )
                time.sleep(backoff)

    raise RuntimeError(
        f"{model} failed to return valid {schema.__name__} JSON after {max_retries} tries"
    ) from last_error