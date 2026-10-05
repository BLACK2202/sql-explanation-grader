from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def call_json(
    model: str,
    system: str,
    user: str,
    schema: type[BaseModel],
    *,
    temperature: float = 0.0,
    seed: int | None = 42,
    num_predict: int = 3000,
    num_ctx: int = 8192,
    top_p: float = 1.0,
    top_k: int = 0,
    repeat_penalty: float = 1.0,
    max_retries: int = 3,
    ollama_host: str | None = None,
    keep_alive: str | int = "10m",
) -> tuple[BaseModel, dict[str, Any]]:
    """Call Ollama and return parsed structured output plus reproducibility metrics."""
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError("The 'ollama' package is required. Run: python -m pip install -r requirements.txt") from exc

    client = ollama.Client(host=ollama_host) if ollama_host else ollama
    options = {
        "temperature": temperature,
        "seed": seed,
        "num_predict": num_predict,
        "num_ctx": num_ctx,
        "top_p": top_p,
        "top_k": top_k,
        "repeat_penalty": repeat_penalty,
    }
    started = time.perf_counter()
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                format=schema.model_json_schema(),
                options=options,
                keep_alive=keep_alive,
            )
            parsed = schema.model_validate_json(response.message.content)
            latency = time.perf_counter() - started
            in_tok = int(response.prompt_eval_count or 0)
            out_tok = int(response.eval_count or 0)
            return parsed, {
                "latency_s": round(latency, 4),
                "latency": round(latency, 2),
                "in_tok": in_tok,
                "out_tok": out_tok,
                "tok_per_s": round(out_tok / latency, 2) if latency > 0 else 0.0,
                "attempts": attempt,
                "usd": 0.0,
            }
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                delay = 2 ** (attempt - 1)
                logger.warning("%s attempt %d/%d failed: %s; retrying in %ss", model, attempt, max_retries, exc, delay)
                time.sleep(delay)

    raise RuntimeError(f"{model} failed to return valid {schema.__name__} JSON after {max_retries} attempts") from last_error
