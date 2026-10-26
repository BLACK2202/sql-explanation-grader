"""LLM clients that return validated structured (JSON) output.

Two roles use an LLM:
  * the generator writes the SQL explanation,
  * the judge grades an explanation.

Both go through `LLMClient.generate_json`, which asks the model for JSON that
matches a Pydantic schema, validates it, retries with exponential backoff, and
returns timing and token counts.

Providers:
  ollama  local models through the Ollama HTTP API (default, no API key)
  openai  any OpenAI-compatible /chat/completions endpoint (optional)
  fake    scripted responses for the offline test-suite
"""
from __future__ import annotations

import json
import logging
import time
from typing import Callable, Protocol, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    provider: str

    def generate_json(self, model: str, system: str, user: str, schema: type[T], options: dict | None = None,
                      max_retries: int = 3) -> tuple[T, dict]: ...


def _metrics(t0: float, in_tok: int | None, out_tok: int | None, model: str, attempts: int, provider: str) -> dict:
    return {
        "latency": round(time.time() - t0, 2),
        "usd": 0.0,  # local inference: no billing (kept for backward compatibility)
        "in_tok": int(in_tok or 0),
        "out_tok": int(out_tok or 0),
        "model": model,
        "provider": provider,
        "attempts": attempts,
    }


class OllamaClient:
    provider = "ollama"

    def __init__(self, host: str = "http://localhost:11434", timeout_s: float = 240.0) -> None:
        import ollama

        self.host = host
        self.client = ollama.Client(host=host, timeout=timeout_s)

    def generate_json(self, model: str, system: str, user: str, schema: type[T], options: dict | None = None,
                      max_retries: int = 3) -> tuple[T, dict]:
        t0 = time.time()
        last: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                r = self.client.chat(
                    model=model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    format=schema.model_json_schema(),
                    options=dict(options or {}),
                )
                parsed = schema.model_validate_json(r.message.content)
                return parsed, _metrics(t0, r.prompt_eval_count, r.eval_count, model, attempt, self.provider)
            except Exception as exc:  # network, model or validation error
                last = exc
                if attempt < max_retries:
                    wait = 2 ** (attempt - 1)
                    logger.warning("%s attempt %d/%d failed: %s; retrying in %ss", model, attempt, max_retries, exc, wait)
                    time.sleep(wait)
        raise LLMError(f"{model} did not return valid {schema.__name__} JSON after {max_retries} attempts: {last}") from last

    def list_models(self) -> list[dict]:
        out = []
        for m in self.client.list().models:
            out.append({"name": m.model, "digest": m.digest, "size": m.size,
                        "parameter_size": getattr(m.details, "parameter_size", None),
                        "quantization": getattr(m.details, "quantization_level", None)})
        return out

    def model_identity(self, model: str) -> dict:
        try:
            for m in self.list_models():
                if m["name"] == model or m["name"] == f"{model}:latest":
                    return m
        except Exception as exc:  # pragma: no cover - depends on a running server
            return {"name": model, "error": str(exc)}
        return {"name": model, "error": "model not pulled"}


class OpenAICompatibleClient:
    provider = "openai"

    def __init__(self, base_url: str, api_key: str, timeout_s: float = 240.0) -> None:
        import httpx

        if not api_key:
            raise LLMError("SQLRAG_OPENAI_API_KEY is not set")
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(timeout=timeout_s, headers={"Authorization": f"Bearer {api_key}"})

    def generate_json(self, model: str, system: str, user: str, schema: type[T], options: dict | None = None,
                      max_retries: int = 3) -> tuple[T, dict]:
        opts = options or {}
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": opts.get("temperature", 0),
            "seed": opts.get("seed"),
            "max_tokens": opts.get("num_predict", 1024),
            "response_format": {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}},
        }
        t0 = time.time()
        last: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                r = self.http.post(f"{self.base_url}/chat/completions", json=body)
                r.raise_for_status()
                data = r.json()
                parsed = schema.model_validate_json(data["choices"][0]["message"]["content"])
                usage = data.get("usage", {})
                return parsed, _metrics(t0, usage.get("prompt_tokens"), usage.get("completion_tokens"), model, attempt, self.provider)
            except Exception as exc:
                last = exc
                if attempt < max_retries:
                    time.sleep(2 ** (attempt - 1))
        raise LLMError(f"{model} failed after {max_retries} attempts: {last}") from last

    def model_identity(self, model: str) -> dict:
        return {"name": model, "provider": self.provider, "base_url": self.base_url}


class FakeLLMClient:
    """Deterministic client for tests: `responder(model, system, user, schema)` returns a dict or JSON text."""

    provider = "fake"

    def __init__(self, responder: Callable[[str, str, str, type[BaseModel]], dict | str]) -> None:
        self.responder = responder
        self.calls: list[dict] = []

    def generate_json(self, model: str, system: str, user: str, schema: type[T], options: dict | None = None,
                      max_retries: int = 3) -> tuple[T, dict]:
        t0 = time.time()
        self.calls.append({"model": model, "system": system, "user": user, "schema": schema.__name__, "options": options})
        payload = self.responder(model, system, user, schema)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        try:
            parsed = schema.model_validate_json(text)
        except Exception as exc:
            raise LLMError(f"fake response did not validate: {exc}") from exc
        return parsed, _metrics(t0, len(system.split()) + len(user.split()), len(text.split()), model, 1, self.provider)

    def model_identity(self, model: str) -> dict:
        return {"name": model, "provider": "fake"}


def make_client(settings) -> LLMClient:
    if settings.llm_provider == "ollama":
        return OllamaClient(settings.ollama_host, settings.request_timeout_s)
    if settings.llm_provider == "openai":
        return OpenAICompatibleClient(settings.openai_base_url, settings.openai_api_key, settings.request_timeout_s)
    raise LLMError(f"unknown LLM provider {settings.llm_provider!r}")
