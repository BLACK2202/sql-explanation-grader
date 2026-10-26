"""Runtime configuration, read from environment variables.

Every setting has a local-development default that works without a paid API
key. See .env.example for the documented list.
"""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field, fields

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_dotenv(path: pathlib.Path = ROOT / ".env") -> list[str]:
    """Load KEY=VALUE lines from .env into os.environ without overriding real
    environment variables. Inline comments (" # ...") and quotes are stripped.
    Set SQLRAG_NO_DOTENV=1 to skip. Returns the keys that were set."""
    if os.environ.get("SQLRAG_NO_DOTENV") == "1" or not path.is_file():
        return []
    loaded = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if value[:1] in {'"', "'"} and value.count(value[0]) >= 2:
            value = value[1:value.index(value[0], 1)]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


load_dotenv()


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(_env(name, str(default)))


@dataclass
class Settings:
    # Storage -----------------------------------------------------------
    store_dir: pathlib.Path = field(default_factory=lambda: _path("SQLRAG_STORE_DIR", ROOT / "rag_store"))
    corpus_manifest: pathlib.Path = field(default_factory=lambda: _path("SQLRAG_CORPUS_MANIFEST", ROOT / "corpus" / "manifest.json"))
    uploads_dir: pathlib.Path = field(default_factory=lambda: _path("SQLRAG_UPLOADS_DIR", ROOT / "corpus" / "uploads"))
    results_dir: pathlib.Path = field(default_factory=lambda: _path("SQLRAG_RESULTS_DIR", ROOT / "results"))
    data_dir: pathlib.Path = field(default_factory=lambda: _path("SQLRAG_DATA_DIR", ROOT / "data"))
    reports_dir: pathlib.Path = field(default_factory=lambda: _path("SQLRAG_REPORTS_DIR", ROOT / "reports"))

    # Embeddings --------------------------------------------------------
    # sentence-transformers (default, local) | hash (offline tests) | ollama
    embedding_backend: str = field(default_factory=lambda: _env("SQLRAG_EMBEDDING_BACKEND", "sentence-transformers"))
    embedding_model: str = field(default_factory=lambda: _env("SQLRAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    embedding_device: str = field(default_factory=lambda: _env("SQLRAG_EMBEDDING_DEVICE", "cpu"))

    # Chunking ----------------------------------------------------------
    chunk_size: int = field(default_factory=lambda: _env_int("SQLRAG_CHUNK_SIZE", 300))
    chunk_overlap: int = field(default_factory=lambda: _env_int("SQLRAG_CHUNK_OVERLAP", 60))
    min_chunk_tokens: int = field(default_factory=lambda: _env_int("SQLRAG_MIN_CHUNK_TOKENS", 60))

    # Retrieval ---------------------------------------------------------
    retrieval_mode: str = field(default_factory=lambda: _env("SQLRAG_RETRIEVAL_MODE", "hybrid"))  # hybrid | dense | bm25
    top_k: int = field(default_factory=lambda: _env_int("SQLRAG_TOP_K", 4))
    candidates: int = field(default_factory=lambda: _env_int("SQLRAG_CANDIDATES", 20))
    # Hits kept from each feature query before fusing across queries (0 = max(4, top_k)).
    per_query_k: int = field(default_factory=lambda: _env_int("SQLRAG_PER_QUERY_K", 0))
    rrf_k: int = field(default_factory=lambda: _env_int("SQLRAG_RRF_K", 60))
    min_similarity: float = field(default_factory=lambda: _env_float("SQLRAG_MIN_SIMILARITY", 0.45))
    max_context_tokens: int = field(default_factory=lambda: _env_int("SQLRAG_MAX_CONTEXT_TOKENS", 1200))
    # After one covering passage per feature, fill the remaining top_k slots with the
    # best other candidates (True) or stop (False: "focused" context).
    fill_context: bool = field(default_factory=lambda: _env_bool("SQLRAG_FILL_CONTEXT", False))

    # Grounding policy --------------------------------------------------
    # abstain : refuse to explain when evidence does not cover the query
    # fallback: produce a prompt-only answer, clearly labelled as ungrounded
    # off     : always generate with whatever evidence was retrieved
    grounding_policy: str = field(default_factory=lambda: _env("SQLRAG_GROUNDING_POLICY", "abstain"))
    min_feature_coverage: float = field(default_factory=lambda: _env_float("SQLRAG_MIN_FEATURE_COVERAGE", 1.0))

    # LLM ---------------------------------------------------------------
    llm_provider: str = field(default_factory=lambda: _env("SQLRAG_LLM_PROVIDER", "ollama"))  # ollama | openai
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    openai_base_url: str = field(default_factory=lambda: _env("SQLRAG_OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_api_key: str = field(default_factory=lambda: _env("SQLRAG_OPENAI_API_KEY", ""))
    generator_model: str = field(default_factory=lambda: _env("SQLRAG_GENERATOR_MODEL", "qwen2.5:7b"))
    judge_model: str = field(default_factory=lambda: _env("SQLRAG_JUDGE_MODEL", "qwen2.5:3b"))
    temperature: float = field(default_factory=lambda: _env_float("SQLRAG_TEMPERATURE", 0.0))
    seed: int = field(default_factory=lambda: _env_int("SQLRAG_SEED", 42))
    num_ctx: int = field(default_factory=lambda: _env_int("SQLRAG_NUM_CTX", 8192))
    num_predict: int = field(default_factory=lambda: _env_int("SQLRAG_NUM_PREDICT", 1024))
    # Ollama layers to place on the GPU (-1 = let Ollama decide). On a 4 GB GPU Ollama's
    # estimate leaves a 3B model partly on the CPU; 99 forces full offload.
    num_gpu: int = field(default_factory=lambda: _env_int("SQLRAG_NUM_GPU", -1))
    request_timeout_s: float = field(default_factory=lambda: _env_float("SQLRAG_REQUEST_TIMEOUT", 240.0))

    # Prompts -----------------------------------------------------------
    baseline_prompt: pathlib.Path = field(default_factory=lambda: _path("SQLRAG_BASELINE_PROMPT", ROOT / "prompts" / "v2.md"))
    rag_prompt: pathlib.Path = field(default_factory=lambda: _path("SQLRAG_RAG_PROMPT", ROOT / "prompts" / "v4_rag.md"))
    judge_prompt: pathlib.Path = field(default_factory=lambda: _path("SQLRAG_JUDGE_PROMPT", ROOT / "prompts" / "judge_v2_rubric.md"))

    # API ---------------------------------------------------------------
    admin_token: str = field(default_factory=lambda: _env("SQLRAG_ADMIN_TOKEN", ""))
    cors_origins: str = field(default_factory=lambda: _env("SQLRAG_CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"))
    max_upload_mb: float = field(default_factory=lambda: _env_float("SQLRAG_MAX_UPLOAD_MB", 10.0))
    allow_ingest: bool = field(default_factory=lambda: _env_bool("SQLRAG_ALLOW_INGEST", True))

    def generation_options(self) -> dict:
        """Sampling options shared by the baseline and RAG generators and the judge."""
        opts = {
            "temperature": self.temperature,
            "seed": self.seed,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }
        if self.num_gpu >= 0:
            opts["num_gpu"] = self.num_gpu
        return opts

    def retrieval_config(self) -> dict:
        return {
            "mode": self.retrieval_mode,
            "top_k": self.top_k,
            "candidates": self.candidates,
            "per_query_k": self.per_query_k,
            "rrf_k": self.rrf_k,
            "min_similarity": self.min_similarity,
            "max_context_tokens": self.max_context_tokens,
            "fill_context": self.fill_context,
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "grounding_policy": self.grounding_policy,
            "min_feature_coverage": self.min_feature_coverage,
        }

    def public_dict(self) -> dict:
        """Settings that are safe to show in the UI or store in experiment metadata."""
        out = {}
        for f in fields(self):
            if f.name in {"openai_api_key", "admin_token"}:
                continue
            value = getattr(self, f.name)
            out[f.name] = str(value) if isinstance(value, pathlib.Path) else value
        return out


def get_settings(**overrides) -> Settings:
    settings = Settings()
    for key, value in overrides.items():
        if value is not None:
            setattr(settings, key, value)
    return settings
