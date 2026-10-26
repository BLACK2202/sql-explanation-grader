"""Embedding backends.

An embedding model maps a passage of text to a fixed-length vector so that
passages about similar topics end up close together. The vectors are used
only to *find* passages; the LLM never sees them. The LLM receives the text
of the passages that retrieval selects.

Backends:
  sentence-transformers  default; BAAI/bge-small-en-v1.5 (384 dims, 33M params,
                         512-token window, runs on CPU)
  hash                   deterministic feature-hashing vectors with no model
                         download; used by the offline test-suite and CI. It
                         is lexical, not semantic, and is not meant for real use.
  ollama                 any Ollama embedding model (e.g. nomic-embed-text)
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

_TOKEN = re.compile(r"[a-z0-9_]+")

# BGE models were trained with this instruction prepended to short queries.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    name: str
    dimension: int
    max_seq_length: int | None

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def count_model_tokens(self, text: str) -> int | None: ...


def _l2(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class HashEmbedder:
    """Feature-hashing bag of words + bigrams. Deterministic and dependency-free."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.name = f"hash-{dimension}"
        self.max_seq_length = None

    def _vector(self, text: str) -> list[float]:
        tokens = _TOKEN.findall(text.lower())
        feats = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        vec = [0.0] * self.dimension
        for feat in feats:
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
            vec[h % self.dimension] += 1.0 if (h >> 63) & 1 else -1.0
        return _l2(vec)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def count_model_tokens(self, text: str) -> int | None:
        return None


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu", batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import kept lazy

        self.model = SentenceTransformer(model_name, device=device)
        self.name = model_name
        self.batch_size = batch_size
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dimension = int(get_dim())
        self.max_seq_length = int(getattr(self.model, "max_seq_length", 512) or 512)
        self.query_instruction = BGE_QUERY_INSTRUCTION if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        vec = self.model.encode([self.query_instruction + text], normalize_embeddings=True, show_progress_bar=False)[0]
        return vec.tolist()

    def count_model_tokens(self, text: str) -> int | None:
        return len(self.model.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])


class OllamaEmbedder:
    def __init__(self, model_name: str = "nomic-embed-text", host: str = "http://localhost:11434") -> None:
        import ollama

        self.client = ollama.Client(host=host)
        self.name = f"ollama/{model_name}"
        self.model_name = model_name
        self.max_seq_length = None
        self.dimension = len(self.embed_query("dimension probe"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out = self.client.embed(model=self.model_name, input=texts)
        return [_l2(list(v)) for v in out["embeddings"]]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def count_model_tokens(self, text: str) -> int | None:
        return None


_CACHE: dict[tuple[str, str, str], Embedder] = {}


def get_embedder(backend: str, model_name: str, device: str = "cpu", ollama_host: str = "http://localhost:11434") -> Embedder:
    """Return a (cached) embedder. Loading a model takes seconds, so reuse it."""
    key = (backend, model_name, device)
    if key in _CACHE:
        return _CACHE[key]
    if backend == "hash":
        emb: Embedder = HashEmbedder()
    elif backend == "sentence-transformers":
        emb = SentenceTransformerEmbedder(model_name, device=device)
    elif backend == "ollama":
        emb = OllamaEmbedder(model_name, host=ollama_host)
    else:
        raise ValueError(f"unknown embedding backend {backend!r}")
    _CACHE[key] = emb
    return emb
