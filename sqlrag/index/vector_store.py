"""Persistent vector store backed by ChromaDB.

ChromaDB keeps the chunk text, its metadata and its embedding on disk
(``rag_store/chroma``) and answers nearest-neighbour queries with an HNSW
index using cosine distance. We pass our own embeddings, so the store never
downloads or runs a model by itself.
"""
from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass


@dataclass
class Hit:
    chunk_id: str
    text: str
    metadata: dict
    similarity: float | None = None


def collection_name_for(embedding_name: str, prefix: str = "sql_docs") -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", embedding_name).strip("-._").lower()
    name = f"{prefix}__{slug}"[:60]
    return name.rstrip("-._") or prefix


class VectorStore:
    def __init__(self, path: pathlib.Path | str | None, collection: str, *, embedding_name: str, dimension: int) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        chroma_settings = ChromaSettings(anonymized_telemetry=False, allow_reset=False)
        if path is None:
            self.client = chromadb.EphemeralClient(settings=chroma_settings)
        else:
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(path), settings=chroma_settings)
        self.collection_name = collection
        self.embedding_name = embedding_name
        self.dimension = dimension
        self.collection = self.client.get_or_create_collection(
            name=collection,
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "embedding_model": embedding_name, "dimension": dimension},
        )
        stored = (self.collection.metadata or {}).get("embedding_model")
        if stored and stored != embedding_name:
            raise RuntimeError(
                f"collection {collection!r} was built with {stored!r}, not {embedding_name!r}; "
                "re-ingest with --rebuild"
            )

    # -- writes ----------------------------------------------------------
    def upsert(self, ids: list[str], embeddings: list[list[float]], texts: list[str], metadatas: list[dict]) -> None:
        for start in range(0, len(ids), 256):
            sl = slice(start, start + 256)
            self.collection.upsert(ids=ids[sl], embeddings=embeddings[sl], documents=texts[sl], metadatas=metadatas[sl])

    def delete_document(self, doc_id: str) -> int:
        existing = self.collection.get(where={"doc_id": doc_id}, include=[])
        ids = existing.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def reset(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_name, "dimension": self.dimension},
        )

    # -- reads -----------------------------------------------------------
    def count(self) -> int:
        return self.collection.count()

    def count_document(self, doc_id: str) -> int:
        return len(self.collection.get(where={"doc_id": doc_id}, include=[]).get("ids", []))

    def query(self, embedding: list[float], k: int, where: dict | None = None) -> list[Hit]:
        n = self.count()
        if n == 0:
            return []
        res = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(k, n),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for cid, text, md, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(Hit(cid, text, dict(md or {}), similarity=1.0 - float(dist)))
        return hits

    def get_all(self, where: dict | None = None) -> list[Hit]:
        res = self.collection.get(where=where or None, include=["documents", "metadatas"])
        return [Hit(cid, text, dict(md or {})) for cid, text, md in zip(res["ids"], res["documents"], res["metadatas"])]

    def get_embeddings(self, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}
        res = self.collection.get(ids=ids, include=["embeddings"])
        return {cid: list(map(float, emb)) for cid, emb in zip(res["ids"], res["embeddings"])}
