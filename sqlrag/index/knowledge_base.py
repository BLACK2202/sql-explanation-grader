"""The knowledge base: ingestion pipeline + persistent index + registry.

Pipeline for each document:
    parse -> clean -> validate -> content hash -> (skip if unchanged or a
    duplicate) -> chunk -> drop chunks already indexed for another document
    -> embed -> replace the document's old chunks -> update the registry

Ingestion is idempotent: running it twice with the same files and settings
changes nothing the second time. A document whose text changed has its old
chunks deleted before the new ones are written, so no stale chunks remain.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import pathlib
import threading
import time
from dataclasses import dataclass, field

from ..config import ROOT, Settings
from ..ingest.chunking import ChunkingConfig, chunk_document
from ..ingest.cleaning import clean_document, validate_document
from ..ingest.models import Chunk, DocumentMeta, ParsedDocument
from ..ingest.parsers import parse_bytes
from .bm25 import BM25Index
from .embeddings import Embedder, get_embedder
from .vector_store import Hit, VectorStore, collection_name_for

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    duplicates: list[dict] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    chunks_written: int = 0
    chunks_deleted: int = 0
    chunks_skipped_duplicate: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def load_manifest(path: pathlib.Path) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def manifest_entries(manifest: dict, *, include_optional: bool = False, doc_ids: list[str] | None = None) -> list[tuple[pathlib.Path, DocumentMeta]]:
    lock_path = ROOT / "corpus" / "manifest.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    out = []
    for d in manifest["documents"]:
        if doc_ids is not None and d["doc_id"] not in doc_ids:
            continue
        if doc_ids is None and not include_optional and not d.get("ingest_by_default", True):
            continue
        meta = DocumentMeta(
            doc_id=d["doc_id"],
            title=d.get("title", ""),
            source_url=d.get("url", ""),
            license=d.get("license", "unknown"),
            doc_type=d.get("doc_type", "reference"),
            dialect=d.get("dialect", "sqlite"),
            categories=tuple(d.get("categories", [])),
            path=d["path"],
            retrieved_at=lock.get(d["doc_id"], {}).get("retrieved_at", ""),
        )
        out.append((ROOT / d["path"], meta))
    return out


class KnowledgeBase:
    def __init__(self, settings: Settings, *, embedder: Embedder | None = None, in_memory: bool = False) -> None:
        self.settings = settings
        self.embedder = embedder or get_embedder(
            settings.embedding_backend, settings.embedding_model, settings.embedding_device, settings.ollama_host
        )
        self.chunking = ChunkingConfig(settings.chunk_size, settings.chunk_overlap, settings.min_chunk_tokens)
        self.collection_name = collection_name_for(f"{self.embedder.name}-c{self.chunking.chunk_size}")
        chroma_path = None if in_memory else pathlib.Path(settings.store_dir) / "chroma"
        self.store = VectorStore(chroma_path, self.collection_name, embedding_name=self.embedder.name, dimension=self.embedder.dimension)
        self.registry_path = None if in_memory else pathlib.Path(settings.store_dir) / f"{self.collection_name}.registry.json"
        self.registry = self._load_registry()
        self._lock = threading.RLock()
        self._bm25: BM25Index | None = None
        self._chunk_cache: dict[str, Hit] | None = None

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------
    def _load_registry(self) -> dict:
        if self.registry_path and self.registry_path.exists():
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        return {"version": 1, "collection": self.collection_name, "documents": {}}

    def _save_registry(self) -> None:
        self.registry.update(
            {
                "collection": self.collection_name,
                "embedding_model": self.embedder.name,
                "dimension": self.embedder.dimension,
                "chunking": {**self.chunking.__dict__, "fingerprint": self.chunking.fingerprint},
                "updated_at": _now(),
            }
        )
        if self.registry_path is None:
            return
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.registry_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.registry, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.registry_path)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def _invalidate(self) -> None:
        self._bm25 = None
        self._chunk_cache = None

    def parse(self, data: bytes, filename: str, meta: DocumentMeta) -> ParsedDocument:
        doc = clean_document(parse_bytes(data, filename, meta))
        validate_document(doc)
        return doc

    def ingest(self, items: list[tuple[pathlib.Path | bytes, DocumentMeta] | tuple[bytes, DocumentMeta, str]], *, force: bool = False) -> IngestReport:
        """Ingest (path, meta) pairs or (bytes, meta, filename) triples."""
        report = IngestReport()
        t0 = time.time()
        with self._lock:
            docs_reg: dict = self.registry.setdefault("documents", {})
            for item in items:
                meta = item[1]
                try:
                    if isinstance(item[0], (bytes, bytearray)):
                        data, filename = bytes(item[0]), item[2]  # type: ignore[misc]
                    else:
                        path = pathlib.Path(item[0])
                        data, filename = path.read_bytes(), path.name
                    doc = self.parse(data, filename, meta)
                except Exception as exc:  # parse/validation failures are reported, not fatal
                    logger.warning("ingest failed for %s: %s", meta.doc_id, exc)
                    report.failed.append({"doc_id": meta.doc_id, "error": str(exc)})
                    docs_reg[meta.doc_id] = {**docs_reg.get(meta.doc_id, {}), "status": "failed", "error": str(exc), "title": meta.title}
                    continue

                content_hash = doc.content_hash
                prev = docs_reg.get(meta.doc_id)
                in_store = self.store.count_document(meta.doc_id)
                if (
                    not force and prev and prev.get("status") == "indexed"
                    and prev.get("content_hash") == content_hash
                    and prev.get("chunk_fingerprint") == self.chunking.fingerprint
                    and in_store == prev.get("n_chunks")
                ):
                    report.unchanged.append(meta.doc_id)
                    continue

                dup_of = next(
                    (did for did, rec in docs_reg.items()
                     if did != meta.doc_id and rec.get("status") == "indexed" and rec.get("content_hash") == content_hash),
                    None,
                )
                if dup_of:
                    report.duplicates.append({"doc_id": meta.doc_id, "duplicate_of": dup_of})
                    docs_reg[meta.doc_id] = {"status": "duplicate", "duplicate_of": dup_of, "title": doc.meta.title,
                                             "content_hash": content_hash, "checked_at": _now()}
                    continue

                chunks = chunk_document(doc, self.chunking)
                chunks, skipped = self._drop_duplicate_chunks(chunks, meta.doc_id)
                report.chunks_skipped_duplicate += skipped
                embeddings = self.embedder.embed_documents([c.index_text for c in chunks]) if chunks else []
                report.chunks_deleted += self.store.delete_document(meta.doc_id)
                if chunks:
                    self.store.upsert(
                        [c.chunk_id for c in chunks], embeddings, [c.text for c in chunks], [c.metadata() for c in chunks]
                    )
                report.chunks_written += len(chunks)
                over_window = None
                if self.embedder.max_seq_length:
                    over_window = sum(
                        1 for c in chunks if (self.embedder.count_model_tokens(c.index_text) or 0) > self.embedder.max_seq_length
                    )
                docs_reg[meta.doc_id] = {
                    "status": "indexed",
                    "title": doc.meta.title,
                    "source_url": doc.meta.source_url,
                    "license": doc.meta.license,
                    "doc_type": doc.meta.doc_type,
                    "dialect": doc.meta.dialect,
                    "categories": list(doc.meta.categories),
                    "path": doc.meta.path,
                    "retrieved_at": doc.meta.retrieved_at,
                    "content_hash": content_hash,
                    "chunk_fingerprint": self.chunking.fingerprint,
                    "n_chunks": len(chunks),
                    "n_tokens": sum(c.n_tokens for c in chunks),
                    "n_blocks": len(doc.blocks),
                    "chunks_over_embedding_window": over_window,
                    "ingested_at": _now(),
                }
                (report.updated if prev and prev.get("status") == "indexed" else report.added).append(meta.doc_id)
            self._save_registry()
            self._invalidate()
        report.seconds = round(time.time() - t0, 2)
        return report

    def _drop_duplicate_chunks(self, chunks: list[Chunk], doc_id: str) -> tuple[list[Chunk], int]:
        """Skip chunks whose exact text is already indexed for another document."""
        existing = {
            h.metadata.get("content_hash")
            for h in self.all_chunks().values()
            if h.metadata.get("doc_id") != doc_id
        }
        seen: set[str] = set()
        kept = []
        for c in chunks:
            if c.content_hash in existing or c.content_hash in seen:
                continue
            seen.add(c.content_hash)
            kept.append(c)
        return kept, len(chunks) - len(kept)

    def ingest_manifest(self, *, include_optional: bool = False, doc_ids: list[str] | None = None, force: bool = False) -> IngestReport:
        manifest = load_manifest(self.settings.corpus_manifest)
        return self.ingest(manifest_entries(manifest, include_optional=include_optional, doc_ids=doc_ids), force=force)

    def remove_document(self, doc_id: str) -> int:
        with self._lock:
            n = self.store.delete_document(doc_id)
            self.registry.get("documents", {}).pop(doc_id, None)
            self._save_registry()
            self._invalidate()
            return n

    def rebuild(self) -> None:
        with self._lock:
            self.store.reset()
            self.registry["documents"] = {}
            self._save_registry()
            self._invalidate()

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------
    def all_chunks(self) -> dict[str, Hit]:
        if self._chunk_cache is None:
            self._chunk_cache = {h.chunk_id: h for h in self.store.get_all()}
        return self._chunk_cache

    def bm25(self) -> BM25Index:
        with self._lock:
            if self._bm25 is None:
                chunks = self.all_chunks()
                ids = sorted(chunks)
                texts = [f"{chunks[i].metadata.get('title', '')} > {chunks[i].metadata.get('section', '')}\n{chunks[i].text}" for i in ids]
                self._bm25 = BM25Index().build(ids, texts)
            return self._bm25

    def list_documents(self) -> list[dict]:
        docs = []
        for doc_id, rec in sorted(self.registry.get("documents", {}).items()):
            docs.append({"doc_id": doc_id, **rec})
        return docs

    def stats(self) -> dict:
        docs = self.registry.get("documents", {})
        indexed = [d for d in docs.values() if d.get("status") == "indexed"]
        chunks = self.all_chunks()
        tokens = [int(h.metadata.get("n_tokens", 0)) for h in chunks.values()]
        return {
            "collection": self.collection_name,
            "embedding_model": self.embedder.name,
            "embedding_dimension": self.embedder.dimension,
            "embedding_max_seq_length": self.embedder.max_seq_length,
            "chunking": {**self.chunking.__dict__, "fingerprint": self.chunking.fingerprint},
            "documents_indexed": len(indexed),
            "documents_failed": sum(1 for d in docs.values() if d.get("status") == "failed"),
            "documents_duplicate": sum(1 for d in docs.values() if d.get("status") == "duplicate"),
            "chunks": len(chunks),
            "chunks_in_vector_store": self.store.count(),
            "avg_chunk_tokens": round(sum(tokens) / len(tokens), 1) if tokens else 0,
            "max_chunk_tokens": max(tokens) if tokens else 0,
            "chunks_over_embedding_window": sum(int(d.get("chunks_over_embedding_window") or 0) for d in indexed),
            "updated_at": self.registry.get("updated_at"),
        }
