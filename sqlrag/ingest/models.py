"""Plain data containers shared by the ingestion stages."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

BlockKind = Literal["heading", "paragraph", "code", "table"]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Block:
    """One structural unit of a parsed document."""

    kind: BlockKind
    text: str
    level: int = 0                       # heading level (1 = top) for headings
    section_path: tuple[str, ...] = ()   # headings above this block
    page: int | None = None              # 1-based page number (PDFs only)


@dataclass
class DocumentMeta:
    doc_id: str
    title: str
    source_url: str = ""
    license: str = "unknown"
    doc_type: str = "user_upload"
    dialect: str = "sqlite"
    categories: tuple[str, ...] = ()
    path: str = ""
    retrieved_at: str = ""


@dataclass
class ParsedDocument:
    meta: DocumentMeta
    blocks: list[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)

    @property
    def content_hash(self) -> str:
        """Hash of the cleaned text; identical content gives an identical hash."""
        return sha256_text(self.text)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str                 # body text shown to users and to the LLM
    section: str              # "Heading > Subheading" (may join several short sections with " / ")
    page_start: int | None
    page_end: int | None
    n_tokens: int
    content_hash: str
    meta: DocumentMeta

    @property
    def header(self) -> str:
        return f"{self.meta.title} > {self.section}" if self.section else self.meta.title

    @property
    def index_text(self) -> str:
        """Text used for embeddings and BM25: a context header plus the body."""
        return f"{self.header}\n{self.text}"

    def metadata(self) -> dict:
        """Flat metadata for the vector store (values must be str/int/float/bool)."""
        md: dict = {
            "doc_id": self.doc_id,
            "title": self.meta.title,
            "section": self.section,
            "source_url": self.meta.source_url,
            "license": self.meta.license,
            "doc_type": self.meta.doc_type,
            "dialect": self.meta.dialect,
            "categories": ",".join(self.meta.categories),
            "chunk_index": self.chunk_index,
            "n_tokens": self.n_tokens,
            "content_hash": self.content_hash,
            "page_start": self.page_start if self.page_start is not None else -1,
            "page_end": self.page_end if self.page_end is not None else -1,
        }
        for cat in self.meta.categories:
            md[f"cat__{cat}"] = True
        return md
