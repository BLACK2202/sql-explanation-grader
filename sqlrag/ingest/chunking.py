"""Structure-aware chunking.

Design rules (each one is covered by tests/test_chunking.py):

1. Chunks never mix two documents.
2. A chunk normally stays inside one section. A section that is too small
   to stand alone (< min_chunk_tokens) is merged with the next one, and the
   chunk's section label lists both ("A / B").
3. SQL code examples are atomic: a code block is never split unless it is
   longer than a whole chunk, and then only at line boundaries.
4. Long paragraphs are split at sentence boundaries.
5. Consecutive chunks of the same section overlap by up to `overlap` tokens
   of whole sentences, so a fact cut at a boundary is still retrievable.
6. Chunk ids contain the document id, the chunk index and a content hash, so
   re-chunking unchanged text gives the same ids (idempotent ingestion).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from .cleaning import count_tokens
from .models import Block, Chunk, ParsedDocument, sha256_text

_SENTENCE = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9(\"'`])")


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 400
    overlap: int = 60
    min_chunk_tokens: int = 60

    def __post_init__(self) -> None:
        if self.chunk_size < 50:
            raise ValueError("chunk_size must be at least 50 tokens")
        if not 0 <= self.overlap < self.chunk_size // 2:
            raise ValueError("overlap must be >= 0 and less than half of chunk_size")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps({"v": 2, **self.__dict__}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


@dataclass
class _Unit:
    block_idx: int
    kind: str
    text: str
    tokens: int
    section: tuple[str, ...]
    page: int | None


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE.split(text) if p.strip()]
    return parts or [text]


def _units_for_block(idx: int, block: Block, cfg: ChunkingConfig) -> list[_Unit]:
    tokens = count_tokens(block.text)
    if block.kind == "code":
        if tokens <= cfg.chunk_size:
            return [_Unit(idx, "code", block.text, tokens, block.section_path, block.page)]
        # Oversized example: split at line boundaries into chunk-sized pieces.
        units, cur, cur_tok = [], [], 0
        for line in block.text.split("\n"):
            t = count_tokens(line) + 1
            if cur and cur_tok + t > cfg.chunk_size:
                units.append(_Unit(idx, "code", "\n".join(cur), cur_tok, block.section_path, block.page))
                cur, cur_tok = [], 0
            cur.append(line)
            cur_tok += t
        if cur:
            units.append(_Unit(idx, "code", "\n".join(cur), cur_tok, block.section_path, block.page))
        return units
    if block.kind == "table":
        rows = block.text.split("\n")
        units, cur, cur_tok = [], [], 0
        for row in rows:
            t = count_tokens(row)
            if cur and cur_tok + t > cfg.chunk_size // 2:
                units.append(_Unit(idx, "table", "\n".join(cur), cur_tok, block.section_path, block.page))
                cur, cur_tok = [], 0
            cur.append(row)
            cur_tok += t
        if cur:
            units.append(_Unit(idx, "table", "\n".join(cur), cur_tok, block.section_path, block.page))
        return units
    # Paragraph: sentence units so overlap and splitting respect sentences.
    units = []
    for sentence in _split_sentences(block.text):
        t = count_tokens(sentence)
        if t > cfg.chunk_size:  # pathological run-on text: hard split on words
            words = sentence.split()
            step = max(1, int(len(words) * cfg.chunk_size / t))
            for i in range(0, len(words), step):
                piece = " ".join(words[i:i + step])
                units.append(_Unit(idx, "paragraph", piece, count_tokens(piece), block.section_path, block.page))
        else:
            units.append(_Unit(idx, "paragraph", sentence, t, block.section_path, block.page))
    return units


def _render(units: list[_Unit]) -> str:
    out: list[str] = []
    prev: _Unit | None = None
    for u in units:
        body = f"```\n{u.text}\n```" if u.kind == "code" else u.text
        if prev is None:
            out.append(body)
        elif prev.block_idx == u.block_idx and u.kind == "paragraph":
            out.append(" " + body)
        elif prev.block_idx == u.block_idx:
            out.append("\n" + body)
        else:
            out.append("\n\n" + body)
        prev = u
    return "".join(out).strip()


def _section_label(units: list[_Unit]) -> str:
    labels: list[str] = []
    for u in units:
        label = " > ".join(u.section)
        if label and label not in labels:
            labels.append(label)
    return " / ".join(labels[:3])


def _only_overlap(current: list[_Unit], groups: list[list[_Unit]]) -> bool:
    """True when `current` holds nothing but the overlap copied from the last chunk."""
    if not groups or not current:
        return False
    last = groups[-1]
    return all(any(u is g for g in last) for u in current)


def chunk_document(doc: ParsedDocument, cfg: ChunkingConfig | None = None) -> list[Chunk]:
    cfg = cfg or ChunkingConfig()
    body_blocks = [(i, b) for i, b in enumerate(doc.blocks) if b.kind != "heading"]

    groups: list[list[_Unit]] = []   # one entry per emitted chunk
    current: list[_Unit] = []
    current_tokens = 0
    current_section: tuple[str, ...] | None = None

    def emit(carry_overlap: bool) -> None:
        nonlocal current, current_tokens
        if not current:
            return
        groups.append(current)
        tail: list[_Unit] = []
        if carry_overlap and cfg.overlap:
            budget = cfg.overlap
            for u in reversed(current):
                if u.kind != "paragraph" or u.tokens > budget:
                    break
                tail.insert(0, u)
                budget -= u.tokens
        current = tail
        current_tokens = sum(u.tokens for u in tail)

    for idx, block in body_blocks:
        if current_section is not None and block.section_path != current_section:
            # Section boundary: close the chunk unless it is too small to stand alone.
            if current_tokens >= cfg.min_chunk_tokens:
                emit(carry_overlap=False)
        current_section = block.section_path
        for unit in _units_for_block(idx, block, cfg):
            if current and current_tokens + unit.tokens > cfg.chunk_size:
                if _only_overlap(current, groups):
                    current, current_tokens = [], 0
                else:
                    emit(carry_overlap=True)
                    if current_tokens + unit.tokens > cfg.chunk_size:
                        current, current_tokens = [], 0
            current.append(unit)
            current_tokens += unit.tokens
    if current and not _only_overlap(current, groups):
        groups.append(current)

    chunks: list[Chunk] = []
    for units in groups:
        text = _render(units)
        if not text:
            continue
        pages = [u.page for u in units if u.page is not None]
        content_hash = sha256_text(text)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.meta.doc_id}::{len(chunks):04d}::{content_hash[:10]}",
                doc_id=doc.meta.doc_id,
                chunk_index=len(chunks),
                text=text,
                section=_section_label(units),
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                n_tokens=count_tokens(text),
                content_hash=content_hash,
                meta=doc.meta,
            )
        )
    return chunks
