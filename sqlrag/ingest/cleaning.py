"""Text normalisation and validation applied to every parsed block."""
from __future__ import annotations

import re
import unicodedata

from .models import Block, ParsedDocument

_WS = re.compile(r"[ \t\f\v  -​  　]+")
_MANY_NL = re.compile(r"\n{3,}")
_HEADING_NUMBER = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[A-Z]\.)\s+")
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_DIAGRAM_CAPTION = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*:$")


def count_tokens(text: str) -> int:
    """Approximate sub-word token count (words + punctuation marks).

    Within ~15% of WordPiece counts on English documentation, which is good
    enough for chunk sizing. The exact embedding-tokenizer count is reported
    by the index statistics (see sqlrag.index.embeddings).
    """
    return len(_TOKEN.findall(text))


def normalize_prose(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = re.sub(r" *\n *", " ", text)
    return text.strip()


def normalize_code(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    # Remove common indentation so examples render cleanly.
    indents = [len(ln) - len(ln.lstrip(" ")) for ln in lines if ln.strip()]
    cut = min(indents) if indents else 0
    return _MANY_NL.sub("\n\n", "\n".join(ln[cut:] for ln in lines))


def clean_heading(text: str) -> str:
    text = normalize_prose(text)
    text = text.replace("¶", "").rstrip("#").strip()
    text = _HEADING_NUMBER.sub("", text)
    return text.strip(" .:")


def clean_document(doc: ParsedDocument, *, min_paragraph_chars: int = 3) -> ParsedDocument:
    """Normalise whitespace, drop empty or duplicate-consecutive blocks."""
    cleaned: list[Block] = []
    for block in doc.blocks:
        if block.kind == "code":
            text = normalize_code(block.text)
        elif block.kind == "heading":
            text = clean_heading(block.text)
        elif block.kind == "table":
            text = "\n".join(normalize_prose(row) for row in block.text.split("\n") if row.strip())
        else:
            text = normalize_prose(block.text)
        if len(text) < min_paragraph_chars:
            continue
        if block.kind == "paragraph" and _DIAGRAM_CAPTION.match(text):
            continue  # caption of a syntax diagram whose image was removed
        if cleaned and cleaned[-1].kind == block.kind and cleaned[-1].text == text:
            continue
        cleaned.append(Block(block.kind, text, block.level, block.section_path, block.page))
    return ParsedDocument(meta=doc.meta, blocks=cleaned)


class DocumentValidationError(ValueError):
    pass


def validate_document(doc: ParsedDocument, *, min_tokens: int = 20) -> None:
    """Reject documents that are empty or look like binary garbage."""
    text = doc.text
    if count_tokens(text) < min_tokens:
        raise DocumentValidationError(f"{doc.meta.doc_id}: too little text after parsing ({count_tokens(text)} tokens)")
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in text)
    if printable / max(1, len(text)) < 0.95:
        raise DocumentValidationError(f"{doc.meta.doc_id}: text is mostly non-printable; is this a scanned or binary file?")
    if not any(b.kind in {"paragraph", "code", "table"} for b in doc.blocks):
        raise DocumentValidationError(f"{doc.meta.doc_id}: no body text found")
