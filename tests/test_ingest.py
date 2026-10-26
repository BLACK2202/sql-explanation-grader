"""Parsing, cleaning and chunking."""
from __future__ import annotations

import pytest

from conftest import FIXTURES
from sqlrag.ingest.chunking import ChunkingConfig, chunk_document
from sqlrag.ingest.cleaning import DocumentValidationError, clean_document, count_tokens, validate_document
from sqlrag.ingest.models import Block, DocumentMeta, ParsedDocument
from sqlrag.ingest.parsers import UnsupportedDocumentError, parse_bytes, parse_file


def meta(doc_id="d1"):
    return DocumentMeta(doc_id=doc_id, title="", license="test")


def test_html_parser_keeps_structure_and_drops_noise():
    doc = clean_document(parse_file(FIXTURES / "joins.html", meta()))
    text = doc.text
    assert "never executed" not in text          # <script>
    assert "color:red" not in text               # <style>
    assert "Home" not in text and "Copyright footer" not in text  # menu, footer
    assert "select-stmt:" not in text             # orphan syntax-diagram caption
    headings = [b.text for b in doc.blocks if b.kind == "heading"]
    assert headings == ["Joining tables", "INNER JOIN", "LEFT JOIN"]  # numbering removed
    code = [b for b in doc.blocks if b.kind == "code"]
    assert len(code) == 1 and "JOIN orders o ON o.customer_id = c.id" in code[0].text
    left = [b for b in doc.blocks if "LEFT OUTER JOIN" in b.text][0]
    assert left.section_path == ("Joining tables", "LEFT JOIN")
    assert any(b.kind == "table" and "kept, right side NULL" in b.text for b in doc.blocks)


def test_markdown_parser_headings_fences_lists():
    doc = clean_document(parse_file(FIXTURES / "grouping.md", meta()))
    assert doc.meta.title == "Grouping rows"
    kinds = [b.kind for b in doc.blocks]
    assert "code" in kinds
    code = next(b for b in doc.blocks if b.kind == "code")
    assert code.text.startswith("SELECT city, count(*)")
    having = next(b for b in doc.blocks if b.text.startswith("The HAVING clause"))
    assert having.section_path == ("Grouping rows", "HAVING")


def test_text_parser_paragraphs_and_indented_code():
    doc = clean_document(parse_file(FIXTURES / "ordering.txt", meta()))
    assert [b.kind for b in doc.blocks] == ["paragraph", "paragraph", "code"]


def test_pdf_parser_extracts_pages_and_headings(tmp_path):
    fitz = pytest.importorskip("fitz")
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Window functions", fontsize=20)
    page.insert_text((72, 110), "A window function computes a value for every row using related rows.", fontsize=11)
    page.insert_text((72, 128), "Rows are not collapsed into groups, unlike GROUP BY.", fontsize=11)
    page2 = pdf.new_page()
    page2.insert_text((72, 72), "The PARTITION BY clause splits rows into partitions for the calculation.", fontsize=11)
    data = pdf.tobytes()
    doc = clean_document(parse_bytes(data, "notes.pdf", meta("pdf1")))
    assert any(b.kind == "heading" and b.text == "Window functions" for b in doc.blocks)
    pages = {b.page for b in doc.blocks}
    assert pages == {1, 2}
    chunks = chunk_document(doc, ChunkingConfig(100, 10, 5))
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 2


def test_rejects_unsupported_and_fake_pdf():
    with pytest.raises(UnsupportedDocumentError):
        parse_bytes(b"MZ\x90\x00", "virus.exe", meta())
    with pytest.raises(UnsupportedDocumentError):
        parse_bytes(b"not a pdf", "fake.pdf", meta())


def test_validation_rejects_empty_documents():
    doc = ParsedDocument(meta(), [Block("heading", "Only a title")])
    with pytest.raises(DocumentValidationError):
        validate_document(doc)


def _doc(blocks):
    return ParsedDocument(DocumentMeta(doc_id="c1", title="T"), blocks)


def test_chunking_never_splits_code_blocks():
    code = "SELECT a,\n       b\nFROM t\nWHERE a > 1\nORDER BY b;"
    prose = " ".join(f"Sentence number {i} explains something about SQL." for i in range(40))
    doc = _doc([Block("paragraph", prose, section_path=("S",)), Block("code", code, section_path=("S",)),
                Block("paragraph", prose, section_path=("S",))])
    chunks = chunk_document(doc, ChunkingConfig(60, 10, 10))
    holders = [c for c in chunks if "WHERE a > 1" in c.text]
    assert len(holders) == 1 and "SELECT a," in holders[0].text and "ORDER BY b;" in holders[0].text
    assert all(c.n_tokens <= 60 + 15 for c in chunks)


def test_chunking_respects_sections_and_merges_tiny_ones():
    big = " ".join(f"Rows in part {i} are filtered here." for i in range(12))
    doc = _doc([
        Block("paragraph", "Tiny intro.", section_path=("A",)),
        Block("paragraph", big, section_path=("B",)),
        Block("paragraph", big, section_path=("C",)),
    ])
    chunks = chunk_document(doc, ChunkingConfig(80, 0, 20))
    assert chunks[0].section.startswith("A / B")          # tiny section merged forward
    assert not any("A" in c.section and "C" in c.section for c in chunks)
    assert any(c.section == "C" for c in chunks)


def test_chunk_overlap_repeats_trailing_sentences():
    prose = " ".join(f"Fact {i} is stated in this sentence." for i in range(30))
    doc = _doc([Block("paragraph", prose, section_path=("S",))])
    chunks = chunk_document(doc, ChunkingConfig(60, 20, 10))
    assert len(chunks) >= 3
    last_sentence = chunks[0].text.split(". ")[-1].rstrip(".")
    assert last_sentence in chunks[1].text


def test_chunk_ids_are_deterministic_and_content_addressed():
    doc = clean_document(parse_file(FIXTURES / "grouping.md", meta("g")))
    a = [c.chunk_id for c in chunk_document(doc, ChunkingConfig(80, 10, 10))]
    b = [c.chunk_id for c in chunk_document(doc, ChunkingConfig(80, 10, 10))]
    assert a == b
    assert all(cid.startswith("g::") for cid in a)


def test_chunking_config_validation_and_fingerprint():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=20)
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=100, overlap=60)
    assert ChunkingConfig(400, 60).fingerprint != ChunkingConfig(400, 50).fingerprint


def test_count_tokens_is_reasonable():
    assert count_tokens("SELECT a, b FROM t;") == 7  # SELECT a , b FROM t ;
