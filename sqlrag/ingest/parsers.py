"""Parsers that turn HTML, PDF, Markdown and plain-text files into blocks.

Every parser returns a ParsedDocument whose blocks keep the document's
reading order and the heading path above each block. Parsers only read
files; they never execute embedded content (scripts, macros, links).
"""
from __future__ import annotations

import pathlib
import re

from .models import Block, DocumentMeta, ParsedDocument

SUPPORTED_EXTENSIONS = {".html", ".htm", ".pdf", ".md", ".markdown", ".txt"}


class UnsupportedDocumentError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _SectionTracker:
    """Keeps the current heading path while walking a document."""

    def __init__(self) -> None:
        self.stack: list[tuple[int, str]] = []

    def push(self, level: int, text: str) -> None:
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, text))

    @property
    def path(self) -> tuple[str, ...]:
        return tuple(text for _, text in self.stack)


def _heading_text(raw: str) -> str:
    from .cleaning import clean_heading

    return clean_heading(raw)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
_NOISE_CLASS = re.compile(
    r"(^|[\s_-])(menu|mainmenu|submenu|searchmenu|nav|navbar|navheader|navfooter|toc|fancy_toc|fancy-toc\d?|"
    r"breadcrumb|footer|header|tagline|logo|sidebar|id_link|skip|cookie|banner)($|[\s_-])",
    re.IGNORECASE,
)
_NOISE_IDS = {"docComments", "pgFooter", "pgHeader", "toc", "menu", "footer", "header"}
_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_CONTAINERS = {"div", "section", "article", "main", "body", "ul", "ol", "dl", "blockquote", "span", "center", "font", "form"}
_PARAGRAPHS = {"p", "li", "dt", "dd", "caption", "figcaption"}


def _strip_noise(soup) -> None:
    for tag in soup(["script", "style", "svg", "nav", "header", "footer", "form", "noscript", "img", "iframe", "button", "input"]):
        tag.decompose()
    for tag in list(soup.find_all(True)):
        if getattr(tag, "decomposed", False):
            continue
        attrs = getattr(tag, "attrs", None) or {}
        classes = " ".join(attrs.get("class", []) or [])
        tag_id = attrs.get("id", "") or ""
        if (classes and _NOISE_CLASS.search(classes)) or tag_id in _NOISE_IDS:
            tag.decompose()


def _main_root(soup):
    for selector in ("#docContent", "main", "article", "div.fancy", "body"):
        node = soup.select_one(selector)
        if node is not None:
            return node
    return soup


def _table_text(table) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def parse_html(raw: str, meta: DocumentMeta) -> ParsedDocument:
    from bs4 import BeautifulSoup, NavigableString, Tag

    soup = BeautifulSoup(raw, "html.parser")
    if not meta.title:
        title_tag = soup.find("title")
        meta.title = title_tag.get_text(strip=True) if title_tag else meta.doc_id
    _strip_noise(soup)
    root = _main_root(soup)

    tracker = _SectionTracker()
    blocks: list[Block] = []
    buffer: list[str] = []

    def flush() -> None:
        text = " ".join(s for s in buffer if s.strip())
        buffer.clear()
        if text.strip():
            blocks.append(Block("paragraph", text, section_path=tracker.path))

    def has_block_children(tag) -> bool:
        return tag.find(["p", "pre", "table", "ul", "ol", "dl", "blockquote", *list(_HEADINGS)]) is not None

    def walk(node) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                if child.__class__.__name__ in {"Comment", "Doctype", "Declaration", "ProcessingInstruction"}:
                    continue
                if str(child).strip():
                    buffer.append(str(child))
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name.lower()
            if name in _HEADINGS:
                flush()
                text = _heading_text(child.get_text(" ", strip=True))
                if text:
                    tracker.push(_HEADINGS[name], text)
                    blocks.append(Block("heading", text, level=_HEADINGS[name], section_path=tracker.path))
            elif name == "pre":
                flush()
                blocks.append(Block("code", child.get_text(), section_path=tracker.path))
            elif name == "table":
                flush()
                if child.find("pre") is not None and len(child.find_all("tr")) <= 2:
                    walk(child)  # layout table that only wraps a code sample
                else:
                    blocks.append(Block("table", _table_text(child), section_path=tracker.path))
            elif name in {"tr", "td", "th", "tbody", "thead"}:
                walk(child)
            elif name in _PARAGRAPHS:
                flush()
                if has_block_children(child):
                    walk(child)
                    flush()
                else:
                    text = child.get_text(" ", strip=True)
                    if name == "li":
                        text = "- " + text
                    blocks.append(Block("paragraph", text, section_path=tracker.path))
            elif name in _CONTAINERS:
                if name in {"div", "section", "article", "main", "blockquote", "ul", "ol", "dl"}:
                    flush()
                walk(child)
                if name in {"div", "section", "article", "main", "blockquote", "ul", "ol", "dl"}:
                    flush()
            elif name == "br":
                buffer.append(" ")
            else:  # inline element (a, b, i, code, tt, em, strong, sup, ...)
                text = child.get_text(" ", strip=False)
                if text.strip():
                    buffer.append(text)
        return None

    walk(root)
    flush()
    return ParsedDocument(meta=meta, blocks=blocks)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_MD_FENCE = re.compile(r"^(```|~~~)")


def parse_markdown(raw: str, meta: DocumentMeta) -> ParsedDocument:
    tracker = _SectionTracker()
    blocks: list[Block] = []
    para: list[str] = []
    lines = raw.replace("\r\n", "\n").split("\n")
    i = 0

    def flush() -> None:
        if para:
            blocks.append(Block("paragraph", " ".join(para), section_path=tracker.path))
            para.clear()

    while i < len(lines):
        line = lines[i]
        fence = _MD_FENCE.match(line.strip())
        if fence:
            flush()
            marker = fence.group(1)
            code: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(marker):
                code.append(lines[i])
                i += 1
            blocks.append(Block("code", "\n".join(code), section_path=tracker.path))
            i += 1
            continue
        heading = _MD_HEADING.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            text = _heading_text(heading.group(2))
            if text:
                if not meta.title and level == 1:
                    meta.title = text
                tracker.push(level, text)
                blocks.append(Block("heading", text, level=level, section_path=tracker.path))
        elif not line.strip():
            flush()
        elif re.match(r"^\s*([-*+]|\d+[.)])\s+", line):
            flush()
            para.append(line.strip())
            flush()
        elif line.startswith("    ") and not para:
            code = [line[4:]]
            while i + 1 < len(lines) and (lines[i + 1].startswith("    ") or not lines[i + 1].strip()):
                i += 1
                code.append(lines[i][4:])
            blocks.append(Block("code", "\n".join(code), section_path=tracker.path))
        else:
            para.append(line.strip())
        i += 1
    flush()
    if not meta.title:
        meta.title = meta.doc_id
    return ParsedDocument(meta=meta, blocks=blocks)


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------
def parse_text(raw: str, meta: DocumentMeta) -> ParsedDocument:
    blocks = []
    for para in re.split(r"\n\s*\n", raw.replace("\r\n", "\n")):
        if not para.strip():
            continue
        indented = all(ln.startswith(("    ", "\t")) for ln in para.split("\n") if ln.strip())
        blocks.append(Block("code" if indented else "paragraph", para))
    if not meta.title:
        meta.title = meta.doc_id
    return ParsedDocument(meta=meta, blocks=blocks)


# ---------------------------------------------------------------------------
# PDF (PyMuPDF)
# ---------------------------------------------------------------------------
_MONO_FONT = re.compile(r"(mono|courier|consol|menlo|code)", re.IGNORECASE)


def parse_pdf(data: bytes, meta: DocumentMeta) -> ParsedDocument:
    import fitz  # PyMuPDF

    blocks: list[Block] = []
    tracker = _SectionTracker()
    with fitz.open(stream=data, filetype="pdf") as pdf:
        if pdf.needs_pass:
            raise UnsupportedDocumentError("encrypted PDFs are not supported")
        if not meta.title:
            meta.title = (pdf.metadata or {}).get("title") or meta.doc_id
        # Collect span sizes to find the body font size.
        sizes: list[float] = []
        pages = []
        for page in pdf:
            d = page.get_text("dict")
            pages.append(d)
            for b in d.get("blocks", []):
                for ln in b.get("lines", []):
                    for sp in ln.get("spans", []):
                        if sp.get("text", "").strip():
                            sizes.append(round(sp.get("size", 0), 1))
        body_size = sorted(sizes)[len(sizes) // 2] if sizes else 10.0

        for page_no, d in enumerate(pages, start=1):
            for b in d.get("blocks", []):
                if b.get("type", 0) != 0:
                    continue  # image block
                lines = []
                spans = [sp for ln in b.get("lines", []) for sp in ln.get("spans", []) if sp.get("text", "").strip()]
                if not spans:
                    continue
                for ln in b.get("lines", []):
                    lines.append("".join(sp.get("text", "") for sp in ln.get("spans", [])))
                text = "\n".join(lines).strip()
                max_size = max(sp.get("size", 0) for sp in spans)
                is_bold = all("bold" in sp.get("font", "").lower() or (sp.get("flags", 0) & 16) for sp in spans)
                mono = sum(bool(_MONO_FONT.search(sp.get("font", ""))) for sp in spans) / len(spans) > 0.6
                short = len(text) <= 120 and "\n" not in text
                if short and (max_size >= body_size * 1.15 or (is_bold and len(text) < 80)):
                    level = 1 if max_size >= body_size * 1.5 else 2 if max_size >= body_size * 1.15 else 3
                    heading = _heading_text(text)
                    if heading:
                        tracker.push(level, heading)
                        blocks.append(Block("heading", heading, level=level, section_path=tracker.path, page=page_no))
                elif mono:
                    blocks.append(Block("code", text, section_path=tracker.path, page=page_no))
                else:
                    blocks.append(Block("paragraph", text.replace("\n", " "), section_path=tracker.path, page=page_no))
    return ParsedDocument(meta=meta, blocks=blocks)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def parse_bytes(data: bytes, filename: str, meta: DocumentMeta) -> ParsedDocument:
    ext = pathlib.Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentError(f"unsupported file type {ext!r}; use one of {sorted(SUPPORTED_EXTENSIONS)}")
    if ext == ".pdf":
        if not data.startswith(b"%PDF"):
            raise UnsupportedDocumentError("file does not look like a PDF")
        return parse_pdf(data, meta)
    text = data.decode("utf-8", errors="replace")
    if ext in {".html", ".htm"}:
        return parse_html(text, meta)
    if ext in {".md", ".markdown"}:
        return parse_markdown(text, meta)
    return parse_text(text, meta)


def parse_file(path: pathlib.Path, meta: DocumentMeta) -> ParsedDocument:
    path = pathlib.Path(path)
    return parse_bytes(path.read_bytes(), path.name, meta)
