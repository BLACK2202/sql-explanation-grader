"""Citation validation and claim-support checks for RAG explanations.

Checks performed on every RAG answer (all deterministic):
  * citation validity: every [S#] marker and every listed citation must name a
    source that was actually given to the model;
  * claim support: for each sentence carrying a marker, the share of its
    content words that occur in the cited passage(s) (lexical support), and
    optionally the embedding similarity between sentence and passage;
  * coverage of the text by citations (how many sentences cite something).

Lexical support is a heuristic: it shows whether a cited passage plausibly
talks about the same thing, not that the sentence is true.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field

from .retrieval import RetrievedChunk
from .schemas import Citation

_MARKER = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S?\d+)*)\s*\]")
# A sentence ends at . ! or ?, optionally followed by source markers ("... rows. [S1]"),
# when the next sentence starts or the text ends.
_SENT_END = re.compile(r"[.!?](?:\s*\[\s*S\d+(?:\s*[,;]\s*S?\d+)*\s*\])*(?=\s+[A-Z\"'(\[]|\s*$)")


def split_sentences(text: str) -> list[str]:
    out, start = [], 0
    for m in _SENT_END.finditer(text):
        out.append(text[start:m.end()].strip())
        start = m.end()
    if text[start:].strip():
        out.append(text[start:].strip())
    return [s for s in out if s]
_WORD = re.compile(r"[a-z][a-z0-9_]+")
_STOP = frozenset(
    "the a an and or of to in on for with by from that this these those is are was were be been it its as at "
    "which who whom whose what when where how than then them they their there each any all only also into "
    "such not no but if so query returns return returned shows show lists list gives give rows row value values "
    "table tables column columns result results data using uses use used will would can".split()
)
SUPPORT_THRESHOLD = 0.25


@dataclass
class CitationReport:
    markers: list[str] = field(default_factory=list)
    invalid_markers: list[str] = field(default_factory=list)
    listed: list[str] = field(default_factory=list)
    invalid_listed: list[str] = field(default_factory=list)
    n_sentences: int = 0
    n_cited_sentences: int = 0
    sentence_support: list[dict] = field(default_factory=list)
    citation_validity: float | None = None   # valid markers / all markers
    support_rate: float | None = None        # supported cited sentences / cited sentences
    sources_used: list[str] = field(default_factory=list)
    sources_unused: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _expand(marker_group: str) -> list[str]:
    out = []
    for part in re.split(r"[,;]", marker_group):
        part = part.strip()
        if not part:
            continue
        out.append(part if part.upper().startswith("S") else f"S{part}")
    return [p.upper() for p in out]


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def lexical_support(sentence: str, passage: str) -> float:
    words = content_words(_MARKER.sub("", sentence))
    if not words:
        return 0.0
    passage_words = content_words(passage)
    # Count a word as supported if the passage has it or a word sharing its first 5 letters (plural/tense).
    stems = {w[:5] for w in passage_words}
    hit = sum(1 for w in words if w in passage_words or w[:5] in stems)
    return hit / len(words)


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def validate_citations(explanation: str, citations: list[Citation], sources: list[RetrievedChunk],
                       embedder=None) -> CitationReport:
    by_label = {c.source_label.upper(): c for c in sources}
    rep = CitationReport()
    for group in _MARKER.findall(explanation):
        rep.markers.extend(_expand(group))
    rep.invalid_markers = [m for m in rep.markers if m not in by_label]
    rep.listed = [c.source_id.strip().upper() for c in citations]
    rep.invalid_listed = [m for m in rep.listed if m not in by_label]
    if rep.markers:
        rep.citation_validity = round(1 - len(rep.invalid_markers) / len(rep.markers), 4)

    sentences = split_sentences(explanation.strip())
    rep.n_sentences = len(sentences)
    supported = 0
    for sent in sentences:
        cited = []
        for group in _MARKER.findall(sent):
            cited.extend(_expand(group))
        if not cited:
            continue
        rep.n_cited_sentences += 1
        valid = [m for m in cited if m in by_label]
        lex = max((lexical_support(sent, by_label[m].text) for m in valid), default=0.0)
        sem = None
        if embedder is not None and valid:
            sv = embedder.embed_query(_MARKER.sub("", sent))
            pv = embedder.embed_documents([by_label[m].text for m in valid])
            sem = round(max(_cos(sv, p) for p in pv), 4)
        ok = bool(valid) and lex >= SUPPORT_THRESHOLD
        supported += int(ok)
        rep.sentence_support.append({
            "sentence": _MARKER.sub("", sent).strip(), "cited": cited, "valid": valid,
            "lexical_support": round(lex, 3), "semantic_similarity": sem, "supported": ok,
        })
    if rep.n_cited_sentences:
        rep.support_rate = round(supported / rep.n_cited_sentences, 4)
    used = {m for m in rep.markers + rep.listed if m in by_label}
    rep.sources_used = sorted(used, key=lambda s: int(s[1:]))
    rep.sources_unused = [label for label in sorted(by_label, key=lambda s: int(s[1:])) if label not in used]
    return rep
