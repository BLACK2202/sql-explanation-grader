"""Okapi BM25 keyword index.

BM25 scores a passage by how many query terms it contains, weighting rare
terms more (inverse document frequency) and damping very long passages.
It complements dense retrieval: dense vectors capture meaning, BM25 catches
exact terms such as function names (``json_extract``) or keywords (``HAVING``).

The tokenizer keeps SQL identifiers intact and adds a few multi-word SQL
phrases as single tokens ("left_join", "group_by", "is_null", ...), which
helps exact matching of SQL constructs.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_WORD = re.compile(r"[a-z_][a-z0-9_]*|\d+")
_PHRASES = [
    ("left", "outer", "join"), ("left", "join"), ("right", "join"), ("full", "outer", "join"), ("inner", "join"),
    ("cross", "join"), ("natural", "join"), ("group", "by"), ("order", "by"), ("partition", "by"),
    ("is", "null"), ("is", "not", "null"), ("not", "in"), ("not", "exists"), ("union", "all"),
    ("row", "number"), ("dense", "rank"), ("common", "table", "expression"), ("window", "function"),
]
STOPWORDS = frozenset(
    "a an and are as at be been but by can do does for from has have if in into is it its may more most "
    "no not of on or other so such than that the their them then there these they this those to was were "
    "what when where which while who will with would you your also each only same both any all".split()
)
# Words that are stop words in prose but meaningful SQL keywords.
# ("is", "by", "on" only matter inside phrases such as "is_null" or "group_by".)
SQL_KEEP = frozenset({"in", "not", "all", "any", "between", "exists", "null", "where"})


def tokenize(text: str) -> list[str]:
    words = _WORD.findall(text.lower())
    tokens = [w for w in words if w not in STOPWORDS or w in SQL_KEEP]
    phrases = []
    for n in (3, 2):
        for i in range(len(words) - n + 1):
            tup = tuple(words[i:i + n])
            if tup in _PHRASES:
                phrases.append("_".join(tup))
    return tokens + phrases


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.ids: list[str] = []
        self.doc_freqs: list[Counter] = []
        self.doc_len: list[int] = []
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0

    def build(self, ids: list[str], texts: list[str]) -> "BM25Index":
        self.ids = list(ids)
        self.doc_freqs = [Counter(tokenize(t)) for t in texts]
        self.doc_len = [sum(c.values()) for c in self.doc_freqs]
        n = len(texts)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0
        df: Counter = Counter()
        for c in self.doc_freqs:
            df.update(c.keys())
        # BM25+ style non-negative IDF.
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        return self

    def __len__(self) -> int:
        return len(self.ids)

    def scores(self, query: str) -> list[float]:
        q = Counter(tokenize(query))
        out = []
        for freqs, dl in zip(self.doc_freqs, self.doc_len):
            s = 0.0
            for term in q:
                tf = freqs.get(term)
                if not tf:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += idf * tf * (self.k1 + 1) / denom
            out.append(s)
        return out

    def search(self, query: str, k: int = 10, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        scored = [
            (cid, s) for cid, s in zip(self.ids, self.scores(query))
            if s > 0 and (allowed is None or cid in allowed)
        ]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]
