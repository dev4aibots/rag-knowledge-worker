"""Hybrid retrieval: TF-IDF + BM25 combined with Reciprocal Rank Fusion.

All ranking is deterministic and dependency-free (pure Python stdlib).
No sentence-transformers, no torch, no network calls — the corpus and index
are shipped with the repo so the demo cannot drift from the tested code.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on whitespace, strip leading/trailing punctuation.

    Internal punctuation (hyphens, dots) is preserved so "1.2", "row-level",
    "tf-idf" remain single tokens and match expected keywords.
    """
    import string
    strip_chars = string.punctuation
    tokens = []
    for raw in text.lower().split():
        t = raw.strip(strip_chars)
        if t:
            tokens.append(t)
    return tokens


@dataclass
class Document:
    id: str
    title: str
    text: str
    tokens: list[str] = field(init=False)
    tf: Counter = field(init=False, compare=False)
    length: int = field(init=False, compare=False)

    def __post_init__(self) -> None:
        self.tokens = _tokenize(self.text)
        self.tf = Counter(self.tokens)
        self.length = len(self.tokens)


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    tokens: list[str] = field(init=False)
    tf: Counter = field(init=False, compare=False)
    length: int = field(init=False, compare=False)

    def __post_init__(self) -> None:
        self.tokens = _tokenize(self.text)
        self.tf = Counter(self.tokens)
        self.length = len(self.tokens)


class Index:
    """In-memory inverted index supporting TF-IDF and BM25 scoring."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.n = len(chunks)
        self.avgdl = sum(c.length for c in chunks) / max(self.n, 1)

        # df[token] = number of chunks containing token
        self.df: Counter = Counter()
        for c in chunks:
            for token in set(c.tokens):
                self.df[token] += 1

        # postings[token] = list of chunk ids containing token
        self.postings: dict[str, list[str]] = defaultdict(list)
        for c in chunks:
            for token in set(c.tokens):
                self.postings[token].append(c.id)

    def _idf(self, token: str) -> float:
        return math.log((self.n - self.df[token] + 0.5) / (self.df[token] + 0.5) + 1.0)

    def tfidf_score(self, query_tokens: list[str], chunk: Chunk) -> float:
        """Cosine similarity between query and chunk TF-IDF vectors."""
        if not query_tokens or chunk.length == 0:
            return 0.0

        # Query TF-IDF vector
        q_tf = Counter(query_tokens)
        q_vec: dict[str, float] = {}
        for token, qf in q_tf.items():
            q_vec[token] = (1.0 + math.log(qf)) * self._idf(token)

        # Chunk TF-IDF vector
        c_vec: dict[str, float] = {}
        for token, cf in chunk.tf.items():
            if token in q_vec:  # only tokens in query matter for dot product
                c_vec[token] = (1.0 + math.log(cf)) * self._idf(token)

        dot = sum(q_vec[t] * c_vec.get(t, 0.0) for t in q_vec)
        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        c_norm = math.sqrt(sum(v * v for v in c_vec.values()))
        if q_norm == 0.0 or c_norm == 0.0:
            return 0.0
        return dot / (q_norm * c_norm)

    def bm25_score(self, query_tokens: list[str], chunk: Chunk) -> float:
        """BM25 score for a chunk against a query."""
        if not query_tokens or chunk.length == 0:
            return 0.0

        score = 0.0
        for token in set(query_tokens):
            if token not in self.df:
                continue
            idf = self._idf(token)
            cf = chunk.tf.get(token, 0)
            if cf == 0:
                continue
            numerator = cf * (self.k1 + 1.0)
            denominator = cf + self.k1 * (1.0 - self.b + self.b * chunk.length / self.avgdl)
            score += idf * (numerator / denominator)
        return score

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float, float, float]]:
        """Hybrid search: TF-IDF + BM25 combined with Reciprocal Rank Fusion.

        Returns list of (chunk, tfidf_score, bm25_score, rrf_score) sorted by RRF.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # Rank by TF-IDF
        tfidf_ranked = sorted(
            self.chunks,
            key=lambda c: self.tfidf_score(query_tokens, c),
            reverse=True,
        )
        tfidf_rank = {c.id: i + 1 for i, c in enumerate(tfidf_ranked)}

        # Rank by BM25
        bm25_ranked = sorted(
            self.chunks,
            key=lambda c: self.bm25_score(query_tokens, c),
            reverse=True,
        )
        bm25_rank = {c.id: i + 1 for i, c in enumerate(bm25_ranked)}

        # Reciprocal Rank Fusion: score = sum(1 / (k + rank))
        k = 60  # standard RRF constant
        chunk_by_id = {c.id: c for c in self.chunks}

        results: dict[str, tuple[Chunk, float, float, float]] = {}
        for c in self.chunks:
            t = self.tfidf_score(query_tokens, c)
            b = self.bm25_score(query_tokens, c)
            rrf = 1.0 / (k + tfidf_rank[c.id]) + 1.0 / (k + bm25_rank[c.id])
            results[c.id] = (c, t, b, rrf)

        return sorted(results.values(), key=lambda x: x[3], reverse=True)[:top_k]
