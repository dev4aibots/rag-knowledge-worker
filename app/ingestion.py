"""Document ingestion: chunking and index building."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .retrieval import Chunk, Document, Index


def chunk_document(doc: Document, chunk_size: int = 200, overlap: int = 40) -> list[Chunk]:
    """Split a document into overlapping token-based chunks.

    Each chunk carries the source document id so answers can cite where
    information came from. Overlap ensures no fact is split across a
    chunk boundary and lost to retrieval.
    """
    tokens = doc.tokens
    if not tokens:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        text = " ".join(chunk_tokens)
        chunks.append(
            Chunk(
                id=f"{doc.id}-{idx}",
                doc_id=doc.id,
                text=text,
            )
        )
        idx += 1
        if end >= len(tokens):
            break
        start = end - overlap
        if start < 0:
            start = 0

    return chunks


def build_index(documents: Iterable[Document], chunk_size: int = 200, overlap: int = 40) -> Index:
    """Chunk all documents and build a retrieval index."""
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
    return Index(chunks=all_chunks)
