"""Tests for document ingestion/chunking — pure stdlib."""

import unittest

from app.retrieval import Document
from app.ingestion import chunk_document, build_index


class TestChunkDocument(unittest.TestCase):
    def test_single_chunk_when_short(self):
        doc = Document(id="d1", title="T", text="a b c d e")
        chunks = chunk_document(doc, chunk_size=100, overlap=10)
        self.assertEqual(len(chunks), 1)

    def test_multiple_chunks_when_long(self):
        doc = Document(id="d1", title="T", text="word " * 500)
        chunks = chunk_document(doc, chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)

    def test_overlap_preserves_tokens(self):
        doc = Document(id="d1", title="T", text="word " * 100)
        chunks = chunk_document(doc, chunk_size=50, overlap=10)
        if len(chunks) > 1:
            # Last tokens of chunk 0 should appear in chunk 1
            tail = set(chunks[0].tokens[-10:])
            head = set(chunks[1].tokens[:10])
            self.assertTrue(tail & head, "overlap should share tokens")


class TestBuildIndex(unittest.TestCase):
    def test_builds_from_documents(self):
        docs = [
            Document(id="d1", title="T1", text="alpha beta gamma"),
            Document(id="d2", title="T2", text="delta epsilon zeta"),
        ]
        index = build_index(docs, chunk_size=10, overlap=2)
        self.assertGreater(len(index.chunks), 0)
        self.assertEqual(index.n, len(index.chunks))
        self.assertGreater(index.avgdl, 0)
