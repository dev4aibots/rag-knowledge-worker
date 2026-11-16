"""Tests for the retrieval module — pure stdlib, no network, no ML deps."""

import unittest

from app.retrieval import Document, Chunk, Index, _tokenize


class TestTokenize(unittest.TestCase):
    def test_lowercases_and_splits(self):
        self.assertEqual(_tokenize("Hello, World!"), ["hello", "world"])

    def test_drops_punctuation(self):
        # Internal punctuation (hyphens, dots) is preserved as part of tokens
        self.assertEqual(_tokenize("a.b-c"), ["a.b-c"])
        # Leading/trailing punctuation is stripped
        self.assertEqual(_tokenize("(hello)"), ["hello"])

    def test_empty(self):
        self.assertEqual(_tokenize(""), [])
        self.assertEqual(_tokenize("!!!"), [])


class TestIndex(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            Chunk(id="c1", doc_id="d1", text="The quick brown fox jumps over the lazy dog"),
            Chunk(id="c2", doc_id="d1", text="A quick brown dog sleeps on the sofa"),
            Chunk(id="c3", doc_id="d2", text="The lazy cat sits on the mat"),
        ]
        self.index = Index(chunks=self.chunks)

    def test_search_returns_results(self):
        results = self.index.search("quick brown fox", top_k=3)
        self.assertGreaterEqual(len(results), 1)
        # Each result: (chunk, tfidf, bm25, rrf)
        self.assertEqual(len(results[0]), 4)

    def test_exact_match_ranks_first(self):
        results = self.index.search("quick brown fox", top_k=3)
        self.assertEqual(results[0][0].id, "c1")

    def test_empty_query(self):
        results = self.index.search("", top_k=3)
        self.assertEqual(results, [])

    def test_tfidf_and_bm25_present(self):
        results = self.index.search("dog", top_k=3)
        for _, tfidf, bm25, rrf in results:
            self.assertIsInstance(tfidf, float)
            self.assertIsInstance(bm25, float)
            self.assertIsInstance(rrf, float)

    def test_rrf_is_deterministic(self):
        r1 = self.index.search("quick dog", top_k=3)
        r2 = self.index.search("quick dog", top_k=3)
        self.assertEqual([x[0].id for x in r1], [x[0].id for x in r2])


class TestChunking(unittest.TestCase):
    def test_chunk_document(self):
        from app.ingestion import chunk_document
        doc = Document(id="d1", title="T", text="a " * 100)
        chunks = chunk_document(doc, chunk_size=30, overlap=5)
        self.assertGreater(len(chunks), 1)
        # Overlap means consecutive chunks share tokens
        self.assertLessEqual(len(chunks[0].tokens), 30)


if __name__ == "__main__":
    unittest.main()
