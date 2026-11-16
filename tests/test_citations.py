"""Tests for citation verification — pure stdlib."""

import unittest

from app.citations import Citation, Answer, verify_citations, extract_claim_terms


class TestExtractTerms(unittest.TestCase):
    def test_stops_common_words(self):
        terms = extract_claim_terms("The quick brown fox jumps over the lazy dog")
        self.assertIn("quick", terms)
        self.assertIn("brown", terms)
        self.assertNotIn("the", terms)
        self.assertNotIn("over", terms)

    def test_drops_short_tokens(self):
        terms = extract_claim_terms("AI RAG system")
        self.assertIn("rag", terms)
        self.assertNotIn("ai", terms)  # length < 3


class TestVerifyCitations(unittest.TestCase):
    def test_grounded_when_terms_cited(self):
        ans = verify_citations(
            "The quick brown fox jumps",
            [Citation(chunk_id="c1", doc_id="d1", text="The quick brown fox jumps over the lazy dog", relevance=0.9)],
        )
        self.assertTrue(ans.grounded)
        self.assertEqual(ans.ungrounded_terms, [])

    def test_ungrounded_when_term_missing(self):
        ans = verify_citations(
            "The quick blue fox jumps",
            [Citation(chunk_id="c1", doc_id="d1", text="The quick brown fox jumps over the lazy dog", relevance=0.9)],
        )
        self.assertFalse(ans.grounded)
        self.assertIn("blue", ans.ungrounded_terms)

    def test_empty_answer_is_grounded(self):
        ans = verify_citations("The a an", [Citation(chunk_id="c1", doc_id="d1", text="hello world", relevance=0.5)])
        self.assertTrue(ans.grounded)
