"""Citation verification: every answer must cite the chunks it used.

A claim is "grounded" only if its key terms appear in a retrieved chunk
that is explicitly cited. Ungrounded claims are flagged so the UI can
mark them or the caller can refuse to answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .retrieval import Chunk


@dataclass
class Citation:
    chunk_id: str
    doc_id: str
    text: str
    relevance: float


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = True
    ungrounded_terms: list[str] = field(default_factory=list)


def extract_claim_terms(text: str) -> set[str]:
    """Extract meaningful terms from an answer for grounding checks.

    Removes common stop words and keeps tokens that look like entities,
    technical terms, or numbers.
    """
    stop = {
        "the", "a", "an", "and", "or", "but", "if", "then", "else", "when",
        "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "can", "must", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "it", "its", "this", "that",
        "over", "under", "into", "onto", "upon",
        "these", "those", "i", "you", "he", "she", "we", "they", "them",
        "their", "my", "your", "his", "her", "our", "not", "no", "yes",
        "so", "such", "just", "only", "also", "very", "really", "quite",
    }
    import re
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in stop and len(t) >= 3}


def verify_citations(answer_text: str, citations: Iterable[Citation]) -> Answer:
    """Check that every significant term in the answer appears in a cited chunk.

    Returns an Answer with grounded=True only if all claim terms are covered
    by at least one cited chunk. Ungrounded terms are listed for the caller.
    """
    claim_terms = extract_claim_terms(answer_text)
    cited_text = " ".join(c.text for c in citations).lower()
    cited_tokens = set(__import__("re").findall(r"[a-z0-9]+", cited_text))

    ungrounded = sorted(t for t in claim_terms if t not in cited_tokens)

    citation_list = list(citations)
    return Answer(
        text=answer_text,
        citations=citation_list,
        grounded=len(ungrounded) == 0,
        ungrounded_terms=ungrounded,
    )


def format_citations(citations: Iterable[Citation]) -> str:
    """Render citations as a compact inline reference list."""
    items = list(citations)
    if not items:
        return ""
    refs = []
    for i, c in enumerate(items, 1):
        refs.append(f"[{i}] {c.doc_id}")
    return " Sources: " + "; ".join(refs)
