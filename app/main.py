"""FastAPI application: hybrid RAG with verified citations.

Endpoints:
  GET  /health          — service health
  POST /query           — hybrid retrieval + answer with citations
  GET  /corpus          — list indexed documents
  POST /eval            — run the evaluation suite
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .citations import Answer, Citation, verify_citations
from .ingestion import build_index
from .retrieval import Chunk, Document, Index

app = FastAPI(title="RAG Knowledge Worker", version="1.0.0")

# ---------------------------------------------------------------------------
# In-memory index (built at startup from the shipped corpus)
# ---------------------------------------------------------------------------
_INDEX: Index | None = None
_DOCUMENTS: list[Document] = []
_CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "corpus.json"


def _load_corpus() -> list[Document]:
    raw = json.loads(_CORPUS_PATH.read_text(encoding="utf-8"))
    docs = []
    for item in raw["documents"]:
        docs.append(
            Document(
                id=item["id"],
                title=item["title"],
                text=item["text"],
            )
        )
    return docs


@app.on_event("startup")
def _startup() -> None:
    global _INDEX, _DOCUMENTS
    _DOCUMENTS = _load_corpus()
    _INDEX = build_index(_DOCUMENTS, chunk_size=160, overlap=32)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    q: str
    top_k: int = 5


class QueryResponse(BaseModel):
    query: str
    answer: str
    grounded: bool
    ungrounded_terms: list[str]
    citations: list[dict[str, Any]]
    retrieval_scores: list[dict[str, Any]]


class EvalResponse(BaseModel):
    total: int
    grounded: int
    ungrounded: int
    pass_rate: float
    dev_pass_rate: float
    test_pass_rate: float
    gate_pass: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "indexed_chunks": str(len(_INDEX.chunks) if _INDEX else 0)}


@app.get("/corpus")
def corpus() -> list[dict[str, str]]:
    if not _DOCUMENTS:
        raise HTTPException(status_code=503, detail="corpus not loaded")
    return [{"id": d.id, "title": d.title} for d in _DOCUMENTS]


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if not _INDEX:
        raise HTTPException(status_code=503, detail="index not ready")

    results = _INDEX.search(req.q, top_k=req.top_k)

    # Build a deterministic extractive answer from all retrieved chunks.
    top_chunks = [r[0] for r in results]
    answer_text = " ".join(c.text for c in top_chunks)

    citations = [
        Citation(chunk_id=c.id, doc_id=c.doc_id, text=c.text, relevance=r)
        for c, _, _, r in results
    ]
    verified = verify_citations(answer_text, citations)

    return QueryResponse(
        query=req.q,
        answer=verified.text,
        grounded=verified.grounded,
        ungrounded_terms=verified.ungrounded_terms,
        citations=[
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text[:200],
                "relevance": round(c.relevance, 4),
            }
            for c in verified.citations
        ],
        retrieval_scores=[
            {
                "chunk_id": c.id,
                "doc_id": c.doc_id,
                "tfidf": round(t, 4),
                "bm25": round(b, 4),
                "rrf": round(r, 4),
            }
            for c, t, b, r in results
        ],
    )


@app.post("/eval", response_model=EvalResponse)
def run_eval() -> EvalResponse:
    if not _INDEX:
        raise HTTPException(status_code=503, detail="index not ready")

    raw = json.loads(_CORPUS_PATH.read_text(encoding="utf-8"))
    questions = raw["eval_questions"]
    split = raw.get("eval_split", {"dev": 0.8, "test": 0.2})
    n_dev = math.ceil(len(questions) * split["dev"])
    dev_q = questions[:n_dev]
    test_q = questions[n_dev:]

    def grade(question: dict[str, Any]) -> bool:
        req = QueryRequest(q=question["q"], top_k=8)
        resp = query(req)
        # A question passes if the answer is grounded and mentions the
        # expected answer keyword somewhere in the retrieved text.
        expected = question.get("expected", "").lower()
        all_text = (resp.answer + " " + " ".join(c["text"] for c in resp.citations)).lower()
        return resp.grounded and (not expected or expected in all_text)

    dev_pass = sum(1 for q in dev_q if grade(q))
    test_pass = sum(1 for q in test_q if grade(q))
    total_pass = dev_pass + test_pass
    total = len(questions)

    return EvalResponse(
        total=total,
        grounded=total_pass,
        ungrounded=total - total_pass,
        pass_rate=round(total_pass / max(total, 1), 4),
        dev_pass_rate=round(dev_pass / max(len(dev_q), 1), 4),
        test_pass_rate=round(test_pass / max(len(test_q), 1), 4),
        gate_pass=(total_pass / max(total, 1)) >= 0.90,
    )


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    ui_path = Path(__file__).resolve().parent.parent / "ui" / "index.html"
    return ui_path.read_text(encoding="utf-8")
