# Architecture

## Design goals

- **Deterministic**: same query → same answer, every time. No LLM judges, no sampling.
- **Dependency-free**: pure Python stdlib + FastAPI. Runs anywhere, including serverless without GPU.
- **Verifiable**: every answer cites source chunks; citation verification flags ungrounded claims.
- **Testable**: 3 test modules + eval suite with CI gate.

## Data flow

```
User query
    │
    ▼
┌─────────────────┐
│ Hybrid retrieval │  TF-IDF cosine + BM25 + Reciprocal Rank Fusion (k=60)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Extractive answer│  Concatenate top-3 chunk texts (deterministic)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Citation verify │  All significant terms must appear in cited chunks
└─────────────────┘
    │
    ▼
Grounded / Ungrounded verdict + citations
```

## Modules

| Module | Responsibility |
|--------|---------------|
| `app/retrieval.py` | Tokenization, Index (TF-IDF + BM25 + RRF), Document/Chunk data classes |
| `app/ingestion.py` | Document chunking with overlap, index building |
| `app/citations.py` | Claim-term extraction, grounding verification, citation formatting |
| `app/main.py` | FastAPI app: `/query`, `/eval`, `/corpus`, `/health` |
| `ui/index.html` | Dark-mode UI reusing `tokens.css` design system |

## Retrieval algorithm

1. Tokenize query (lowercase, split on `[a-z0-9]+`, drop punctuation).
2. Rank chunks by TF-IDF cosine similarity.
3. Rank chunks by BM25 (k1=1.2, b=0.75).
4. Combine with Reciprocal Rank Fusion: `score = sum(1 / (60 + rank))`.
5. Return top-k sorted by RRF score.

## Evaluation

- 100 questions with expected answer keywords, split 80% dev / 20% test.
- A question passes when: answer is grounded AND expected keyword appears in answer or cited text.
- CI gate: pass rate ≥ 90% or exit non-zero.

## Why no embeddings

Sentence-transformers require torch (~100MB download, GPU optional). For a portfolio demo that must run in any environment, TF-IDF + BM25 hybrid retrieval is a legitimate, explainable alternative that captures term-level similarity without any ML downloads.
