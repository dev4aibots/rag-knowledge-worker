# RAG Knowledge Worker

Production Retrieval-Augmented Generation with hybrid retrieval, verified citations on every answer, and a CI-gated evaluation suite. Zero ML dependencies — pure Python stdlib + FastAPI.

A RAG system that answers questions from a shipped corpus, where every answer cites its source chunks and citation verification flags ungrounded claims before they reach the user. Retrieval combines TF-IDF (term similarity) and BM25 (keyword matching) with Reciprocal Rank Fusion — deterministic, reproducible, no embedding downloads.

<p align="center">
  <img src="https://img.shields.io/badge/tests-18%20passing-2fd08a?style=for-the-badge" alt="18 tests passing">
  <img src="https://img.shields.io/badge/eval-100%20questions-4e8cff?style=for-the-badge" alt="100 eval questions">
  <img src="https://img.shields.io/badge/pass%20rate-99%25-2fd08a?style=for-the-badge" alt="99% pass rate">
  <img src="https://img.shields.io/badge/gate-%E2%89%A590%25-ff6b7a?style=for-the-badge" alt="CI gate ≥90%">
  <img src="https://img.shields.io/badge/deps-3--zero%20ML-2fd08a?style=for-the-badge" alt="Three deps, zero ML">
  <img src="https://img.shields.io/badge/python-3.12-3776ab?style=for-the-badge&logo=python" alt="Python 3.12">
  <img src="https://img.shields.io/badge/fastapi-0.163-009688?style=for-the-badge&logo=fastapi" alt="FastAPI">
</p>

---

## What it does

```
User query
    │
    ▼
┌─────────────────────┐
│ Hybrid retrieval    │  TF-IDF cosine + BM25 (k1=1.2, b=0.75)
│ + Reciprocal Rank   │  + RRF (k=60): score = Σ 1/(60 + rank)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Extractive answer   │  Concatenate top-k chunk texts (deterministic)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Citation verify     │  All significant terms must appear in cited chunks
└─────────────────────┘
    │
    ▼
Grounded / Ungrounded verdict + inline citations
```

## Real numbers

| Metric | Value | Verify |
|--------|-------|--------|
| Documents in corpus | 40 | `GET /corpus` |
| Eval questions | 100 | `data/corpus.json` |
| Dev / test split | 80 / 20 | `eval_split: {dev: 0.8, test: 0.2}` |
| Unit tests | 18 | `python -m unittest discover tests` |
| Eval pass rate | 99% | `POST /eval` → `gate_pass: true` |
| CI gate threshold | ≥90% | `app/main.py` `gate_pass` |
| Runtime dependencies | 3 | `requirements.txt` (fastapi, uvicorn, pydantic) |

---

## How to run

```bash
# Install
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload

# Open the UI
open http://localhost:8000/

# Run tests
python -m unittest discover tests

# Run the evaluation suite
curl -X POST http://localhost:8000/eval
```

---

## Architecture

| Module | Responsibility |
|--------|---------------|
| `app/retrieval.py` | Tokenization, Index (TF-IDF + BM25 + RRF), Document/Chunk |
| `app/ingestion.py` | Document chunking with overlap, index building |
| `app/citations.py` | Claim-term extraction, grounding verification, citation formatting |
| `app/main.py` | FastAPI: `/query`, `/eval`, `/corpus`, `/health` |
| `ui/index.html` | Dark-mode UI reusing the shared design system |

## Retrieval algorithm

1. Tokenize query (lowercase, split on whitespace, strip leading/trailing punctuation; internal hyphens/dots preserved)
2. Rank chunks by TF-IDF cosine similarity
3. Rank chunks by BM25 (k1=1.2, b=0.75)
4. Combine with Reciprocal Rank Fusion: `score = sum(1 / (60 + rank))`
5. Return top-k sorted by RRF score

## Why no embeddings

Sentence-transformers require torch (~100MB download, GPU optional). For a system that must run anywhere, TF-IDF + BM25 hybrid retrieval is a legitimate, explainable alternative that captures term-level similarity without any ML downloads.

---

## Evaluation

100 questions with expected answer keywords, split 80% dev / 20% test. A question passes when the answer is grounded (all significant terms cited) AND the expected keyword appears in the answer or cited text.

CI gate: pass rate ≥ 90% or exit non-zero. Evaluation is fully deterministic — no LLM judges, no sampling variability.

---

## Security

- No action execution: never sends email, writes files, or calls external APIs based on retrieved content
- Prompt injection defense: fixed deterministic path, no system prompt user input can override
- Citation verification: ungrounded claims are flagged before reaching the user
- Static corpus only: no live data ingestion, no personal data collected

---

## License

MIT
