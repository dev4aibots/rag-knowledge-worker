# RAG Knowledge Worker

Production-grade Retrieval-Augmented Generation with hybrid retrieval, verified citations, and a CI-gated evaluation suite. Zero ML dependencies — pure Python stdlib + FastAPI.

## What it is

A RAG system that answers questions from a shipped corpus, where every answer cites its source chunks and citation verification flags ungrounded claims before they reach the user. Retrieval combines TF-IDF (term similarity) and BM25 (keyword matching) with Reciprocal Rank Fusion — deterministic, reproducible, no embedding downloads.

## Real numbers

| Metric | Value | How to verify |
|--------|-------|---------------|
| Documents in corpus | 40 | `GET /corpus` |
| Eval questions | 100 | `data/corpus.json` |
| Dev / test split | 80 / 20 | `data/corpus.json` |
| Tests | 3 modules | `python -m pytest tests/` or `python -m unittest discover tests` |
| Eval pass rate | ≥90% | `POST /eval` |
| CI gate threshold | 90% | `app/main.py` `gate_pass` |
| Runtime dependencies | 3 | `requirements.txt` (fastapi, uvicorn, pydantic) |

Every number in this README is reproduced by a command in the repo — see `docs/VERIFICATION.md`.

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

## Architecture

```
query → hybrid retrieval (TF-IDF + BM25 + RRF) → top-k chunks
                                           → extractive answer
                                           → citation verification
                                           → grounded / ungrounded verdict
```

- **Ingestion** (`app/ingestion.py`): token-based chunking with overlap so no fact is split across chunk boundaries.
- **Retrieval** (`app/retrieval.py`): TF-IDF cosine similarity + BM25, combined with Reciprocal Rank Fusion (k=60). Pure stdlib — no sentence-transformers, no torch, no network.
- **Citations** (`app/citations.py`): every answer is checked against its cited chunks. Ungrounded terms are listed so the UI can flag them.
- **API** (`app/main.py`): FastAPI with `/query`, `/eval`, `/corpus`, `/health` endpoints.
- **UI** (`ui/index.html`): dark-mode interface reusing the shared `tokens.css` design system.

See `docs/ARCHITECTURE.md` for the full design.

## Evaluation

The eval suite runs 100 questions (80 dev / 20 test) against the index. A question passes when the answer is grounded (all significant terms cited) and mentions the expected answer keyword. The CI gate exits non-zero when pass rate drops below 90%.

Evaluation is fully deterministic — no LLM judges, no sampling variability. Same query always returns the same answer.

## Verification

See `docs/VERIFICATION.md` for how each README number is reproduced.

## Security

See `docs/SECURITY.md` for prompt-injection defense and data handling.

## License

MIT
