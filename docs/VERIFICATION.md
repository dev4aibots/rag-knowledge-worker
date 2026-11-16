# Verification

Every number in the README is reproduced by a command in this repo.

## 1. Documents in corpus (40)

```bash
python -c "import json; d=json.load(open('data/corpus.json')); print(len(d['documents']))"
# → 40
```

## 2. Eval questions (100) + dev/test split (80/20)

```bash
python -c "import json; d=json.load(open('data/corpus.json')); q=d['eval_questions']; print(len(q), int(len(q)*0.8), len(q)-int(len(q)*0.8))"
# → 100 80 20
```

## 3. Tests (3 modules)

```bash
python -m unittest discover tests -v
# → 3 test modules, all green
```

## 4. Eval pass rate (≥90%)

```bash
curl -s -X POST http://localhost:8000/eval | python -m json.tool
# → pass_rate >= 0.90, gate_pass: true
```

Requires the server running: `uvicorn app.main:app --reload`.

## 5. Zero ML dependencies

```bash
cat requirements.txt
# → fastapi, uvicorn, pydantic only (no sentence-transformers, no torch)
```

## 6. Deterministic output

```bash
curl -s -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{"q":"What is RAG?"}' | python -c "import sys,json; d=json.load(sys.stdin); print(d['answer'][:80])"
# → identical output on every call
```
