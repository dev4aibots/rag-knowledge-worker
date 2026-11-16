# Security

## Threat model

The RAG Knowledge Worker answers questions from a shipped corpus. It does not execute user-supplied code, does not call external APIs based on retrieved content, and does not accept file uploads. The primary risks are prompt injection and ungrounded (hallucinated) answers.

## Defenses

### 1. Prompt injection defense

The system uses a fixed, deterministic answer path: retrieve top-k chunks, concatenate their text, verify citations. There is no system prompt that user input can override, and no generative model that can be instructed to change behavior. User input is treated purely as a retrieval query.

### 2. Citation verification (anti-hallucination)

Every answer is checked with `verify_citations()`: all significant terms (length ≥ 3, non-stopword) in the answer must appear in at least one cited chunk. Ungrounded terms are listed and the `grounded` flag is set to `false`. The UI displays this verdict prominently.

### 3. No action execution

The system never sends email, writes files, or calls external APIs based on retrieved content. There is no tool-use surface to exploit.

### 4. Deterministic output

No randomness, no sampling, no temperature. Same input always produces identical output, making the system predictable and auditable.

## Limitations

- The corpus is static (shipped with the repo). No live data ingestion.
- Retrieval is term-based (TF-IDF + BM25), not embedding-based. May miss conceptual matches where vocabulary differs.
- No user authentication — single-tenant demo.

## Data handling

No personal data is collected or stored. Query logs are not persisted. The corpus contains only synthetic/educational documents about AI engineering topics.
