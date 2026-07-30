# Architecture

```mermaid
flowchart LR
    UI[LiteShip 0.10 / Astro 7] -->|HTTP + replayable SSE| API[FastAPI]
    API --> RUN[One Agents SDK investigator]
    RUN --> TOOLS[Search · Read · Inspect · Verify]
    TOOLS --> FTS[SQLite + FTS5]
    TOOLS --> TV[TurboVec IdMapIndex]
    TOOLS --> EMB[OpenAI embeddings]
    RUN --> MODEL[GPT-5.6 Terra]
    API --> LEDGER[Append-only receipt ledger]
    LEDGER --> JSON[Canonical JSON + SHA-256]
    LEDGER --> HTML[Human receipt]
```

The corpus builder extracts only explicitly selected source pages, chunks without crossing page boundaries, embeds and normalizes text, writes SQLite and TurboVec into a temporary directory, validates them, and atomically publishes an immutable manifest version.
