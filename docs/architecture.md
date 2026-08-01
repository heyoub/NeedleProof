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

The corpus builder extracts only explicitly selected source pages, chunks without crossing page boundaries, embeds and normalizes text, and writes SQLite and TurboVec into a temporary directory. The manifest commits to both artifacts and canonical ordered chunk records; startup reconciles counts, source hashes, and the complete SQLite/TurboVec ID set before serving the version.

Runs belong to a server-generated browser session. One guarded lifecycle handles live and rehearsal execution, cancellation, timeout, failure, shutdown, and restart recovery. Terminal ledger events are withheld from SSE until the terminal envelope, independently stored receipt digest, and receipt path are durable.
