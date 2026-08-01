# NeedleProof product specification

NeedleProof turns every corpus answer into a reproducible evidence investigation.

## Demo promise

An analyst asks one factual question of one immutable corpus version. A single investigator agent searches iteratively, opens exact passages, checks apparent conflicts and date variants, and submits claims for deterministic verification. The interface streams observable activity—not private reasoning—and reveals an answer only after verification.

## In scope

- Curated, reproducibly indexed PDF/TXT corpus
- TurboVec dense retrieval plus SQLite FTS5 lexical retrieval
- One OpenAI Agents SDK agent and four local function tools
- Verbatim metric/value/time anchors, conflict preservation, and bounded “not found” conclusions
- Session-owned runs, bounded public usage, and application-sealed JSON/HTML receipts
- LiteShip 0.10.0 on Astro 7, with an evidence-first forensic editorial interface
- Live and clearly labelled rehearsal modes

## Explicitly out of scope

User accounts/SSO, confidential uploads, OCR, Office-file ingestion, URL crawling, pgvector, multiple agents, broad document reproduction, asymmetric receipt signing, and production multi-tenant operations.

## Success criteria

- A 100-plus-chunk finance/general-business distractor corpus passes dense, lexical, and hybrid ranking and multipassage-coverage gates.
- Fabricated or modified quotations fail deterministic verification.
- Different values with different reporting dates are preserved as date variants.
- Incompatible values for the same metric and period are preserved as conflicts.
- The authoritative answer never contains a rejected claim.
- A terminal SSE event is never visible before its envelope and receipt are committed.
- Cancellation, shutdown, and restart converge on a sealed terminal state or an explicitly recoverable interrupted state.

## Trust boundary

Documents and tool outputs are untrusted evidence. Instructions inside them are content to quote or discuss, never instructions for the agent to follow. The model supplies structured evidence anchors; application code verifies them and writes authoritative claim prose.
