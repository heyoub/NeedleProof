# NeedleProof

**NeedleProof turns every corpus answer into a reproducible evidence investigation.**

Ask a closed corpus a real question. Watch one agent search, inspect, compare, and verify its evidence. The final answer appears only after deterministic application code has checked quotation, metric, value, and temporal anchors; every run leaves an application-sealed, downloadable receipt.

Built for the Finance of the Future event at Tactix in Philadelphia on July 29, 2026.

## What ships

- One `NeedleProof Corpus Investigator` using GPT-5.6 Terra with medium reasoning
- Four sequential tools: `search_corpus`, `read_chunks`, `inspect_document`, and `verify_evidence`
- TurboVec dense retrieval plus SQLite FTS5 lexical search and reciprocal-rank fusion
- Immutable, artifact-bound event corpus at 768 embedding dimensions
- Exact quote/metric/value/time anchors, date variants, conflicts, and bounded missing-evidence verification
- Append-only hash-chained execution ledger with canonical JSON and human-readable HTML receipts
- Semantic SSE activity with replay, `Last-Event-ID`, cancellation, and terminal-state recovery
- LiteShip 0.10.0 / Astro 7 forensic editorial interface—no React and no chatbot theater
- A real sealed rehearsal run for live-demo continuity

The primary corpus is the supplied Hamilton Lane/Northbank Advisors finance challenge. Its answer-key page is deliberately excluded from indexing.

## One-command demo

Prerequisites: Linux x86-64, Python 3.11–3.13, [uv](https://docs.astral.sh/uv/), Node 22.13+, pnpm 10.32.1, and an OpenAI-compatible project key with access to `gpt-5.6-terra` plus `text-embedding-3-small`.

```bash
cp .env.example .env
# Add OPENAI_API_KEY to .env. Never commit it.
uv sync
pnpm install --frozen-lockfile
pnpm demo
```

Open [http://127.0.0.1:4321](http://127.0.0.1:4321). `pnpm demo` verifies the seeded corpus, starts FastAPI on port 8000, starts Astro on port 4321, and proxies `/api` from the web app to the API.

The process starts even when OpenAI is unavailable, so rehearsal, corpus inspection, and existing receipts remain usable. `/api/live` reports local liveness; `/api/ready` reports corpus/model readiness. A live run performs or reuses a cached capability check and fails precisely if Terra is unavailable—NeedleProof never silently changes models.

## Golden demo

Use the synthesis prompt:

> Summarize the company’s scale and fee economics and identify anything questionable.

The investigation should preserve:

- AUM of `$146.1 billion` at December 31, 2025 and `$142 billion` at March 31, 2026 as date variants
- fee-earning AUM of `$82 billion` and the unresolved `$8.2 billion` value as a genuine conflict
- assets under advisement of `$905 billion`
- management and advisory fees of `$584.2 million`
- fee-related earnings of `$345 million`
- blended fee rate of `67 basis points`
- no invented headcount

These values live only in golden expectations and source documents. Application behavior does not hard-code the answers.

After the answer appears:

1. Open an evidence chip to compare the exact extracted passage with the original PDF page.
2. Select **Test an altered quotation** to watch deterministic verification reject a planted modification.
3. Inspect the HTML receipt or download its canonical JSON.
4. Use **Replay rehearsal receipt** in the footer if live API continuity is needed.

## Architecture

```text
LiteShip 0.10.0 / Astro 7
          │ HTTP + replayable SSE
          ▼
       FastAPI
          ├── OpenAI Agents SDK 0.18.3 ── GPT-5.6 Terra
          ├── OpenAI embeddings ───────── text-embedding-3-small / 768d
          ├── corpus tools ─────────────── search / read / inspect / verify
          ├── TurboVec 0.8.0 ───────────── persisted .tvim dense index
          ├── SQLite FTS5 ──────────────── lexical retrieval and metadata
          └── receipt ledger ───────────── SQLite + sealed canonical JSON
```

The longer diagram and trust boundaries are in [docs/architecture.md](docs/architecture.md) and [docs/product-spec.md](docs/product-spec.md).

## Verification model

The model produces typed observations rather than a publishable proposition or proposed verdict. It must copy a canonical metric anchor and the smallest contiguous assertion beginning at that complete metric or a bound leading temporal anchor and ending at the proved value or its bound trailing temporal anchor, plus the exact value, observation kind, evidence relation, and enclosing quotation. The server reruns the deterministic verifier after the agent finishes and constructs displayed claim prose from verified observations; the model's draft prose is never published. The verifier checks source integrity and a closed positive-binding contract—not arbitrary English entailment—and leaves unknown syntax and unauthorized value roles unverified. See [the verification boundary](docs/verification-boundary.md). Valid review findings follow the repository's [quality feedback loop](docs/quality-feedback-loop.md): each becomes a shared invariant and enforced shape family, not a one-line patch.

Verification checks include:

- contiguous quote and assertion presence after Unicode NFKC, whitespace folding, PDF line-break dehyphenation, and recorded case-folded comparison
- chunk membership in the current corpus version
- metric-anchor identity plus an authorized direct/anaphoric binding profile, explicit value role, currency, magnitude, percentages, basis points, units, and span-bound temporal anchors
- supporting evidence for every accepted value; contextual or contradicting evidence cannot authorize a claim by itself
- locally value-bound contradiction language versus canonically equivalent or genuinely different reporting periods
- a server-owned four-probe absence protocol plus a breadth-limited exhaustive complete-metric phrase scan; it opens every unique candidate, analyzes the complete local clause while storing a bounded display excerpt, independently re-derives the proof at authorization time, and refuses `not_found` when retrieval is incomplete, over-broad, or any metric-adjacent value or qualified predicate needs review

A rejected claim is removed from the authoritative answer. A timeout, cancellation, invalid structured response, tool failure, or rate limit seals a non-authoritative partial receipt instead.

## Receipts

Every sealed receipt includes:

- ordered search/read/inspect/verify events and their hash chain
- TurboVec IDs, dense/lexical ranks, scores, pages, previews, timings, and chunk hashes
- exact selected quotations plus quote/value verification outcomes
- all model and embedding calls, response/request IDs when available, token use, observable retry counts, and sanitized errors; hidden client retries are disabled
- corpus manifest digest, agent-instruction/tool-schema/binding/absence hashes, verifier version, Git/image revision, dependency-lock digests, and all material runtime limits
- schema-visible live/migration provenance, hash-addressed trusted source artifacts for migrated
  receipts, and read-only integrity validation for retained schema-1.2 receipts

Receipts are application-sealed and integrity-checked, not cryptographically signed for third-party authentication. The expected digest is stored independently in SQLite, and JSON/HTML is revalidated whenever it is served or replayed. A migrated receipt is accepted only when its declared source resolves through a code-owned migration record, the retained source artifact reproduces that canonical digest, the source run/corpus/status/question/verifier identity matches, and the entire migrated receipt reproduces the record's canonical target digest. The target digest binds the answer, claims, evidence, events, configuration, and provenance rather than trusting a selected field list.

A server-generated `HttpOnly`, `SameSite=Lax` browser cookie owns every run. Read, SSE, cancel, receipt, and quote-verification routes enforce that ownership; run IDs are identifiers rather than bearer credentials. Live runs are bounded per browser and by per-session/IP rates plus hourly/daily token budgets. Each live admission atomically reserves conservative token capacity in SQLite. Before every model call, NeedleProof reserves a byte-based input ceiling plus an API-enforced output-token cap; each persisted call atomically expands the reservation to at least actual cumulative usage. Forwarded client IPs are honored only when the immediate transport peer belongs to an explicitly configured trusted-proxy CIDR; direct clients cannot forge `CF-Connecting-IP` to rotate their rate-limit key. The trusted proxy must strip client-supplied copies of that header. Receipt routes send `noindex, nofollow`. The event build supports only the bundled public/fictional corpus; confidential uploads are explicitly unsupported.

## Commands

```bash
pnpm demo                 # ensure corpus, then run API + web
pnpm test                 # Python tests, Astro type-check, production web build
pnpm golden               # live seeded-corpus passage-presence smoke test
pnpm receipt:rehearsal    # validate receipt digest, hash chain, and golden statuses
pnpm corpus:build         # reproducibly rebuild and atomically publish the corpus
pnpm corpus:ensure        # verify the current manifest or build when absent
```

The five-chunk seeded golden is intentionally described as a passage-presence smoke test, not a ranking benchmark. The offline retrieval gate builds a 100-plus-chunk haystack with overlapping finance terms, planted numeric decoys, instruction-like text, the Hamilton Lane material, and the Fairmount Studio general-business memo. It measures dense, lexical, and hybrid ranking, first-support reciprocal rank, multipassage conflict coverage, document/date filters, and ensures `k` is materially smaller than the corpus.

## Docker

```bash
cp .env.example .env
# Add OPENAI_API_KEY to .env.
docker compose up --build
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The multi-stage image builds the LiteShip static frontend and serves it from FastAPI. Compose initializes a named `needleproof-data` volume from the image's validated corpus, then persists SQLite runs, receipts, corpus manifests, and `.tvim` files across restarts.

## Corpus lifecycle

`data/corpus-source.json` explicitly selects supported text-based PDF/TXT sources and physical pages. The builder:

1. extracts page text and printed labels without OCR;
2. creates page-bounded semantic chunks and stable opaque external IDs (converted to unsigned 64-bit values only inside the TurboVec adapter);
3. embeds and L2-normalizes chunks;
4. writes SQLite/FTS5 and TurboVec into a temporary version directory;
5. validates document/row counts and the full SQLite-to-TurboVec ID set;
6. commits hashes of `corpus.sqlite3`, `index.tvim`, copied documents, and canonical ordered chunk records into the manifest;
7. hashes the canonical manifest and atomically advances `data/corpora/current.json` only after validation.

Adding or removing a document creates a new immutable version. PDFs with no usable text fail clearly; OCR, DOCX, XLSX ingestion, crawlers, auth, pgvector, and confidential uploads remain intentionally out of scope.

## Repository map

```text
apps/web/                 LiteShip/Astro interface
apps/api/                 FastAPI, agent, retrieval, verifier, receipts
packages/contracts/       browser and canonical receipt contracts
data/event-pack/          authoritative supplied event materials
data/corpora/             immutable seeded corpus versions
data/rehearsal/           featured sealed live receipt
docs/                     product spec, architecture, demo script
tests/                    deterministic and adapter suite
```

See [docs/demo-script.md](docs/demo-script.md) for the two-minute presentation flow.

## Critical pins

- `openai-agents==0.18.3`
- `openai==2.45.0`
- `turbovec==0.8.0`
- `@czap/astro`, `@czap/core`, and `@czap/web` `0.10.0`
- `astro==7.1.4`
- Python `>=3.11,<3.14`

Both `uv.lock` and `pnpm-lock.yaml` are committed. LiteShip’s exact Astro, Effect, and `@czap/*` versions come from the 0.10.0 starter and lockfile rather than older indexed documentation.

NeedleProof is available under the [MIT License](LICENSE).

---

**Inspect the evidence, download the receipt, or ask the corpus another question.**
