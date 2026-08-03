from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .chunk_ids import ChunkId
from .config import Settings
from .corpus import load_current_manifest
from .db import AppDatabase
from .models import CorpusSummary, RunCreateRequest, RunCreateResponse, RunStatus
from .readiness import ModelAvailability
from .receipt import receipt_html, validate_receipt
from .retrieval import CorpusStore
from .security import (
    PublicUsageLimiter,
    UsageLimitError,
    new_browser_session,
    resolve_client_ip,
    session_digest,
    valid_browser_session,
)
from .service import (
    TERMINAL_EVENT_TYPES,
    InvestigationService,
    RunCapacityError,
    SessionLiveRunError,
)
from .util import normalize_evidence_text

TERMINAL_STATUSES = {
    RunStatus.COMPLETED,
    RunStatus.INCOMPLETE,
    RunStatus.CANCELLED,
    RunStatus.FAILED,
    RunStatus.INTERRUPTED,
}


class QuoteChallengeRequest(BaseModel):
    run_id: str = Field(pattern=r"^run_[0-9a-f]{32}$")
    corpus_version: str = Field(pattern=r"^v_[0-9a-f]{16}$")
    chunk_id: ChunkId
    quote: str = Field(min_length=1, max_length=4000)

    @field_validator("quote")
    @classmethod
    def require_normalized_quote_text(cls, value: str) -> str:
        normalized, _ = normalize_evidence_text(value)
        if not normalized:
            raise ValueError("quote must contain non-whitespace evidence text")
        return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    database = AppDatabase(settings.app_db_path)
    await database.initialize()
    retention_cutoff = (
        datetime.now(UTC) - timedelta(days=settings.receipt_retention_days)
    ).isoformat()
    expired_receipts = await database.delete_runs_created_before(retention_cutoff)
    receipts_root = settings.receipts_dir.resolve()
    for expired in expired_receipts:
        path = Path(expired).resolve()
        if path.parent == receipts_root:
            path.unlink(missing_ok=True)
    corpus = CorpusStore(settings)
    app.state.settings = settings
    app.state.database = database
    app.state.corpus = corpus
    app.state.usage_limiter = PublicUsageLimiter(settings, database)
    app.state.service = InvestigationService(settings, database, corpus, app.state.usage_limiter)
    await app.state.service.reconcile_abandoned_runs()
    app.state.model_availability = ModelAvailability(settings)
    model_probe = asyncio.create_task(app.state.model_availability.refresh())
    yield
    await app.state.service.shutdown()
    if not model_probe.done():
        model_probe.cancel()
    with suppress(asyncio.CancelledError):
        await model_probe


_public_demo = os.getenv("NEEDLEPROOF_PUBLIC_DEMO", "false").casefold() in {
    "1",
    "true",
    "yes",
}

app = FastAPI(
    title="NeedleProof API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None if _public_demo else "/api/docs",
    openapi_url=None if _public_demo else "/api/openapi.json",
)


@app.middleware("http")
async def browser_session(request: Request, call_next):
    settings: Settings = request.app.state.settings
    token = request.cookies.get(settings.session_cookie_name)
    created = not valid_browser_session(token)
    if created:
        token = new_browser_session()
    request.state.session_id = session_digest(token)
    request.state.client_ip = resolve_client_ip(
        request.client.host if request.client else None,
        request.headers.get("cf-connecting-ip"),
        settings.trusted_proxy_cidrs,
    )
    response = await call_next(request)
    if created:
        response.set_cookie(
            settings.session_cookie_name,
            token,
            httponly=True,
            secure=settings.session_cookie_secure,
            samesite="lax",
            max_age=60 * 60 * 24 * settings.receipt_retention_days,
            path="/",
        )
    return response


def _services(request: Request) -> tuple[AppDatabase, CorpusStore, InvestigationService]:
    return request.app.state.database, request.app.state.corpus, request.app.state.service


async def _owned_run_row(request: Request, run_id: str) -> dict[str, object]:
    database: AppDatabase = request.app.state.database
    row = await database.get_run_row(run_id)
    if not row or row.get("session_id") != request.state.session_id:
        raise HTTPException(status_code=404, detail="Investigation run not found")
    return row


@app.get("/api/live")
async def live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/api/ready")
async def ready(request: Request) -> JSONResponse:
    _, corpus, _ = _services(request)
    availability: ModelAvailability = request.app.state.model_availability
    status_code = 200 if availability.ready else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if availability.ready else "degraded",
            "model": request.app.state.settings.model,
            "model_ready": availability.ready,
            "model_error": availability.error,
            "corpus_version": corpus.corpus_version,
            "manifest_sha256": corpus.manifest_sha256,
        },
    )


@app.get("/api/health")
async def health(request: Request) -> JSONResponse:
    return await ready(request)


@app.get("/api/corpus", response_model=CorpusSummary)
async def corpus_summary(request: Request) -> CorpusSummary:
    settings: Settings = request.app.state.settings
    manifest = load_current_manifest(settings)
    return CorpusSummary(
        corpus_id=manifest["corpus_id"],
        display_name=manifest["display_name"],
        corpus_version=manifest["corpus_version"],
        manifest_sha256=manifest["manifest_sha256"],
        document_count=manifest["document_count"],
        chunk_count=manifest["chunk_count"],
        embedding_model=manifest["embedding_model"],
        embedding_dimensions=manifest["embedding_dimensions"],
        turbovec_version=manifest["turbovec_version"],
    )


@app.post("/api/runs", response_model=RunCreateResponse, status_code=202)
async def create_run(request: Request, body: RunCreateRequest) -> RunCreateResponse:
    _, _, service = _services(request)
    try:
        if not body.rehearsal:
            availability: ModelAvailability = request.app.state.model_availability
            await availability.require()
        return await service.create_run(
            body,
            session_id=request.state.session_id,
            client_ip=request.state.client_ip,
        )
    except RunCapacityError as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": "2"},
        ) from exc
    except SessionLiveRunError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except UsageLimitError as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}")
async def get_run(request: Request, run_id: str):
    database, _, _ = _services(request)
    await _owned_run_row(request, run_id)
    envelope = await database.get_envelope(run_id)
    if not envelope:
        raise HTTPException(status_code=404, detail="Investigation run not found")
    return envelope


@app.post("/api/runs/{run_id}/cancel", status_code=202)
async def cancel_run(request: Request, run_id: str) -> dict[str, str]:
    _, _, service = _services(request)
    await _owned_run_row(request, run_id)
    if not await service.cancel(run_id):
        raise HTTPException(status_code=409, detail="Investigation is already terminal")
    return {"run_id": run_id, "status": "cancelling"}


@app.get("/api/runs/{run_id}/events")
async def run_events(
    request: Request,
    run_id: str,
    last_event_id_header: str | None = Header(default=None, alias="Last-Event-ID"),
    last_event_id_query: int | None = Query(default=None, alias="lastEventId"),
) -> StreamingResponse:
    database, _, _ = _services(request)
    await _owned_run_row(request, run_id)
    try:
        after = last_event_id_query or int(last_event_id_header or 0)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid Last-Event-ID") from exc

    async def stream() -> AsyncIterator[str]:
        cursor = after
        quiet_polls = 0
        while True:
            if await request.is_disconnected():
                break
            events = await database.list_events(run_id, after=cursor)
            terminal_sent = False
            if events:
                quiet_polls = 0
                for event in events:
                    if event["type"] in TERMINAL_EVENT_TYPES:
                        row = await database.get_run_row(run_id)
                        committed = bool(
                            row
                            and RunStatus(str(row["status"])) in TERMINAL_STATUSES
                            and row.get("receipt_path")
                        )
                        if not committed:
                            # Keep the cursor before this event so reconnect/replay
                            # cannot observe completion ahead of durable state.
                            break
                    cursor = int(event["sequence"])
                    yield f"id: {cursor}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                    terminal_sent = event["type"] in TERMINAL_EVENT_TYPES
                if terminal_sent:
                    break
            else:
                quiet_polls += 1
                if quiet_polls >= 40:
                    quiet_polls = 0
                    yield ": heartbeat\n\n"
            row = await database.get_run_row(run_id)
            if (
                row
                and RunStatus(str(row["status"])) in TERMINAL_STATUSES
                and row.get("receipt_path")
                and not events
            ):
                break
            await asyncio.sleep(0.1)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "X-Robots-Tag": "noindex, nofollow",
        },
    )


def _validated_receipt(row: dict[str, object]) -> tuple[Path, dict[str, object]]:
    value = row.get("receipt_path")
    if not value:
        raise HTTPException(status_code=409, detail="Receipt has not been sealed")
    path = Path(str(value)).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail="Receipt file is unavailable")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_receipt(receipt)
    if errors:
        raise HTTPException(status_code=409, detail="Receipt integrity validation failed")
    if receipt.get("receipt_sha256") != row.get("receipt_sha256"):
        raise HTTPException(status_code=409, detail="Receipt database digest does not match")
    return path, receipt


@app.get("/api/runs/{run_id}/receipt.json")
async def receipt_json(request: Request, run_id: str) -> FileResponse:
    row = await _owned_run_row(request, run_id)
    path, _ = _validated_receipt(row)
    return FileResponse(
        path,
        media_type="application/json",
        filename=f"needleproof-{run_id}.json",
        headers={"X-Robots-Tag": "noindex, nofollow"},
    )


@app.get("/api/runs/{run_id}/receipt", response_class=HTMLResponse)
async def receipt_page(request: Request, run_id: str) -> HTMLResponse:
    row = await _owned_run_row(request, run_id)
    _, receipt = _validated_receipt(row)
    return HTMLResponse(receipt_html(receipt), headers={"X-Robots-Tag": "noindex, nofollow"})


@app.get("/api/corpora/{corpus_version}/documents/{document_id}/pdf")
async def versioned_document_pdf(
    request: Request, corpus_version: str, document_id: str
) -> FileResponse:
    settings: Settings = request.app.state.settings
    try:
        corpus = CorpusStore(settings, corpus_version)
        path = corpus.source_pdf(document_id)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
    return FileResponse(
        path,
        media_type="application/pdf",
        headers={"X-Robots-Tag": "noindex, nofollow"},
    )


@app.get("/api/documents/{document_id}/pdf")
async def document_pdf(request: Request, document_id: str) -> FileResponse:
    _, corpus, _ = _services(request)
    return await versioned_document_pdf(request, corpus.corpus_version, document_id)


@app.post("/api/verify/quote")
async def quote_challenge(request: Request, body: QuoteChallengeRequest) -> JSONResponse:
    database, current_corpus, _ = _services(request)
    row = await _owned_run_row(request, body.run_id)
    if body.corpus_version != row["corpus_version"]:
        raise HTTPException(status_code=404, detail="Corpus version not found in this run")
    envelope = await database.get_envelope(body.run_id)
    cited_chunk_ids = {
        evidence.chunk_id
        for claim in (envelope.claims if envelope else [])
        for evidence in claim.evidence
    }
    if body.chunk_id not in cited_chunk_ids:
        raise HTTPException(status_code=404, detail="Cited chunk not found in this run")
    corpus = (
        current_corpus
        if current_corpus.corpus_version == body.corpus_version
        else CorpusStore(request.app.state.settings, body.corpus_version)
    )
    chunks = corpus.get_chunks([body.chunk_id], neighbor_radius=0)
    if not chunks:
        raise HTTPException(status_code=404, detail="Chunk not found")
    normalized_quote, operations = normalize_evidence_text(body.quote)
    if not normalized_quote:
        raise HTTPException(status_code=422, detail="Quote contains no evidence text")
    matched = normalized_quote in chunks[0].normalized_text
    return JSONResponse(
        {
            "chunk_id": body.chunk_id,
            "quote_found": matched,
            "status": "verified" if matched else "rejected",
            "normalization_operations": operations,
        }
    )


web_dist = Path(os.getenv("NEEDLEPROOF_WEB_DIST", "apps/web/dist"))
if web_dist.is_dir():
    app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
