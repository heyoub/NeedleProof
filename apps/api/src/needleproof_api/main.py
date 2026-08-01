from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from .config import Settings
from .corpus import load_current_manifest
from .db import AppDatabase
from .models import CorpusSummary, RunCreateRequest, RunCreateResponse, RunStatus
from .receipt import receipt_html
from .retrieval import CorpusStore
from .service import InvestigationService, RunCapacityError
from .util import normalize_evidence_text

TERMINAL_STATUSES = {
    RunStatus.COMPLETED,
    RunStatus.INCOMPLETE,
    RunStatus.CANCELLED,
    RunStatus.FAILED,
}


class QuoteChallengeRequest(BaseModel):
    chunk_id: int
    quote: str = Field(min_length=1, max_length=4000)


async def verify_model_access(settings: Settings) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required. NeedleProof will not start without it.")
    try:
        # Model catalog visibility can differ from invocation access, so the startup
        # gate exercises the same Responses endpoint the agent will use.
        await AsyncOpenAI().responses.create(
            model=settings.model,
            input="Reply with OK.",
            reasoning={"effort": "none"},
            max_output_tokens=16,
            store=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Configured model {settings.model!r} is unavailable to this OpenAI project. "
            "NeedleProof will not silently substitute another model."
        ) from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    await verify_model_access(settings)
    database = AppDatabase(settings.app_db_path)
    await database.initialize()
    corpus = CorpusStore(settings)
    app.state.settings = settings
    app.state.database = database
    app.state.corpus = corpus
    app.state.service = InvestigationService(settings, database, corpus)
    yield
    for task in list(app.state.service._tasks.values()):
        task.cancel()


app = FastAPI(
    title="NeedleProof API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)


def _services(request: Request) -> tuple[AppDatabase, CorpusStore, InvestigationService]:
    return request.app.state.database, request.app.state.corpus, request.app.state.service


@app.get("/api/health")
async def health(request: Request) -> dict[str, str]:
    _, corpus, _ = _services(request)
    return {
        "status": "ok",
        "model": request.app.state.settings.model,
        "corpus_version": corpus.corpus_version,
        "manifest_sha256": corpus.manifest_sha256,
    }


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
        return await service.create_run(body)
    except RunCapacityError as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": "2"},
        ) from exc


@app.get("/api/runs/{run_id}")
async def get_run(request: Request, run_id: str):
    database, _, _ = _services(request)
    envelope = await database.get_envelope(run_id)
    if not envelope:
        raise HTTPException(status_code=404, detail="Investigation run not found")
    return envelope


@app.post("/api/runs/{run_id}/cancel", status_code=202)
async def cancel_run(request: Request, run_id: str) -> dict[str, str]:
    database, _, service = _services(request)
    if not await database.get_run_row(run_id):
        raise HTTPException(status_code=404, detail="Investigation run not found")
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
    if not await database.get_run_row(run_id):
        raise HTTPException(status_code=404, detail="Investigation run not found")
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
            if events:
                quiet_polls = 0
                for event in events:
                    cursor = int(event["sequence"])
                    yield f"id: {cursor}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            else:
                quiet_polls += 1
                if quiet_polls >= 40:
                    quiet_polls = 0
                    yield ": heartbeat\n\n"
            envelope = await database.get_envelope(run_id)
            if envelope and envelope.status in TERMINAL_STATUSES and not events:
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


def _receipt_path(row: dict[str, object]) -> Path:
    value = row.get("receipt_path")
    if not value:
        raise HTTPException(status_code=409, detail="Receipt has not been sealed")
    path = Path(str(value)).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail="Receipt file is unavailable")
    return path


@app.get("/api/runs/{run_id}/receipt.json")
async def receipt_json(request: Request, run_id: str) -> FileResponse:
    database, _, _ = _services(request)
    row = await database.get_run_row(run_id)
    if not row:
        raise HTTPException(status_code=404, detail="Investigation run not found")
    return FileResponse(
        _receipt_path(row),
        media_type="application/json",
        filename=f"needleproof-{run_id}.json",
        headers={"X-Robots-Tag": "noindex, nofollow"},
    )


@app.get("/api/runs/{run_id}/receipt", response_class=HTMLResponse)
async def receipt_page(request: Request, run_id: str) -> HTMLResponse:
    database, _, _ = _services(request)
    row = await database.get_run_row(run_id)
    if not row:
        raise HTTPException(status_code=404, detail="Investigation run not found")
    receipt = json.loads(_receipt_path(row).read_text(encoding="utf-8"))
    return HTMLResponse(receipt_html(receipt), headers={"X-Robots-Tag": "noindex, nofollow"})


@app.get("/api/documents/{document_id}/pdf")
async def document_pdf(request: Request, document_id: str) -> FileResponse:
    _, corpus, _ = _services(request)
    try:
        path = corpus.source_pdf(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
    return FileResponse(path, media_type="application/pdf")


@app.post("/api/verify/quote")
async def quote_challenge(request: Request, body: QuoteChallengeRequest) -> JSONResponse:
    _, corpus, _ = _services(request)
    chunks = corpus.get_chunks([body.chunk_id], neighbor_radius=0)
    if not chunks:
        raise HTTPException(status_code=404, detail="Chunk not found")
    normalized_quote, operations = normalize_evidence_text(body.quote)
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
