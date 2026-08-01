from __future__ import annotations

import asyncio
import shutil
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from needleproof_api.config import Settings
from needleproof_api.db import AppDatabase
from needleproof_api.main import create_run
from needleproof_api.models import RunCreateRequest, RunStatus
from needleproof_api.retrieval import CorpusStore
from needleproof_api.service import InvestigationService, RunCapacityError
from needleproof_api.util import new_run_id


@pytest.mark.asyncio
async def test_rehearsal_replays_and_seals_without_model_access(tmp_path):
    shutil.copytree(Path("data/corpora"), tmp_path / "corpora")
    (tmp_path / "rehearsal").mkdir()
    shutil.copy2(Path("data/rehearsal/featured.json"), tmp_path / "rehearsal/featured.json")

    settings = Settings(data_dir=tmp_path)
    database = AppDatabase(settings.app_db_path)
    await database.initialize()
    service = InvestigationService(settings, database, CorpusStore(settings))
    request = RunCreateRequest(question="Replay the featured investigation", rehearsal=True)
    run_id = new_run_id()
    await database.create_run(
        run_id=run_id,
        session_id=None,
        question=request.question,
        corpus_id=service.corpus.corpus_id,
        corpus_version=service.corpus.corpus_version,
        manifest_sha256=service.corpus.manifest_sha256,
    )
    await service._replay(run_id, request)

    envelope = await database.get_envelope(run_id)
    assert envelope is not None
    assert envelope.status == RunStatus.COMPLETED
    assert envelope.answer
    assert {claim.status.value for claim in envelope.claims} >= {"conflict", "date_variant"}
    assert settings.receipts_dir.joinpath(f"{run_id}.json").exists()
    events = await database.list_events(run_id)
    assert events[0]["type"] == "run.started"
    assert events[-1]["type"] == "run.completed"


@pytest.mark.asyncio
async def test_admission_rejects_before_persistence_and_counts_rehearsals(
    tmp_path, monkeypatch
):
    shutil.copytree(Path("data/corpora"), tmp_path / "corpora")
    settings = Settings(data_dir=tmp_path, max_concurrent_runs=1)
    database = AppDatabase(settings.app_db_path)
    await database.initialize()
    service = InvestigationService(settings, database, CorpusStore(settings))
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_run(_run_id, _request):
        entered.set()
        await release.wait()

    monkeypatch.setattr(service, "_execute", blocked_run)
    monkeypatch.setattr(service, "_replay", blocked_run)

    first = await service.create_run(
        RunCreateRequest(question="Hold the rehearsal slot", rehearsal=True)
    )
    await entered.wait()

    with pytest.raises(RunCapacityError):
        await service.create_run(
            RunCreateRequest(question="This live run must be rejected", rehearsal=False)
        )

    with sqlite3.connect(settings.app_db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1

    release.set()
    await service._tasks[first.run_id]
    await asyncio.sleep(0)
    second = await service.create_run(
        RunCreateRequest(question="Capacity is available again", rehearsal=False)
    )
    await service._tasks[second.run_id]


@pytest.mark.asyncio
async def test_run_endpoint_reports_capacity_with_retry_hint():
    class AtCapacity:
        async def create_run(self, _body):
            raise RunCapacityError("NeedleProof is at investigation capacity.")

    state = SimpleNamespace(database=None, corpus=None, service=AtCapacity())
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    with pytest.raises(HTTPException) as raised:
        await create_run(request, RunCreateRequest(question="Try a bounded run"))

    assert raised.value.status_code == 429
    assert raised.value.headers == {"Retry-After": "2"}
