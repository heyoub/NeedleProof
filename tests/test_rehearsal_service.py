from __future__ import annotations

import asyncio
import json
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
from needleproof_api.service import (
    InvestigationService,
    RunCapacityError,
    SessionLiveRunError,
)
from needleproof_api.util import canonical_json, sha256_text


def reseal_receipt(receipt):
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = sha256_text(canonical_json(unsigned))


@pytest.mark.asyncio
async def test_rehearsal_replays_and_seals_without_model_access(tmp_path):
    shutil.copytree(Path("data/corpora"), tmp_path / "corpora")
    (tmp_path / "rehearsal").mkdir()
    shutil.copy2(Path("data/rehearsal/featured.json"), tmp_path / "rehearsal/featured.json")

    settings = Settings(data_dir=tmp_path)
    database = AppDatabase(settings.app_db_path)
    await database.initialize()
    service = InvestigationService(settings, database, CorpusStore(settings))
    created = await service.create_run(
        RunCreateRequest(question="Replay the featured investigation", rehearsal=True),
        session_id="test-session",
    )
    run_id = created.run_id
    await service._tasks[run_id]

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
async def test_admission_rejects_before_persistence_and_counts_rehearsals(tmp_path, monkeypatch):
    shutil.copytree(Path("data/corpora"), tmp_path / "corpora")
    settings = Settings(data_dir=tmp_path, max_concurrent_runs=1)
    database = AppDatabase(settings.app_db_path)
    await database.initialize()
    service = InvestigationService(settings, database, CorpusStore(settings))
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_replay(_run_id, _request, _ledger):
        entered.set()
        await release.wait()

    async def blocked_live(_run_id, _request, _session_id, _ledger):
        entered.set()
        await release.wait()

    monkeypatch.setattr(service, "_run_live", blocked_live)
    monkeypatch.setattr(service, "_run_rehearsal", blocked_replay)

    first = await service.create_run(
        RunCreateRequest(question="Hold the rehearsal slot", rehearsal=True),
        session_id="session-a",
    )
    await entered.wait()

    with pytest.raises(RunCapacityError):
        await service.create_run(
            RunCreateRequest(question="This live run must be rejected", rehearsal=False),
            session_id="session-b",
        )

    with sqlite3.connect(settings.app_db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1

    release.set()
    await service._tasks[first.run_id]
    await asyncio.sleep(0)
    second = await service.create_run(
        RunCreateRequest(question="Capacity is available again", rehearsal=False),
        session_id="session-a",
    )
    await service._tasks[second.run_id]


@pytest.mark.asyncio
async def test_run_endpoint_reports_capacity_with_retry_hint():
    class AtCapacity:
        async def create_run(self, _body, *, session_id, client_ip):
            raise RunCapacityError("NeedleProof is at investigation capacity.")

    class AllowUsage:
        async def admit(self, **_kwargs):
            return None

    class AvailableModel:
        async def require(self):
            return None

    state = SimpleNamespace(
        database=None,
        corpus=None,
        service=AtCapacity(),
        usage_limiter=AllowUsage(),
        model_availability=AvailableModel(),
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=state),
        state=SimpleNamespace(session_id="session-a", client_ip="127.0.0.1"),
    )

    with pytest.raises(HTTPException) as raised:
        await create_run(request, RunCreateRequest(question="Try a bounded run"))

    assert raised.value.status_code == 429
    assert raised.value.headers == {"Retry-After": "2"}


@pytest.mark.asyncio
async def test_one_active_live_run_per_browser_session(tmp_path, monkeypatch):
    shutil.copytree(Path("data/corpora"), tmp_path / "corpora")
    settings = Settings(data_dir=tmp_path, max_concurrent_runs=2)
    database = AppDatabase(settings.app_db_path)
    await database.initialize()
    service = InvestigationService(settings, database, CorpusStore(settings))
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_live(_run_id, _request, _session_id, _ledger):
        entered.set()
        await release.wait()

    monkeypatch.setattr(service, "_run_live", blocked_live)
    first = await service.create_run(
        RunCreateRequest(question="First live run"), session_id="alice"
    )
    await entered.wait()

    with pytest.raises(SessionLiveRunError):
        await service.create_run(RunCreateRequest(question="Second live run"), session_id="alice")

    with sqlite3.connect(settings.app_db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1

    release.set()
    await service._tasks[first.run_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["tampered", "wrong_corpus", "fabricated_quote"])
async def test_rehearsal_rejects_untrusted_or_stale_source_receipts(tmp_path, mutation):
    shutil.copytree(Path("data/corpora"), tmp_path / "corpora")
    (tmp_path / "rehearsal").mkdir()
    source = json.loads(Path("data/rehearsal/featured.json").read_text(encoding="utf-8"))
    if mutation == "tampered":
        source["answer"] = f"{source['answer']} tampered"
    elif mutation == "wrong_corpus":
        source["corpus_manifest_sha256"] = "0" * 64
        reseal_receipt(source)
    else:
        claim = next(claim for claim in source["claims"] if claim["status"] == "verified")
        claim["values"][0]["value"] = "$346 million"
        claim["values"][0]["evidence"][0]["exact_quote"] = "Fee-related earnings were $346 million."
        reseal_receipt(source)
    (tmp_path / "rehearsal" / "featured.json").write_text(
        json.dumps(source, indent=2) + "\n", encoding="utf-8"
    )

    settings = Settings(data_dir=tmp_path)
    database = AppDatabase(settings.app_db_path)
    await database.initialize()
    service = InvestigationService(settings, database, CorpusStore(settings))
    created = await service.create_run(
        RunCreateRequest(question="Reject bad rehearsal", rehearsal=True),
        session_id="alice",
    )
    task = service._tasks[created.run_id]
    await task

    row = await database.get_run_row(created.run_id)
    assert row is not None
    assert row["status"] == RunStatus.FAILED.value
    assert row["receipt_path"]
    assert row["error_json"]
