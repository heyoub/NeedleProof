from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from needleproof_api.config import Settings
from needleproof_api.db import AppDatabase
from needleproof_api.main import run_events
from needleproof_api.models import RunCreateRequest, RunStatus
from needleproof_api.receipt import RunLedger
from needleproof_api.retrieval import CorpusStore
from needleproof_api.security import PublicUsageLimiter
from needleproof_api.service import InvestigationService
from needleproof_api.util import new_run_id, utc_now_iso


def lifecycle_service(tmp_path: Path) -> tuple[InvestigationService, AppDatabase, Settings]:
    shutil.copytree(Path("data/corpora"), tmp_path / "corpora")
    (tmp_path / "rehearsal").mkdir()
    shutil.copy2(Path("data/rehearsal/featured.json"), tmp_path / "rehearsal/featured.json")
    settings = Settings(data_dir=tmp_path)
    database = AppDatabase(settings.app_db_path)
    limiter = PublicUsageLimiter(settings, database)
    return (
        InvestigationService(settings, database, CorpusStore(settings), limiter),
        database,
        settings,
    )


@pytest.mark.asyncio
async def test_rehearsal_cancellation_seals_terminal_receipt(tmp_path, monkeypatch):
    service, database, settings = lifecycle_service(tmp_path)
    await database.initialize()
    entered = asyncio.Event()

    async def blocked_rehearsal(_run_id, _request, _ledger):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(service, "_run_rehearsal", blocked_rehearsal)
    created = await service.create_run(
        RunCreateRequest(question="Cancel this replay", rehearsal=True),
        session_id="alice",
    )
    task = service._tasks[created.run_id]
    await entered.wait()

    assert await service.cancel(created.run_id)
    await task

    row = await database.get_run_row(created.run_id)
    assert row is not None
    assert row["status"] == RunStatus.CANCELLED.value
    assert Path(str(row["receipt_path"])).exists()
    events = await database.list_events(created.run_id)
    assert events[-1]["type"] == "run.cancelled"
    assert settings.receipts_dir.joinpath(f"{created.run_id}.json").exists()


@pytest.mark.asyncio
async def test_immediate_cancellation_still_enters_guarded_finalizer(tmp_path):
    service, database, _settings = lifecycle_service(tmp_path)
    await database.initialize()
    created = await service.create_run(
        RunCreateRequest(question="Cancel before run started", rehearsal=True),
        session_id="alice",
    )
    task = service._tasks[created.run_id]

    assert await service.cancel(created.run_id)
    await task

    row = await database.get_run_row(created.run_id)
    assert row is not None
    assert row["status"] == RunStatus.CANCELLED.value
    assert Path(str(row["receipt_path"])).exists()


@pytest.mark.asyncio
async def test_live_cancellation_releases_reserved_model_tokens(tmp_path, monkeypatch):
    service, database, settings = lifecycle_service(tmp_path)
    await database.initialize()
    entered = asyncio.Event()

    async def blocked_live(_run_id, _request, _session_id, _ledger):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(service, "_run_live", blocked_live)
    created = await service.create_run(
        RunCreateRequest(question="Cancel this live run"),
        session_id="alice",
        client_ip="192.0.2.20",
    )
    task = service._tasks[created.run_id]
    await entered.wait()
    assert await database.get_model_token_reservation(created.run_id) == (
        settings.model_token_reservation_per_run
    )

    assert await service.cancel(created.run_id)
    await task

    assert await database.get_model_token_reservation(created.run_id) is None
    row = await database.get_run_row(created.run_id)
    assert row is not None
    assert row["status"] == RunStatus.CANCELLED.value


@pytest.mark.asyncio
async def test_run_creation_failure_releases_reservation(tmp_path, monkeypatch):
    service, database, settings = lifecycle_service(tmp_path)
    await database.initialize()
    settings.max_runs_per_session_per_hour = 2
    settings.max_runs_per_ip_per_hour = 2
    original_create_run = database.create_run

    async def fail_create_run(**_kwargs):
        raise OSError("injected run persistence failure")

    monkeypatch.setattr(database, "create_run", fail_create_run)
    for attempt in range(2):
        with pytest.raises(OSError, match="injected run persistence failure"):
            await service.create_run(
                RunCreateRequest(question=f"Fail after budget reservation {attempt}"),
                session_id="alice",
                client_ip="192.0.2.21",
            )

    assert await database.release_terminal_or_orphan_token_reservations() == 0
    assert not service._admitted_run_ids

    monkeypatch.setattr(database, "create_run", original_create_run)
    recovered = await service.create_run(
        RunCreateRequest(question="Persist after database recovery", rehearsal=True),
        session_id="alice",
        client_ip="192.0.2.21",
    )
    await service._tasks[recovered.run_id]
    row = await database.get_run_row(recovered.run_id)
    assert row is not None
    assert row["status"] == RunStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_terminal_sse_waits_for_terminal_database_commit(tmp_path, monkeypatch):
    service, database, _settings = lifecycle_service(tmp_path)
    await database.initialize()
    commit_entered = asyncio.Event()
    release_commit = asyncio.Event()
    original_commit = service._commit_terminal

    async def blocked_commit(envelope, receipt_path, error):
        commit_entered.set()
        await release_commit.wait()
        await original_commit(envelope, receipt_path, error)

    monkeypatch.setattr(service, "_commit_terminal", blocked_commit)
    created = await service.create_run(
        RunCreateRequest(question="Replay with a commit barrier", rehearsal=True),
        session_id="alice",
    )
    task = service._tasks[created.run_id]
    await commit_entered.wait()

    row = await database.get_run_row(created.run_id)
    assert row is not None
    assert row["status"] == RunStatus.RUNNING.value
    persisted_events = await database.list_events(created.run_id)
    assert persisted_events[-1]["type"] == "run.completed"

    async def connected() -> bool:
        return False

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(database=database, corpus=service.corpus, service=service)
        ),
        state=SimpleNamespace(session_id="alice"),
        is_disconnected=connected,
    )
    response = await run_events(request, created.run_id, None, None)
    iterator = response.body_iterator
    for _event in persisted_events[:-1]:
        chunk = await anext(iterator)
        assert "run.completed" not in chunk

    terminal_chunk = asyncio.create_task(anext(iterator))
    await asyncio.sleep(0.15)
    assert not terminal_chunk.done()

    release_commit.set()
    await task
    chunk = await asyncio.wait_for(terminal_chunk, timeout=1)
    assert "run.completed" in chunk
    envelope = await database.get_envelope(created.run_id)
    assert envelope is not None
    assert envelope.status == RunStatus.COMPLETED


@pytest.mark.asyncio
async def test_startup_reconciles_abandoned_running_run(tmp_path):
    service, database, _settings = lifecycle_service(tmp_path)
    await database.initialize()
    run_id = new_run_id()
    assert service.usage_limiter is not None
    await service.usage_limiter.admit(
        run_id=run_id,
        session_id="alice",
        client_ip="192.0.2.22",
        rehearsal=False,
    )
    await database.create_run(
        run_id=run_id,
        session_id="alice",
        rehearsal=False,
        question="Interrupted by a restart",
        corpus_id=service.corpus.corpus_id,
        corpus_version=service.corpus.corpus_version,
        manifest_sha256=service.corpus.manifest_sha256,
    )
    await database.update_run(run_id, status=RunStatus.RUNNING, started_at=utc_now_iso())
    ledger = RunLedger(run_id, database, service.settings)
    await ledger.append("run.started", {"question": "Interrupted by a restart"})

    await service.reconcile_abandoned_runs()

    row = await database.get_run_row(run_id)
    assert row is not None
    assert row["status"] == RunStatus.INTERRUPTED.value
    assert Path(str(row["receipt_path"])).exists()
    events = await database.list_events(run_id)
    assert events[-1]["type"] == "run.interrupted"
    assert await database.get_model_token_reservation(run_id) is None


@pytest.mark.asyncio
async def test_receipt_recovers_when_terminal_database_update_initially_fails(
    tmp_path, monkeypatch
):
    service, database, settings = lifecycle_service(tmp_path)
    await database.initialize()
    original_commit = service._commit_terminal
    failed_once = False

    async def fail_once(envelope, receipt_path, error):
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise RuntimeError("injected terminal database failure")
        await original_commit(envelope, receipt_path, error)

    monkeypatch.setattr(service, "_commit_terminal", fail_once)
    created = await service.create_run(
        RunCreateRequest(question="Recover committed receipt", rehearsal=True),
        session_id="alice",
    )
    task = service._tasks[created.run_id]
    await task
    assert settings.receipts_dir.joinpath(f"{created.run_id}.json").exists()
    row = await database.get_run_row(created.run_id)
    assert row is not None
    assert row["status"] == RunStatus.INTERRUPTED.value
    assert row["receipt_path"]
    assert row["receipt_sha256"]
    events = await database.list_events(created.run_id)
    assert events[-1]["type"] == "run.interrupted"

    async def connected() -> bool:
        return False

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(database=database, corpus=service.corpus, service=service)
        ),
        state=SimpleNamespace(session_id="alice"),
        is_disconnected=connected,
    )
    response = await run_events(request, created.run_id, None, None)
    streamed = "".join([chunk async for chunk in response.body_iterator])
    assert "run.interrupted" in streamed
    assert "run.completed" not in streamed

    await service.reconcile_abandoned_runs()
    recovered = await database.get_run_row(created.run_id)
    assert recovered is not None
    assert recovered["status"] == RunStatus.COMPLETED.value
    assert recovered["receipt_path"]


@pytest.mark.asyncio
async def test_reconciliation_preserves_valid_receipt_when_commit_is_unavailable(
    tmp_path, monkeypatch
):
    service, database, _settings = lifecycle_service(tmp_path)
    await database.initialize()
    created = await service.create_run(
        RunCreateRequest(question="Preserve this sealed result", rehearsal=True),
        session_id="alice",
    )
    await service._tasks[created.run_id]
    row = await database.get_run_row(created.run_id)
    assert row is not None
    receipt_path = Path(str(row["receipt_path"]))
    original_receipt = receipt_path.read_bytes()
    original_commit = service._commit_terminal
    await database.update_run(created.run_id, status=RunStatus.INTERRUPTED)

    async def unavailable_commit(_envelope, _receipt_path, _error):
        raise OSError("injected recovery persistence failure")

    monkeypatch.setattr(service, "_commit_terminal", unavailable_commit)
    await service.reconcile_abandoned_runs()

    preserved = await database.get_run_row(created.run_id)
    assert preserved is not None
    assert preserved["status"] == RunStatus.INTERRUPTED.value
    assert receipt_path.read_bytes() == original_receipt

    monkeypatch.setattr(service, "_commit_terminal", original_commit)
    await service.reconcile_abandoned_runs()
    recovered = await database.get_run_row(created.run_id)
    assert recovered is not None
    assert recovered["status"] == RunStatus.COMPLETED.value
    assert receipt_path.read_bytes() == original_receipt


@pytest.mark.asyncio
async def test_receipt_write_failure_becomes_recoverable_interruption(tmp_path, monkeypatch):
    service, database, _settings = lifecycle_service(tmp_path)
    await database.initialize()
    original_seal = RunLedger.seal

    async def fail_seal(self, *args, **kwargs):
        raise OSError("injected receipt filesystem failure")

    monkeypatch.setattr(RunLedger, "seal", fail_seal)
    created = await service.create_run(
        RunCreateRequest(question="Recover failed receipt write", rehearsal=True),
        session_id="alice",
    )
    task = service._tasks[created.run_id]
    await task

    row = await database.get_run_row(created.run_id)
    assert row is not None
    assert row["status"] == RunStatus.INTERRUPTED.value
    assert row["receipt_path"] is None

    monkeypatch.setattr(RunLedger, "seal", original_seal)
    await service.reconcile_abandoned_runs()
    recovered = await database.get_run_row(created.run_id)
    assert recovered is not None
    assert recovered["status"] == RunStatus.INTERRUPTED.value
    assert Path(str(recovered["receipt_path"])).exists()
