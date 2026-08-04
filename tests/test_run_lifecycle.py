from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from types import SimpleNamespace

import needleproof_api.main as main_module
import pytest
from fastapi import HTTPException
from needleproof_api.config import Settings
from needleproof_api.db import AppDatabase
from needleproof_api.main import get_run, receipt_json, receipt_page, run_events
from needleproof_api.models import RunCreateRequest, RunStatus
from needleproof_api.receipt import RunLedger
from needleproof_api.retrieval import CorpusStore
from needleproof_api.security import PublicUsageLimiter
from needleproof_api.service import InvestigationService, ReceiptRecoveryOutcome
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
async def test_retention_delete_cascades_to_run_children(tmp_path):
    database = AppDatabase(tmp_path / "cascade.sqlite3")
    await database.initialize()
    run_id = new_run_id()
    await database.create_run(
        run_id=run_id,
        session_id="cascade-owner",
        rehearsal=True,
        question="Verify cascade cleanup",
        corpus_id="corpus",
        corpus_version="v_0123456789abcdef",
        manifest_sha256="digest",
    )
    await database.append_event_row(
        run_id=run_id,
        sequence=1,
        event_type="run.started",
        occurred_at=utc_now_iso(),
        payload={},
        previous_hash="0" * 64,
        event_hash="1" * 64,
    )
    await database.append_openai_call(
        run_id,
        1,
        {"operation": "embedding", "started_at": utc_now_iso(), "token_usage": {}},
    )

    await database.delete_runs_created_before("9999-12-31T23:59:59+00:00")

    async with database._connect() as connection:
        event_cursor = await connection.execute("SELECT COUNT(*) FROM ledger_events")
        call_cursor = await connection.execute("SELECT COUNT(*) FROM openai_calls")
        event_count = (await event_cursor.fetchone())[0]
        call_count = (await call_cursor.fetchone())[0]
    assert event_count == 0
    assert call_count == 0


@pytest.mark.asyncio
async def test_transient_receipt_read_failure_does_not_delete_sealed_file(tmp_path, monkeypatch):
    service, database, settings = lifecycle_service(tmp_path)
    await database.initialize()
    run_id = new_run_id()
    await database.create_run(
        run_id=run_id,
        session_id="receipt-owner",
        rehearsal=True,
        question="Preserve unreadable receipt",
        corpus_id=service.corpus.corpus_id,
        corpus_version=service.corpus.corpus_version,
        manifest_sha256=service.corpus.manifest_sha256,
    )
    settings.receipts_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = settings.receipts_dir / f"{run_id}.json"
    receipt_path.write_text("{}", encoding="utf-8")
    original_read_text = Path.read_text

    def fail_only_for_receipt(path, *args, **kwargs):
        if path == receipt_path:
            raise PermissionError("temporary read failure")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_only_for_receipt)
    await service.reconcile_abandoned_runs()

    assert receipt_path.exists()
    row = await database.get_run_row(run_id)
    assert row is not None
    assert row["status"] == RunStatus.QUEUED.value


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
async def test_explicit_zero_sse_cursor_overrides_nonzero_header(tmp_path):
    service, database, _settings = lifecycle_service(tmp_path)
    await database.initialize()
    created = await service.create_run(
        RunCreateRequest(question="Replay the complete event history", rehearsal=True),
        session_id="alice",
    )
    await service._tasks[created.run_id]

    async def connected() -> bool:
        return False

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(database=database, corpus=service.corpus, service=service)
        ),
        state=SimpleNamespace(session_id="alice"),
        is_disconnected=connected,
    )
    response = await run_events(request, created.run_id, "999", 0)
    streamed = "".join([chunk async for chunk in response.body_iterator])

    assert "id: 1\n" in streamed
    assert "run.started" in streamed
    assert "run.completed" in streamed


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
    commit_attempts = 0

    async def fail_twice(envelope, receipt_path, error):
        nonlocal commit_attempts
        commit_attempts += 1
        if commit_attempts <= 2:
            raise RuntimeError("injected terminal database failure")
        await original_commit(envelope, receipt_path, error)

    monkeypatch.setattr(service, "_commit_terminal", fail_twice)
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
    iterator = response.body_iterator
    first_terminal_index = next(
        index
        for index, event in enumerate(events)
        if event["type"].startswith("run.") and event["type"] != "run.started"
    )
    for _event in events[:first_terminal_index]:
        chunk = await anext(iterator)
        assert "run.completed" not in chunk
        assert "run.interrupted" not in chunk

    for route in (receipt_json, receipt_page):
        with pytest.raises(HTTPException, match="awaiting terminal-state reconciliation") as raised:
            await route(request, created.run_id)
        assert raised.value.status_code == 409

    restored = await get_run(request, created.run_id)
    assert restored.status == RunStatus.COMPLETED
    assert commit_attempts == 3

    terminal_chunk = asyncio.create_task(anext(iterator))
    chunk = await asyncio.wait_for(terminal_chunk, timeout=1)
    assert "run.completed" in chunk
    assert "run.interrupted" not in chunk
    recovered = await database.get_run_row(created.run_id)
    assert recovered is not None
    assert recovered["status"] == RunStatus.COMPLETED.value
    assert recovered["receipt_path"]
    assert (await receipt_json(request, created.run_id)).status_code == 200
    assert (await receipt_page(request, created.run_id)).status_code == 200


@pytest.mark.asyncio
async def test_get_run_returns_retryable_error_while_receipt_recovery_is_unavailable(
    tmp_path, monkeypatch
):
    service, database, _settings = lifecycle_service(tmp_path)
    await database.initialize()
    created = await service.create_run(
        RunCreateRequest(question="Keep retryable recovery nonterminal", rehearsal=True),
        session_id="alice",
    )
    await service._tasks[created.run_id]
    await database.update_run(created.run_id, status=RunStatus.INTERRUPTED)

    async def unavailable_commit(_envelope, _receipt_path, _error):
        raise OSError("injected persistent recovery failure")

    monkeypatch.setattr(service, "_commit_terminal", unavailable_commit)

    async def connected() -> bool:
        return False

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(database=database, corpus=service.corpus, service=service)
        ),
        state=SimpleNamespace(session_id="alice"),
        is_disconnected=connected,
    )

    with pytest.raises(HTTPException, match="temporarily unavailable") as raised:
        await get_run(request, created.run_id)

    assert raised.value.status_code == 503
    assert raised.value.headers == {"Retry-After": "1"}
    row = await database.get_run_row(created.run_id)
    assert row is not None
    assert row["status"] == RunStatus.INTERRUPTED.value


@pytest.mark.asyncio
async def test_terminal_sse_retries_transient_receipt_reconciliation_on_same_connection(
    tmp_path, monkeypatch
):
    service, database, _settings = lifecycle_service(tmp_path)
    await database.initialize()
    original_commit = service._commit_terminal
    commit_attempts = 0

    async def fail_twice_then_recover(envelope, receipt_path, error):
        nonlocal commit_attempts
        commit_attempts += 1
        if commit_attempts <= 2:
            raise OSError("injected transient terminal persistence failure")
        await original_commit(envelope, receipt_path, error)

    monkeypatch.setattr(service, "_commit_terminal", fail_twice_then_recover)
    created = await service.create_run(
        RunCreateRequest(question="Retry recovery without reconnecting", rehearsal=True),
        session_id="alice",
    )
    await service._tasks[created.run_id]
    interrupted = await database.get_run_row(created.run_id)
    assert interrupted is not None
    assert interrupted["status"] == RunStatus.INTERRUPTED.value

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

    async def read_terminal_event() -> str:
        async for chunk in response.body_iterator:
            if "run.completed" in chunk:
                return chunk
        raise AssertionError("SSE stream closed before terminal recovery")

    terminal_chunk = await asyncio.wait_for(read_terminal_event(), timeout=2)

    assert "run.interrupted" not in terminal_chunk
    assert commit_attempts == 3
    recovered = await database.get_run_row(created.run_id)
    assert recovered is not None
    assert recovered["status"] == RunStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_terminal_sse_retries_transient_receipt_read_on_same_connection(
    tmp_path, monkeypatch
):
    service, database, _settings = lifecycle_service(tmp_path)
    await database.initialize()
    original_commit = service._commit_terminal
    failed_commit = False

    async def fail_initial_commit(envelope, receipt_path, error):
        nonlocal failed_commit
        if not failed_commit:
            failed_commit = True
            raise OSError("injected initial terminal persistence failure")
        await original_commit(envelope, receipt_path, error)

    monkeypatch.setattr(service, "_commit_terminal", fail_initial_commit)
    created = await service.create_run(
        RunCreateRequest(question="Retry transient receipt reads", rehearsal=True),
        session_id="alice",
    )
    await service._tasks[created.run_id]
    interrupted = await database.get_run_row(created.run_id)
    assert interrupted is not None
    assert interrupted["status"] == RunStatus.INTERRUPTED.value

    original_integrity_check = main_module._integrity_checked_receipt
    integrity_attempts = 0

    def fail_first_integrity_read(row):
        nonlocal integrity_attempts
        integrity_attempts += 1
        if integrity_attempts == 1:
            raise OSError("injected transient receipt read failure")
        return original_integrity_check(row)

    monkeypatch.setattr(main_module, "_integrity_checked_receipt", fail_first_integrity_read)

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

    async def read_terminal_event() -> str:
        async for chunk in response.body_iterator:
            if "run.completed" in chunk:
                return chunk
        raise AssertionError("SSE stream closed after a transient receipt read failure")

    terminal_chunk = await asyncio.wait_for(read_terminal_event(), timeout=2)

    assert "run.interrupted" not in terminal_chunk
    assert integrity_attempts >= 2
    recovered = await database.get_run_row(created.run_id)
    assert recovered is not None
    assert recovered["status"] == RunStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_receipt_recovery_rejects_receipt_from_another_run(tmp_path):
    service, database, _settings = lifecycle_service(tmp_path)
    await database.initialize()
    first = await service.create_run(
        RunCreateRequest(question="First sealed result", rehearsal=True),
        session_id="alice",
    )
    second = await service.create_run(
        RunCreateRequest(question="Second sealed result", rehearsal=True),
        session_id="bob",
    )
    await asyncio.gather(service._tasks[first.run_id], service._tasks[second.run_id])
    first_row = await database.get_run_row(first.run_id)
    second_row = await database.get_run_row(second.run_id)
    assert first_row is not None
    assert second_row is not None
    first_receipt = Path(str(first_row["receipt_path"]))
    second_receipt = Path(str(second_row["receipt_path"]))
    await database.update_run(
        first.run_id,
        status=RunStatus.INTERRUPTED,
        answer=None,
        result_json=None,
        receipt_sha256=None,
    )
    first_receipt.write_bytes(second_receipt.read_bytes())

    outcome = await service.reconcile_pending_receipt(first.run_id)

    assert outcome == ReceiptRecoveryOutcome.UNRECOVERABLE
    rejected = await database.get_run_row(first.run_id)
    assert rejected is not None
    assert rejected["status"] == RunStatus.INTERRUPTED.value
    assert rejected["answer"] is None
    assert rejected["result_json"] is None
    assert not first_receipt.exists()
    assert second_receipt.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("corrupt_receipt", ["{not valid json", "null", '{"events": null}'])
async def test_terminal_sse_closes_for_unrecoverable_receipt_corruption(tmp_path, corrupt_receipt):
    service, database, _settings = lifecycle_service(tmp_path)
    await database.initialize()
    created = await service.create_run(
        RunCreateRequest(question="Detect corrupt terminal receipt", rehearsal=True),
        session_id="alice",
    )
    await service._tasks[created.run_id]
    row = await database.get_run_row(created.run_id)
    assert row is not None
    receipt_path = Path(str(row["receipt_path"]))
    receipt_path.write_text(corrupt_receipt, encoding="utf-8")
    events = await database.list_events(created.run_id)

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
    first_terminal_index = next(
        index for index, event in enumerate(events) if event["type"] == "run.completed"
    )
    for _event in events[:first_terminal_index]:
        await anext(iterator)

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(iterator), timeout=1)


@pytest.mark.asyncio
async def test_terminal_event_read_failure_marks_run_interrupted(tmp_path, monkeypatch):
    service, database, _settings = lifecycle_service(tmp_path)
    await database.initialize()
    original_rehearsal = service._run_rehearsal
    original_list_events = database.list_events
    armed = False

    async def rehearsal_then_arm(*args, **kwargs):
        nonlocal armed
        state = await original_rehearsal(*args, **kwargs)
        armed = True
        return state

    async def fail_once_when_armed(*args, **kwargs):
        nonlocal armed
        if armed:
            armed = False
            raise OSError("injected terminal event read failure")
        return await original_list_events(*args, **kwargs)

    monkeypatch.setattr(service, "_run_rehearsal", rehearsal_then_arm)
    monkeypatch.setattr(database, "list_events", fail_once_when_armed)
    created = await service.create_run(
        RunCreateRequest(question="Recover terminal event persistence", rehearsal=True),
        session_id="alice",
    )
    await service._tasks[created.run_id]

    row = await database.get_run_row(created.run_id)
    assert row is not None
    assert row["status"] == RunStatus.INTERRUPTED.value
    events = await original_list_events(created.run_id)
    assert events[-1]["type"] == "run.interrupted"


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

    async def read_interruption() -> str:
        async for chunk in response.body_iterator:
            if "run.interrupted" in chunk:
                return chunk
        raise AssertionError("SSE stream closed without the receiptless interruption")

    terminal_chunk = await asyncio.wait_for(read_interruption(), timeout=1)
    assert "run.completed" not in terminal_chunk

    monkeypatch.setattr(RunLedger, "seal", original_seal)
    await service.reconcile_abandoned_runs()
    recovered = await database.get_run_row(created.run_id)
    assert recovered is not None
    assert recovered["status"] == RunStatus.INTERRUPTED.value
    assert Path(str(recovered["receipt_path"])).exists()
